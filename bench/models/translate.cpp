// Offline translation with PFN's PLaMo 3 Translate (its GGUFs): the text on
// stdin, its translation on stdout, on this repo's own kernels, GGUF loader and
// tokenizer.
//
// Usage: translate MODEL.gguf --to LANG [--from LANG] [--form plain|polite]
//                  [--context TEXT] [--glossary SOURCE=TARGET]... [--max-new N] < text
//
// The prompt is PLaMo's PFML3 as the model's chat_template.jinja writes it (one
// source text, its options), checked as that template checks them: a language
// code lowercase (en, ja, zh-hans, ja(easy)), a form only into Japanese, no
// text empty or holding a <|plamo: marker. The text is kept as it is, white
// space and all. Greedy, as PFN's examples decode; the translation ends where
// the model writes <|plamo:tag|> or <|plamo:bos|> (tokenizer::is_eog), or after
// --max-new tokens (default: twice the prompt's, at least 512). The KV cache
// holds the prompt and those, up to the 32,768 tokens the model is made for.

#include "qwen.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qm = qwenmodel;

namespace {

struct request {
  std::string text, to, from, form, context;
  std::vector<std::pair<std::string, std::string>> glossary;
};

// The template's check_text: a string with something besides white space (here
// ASCII's; the template's trim takes Unicode's too, so a text of only U+3000 passes
// here where it raises there), and no PFML marker (the prompt's structure is those
// markers).
void check_text(const std::string& v, const char* name, bool allow_empty = false) {
  if (!allow_empty && v.find_first_not_of(" \t\n\r\f\v") == std::string::npos)
    throw std::invalid_argument(std::string(name) + " must not be empty");
  if (v.find("<|plamo:") != std::string::npos)
    throw std::invalid_argument(std::string(name) + " must not contain <|plamo: (PFML's markers)");
}
// The template's check_lang: ja(easy), or 2 or 3 lowercase letters with an
// optional 4-letter script (zh-hans).
void check_lang(const std::string& v, const char* name) {
  check_text(v, name);
  if (v == "ja(easy)") return;
  const size_t dash = v.find('-');
  const size_t head = dash == std::string::npos ? v.size() : dash;
  const bool shape = (head == 2 || head == 3) &&
                     (dash == std::string::npos || (v.size() - dash - 1 == 4 && v.find('-', dash + 1) == std::string::npos));
  const bool lower = v.find_first_not_of("abcdefghijklmnopqrstuvwxyz-") == std::string::npos;
  if (!shape || !lower)
    throw std::invalid_argument(std::string(name) + " must be a lowercase language code: en, ja, zh-hans, ja(easy)");
}

// PFML3, as chat_template.jinja renders one source text with these options.
std::string pfml3(const request& r) {
  check_text(r.text, "the text");
  check_lang(r.to, "--to");
  if (!r.from.empty()) check_lang(r.from, "--from");
  check_text(r.context, "--context", /*allow_empty=*/true);
  if (!r.form.empty() && ((r.form != "plain" && r.form != "polite") || r.to != "ja"))
    throw std::invalid_argument("--form is plain or polite, and only with --to ja");
  for (const auto& [s, t] : r.glossary) {
    check_text(s, "a glossary source");
    check_text(t, "a glossary target");
  }
  std::string p = "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation";
  if (!r.context.empty()) p += "<|plamo:tag|>context<|plamo:msg|>" + r.context;
  if (!r.glossary.empty()) {
    p += "<|plamo:tag|>glossary";
    for (const auto& [s, t] : r.glossary) p += "<|plamo:index|>" + s + "<|plamo:item|>" + t;
  }
  p += "<|plamo:tag|>input";
  if (!r.from.empty()) p += "<|plamo:key|>lang<|plamo:val|>" + r.from;
  p += "<|plamo:msg|>" + r.text + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + r.to;
  if (!r.form.empty()) p += "<|plamo:key|>form<|plamo:val|>" + r.form;
  return p + "<|plamo:msg|>";
}

// Where a decoded prefix may be printed: before an unfinished UTF-8 sequence at
// its end (a code point split across byte tokens).
size_t utf8_complete(const std::string& s) {
  size_t i = s.size(), back = 0;
  while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) i--, back++;
  if (i == 0) return s.size();
  const auto lead = static_cast<unsigned char>(s[i - 1]);
  const size_t len = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
  return back + 1 < len ? i - 1 : s.size();
}

int usage() {
  std::fprintf(stderr,
               "usage: translate MODEL.gguf --to LANG [--from LANG] [--form plain|polite]\n"
               "                 [--context TEXT] [--glossary SOURCE=TARGET]... [--max-new N] < text\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string path = argv[1];
  request r;
  int64_t max_new = 0;  // 0: the default
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (a == "--to") r.to = v;
    else if (a == "--from") r.from = v;
    else if (a == "--form") r.form = v;
    else if (a == "--context") r.context = v;
    else if (a == "--max-new") {
      char* end = nullptr;
      max_new = std::strtoll(v.c_str(), &end, 10);
      if (*end || max_new <= 0) return usage();
    } else if (a == "--glossary") {
      const size_t eq = v.find('=');
      if (eq == std::string::npos) return usage();
      r.glossary.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else {
      return usage();
    }
  }
  if (r.to.empty()) return usage();
  r.text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());

  try {
    const std::string prompt = pfml3(r);
    if (!tl::gpu_available()) throw std::runtime_error("no GPU device");
    tl::tokenizer tok(path);
    qm::gg::model m(path);
    if (m.kv("general.architecture").as_str() != "plamo3")
      throw std::invalid_argument(path + " is not a PLaMo 3 model");
    const std::vector<int> ids = tok.encode(prompt);
    constexpr int64_t context_limit = 32768;  // the model card's, input and output together
    if (max_new == 0) max_new = std::max<int64_t>(512, 2 * (int64_t)ids.size());
    const int64_t maxc = std::min<int64_t>((int64_t)ids.size() + max_new, context_limit);
    if ((int64_t)ids.size() >= maxc)
      throw std::invalid_argument(std::to_string(ids.size()) + " prompt tokens, past the model's " +
                                  std::to_string(context_limit));
    tl::use_gpu();
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, maxc, /*oracle=*/false,
                            /*f16_acc=*/true, /*quant=*/qm::quant_file(m));

    const double t0 = qm::StepProf::now_ms();
    qm::captured_decoder dec;
    int64_t next = dec.begin(M, ids);
    const double t1 = qm::StepProf::now_ms();
    // A token's text as it comes (a plamo3 decode is its tokens' bytes joined),
    // held back while it ends inside a UTF-8 sequence.
    std::string pending;
    char last = '\n';
    int64_t n = 0;
    for (; next >= 0 && !tok.is_eog((int)next) && n < maxc - (int64_t)ids.size(); n++) {
      pending += tok.decode({(int)next});
      const size_t upto = utf8_complete(pending);
      if (upto) {
        std::fwrite(pending.data(), 1, upto, stdout);
        std::fflush(stdout);
        last = pending[upto - 1];
        pending.erase(0, upto);
      }
      next = dec.next(M);
    }
    const bool cut = next >= 0 && !tok.is_eog((int)next);
    dec.destroy();
    std::fwrite(pending.data(), 1, pending.size(), stdout);
    if (!pending.empty()) last = pending.back();
    if (last != '\n') std::printf("\n");  // a translation keeps the text's last newline
    std::fflush(stdout);                    // before stderr's lines
    const double decode_s = (qm::StepProf::now_ms() - t1) / 1e3;
    if (cut)
      std::fprintf(stderr, "translate: stopped at %lld tokens (--max-new or the context), cut short\n",
                   (long long)n);
    std::fprintf(stderr, "(%zu prompt tokens %.2f s, %lld tokens %.2f s, %.1f tok/s)\n", ids.size(),
                 (t1 - t0 - dec.capture_ms) / 1e3, (long long)n, decode_s, n / std::max(decode_s, 1e-9));
    return 0;
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "translate: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "translate: %s\n", e.what());
    return 1;
  }
}
