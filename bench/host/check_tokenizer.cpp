// Token-exact check of tl::tokenizer (include/tokenizer.h) against ground
// truth for the file's family: HF's Qwen2.5-0.5B-Instruct tokenizer
// (tokenizer_oracle_data.h, GPT-2 BPE), or llama.cpp's llama-tokenize
// (tools/tokenizer_oracle.py) on Qwen3.5, Llama 3 (GPT-2 BPE, each its own
// pre-tokenizer), Gemma 3 (SentencePiece) and PLaMo 3 (its unigram).
// argv[1] = gguf path. Exit 0 iff every case encodes to the exact id
// sequence and decode() round-trips the text.

#include <cstdio>
#include <string>
#include <vector>

#include "../../include/tokenizer.h"
#include "tokenizer_oracle_data.h"
#include "tokenizer_oracle_gemma3.h"
#include "tokenizer_oracle_llama3.h"
#include "tokenizer_oracle_plamo3.h"
#include "tokenizer_oracle_qwen35.h"

struct oracle_case {
  std::string text;
  std::vector<int> ids;
};
// A header's oracle cases (each header its own Case type) as oracle_case.
template <auto cases>
static std::vector<oracle_case> cases_of() {
  std::vector<oracle_case> out;
  for (const auto& c : cases()) out.push_back({c.text, c.ids});
  return out;
}
// What a tokenizer makes of text that is not UTF-8, where llama.cpp reads each
// bad byte as U+FFFD: a byte-level BPE keeps the bytes, so its oracle's ids do
// not hold (skip); PLaMo 3's reads U+FFFD too, so they do but the text does not
// come back (ids); SentencePiece's falls back to the bytes themselves (exact).
enum class not_utf8 { skip, ids, exact };
// A family's ground truth, by the file's architecture and pre-tokenizer (a
// vocabulary each), the token its chat template closes a turn with, and what
// its tokenizer leaves out of the text it gives back (PLaMo 3: U+EE00, a cut).
struct family {
  const char *arch, *pre;
  std::vector<oracle_case> (*cases)();
  const char* turn;
  not_utf8 on_bad_utf8;
  const char* dropped = nullptr;
};
static const family families[] = {
    {"qwen2", "qwen2", cases_of<tokoracle::cases>, "<|im_end|>", not_utf8::skip},
    {"qwen3", "qwen2", cases_of<tokoracle::cases>, "<|im_end|>", not_utf8::skip},
    {"qwen35", "qwen35", cases_of<tokoracle_qwen35::cases>, "<|im_end|>", not_utf8::skip},
    {"llama", "llama-bpe", cases_of<tokoracle_llama3::cases>, "<|eot_id|>", not_utf8::skip},
    {"gemma3", "default", cases_of<tokoracle_gemma3::cases>, "<end_of_turn>", not_utf8::exact},
    {"plamo3", "default", cases_of<tokoracle_plamo3::cases>, "<|plamo:tag|>", not_utf8::ids,
     "\xee\xb8\x80"},
};
// The text as the family's decode gives it back: without what it drops.
static std::string given_back(const family& f, std::string s) {
  if (!f.dropped) return s;
  const std::string d = f.dropped;
  for (size_t p; (p = s.find(d)) != std::string::npos;) s.erase(p, d.size());
  return s;
}

// Whether s is UTF-8: each lead byte followed by the continuation bytes it names.
static bool utf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const auto b = static_cast<unsigned char>(s[i]);
    const size_t n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
    if (!n || i + n > s.size()) return false;
    for (size_t k = 1; k < n; k++)
      if ((static_cast<unsigned char>(s[i + k]) >> 6) != 2) return false;
    i += n;
  }
  return true;
}

static void print_ids(const char* label, const std::vector<int>& v) {
  std::printf("  %s [", label);
  for (size_t i = 0; i < v.size(); i++)
    std::printf("%s%d", i ? "," : "", v[i]);
  std::printf("] (%zu)\n", v.size());
}

int main(int argc, char** argv) {
  std::string path = argc > 1
      ? argv[1]
      : "/home/yuji/models/qwen2.5-0.5b-instruct-fp16.gguf";

  const family* fam = nullptr;
  {
    tl::gguf::model m(path);
    const std::string arch = m.kv("general.architecture").as_str(),
                      pre = m.has("tokenizer.ggml.pre") ? m.kv("tokenizer.ggml.pre").as_str() : "";
    for (const auto& f : families)
      if (arch == f.arch && pre == f.pre) fam = &f;
    if (!fam) {
      std::printf("no oracle for architecture %s, pre-tokenizer %s\n", arch.c_str(), pre.c_str());
      return 2;
    }
  }
  tl::tokenizer tok(path);
  std::printf("tokenizer loaded from %s (%s, %s; bos=%d eos=%d)\n", path.c_str(), fam->arch,
              fam->pre, tok.bos_id(), tok.eos_id());

  bool all_ok = true;
  const std::vector<oracle_case> cases = fam->cases();
  for (size_t c = 0; c < cases.size(); c++) {
    const auto& cs = cases[c];
    const bool not_text = !utf8(cs.text);
    if (not_text && fam->on_bad_utf8 == not_utf8::skip) {
      std::printf("case %zu: skipped (not UTF-8: llama.cpp reads U+FFFD, this BPE the bytes)\n", c);
      continue;
    }
    std::vector<int> got = tok.encode(cs.text);

    bool ids_ok = (got == cs.ids);
    std::string rt = tok.decode(got);
    const bool rt_checked = !(not_text && fam->on_bad_utf8 == not_utf8::ids);
    bool rt_ok = !rt_checked || rt == given_back(*fam, cs.text);

    if (ids_ok && rt_ok) {
      std::printf("case %zu: PASS (%zu ids, round-trip %s)\n", c, got.size(),
                  rt_checked ? "ok" : "not checked: not UTF-8");
    } else {
      all_ok = false;
      std::printf("case %zu: FAIL%s%s\n", c, ids_ok ? "" : " [ids]",
                  rt_ok ? "" : " [round-trip]");
      if (!ids_ok) {
        print_ids("expected:", cs.ids);
        print_ids("got:     ", got);
        size_t n = cs.ids.size() < got.size() ? cs.ids.size() : got.size();
        size_t d = 0;
        while (d < n && cs.ids[d] == got[d]) d++;
        std::printf("  first diff at index %zu (expected %s, got %s)\n", d,
                    d < cs.ids.size() ? std::to_string(cs.ids[d]).c_str() : "<end>",
                    d < got.size() ? std::to_string(got[d]).c_str() : "<end>");
      }
      if (!rt_ok)
        std::printf("  round-trip mismatch: got %zu bytes, want %zu bytes\n",
                    rt.size(), given_back(*fam, cs.text).size());
    }
  }

  // The generation ends at EOS and at the family's closing token of a turn (the
  // chat template's), which is not EOS in Gemma's file; plain text ends nothing.
  const std::vector<int> turn = tok.encode(fam->turn),
                         text = tok.encode("Hello");
  const bool eog_ok = turn.size() == 1 && tok.is_eog(turn[0]) && tok.is_eog(tok.eos_id()) &&
                      !text.empty() && !tok.is_eog(text[0]);
  std::printf("end of generation (EOS, the turn's closing token, not text): %s\n",
              eog_ok ? "PASS" : "FAIL");
  all_ok &= eog_ok;

  if (all_ok) {
    std::printf("ALL OK\n");
    return 0;
  }
  return 1;
}
