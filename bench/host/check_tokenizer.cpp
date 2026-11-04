// Token-exact check of tl::tokenizer (include/tokenizer.h) against ground
// truth for the file's family: HF's Qwen2.5-0.5B-Instruct tokenizer
// (tokenizer_oracle_data.h, GPT-2 BPE), or llama.cpp's llama-tokenize
// (tools/tokenizer_oracle.py) on Qwen3.5, Llama 3 (GPT-2 BPE, each its own
// pre-tokenizer) and Gemma 3 (SentencePiece).
// argv[1] = gguf path. Exit 0 iff every case encodes to the exact id
// sequence and decode() round-trips the text.

#include <cstdio>
#include <string>
#include <vector>

#include "../../include/tokenizer.h"
#include "tokenizer_oracle_data.h"
#include "tokenizer_oracle_gemma3.h"
#include "tokenizer_oracle_llama3.h"
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
// A family's ground truth, by the file's architecture and pre-tokenizer (a
// vocabulary each), and the token its chat template closes a turn with. A
// byte-level BPE (bpe) keeps the bytes of text that is not UTF-8 where
// llama.cpp reads each bad one as U+FFFD, so its oracle's such cases are skipped.
struct family {
  const char *arch, *pre;
  std::vector<oracle_case> (*cases)();
  const char* turn;
  bool bpe;
};
static const family families[] = {
    {"qwen2", "qwen2", cases_of<tokoracle::cases>, "<|im_end|>", true},
    {"qwen3", "qwen2", cases_of<tokoracle::cases>, "<|im_end|>", true},
    {"qwen35", "qwen35", cases_of<tokoracle_qwen35::cases>, "<|im_end|>", true},
    {"llama", "llama-bpe", cases_of<tokoracle_llama3::cases>, "<|eot_id|>", true},
    {"gemma3", "default", cases_of<tokoracle_gemma3::cases>, "<end_of_turn>", false},
};

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
    if (fam->bpe && !utf8(cs.text)) {
      std::printf("case %zu: skipped (not UTF-8: llama.cpp reads U+FFFD, this BPE the bytes)\n", c);
      continue;
    }
    std::vector<int> got = tok.encode(cs.text);

    bool ids_ok = (got == cs.ids);
    std::string rt = tok.decode(got);
    bool rt_ok = (rt == cs.text);

    if (ids_ok && rt_ok) {
      std::printf("case %zu: PASS (%zu ids, round-trip ok)\n", c, got.size());
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
                    rt.size(), cs.text.size());
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
