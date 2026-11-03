// Token-exact check of tl::tokenizer (include/tokenizer.h) against ground
// truth for the file's family: HF's Qwen2.5-0.5B-Instruct tokenizer
// (tokenizer_oracle_data.h, GPT-2 BPE) or llama.cpp's llama-tokenize on Gemma 3
// (tokenizer_oracle_gemma3.h, SentencePiece; tools/tokenizer_oracle.py).
// argv[1] = gguf path. Exit 0 iff every case encodes to the exact id
// sequence and decode() round-trips the text.

#include <cstdio>
#include <string>
#include <vector>

#include "../../include/tokenizer.h"
#include "tokenizer_oracle_data.h"
#include "tokenizer_oracle_gemma3.h"

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

  tl::tokenizer tok(path);
  std::printf("tokenizer loaded from %s (bos=%d eos=%d)\n", path.c_str(),
              tok.bos_id(), tok.eos_id());

  bool all_ok = true;
  struct oracle_case {
    std::string text;
    std::vector<int> ids;
  };
  std::vector<oracle_case> cases;
  if (tl::gguf::model(path).kv("general.architecture").as_str() == "gemma3") {
    for (auto& c : tokoracle_gemma3::cases()) cases.push_back({c.text, c.ids});
  } else {
    for (auto& c : tokoracle::cases()) cases.push_back({c.text, c.ids});
  }
  for (size_t c = 0; c < cases.size(); c++) {
    const auto& cs = cases[c];
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

  if (all_ok) {
    std::printf("ALL OK\n");
    return 0;
  }
  return 1;
}
