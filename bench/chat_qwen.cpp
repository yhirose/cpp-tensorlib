// M9 "actually chat" — the real thing. Load a Qwen2.5-0.5B-Instruct GGUF, tokenize
// a prompt with the Qwen2 chat template (tl::tokenizer), run the decoder
// (qwen_model.h) greedily, and detokenize the generated ids back to text. This is
// the end-to-end proof: prompt string in, generated text out, on hand-written CUDA
// kernels + own GGUF loader + own BPE tokenizer, zero third-party runtime deps.
//
// Usage: chat_qwen [model.gguf] ["your prompt"]
// Greedy decoding (deterministic); stops at <|im_end|> or the token budget.

#include "qwen_model.h"
#include "tokenizer.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

namespace qm = qwenmodel;

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no CUDA device — skipping chat\n");
    return 0;
  }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  std::string user = argc > 2 ? argv[2]
                              : "Give me a short introduction to large language models.";
  const int64_t MAX_NEW = 256;

  std::printf("loading %s\n", path.c_str());
  tl::tokenizer tok(path);
  qm::gg::model m(path);
  if (!qm::check_config(m)) {
    std::printf("config mismatch — hard-coded for Qwen2.5-0.5B\n");
    return 2;
  }
  tl::use_gpu();
  // bf16 weight storage: ~1GB (under the WSL2 2GB sysmem cliff), decode GEMV
  // consumes it natively. Pass "f32" as argv[3] to force exact F32 (~2GB).
  tl::dtype wdt = (argc > 3 && std::string(argv[3]) == "f32") ? tl::dtype::f32
                                                              : tl::dtype::bf16;
  std::printf("weight storage: %s\n", wdt == tl::dtype::bf16 ? "bf16" : "f32");
  auto t_build = clk::now();
  qm::Model M = qm::build(m, wdt);
  double build_ms = ms_since(t_build);

  // Qwen2 chat template (matches the model's default system prompt).
  std::string templated =
      "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
      "helpful assistant.<|im_end|>\n<|im_start|>user\n" +
      user + "<|im_end|>\n<|im_start|>assistant\n";
  std::vector<int> ids = tok.encode(templated);
  std::printf("prompt: %s\n%zu prompt tokens; generating (greedy)...\n\n",
              user.c_str(), ids.size());

  // Prefill: feed the prompt token by token (== causal prefill for the cache).
  int64_t pos = 0;
  std::vector<float> logits;
  auto t_prefill = clk::now();
  for (size_t i = 0; i < ids.size(); i++) logits = qm::step(M, ids[i], pos++);
  double prefill_ms = ms_since(t_prefill);

  // Greedy generate until <|im_end|> or the budget.
  std::vector<int> gen;
  int64_t next = qm::argmax(logits);
  auto t_dec = clk::now();
  for (int64_t i = 0; i < MAX_NEW; i++) {
    if (next == tok.eos_id()) break;
    gen.push_back((int)next);
    logits = qm::step(M, next, pos++);
    next = qm::argmax(logits);
  }
  double dec_ms = ms_since(t_dec);

  std::printf("=== assistant ===\n%s\n", tok.decode(gen).c_str());
  std::printf(
      "\n(%zu tokens)  build %.0f ms | prefill %zu tok %.0f ms (%.1f tok/s) | "
      "decode %zu tok %.0f ms (%.1f tok/s)\n",
      gen.size(), build_ms, ids.size(), prefill_ms, ids.size() * 1000.0 / prefill_ms,
      gen.size(), dec_ms, gen.size() * 1000.0 / dec_ms);
  return 0;
}
