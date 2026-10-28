// The two numbers llama-bench reports, measured the same way on this library's
// path, so a Qwen GGUF can be put through both and the rows read side by side:
//
//   pp<P>  a P-token prompt into an empty cache (or one already holding d
//          tokens): the batched prefill, through the last token's logits.
//   tg<N>  N tokens decoded one at a time after d cached ones: the chat's own
//          decoder (CUDA-graph replay where the backend captures), each step
//          through its greedy token. Capturing is set-up, like llama-bench's
//          warm-up run, and is not timed.
//
// Each test runs once untimed, then `reps` times; the row is the mean ± sample
// standard deviation in tok/s, as llama-bench prints it, and the median, which
// is the steadier figure on WSL2. Token ids are a fixed cycle: nothing here
// depends on their values.
//
// Usage: bench_qwen_llama model.gguf [P=512] [N=128] [reps=5] [d=0] [f32acc]

#include "../../models/qwen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qm = qwenmodel;

// Back to an empty sequence, then `d` tokens of context (untimed).
static bool fill_depth(qm::Model& M, int64_t d) {
  qm::reset_cache(M);
  return d == 0 || qm::prefill_batched(M, qm::fixture_ids(d)) >= 0;
}

static double pp_ms(qm::Model& M, int64_t P, int64_t d) {
  if (!fill_depth(M, d)) return -1;
  const std::vector<int> ids = qm::fixture_ids(P);
  const double t0 = qm::StepProf::now_ms();
  if (qm::prefill_batched(M, ids) < 0) return -1;  // its argmax is the sync
  return qm::StepProf::now_ms() - t0;
}

static double tg_ms(qm::Model& M, int64_t N, int64_t d) {
  if (!fill_depth(M, d)) return -1;
  qm::captured_decoder dec;
  dec.init(M, qm::PREFILL_IDS[0], qm::seq_cache(M).pos);
  int64_t tok = qm::PREFILL_IDS[0];
  const double t0 = qm::StepProf::now_ms();
  for (int64_t i = 0; i < N && tok >= 0; i++) tok = dec.step(M, tok);
  const double ms = qm::StepProf::now_ms() - t0;
  dec.destroy();
  return tok >= 0 ? ms : -1;
}

template <class F>
static void row(const char* test, int64_t tokens, int reps, F run) {
  if (run() < 0) {  // the warm-up, which also says whether the path exists
    std::printf("| %-8s | unavailable on this backend |\n", test);
    return;
  }
  std::vector<double> ts;
  for (int r = 0; r < reps; r++) ts.push_back(tokens * 1000.0 / run());
  double mean = 0, var = 0;
  for (double t : ts) mean += t;
  mean /= reps;
  for (double t : ts) var += (t - mean) * (t - mean);
  const double sd = reps > 1 ? std::sqrt(var / (reps - 1)) : 0.0;
  std::sort(ts.begin(), ts.end());
  const double med = reps % 2 ? ts[reps / 2] : (ts[reps / 2 - 1] + ts[reps / 2]) / 2;
  std::printf("| %-8s | %10.2f ± %7.2f | %10.2f |\n", test, mean, sd, med);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.gguf [P=512] [N=128] [reps=5] [d=0] [f32acc]\n",
                 argv[0]);
    return 2;
  }
  if (!tl::gpu_available()) {
    std::printf("no GPU device\n");
    return 0;
  }
  const int64_t P = argc > 2 ? std::atoll(argv[2]) : 512;
  const int64_t N = argc > 3 ? std::atoll(argv[3]) : 128;
  const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
  const int64_t d = argc > 5 ? std::atoll(argv[5]) : 0;
  const bool f32acc = argc > 6 && std::string(argv[6]) == "f32acc";
  qm::gg::model m(argv[1]);
  tl::use_gpu();
  qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, /*oracle=*/false, !f32acc,
                           /*quant=*/qm::kquant_file(m) && !std::getenv("TL_DEQUANT"));
  const auto* t = m.tensor("blk.0.ffn_up.weight");
  std::printf("%s | %s | %lld layers | weights %s (file %s) | depth %lld | prefill sums %s\n",
              argv[1], M.c.arch.c_str(), (long long)M.c.NL,
              M.outwT.dt() == tl::dtype::bf16 ? "bf16" : "f32",
              t && t->type == qm::gg::ggml_type::bf16 ? "BF16" : "F16", (long long)d,
              M.f16_acc ? "f16" : "f32");
  std::printf("| test     |       t/s (mean ± sd) |   t/s (med) |\n");
  const std::string pp = "pp" + std::to_string(P), tg = "tg" + std::to_string(N);
  row(pp.c_str(), P, reps, [&] { return pp_ms(M, P, d); });
  if (N > 0) row(tg.c_str(), N, reps, [&] { return tg_ms(M, N, d); });  // N=0: prefill only
  return 0;
}
