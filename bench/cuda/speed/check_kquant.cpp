// A Q4_K_M GGUF run two ways on the same tokens: its K-quant blocks dequantized
// to bf16 at load (the existing paths, the oracle) and kept as blocks, read by
// the K-quant GEMVs (Model.quant). Both decode by the imperative step and the
// native run is fed the oracle's own tokens (teacher forcing), so the only
// difference is how a weight's bytes are read: each step's logits agree to the
// bf16 rounding the oracle applies to every dequantized weight, and the top
// token agrees except where the oracle's top two are within that rounding.
//
// Usage: check_kquant model-Q4_K_M.gguf [steps=32]

#include "../../models/qwen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace qm = qwenmodel;

// Decode `steps` tokens; `force` (when given) is the token fed at each step,
// else the greedy one. Returns the fed tokens, logits per step.
static std::vector<int64_t> run(qm::Model& M, int steps, std::vector<std::vector<float>>& logits,
                                const std::vector<int64_t>* force) {
  qm::reset_cache(M);
  std::vector<int64_t> fed;
  int64_t id = qm::PREFILL_IDS[0], pos = 0;
  for (int s = 0; s < steps; s++) {
    fed.push_back(id);
    const float* lg = qm::imperative_logits(M, id, pos++);
    logits.emplace_back(lg, lg + M.c.VOCAB);
    id = force ? (*force)[(size_t)s + 1 < force->size() ? (size_t)s + 1 : (size_t)s]
               : qm::argmax(logits.back());
  }
  return fed;
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) { std::printf("no CUDA device — skipping\n"); return 0; }
  if (argc < 2) { std::fprintf(stderr, "usage: %s model-Q4_K_M.gguf [steps]\n", argv[0]); return 2; }
  const int steps = argc > 2 ? std::atoi(argv[2]) : 32;
  qm::gg::model m(argv[1]);
  tl::use_gpu();
  std::vector<std::vector<float>> la, lb;
  std::vector<int64_t> fed;
  {
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, false, true, /*quant=*/false);
    fed = run(M, steps, la, nullptr);
  }
  {
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, false, true, /*quant=*/true);
    if (!M.quant) { std::printf("native K-quant path unavailable\n"); return 1; }
    run(M, steps, lb, &fed);
  }
  double worst = 0;
  int agree = 0, tie_flips = 0, bad = 0;
  for (int s = 0; s < steps; s++) {
    double mx = 0, ref = 0;
    for (size_t i = 0; i < la[s].size(); i++) {
      mx = std::max(mx, (double)std::fabs(la[s][i] - lb[s][i]));
      ref = std::max(ref, (double)std::fabs(la[s][i]));
    }
    worst = std::max(worst, mx / (1.0 + ref));
    const int64_t ta = qm::argmax(la[s]), tb = qm::argmax(lb[s]);
    if (ta == tb) { agree++; continue; }
    // A different top token is fine only if the oracle itself had them within
    // the rounding it carries (a near tie).
    const double gap = la[s][ta] - la[s][tb];
    if (gap < 4e-2 * (1.0 + ref)) tie_flips++; else bad++;
  }
  std::printf("%d steps | top-1 agrees %d, near-tie flips %d, real misses %d | logits worst |d|/(1+max) %.2e\n",
              steps, agree, tie_flips, bad, worst);
  return bad == 0 && worst < 2e-2 ? 0 : 1;
}
