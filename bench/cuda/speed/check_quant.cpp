// A Q4_K_M GGUF run two ways on the same tokens: its block-quantized blocks dequantized
// to bf16 at load (the existing paths, the oracle) and kept as blocks, read by
// the block-quantized GEMVs (Model.quant). Both decode by the imperative step and the
// native run is fed the oracle's own tokens (teacher forcing), so the only
// difference is how a weight's bytes are read: each step's logits agree to the
// bf16 rounding the oracle applies to every dequantized weight, and the top
// token agrees except where the oracle's top two are within that rounding.
//
// The native run is made twice: reading the activation as float (the exact one,
// which should agree to the oracle's bf16 rounding) and as int8 (Model.q8act,
// the Q4_K GEMVs' dp4a form: the activation rounded to 8 bits per 32 values, as
// llama.cpp does), which is held to a looser bound and to the same top token
// except at near ties.
//
// Usage: check_quant model-Q4_K_M.gguf [steps=32]

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
  std::vector<std::vector<float>> la;
  std::vector<int64_t> fed;
  {
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, false, true, /*quant=*/false);
    fed = run(M, steps, la, nullptr);
  }
  bool ok_all = true;
  {
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, false, true, /*quant=*/true);
    if (!M.quant) { std::printf("native block-quantized path unavailable\n"); return 1; }
    const bool has_q8 = M.q8act;
    for (bool q8 : {false, true}) {
      if (q8 && !has_q8) {  // the backend or the environment has no int8 form
        std::printf("int8 activations unavailable (TL_NO_Q8ACT or no dp4a): skipped\n");
        continue;
      }
      M.q8act = q8;
      std::vector<std::vector<float>> lb;
      run(M, steps, lb, &fed);
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
      const double bound = q8 ? 3e-1 : 2e-2;
      const bool ok = bad == 0 && worst < bound;
      ok_all = ok_all && ok;
      std::printf("%d steps, %s activations | top-1 agrees %d, near-tie flips %d, real misses %d | logits worst |d|/(1+max) %.2e (< %.0e) %s\n",
                  steps, q8 ? "int8" : "float", agree, tie_flips, bad, worst, bound, ok ? "OK" : "FAIL");
    }
  }
  // The batched prefill over the same blocks: a prompt through GEMMs that
  // dequantize as they stage, against the bf16 weights' prefill, in the fast
  // form (f16 sums) and the exact one a non-finite prompt is redone in.
  bool prefill_ok = true;
  for (bool exact : {false, true}) {
    std::vector<float> pa_, pb_;
    int64_t ta0 = -1, tb0 = -1;
    const std::vector<int> ids = qm::fixture_ids(200);
    for (int pass = 0; pass < 2; pass++) {
      qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, false, true, /*quant=*/pass == 1);
      qm::reset_cache(M);
      const int64_t t = qm::prefill_batched(M, ids, qm::PREFILL_CHUNK, exact);
      const float* lg = qm::host_read(M.scratch.logits);
      (pass ? pb_ : pa_).assign(lg, lg + M.c.VOCAB);
      (pass ? tb0 : ta0) = t;
    }
    double pmx = 0, pref = 0;
    for (size_t i = 0; i < pa_.size(); i++) {
      pmx = std::max(pmx, (double)std::fabs(pa_[i] - pb_[i]));
      pref = std::max(pref, (double)std::fabs(pa_[i]));
    }
    const double err = pmx / (1.0 + pref);
    const bool ok = err < 2e-2 && (ta0 == tb0 || pa_[ta0] - pa_[tb0] < 4e-2 * (1.0 + pref));
    prefill_ok = prefill_ok && ok;
    std::printf("prefill 200 tokens (%s) | first token %lld vs %lld | logits worst |d|/(1+max) %.2e %s\n",
                exact ? "exact" : "f16 sums", (long long)ta0, (long long)tb0, err, ok ? "OK" : "FAIL");
  }
  return ok_all && prefill_ok ? 0 : 1;
}
