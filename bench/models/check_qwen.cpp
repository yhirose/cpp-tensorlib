// M9 "actually chat" — numeric proof. Load a real Qwen2.5-0.5B-Instruct, Qwen3
// or Qwen3.5 GGUF (F16 weights) and run the decoder end to end, verified against
// a tight numpy reference that reads the SAME F16 weights (tools/qwen_oracle.py
// wrote qwen_oracle_data.h / qwen3_oracle_data.h / qwen35_oracle_data.h; the
// model's arch picks one). Identical
// weights => the C++ forward must match the reference to ~1e-4 at every checkpoint
// (embedding, layer-0 residual, final norm, top-5 logits) AND reproduce the greedy
// token sequence exactly. Proves weight loading + GGML->our layout transpose +
// Qwen2 structure + the head_dim=64 attention kernels, on whatever backend
// gpu:: resolves to. Tokenizer/sampling are
// separate (see chat_qwen). Model path: argv[1] or ~/models/...fp16.gguf.

#include "qwen.h"
#include "qwen35_oracle_data.h"
#include "qwen3_oracle_data.h"
#include "qwen_oracle_data.h"

#include <cstdio>
#include <vector>

namespace qm = qwenmodel;

// One model's checkpoints, whichever generated header they came from.
struct Oracle {
  const float *emb_last, *l0_last, *fnorm_last, *top5_vals;
  const int *top5_ids, *greedy30, *prompt_ids;
  size_t n_greedy, n_prompt;
};
#define TL_ORACLE(ns)                                                       \
  Oracle {                                                                  \
    ns::emb_last, ns::l0_last, ns::fnorm_last, ns::top5_vals, ns::top5_ids, \
        ns::greedy30, ns::prompt_ids, sizeof(ns::greedy30) / sizeof(int),   \
        sizeof(ns::prompt_ids) / sizeof(int)                                \
  }

static double maxrel(const std::vector<float>& a, const float* b, int64_t n) {
  double mr = 0;
  for (int64_t i = 0; i < n; i++)
    mr = std::max(mr, std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)b[i])));
  return mr;
}

// A hybrid model (Qwen3.5) has no array path, so its checkpoints are read off
// the imperative forward: the final norm the last layer's seam leaves in
// scratch, the logits, and the greedy continuation, at F32 against the oracle.
// Then the bf16 weights (the chat's) must reproduce the same greedy opening.
static bool check_hybrid(const qm::gg::model& m, const Oracle& O) {
  const int64_t NP = (int64_t)O.n_prompt, NG = (int64_t)O.n_greedy;
  std::vector<int64_t> want;
  bool ok;
  {
    qm::Model M = qm::build(m, tl::dtype::f32);
    const qm::Config& C = M.c;
    std::printf("model built — %lld layers (%lld attention), head_dim=%lld, DeltaNet %lldx[%lld,%lld]\n",
                (long long)C.NL, (long long)(C.NL / C.INTERVAL), (long long)C.HD,
                (long long)C.SH, (long long)C.SDK, (long long)C.SDV);
    for (int64_t i = 0; i + 1 < NP; i++) qm::step_imperative(M, O.prompt_ids[i], i);
    const float* lp = qm::imperative_logits(M, O.prompt_ids[NP - 1], NP - 1);
    std::vector<float> logits(lp, lp + C.VOCAB);
    const float* fp = qm::host_read(M.scratch.hb);
    double fn_mr = maxrel(std::vector<float>(fp, fp + C.NE), O.fnorm_last, C.NE);
    double logit_mr = 0;
    for (int t = 0; t < 5; t++) {
      const double ref = O.top5_vals[t];
      logit_mr = std::max(logit_mr, std::fabs(logits[O.top5_ids[t]] - ref) / (1.0 + std::fabs(ref)));
    }
    const int64_t top1 = qm::argmax(logits);
    std::printf("\ncheckpoints vs numpy-on-same-F16-weights (imperative path):\n");
    std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top1, O.top5_ids[0],
                top1 == O.top5_ids[0] ? "OK" : "FAIL");
    int64_t tok = top1, p = NP;
    bool greedy_ok = true;
    std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      want.push_back(tok);
      if (tok != O.greedy30[i]) greedy_ok = false;
      tok = qm::step_imperative(M, tok, p++);
    }
    std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");
    ok = fn_mr < 5e-3 && logit_mr < 5e-3 && top1 == O.top5_ids[0] && greedy_ok;
  }
  // A backend without the bf16-weight kernels (bf16_gemm, as the non-hybrid
  // check asks) has no reader for the bf16 lm_head either: skip as it does.
  if (!qm::gpu::caps::bf16_gemm) {
    std::printf("\nbf16 weights: backend has no bf16 weight GEMM — skipped\n");
    return ok;
  }
  const int64_t NB = 8;
  qm::Model B = qm::build(m, tl::dtype::bf16, false, false, 4096, /*oracle=*/false);
  int64_t tok = 0, p = 0;
  for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(B, O.prompt_ids[i], p++);
  bool bf16_ok = true;
  std::printf("\nbf16 weights, greedy %lld:\n  cpp:", (long long)NB);
  for (int64_t i = 0; i < NB; i++) {
    std::printf(" %lld", (long long)tok);
    if (tok != want[i]) bf16_ok = false;
    tok = qm::step_imperative(B, tok, p++);
  }
  std::printf("\n  greedy %s\n", bf16_ok ? "MATCH" : "DIVERGE");
  return ok && bf16_ok;
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no GPU device — skipping Qwen forward check\n");
    return 0;
  }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  std::printf("loading %s\n", path.c_str());
  qm::gg::model m(path);
  const qm::Config C = qm::config_from_gguf(m);
  // Checkpoints exist for Qwen2.5-0.5B (24 layers, 896 wide), Qwen3-0.6B (28,
  // 1024) and Qwen3.5-0.8B (24, 1024).
  const bool q3 = C.arch == "qwen3", q35 = C.arch == "qwen35";
  const int64_t want_nl = q3 ? 28 : 24, want_ne = q3 || q35 ? 1024 : 896;
  if (C.NL != want_nl || C.NE != want_ne) {
    std::printf("config mismatch — no oracle for this model\n");
    return 2;
  }
  const Oracle O = q35 ? TL_ORACLE(qwen35oracle)
                   : q3 ? TL_ORACLE(qwen3oracle)
                        : TL_ORACLE(qwenoracle);
  tl::use_gpu();
  if (C.hybrid()) {
    const bool ok = check_hybrid(m, O);
    std::printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
  }
  std::printf("building model (F32, transposing linear weights)...\n");
  const int64_t NP = (int64_t)O.n_prompt;
  bool imp_ok = true, bf16_ok = true, ok_f32 = false;
  {  // scoped so the F32 weights are back in the pool before the bf16 build
  qm::Model M = qm::build(m);
  std::printf("model built — %lld layers, GQA %lldq/%lldkv, head_dim=%lld\n",
              (long long)C.NL, (long long)C.NH, (long long)C.NKV, (long long)C.HD);

  std::vector<float> l0, fnorm, last_logits;
  int64_t pos = 0;
  for (int64_t i = 0; i < NP; i++) {
    bool last = (i == NP - 1);
    last_logits = qm::step(M, O.prompt_ids[i], pos,
                           last ? &l0 : nullptr, last ? &fnorm : nullptr);
    pos++;
  }
  tl::array el = qm::embed_row(M, O.prompt_ids[NP - 1]);
  el.eval();
  double emb_mr = maxrel(std::vector<float>(el.raw(), el.raw() + C.NE),
                         O.emb_last, C.NE);
  double l0_mr = maxrel(l0, O.l0_last, C.NE);
  double fn_mr = maxrel(fnorm, O.fnorm_last, C.NE);

  std::vector<int64_t> top5;
  { std::vector<float> lg = last_logits;
    for (int t = 0; t < 5; t++) { int64_t bi = qm::argmax(lg); top5.push_back(bi); lg[bi] = -1e30f; } }
  bool top1_ok = top5[0] == O.top5_ids[0];
  double logit_mr = 0;
  for (int t = 0; t < 5; t++) {
    int64_t id = O.top5_ids[t];
    logit_mr = std::max(logit_mr, std::fabs((double)last_logits[id] - O.top5_vals[t]) /
                                      (1.0 + std::fabs((double)O.top5_vals[t])));
  }

  std::printf("\ncheckpoints vs numpy-on-same-F16-weights (expect ~1e-4):\n");
  std::printf("  embedding  maxrel %.2e %s\n", emb_mr, emb_mr < 1e-3 ? "OK" : "FAIL");
  std::printf("  layer0 res maxrel %.2e %s\n", l0_mr, l0_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top5[0],
              O.top5_ids[0], top1_ok ? "OK" : "FAIL");

  const int64_t NG = (int64_t)O.n_greedy;
  std::vector<int64_t> gen;
  int64_t next = qm::argmax(last_logits);
  for (int64_t i = 0; i < NG; i++) {
    gen.push_back(next);
    std::vector<float> lg = qm::step(M, next, pos);
    pos++;
    next = qm::argmax(lg);
  }
  bool greedy_ok = true;
  for (int64_t i = 0; i < NG; i++) if (gen[i] != O.greedy30[i]) greedy_ok = false;
  std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
  for (auto t : gen) std::printf(" %lld", (long long)t);
  std::printf("\n  ref:");
  for (int64_t i = 0; i < NG; i++) std::printf(" %d", O.greedy30[i]);
  std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");

  // The same sequence again, on the imperative path: the fused model-path
  // kernels (rmsnorm/rmsnorm_res/swiglu, the decode GEMVs, kv_append,
  // attn_decode, argmax), or their generic compositions on a backend without
  // them, instead of the array compositions the checkpoints above validated.
  // They are meant to compute the same thing, so at F32 the greedy sequence
  // must be identical — which makes every one of them gated against the numpy
  // reference too, without a second oracle.
  {
    qm::reset_cache(M);
    int64_t p = 0, tok = 0;
    for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(M, O.prompt_ids[i], p++);
    std::printf("\nimperative path (same weights, fused kernels):\n  cpp:");
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      if (tok != O.greedy30[i]) imp_ok = false;
      tok = qm::step_imperative(M, tok, p++);
    }
    std::printf("\n  greedy %s\n", imp_ok ? "MATCH" : "DIVERGE");
  }

  ok_f32 = emb_mr < 1e-3 && l0_mr < 5e-3 && fn_mr < 5e-3 && logit_mr < 5e-3 &&
           top1_ok && greedy_ok && imp_ok;
  }

  // The bf16 weight path is the one a chat actually runs: [N,K] weights for
  // the row GEMV, and the batched prefill's GEMM over a whole chunk of prompt
  // at once. Rounding makes it a different program from the F32 reference
  // above, so the gate is internal consistency — the batched prefill plus the
  // imperative decode must produce the same greedy tokens as THIS model's own
  // array path, which reaches the same rounded weights through .dot() and the
  // separate projections. It is the wiring that check catches: a decode GEMV
  // pointed at the [K,N] oracle weight instead of the [N,K] copy reads the
  // right number of bytes in the wrong order, and nothing but a comparison
  // like this one notices.
  const int64_t NB = 8;
  {
    std::printf("\nbf16 weights (row GEMV + batched prefill) vs this model's array path:\n");
    qm::Model B = qm::build(m, tl::dtype::bf16);
    std::vector<int> ids(O.prompt_ids, O.prompt_ids + NP);
    if (!qm::can_prefill_batched(B, NP)) {
      std::printf("  backend has no bf16 weight GEMM — skipped\n");
    } else {
      std::vector<int64_t> want, got;
      {
        qm::reset_cache(B);
        int64_t p = 0;
        std::vector<float> lg;
        for (int64_t i = 0; i < NP; i++) lg = qm::step(B, ids[i], p++);
        int64_t next = qm::argmax(lg);
        for (int64_t i = 0; i < NB; i++) {
          want.push_back(next);
          lg = qm::step(B, next, p++);
          next = qm::argmax(lg);
        }
      }
      qm::reset_cache(B);
      int64_t next = qm::prefill_batched(B, ids);
      int64_t p = qm::seq_cache(B).pos;
      for (int64_t i = 0; i < NB; i++) {
        got.push_back(next);
        next = qm::step_imperative(B, next, p++);
      }
      std::printf("  array:");
      for (auto t : want) std::printf(" %lld", (long long)t);
      std::printf("\n  fused:");
      for (auto t : got) std::printf(" %lld", (long long)t);
      bf16_ok = got == want;
      std::printf("\n  greedy %s\n", bf16_ok ? "MATCH" : "DIVERGE");
    }
  }

  bool ok = ok_f32 && bf16_ok;
  std::printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
