// M9 "actually chat" — numeric proof. Load a real Qwen2.5-0.5B-Instruct GGUF (F16
// weights) and run the decoder end to end, verified against a tight numpy
// reference that reads the SAME F16 weights (qwen_oracle_data.h here). Identical
// weights => the C++ forward must match the reference to ~1e-4 at every checkpoint
// (embedding, layer-0 residual, final norm, top-5 logits) AND reproduce the greedy
// token sequence exactly. Proves weight loading + GGML->our layout transpose +
// Qwen2 structure + the head_dim=64 attention kernels, on whatever backend
// gpu:: resolves to. Tokenizer/sampling are
// separate (see chat_qwen). Model path: argv[1] or ~/models/...fp16.gguf.

#include "qwen2.h"
#include "qwen_oracle_data.h"

#include <cstdio>
#include <vector>

namespace qm = qwenmodel;

static double maxrel(const std::vector<float>& a, const float* b, int64_t n) {
  double mr = 0;
  for (int64_t i = 0; i < n; i++)
    mr = std::max(mr, std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)b[i])));
  return mr;
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no GPU device — skipping Qwen forward check\n");
    return 0;
  }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  std::printf("loading %s\n", path.c_str());
  qm::gg::model m(path);
  if (!qm::check_config(m)) {
    std::printf("config mismatch — this driver is hard-coded for Qwen2.5-0.5B\n");
    return 2;
  }
  tl::use_gpu();
  std::printf("building model (F32, transposing linear weights)...\n");
  const int64_t NP = sizeof(qwenoracle::prompt_ids) / sizeof(int);
  bool imp_ok = true, bf16_ok = true, ok_f32 = false;
  {  // scoped so the F32 weights are back in the pool before the bf16 build
  qm::Model M = qm::build(m);
  std::printf("model built — %lld layers, GQA %lldq/%lldkv, head_dim=%lld\n",
              (long long)qm::NL, (long long)qm::NH, (long long)qm::NKV, (long long)qm::HD);

  std::vector<float> l0, fnorm, last_logits;
  int64_t pos = 0;
  for (int64_t i = 0; i < NP; i++) {
    bool last = (i == NP - 1);
    last_logits = qm::step(M, qwenoracle::prompt_ids[i], pos,
                           last ? &l0 : nullptr, last ? &fnorm : nullptr);
    pos++;
  }
  tl::array el = qm::embed_row(M, qwenoracle::prompt_ids[NP - 1]);
  el.eval();
  double emb_mr = maxrel(std::vector<float>(el.raw(), el.raw() + qm::NE),
                         qwenoracle::emb_last, qm::NE);
  double l0_mr = maxrel(l0, qwenoracle::l0_last, qm::NE);
  double fn_mr = maxrel(fnorm, qwenoracle::fnorm_last, qm::NE);

  std::vector<int64_t> top5;
  { std::vector<float> lg = last_logits;
    for (int t = 0; t < 5; t++) { int64_t bi = qm::argmax(lg); top5.push_back(bi); lg[bi] = -1e30f; } }
  bool top1_ok = top5[0] == qwenoracle::top5_ids[0];
  double logit_mr = 0;
  for (int t = 0; t < 5; t++) {
    int64_t id = qwenoracle::top5_ids[t];
    logit_mr = std::max(logit_mr, std::fabs((double)last_logits[id] - qwenoracle::top5_vals[t]) /
                                      (1.0 + std::fabs((double)qwenoracle::top5_vals[t])));
  }

  std::printf("\ncheckpoints vs numpy-on-same-F16-weights (expect ~1e-4):\n");
  std::printf("  embedding  maxrel %.2e %s\n", emb_mr, emb_mr < 1e-3 ? "OK" : "FAIL");
  std::printf("  layer0 res maxrel %.2e %s\n", l0_mr, l0_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top5[0],
              qwenoracle::top5_ids[0], top1_ok ? "OK" : "FAIL");

  const int64_t NG = sizeof(qwenoracle::greedy30) / sizeof(int);
  std::vector<int64_t> gen;
  int64_t next = qm::argmax(last_logits);
  for (int64_t i = 0; i < NG; i++) {
    gen.push_back(next);
    std::vector<float> lg = qm::step(M, next, pos);
    pos++;
    next = qm::argmax(lg);
  }
  bool greedy_ok = true;
  for (int64_t i = 0; i < NG; i++) if (gen[i] != qwenoracle::greedy30[i]) greedy_ok = false;
  std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
  for (auto t : gen) std::printf(" %lld", (long long)t);
  std::printf("\n  ref:");
  for (int64_t i = 0; i < NG; i++) std::printf(" %d", qwenoracle::greedy30[i]);
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
    for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(M, qwenoracle::prompt_ids[i], p++);
    std::printf("\nimperative path (same weights, fused kernels):\n  cpp:");
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      if (tok != qwenoracle::greedy30[i]) imp_ok = false;
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
    std::vector<int> ids(qwenoracle::prompt_ids, qwenoracle::prompt_ids + NP);
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
      int64_t p = B.layers[0].cache.pos;
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
