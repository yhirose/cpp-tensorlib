// M9 "actually chat" — numeric proof. Load a real Qwen2.5-0.5B-Instruct GGUF (F16
// weights) and run the decoder end to end, verified against a tight numpy
// reference that reads the SAME F16 weights (bench/cuda/qwen_oracle_data.h). Identical
// weights => the C++ forward must match the reference to ~1e-4 at every checkpoint
// (embedding, layer-0 residual, final norm, top-5 logits) AND reproduce the greedy
// token sequence exactly. Proves weight loading + GGML->our layout transpose +
// Qwen2 structure + the head_dim=64 attention kernels. Tokenizer/sampling are
// separate (see chat_qwen). Model path: argv[1] or ~/models/...fp16.gguf.

#include "qwen_model.h"
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
    std::printf("no CUDA device — skipping Qwen forward check\n");
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
  qm::Model M = qm::build(m);
  std::printf("model built — %lld layers, GQA %lldq/%lldkv, head_dim=%lld\n",
              (long long)qm::NL, (long long)qm::NH, (long long)qm::NKV, (long long)qm::HD);

  const int64_t NP = sizeof(qwenoracle::prompt_ids) / sizeof(int);
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

  bool ok = emb_mr < 1e-3 && l0_mr < 5e-3 && fn_mr < 5e-3 && logit_mr < 5e-3 &&
            top1_ok && greedy_ok;
  std::printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
