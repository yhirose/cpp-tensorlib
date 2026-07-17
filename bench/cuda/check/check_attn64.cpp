// M9 head_dim generalization proof: the attention/KV kernels are templated on
// head_dim and instantiated for {64,128}. The existing suite covers D=128; this
// exercises the D=64 instantiations (Qwen2 = 896/14 = 64) against the same
// from-scratch CPU references that validate D=128 in bench_attn_decode. Covers
// every head-dim-specific kernel at D=64: kv_append, decode attention (both the
// single-block and the split-KV path, straddling the ctx>=256 boundary), kv_fill,
// causal prefill, and the prefill->decode handoff. Correctness-only (no timing);
// returns nonzero on any mismatch, so it gates the generalization as a ctest.
//
// GQA shape 14 q / 2 kv (group 7) mirrors the real first target Qwen2.5-0.5B.

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tl::cuda;

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping D=64 attention check\n");
    return 0;
  }
  const int64_t D = 64, HQ = 14, HKV = 2, group = HQ / HKV, MAXC = 2048;
  const float scale = 1.0f / std::sqrt((float)D);
  std::printf("D=64 attention check — %lld q / %lld kv heads (GQA %lld), D=%lld\n",
              (long long)HQ, (long long)HKV, (long long)group, (long long)D);
  bool ok = true;

  uint32_t st = 20260709u;
  auto rnd = [&] {
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (int32_t)st * (1.0f / 2147483648.0f);
  };

  // ---- KV cache + GQA at D=64: grow one token at a time, verify prefix reads +
  // GQA mapping at checkpoints straddling the split-KV boundary (ctx>=256). ----
  {
    kv_cache cache;
    if (!cache.init(HKV, MAXC, D)) { std::printf("  cache init failed\n"); return 1; }
    float *hkn = nullptr, *hvn = nullptr, *hq = nullptr, *ho = nullptr;
    void* kn = alloc(HKV * D * 4, &hkn);
    void* vn = alloc(HKV * D * 4, &hvn);
    void* q = alloc(HQ * D * 4, &hq);
    void* o = alloc(HQ * D * 4, &ho);
    std::vector<float> Khist((size_t)HKV * MAXC * D), Vhist((size_t)HKV * MAXC * D);

    const int64_t checks[] = {1, 2, 63, 64, 127, 128, 255, 256, 257, 512, 600};
    size_t ci = 0;
    double maxrel = 0;
    std::vector<float> scores(MAXC);
    const int64_t STEPS = 600;
    for (int64_t step = 1; step <= STEPS; step++) {
      int64_t pos = step - 1;
      for (int64_t i = 0; i < HKV * D; i++) {
        float kv = rnd();
        hkn[i] = kv; hvn[i] = rnd();
        int64_t head = i / D, d = i % D;
        Khist[(size_t)head * MAXC * D + pos * D + d] = kv;
        Vhist[(size_t)head * MAXC * D + pos * D + d] = hvn[i];
      }
      sync_to_host(kn, true);
      sync_to_host(vn, true);
      if (!cache.append(kn, vn)) { std::printf("  append failed pos %lld\n", (long long)pos); return 1; }

      if (ci < sizeof(checks) / sizeof(checks[0]) && step == checks[ci]) {
        ci++;
        for (int64_t i = 0; i < HQ * D; i++) hq[i] = rnd();
        sync_to_host(q, true);
        cache.attn(q, o, HQ, scale);
        flush();
        sync_to_host(o, false);
        int64_t L = step;
        for (int64_t qh = 0; qh < HQ; qh++) {
          int64_t kvh = qh / group;
          const float* qhp = hq + qh * D;
          const float* Kh = Khist.data() + (size_t)kvh * MAXC * D;
          const float* Vh = Vhist.data() + (size_t)kvh * MAXC * D;
          float mx = -1e30f;
          for (int64_t j = 0; j < L; j++) {
            float s = 0;
            for (int64_t d = 0; d < D; d++) s += qhp[d] * Kh[j * D + d];
            scores[j] = s * scale;
            mx = std::max(mx, scores[j]);
          }
          float sum = 0;
          for (int64_t j = 0; j < L; j++) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
          for (int64_t d = 0; d < D; d++) {
            float acc = 0;
            for (int64_t j = 0; j < L; j++) acc += scores[j] * Vh[j * D + d];
            float ref = acc / sum, got = ho[qh * D + d];
            maxrel = std::max<double>(maxrel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
          }
        }
      }
    }
    ok &= maxrel < 1e-4;
    std::printf("  kv cache grow to %lld tokens (%zu checkpoints, split-KV @>=256) "
                "— maxrel %.1e %s\n", (long long)cache.pos,
                sizeof(checks) / sizeof(checks[0]), maxrel, maxrel < 1e-4 ? "OK" : "FAIL");
    cache.destroy();
    release(kn, 0, nullptr); release(vn, 0, nullptr);
    release(q, 0, nullptr);  release(o, 0, nullptr);
  }

  // ---- Causal prefill at D=64 + prefill->decode handoff. ----
  {
    const int64_t T = 300;
    float *hks = nullptr, *hvs = nullptr, *hqp = nullptr, *hop = nullptr;
    void* ks = alloc(HKV * T * D * 4, &hks);
    void* vs = alloc(HKV * T * D * 4, &hvs);
    void* qp = alloc(HQ * T * D * 4, &hqp);
    void* op = alloc(HQ * T * D * 4, &hop);
    for (int64_t i = 0; i < HKV * T * D; i++) { hks[i] = rnd(); hvs[i] = rnd(); }
    for (int64_t i = 0; i < HQ * T * D; i++) hqp[i] = rnd();

    kv_cache cache;
    if (!cache.init(HKV, MAXC, D)) { std::printf("  cache init failed\n"); return 1; }
    cache.prefill(qp, ks, vs, op, T, HQ, scale);
    flush();
    sync_to_host(op, false);

    double maxrel = 0;
    std::vector<float> scores(T);
    for (int64_t qh = 0; qh < HQ; qh++) {
      int64_t kvh = qh / group;
      const float* Kh = hks + kvh * T * D;
      const float* Vh = hvs + kvh * T * D;
      for (int64_t p = 0; p < T; p++) {
        const float* qhp = hqp + (qh * T + p) * D;
        float mx = -1e30f;
        for (int64_t j = 0; j <= p; j++) {
          float s = 0;
          for (int64_t d = 0; d < D; d++) s += qhp[d] * Kh[j * D + d];
          scores[j] = s * scale;
          mx = std::max(mx, scores[j]);
        }
        float sum = 0;
        for (int64_t j = 0; j <= p; j++) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
        for (int64_t d = 0; d < D; d++) {
          float acc = 0;
          for (int64_t j = 0; j <= p; j++) acc += scores[j] * Vh[j * D + d];
          float ref = acc / sum, got = hop[(qh * T + p) * D + d];
          maxrel = std::max<double>(maxrel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
        }
      }
    }
    ok &= maxrel < 1e-4;
    std::printf("  causal prefill T=%lld — maxrel %.1e %s (cache pos %lld)\n",
                (long long)T, maxrel, maxrel < 1e-4 ? "OK" : "FAIL", (long long)cache.pos);

    // decode handoff: append one token at pos=T, attend over 0..T.
    float *hkn = nullptr, *hvn = nullptr, *hq1 = nullptr, *ho1 = nullptr;
    void* kn = alloc(HKV * D * 4, &hkn);
    void* vn = alloc(HKV * D * 4, &hvn);
    void* q1 = alloc(HQ * D * 4, &hq1);
    void* o1 = alloc(HQ * D * 4, &ho1);
    for (int64_t i = 0; i < HKV * D; i++) { hkn[i] = rnd(); hvn[i] = rnd(); }
    for (int64_t i = 0; i < HQ * D; i++) hq1[i] = rnd();
    sync_to_host(kn, true); sync_to_host(vn, true); sync_to_host(q1, true);
    cache.append(kn, vn);
    cache.attn(q1, o1, HQ, scale);
    flush();
    sync_to_host(o1, false);

    double drel = 0;
    std::vector<float> sc(T + 1);
    for (int64_t qh = 0; qh < HQ; qh++) {
      int64_t kvh = qh / group;
      const float* Kh = hks + kvh * T * D;
      const float* Vh = hvs + kvh * T * D;
      const float* q1p = hq1 + qh * D;
      float mx = -1e30f;
      for (int64_t j = 0; j <= T; j++) {
        const float* kj = (j < T) ? Kh + j * D : hkn + kvh * D;
        float s = 0;
        for (int64_t d = 0; d < D; d++) s += q1p[d] * kj[d];
        sc[j] = s * scale;
        mx = std::max(mx, sc[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j <= T; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
      for (int64_t d = 0; d < D; d++) {
        float acc = 0;
        for (int64_t j = 0; j <= T; j++) {
          const float* vj = (j < T) ? Vh + j * D : hvn + kvh * D;
          acc += sc[j] * vj[d];
        }
        float ref = acc / sum, got = ho1[qh * D + d];
        drel = std::max<double>(drel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
      }
    }
    ok &= drel < 1e-4;
    std::printf("  prefill->decode handoff @pos=%lld — maxrel %.1e %s\n",
                (long long)cache.pos - 1, drel, drel < 1e-4 ? "OK" : "FAIL");

    cache.destroy();
    release(ks, 0, nullptr); release(vs, 0, nullptr);
    release(qp, 0, nullptr); release(op, 0, nullptr);
    release(kn, 0, nullptr); release(vn, 0, nullptr);
    release(q1, 0, nullptr); release(o1, 0, nullptr);
  }

  std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
