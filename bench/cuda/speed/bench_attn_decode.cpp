// M9 proof: fused decode attention vs the unfused 3-op path, on the CUDA layer.
// The decode-token time is dominated by attention (bench_llm: ~72%), but that
// cost is launch/materialize overhead — the array path runs q·Kᵀ, softmax, ·V
// as 3 separate kernels with 2 intermediate materializations, per head per
// layer (heads×layers ≈ 1024 times). tl_attn_decode_f32 fuses them with the
// online-softmax recurrence: the ctx-long scores never touch memory and K,V are
// read exactly once, so the kernel runs at the decode floor (KV bandwidth).
//
// Verifies against a CPU reference (scale·q·Kᵀ → softmax → ·V), then reports the
// fused ms for one layer's heads + achieved KV GB/s + the implied full-model
// (×layers) decode-attention time to compare against bench_llm's number.
// Direct cuda:: API, no cudart (timing via cuda::flush + steady_clock).

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tl::cuda;
using clk = std::chrono::steady_clock;

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping attention bench\n");
    return 0;
  }
  const int64_t H = 32, D = 128, LAYERS = 32;  // llama-7B decoder
  const float scale = 1.0f / std::sqrt((float)D);
  const int R = 30, ROUNDS = 7;
  const int64_t ctxs[] = {512, 1024, 2048, 4096};

  std::printf("fused decode attention — H=%lld heads, D=%lld, scale=1/sqrt(D)\n",
              (long long)H, (long long)D);
  std::printf("f32 = 4B K,V (baseline); bf16 = 2B K,V (M9 KV bandwidth lever, "
              "~2x the KV floor). maxrel is vs the f32 CPU reference.\n");
  std::printf("%-8s %9s %9s %8s   %9s %9s %9s   %10s\n", "ctx", "f32 ms",
              "bf16 ms", "f32/bf16", "f32 GB/s", "bf16 GB/s", "bf16 maxrel",
              "model bf16");
  for (int64_t ctx : ctxs) {
    float *hq = nullptr, *hK = nullptr, *hV = nullptr, *ho = nullptr;
    void* q = alloc(H * D * 4, &hq);
    void* K = alloc(H * ctx * D * 4, &hK);
    void* V = alloc(H * ctx * D * 4, &hV);
    void* o = alloc(H * D * 4, &ho);
    // bf16 K,V cache holding the same values (narrowed): filled once via kv_fill
    // (f32 in -> bf16 out), the exact write path the model uses.
    void* Kb = alloc(H * ctx * D * 2, nullptr);
    void* Vb = alloc(H * ctx * D * 2, nullptr);

    uint32_t st = 99u;
    auto rnd = [&] {
      st ^= st << 13;
      st ^= st >> 17;
      st ^= st << 5;
      return (int32_t)st * (1.0f / 2147483648.0f);
    };
    for (int64_t i = 0; i < H * D; i++) hq[i] = rnd();
    for (int64_t i = 0; i < H * ctx * D; i++) {
      hK[i] = rnd();
      hV[i] = rnd();
    }
    // Narrow the f32 K,V into the bf16 cache buffers (kv_fill: [H,ctx,D] -> cache
    // rows [0,ctx), kv_max=ctx so no padding). H_kv = H here (no GQA).
    kv_fill(Kb, Vb, K, V, ctx, ctx, H, D, /*kv_bf16=*/true);
    flush();

    auto run = [&](bool bf16) {
      attn_decode(q, bf16 ? Kb : K, bf16 ? Vb : V, o, H, H, ctx, ctx, D, scale,
                  bf16);
    };
    run(true);
    flush();
    sync_to_host(o, false);

    // CPU reference per head: softmax(scale·q·Kᵀ)·V (f32 values). Compared to the
    // bf16-cache run below — so maxrel is the bf16 quantization error, end to end.
    double maxrel = 0;
    std::vector<float> scores(ctx);
    for (int64_t h = 0; h < H; h++) {
      const float* qh = hq + h * D;
      const float* Kh = hK + h * ctx * D;
      const float* Vh = hV + h * ctx * D;
      float mx = -1e30f;
      for (int64_t j = 0; j < ctx; j++) {
        float s = 0;
        for (int64_t d = 0; d < D; d++) s += qh[d] * Kh[j * D + d];
        scores[j] = s * scale;
        mx = std::max(mx, scores[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j < ctx; j++) {
        scores[j] = std::exp(scores[j] - mx);
        sum += scores[j];
      }
      for (int64_t d = 0; d < D; d++) {
        float acc = 0;
        for (int64_t j = 0; j < ctx; j++) acc += scores[j] * Vh[j * D + d];
        float ref = acc / sum;
        float got = ho[h * D + d];
        maxrel = std::max<double>(maxrel,
                                  std::fabs(got - ref) / (1.0 + std::fabs(ref)));
      }
    }

    auto time = [&](bool bf16) {
      std::vector<double> ms;
      for (int r = 0; r < ROUNDS; r++) {
        auto t0 = clk::now();
        for (int i = 0; i < R; i++) run(bf16);
        flush();
        ms.push_back(
            std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
      }
      return median(ms);
    };
    double f32_ms = time(false), bf16_ms = time(true);
    double f32_gbs = 2.0 * H * ctx * D * 4 / (f32_ms * 1e6);
    double bf16_gbs = 2.0 * H * ctx * D * 2 / (bf16_ms * 1e6);
    std::printf("%-8lld %9.4f %9.4f %8.2f   %9.1f %9.1f %9.1e   %8.2f ms\n",
                (long long)ctx, f32_ms, bf16_ms, f32_ms / bf16_ms, f32_gbs,
                bf16_gbs, maxrel, bf16_ms * LAYERS);

    release(q, 0, nullptr);
    release(K, 0, nullptr);
    release(V, 0, nullptr);
    release(o, 0, nullptr);
    release(Kb, 0, nullptr);
    release(Vb, 0, nullptr);
  }
  std::printf(
      "\n(compare model ms vs bench_llm's unfused decode attention ~48 ms)\n");

  // ---- REAL Qwen2.5-0.5B decode shape: HQ=14, HKV=2, D=64 (the actual target).
  // Tiny KV footprint (2 kv heads x ctx x 64) at short chat contexts, so this is
  // the honest test of whether bf16 KV moves the needle on THIS model vs the
  // llama-7B ceiling above. Per-layer attn ms x24 layers = the full-model delta.
  {
    const int64_t HQ = 14, HKV = 2, D = 64;
    const int64_t group = HQ / HKV, LY = 24;
    const float sc = 1.0f / std::sqrt((float)D);
    const int64_t qctxs[] = {128, 256, 512, 1024, 2048};
    std::printf("\nQwen2.5-0.5B decode attn — %lld q / %lld kv heads, D=%lld, "
                "x%lld layers\n", (long long)HQ, (long long)HKV, (long long)D,
                (long long)LY);
    std::printf("%-8s %9s %9s %8s   %11s %11s\n", "ctx", "f32 us", "bf16 us",
                "f32/bf16", "f32 model", "bf16 model");
    for (int64_t ctx : qctxs) {
      float *hq = nullptr, *hK = nullptr, *hV = nullptr;
      void* q = alloc(HQ * D * 4, &hq);
      void* K = alloc(HKV * ctx * D * 4, &hK);
      void* V = alloc(HKV * ctx * D * 4, &hV);
      void* o = alloc(HQ * D * 4, nullptr);
      void* Kb = alloc(HKV * ctx * D * 2, nullptr);
      void* Vb = alloc(HKV * ctx * D * 2, nullptr);
      uint32_t st = 7u;
      auto rnd = [&] { st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                       return (int32_t)st * (1.0f / 2147483648.0f); };
      for (int64_t i = 0; i < HQ * D; i++) hq[i] = rnd();
      for (int64_t i = 0; i < HKV * ctx * D; i++) { hK[i] = rnd(); hV[i] = rnd(); }
      kv_fill(Kb, Vb, K, V, ctx, ctx, HKV, D, true);
      flush();
      auto run = [&](bool bf16) {
        attn_decode(q, bf16 ? Kb : K, bf16 ? Vb : V, o, HQ, HKV, ctx, ctx, D, sc,
                    bf16);
      };
      auto time = [&](bool bf16) {
        run(bf16); flush();
        std::vector<double> ms;
        for (int r = 0; r < ROUNDS; r++) {
          auto t0 = clk::now();
          for (int i = 0; i < R; i++) run(bf16);
          flush();
          ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
        }
        return median(ms);
      };
      double f = time(false), b = time(true);
      std::printf("%-8lld %9.2f %9.2f %8.2f   %8.3f ms %8.3f ms\n", (long long)ctx,
                  f * 1000, b * 1000, f / b, f * LY, b * LY);
      release(q, 0, nullptr); release(K, 0, nullptr); release(V, 0, nullptr);
      release(o, 0, nullptr); release(Kb, 0, nullptr); release(Vb, 0, nullptr);
    }
  }

  // ---- KV cache + GQA: grow the cache one token at a time, verify prefix reads
  // + the GQA head mapping against a from-scratch CPU reference at checkpoint
  // positions (covering the split-KV boundary at ctx>=256 and its rounding). ----
  {
    const int64_t HQ = 32, HKV = 8, D = 128, MAXC = 2048;  // llama-3 style GQA
    const int64_t group = HQ / HKV;
    const float scale = 1.0f / std::sqrt((float)D);
    std::printf(
        "\nKV cache + GQA — %lld q heads / %lld kv heads (group %lld), "
        "max_ctx=%lld\n",
        (long long)HQ, (long long)HKV, (long long)group, (long long)MAXC);

    kv_cache cache;
    if (!cache.init(HKV, MAXC, D)) {
      std::printf("  cache init failed\n");
      return 1;
    }
    // Per-step k,v projections ([HKV,D]) and the query ([HQ,D]); host mirrors.
    float *hkn = nullptr, *hvn = nullptr, *hq = nullptr, *ho = nullptr;
    void* kn = alloc(HKV * D * 4, &hkn);
    void* vn = alloc(HKV * D * 4, &hvn);
    void* q = alloc(HQ * D * 4, &hq);
    void* o = alloc(HQ * D * 4, &ho);

    // Full host history of what we appended, laid out exactly like the device
    // cache ([HKV, MAXC, D]) so the reference indexing mirrors the kernel.
    std::vector<float> Khist((size_t)HKV * MAXC * D);
    std::vector<float> Vhist((size_t)HKV * MAXC * D);

    uint32_t st = 1234567u;
    auto rnd = [&] {
      st ^= st << 13;
      st ^= st >> 17;
      st ^= st << 5;
      return (int32_t)st * (1.0f / 2147483648.0f);
    };

    // Checkpoints: split-KV kicks in at ctx>=256; probe around that boundary and
    // the chunk rounding, plus the single-block regime and a long context.
    const int64_t checks[] = {1,   2,   127, 128,  129,  255,
                              256, 257, 512, 1024, 2048};
    size_t ci = 0;
    double maxrel = 0;
    std::vector<float> scores(MAXC);
    for (int64_t step = 1; step <= MAXC; step++) {
      int64_t pos = step - 1;  // row this token lands on
      for (int64_t i = 0; i < HKV * D; i++) {
        float kv = rnd();
        hkn[i] = kv;
        hvn[i] = rnd();
        // mirror into the host history at [head, pos, d]
        int64_t head = i / D, d = i % D;
        Khist[(size_t)head * MAXC * D + pos * D + d] = kv;
        Vhist[(size_t)head * MAXC * D + pos * D + d] = hvn[i];
      }
      sync_to_host(kn, true);  // host just wrote → re-upload on next device read
      sync_to_host(vn, true);
      if (!cache.append(kn, vn)) {
        std::printf("  append failed at pos %lld\n", (long long)pos);
        return 1;
      }

      if (ci < sizeof(checks) / sizeof(checks[0]) && step == checks[ci]) {
        ci++;
        for (int64_t i = 0; i < HQ * D; i++) hq[i] = rnd();
        sync_to_host(q, true);
        cache.attn(q, o, HQ, scale);
        flush();
        sync_to_host(o, false);

        int64_t L = step;  // valid length = number of tokens appended
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
          for (int64_t j = 0; j < L; j++) {
            scores[j] = std::exp(scores[j] - mx);
            sum += scores[j];
          }
          for (int64_t d = 0; d < D; d++) {
            float acc = 0;
            for (int64_t j = 0; j < L; j++) acc += scores[j] * Vh[j * D + d];
            float ref = acc / sum;
            float got = ho[qh * D + d];
            maxrel = std::max<double>(
                maxrel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
          }
        }
      }
    }
    std::printf("  grew to %lld tokens over %zu checkpoints — maxrel %.1e %s\n",
                (long long)cache.pos, sizeof(checks) / sizeof(checks[0]), maxrel,
                maxrel < 1e-4 ? "OK" : "FAIL");

    // Steady-state timing at full context (KV bandwidth = HKV*pos*D*2*4 bytes).
    std::vector<double> ms;
    for (int r = 0; r < ROUNDS; r++) {
      auto t0 = clk::now();
      for (int i = 0; i < R; i++) cache.attn(q, o, HQ, scale);
      flush();
      ms.push_back(
          std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
    }
    double layer_ms = median(ms);
    double kv_gbs = 2.0 * HKV * cache.pos * D * 4 / (layer_ms * 1e6);
    std::printf(
        "  attn @ ctx=%lld: %.4f ms/layer, %.1f KV GB/s (%lld kv heads vs %lld "
        "in MHA → %.1fx less KV traffic)\n",
        (long long)cache.pos, layer_ms, kv_gbs, (long long)HKV, (long long)HQ,
        (double)HQ / HKV);

    cache.destroy();
    release(kn, 0, nullptr);
    release(vn, 0, nullptr);
    release(q, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- Causal prefill: process a T-token prompt at once (query p attends keys
  // 0..p), fill the cache, then verify a decode step continues from pos=T. ----
  {
    const int64_t HQ = 32, HKV = 8, D = 128, MAXC = 2048, T = 512;
    const int64_t group = HQ / HKV;
    const float scale = 1.0f / std::sqrt((float)D);
    std::printf(
        "\ncausal prefill — %lld q / %lld kv heads, T=%lld, max_ctx=%lld\n",
        (long long)HQ, (long long)HKV, (long long)T, (long long)MAXC);

    float *hks = nullptr, *hvs = nullptr, *hqp = nullptr, *hop = nullptr;
    void* ks = alloc(HKV * T * D * 4, &hks);  // prompt k,v [HKV,T,D]
    void* vs = alloc(HKV * T * D * 4, &hvs);
    void* qp = alloc(HQ * T * D * 4, &hqp);  // prompt queries [HQ,T,D]
    void* op = alloc(HQ * T * D * 4, &hop);

    uint32_t st = 424242u;
    auto rnd = [&] {
      st ^= st << 13;
      st ^= st >> 17;
      st ^= st << 5;
      return (int32_t)st * (1.0f / 2147483648.0f);
    };
    for (int64_t i = 0; i < HKV * T * D; i++) {
      hks[i] = rnd();
      hvs[i] = rnd();
    }
    for (int64_t i = 0; i < HQ * T * D; i++) hqp[i] = rnd();

    kv_cache cache;
    if (!cache.init(HKV, MAXC, D)) {
      std::printf("  cache init failed\n");
      return 1;
    }
    cache.prefill(qp, ks, vs, op, T, HQ, scale);
    flush();
    sync_to_host(op, false);

    // CPU causal reference: out[qh,p,:] = softmax over keys 0..p of q·kᵀ · v.
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
        for (int64_t j = 0; j <= p; j++) {
          scores[j] = std::exp(scores[j] - mx);
          sum += scores[j];
        }
        for (int64_t d = 0; d < D; d++) {
          float acc = 0;
          for (int64_t j = 0; j <= p; j++) acc += scores[j] * Vh[j * D + d];
          float ref = acc / sum;
          float got = hop[(qh * T + p) * D + d];
          maxrel = std::max<double>(
              maxrel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
        }
      }
    }
    std::printf("  prefill T=%lld — maxrel %.1e %s (cache pos now %lld)\n",
                (long long)T, maxrel, maxrel < 1e-4 ? "OK" : "FAIL",
                (long long)cache.pos);

    // Decode continues: append one token at pos=T, attend over 0..T. Verifies
    // the prefill→decode handoff (prefill-filled rows + the appended row).
    float *hkn = nullptr, *hvn = nullptr, *hq1 = nullptr, *ho1 = nullptr;
    void* kn = alloc(HKV * D * 4, &hkn);
    void* vn = alloc(HKV * D * 4, &hvn);
    void* q1 = alloc(HQ * D * 4, &hq1);
    void* o1 = alloc(HQ * D * 4, &ho1);
    for (int64_t i = 0; i < HKV * D; i++) {
      hkn[i] = rnd();
      hvn[i] = rnd();
    }
    for (int64_t i = 0; i < HQ * D; i++) hq1[i] = rnd();
    sync_to_host(kn, true);
    sync_to_host(vn, true);
    sync_to_host(q1, true);
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
      for (int64_t j = 0; j <= T; j++) {  // keys 0..T (T from prefill + 1 new)
        const float* kj = (j < T) ? Kh + j * D : hkn + kvh * D;
        float s = 0;
        for (int64_t d = 0; d < D; d++) s += q1p[d] * kj[d];
        sc[j] = s * scale;
        mx = std::max(mx, sc[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j <= T; j++) {
        sc[j] = std::exp(sc[j] - mx);
        sum += sc[j];
      }
      for (int64_t d = 0; d < D; d++) {
        float acc = 0;
        for (int64_t j = 0; j <= T; j++) {
          const float* vj = (j < T) ? Vh + j * D : hvn + kvh * D;
          acc += sc[j] * vj[d];
        }
        float ref = acc / sum;
        float got = ho1[qh * D + d];
        drel = std::max<double>(drel,
                                std::fabs(got - ref) / (1.0 + std::fabs(ref)));
      }
    }
    std::printf("  decode step at pos=%lld — maxrel %.1e %s\n",
                (long long)cache.pos - 1, drel, drel < 1e-4 ? "OK" : "FAIL");

    // Prefill timing.
    std::vector<double> ms;
    for (int r = 0; r < ROUNDS; r++) {
      auto t0 = clk::now();
      for (int i = 0; i < R; i++)
        attn_prefill(qp, cache.K, cache.V, op, HQ, HKV, T, MAXC, D, scale);
      flush();
      ms.push_back(
          std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
    }
    double pf_ms = median(ms);
    std::printf("  prefill attn: %.4f ms for T=%lld (%.1f Ktok/s)\n", pf_ms,
                (long long)T, T / pf_ms);

    cache.destroy();
    release(ks, 0, nullptr);
    release(vs, 0, nullptr);
    release(qp, 0, nullptr);
    release(op, 0, nullptr);
    release(kn, 0, nullptr);
    release(vn, 0, nullptr);
    release(q1, 0, nullptr);
    release(o1, 0, nullptr);
  }
  return 0;
}
