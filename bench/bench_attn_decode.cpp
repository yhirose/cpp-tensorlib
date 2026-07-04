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
  std::printf("%-8s %10s %10s %10s   %14s\n", "ctx", "ms/layer", "KV GB/s",
              "maxrel", "model ms (x%d)");
  for (int64_t ctx : ctxs) {
    float *hq = nullptr, *hK = nullptr, *hV = nullptr, *ho = nullptr;
    void* q = alloc(H * D * 4, &hq);
    void* K = alloc(H * ctx * D * 4, &hK);
    void* V = alloc(H * ctx * D * 4, &hV);
    void* o = alloc(H * D * 4, &ho);

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

    attn_decode(q, K, V, o, H, ctx, D, scale);
    flush();
    sync_to_host(o, false);

    // CPU reference per head: softmax(scale·q·Kᵀ)·V
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

    std::vector<double> ms;
    for (int r = 0; r < ROUNDS; r++) {
      auto t0 = clk::now();
      for (int i = 0; i < R; i++) attn_decode(q, K, V, o, H, ctx, D, scale);
      flush();
      ms.push_back(
          std::chrono::duration<double, std::milli>(clk::now() - t0).count() /
          R);
    }
    double layer_ms = median(ms);
    double kv_gbs = 2.0 * H * ctx * D * 4 / (layer_ms * 1e6);
    std::printf("%-8lld %10.4f %10.1f %10.1e   %11.2f ms\n", (long long)ctx,
                layer_ms, kv_gbs, maxrel, layer_ms * LAYERS);

    release(q, 0, nullptr);
    release(K, 0, nullptr);
    release(V, 0, nullptr);
    release(o, 0, nullptr);
  }
  std::printf(
      "\n(compare model ms vs bench_llm's unfused decode attention ~48 ms)\n");
  return 0;
}
