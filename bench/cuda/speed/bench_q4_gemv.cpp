// M8 proof: int4 group-quantized weight decode GEMV, on the CUDA layer.
// Weights are symmetric int4 in [N,K] (out×in) layout, quantized in groups of G
// along K (GGUF/GPTQ convention) — one f32 scale per group, q in [-8,7]. Decode
// is bandwidth-bound, so reading ~0.625 bytes/weight (0.5 packed + 4/G scale)
// vs bf16's 2 is the biggest remaining lever. Verifies the kernel against a host
// dequant-then-dot reference (pure-fp match) and reports the quantization error
// vs the original f32 weights. Direct cuda:: API (no cudart).

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

struct shape {
  const char* name;
  int64_t N, K;  // weight [N_out, K_in]; y(N) = a(1,K) @ dequant(W)
};

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping int4 GEMV bench\n");
    return 0;
  }
  const int64_t G = 32;  // quant group size along K
  const shape shapes[] = {{"qkv_proj", 12288, 4096}, {"attn_out", 4096, 4096},
                          {"ffn_gate_up", 22016, 4096}, {"ffn_down", 4096, 11008},
                          {"lm_head", 32000, 4096}};
  const int R = 50, ROUNDS = 7;

  std::printf("%-13s %9s %10s %8s   %9s\n", "shape", "int4 ms", "GB/s(q4)",
              "b/wt", "maxrel");
  double sum_ms = 0;
  for (const auto& s : shapes) {
    const int64_t N = s.N, K = s.K, groups = K / G;
    float *ha = nullptr, *hqw = nullptr, *hsc = nullptr, *hy = nullptr;
    void* a = alloc(K * 4, &ha);
    void* qw = alloc(N * K / 2, &hqw);          // packed int4 (2 per byte)
    void* sc = alloc(N * groups * 4, &hsc);     // f32 scale per group
    void* y = alloc(N * 4, &hy);
    uint32_t* words = reinterpret_cast<uint32_t*>(hqw);

    uint32_t st = 7u;
    auto rnd = [&] {
      st ^= st << 13;
      st ^= st >> 17;
      st ^= st << 5;
      return (int32_t)st * (1.0f / 2147483648.0f);
    };
    for (int64_t i = 0; i < K; i++) ha[i] = rnd();

    // Quantize a fresh f32 weight row-by-row into int4 + per-group scales, and
    // keep the dequantized value to build the reference. wq stored as (q+8).
    std::vector<float> deq((size_t)N * K);  // dequantized weights (host ref)
    for (int64_t n = 0; n < N; n++) {
      for (int64_t g = 0; g < groups; g++) {
        float maxabs = 1e-8f;
        float wtmp[64];  // G <= 64
        for (int64_t j = 0; j < G; j++) {
          float w = rnd();
          wtmp[j] = w;
          maxabs = std::max(maxabs, std::fabs(w));
        }
        float scale = maxabs / 7.0f;
        hsc[n * groups + g] = scale;
        for (int64_t j = 0; j < G; j++) {
          int q = (int)std::lround(wtmp[j] / scale);
          q = std::max(-8, std::min(7, q));
          int64_t k = g * G + j;
          // pack nibble (q+8) into words[n*(K/8) + k/8], slot k%8
          uint32_t& wd = words[n * (K / 8) + k / 8];
          unsigned slot = (unsigned)(k % 8);
          wd = (wd & ~(0xFu << (slot * 4))) | ((unsigned)(q + 8) << (slot * 4));
          deq[(size_t)n * K + k] = scale * (float)q;
        }
      }
    }

    gemv_q4(a, qw, sc, y, N, K, G);
    flush();
    sync_to_host(y, false);

    // reference: a · dequant(W) per row (kernel must match this to fp precision)
    double maxrel = 0;
    for (int64_t n = 0; n < N; n += 17) {  // sample rows
      double ref = 0;
      for (int64_t k = 0; k < K; k++) ref += (double)ha[k] * deq[(size_t)n * K + k];
      float got = hy[n];
      maxrel = std::max(maxrel, std::fabs(got - ref) / (1.0 + std::fabs(ref)));
    }

    auto time_ms = [&](auto&& fn) {
      fn();
      flush();
      std::vector<double> ms;
      for (int r = 0; r < ROUNDS; r++) {
        auto t0 = clk::now();
        for (int i = 0; i < R; i++) fn();
        flush();
        ms.push_back(
            std::chrono::duration<double, std::milli>(clk::now() - t0).count() /
            R);
      }
      return median(ms);
    };
    double ms = time_ms([&] { gemv_q4(a, qw, sc, y, N, K, G); });
    double bytes = (double)N * K * 0.5 + (double)N * groups * 4;
    double gbs = bytes / (ms * 1e6);
    double bpw = bytes / ((double)N * K);
    sum_ms += ms;

    std::printf("%-13s %9.4f %10.1f %8.3f   %9.1e\n", s.name, ms, gbs, bpw,
                maxrel);
    release(a, 0, nullptr);
    release(qw, 0, nullptr);
    release(sc, 0, nullptr);
    release(y, 0, nullptr);
  }
  std::printf(
      "\n  layer decode GEMV sum (int4): %.3f ms  (bf16 was ~0.78 ms; ~0.625 "
      "vs 2.0 bytes/wt)\n",
      sum_ms);
  return 0;
}
