// M7 Path-B proof: bf16-weight decode GEMV vs f32-weight, on the CUDA layer.
// Isolates the storage-width lever without touching the array/dtype system —
// the cuda:: facade takes opaque device buffers, so a bf16 weight buffer + a
// widen-on-load kernel are all it takes. Decode is batch~1 (GEMV) and memory-
// bandwidth-bound: the K×N weight B dominates traffic, so halving its bytes
// (f32->bf16) should ~halve the time. Both kernels are structurally identical;
// only B's dtype differs, so f32_ms/bf16_ms isolates the win, and the achieved
// GB/s (near the device's peak) confirms both are bandwidth-bound. Weights round
// to bf16 (RNE); the accumulator stays F32, so the error is bf16's 8-bit
// mantissa on the weights only.
//
// Direct cuda:: API (like bench_cuda_gemm) but no cudart/cuBLAS: timing is
// steady_clock around cuda::flush() (driver CtxSynchronize) over R launches.
// Measured on the RTX 3090 under WSL2. Shapes are a llama-7B decoder layer.

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tl::cuda;
using clk = std::chrono::steady_clock;

// f32 -> bf16 with round-to-nearest-even (top 16 bits + rounding bias).
static uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct shape {
  const char* name;
  int64_t K, N;  // y(N) = a(1,K) @ B(K,N)
};

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping bf16 GEMV bench\n");
    return 0;
  }

  // llama-7B decoder-layer weight shapes (K=in, N=out), decode = 1 row.
  const shape shapes[] = {{"qkv_proj", 4096, 12288}, {"attn_out", 4096, 4096},
                          {"ffn_gate_up", 4096, 22016}, {"ffn_down", 11008, 4096},
                          {"lm_head", 4096, 32000}};
  const int R = 50, ROUNDS = 7;

  std::printf("%-13s %9s %9s %7s   %9s %9s %9s\n", "shape", "f32 ms", "bf16 ms",
              "x", "f32 GB/s", "bf16 GB/s", "maxrel");
  double sum_f32 = 0, sum_bf16 = 0;
  for (const auto& s : shapes) {
    const int64_t K = s.K, N = s.N;
    float *ha = nullptr, *hBf = nullptr, *hBb = nullptr, *hyf = nullptr,
          *hyb = nullptr;
    void* a = alloc(K * 4, &ha);
    void* Bf = alloc(K * N * 4, &hBf);
    void* Bb = alloc(K * N * 2, &hBb);  // bf16 weights (half the bytes)
    void* yf = alloc(N * 4, &hyf);
    void* yb = alloc(N * 4, &hyb);

    uint32_t st = 12345u;
    auto rnd = [&] {
      st ^= st << 13;
      st ^= st >> 17;
      st ^= st << 5;
      return static_cast<int32_t>(st) * (1.0f / 2147483648.0f);
    };
    for (int64_t i = 0; i < K; i++) ha[i] = rnd();
    uint16_t* Bb16 = reinterpret_cast<uint16_t*>(hBb);
    for (int64_t i = 0; i < K * N; i++) {
      float v = rnd();
      hBf[i] = v;
      Bb16[i] = f32_to_bf16(v);
    }

    // correctness: bf16-weight result vs f32-weight result
    gemv_f32(a, Bf, yf, N, K);
    gemv_bf16(a, Bb, yb, N, K);
    flush();
    sync_to_host(yf, false);
    sync_to_host(yb, false);
    double maxrel = 0;
    for (int64_t i = 0; i < N; i++) {
      double e = std::fabs(hyf[i] - hyb[i]) / (1.0 + std::fabs(hyf[i]));
      maxrel = std::max(maxrel, e);
    }

    auto time_ms = [&](auto&& fn) {
      fn();
      flush();  // warmup
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
    double f32_ms = time_ms([&] { gemv_f32(a, Bf, yf, N, K); });
    double bf16_ms = time_ms([&] { gemv_bf16(a, Bb, yb, N, K); });
    double f32_gbs = static_cast<double>(K) * N * 4 / (f32_ms * 1e6);
    double bf16_gbs = static_cast<double>(K) * N * 2 / (bf16_ms * 1e6);
    sum_f32 += f32_ms;
    sum_bf16 += bf16_ms;

    std::printf("%-13s %9.4f %9.4f %6.2fx   %9.1f %9.1f %9.1e\n", s.name, f32_ms,
                bf16_ms, f32_ms / bf16_ms, f32_gbs, bf16_gbs, maxrel);

    release(a, 0, nullptr);
    release(Bf, 0, nullptr);
    release(Bb, 0, nullptr);
    release(yf, 0, nullptr);
    release(yb, 0, nullptr);
  }
  std::printf("\n  layer decode GEMV sum: f32 %.3f ms  bf16 %.3f ms  (%.2fx)\n",
              sum_f32, sum_bf16, sum_f32 / sum_bf16);
  return 0;
}
