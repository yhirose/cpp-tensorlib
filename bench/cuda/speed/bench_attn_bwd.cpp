// The fused causal-attention pullback against its own forward, on the CUDA
// layer: what a training step pays per attention layer, and how that splits
// between the dq kernel and the dK/dV one. Both walk the same causal half the
// forward does, with the scores in registers, so the forward is the yardstick —
// dq does two dot products per (query, key) where the forward does one, and
// dK/dV two accumulations where the forward does one.
//
// Direct cuda:: API, no cudart (timing via cuda::flush + cpu_barrier and
// steady_clock), so a change to either kernel can be measured without a
// consumer's build in the way.

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

static void fill_random(float* p, int64_t n, uint32_t seed) {
  uint32_t s = seed;
  for (int64_t i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;
    p[i] = (float)((s >> 8) & 0xffff) / 32768.0f - 1.0f;
  }
}

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping attention backward bench\n");
    return 0;
  }
  struct shape {
    int64_t H, T, D;
  };
  // Heads here are batch × heads, the shape a training step actually launches:
  // the first row is one layer of the GPT-medium benchmark.
  const shape shapes[] = {{64, 512, 64}, {64, 1024, 64}, {32, 512, 128}};
  const int R = 5, ROUNDS = 7;

  std::printf("fused causal attention, forward and its two pullback halves\n");
  std::printf("%-18s %10s %10s %10s %10s %9s\n", "H x T x D", "fwd ms", "dq ms",
              "dkv ms", "bwd ms", "bwd/fwd");
  for (const shape& s : shapes) {
    const int64_t n = s.H * s.T * s.D, bytes = n * 4;
    const float scale = 1.0f / std::sqrt((float)s.D);
    float *hq = nullptr, *hK = nullptr, *hV = nullptr, *hG = nullptr;
    void* q = alloc(bytes, &hq);
    void* K = alloc(bytes, &hK);
    void* V = alloc(bytes, &hV);
    void* dO = alloc(bytes, &hG);
    void* out = alloc(bytes, nullptr);
    void* dq = alloc(bytes, nullptr);
    void* dK = alloc(bytes, nullptr);
    void* dV = alloc(bytes, nullptr);
    void* stats = alloc(2 * s.H * s.T * 4, nullptr);
    fill_random(hq, n, 1);
    fill_random(hK, n, 2);
    fill_random(hV, n, 3);
    fill_random(hG, n, 4);

    auto run_fwd = [&] {
      attn_prefill(q, K, V, out, s.H, s.H, s.T, s.T, s.D, scale);
    };
    auto run_dq = [&] {
      attn_prefill_dq(q, K, V, dO, out, dq, stats, s.H, s.T, s.D, scale);
    };
    auto run_dkv = [&] {
      attn_prefill_dkv(q, K, V, dO, stats, dK, dV, s.H, s.T, s.D, scale);
    };
    auto time_it = [&](auto&& fn) {
      fn();
      flush();
      cpu_barrier();
      std::vector<double> ms;
      for (int r = 0; r < ROUNDS; r++) {
        auto t0 = clk::now();
        for (int i = 0; i < R; i++) fn();
        flush();
        cpu_barrier();
        ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0)
                         .count() /
                     R);
      }
      return median(std::move(ms));
    };

    run_fwd();  // stats and dq read the forward's output
    flush();
    cpu_barrier();
    const double f = time_it(run_fwd), a = time_it(run_dq), b = time_it(run_dkv);
    char name[32];
    std::snprintf(name, sizeof(name), "%lld x %lld x %lld", (long long)s.H,
                  (long long)s.T, (long long)s.D);
    std::printf("%-18s %10.3f %10.3f %10.3f %10.3f %9.2f\n", name, f, a, b,
                a + b, (a + b) / f);

    release(q, bytes, hq);
    release(K, bytes, hK);
    release(V, bytes, hV);
    release(dO, bytes, hG);
    release(out, bytes, nullptr);
    release(dq, bytes, nullptr);
    release(dK, bytes, nullptr);
    release(dV, bytes, nullptr);
    release(stats, 2 * s.H * s.T * 4, nullptr);
  }
  return 0;
}
