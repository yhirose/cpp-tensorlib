// The fused causal attention and its pullback on the Metal layer: what a
// training step pays per attention layer, split into the forward, the dq half
// and the dK/dV half. bench/cuda/speed/bench_attn_bwd.cpp is the CUDA twin.
//
// Direct metal:: API (timing via metal::flush + cpu_barrier and steady_clock),
// so a change to one kernel can be measured without a consumer's build.

#include "metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tl::metal;
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
    std::printf("no Metal device — skipping attention bench\n");
    return 0;
  }
  struct shape {
    int64_t H, T, D;
  };
  // Heads here are batch × heads: one layer of the GPT small, medium and large
  // steps in culebra's benchmarks/tensor, then a D=128 head.
  const shape shapes[] = {
      {32, 256, 64}, {64, 512, 64}, {48, 1024, 64}, {32, 512, 128}};
  const int R = 5, ROUNDS = 7;

  std::printf("fused causal attention, forward and its two pullback halves\n");
  std::printf("%-18s %10s %10s %10s %10s\n", "H x T x D", "fwd ms", "dq ms",
              "dkv ms", "TFLOPS");
  for (const shape& s : shapes) {
    const int64_t n = s.H * s.T * s.D, bytes = n * 4;
    const float scale = 1.0f / std::sqrt((float)s.D);
    // release() pools each buffer with its own contents pointer.
    float *hq, *hK, *hV, *hG, *hout, *hdq, *hdK, *hdV, *hst;
    void* q = alloc(bytes, &hq);
    void* K = alloc(bytes, &hK);
    void* V = alloc(bytes, &hV);
    void* dO = alloc(bytes, &hG);
    void* out = alloc(bytes, &hout);
    void* dq = alloc(bytes, &hdq);
    void* dK = alloc(bytes, &hdK);
    void* dV = alloc(bytes, &hdV);
    void* stats = alloc(2 * s.H * s.T * 4, &hst);
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
        ms.push_back(
            std::chrono::duration<double, std::milli>(clk::now() - t0).count() /
            R);
      }
      return median(std::move(ms));
    };

    run_fwd();  // dq reads the forward's output, dK/dV the dq half's stats
    run_dq();
    flush();
    cpu_barrier();
    const double f = time_it(run_fwd), a = time_it(run_dq), b = time_it(run_dkv);
    // Causal half of the products: 2 in the forward, 3 in dq, 4 in dK/dV.
    const double flop = 9.0 * s.H * s.T * s.T * s.D;
    char name[32];
    std::snprintf(name, sizeof(name), "%lld x %lld x %lld", (long long)s.H,
                  (long long)s.T, (long long)s.D);
    std::printf("%-18s %10.3f %10.3f %10.3f %10.2f\n", name, f, a, b,
                flop / ((f + a + b) * 1e9));

    release(q, bytes, hq);
    release(K, bytes, hK);
    release(V, bytes, hV);
    release(dO, bytes, hG);
    release(out, bytes, hout);
    release(dq, bytes, hdq);
    release(dK, bytes, hdK);
    release(dV, bytes, hdV);
    release(stats, 2 * s.H * s.T * 4, hst);
  }
  return 0;
}
