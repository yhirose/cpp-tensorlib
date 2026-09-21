// Cross-entropy's three kernels against the softmax the composed form
// materializes: the row logsumexp the forward reduces with, the trailing-axis
// gather that names each row's target, and the fused pullback. The softmax is
// the yardstick — it reads [rows, cols] and writes it back, where logsumexp
// only reads and the pullback reads once and writes once, so each kernel's
// implied bandwidth says whether it sits near the memory floor.
//
// Direct cuda:: API, no cudart (timing via cuda::flush + cpu_barrier and
// steady_clock), so a kernel change can be measured without a consumer's
// build in the way.

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "gpu.h"  // cuda.h plus the shared ops (tl::gpu resolves to cuda here)

#include "array.h"  // the second table: the same work through the graph

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tl::cuda;
namespace gpu = tl::gpu;  // the shared ops; tl::gpu resolves to cuda here
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

// GB/s a kernel moving `bytes` in `ms` implies.
static double gbs(int64_t bytes, double ms) {
  return (double)bytes / 1.0e9 / (ms / 1.0e3);
}

int main() {
  if (!available()) {
    std::printf("no CUDA device — skipping cross-entropy bench\n");
    return 0;
  }
  struct shape {
    int64_t rows, cols;
  };
  // A training step's head: batch x seq rows against the vocabulary. The
  // middle row is the GPT-medium benchmark's own.
  const shape shapes[] = {{2048, 4096}, {4096, 8192}, {4096, 16384}};
  const int R = 5, ROUNDS = 7;

  std::printf("cross-entropy kernels, ms (implied GB/s)\n");
  std::printf("%-14s %16s %16s %10s %16s\n", "rows x cols", "logsumexp",
              "softmax", "gather", "xent_bwd");
  for (const shape& s : shapes) {
    const int64_t n = s.rows * s.cols, bytes = n * 4, row_bytes = s.rows * 4;
    float *hx = nullptr, *ht = nullptr, *hg = nullptr;
    void* x = alloc(bytes, &hx);
    void* probs = alloc(bytes, nullptr);
    void* dx = alloc(bytes, nullptr);
    void* lse = alloc(row_bytes, nullptr);
    void* tgt = alloc(row_bytes, &ht);
    void* g = alloc(row_bytes, &hg);
    void* picked = alloc(row_bytes, nullptr);
    fill_random(hx, n, 1);
    fill_random(hg, s.rows, 2);
    for (int64_t i = 0; i < s.rows; i++) {
      ht[i] = (float)((i * 7919) % s.cols);  // ops.cul's own label sequence
    }

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

    const double lse_ms = time_it(
        [&] { gpu::row_logsumexp({x, 0}, {lse, 0}, s.rows, s.cols, 1.0f, 0.0f); });
    const double sm_ms = time_it([&] {
      gpu::row_op(gpu::kop::softmax, {x, 0}, {probs, 0}, s.rows, s.cols, 1.0f,
                  0.0f);
    });
    const double gat_ms = time_it([&] {
      gpu::gather_from_axis({x, 0}, {tgt, 0}, {picked, 0}, s.rows, s.cols);
    });
    const double bwd_ms = time_it([&] {
      gpu::xent_bwd({x, 0}, {lse, 0}, {tgt, 0}, {g, 0}, {dx, 0}, s.rows, s.cols);
    });

    char name[32];
    std::snprintf(name, sizeof(name), "%lld x %lld", (long long)s.rows,
                  (long long)s.cols);
    // logsumexp reads the matrix; softmax reads and writes it; the pullback
    // reads it and writes a gradient the same size. gather touches one
    // element per row, so its bytes are not worth a rate.
    std::printf("%-14s %8.3f (%5.0f) %8.3f (%5.0f) %10.3f %8.3f (%5.0f)\n",
                name, lse_ms, gbs(bytes, lse_ms), sm_ms, gbs(2 * bytes, sm_ms),
                gat_ms, bwd_ms, gbs(2 * bytes, bwd_ms));

    release(x, bytes, hx);
    release(probs, bytes, nullptr);
    release(dx, bytes, nullptr);
    release(lse, row_bytes, nullptr);
    release(tgt, row_bytes, ht);
    release(g, row_bytes, hg);
    release(picked, row_bytes, nullptr);
  }

  // The same work again, but composed through the array graph rather than
  // called kernel by kernel. The gap between the two tables is what the
  // evaluator adds per op — launch, dispatch gates, and any host read that
  // makes the queue wait — which is what a consumer actually pays.
  std::printf("\nthrough the array graph, ms\n");
  std::printf("%-14s %10s %10s %10s %10s\n", "rows x cols", "softmax",
              "logsumexp", "gather", "xent fwd");
  tl::device_ = tl::device_type::gpu;
  for (const shape& s : shapes) {
    const int64_t n = s.rows * s.cols;
    std::vector<float> hx(n);
    fill_random(hx.data(), n, 1);
    std::vector<float> ht(s.rows);
    for (int64_t i = 0; i < s.rows; i++) {
      ht[i] = (float)((i * 7919) % s.cols);
    }
    auto x = tl::array::from(std::move(hx), {s.rows, s.cols});
    auto idx = tl::array::from(std::move(ht), {s.rows});
    const float ceiling = -std::log(1.0e-15f);

    auto time_arr = [&](auto&& build) {
      build().eval();
      flush();
      cpu_barrier();
      std::vector<double> ms;
      for (int r = 0; r < ROUNDS; r++) {
        auto t0 = clk::now();
        for (int i = 0; i < R; i++) build().eval();
        flush();
        cpu_barrier();
        ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0)
                         .count() /
                     R);
      }
      return median(std::move(ms));
    };

    const double sm = time_arr([&] { return x.softmax(); });
    const double lse = time_arr([&] { return x.logsumexp(1); });
    const double gat = time_arr([&] { return tl::gather_from_axis(x, idx); });
    // culebra's own forward, op for op.
    const double fwd = time_arr([&] {
      return (x.logsumexp(1) - tl::gather_from_axis(x, idx))
          .clamp(-std::numeric_limits<float>::max(), ceiling);
    });

    char name[32];
    std::snprintf(name, sizeof(name), "%lld x %lld", (long long)s.rows,
                  (long long)s.cols);
    std::printf("%-14s %10.3f %10.3f %10.3f %10.3f\n", name, sm, lse, gat, fwd);
  }
  return 0;
}
