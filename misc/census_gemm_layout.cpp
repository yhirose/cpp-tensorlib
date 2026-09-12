// GEMM operand-layout census: the same product with each operand plain or a
// transposed view (NN, NT, TN, TT), on the CPU and the GPU, in µs. The
// attention scores gemm is NT (Q · Kᵀ) and gradients are TN/NT, so a backend
// whose fast kernel takes only NN shows up here as a cliff. On the RTX 3090
// (2026-09-12) the CUDA register-blocked kernel is NN-only and NT at
// 512x1024x512 fell to the one-output-per-thread fallback: 1057 µs against
// 117 µs for NN. Medians, warm-up first (misc/census.cpp discipline).
#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using tl::array;
using clk = std::chrono::steady_clock;

static array rnd(tl::shape_t s, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1;
  for (auto x : s) n *= (size_t)x;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return array::from(std::move(v), std::move(s));
}

static double med_us(int trials, const std::function<void()>& f) {
  for (int i = 0; i < 3; i++) f();
  std::vector<double> ts;
  for (int i = 0; i < trials; i++) {
    auto t0 = clk::now();
    f();
    ts.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

int main() {
  struct S { int64_t m, k, n; };
  // MNIST gradient shape, a square, and the attention-scores shape.
  const S shapes[] = {{30, 10, 784}, {256, 256, 256}, {512, 1024, 512}};
  const bool gpu = tl::gpu_available();
  std::printf("%-16s %-6s %10s %10s %10s %10s\n", "MxKxN", "device", "NN", "NT", "TN", "TT");
  for (const auto& s : shapes) {
    // A [m,k] plain; At [k,m] so At.transpose() is A as a view. Same for B.
    auto A = rnd({s.m, s.k}, 1), At = rnd({s.k, s.m}, 2);
    auto B = rnd({s.k, s.n}, 3), Bt = rnd({s.n, s.k}, 4);
    for (int dev = 0; dev < (gpu ? 2 : 1); dev++) {
      if (dev == 0) tl::use_cpu(); else tl::use_gpu();
      const int trials = s.m * s.n * s.k > 100'000'000 ? 9 : 21;
      double nn = med_us(trials, [&] { A.dot(B).eval(); });
      double nt = med_us(trials, [&] { A.dot(Bt.transpose()).eval(); });
      double tn = med_us(trials, [&] { At.transpose().dot(B).eval(); });
      double tt = med_us(trials, [&] { At.transpose().dot(Bt.transpose()).eval(); });
      char buf[40];
      std::snprintf(buf, sizeof buf, "%lldx%lldx%lld", (long long)s.m, (long long)s.k, (long long)s.n);
      std::printf("%-16s %-6s %8.1f us %8.1f us %8.1f us %8.1f us\n", buf,
                  dev == 0 ? "cpu" : "gpu", nn, nt, tn, tt);
    }
  }
  tl::use_cpu();
  return 0;
}
