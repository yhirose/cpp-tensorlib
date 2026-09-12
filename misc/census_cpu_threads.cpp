// CPU gemm thread-count census: how many threads a gemm of a given size
// should use. cpu::sgemm splits its M panels across up to
// M*N*K / min_work_per_thread_() threads (cpu.h); this prints, for a sweep
// of shapes, the gemm's time at each thread count so the floor can be read
// off: the smallest work per thread at which adding a thread still helps.
// Interleaved, warmup, medians (the misc/census.cpp discipline).
//
// The thread cap is applied through TL_CPU_MIN_WORK (read once at process
// start), so each column is one process: run.sh-style
//   for w in 100000 300000 1000000 3000000; do
//     TL_CPU_MIN_WORK=$w ./tensorlib_census_cpu_threads; done
// or read the "threads" column here, which reports the cap that resulted.
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
  for (int i = 0; i < 4; i++) f();
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
  tl::use_cpu();
  const int64_t floor = tl::cpu::min_work_per_thread_();
  const int pool = tl::cpu::thread_pool::instance().size();
  std::printf("pool %d threads, TL_CPU_MIN_WORK %lld\n", pool, (long long)floor);
  std::printf("  %-18s %-12s %8s %10s\n", "shape (MxKxN)", "work", "threads", "median");
  struct S { int64_t m, k, n; };
  // MNIST-step shapes, then squares across the crossover.
  const S shapes[] = {{10, 30, 10},   {30, 784, 10},  {30, 10, 784},  {64, 64, 64},
                      {96, 96, 96},   {128, 128, 128}, {160, 160, 160}, {192, 192, 192},
                      {256, 256, 256}, {384, 384, 384}, {512, 512, 512}, {1024, 1024, 1024}};
  for (const auto& s : shapes) {
    auto a = rnd({s.m, s.k}, 1), b = rnd({s.k, s.n}, 2);
    const int64_t work = s.m * s.n * s.k;
    const int threads = (int)std::max<int64_t>(1, std::min<int64_t>(pool, work / floor));
    double t = med_us(work > 100'000'000 ? 7 : 31, [&] { a.dot(b).eval(); });
    char buf[40];
    std::snprintf(buf, sizeof buf, "%lldx%lldx%lld", (long long)s.m, (long long)s.k, (long long)s.n);
    std::printf("  %-18s %-12lld %8d %9.1f us\n", buf, (long long)work, threads, t);
  }
  return 0;
}
