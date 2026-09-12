// Thread-pool round-trip census: what a parallel_for costs with nothing to
// do — the wake + join of p threads (empty body) — per thread count, both
// back-to-back (the training-step pattern) and after a 1 ms idle. Then the
// thread floor this implies: a second thread pays for itself only when the
// work it takes over outlasts the round trip, i.e. work > 2 x latency x
// single-thread throughput; compare that against cpu::min_work_per_thread_().
// On a 20-thread Zen under WSL2 (2026-09-12): 2 threads 20 µs, +7 µs per
// extra thread, 20 threads 146 µs; 54k MAC/µs -> floor 2.1e6, the census
// value. Medians (misc/census.cpp discipline).
#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static double median_us(int trials, const std::function<void()>& f) {
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
  auto& pool = tl::cpu::thread_pool::instance();
  std::printf("pool %d threads\n", pool.size());
  std::function<void(int64_t, int64_t)> nop = [](int64_t, int64_t) {};
  const double sleep_only =
      median_us(50, [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
  std::printf("  %-8s %14s %14s\n", "threads", "back-to-back", "after 1ms idle");
  double latency2 = 0;
  for (int p : {2, 3, 4, 6, 8, 12, 16, 20, 32, 64}) {
    if (p > pool.size()) break;
    double hot = median_us(200, [&] { pool.parallel_for(p, nop, p); });
    double cold = median_us(50, [&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      pool.parallel_for(p, nop, p);
    });
    if (p == 2) latency2 = hot;
    std::printf("  %-8d %11.1f us %11.1f us\n", p, hot, cold - sleep_only);
  }
  // Single-thread gemm throughput at 128^3 (below the default floor, so it
  // runs on the calling thread whatever the pool says).
  tl::use_cpu();
  auto a = tl::array::ones({128, 128}), b = tl::array::ones({128, 128});
  a.dot(b).eval();
  double t = median_us(31, [&] { a.dot(b).eval(); });
  double mac_per_us = 2097152.0 / t;
  std::printf("128^3 single-thread: %.1f us -> %.0f MAC/us\n", t, mac_per_us);
  std::printf("implied floor 2 x %.1f us x %.0f = %.2e MAC/thread "
              "(cpu::min_work_per_thread_() = %.2e)\n",
              latency2, mac_per_us, 2.0 * latency2 * mac_per_us,
              (double)tl::cpu::min_work_per_thread_());
  return 0;
}
