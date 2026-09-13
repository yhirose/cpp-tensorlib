// Thread-pool round-trip census: what a parallel_for costs with nothing to
// do — signalling p threads and joining (empty body) — per thread count,
// both back-to-back (the pool hot, workers still spinning: the op-chain
// pattern) and after a 1 ms idle (workers asleep: the cold wake through the
// helping tree). Then the thread floor this implies: a thread pays for
// itself only when its share outlasts the full-width hot round, i.e. work >
// latency(all) x single-thread throughput. cpu::min_work_per_thread_()
// derives its floor by the same arithmetic on first use, so the two figures
// printed last should agree up to its 1e5 clamp (TL_CPU_MIN_WORK pins the
// runtime's). On a 20-thread i7-12700KF under WSL2 (2026-09-13): hot 2
// threads 0.2 µs, 20 threads 3 µs; cold 20 threads 47 µs (was 146 with
// sequential wakes). Medians (misc/census.cpp discipline).
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
  double latency_all = 0;
  for (int p : {2, 3, 4, 6, 8, 12, 16, 20, 32, 64}) {
    if (p > pool.size()) break;
    double hot = median_us(200, [&] { pool.parallel_for(p, nop, p); });
    double cold = median_us(50, [&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      pool.parallel_for(p, nop, p);
    });
    latency_all = hot;
    std::printf("  %-8d %11.1f us %11.1f us\n", p, hot, cold - sleep_only);
  }
  // Single-thread gemm throughput at 128^3 (pinned to one thread: the floor
  // is derived from it, so it must not depend on the floor).
  constexpr int64_t N = 128;
  std::vector<float> a(N * N, 1.0f), b(N * N, 1.0f), c(N * N);
  auto probe = [&] {
    tl::cpu::sgemm_(a.data(), N, 1, b.data(), N, 1, c.data(), N, N, N, 1.0f, 1);
  };
  probe();
  double t = median_us(31, probe);
  double mac_per_us = (double)(N * N * N) / t;
  std::printf("128^3 single-thread: %.1f us -> %.0f MAC/us\n", t, mac_per_us);
  std::printf("implied floor %.1f us x %.0f = %.2e MAC/thread "
              "(cpu::min_work_per_thread_() = %.2e)\n",
              latency_all, mac_per_us, latency_all * mac_per_us,
              (double)tl::cpu::min_work_per_thread_());
  return 0;
}
