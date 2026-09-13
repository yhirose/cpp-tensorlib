// Thread-pool round-trip census: what a parallel_for costs with nothing to
// do — signalling p threads and joining (empty body) — per thread count,
// both back-to-back (the pool hot, workers still spinning: the op-chain
// pattern) and after a 1 ms idle (workers asleep: the cold wake through the
// helping tree). Then the thread floor this implies: a thread pays for
// itself only when its share outlasts the full-width hot round, i.e. work >
// latency(all) x single-thread throughput. cpu::min_work_per_thread_()
// derives its floor from the same two probes on first use (cpu.h
// pool_round_trip_us_ / single_thread_mac_per_us_), clamped below by
// kMinWorkFloor, so the last two figures agree up to that clamp
// (TL_CPU_MIN_WORK pins the runtime's). On a 20-thread i7-12700KF under
// WSL2 (2026-09-13): hot 2 threads 0.2 µs, 20 threads 3 µs; cold 20 threads
// 47 µs (was 146 with sequential wakes). Medians (misc/census.cpp
// discipline).
#include <tensorlib.h>

#include <chrono>
#include <cstdio>
#include <thread>

int main() {
  using tl::cpu::median_us_;
  auto& pool = tl::cpu::thread_pool::instance();
  std::printf("pool %d threads\n", pool.size());
  auto nop = [](int64_t, int64_t) {};
  const double sleep_only =
      median_us_(50, [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
  std::printf("  %-8s %14s %14s\n", "threads", "back-to-back", "after 1ms idle");
  for (int p : {2, 3, 4, 6, 8, 12, 16, 20, 32, 64}) {
    if (p > pool.size()) break;
    double hot = median_us_(200, [&] { pool.parallel_for(p, nop, p); });
    double cold = median_us_(50, [&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      pool.parallel_for(p, nop, p);
    });
    std::printf("  %-8d %11.1f us %11.1f us\n", p, hot, cold - sleep_only);
  }
  double round_trip = tl::cpu::pool_round_trip_us_();
  double mac_per_us = tl::cpu::single_thread_mac_per_us_();
  std::printf("128^3 single-thread: %.0f MAC/us; full-width hot round %.1f us\n",
              mac_per_us, round_trip);
  std::printf("implied floor max(%.0e, %.1f us x %.0f = %.2e) MAC/thread "
              "(cpu::min_work_per_thread_() = %.2e)\n",
              (double)tl::cpu::kMinWorkFloor, round_trip, mac_per_us,
              round_trip * mac_per_us, (double)tl::cpu::min_work_per_thread_());
  return 0;
}
