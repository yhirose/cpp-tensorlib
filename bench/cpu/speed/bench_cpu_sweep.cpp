// Block-size sweep harness for the M5 tuning sprint: calls cpu::sgemm
// directly (no graph/eval), so it compiles in ~1s and isolates the GEMM.
// Compile per config with -DTL_CPU_MC/KC/NC (the cpu.h constants are
// #ifndef-guarded for exactly this):
//   c++ -std=c++2b -O2 -I include -DTL_CPU_KC=512 bench/cpu/speed/bench_cpu_sweep.cpp -o sw && ./sw
// Interleave configs and take the per-size medians (performance-notes.md
// measurement discipline; check uptime < ~4 first).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu.h"

using clk = std::chrono::steady_clock;

int main() {
  std::printf("MC=%lld KC=%lld NC=%lld:", (long long)tl::cpu::MC,
              (long long)tl::cpu::KC, (long long)tl::cpu::NC);
  for (int64_t m : {256, 512, 1024, 2048}) {
    std::mt19937 g(42);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<float> A(m * m), B(m * m), C(m * m);
    for (auto& x : A) x = d(g);
    for (auto& x : B) x = d(g);
    auto run = [&] {
      tl::cpu::sgemm(A.data(), m, 1, B.data(), m, 1, C.data(), m, m, m, 1.0f);
    };
    for (int i = 0; i < 3; i++) run();
    std::vector<double> ts;
    int trials = m >= 2048 ? 5 : 9;
    for (int i = 0; i < trials; i++) {
      auto t0 = clk::now();
      run();
      ts.push_back(std::chrono::duration<double>(clk::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    double sec = ts[ts.size() / 2];
    std::printf("  %lld:%6.1f", (long long)m, 2.0 * m * m * m / sec / 1e9);
  }
  std::printf("  GF/s\n");
  return 0;
}
