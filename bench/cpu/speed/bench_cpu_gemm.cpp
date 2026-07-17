// Own CPU GEMM benchmark harness — for the M5 tuning sprint (see
// docs/roadmap.md "M5 remaining" and performance-notes.md "Own CPU GEMM
// first cut"). Compares cpu::sgemm (own NEON/scalar) against the ref::
// naive loop and, when built with -DBENCH_HAS_OPENBLAS, against OpenBLAS.
// Caveat, measured 2026-07-03: Homebrew's OpenBLAS on M1 appears to use
// generic ARMV8 microkernels, not M1-tuned ones — the untuned own kernel
// already beats it (~1.8x at 1024^3, threads matched), implausible against
// a properly-tuned OpenBLAS. So treat the Mac OpenBLAS number as a loose
// lower-bound sanity check, NOT the real gate. The better Mac metric is %
// of NEON fp32 peak (~410 GFLOP/s on M1 Pro; own is ~54-64%). The
// definitive OpenBLAS-90% gate is on the x86 box with a tuned build.
//
// Build (own vs ref only):
//   c++ -std=c++2b -O2 -I include bench/cpu/speed/bench_cpu_gemm.cpp -o /tmp/bcg \
//       -framework Accelerate -framework Metal -framework Foundation
// With OpenBLAS (the gate) on macOS/Homebrew:
//   c++ -std=c++2b -O2 -I include -DBENCH_HAS_OPENBLAS \
//       -I/opt/homebrew/opt/openblas/include bench/cpu/speed/bench_cpu_gemm.cpp \
//       -o /tmp/bcg -framework Accelerate -framework Metal -framework Foundation \
//       -L/opt/homebrew/opt/openblas/lib -lopenblas
//
// Measurement discipline (performance-notes.md): check `uptime` < ~4 first,
// interleave, take medians. Numbers here are directional if the machine is
// loaded.

#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#ifdef BENCH_HAS_OPENBLAS
#include <cblas.h>
// OpenBLAS thread control (extension). Match own's thread count or the gate
// is a thread-count race, not a kernel comparison — Homebrew OpenBLAS
// otherwise runs far below its NEON potential here.
extern "C" void openblas_set_num_threads(int);
#endif

using tl::array;
using clk = std::chrono::steady_clock;

static array rnd(tl::shape_t s, unsigned sd) {
  std::mt19937 g(sd);
  std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1;
  for (auto x : s) n *= static_cast<size_t>(x);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return array::from(std::move(v), std::move(s));
}

static double med(int trials, const std::function<void()>& f) {
  for (int i = 0; i < 3; i++) f();
  std::vector<double> ts;
  for (int i = 0; i < trials; i++) {
    auto a = clk::now();
    f();
    ts.push_back(std::chrono::duration<double, std::milli>(clk::now() - a).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

int main() {
  std::printf("own CPU GEMM — threads=%d\n",
              tl::cpu::thread_pool::instance().size());
#ifdef BENCH_HAS_OPENBLAS
  openblas_set_num_threads(tl::cpu::thread_pool::instance().size());
  std::printf("gate: OpenBLAS (ARM = same NEON ISA, threads matched)\n");
#else
  std::printf("(build with -DBENCH_HAS_OPENBLAS for the gate)\n");
#endif

  for (int64_t m : {128, 256, 512, 1024, 2048}) {
    auto a = rnd({m, m}, 1), b = rnd({m, m}, 2);
    double gf = 2.0 * m * m * m;

    tl::use_accelerate_ = false;
    tl::cpu::enabled_ = true;
    double own = med(9, [&] { a.dot(b).eval(); });

    tl::cpu::enabled_ = false;
    double ref = (m <= 1024) ? med(3, [&] { a.dot(b).eval(); }) : 0.0;
    tl::cpu::enabled_ = true;

    double blas = 0.0;
#ifdef BENCH_HAS_OPENBLAS
    {
      std::vector<float> A(m * m), B(m * m), C(m * m);
      const float* pa = a.data();
      const float* pb = b.data();
      std::copy(pa, pa + m * m, A.begin());
      std::copy(pb, pb + m * m, B.begin());
      blas = med(9, [&] {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0f,
                    A.data(), m, B.data(), m, 0.0f, C.data(), m);
      });
    }
#endif

    std::printf("  %4lld^3  own %8.3f ms (%7.1f GFLOP/s)", (long long)m, own,
                gf / (own * 1e6));
    if (ref > 0) std::printf("   ref %9.3f ms (%.1fx)", ref, ref / own);
    if (blas > 0)
      std::printf("   openblas %8.3f ms   own/blas %.0f%%", blas,
                  100.0 * blas / own);
    std::printf("\n");
  }
  return 0;
}
