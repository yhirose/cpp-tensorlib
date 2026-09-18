#pragma once

// Own CPU backend (M5) — the portable SGEMM for platforms without
// Accelerate (Linux/Windows; also runnable on Apple for validation).
//
// BLIS-style GEMM: cache-blocking loops + packing (architecture-independent)
// around a register-blocked microkernel selected at runtime by ISA/CPUID
// (select_ukernel). Raw-pointer API (no dependency on array.h, like metal.h),
// so array.h calls it with plain buffers + strides; packing is stride-aware
// so transposed views feed in place (no materialization), matching the
// Accelerate path.
//
// Scope (M5 first cut): SGEMM only. Elementwise / reductions ride array.h's
// contiguous flat-loop fast paths, which are memory-bound and autovectorize
// — the compute-bound GEMM is what needs the hand-blocked kernel.
//
// Status: scaffolding + scalar microkernel (portable, correct on any ISA);
// NEON microkernel (Apple Silicon / ARM Linux, native), NEON-tuned on M1 Pro
// (lane-indexed FMA, K-unroll, prefetch, thread-local pack buffers, MR-panel-
// granular parallelism — see performance-notes.md); AVX2 microkernel (x86),
// executed and tuned on the i7-12700KF box (2026-07-03): the 6×16 tile (12
// ymm accumulators) reaches ~91% of single-P-core AVX2 peak and ~106% of
// OpenBLAS at 2048³, which settled its A/B against the original 8×8 tile (that
// kernel is gone; see the 6×16 comment for what it was and why it lost).
// Each kernel packs to its OWN tile — 8×8 for scalar/NEON, 6×16 for
// AVX2 — carried in the ukernel_desc the driver reads (no longer one shared
// layout). AVX-512 (also NR=16) reuses the 6×16 pack layout and is deferred;
// see docs/roadmap.md and docs/performance-notes.md.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TL_CPU_NEON 1
#endif

// x86 detection covers both the GCC/Clang macros and the MSVC ones (_M_X64 /
// _M_IX86 — MSVC defines neither __x86_64__ nor __i386__).
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include <immintrin.h>
#define TL_CPU_X86 1
#if defined(_MSC_VER)
#include <intrin.h>  // __cpuidex / _xgetbv
#endif
#endif

// Portable helper macros bridging the GCC/Clang builtins to MSVC:
//   TL_TARGET(feat) — per-function ISA opt-in, e.g. TL_TARGET("avx2,fma").
//     GCC/Clang need it so an ISA-specific kernel can be compiled in a
//     baseline-x86 TU and dispatched at runtime; MSVC has no such attribute
//     and accepts the intrinsics unconditionally, so it expands to nothing
//     there. Parametrized on the feature string so the deferred AVX-512 kernel
//     reuses it as TL_TARGET("avx512f,...") rather than a second macro.
//   TL_PREFETCH(p)  — software prefetch hint; a no-op where unavailable.
#if defined(_MSC_VER)
#define TL_TARGET(feat)
#else
#define TL_TARGET(feat) __attribute__((target(feat)))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TL_PREFETCH(p) __builtin_prefetch(p)
#elif defined(_MSC_VER) && defined(TL_CPU_X86)
#define TL_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#else
#define TL_PREFETCH(p) ((void)0)
#endif

#include "cpu_threadpool.h"

namespace tl {
namespace cpu {

// Gates every own-CPU kernel in the eval dispatch — the GEMMs, the batched
// and attention paths, and the pool-parallel elementwise/softmax drivers in
// array.h. Default on: it is the primary CPU backend off-Apple, and on Apple
// it sits below Accelerate (only reached when use_accelerate_ is off). Oracle
// tests set it false (with use_accelerate_ false) so the ref:: walkers, single
// threaded and strided, are what the fast paths are compared against.
inline bool enabled_ = true;

// Register-block tile and cache-block sizes. MC/KC from the 2026-07-03
// M1 Pro sweep (KC=512 best at 1024³+; see performance-notes.md — the
// machine was loaded, so re-confirm in a quiet census). Overridable via
// -D for the tuning sweep (bench sweep harness).
#ifndef TL_CPU_MC
#define TL_CPU_MC 128
#endif
#ifndef TL_CPU_KC
#define TL_CPU_KC 512
#endif
#ifndef TL_CPU_NC
#define TL_CPU_NC 2048
#endif
constexpr int MR = 8, NR = 8;
constexpr int64_t MC = TL_CPU_MC, KC = TL_CPU_KC, NC = TL_CPU_NC;

// The gemm proper, with the thread cap chosen by the caller (defined after
// the packers; declared here for the floor probe below).
inline void sgemm_(const float* A, int64_t as0, int64_t as1, const float* B,
                   int64_t bs0, int64_t bs1, float* C, int64_t m, int64_t n,
                   int64_t k, float alpha, int max_threads);

// How much of a job (M*N*K multiply-adds, or an elementwise op's elements
// weighted by kStreamMacs) each thread must get before the split is worth a
// thread: a job below this runs on the calling thread, and a medium one
// signals only the workers it can feed.
//
// kMinWorkFloor is the design point: workers spin between jobs
// (cpu_threadpool.h), so a hot full-width round costs a few µs and the
// transformer block census (2026-09-13) reads the same at 1e5 and 3e5 and
// loses on layer_norm from 1e6 up, where its 9 elementwise ops fall back to
// one or two threads. The derivation below is the escape for a host whose
// round trip is not a few µs (spinning off with TL_CPU_SPIN_US=0, a VM that
// deschedules spinners): a thread pays for itself only when its share
// outlasts a full-width round of the pool (signalling every worker, each
// taking and finishing its chunks, the join), i.e.
//   floor = max(kMinWorkFloor, round_trip(all threads, pool hot) × MAC/µs)
// (misc/census_pool_latency.cpp prints the same arithmetic from the two
// probes below). About 1 ms, once, on the first job. The model charges a
// two-thread split the full-width round; harmless while both are µs. Env
// TL_CPU_MIN_WORK pins it, like TL_BATCH_MATMUL_BIAS.
constexpr int64_t kMinWorkFloor = 100'000;

// Median µs of `trials` calls.
template <class F>
inline double median_us_(int trials, F&& f) {
  using clk = std::chrono::steady_clock;
  std::vector<double> ts(static_cast<size_t>(trials));
  for (double& t : ts) {
    auto t0 = clk::now();
    f();
    t = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
  }
  std::nth_element(ts.begin(), ts.begin() + trials / 2, ts.end());
  return ts[static_cast<size_t>(trials / 2)];
}

// A hot full-width empty parallel_for, median of 25 after 5 warm-ups.
inline double pool_round_trip_us_() {
  auto& pool = thread_pool::instance();
  const int p = pool.size();
  auto nop = [](int64_t, int64_t) {};
  for (int i = 0; i < 5; i++) pool.parallel_for(p, nop, p);
  return median_us_(25, [&] { pool.parallel_for(p, nop, p); });
}

// Single-thread gemm throughput at 128^3, median of 5.
inline double single_thread_mac_per_us_() {
  constexpr int64_t N = 128;
  std::vector<float> a(N * N, 1.0f), b(N * N, 1.0f), c(N * N);
  auto probe = [&] {
    sgemm_(a.data(), N, 1, b.data(), N, 1, c.data(), N, N, N, 1.0f, 1);
  };
  probe();
  return static_cast<double>(N * N * N) / median_us_(5, probe);
}

inline int64_t min_work_per_thread_() {
  static const int64_t v = []() -> int64_t {
    if (const char* e = std::getenv("TL_CPU_MIN_WORK")) {
      char* end = nullptr;
      long long x = std::strtoll(e, &end, 10);
      if (end != e && x > 0) return static_cast<int64_t>(x);
    }
    if (thread_pool::instance().size() < 2) return 2'000'000;  // nothing to split across
    double derived = pool_round_trip_us_() * single_thread_mac_per_us_();
    return static_cast<int64_t>(
        std::clamp(derived, static_cast<double>(kMinWorkFloor), 1e9));
  }();
  return v;
}

namespace detail {

// Pack an mrt-tall, kc-wide panel of A (with arbitrary strides, alpha folded
// in) into column-major micropanels: ap[p*mrt + i] = alpha * A[i*as0 + p*as1].
// `mrt` is the selected kernel's tile MR (8 for scalar/NEON, 6 for AVX2 6×16);
// rows past `mr` are zero-padded to mrt so the microkernel runs a full tile.
inline void pack_a_panel(const float* A, int64_t as0, int64_t as1, int mr,
                         int mrt, int64_t kc, float alpha, float* ap) {
  for (int64_t p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) ap[p * mrt + i] = alpha * A[i * as0 + p * as1];
    for (int i = mr; i < mrt; i++) ap[p * mrt + i] = 0.0f;
  }
}

// Pack a kc-tall, nrt-wide panel of B into row-major micropanels:
// bp[p*nrt + j] = B[p*bs0 + j*bs1]. `nrt` is the kernel's tile NR (8 or 16);
// cols past `nr` are zero-padded to nrt.
inline void pack_b_panel(const float* B, int64_t bs0, int64_t bs1, int nr,
                         int nrt, int64_t kc, float* bp) {
  for (int64_t p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) bp[p * nrt + j] = B[p * bs0 + j * bs1];
    for (int j = nr; j < nrt; j++) bp[p * nrt + j] = 0.0f;
  }
}

// Microkernel: C[mr×nr] += (packed A panel) · (packed B panel), accumulating
// over kc. C is row-major with leading dimension ldc. Edge tiles (mr<MR or
// nr<NR) read the zero-padded panels and store only the valid mr×nr corner.
//
// One implementation per ISA, all sharing the MR=8/NR=8 packed layout and
// this signature so they are interchangeable behind a function pointer
// (select_ukernel below). The scalar version is always compiled (portable
// fallback + oracle for the vector ones); NEON and AVX2 are compiled only
// where their headers exist. AVX2 is `target`-attributed so it can sit in a
// baseline-x86 TU and be reached only after a CPUID check.
using ukernel_fn = void (*)(int64_t, const float*, const float*, float*,
                            int64_t, int, int);

// A microkernel plus the register-block tile it packs to. The tile is a
// property of the kernel — 8×8 for scalar/NEON, 6×16 for the AVX2 kernel (6
// rows × 2 ymm = 12 accumulators, the canonical Haswell+ blocking) — so the
// driver reads mr/nr here to size the packed panels. This is the "second
// packing layout" the roadmap anticipated; the deferred AVX-512 kernel reuses
// the NR=16 layout. select_ukernel picks one descriptor once per process.
struct ukernel_desc {
  ukernel_fn fn;
  int mr, nr;
};

inline void ukernel_scalar(int64_t kc, const float* ap, const float* bp,
                           float* c, int64_t ldc, int mr, int nr) {
  float ab[MR][NR];
  for (int i = 0; i < MR; i++)
    for (int j = 0; j < NR; j++) ab[i][j] = 0.0f;
  for (int64_t p = 0; p < kc; p++) {
    const float* a = ap + p * MR;
    const float* b = bp + p * NR;
    for (int i = 0; i < MR; i++)
      for (int j = 0; j < NR; j++) ab[i][j] += a[i] * b[j];
  }
  for (int i = 0; i < mr; i++)
    for (int j = 0; j < nr; j++) c[i * ldc + j] += ab[i][j];
}

#ifdef TL_CPU_NEON
inline void ukernel_neon(int64_t kc, const float* ap, const float* bp,
                         float* c, int64_t ldc, int mr, int nr) {
  // 8×8 tile: two float32x4 accumulators per row (cols 0-3, 4-7). A is
  // loaded as two vectors and consumed with lane-indexed FMA — no per-
  // element vdupq (8 dup µops/k-step in the first cut). 16 accumulators +
  // 2 A + 2 B vectors = 20 of the 32 NEON registers.
  float32x4_t ab0[MR], ab1[MR];
  for (int i = 0; i < MR; i++) {
    ab0[i] = vdupq_n_f32(0.0f);
    ab1[i] = vdupq_n_f32(0.0f);
  }
#define TL_CPU_KSTEP(p)                                        \
  {                                                            \
    float32x4_t b0 = vld1q_f32(bp + (p)*NR);                   \
    float32x4_t b1 = vld1q_f32(bp + (p)*NR + 4);               \
    float32x4_t a0 = vld1q_f32(ap + (p)*MR);                   \
    float32x4_t a1 = vld1q_f32(ap + (p)*MR + 4);               \
    ab0[0] = vfmaq_laneq_f32(ab0[0], b0, a0, 0);               \
    ab1[0] = vfmaq_laneq_f32(ab1[0], b1, a0, 0);               \
    ab0[1] = vfmaq_laneq_f32(ab0[1], b0, a0, 1);               \
    ab1[1] = vfmaq_laneq_f32(ab1[1], b1, a0, 1);               \
    ab0[2] = vfmaq_laneq_f32(ab0[2], b0, a0, 2);               \
    ab1[2] = vfmaq_laneq_f32(ab1[2], b1, a0, 2);               \
    ab0[3] = vfmaq_laneq_f32(ab0[3], b0, a0, 3);               \
    ab1[3] = vfmaq_laneq_f32(ab1[3], b1, a0, 3);               \
    ab0[4] = vfmaq_laneq_f32(ab0[4], b0, a1, 0);               \
    ab1[4] = vfmaq_laneq_f32(ab1[4], b1, a1, 0);               \
    ab0[5] = vfmaq_laneq_f32(ab0[5], b0, a1, 1);               \
    ab1[5] = vfmaq_laneq_f32(ab1[5], b1, a1, 1);               \
    ab0[6] = vfmaq_laneq_f32(ab0[6], b0, a1, 2);               \
    ab1[6] = vfmaq_laneq_f32(ab1[6], b1, a1, 2);               \
    ab0[7] = vfmaq_laneq_f32(ab0[7], b0, a1, 3);               \
    ab1[7] = vfmaq_laneq_f32(ab1[7], b1, a1, 3);               \
  }
  int64_t p = 0;
  for (; p + 4 <= kc; p += 4) {
    TL_PREFETCH(ap + (p + 16) * MR);
    TL_PREFETCH(bp + (p + 16) * NR);
    TL_CPU_KSTEP(p);
    TL_CPU_KSTEP(p + 1);
    TL_CPU_KSTEP(p + 2);
    TL_CPU_KSTEP(p + 3);
  }
  for (; p < kc; p++) TL_CPU_KSTEP(p);
#undef TL_CPU_KSTEP
  if (mr == MR && nr == NR) {
    for (int i = 0; i < MR; i++) {
      vst1q_f32(c + i * ldc, vaddq_f32(vld1q_f32(c + i * ldc), ab0[i]));
      vst1q_f32(c + i * ldc + 4, vaddq_f32(vld1q_f32(c + i * ldc + 4), ab1[i]));
    }
  } else {
    float tmp[MR][NR];
    for (int i = 0; i < MR; i++) {
      vst1q_f32(tmp[i], ab0[i]);
      vst1q_f32(tmp[i] + 4, ab1[i]);
    }
    for (int i = 0; i < mr; i++)
      for (int j = 0; j < nr; j++) c[i * ldc + j] += tmp[i][j];
  }
}
#endif  // TL_CPU_NEON

#ifdef TL_CPU_X86
// AVX2 6×16: 6 rows × 2 ymm columns = 12 accumulators, the canonical Haswell+
// register blocking (BLIS/OpenBLAS haswell use it). Per k-step: two B loads
// (16 floats) each FMA'd against a broadcast of the 6 A values → 12 FMA on 12
// accumulators, using 12 + 2(B) + 1(broadcast) = 15 of the 16 ymm registers.
// The first cut was an 8×8 tile (the direct x86 analogue of the NEON lane-FMA
// kernel): only 8 accumulators, half the register file, so it reloaded B
// relative to compute. 6×16 raises in-register reuse — the reason it is the
// standard AVX2 GEMM tile — and measured ~1.2x it on the i7-12700KF, so the
// A/B ended there and only this kernel remains. Packs to MR=6/NR=16, a layout
// distinct from the 8×8 scalar/NEON one; the driver picks it via the
// descriptor. K-unrolled ×4 with prefetch, like the NEON kernel.
// TL_TARGET("avx2,fma") lets it compile in a baseline-x86 TU and be reached
// only after select_ukernel's CPUID check.
TL_TARGET("avx2,fma") inline void ukernel_avx2_6x16(
    int64_t kc, const float* ap, const float* bp, float* c, int64_t ldc,
    int mr, int nr) {
  constexpr int KMR = 6, KNR = 16;
  __m256 ab0[KMR], ab1[KMR];
  for (int i = 0; i < KMR; i++) {
    ab0[i] = _mm256_setzero_ps();
    ab1[i] = _mm256_setzero_ps();
  }
#define TL_CPU_KSTEP(p)                                            \
  {                                                                \
    __m256 b0 = _mm256_loadu_ps(bp + (p)*KNR);                     \
    __m256 b1 = _mm256_loadu_ps(bp + (p)*KNR + 8);                 \
    const float* a = ap + (p)*KMR;                                 \
    __m256 av;                                                     \
    av = _mm256_broadcast_ss(a + 0);                               \
    ab0[0] = _mm256_fmadd_ps(av, b0, ab0[0]);                      \
    ab1[0] = _mm256_fmadd_ps(av, b1, ab1[0]);                      \
    av = _mm256_broadcast_ss(a + 1);                               \
    ab0[1] = _mm256_fmadd_ps(av, b0, ab0[1]);                      \
    ab1[1] = _mm256_fmadd_ps(av, b1, ab1[1]);                      \
    av = _mm256_broadcast_ss(a + 2);                               \
    ab0[2] = _mm256_fmadd_ps(av, b0, ab0[2]);                      \
    ab1[2] = _mm256_fmadd_ps(av, b1, ab1[2]);                      \
    av = _mm256_broadcast_ss(a + 3);                               \
    ab0[3] = _mm256_fmadd_ps(av, b0, ab0[3]);                      \
    ab1[3] = _mm256_fmadd_ps(av, b1, ab1[3]);                      \
    av = _mm256_broadcast_ss(a + 4);                               \
    ab0[4] = _mm256_fmadd_ps(av, b0, ab0[4]);                      \
    ab1[4] = _mm256_fmadd_ps(av, b1, ab1[4]);                      \
    av = _mm256_broadcast_ss(a + 5);                               \
    ab0[5] = _mm256_fmadd_ps(av, b0, ab0[5]);                      \
    ab1[5] = _mm256_fmadd_ps(av, b1, ab1[5]);                      \
  }
  int64_t p = 0;
  for (; p + 4 <= kc; p += 4) {
    TL_PREFETCH(ap + (p + 16) * KMR);
    TL_PREFETCH(bp + (p + 16) * KNR);
    TL_CPU_KSTEP(p);
    TL_CPU_KSTEP(p + 1);
    TL_CPU_KSTEP(p + 2);
    TL_CPU_KSTEP(p + 3);
  }
  for (; p < kc; p++) TL_CPU_KSTEP(p);
#undef TL_CPU_KSTEP
  if (mr == KMR && nr == KNR) {
    for (int i = 0; i < KMR; i++) {
      _mm256_storeu_ps(c + i * ldc,
                       _mm256_add_ps(_mm256_loadu_ps(c + i * ldc), ab0[i]));
      _mm256_storeu_ps(c + i * ldc + 8,
                       _mm256_add_ps(_mm256_loadu_ps(c + i * ldc + 8), ab1[i]));
    }
  } else {
    float tmp[KMR][KNR];
    for (int i = 0; i < KMR; i++) {
      _mm256_storeu_ps(tmp[i], ab0[i]);
      _mm256_storeu_ps(tmp[i] + 8, ab1[i]);
    }
    for (int i = 0; i < mr; i++)
      for (int j = 0; j < nr; j++) c[i * ldc + j] += tmp[i][j];
  }
}
// Runtime AVX2+FMA probe. GCC/Clang expose __builtin_cpu_supports; MSVC has no
// equivalent, so hand-roll the CPUID/XGETBV check: leaf-1 ECX[12]=FMA and
// ECX[27]=OSXSAVE, XCR0 must report XMM+YMM state (the OS preserves the ymm
// regs across context switches), and leaf-7 subleaf-0 EBX[5]=AVX2.
inline bool cpu_has_avx2_fma() {
#if defined(_MSC_VER)
  int r[4];
  __cpuid(r, 1);
  const bool fma = (r[2] & (1 << 12)) != 0;
  const bool osxsave = (r[2] & (1 << 27)) != 0;
  if (!osxsave) return false;
  if ((_xgetbv(0) & 0x6) != 0x6) return false;  // XMM + YMM state enabled
  __cpuidex(r, 7, 0);
  const bool avx2 = (r[1] & (1 << 5)) != 0;
  return avx2 && fma;
#else
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif  // TL_CPU_X86

// Pick the microkernel once, by ISA then (on x86) by CPUID. On ARM the choice
// is fixed at compile time (NEON always present); on x86 AVX2 needs a runtime
// probe because a generic build must still run on pre-AVX2 CPUs.
inline ukernel_desc select_ukernel() {
#if defined(TL_CPU_NEON)
  return {&ukernel_neon, MR, NR};
#elif defined(TL_CPU_X86)
  if (cpu_has_avx2_fma()) return {&ukernel_avx2_6x16, 6, 16};
  return {&ukernel_scalar, MR, NR};
#else
  return {&ukernel_scalar, MR, NR};
#endif
}

// The three loops both split paths of sgemm_ are made of. Panels are the
// microkernel's tile: mrt rows of A (an mrt×KC slot each, alpha folded in) and
// nrt columns of B (a kc×nrt slot each — kc, not KC, so a shallow K block does
// not scatter 640 B panels 32 KB apart). Edge panels are clamped to mr/nr and
// zero-padded by the packers.
inline void pack_a_panels(const float* A, int64_t as0, int64_t as1, int64_t m,
                          int64_t pc, int64_t kc, float alpha, int mrt,
                          int64_t ip0, int64_t ip1, float* dst) {
  for (int64_t ip = ip0; ip < ip1; ip++) {
    int64_t i0 = ip * mrt;
    int mr = static_cast<int>(std::min<int64_t>(mrt, m - i0));
    pack_a_panel(A + i0 * as0 + pc * as1, as0, as1, mr, mrt, kc, alpha,
                 dst + (ip - ip0) * mrt * KC);
  }
}
// `np` panels of B starting at column col0, clamped at col_end.
inline void pack_b_panels(const float* B, int64_t bs0, int64_t bs1, int64_t pc,
                          int64_t kc, int nrt, int64_t col0, int64_t col_end,
                          int64_t np, float* dst) {
  for (int64_t q = 0; q < np; q++) {
    int64_t j0 = col0 + q * nrt;
    int nr = static_cast<int>(std::min<int64_t>(nrt, col_end - j0));
    pack_b_panel(B + pc * bs0 + j0 * bs1, bs0, bs1, nr, nrt, kc,
                 dst + q * kc * nrt);
  }
}
// C[rows of panels ip0..ip1, cols col0..] += packed A panels × packed B
// panels: a B panel stays in L1 across the A panels (BLIS's order).
inline void macrokernel(ukernel_fn uk, int64_t kc, const float* apack, int mrt,
                        int64_t m, int64_t ip0, int64_t ip1, const float* bpack,
                        int nrt, int64_t col0, int64_t col_end, int64_t np,
                        float* C, int64_t n) {
  for (int64_t q = 0; q < np; q++) {
    int64_t j0 = col0 + q * nrt;
    int nr = static_cast<int>(std::min<int64_t>(nrt, col_end - j0));
    for (int64_t ip = ip0; ip < ip1; ip++) {
      int64_t i0 = ip * mrt;
      int mr = static_cast<int>(std::min<int64_t>(mrt, m - i0));
      uk(kc, apack + (ip - ip0) * mrt * KC, bpack + q * kc * nrt,
         C + i0 * n + j0, n, mr, nr);
    }
  }
}
// Grow-only scratch: the packers write every element they use (edge panels
// zero-padded by hand), so a resize's value-initialisation is wasted work, and
// a shrink-then-grow across gemms of alternating shapes was a memset per call.
inline float* scratch_(std::vector<float>& v, size_t need) {
  if (v.size() < need) v.resize(need);
  return v.data();
}

}  // namespace detail

// How many threads `macs` multiply-adds are worth: one per floor of work
// (min_work_per_thread_), so a small job never pays for threads it cannot
// feed. The cap every pool split in this backend goes through.
inline int threads_for_(int64_t macs) {
  return static_cast<int>(
      std::min<int64_t>(macs / min_work_per_thread_(), 1 << 30));
}

// C[m×n] = alpha · A · B, C row-major (ld = n), beta = 0 (C overwritten).
// A is m×k with strides (as0, as1); B is k×n with strides (bs0, bs1) — any
// layout, so a transposed view passes its base strides in place.
inline void sgemm(const float* A, int64_t as0, int64_t as1, const float* B,
                  int64_t bs0, int64_t bs1, float* C, int64_t m, int64_t n,
                  int64_t k, float alpha) {
  sgemm_(A, as0, as1, B, bs0, bs1, C, m, n, k, alpha, threads_for_(m * n * k));
}

inline void sgemm_(const float* A, int64_t as0, int64_t as1, const float* B,
                   int64_t bs0, int64_t bs1, float* C, int64_t m, int64_t n,
                   int64_t k, float alpha, int max_threads) {
  if (m == 0 || n == 0) return;
  std::memset(C, 0, static_cast<size_t>(m) * static_cast<size_t>(n) *
                        sizeof(float));
  if (k == 0) return;

  // Microkernel chosen once per process (magic-static init is thread-safe);
  // the indirect call is amortized over the kc-loop's hundreds of FMAs. The
  // descriptor carries the register-block tile (mr/nr) the packers must match.
  static const detail::ukernel_desc ukr = detail::select_ukernel();
  const detail::ukernel_fn ukernel = ukr.fn;
  const int mrt = ukr.mr, nrt = ukr.nr;
  auto& pool = thread_pool::instance();
  const int64_t mpanels = (m + mrt - 1) / mrt;

  // Which axis the threads split. Over M (a tall A) the caller packs a B
  // block once and each thread packs and streams its own A panels. A short M
  // against a wide N (a 256-row activation into a 3072-wide FFN weight) is the
  // other case: every thread would stream the whole 4 MB packed B block, which
  // no L2 holds, for a few rows of A, and each (jc, pc) block would be a fresh
  // round with a serial B pack in front of it — on 8 threads that left them
  // busy 56% of the time. So when A's K block fits a core's L2 and A is the
  // smaller operand to share, split N: per K block one round packs all of A,
  // then one round over the N panels in which each thread packs and owns a
  // slice of B, which fits its L2, and runs all of A against it.
  const bool split_n =
      static_cast<size_t>(m) * KC * sizeof(float) <= (1u << 20) &&
      m <= std::min<int64_t>(n, NC);
  if (split_n) {
    // A's K blocks are packed in groups of as many as fit kApackBudget (all
    // of them for the usual k), each group two rounds: every round costs the
    // pool's wake (~140 µs at 20 threads), so a 3072-deep K in six per-block
    // pairs would spend more waking than a 256×3072×768 gemm computes.
    static thread_local std::vector<float> apack_all;  // [kblock][panel]
    constexpr size_t kApackBudget = 8u << 20;
    const size_t block_floats = static_cast<size_t>(mpanels) * mrt * KC;
    const int64_t kblocks = (k + KC - 1) / KC;
    const int64_t group = std::max<int64_t>(
        1, std::min<int64_t>(kblocks,
                             kApackBudget / (block_floats * sizeof(float))));
    float* apack_p = detail::scratch_(apack_all, group * block_floats);
    const int64_t npanels = (n + nrt - 1) / nrt;
    constexpr int64_t kMaxPanels = NC / 16;  // a thread's B slice at a time
    for (int64_t b0 = 0; b0 < kblocks; b0 += group) {
      const int64_t b1 = std::min(kblocks, b0 + group);
      pool.parallel_for((b1 - b0) * mpanels, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; t++) {
          const int64_t b = b0 + t / mpanels, ip = t % mpanels;
          const int64_t pc = b * KC, kc = std::min<int64_t>(KC, k - pc);
          detail::pack_a_panels(A, as0, as1, m, pc, kc, alpha, mrt, ip, ip + 1,
                                apack_p + t * mrt * KC);
        }
      }, max_threads);
      pool.parallel_for(npanels, [&](int64_t jp0, int64_t jp1) {
        static thread_local std::vector<float> bpack_w;
        for (int64_t jq = jp0; jq < jp1; jq += kMaxPanels) {
          const int64_t np = std::min(kMaxPanels, jp1 - jq);
          for (int64_t b = b0; b < b1; b++) {
            const int64_t pc = b * KC, kc = std::min<int64_t>(KC, k - pc);
            float* bp = detail::scratch_(bpack_w, np * kc * nrt);
            detail::pack_b_panels(B, bs0, bs1, pc, kc, nrt, jq * nrt, n, np, bp);
            detail::macrokernel(ukernel, kc, apack_p + (b - b0) * block_floats,
                                mrt, m, 0, mpanels, bp, nrt, jq * nrt, n, np, C,
                                n);
          }
        }
      }, max_threads);
    }
    return;
  }

  // Split M. bpack lives on the calling thread (packed once per (jc, pc),
  // read by all); apack on each worker. Raw pointer for the workers: naming
  // `bpack` inside the lambda would resolve to each worker's own (empty)
  // thread_local instance.
  static thread_local std::vector<float> bpack;
  float* bpack_p = detail::scratch_(
      bpack, static_cast<size_t>(KC) * ((std::min<int64_t>(NC, n) + nrt - 1) / nrt) * nrt);
  const int64_t panels_per_mc = MC / mrt;
  for (int64_t jc = 0; jc < n; jc += NC) {
    const int64_t nc = std::min<int64_t>(NC, n - jc);
    const int64_t npanels = (nc + nrt - 1) / nrt;
    for (int64_t pc = 0; pc < k; pc += KC) {
      const int64_t kc = std::min<int64_t>(KC, k - pc);
      detail::pack_b_panels(B, bs0, bs1, pc, kc, nrt, jc, jc + nc, npanels,
                            bpack_p);
      // The M dimension at mrt-panel granularity, each thread packing and
      // running its panels in MC-row groups at most (the pool's chunks are
      // usually smaller: on this box balance across unequal cores beat the B
      // traffic a coarser chunk would have saved).
      pool.parallel_for(mpanels, [&](int64_t ip0, int64_t ip1) {
        static thread_local std::vector<float> apack;
        float* ap = detail::scratch_(apack, panels_per_mc * mrt * KC);
        for (int64_t g0 = ip0; g0 < ip1; g0 += panels_per_mc) {
          const int64_t g1 = std::min<int64_t>(g0 + panels_per_mc, ip1);
          detail::pack_a_panels(A, as0, as1, m, pc, kc, alpha, mrt, g0, g1, ap);
          detail::macrokernel(ukernel, kc, ap, mrt, m, g0, g1, bpack_p, nrt, jc,
                              jc + nc, npanels, C, n);
        }
      }, max_threads);
    }
  }
}

namespace detail {
#ifdef TL_CPU_X86
// Cephes' expf, eight lanes: n = floor(x·log2e + 1/2), r = x − n·ln2 (ln2 in
// two parts), a degree-5 polynomial in r, 2^n through the exponent bits.
// Within 7.7e-8 relative (~1.3 ulp) over [-88, 0], the max-shifted exponents
// softmax feeds it; results below FLT_MIN flush to 0.
// "avx2" without "fma" on purpose: nothing may fuse into an FMA, so the
// rounding is the same wherever this runs.
TL_TARGET("avx2") inline __m256 exp8_(__m256 x) {
  // min/max with x second: a NaN comes through (the other operand is kept).
  x = _mm256_min_ps(_mm256_set1_ps(88.3762626647949f), x);
  x = _mm256_max_ps(_mm256_set1_ps(-88.3762626647949f), x);
  __m256 fx = _mm256_floor_ps(_mm256_add_ps(
      _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
      _mm256_set1_ps(0.5f)));
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, z), x), _mm256_set1_ps(1.0f));
  __m256i n = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127));
  return _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(n, 23)));
}

TL_TARGET("avx2") inline void exp_shifted_avx2_(float* dst, const float* src,
                                                int64_t stride, int64_t n,
                                                float shift) {
  const __m256 s = _mm256_set1_ps(shift);
  int64_t i = 0;
  if (stride == 1) {
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(dst + i, exp8_(_mm256_sub_ps(_mm256_loadu_ps(src + i), s)));
  }
  // The tail, or a strided run, through a lane buffer and the same kernel:
  // an element's value does not depend on where in the row it sits.
  for (; i < n; i += 8) {
    const int64_t k = std::min<int64_t>(8, n - i);
    float buf[8] = {};
    for (int64_t j = 0; j < k; j++) buf[j] = src[(i + j) * stride];
    _mm256_storeu_ps(buf, exp8_(_mm256_sub_ps(_mm256_loadu_ps(buf), s)));
    std::memcpy(dst + i, buf, static_cast<size_t>(k) * sizeof(float));
  }
}
#endif  // TL_CPU_X86
}  // namespace detail

// dst[i] = exp(src[i·stride] − shift) over a run of n: softmax's exp pass.
// libm's expf is a scalar call per element (no -ffast-math, so nothing
// vectorizes it), ~80% of a CPU softmax; AVX2 hosts take exp8_ instead.
// Elsewhere this is std::exp, as before.
inline void exp_shifted(float* dst, const float* src, int64_t stride,
                        int64_t n, float shift) {
#ifdef TL_CPU_X86
  static const bool avx2 = detail::cpu_has_avx2_fma();
  if (avx2) return detail::exp_shifted_avx2_(dst, src, stride, n, shift);
#endif
  for (int64_t i = 0; i < n; i++) dst[i] = std::exp(src[i * stride] - shift);
}

}  // namespace cpu
}  // namespace tl
