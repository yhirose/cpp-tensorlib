#pragma once

// Own CPU backend (M5) — the portable SGEMM for platforms without
// Accelerate (Linux/Windows; also runnable on Apple for validation).
//
// BLIS-style GEMM: cache-blocking loops + packing (architecture-independent)
// around a register-blocked microkernel selected per ISA at compile time.
// Raw-pointer API (no dependency on array.h, like metal.h), so array.h calls
// it with plain buffers + strides; packing is stride-aware so transposed
// views feed in place (no materialization), matching the Accelerate path.
//
// Scope (M5 first cut): SGEMM only. Elementwise / reductions ride array.h's
// contiguous flat-loop fast paths, which are memory-bound and autovectorize
// — the compute-bound GEMM is what needs the hand-blocked kernel.
//
// Status: scaffolding + scalar microkernel (portable, correct on any ISA)
// + NEON microkernel (Apple Silicon / ARM Linux, native), NEON-tuned on
// M1 Pro (lane-indexed FMA, K-unroll, prefetch, thread-local pack buffers,
// MR-panel-granular parallelism — see performance-notes.md). AVX2/AVX-512
// microkernels are the next drop-in behind the same interface, validated
// on the x86 box (see docs/roadmap.md).

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TL_CPU_NEON 1
#endif

#include "cpu_threadpool.h"

namespace tl {
namespace cpu {

// Gates the own-CPU GEMM in the eval dispatch. Default on: it is the
// primary CPU GEMM off-Apple, and on Apple it sits below Accelerate (only
// reached when use_accelerate_ is off). Oracle tests set it false (with
// use_accelerate_ false) to force the ref:: path.
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

namespace detail {

// Pack an MR-tall, kc-wide panel of A (with arbitrary strides, alpha folded
// in) into column-major micropanels: ap[p*MR + i] = alpha * A[i*as0 + p*as1].
// Rows past `mr` are zero-padded so the microkernel always runs full MR×NR.
inline void pack_a_panel(const float* A, int64_t as0, int64_t as1, int mr,
                         int64_t kc, float alpha, float* ap) {
  for (int64_t p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) ap[p * MR + i] = alpha * A[i * as0 + p * as1];
    for (int i = mr; i < MR; i++) ap[p * MR + i] = 0.0f;
  }
}

// Pack a kc-tall, NR-wide panel of B into row-major micropanels:
// bp[p*NR + j] = B[p*bs0 + j*bs1]. Cols past `nr` are zero-padded.
inline void pack_b_panel(const float* B, int64_t bs0, int64_t bs1, int nr,
                         int64_t kc, float* bp) {
  for (int64_t p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) bp[p * NR + j] = B[p * bs0 + j * bs1];
    for (int j = nr; j < NR; j++) bp[p * NR + j] = 0.0f;
  }
}

// Microkernel: C[mr×nr] += (packed A panel) · (packed B panel), accumulating
// over kc. C is row-major with leading dimension ldc. Edge tiles (mr<MR or
// nr<NR) read the zero-padded panels and store only the valid mr×nr corner.

#ifdef TL_CPU_NEON
inline void ukernel(int64_t kc, const float* ap, const float* bp, float* c,
                    int64_t ldc, int mr, int nr) {
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
    __builtin_prefetch(ap + (p + 16) * MR);
    __builtin_prefetch(bp + (p + 16) * NR);
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
#else
inline void ukernel(int64_t kc, const float* ap, const float* bp, float* c,
                    int64_t ldc, int mr, int nr) {
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
#endif

}  // namespace detail

// C[m×n] = alpha · A · B, C row-major (ld = n), beta = 0 (C overwritten).
// A is m×k with strides (as0, as1); B is k×n with strides (bs0, bs1) — any
// layout, so a transposed view passes its base strides in place.
inline void sgemm(const float* A, int64_t as0, int64_t as1, const float* B,
                  int64_t bs0, int64_t bs1, float* C, int64_t m, int64_t n,
                  int64_t k, float alpha) {
  if (m == 0 || n == 0) return;
  std::memset(C, 0, static_cast<size_t>(m) * static_cast<size_t>(n) *
                        sizeof(float));
  if (k == 0) return;

  // Reusable pack buffers: thread-local so no per-call allocation (a
  // per-call std::vector dominated small sizes in the first cut). bpack
  // lives on the calling thread (packed once per (jc,pc), read by all);
  // apack lives on each worker thread.
  static thread_local std::vector<float> bpack;
  bpack.resize(static_cast<size_t>(KC) *
               ((std::min<int64_t>(NC, n) + NR - 1) / NR) * NR);
  // Raw pointer for the workers: naming `bpack` inside the lambda would
  // resolve to each worker's own (empty) thread_local instance.
  float* bpack_p = bpack.data();
  auto& pool = thread_pool::instance();

  for (int64_t jc = 0; jc < n; jc += NC) {
    int64_t nc = std::min<int64_t>(NC, n - jc);
    for (int64_t pc = 0; pc < k; pc += KC) {
      int64_t kc = std::min<int64_t>(KC, k - pc);
      // Pack B[pc:pc+kc, jc:jc+nc] into NR-wide row-major micropanels.
      int64_t npanels = (nc + NR - 1) / NR;
      for (int64_t jp = 0; jp < npanels; jp++) {
        int64_t j0 = jp * NR;
        int nr = static_cast<int>(std::min<int64_t>(NR, nc - j0));
        detail::pack_b_panel(B + (pc)*bs0 + (jc + j0) * bs1, bs0, bs1, nr, kc,
                             bpack_p + jp * KC * NR);
      }
      // Parallelize the M dimension at MR-panel granularity (not MC-block):
      // small m (e.g. 256 → 2 MC blocks) would otherwise use only 1-2
      // threads. Each thread packs+computes its contiguous panel range in
      // MC-row groups, preserving the L2 blocking.
      int64_t mpanels = (m + MR - 1) / MR;
      const int64_t panels_per_mc = MC / MR;
      pool.parallel_for(mpanels, [&](int64_t ip0, int64_t ip1) {
        static thread_local std::vector<float> apack;
        apack.resize(static_cast<size_t>(MC) * KC);
        for (int64_t g0 = ip0; g0 < ip1; g0 += panels_per_mc) {
          int64_t g1 = std::min<int64_t>(g0 + panels_per_mc, ip1);
          // Pack A[g0*MR : ..., pc:pc+kc] * alpha into MR-tall panels.
          for (int64_t ip = g0; ip < g1; ip++) {
            int64_t i0 = ip * MR;
            int mr = static_cast<int>(std::min<int64_t>(MR, m - i0));
            detail::pack_a_panel(A + i0 * as0 + pc * as1, as0, as1, mr, kc,
                                 alpha, apack.data() + (ip - g0) * MR * KC);
          }
          // Macrokernel: MR×NR microkernel over the packed panels.
          for (int64_t jp = 0; jp < npanels; jp++) {
            int64_t j0 = jp * NR;
            int nr = static_cast<int>(std::min<int64_t>(NR, nc - j0));
            for (int64_t ip = g0; ip < g1; ip++) {
              int64_t i0 = ip * MR;
              int mr = static_cast<int>(std::min<int64_t>(MR, m - i0));
              detail::ukernel(kc, apack.data() + (ip - g0) * MR * KC,
                              bpack_p + jp * KC * NR,
                              C + i0 * n + (jc + j0), n, mr, nr);
            }
          }
        }
      });
    }
  }
}

}  // namespace cpu
}  // namespace tl
