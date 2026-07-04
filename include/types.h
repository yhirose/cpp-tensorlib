#pragma once

// Device selection. v1 computes in F32 only (BF16 lands later as a storage
// type); dtype is therefore not part of the public type system.

#include <cstdint>

namespace tl {

enum class device_type { cpu, gpu, auto_ };

inline device_type device_ = device_type::cpu;

inline void use_cpu() { device_ = device_type::cpu; }
inline void use_gpu() { device_ = device_type::gpu; }
inline void use_auto() { device_ = device_type::auto_; }

// tl::gpu_available() — true when a GPU backend is compiled in AND a device
// is reachable at runtime — is defined in metal.h (Metal on macOS; the CUDA
// dlopen probe joins it in M6). Where no device exists, gpu/auto modes fall
// back to the CPU path; CI runs all three modes so the fallback stays
// permanently tested.

// Testing/benchmark toggle: when false, evaluation ignores accelerated CPU
// backends and runs everything through the ref:: oracle. Lets tests compare
// backend results against the oracle in-process.
inline bool use_accelerate_ = true;

// Kernel families with distinct CPU/GPU crossover points (silarray's
// use_gpu_(n, class) design). auto_ mode compares an op's work size against
// one threshold per family; the manual cpu/gpu modes ignore them.
enum class kernel_class {
  elementwise,  // unary/binary maps, affine — n = elements
  reduction,    // softmax, row reductions — n = elements
  matmul,       // gemm — n = M*N*K
};

// auto_-mode size thresholds: the measured CPU vs GPU-total (single op +
// flush) standalone crossover per family — the size where GPU-total first
// beats CPU. The GPU backend is fixed at build time by the gpu:: facade
// (CUDA off-Apple when TENSORLIB_CUDA, else Metal on Apple), so the crossover
// is a compile-time property too and the thresholds branch on the same macro.
// These are the conservative standalone values; pipelined graphs amortize the
// flush, so the effective crossover is lower. Re-measure per target with
// misc/census.cpp.
inline int64_t auto_threshold_(kernel_class kc) {
#if defined(TENSORLIB_CUDA)
  // RTX 3090 (sm_86) census 2026-07-04, misc/census.cpp. No AMX rival on the
  // CPU side (own BLIS), so the GPU wins far earlier than on Apple's Metal:
  //   matmul      64^3=2.6e5 already GPU (0.038 vs cpu 0.152 ms) — crossover
  //               is at/below the smallest measured size; below trivial matmul
  //               that CPU handles cheaply anyway.
  //   elementwise 256K tie (0.374 vs 0.377), 1M GPU (0.438 vs 0.575) → ~5e5.
  //   reduction   softmax 16K already GPU (0.034 vs 0.044); CPU's per-row loop
  //               is slow, GPU wins from the smallest measured size.
  switch (kc) {
    case kernel_class::matmul: return 200'000;          // ~58^3; GPU by 64^3
    case kernel_class::elementwise: return 500'000;     // tie ~256K, GPU by 1M
    case kernel_class::reduction: return 16'000;        // GPU from ~16K
  }
#else
  // Metal / M1 Pro census 2026-07-03 (load ~4, interleaved medians):
  //   matmul      1024^3=1.07e9 tie, 1280^3=2.1e9 tie, 1536^3=3.6e9 GPU wins
  //               → crossover ~2e9 (~1260^3). AMX is strong; standalone GPU
  //               matmul only pays off past here.
  //   elementwise 1M cpu wins (0.15 vs 0.25 ms), 4M GPU wins (0.55 vs 0.91)
  //               → crossover ~2e6.
  //   reduction   softmax 256x256=65536 tie, 1024x256=262144 GPU wins clearly
  //               (0.26 vs 0.87 ms) → crossover ~2e5. Set slightly above the
  //               softmax crossover for the lighter row_sum/row_max.
  switch (kc) {
    case kernel_class::matmul: return 2'000'000'000;    // ~1260^3
    case kernel_class::elementwise: return 2'000'000;   // 2M elements
    case kernel_class::reduction: return 200'000;       // ~2e5 elements
  }
#endif
  return 1'000'000'000'000;  // unreachable
}

}  // namespace tl
