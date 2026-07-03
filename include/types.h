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

// auto_-mode size thresholds: the measured CPU (Accelerate) vs GPU-total
// (Metal single op + flush) standalone crossover per family — the size
// where GPU-total first beats CPU. Census 2026-07-03 on M1 Pro (load ~4,
// interleaved medians; misc/census.cpp):
//   matmul      1024^3=1.07e9 tie, 1280^3=2.1e9 tie, 1536^3=3.6e9 GPU wins
//               → crossover ~2e9 (~1260^3). AMX is strong; standalone GPU
//               matmul only pays off past here. Pipelined training amortizes
//               the flush, so the effective threshold could be lower — this
//               is the conservative standalone value.
//   elementwise 1M cpu wins (0.15 vs 0.25 ms), 4M GPU wins (0.55 vs 0.91)
//               → crossover ~2e6.
//   reduction   softmax 256x256=65536 tie, 1024x256=262144 GPU wins clearly
//               (0.26 vs 0.87 ms) → crossover ~2e5. GPU wins reductions
//               early (CPU's per-row loop is slow); set slightly above the
//               softmax crossover for the lighter row_sum/row_max.
inline int64_t auto_threshold_(kernel_class kc) {
  switch (kc) {
    case kernel_class::matmul: return 2'000'000'000;    // ~1260^3
    case kernel_class::elementwise: return 2'000'000;   // 2M elements
    case kernel_class::reduction: return 200'000;       // ~2e5 elements
  }
  return 1'000'000'000'000;  // unreachable
}

}  // namespace tl
