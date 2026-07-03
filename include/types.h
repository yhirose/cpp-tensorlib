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

// auto_-mode size thresholds: the measured CPU vs GPU-total (encode +
// device + sync) crossover per family. PROVISIONAL — taken on a loaded
// machine (2026-07-02, load avg ~9; see performance-notes.md): matmul
// parity at 1024^3 ≈ 1.1e9, clear GPU win at 2048^3; elementwise ~1.4x GPU
// win at 4M contiguous. Re-run the census quiet before trusting the auto
// default for real workloads.
inline int64_t auto_threshold_(kernel_class kc) {
  switch (kc) {
    case kernel_class::matmul: return 500'000'000;      // ~800^3
    case kernel_class::elementwise: return 4'000'000;   // 4M elements
    case kernel_class::reduction: return 8'000'000;     // unmeasured: high
  }
  return 1'000'000'000'000;  // unreachable
}

}  // namespace tl
