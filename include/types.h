#pragma once

// Device selection and storage dtype. Compute is F32 everywhere; bf16 (M7) is
// a STORAGE type only — 16-bit in memory, widened to F32 on load. In slice A1
// a bf16 array is a weight container: the CUDA decode GEMV consumes it
// natively; every other op widens it to an F32 copy first (correct on all
// backends, no kernel changes). Activations, results and scalars stay F32.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace tl {

// f32/bf16 are plain storage widths. q4 (M8) is group-symmetric int4 *weight*
// storage: physically [N,K] (out×in) packed 2 nibbles/byte with per-group f32
// scales appended in the same buffer — the logical array shape stays [K,N] so
// `a.dot(Wq)` type-checks like f32; the decode GEMV reads it natively and every
// other op dequantizes to F32 (to_f32) via the same widen-fallback seam as bf16.
enum class dtype : uint8_t { f32, bf16, q4 };

// Byte width for the plain widths (q4 is variable — use q4_bytes()).
inline int64_t dtype_size(dtype dt) { return dt == dtype::bf16 ? 2 : 4; }

// q4 layout: fixed group size, one buffer = [packed int4 [N][K/2] | scales
// f32 [N][K/kQ4Group]]. K must be a multiple of kQ4Group (the kernel's per-lane
// tail guard handles K % 256 != 0, e.g. Qwen's K=896).
inline constexpr int64_t kQ4Group = 32;
inline int64_t q4_bytes(int64_t N, int64_t K) {
  return N * K / 2 + N * (K / kQ4Group) * 4;
}

// Scalar converters (host side). bf16 = top 16 bits of the F32 pattern;
// narrow rounds to nearest-even (the same rounding the CUDA path uses).
inline uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}
inline float bf16_to_f32(uint16_t h) {
  uint32_t x = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

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
  // Metal / M1 Pro. elementwise/reduction: census 2026-07-03 (load ~4,
  // interleaved medians):
  //   elementwise 1M cpu wins (0.15 vs 0.25 ms), 4M GPU wins (0.55 vs 0.91)
  //               → crossover ~2e6.
  //   reduction   softmax 256x256=65536 tie, 1024x256=262144 GPU wins clearly
  //               (0.26 vs 0.87 ms) → crossover ~2e5. Set slightly above the
  //               softmax crossover for the lighter row_sum/row_max.
  //   matmul      re-anchored 2026-07-18 on the pipelined transformer bench
  //               (STEEL kernels incl. transposed operands, no mid-graph
  //               drains): GPU wins from the d=768 block (largest gemm
  //               256·768·3072 = 6.0e8), CPU still wins at d=512 (2.7e8)
  //               → crossover ~5e8. The old 2e9 (~1260^3) was the standalone
  //               single-op+flush crossover; in-graph the flush amortizes,
  //               and the sticky rule drags the rest of the block along, so
  //               the pipelined anchor is the one auto_ should use.
  switch (kc) {
    case kernel_class::matmul: return 500'000'000;      // ~794^3 (pipelined)
    case kernel_class::elementwise: return 2'000'000;   // 2M elements
    case kernel_class::reduction: return 200'000;       // ~2e5 elements
  }
#endif
  return 1'000'000'000'000;  // unreachable
}

// Batch device bias (auto_ mode). The per-op auto_threshold_ is greedy: it
// decides each op in isolation, so a transformer block's small projection
// gemms (e.g. seq·d·d = 256·768·768 = 1.5e8, below the matmul threshold)
// strand on the CPU while its one big FFN gemm goes to the GPU — the batch
// then thrashes CPU⇄GPU (each hand-off is a blocking flush). This threshold
// is compared against the *sum* of M·N·K over every dot in one evaluation
// batch (graph::run_); above it the whole batch is pinned to the GPU so the
// projections ride along and the pipeline stays on one device. Env override
// TL_BATCH_MATMUL_BIAS (element-product units) for host calibration, mirroring
// the misc/census.cpp re-measure note above.
inline int64_t batch_matmul_bias_threshold_() {
  static const int64_t v = []() -> int64_t {
    if (const char* e = std::getenv("TL_BATCH_MATMUL_BIAS")) {
      char* end = nullptr;
      long long x = std::strtoll(e, &end, 10);
      if (end != e && x > 0) return static_cast<int64_t>(x);
    }
#if defined(TENSORLIB_CUDA)
    return 4'000'000;             // GPU wins early; a couple of small gemms
#else
    // Metal / M1 Pro, calibrated 2026-07-18 on the pipelined transformer bench
    // (position-rotated, per-config min). The batch bias helps only when the
    // whole block is GPU-favourable: pinning d768's ~7e8 attention batch to
    // the GPU is a net loss (encode/flush > the small gemms' savings), while
    // pinning d1024's ~1.2e9 attention batch is a large win (8.73→5.6 ms/iter).
    // Sit the threshold between them; env TL_BATCH_MATMUL_BIAS re-calibrates.
    return 800'000'000;
#endif
  }();
  return v;
}

}  // namespace tl
