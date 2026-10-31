// CUDA kernels for the M6 own-CUDA backend — the non-Apple analogue of
// metal_kernels.metal. AOT-compiled to PTX by nvcc and #embed'd into cuda.h,
// then loaded at runtime via the dlopen'd driver API (cuModuleLoadData +
// cuLaunchKernel). No CUDA runtime, no cuBLAS/CUTLASS — own kernels only.
//
// Conventions mirror the Metal kernels exactly so the graph's fused affine
// epilogue works identically on both backends:
//   * every elementwise/gemm kernel computes  out = expr * scale + offset
//     (fma), so fused scalar chains and dot-then-scale cost zero extra passes;
//   * softmax ignores scale/offset (a pre-softmax affine is not meaningful),
//     matching softmax_ in the MSL source;
//   * row_sum/row_max write one value per row with the affine epilogue.
// View offsets are folded host-side (pointer arithmetic on the managed
// allocation), so kernels take already-offset pointers — simpler than Metal's
// setBuffer:offset:. Kernels are extern "C" for stable, unmangled PTX names.
//
// Correctness-first (M6 stage 1). The SGEMM here is a one-output-per-thread
// tiled loop — the tuned cuBLAS-90% ladder is the stage-2 sprint (roadmap M6).

#include <cuda_bf16.h>  // __nv_bfloat16 + __bfloat162float (M7 storage widen)
#include <cuda_fp16.h>
#include <type_traits>  // __half: the prefill attention's tensor-core operands

// TL_ALIGN(kernel, a0, a1, ...): the byte alignment each leading pointer
// parameter of `kernel` needs, written beside the kernel wherever its loads or
// stores are wider than an element (a float4 read: 16; a float2 store: 8; a
// Q4_K word: 4). The host reads it from the PTX (cuda.h, parse_align_decls_)
// and declines a launch off it, so a misaligned view is refused, never
// misread. Pointers past the last one listed, and every pointer of a kernel
// without one, need only their element's width.
#define TL_ALIGN(kernel, ...) \
  extern "C" __device__ const unsigned char kernel##_ptr_align[] = {__VA_ARGS__};

// The kernels by layer (gpu_abi.h): each file builds on the ones before it.
// Kernels sit in extern "C" blocks for a stable, unmangled symbol each (the
// driver loads them by name); a block closes around the __device__ templates,
// which cannot have C linkage, and their __global__ wrappers re-declare it.
#include "cuda_core.cuh"
#include "cuda_nn.cuh"
#include "cuda_llm.cuh"
