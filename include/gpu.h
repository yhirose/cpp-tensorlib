#pragma once

// The GPU facade: array.h and storage.h dispatch through tl::gpu, so the eval
// seam carries no platform #ifdefs. tl::gpu is two things laid over each other:
//
//   shared    gpu_abi.h (the op vocabulary, span, the kernel ABI) and gpu_ops.h
//             (ops written once, over the backend's `dispatch`):
//               binary / unary / unary_ext / binary_bcast / compare / clamp /
//               scalar_binary / row_op / row_logsumexp / layer_norm /
//               index_select / gather_from_axis / xent_bwd / adam_step
//   backend   whatever the selected backend header declares, reached through
//             the using-directive below. Until an op moves to gpu_ops.h, each
//             backend header declares it with the identical signature — the
//             contract, i.e. everything array.h/storage.h may call:
//   lifecycle  available / pending / flush / cpu_barrier
//   memory     alloc / release / sync_to_host / upload
//   kernels    binary_bcast_nd / where_nd / copy_nd /
//              gemm / gemm_batched / gemm_bias / pad / fold /
//              index_add / scatter_to_axis / sum_to /
//              concat_part / rope / layer_norm_bwd
//   LLM path   gemv_f32 / gemv_bf16 / gemv_q4 / attn_decode / attn_prefill /
//              attn_prefill_dq / attn_prefill_dkv
//   model path kv_append / kv_fill / argmax / rmsnorm / rmsnorm_res / swiglu /
//              split_heads / merge_heads / gemv_bf16_row / gemm_bf16_nt
// A backend with no kernel for one of these returns false and the evaluator
// falls back to the CPU — so the LLM row is real on CUDA and stubs elsewhere
// (tools/check_backend_parity.py checks every name on both lines here
// against cuda.h/metal.h/webgpu.h and fails CI if one drifts unannounced).
// The model path is what a decoder (kv_cache.h, bench/models) runs on raw
// device buffers between its GEMVs and attention; it has no CPU fallback,
// so a model checks the return and keeps to the array ops where it is false.
//
// Beyond the kernels, each backend states what a model may assume of it in
// `caps` (model_path, graph_capture, row_gemv, bf16_gemm, flat_addressing),
// and carries the graph-capture group — graph_available / capture_begin /
// capture_end /
// graph_launch / graph_destroy / upload_u32 / incr_u32 / rope_dpos /
// kv_append_dpos / attn_decode_dpos / attn_dpos_partials_bytes — as no-ops
// where the capability is false, so a decoder is written once and branches on
// caps. `upload` (staging host bytes into a device buffer) is memory, not
// capture: every decoder needs it for the embedding row it feeds each step.
//
// Each backend compiles to stubs unless its own gate holds, so including all of
// them is free: metal.h is real only on __APPLE__, cuda.h only on
// TENSORLIB_CUDA && !__APPLE__, webgpu.h only on TENSORLIB_WEBGPU &&
// __EMSCRIPTEN__. The alias picks the one that can do real work.
//
// This lived at the bottom of cuda.h until M10 — the only place both namespaces
// happened to be visible. That stopped scaling at the third backend, since
// adding a browser GPU meant editing the CUDA header.

#include "cuda.h"
#include "metal.h"
#include "webgpu.h"

namespace tl {

// WebGPU is checked first: a wasm build defines neither __APPLE__ nor
// TENSORLIB_CUDA, but a host build could define TENSORLIB_WEBGPU by accident
// and should not silently take a backend that cannot work there — webgpu::
// is stubs unless __EMSCRIPTEN__ too, so the order is safe either way.
// A using-directive rather than an alias, so tl::gpu can hold the shared layer
// too: a name declared in tl::gpu itself (a shared op) is found first, and one
// that is not falls through to the backend.
namespace gpu {
#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)
using namespace webgpu;
#elif defined(TENSORLIB_CUDA) && !defined(__APPLE__)
using namespace cuda;
#else
using namespace metal;
#endif
}  // namespace gpu

inline bool gpu_available() { return gpu::available(); }

}  // namespace tl

#include "gpu_ops.h"
