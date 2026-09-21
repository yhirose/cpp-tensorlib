#pragma once

// The GPU facade: array.h and storage.h dispatch through tl::gpu, so the eval
// seam carries no platform #ifdefs. tl::gpu is two layers:
//
//   shared    gpu_abi.h — the op vocabulary, `span` (a device handle and a byte
//             offset: the currency every op takes), the kernel ABI and the
//             launch policy — and gpu_ops.h, where every op is written once.
//             An op's signature exists there and nowhere else.
//   backend   the selected backend header, reached through the using-directive
//             below. What a backend provides, its device core:
//               lifecycle  available / pending / flush / cpu_barrier
//               memory     alloc / release / sync_to_host / upload
//               launch     dispatch(kernel, views, params, grid): the one way
//                          a shared op runs a kernel; false for a kernel id the
//                          backend has none for
//               own        the ops it runs its own way (a different algorithm,
//                          several kernels, a host round trip), as static
//                          members of `struct own` with the shared signature.
//                          gpu_ops.h forwards to the ones that exist; an op a
//                          backend does not declare has no stub to keep in step
//               traits     what the launch policy may assume of its kernels
//               caps       what a model may assume (model_path, graph_capture,
//                          row_gemv, bf16_gemm), plus the graph-capture plumbing
//                          — graph_available / capture_begin / capture_end /
//                          graph_launch / graph_destroy / upload_u32 /
//                          attn_dpos_partials_bytes — as no-ops where absent
//
// An op answers false when the backend has no kernel for it, and the evaluator
// falls back to the CPU. The model path (what a decoder runs on raw device
// buffers between its GEMVs and attention: kv_cache.h, bench/models) has no CPU
// fallback, so a model checks the return and keeps to the array ops where it
// is false. gpu::census(kernel) counts launches, which is how a test tells a
// kernel that ran from an op that quietly fell back.
//
// One backend is selected below, by the gate its header is written under:
// webgpu.h under TENSORLIB_WEBGPU && __EMSCRIPTEN__, cuda.h under TENSORLIB_CUDA
// && !__APPLE__, metal.h under __APPLE__, and gpu_null.h — no device, every op
// declines — for a build none of them fits. gpu_null.h is also the template:
// it is everything this file asks of a backend, with nothing in it. Adding a
// backend is a header that fills that in, its kernels, and one branch here.

// WebGPU is checked first: a wasm build defines neither __APPLE__ nor
// TENSORLIB_CUDA, but a host build could define TENSORLIB_WEBGPU by accident
// and should not take a backend that cannot work there.
#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)
#include "webgpu.h"
#define TL_GPU_BACKEND webgpu
#elif defined(TENSORLIB_CUDA) && !defined(__APPLE__)
#include "cuda.h"
#define TL_GPU_BACKEND cuda
#elif defined(__APPLE__)
#include "metal.h"
#define TL_GPU_BACKEND metal
#else
#include "gpu_null.h"
#define TL_GPU_BACKEND null_gpu
#endif

namespace tl {

// A using-directive rather than an alias, so tl::gpu can hold the shared layer
// too: a name declared in tl::gpu itself (a shared op) is found first, and one
// that is not falls through to the backend.
namespace gpu {
using namespace TL_GPU_BACKEND;
}  // namespace gpu

inline bool gpu_available() { return gpu::available(); }

}  // namespace tl

#include "gpu_ops.h"

#undef TL_GPU_BACKEND
