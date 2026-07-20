#pragma once

// The GPU-backend facade: array.h and storage.h dispatch through tl::gpu, so
// the eval seam carries no platform #ifdefs. Every backend header exposes the
// identical API (available/pending/flush/alloc/release/binary/unary/gemm/
// row_op/sync_to_host) and shares tl::metal::kop, which makes the alias below a
// drop-in.
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
#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)
namespace gpu = webgpu;
#elif defined(TENSORLIB_CUDA) && !defined(__APPLE__)
namespace gpu = cuda;
#else
namespace gpu = metal;
#endif

inline bool gpu_available() { return gpu::available(); }

}  // namespace tl
