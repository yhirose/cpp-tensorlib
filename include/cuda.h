#pragma once

// Own CUDA backend (M6) — the non-Apple GPU backend, mirroring metal.h. The
// NVIDIA driver is loaded via dlopen (no link-time CUDA dependency), so a
// binary built with TENSORLIB_CUDA still runs — and falls back to CPU — on a
// machine with no driver. Kernels are AOT-compiled to PTX by nvcc and #embed'd
// (see kernels/tensorlib_cuda.cu), then loaded through the driver API at first
// use. No CUDA runtime, no cuBLAS/CUTLASS.
//
// Memory: a persistent host/device MIRROR per allocation. cuMemAlloc gives a
// real device buffer (`native`, what kernels use); a paired host buffer
// (`contents`, what CPU ops read/write) is malloc'd alongside. A per-allocation
// dirty state (keyed by the device pointer in the leaked context, so views that
// share one storage share one mirror entry) drives lazy copies: H2D before a
// kernel reads a host-dirty buffer, D2H before the CPU reads a device-dirty one
// (array::raw()/data() → gpu::sync_to_host). This replaces the original
// cuMemAllocManaged model: on WSL2 managed pages are never migrated to the
// device (cudaDevAttrConcurrentManagedAccess=0; prefetch/advise return "invalid
// device ordinal"), so compute-bound GEMM on managed memory ran ~88× slower
// than on device memory — the roadmap's pre-authorized device-buffer pivot.
// View offsets are folded host-side into the pointer passed to each kernel.
//
// Real implementation is gated on TENSORLIB_CUDA && !__APPLE__ (Apple uses
// Metal; a plain build gets the stubs below). The API matches metal.h exactly
// — available/pending/flush/alloc/release/binary/unary/gemm/row_op — so the
// eval_one dispatch seam is backend-agnostic and carries no platform #ifdefs.

#include <cstdint>

#include "metal.h"  // reuse tl::metal::kop (platform-independent op enum)
#include "types.h"  // tl::dtype (KV cache storage width)

namespace tl {
namespace cuda {

using kop = tl::metal::kop;

#if defined(TENSORLIB_CUDA) && !defined(__APPLE__)

}  // namespace cuda
}  // namespace tl

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tl {
namespace cuda {

// Dynamic-loader shim: dlopen/dlsym on Unix, LoadLibrary/GetProcAddress on
// Windows (where the driver ships as nvcuda.dll). Symbols are cast to the
// hand-declared function-pointer types by the caller, same as before.
inline void* dl_open(const char* path) {
#ifdef _WIN32
  return reinterpret_cast<void*>(::LoadLibraryA(path));
#else
  return ::dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#endif
}
inline void* dl_sym(void* lib, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(
      ::GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return ::dlsym(lib, name);
#endif
}

// ---- driver API surface (declared by hand; loaded from libcuda via dlopen) ----
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUgraph_st;
struct CUgraphExec_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUstream = CUstream_st*;
using CUgraph = CUgraph_st*;
using CUgraphExec = CUgraphExec_st*;

struct driver {
  CUresult (*Init)(unsigned) = nullptr;
  CUresult (*DeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*DeviceGetCount)(int*) = nullptr;
  CUresult (*DevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*CtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*CtxSynchronize)() = nullptr;
  CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, CUstream, void**,
                           void**) = nullptr;
  CUresult (*MemAlloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*MemFree)(CUdeviceptr) = nullptr;
  CUresult (*MemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
  CUresult (*MemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream) =
      nullptr;
  // CUDA-graph capture (M9 C1-2). Optional: dlsym'd best-effort; graph_ok()
  // gates the fast replay path, everything else works without them.
  CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) =
      nullptr;
  CUresult (*StreamCreate)(CUstream*, unsigned) = nullptr;
  CUresult (*StreamDestroy)(CUstream) = nullptr;
  CUresult (*StreamSynchronize)(CUstream) = nullptr;
  CUresult (*StreamBeginCapture)(CUstream, int /*CUstreamCaptureMode*/) =
      nullptr;
  CUresult (*StreamEndCapture)(CUstream, CUgraph*) = nullptr;
  CUresult (*GraphInstantiate)(CUgraphExec*, CUgraph, unsigned long long) =
      nullptr;
  CUresult (*GraphLaunch)(CUgraphExec, CUstream) = nullptr;
  CUresult (*GraphExecDestroy)(CUgraphExec) = nullptr;
  CUresult (*GraphDestroy)(CUgraph) = nullptr;

  bool ok() const {
    return Init && DeviceGet && DevicePrimaryCtxRetain && CtxSetCurrent &&
           CtxSynchronize && ModuleLoadData && ModuleGetFunction &&
           LaunchKernel && MemAlloc && MemFree && MemcpyHtoD && MemcpyDtoH &&
           MemsetD8;
  }
  bool graph_ok() const {
    return MemcpyHtoDAsync && StreamCreate && StreamBeginCapture &&
           StreamEndCapture && GraphInstantiate && GraphLaunch &&
           GraphExecDestroy && GraphDestroy && StreamSynchronize;
  }
};

// The embedded PTX (nvcc-compiled from kernels/tensorlib_cuda.cu, then turned
// into a C byte array by the build — bin2c style, not C23 #embed, since the
// off-Apple compilers here (g++ 11 / clang 14) predate #embed). The build
// generates tensorlib_cuda_ptx.inc (a comma-separated byte list ending in a
// 0x00 terminator, which cuModuleLoadData requires for PTX) and puts it on the
// include path.
inline const char* ptx_source_() {
  static const unsigned char src[] = {
#include "tensorlib_cuda_ptx.inc"
  };
  return reinterpret_cast<const char*>(src);
}

inline const char* kernel_name_(kop op) {
  switch (op) {
    case kop::add: return "tl_add";
    case kop::sub: return "tl_sub";
    case kop::mul: return "tl_mul";
    case kop::div: return "tl_div";
    case kop::exp_: return "tl_exp";
    case kop::log_: return "tl_log";
    case kop::sqrt_: return "tl_sqrt";
    case kop::sigmoid: return "tl_sigmoid";
    case kop::relu: return "tl_relu";
    case kop::affine: return "tl_affine";
    case kop::softmax: return "tl_softmax";
    case kop::row_sum: return "tl_row_sum";
    case kop::row_max: return "tl_row_max";
    default: return "tl_sgemm";  // sgemm* / steel* all route to tl_sgemm
  }
}

struct context {
  void* lib = nullptr;
  driver d;
  CUcontext ctx = nullptr;
  CUmodule mod = nullptr;
  bool ready = false;
  bool pending = false;
  // The stream every kernel launch / async copy targets. Null = the default
  // stream (the normal path). Temporarily set to a capture stream while
  // recording a CUDA graph, then restored — so no launcher needs a stream arg.
  CUstream stream = nullptr;
  CUstream cap_stream = nullptr;  // dedicated capture stream (created on demand)
  // Diagnostic knob (M9 decode gemv census): when true, gemv_run_ forces gy=1
  // (no split-K, so no MemsetD8Async + atomicAdd combine). Lets the bench and
  // the decode loop A/B the split-K path at the small Qwen shapes. Default off.
  bool no_splitk = false;
  std::unordered_map<int, CUfunction> fns;

  // Host/device mirror per allocation, keyed by the device pointer (== the
  // `native` handle stored in storage). Views sharing a storage share the key,
  // so one dirty state serves every view. loc tracks where the live copy is.
  enum loc { HOST, DEVICE, BOTH };
  struct mirror {
    float* host = nullptr;  // CPU-side buffer (storage.contents/ptr)
    CUdeviceptr dev = 0;    // device buffer (storage.native)
    size_t bytes = 0;
    loc where = HOST;
  };
  std::unordered_map<CUdeviceptr, mirror> mirrors;

  // Size-keyed free list (like Metal's MTLBuffer pool). Released buffers are
  // recycled, not cuMemFree'd — repeated large alloc/free otherwise fragments
  // the driver allocator (decode benches, training that churns activations).
  // Buffers persist until the (leaked) context tears down. Keyed by exact byte
  // size; the workloads that churn reuse identical shapes.
  std::unordered_map<size_t, std::vector<std::pair<CUdeviceptr, float*>>> pool;

  mirror* mirror_(void* native) {
    auto it = mirrors.find(reinterpret_cast<CUdeviceptr>(native));
    return it == mirrors.end() ? nullptr : &it->second;
  }
  // A kernel is about to READ this buffer: ensure the device copy is current.
  void device_read_(void* native) {
    mirror* m = mirror_(native);
    if (m && m->where == HOST) {
      d.MemcpyHtoD(m->dev, m->host, m->bytes);  // sync; serializes on null stream
      m->where = BOTH;
    }
  }
  // A kernel is about to WRITE this buffer: it becomes the live copy.
  void device_write_(void* native) {
    if (mirror* m = mirror_(native)) m->where = DEVICE;
  }

  static context& get() {
    static auto* c = new context();  // leaked: outlives all storage deleters
    return *c;
  }

  context() {
#ifdef _WIN32
    const char* paths[] = {"nvcuda.dll"};
#else
    const char* paths[] = {"/usr/lib/wsl/lib/libcuda.so.1", "libcuda.so.1",
                           "libcuda.so"};
#endif
    for (const char* p : paths) {
      lib = dl_open(p);
      if (lib) break;
    }
    if (!lib) return;  // no driver → available()==false → CPU fallback
    auto S = [&](const char* n) { return dl_sym(lib, n); };
    d.Init = (CUresult(*)(unsigned))S("cuInit");
    d.DeviceGet = (CUresult(*)(CUdevice*, int))S("cuDeviceGet");
    d.DeviceGetCount = (CUresult(*)(int*))S("cuDeviceGetCount");
    d.DevicePrimaryCtxRetain =
        (CUresult(*)(CUcontext*, CUdevice))S("cuDevicePrimaryCtxRetain");
    d.CtxSetCurrent = (CUresult(*)(CUcontext))S("cuCtxSetCurrent");
    d.CtxSynchronize = (CUresult(*)())S("cuCtxSynchronize");
    d.ModuleLoadData = (CUresult(*)(CUmodule*, const void*))S("cuModuleLoadData");
    d.ModuleGetFunction =
        (CUresult(*)(CUfunction*, CUmodule, const char*))S("cuModuleGetFunction");
    d.LaunchKernel =
        (CUresult(*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                     unsigned, unsigned, unsigned, CUstream, void**,
                     void**))S("cuLaunchKernel");
    // _v2 memory calls are the current ABI; fall back to the base name.
    d.MemAlloc = (CUresult(*)(CUdeviceptr*, size_t))S("cuMemAlloc_v2");
    if (!d.MemAlloc) d.MemAlloc = (CUresult(*)(CUdeviceptr*, size_t))S("cuMemAlloc");
    d.MemFree = (CUresult(*)(CUdeviceptr))S("cuMemFree_v2");
    if (!d.MemFree) d.MemFree = (CUresult(*)(CUdeviceptr))S("cuMemFree");
    d.MemcpyHtoD =
        (CUresult(*)(CUdeviceptr, const void*, size_t))S("cuMemcpyHtoD_v2");
    if (!d.MemcpyHtoD)
      d.MemcpyHtoD = (CUresult(*)(CUdeviceptr, const void*, size_t))S("cuMemcpyHtoD");
    d.MemcpyDtoH =
        (CUresult(*)(void*, CUdeviceptr, size_t))S("cuMemcpyDtoH_v2");
    if (!d.MemcpyDtoH)
      d.MemcpyDtoH = (CUresult(*)(void*, CUdeviceptr, size_t))S("cuMemcpyDtoH");
    d.MemsetD8 =
        (CUresult(*)(CUdeviceptr, unsigned char, size_t))S("cuMemsetD8_v2");
    if (!d.MemsetD8)
      d.MemsetD8 = (CUresult(*)(CUdeviceptr, unsigned char, size_t))S("cuMemsetD8");
    // CUDA-graph symbols (optional; graph_ok() gates their use).
    d.MemsetD8Async = (CUresult(*)(CUdeviceptr, unsigned char, size_t,
                                   CUstream))S("cuMemsetD8Async");
    d.MemcpyHtoDAsync = (CUresult(*)(CUdeviceptr, const void*, size_t,
                                     CUstream))S("cuMemcpyHtoDAsync_v2");
    if (!d.MemcpyHtoDAsync)
      d.MemcpyHtoDAsync = (CUresult(*)(CUdeviceptr, const void*, size_t,
                                       CUstream))S("cuMemcpyHtoDAsync");
    d.StreamCreate = (CUresult(*)(CUstream*, unsigned))S("cuStreamCreate");
    d.StreamDestroy = (CUresult(*)(CUstream))S("cuStreamDestroy_v2");
    if (!d.StreamDestroy)
      d.StreamDestroy = (CUresult(*)(CUstream))S("cuStreamDestroy");
    d.StreamSynchronize = (CUresult(*)(CUstream))S("cuStreamSynchronize");
    d.StreamBeginCapture =
        (CUresult(*)(CUstream, int))S("cuStreamBeginCapture_v2");
    if (!d.StreamBeginCapture)
      d.StreamBeginCapture =
          (CUresult(*)(CUstream, int))S("cuStreamBeginCapture");
    d.StreamEndCapture =
        (CUresult(*)(CUstream, CUgraph*))S("cuStreamEndCapture");
    d.GraphInstantiate = (CUresult(*)(CUgraphExec*, CUgraph,
                                      unsigned long long))S(
        "cuGraphInstantiateWithFlags");
    d.GraphLaunch = (CUresult(*)(CUgraphExec, CUstream))S("cuGraphLaunch");
    d.GraphExecDestroy = (CUresult(*)(CUgraphExec))S("cuGraphExecDestroy");
    d.GraphDestroy = (CUresult(*)(CUgraph))S("cuGraphDestroy");
    if (!d.ok()) return;

    if (d.Init(0) != 0) return;
    int cnt = 0;
    if (!d.DeviceGetCount || d.DeviceGetCount(&cnt) != 0 || cnt < 1) return;
    CUdevice dev = 0;
    if (d.DeviceGet(&dev, 0) != 0) return;
    if (d.DevicePrimaryCtxRetain(&ctx, dev) != 0) return;
    d.CtxSetCurrent(ctx);
    if (d.ModuleLoadData(&mod, ptx_source_()) != 0) return;
    ready = true;
  }

  CUfunction fn_(kop op) {
    int key = static_cast<int>(op);
    auto it = fns.find(key);
    if (it != fns.end()) return it->second;
    CUfunction f = nullptr;
    d.ModuleGetFunction(&f, mod, kernel_name_(op));
    fns[key] = f;
    return f;
  }

  // The register-blocked SGEMM fast path (tl_sgemm_rb), cached separately from
  // the kop table since it has no kop of its own.
  CUfunction sgemm_rb_fn = nullptr;
  CUfunction sgemm_rb_() {
    if (!sgemm_rb_fn) d.ModuleGetFunction(&sgemm_rb_fn, mod, "tl_sgemm_rb");
    return sgemm_rb_fn;
  }

  // M7 decode GEMV (f32 and bf16-weight variants), cached like sgemm_rb.
  CUfunction gemv_f32_fn = nullptr, gemv_bf16_fn = nullptr, gemv_bf16v8_fn = nullptr;
  CUfunction gemv_f32_() {
    if (!gemv_f32_fn) d.ModuleGetFunction(&gemv_f32_fn, mod, "tl_gemv_f32");
    return gemv_f32_fn;
  }
  CUfunction gemv_bf16_() {
    if (!gemv_bf16_fn) d.ModuleGetFunction(&gemv_bf16_fn, mod, "tl_gemv_bf16");
    return gemv_bf16_fn;
  }
  CUfunction gemv_bf16v8_() {
    if (!gemv_bf16v8_fn)
      d.ModuleGetFunction(&gemv_bf16v8_fn, mod, "tl_gemv_bf16v8");
    return gemv_bf16v8_fn;
  }
  CUfunction gemv_bf16_row_fn = nullptr;
  CUfunction gemv_bf16_row_() {
    if (!gemv_bf16_row_fn)
      d.ModuleGetFunction(&gemv_bf16_row_fn, mod, "tl_gemv_bf16_row");
    return gemv_bf16_row_fn;
  }

  // M8 int4-weight decode GEMV.
  CUfunction gemv_q4_fn = nullptr;
  CUfunction gemv_q4_() {
    if (!gemv_q4_fn) d.ModuleGetFunction(&gemv_q4_fn, mod, "tl_gemv_q4");
    return gemv_q4_fn;
  }

  // M9 fused decode attention (single-pass + split-KV two-pass).
  // head_dim {64,128} variants (M9): each templated instantiation has its own
  // symbol; the launchers pick by D. The unsuffixed name is the D=128 build.
  // Each attention/KV kernel has an f32 and a bf16-KV-storage instantiation
  // (M9 bf16 KV cache): the bf16 variants read/write K,V as __nv_bfloat16 while
  // q/out/scratch stay f32. The getters pick by (D, kv_bf16); a small 2x2 cache.
  CUfunction attn_decode_fn = nullptr, attn_split_fn = nullptr,
             attn_combine_fn = nullptr;
  CUfunction attn_decode_64_fn = nullptr, attn_split_64_fn = nullptr;
  CUfunction attn_decode_bf16_fn = nullptr, attn_decode_bf16_64_fn = nullptr;
  CUfunction attn_split_bf16_fn = nullptr, attn_split_bf16_64_fn = nullptr;
  CUfunction attn_decode_(int64_t D, bool bf16 = false) {
    CUfunction* slot = bf16 ? (D == 64 ? &attn_decode_bf16_64_fn : &attn_decode_bf16_fn)
                            : (D == 64 ? &attn_decode_64_fn : &attn_decode_fn);
    if (!*slot)
      d.ModuleGetFunction(slot, mod,
                          bf16 ? (D == 64 ? "tl_attn_decode_bf16_64" : "tl_attn_decode_bf16")
                               : (D == 64 ? "tl_attn_decode_f32_64" : "tl_attn_decode_f32"));
    return *slot;
  }
  CUfunction attn_split_(int64_t D, bool bf16 = false) {
    CUfunction* slot = bf16 ? (D == 64 ? &attn_split_bf16_64_fn : &attn_split_bf16_fn)
                            : (D == 64 ? &attn_split_64_fn : &attn_split_fn);
    if (!*slot)
      d.ModuleGetFunction(slot, mod,
                          bf16 ? (D == 64 ? "tl_attn_decode_split_bf16_64" : "tl_attn_decode_split_bf16")
                               : (D == 64 ? "tl_attn_decode_split_64" : "tl_attn_decode_split"));
    return *slot;
  }
  CUfunction attn_combine_() {  // head_dim implicit (blockDim.x) — one symbol
    if (!attn_combine_fn)
      d.ModuleGetFunction(&attn_combine_fn, mod, "tl_attn_combine");
    return attn_combine_fn;
  }

  // M9 KV cache append (scatter one token's k,v into the persistent cache).
  CUfunction kv_append_fn = nullptr, kv_append_bf16_fn = nullptr;
  CUfunction kv_append_(bool bf16 = false) {
    CUfunction* slot = bf16 ? &kv_append_bf16_fn : &kv_append_fn;
    if (!*slot)
      d.ModuleGetFunction(slot, mod, bf16 ? "tl_kv_append_bf16" : "tl_kv_append");
    return *slot;
  }

  // RoPE (rotary position embedding) for q/k.
  CUfunction rope_fn = nullptr;
  CUfunction rope_() {
    if (!rope_fn) d.ModuleGetFunction(&rope_fn, mod, "tl_rope");
    return rope_fn;
  }

  // Device-pos variants (CUDA-graph capture): pos/ctx read from a device scalar
  // so one instantiated graph replays correctly as the decode position advances.
  CUfunction rope_dpos_fn = nullptr, incr_u32_fn = nullptr,
             kv_append_dpos_fn = nullptr, attn_decode_dpos_fn = nullptr,
             attn_decode_dpos_64_fn = nullptr;
  CUfunction rope_dpos_() {
    if (!rope_dpos_fn) d.ModuleGetFunction(&rope_dpos_fn, mod, "tl_rope_dpos");
    return rope_dpos_fn;
  }
  CUfunction incr_u32_() {
    if (!incr_u32_fn) d.ModuleGetFunction(&incr_u32_fn, mod, "tl_incr_u32");
    return incr_u32_fn;
  }
  CUfunction kv_append_dpos_() {
    if (!kv_append_dpos_fn)
      d.ModuleGetFunction(&kv_append_dpos_fn, mod, "tl_kv_append_dpos");
    return kv_append_dpos_fn;
  }
  CUfunction attn_decode_dpos_(int64_t D) {
    CUfunction* slot = D == 64 ? &attn_decode_dpos_64_fn : &attn_decode_dpos_fn;
    if (!*slot)
      d.ModuleGetFunction(slot, mod,
                          D == 64 ? "tl_attn_decode_f32_64_dpos"
                                  : "tl_attn_decode_f32_dpos");
    return *slot;
  }

  // GPU argmax (greedy last-mile): kernel + a persistent 4-byte device result
  // buffer so the per-token result is a 4-byte D2H, not the 608KB logits copy.
  CUfunction argmax_fn = nullptr;
  CUfunction argmax_() {
    if (!argmax_fn) d.ModuleGetFunction(&argmax_fn, mod, "tl_argmax");
    return argmax_fn;
  }
  CUdeviceptr argmax_res = 0;
  CUdeviceptr argmax_res_() {
    if (!argmax_res && d.MemAlloc(&argmax_res, 16) != 0) argmax_res = 0;
    return argmax_res;
  }

  // Fused decode-step ops (imperative path): RMSNorm + SwiGLU.
  CUfunction rmsnorm_fn = nullptr, swiglu_fn = nullptr;
  CUfunction rmsnorm_() {
    if (!rmsnorm_fn) d.ModuleGetFunction(&rmsnorm_fn, mod, "tl_rmsnorm");
    return rmsnorm_fn;
  }
  CUfunction swiglu_() {
    if (!swiglu_fn) d.ModuleGetFunction(&swiglu_fn, mod, "tl_swiglu");
    return swiglu_fn;
  }
  CUfunction add_rmsnorm_fn = nullptr;
  CUfunction add_rmsnorm_() {
    if (!add_rmsnorm_fn)
      d.ModuleGetFunction(&add_rmsnorm_fn, mod, "tl_add_rmsnorm");
    return add_rmsnorm_fn;
  }

  // M9 prefill: bulk cache fill + causal prefill attention.
  CUfunction kv_fill_fn = nullptr, kv_fill_bf16_fn = nullptr,
             attn_prefill_fn = nullptr;
  CUfunction kv_fill_(bool bf16 = false) {
    CUfunction* slot = bf16 ? &kv_fill_bf16_fn : &kv_fill_fn;
    if (!*slot)
      d.ModuleGetFunction(slot, mod, bf16 ? "tl_kv_fill_bf16" : "tl_kv_fill");
    return *slot;
  }
  CUfunction attn_prefill_64_fn = nullptr;
  CUfunction attn_prefill_bf16_fn = nullptr, attn_prefill_bf16_64_fn = nullptr;
  CUfunction attn_prefill_(int64_t D, bool bf16 = false) {
    CUfunction* slot = bf16 ? (D == 64 ? &attn_prefill_bf16_64_fn : &attn_prefill_bf16_fn)
                            : (D == 64 ? &attn_prefill_64_fn : &attn_prefill_fn);
    if (!*slot)
      d.ModuleGetFunction(slot, mod,
                          bf16 ? (D == 64 ? "tl_attn_prefill_bf16_64" : "tl_attn_prefill_bf16")
                               : (D == 64 ? "tl_attn_prefill_f32_64" : "tl_attn_prefill_f32"));
    return *slot;
  }

  // Reusable device scratch for split-KV partials. Grown as needed, reused
  // across attention calls (sequential on the null stream), freed at teardown.
  CUdeviceptr attn_scratch = 0;
  size_t attn_scratch_bytes = 0;
  CUdeviceptr attn_scratch_(size_t bytes) {
    if (bytes > attn_scratch_bytes) {
      if (attn_scratch) d.MemFree(attn_scratch);  // syncs; fine (grows once)
      if (d.MemAlloc(&attn_scratch, bytes) != 0) {
        attn_scratch = 0;
        attn_scratch_bytes = 0;
        return 0;
      }
      attn_scratch_bytes = bytes;
    }
    return attn_scratch;
  }

  // char* pointer arithmetic to fold a byte offset into a managed pointer.
  static float* off_(void* base, int64_t byte_off) {
    return reinterpret_cast<float*>(static_cast<char*>(base) + byte_off);
  }

  bool launch1d_(CUfunction f, unsigned n, void** args, unsigned shared = 0) {
    if (!f) return false;
    unsigned block = 256, grid = (n + block - 1) / block;
    if (grid == 0) grid = 1;
    pending = true;
    return d.LaunchKernel(f, grid, 1, 1, block, 1, 1, shared, stream, args,
                          nullptr) == 0;
  }
};

inline bool available() { return context::get().ready; }

// Diagnostic knob: force gemv to skip split-K (gy=1). See context::no_splitk.
inline void set_no_splitk(bool v) { context::get().no_splitk = v; }
inline bool pending() { return context::get().pending; }

// End the batch: block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  c.d.CtxSynchronize();
  c.pending = false;
}

// ---- CUDA-graph capture (M9 C1-2): record a fixed launch sequence once and
// replay it as a single submit, erasing per-launch host overhead. Only the
// imperative decode step (no host sync / blocking copy mid-stream) is
// capturable; embed staging + argmax happen outside the captured region.
inline bool graph_available() { return context::get().d.graph_ok(); }

// Begin capturing: route every subsequent launch/async-copy onto a private
// capture stream. Drains the default stream first. Returns false if graph
// support is missing. Pair with capture_end().
inline bool capture_begin() {
  auto& c = context::get();
  if (!c.ready || !c.d.graph_ok()) return false;
  if (c.pending) flush();
  if (!c.cap_stream && c.d.StreamCreate(&c.cap_stream, 0) != 0) return false;
  if (c.d.StreamBeginCapture(c.cap_stream, 0 /*GLOBAL*/) != 0) return false;
  c.stream = c.cap_stream;
  return true;
}
// End capture and instantiate an executable graph (nullptr on failure).
// Restores the default stream.
inline CUgraphExec capture_end() {
  auto& c = context::get();
  CUgraph g = nullptr;
  CUresult r = c.d.StreamEndCapture(c.cap_stream, &g);
  c.stream = nullptr;
  if (r != 0 || !g) return nullptr;
  CUgraphExec e = nullptr;
  if (c.d.GraphInstantiate(&e, g, 0) != 0) e = nullptr;
  c.d.GraphDestroy(g);
  return e;
}
// Replay a captured graph on the default stream (marks work pending; the caller
// flushes or reads results as usual).
inline bool graph_launch(CUgraphExec e) {
  auto& c = context::get();
  if (!c.ready || !e) return false;
  if (c.d.GraphLaunch(e, nullptr) != 0) return false;
  c.pending = true;
  return true;
}
inline void graph_destroy(CUgraphExec e) {
  auto& c = context::get();
  if (e && c.d.GraphExecDestroy) c.d.GraphExecDestroy(e);
}

// Blocking H2D into a device buffer's mirror, marking it BOTH (device current).
// Used to pre-stage inputs (e.g. the embedding row) before a capture, so the
// captured region contains no blocking copy.
inline void upload(void* native, const float* src, int64_t n) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m) return;
  std::memcpy(m->host, src, (size_t)n * sizeof(float));
  c.d.MemcpyHtoD(m->dev, m->host, m->bytes);
  m->where = context::BOTH;
}

// Set a device u32 scalar (e.g. the capture pos counter) via its mirror. Raw
// 4-byte H2D — the mirror's host bytes are set to `val` then copied to device.
inline void upload_u32(void* native, unsigned val) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m || m->bytes < 4) return;
  std::memcpy(m->host, &val, 4);
  c.d.MemcpyHtoD(m->dev, m->host, 4);
  m->where = context::BOTH;
}

// Mirror allocation: a device buffer (returned as `native`) paired with a host
// buffer (returned via `contents`). They are DISTINCT memory — the mirror's
// dirty state copies between them on demand (device_read_/sync_to_host). storage
// keeps native != contents, like Metal (MTLBuffer handle vs .contents pointer).
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  size_t nb = bytes > 0 ? (size_t)bytes : 4;
  CUdeviceptr dev = 0;
  float* host = nullptr;
  auto it = c.pool.find(nb);  // reuse a recycled buffer of this exact size
  if (it != c.pool.end() && !it->second.empty()) {
    dev = it->second.back().first;
    host = it->second.back().second;
    it->second.pop_back();
  } else {
    if (c.d.MemAlloc(&dev, nb) != 0) return nullptr;
    host = static_cast<float*>(std::malloc(nb));
    if (!host) {
      c.d.MemFree(dev);
      return nullptr;
    }
  }
  c.mirrors[dev] = context::mirror{host, dev, nb, context::HOST};
  if (contents) *contents = host;
  return reinterpret_cast<void*>(dev);
}

inline void release(void* buf, int64_t, float*) {
  auto& c = context::get();
  if (!c.ready || !buf) return;
  CUdeviceptr dev = reinterpret_cast<CUdeviceptr>(buf);
  auto it = c.mirrors.find(dev);
  if (it == c.mirrors.end()) {
    c.d.MemFree(dev);  // untracked (shouldn't happen); free outright
    return;
  }
  c.pool[it->second.bytes].push_back({dev, it->second.host});  // recycle
  c.mirrors.erase(it);
}

// Reconcile a buffer for a CPU access: flush pending kernels, then D2H if the
// device holds the live copy. for_write invalidates the device copy (the host
// is about to mutate it). No-op for heap storages / unknown pointers.
inline void sync_to_host(void* native, bool for_write) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m) return;
  if (c.pending) flush();
  if (m->where == context::DEVICE) {
    c.d.MemcpyDtoH(m->host, m->dev, m->bytes);
    m->where = context::BOTH;
  }
  if (for_write) m->where = context::HOST;
}

// out = (a OP b) * scale + offset, contiguous; offsets in bytes.
inline bool binary(kop op, void* a, int64_t ao, void* b, int64_t bo, void* out,
                   int64_t oo, int64_t n, float scale, float offset) {
  if (op == kop::pow_) return false;  // no CUDA pow kernel yet — CPU fallback
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(b);
  c.device_write_(out);
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pa, &pb, &po, &un, &scale, &offset};
  return c.launch1d_(c.fn_(op), un, args);
}

// Rank-2 broadcast binary — no CUDA kernel yet; false sends the evaluator
// down the CPU fallback, matching pre-bcast behavior on this backend.
inline bool binary_bcast(kop, void*, int64_t, int64_t, int64_t, void*, int64_t,
                         int64_t, int64_t, void*, int64_t, int64_t, int64_t,
                         float, float) {
  return false;
}

inline bool unary(kop op, void* a, int64_t ao, void* out, int64_t oo, int64_t n,
                  float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_write_(out);
  float* pa = context::off_(a, ao);
  float* po = context::off_(out, oo);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pa, &po, &un, &scale, &offset};
  return c.launch1d_(c.fn_(op), un, args);
}

// M7 decode GEMV: y(n) = a(1,k) @ B(k,n), F32 accumulate. B is either f32 or
// bf16 weights (bf16 halves the dominant K×N weight traffic — the decode
// bandwidth lever). Buffers are opaque device pointers; the kernel interprets
// B's dtype. Offset 0 (contiguous weight/activation operands). Separate from
// gemm() so the M=1 path skips the 128×128 tile that wastes 127 rows.
//
// Split-K when the N/256 column-blocks alone underfill the SMs (small-N layers):
// partition K over gridDim.y, atomicAdd into a pre-zeroed y, so the kernel stays
// bandwidth-bound rather than occupancy-bound. gridDim.y==1 stores directly.
inline bool gemv_run_(CUfunction f, float* pa, float* pB, float* py,
                      void* y_native, unsigned un, unsigned uk,
                      unsigned vcols = 1) {
  auto& c = context::get();
  unsigned per = 256u * vcols;  // output columns covered by one block
  unsigned bx = (un + per - 1) / per;
  if (bx == 0) bx = 1;
  unsigned gy = 1, ksplit = uk;
  const long target = 164;  // ~2 blocks per SM on the 82-SM RTX 3090
  if (!c.no_splitk && static_cast<long>(bx) < target && uk >= 512) {
    unsigned g = static_cast<unsigned>((target + bx - 1) / bx);
    unsigned chunk = (uk + g - 1) / g;
    chunk = (chunk + 31u) & ~31u;
    if (chunk == 0) chunk = 32;
    unsigned s = (uk + chunk - 1) / chunk;
    if (s > 1) {
      gy = s;
      ksplit = chunk;
    }
  }
  if (gy > 1) {
    // Zero y for the split-K atomicAdd. Async on the stream (ordered before the
    // gemv on the same stream) so this stays capturable — a blocking MemsetD8
    // is illegal mid CUDA-graph capture.
    CUdeviceptr yd = reinterpret_cast<CUdeviceptr>(y_native);
    if (c.d.MemsetD8Async) c.d.MemsetD8Async(yd, 0, (size_t)un * 4, c.stream);
    else c.d.MemsetD8(yd, 0, (size_t)un * 4);
  }
  void* args[] = {&pa, &pB, &py, &un, &uk, &ksplit};
  c.pending = true;
  return c.d.LaunchKernel(f, bx, gy, 1, 256, 1, 1, 0, c.stream, args, nullptr) ==
         0;
}
inline bool gemv_f32(void* a, void* B, void* y, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(B);
  c.device_write_(y);
  return gemv_run_(c.gemv_f32_(), context::off_(a, 0), context::off_(B, 0),
                   context::off_(y, 0), y, static_cast<unsigned>(n),
                   static_cast<unsigned>(k));
}
inline bool gemv_bf16(void* a, void* B, void* y, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(B);  // B reinterpreted as __nv_bfloat16* in-kernel
  c.device_write_(y);
  // Vectorized 8-cols/thread path when n%8==0 (all transformer dims) — 16-byte
  // bf16 loads close the bandwidth gap to f32; scalar fallback otherwise.
  bool v8 = (n % 8) == 0;
  return gemv_run_(v8 ? c.gemv_bf16v8_() : c.gemv_bf16_(), context::off_(a, 0),
                   context::off_(B, 0), context::off_(y, 0), y,
                   static_cast<unsigned>(n), static_cast<unsigned>(k),
                   v8 ? 8u : 1u);
}

// Block size (32..256 threads) for the one-block-per-row GEMVs (tl_gemv_bf16_row,
// tl_gemv_q4), chosen per K to minimize the per-thread iteration count (llama.cpp's
// mul_mat_vec_f strategy) — a wide row (large K) gets more warps collaborating on
// the reduction. Ties keep the smaller block size (loop only replaces on strictly
// fewer iters). Always a multiple of 32 and <=256, as both kernels require.
inline unsigned gemv_row_block_size(int64_t k) {
  unsigned best = 32;
  int64_t niter_best = INT64_MAX;
  for (unsigned bs = 32; bs <= 256; bs += 32) {
    int64_t niter = (k + 8 * bs - 1) / (8 * bs);
    if (niter < niter_best) { niter_best = niter; best = bs; }
  }
  return best;
}
// Companion dynamic-shared size for a gemv_row_block_size block: one float per
// warp for the cross-warp reduce, or 0 for a single-warp block (which never
// touches shared). Kept beside the block-size policy so the two can't drift.
inline unsigned gemv_row_smem(unsigned block) {
  return block > 32 ? (block >> 5) * (unsigned)sizeof(float) : 0u;
}

// Warp-per-row bf16 decode GEMV (lever A): y(N) = a(1,K) @ W[N,K], W row-major
// (K contiguous per output row). ONE BLOCK per output row (grid.x == N), no
// split-K — no memset, no atomic combine. The small-N floor-bound lever; see
// tl_gemv_bf16_row. Requires K % 8 == 0 (host-gated; caller falls back to the
// split-K [K,N] path otherwise).
inline bool gemv_bf16_row(void* a, void* B, void* y, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready || (k % 8) != 0) return false;
  c.device_read_(a);
  c.device_read_(B);  // B reinterpreted as __nv_bfloat16* [N][K] in-kernel
  c.device_write_(y);
  float* pa = context::off_(a, 0);
  float* pB = context::off_(B, 0);
  float* py = context::off_(y, 0);
  unsigned uN = static_cast<unsigned>(n), uK = static_cast<unsigned>(k);
  void* args[] = {&pa, &pB, &py, &uN, &uK};
  unsigned block = gemv_row_block_size(k);
  c.pending = true;
  return c.d.LaunchKernel(c.gemv_bf16_row_(), uN, 1, 1, block, 1, 1,
                          gemv_row_smem(block), c.stream, args, nullptr) == 0;
}

// M8 int4-weight decode GEMV: y(N) = a(1,K) @ dequant(Wq[N,K]), F32 accumulate.
// qw = packed int4 [N][K/8] words, scales = f32 [N][K/group]. ONE BLOCK per
// output row (grid.x == N), K-adaptive block size — see gemv_bf16_row.
// K % group == 0, group % 8 == 0 (host-gated); the kernel's per-thread tail
// guard lifts the old K % 256 requirement (Qwen K=896 = 3×256+128 works).
inline bool gemv_q4(void* a, void* qw, void* scales, void* y, int64_t N,
                    int64_t K, int64_t group) {
  auto& c = context::get();
  if (!c.ready || group <= 0 || (K % group) != 0 || (group % 8) != 0)
    return false;
  c.device_read_(a);
  c.device_read_(qw);
  c.device_read_(scales);
  c.device_write_(y);
  float* pa = context::off_(a, 0);
  float* pq = context::off_(qw, 0);
  float* ps = context::off_(scales, 0);
  float* py = context::off_(y, 0);
  unsigned uN = (unsigned)N, uK = (unsigned)K, uG = (unsigned)group;
  void* args[] = {&pa, &pq, &ps, &py, &uN, &uK, &uG};
  unsigned block = gemv_row_block_size(K);
  c.pending = true;
  return c.d.LaunchKernel(c.gemv_q4_(), uN, 1, 1, block, 1, 1,
                          gemv_row_smem(block), c.stream, args, nullptr) == 0;
}

// M9 fused decode attention: out(h,:) = softmax(scale·q(h,:)·K(kv,:)^T)·V(kv) in
// one pass. q [n_q_heads,D], out [n_q_heads,D]; K/V are a [n_kv_heads,kv_max,D]
// cache read over its valid prefix [0,ctx) (kv_max==ctx is the no-cache case).
// GQA: q head h reads kv head h/(n_q_heads/n_kv_heads). Contiguous, D∈{64,128}.
inline bool attn_decode(void* q, void* K, void* V, void* out, int64_t n_q_heads,
                        int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                        int64_t D, float scale, bool kv_bf16 = false) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  c.device_read_(q);
  c.device_read_(K);
  c.device_read_(V);
  c.device_write_(out);
  float* pq = context::off_(q, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  float* po = context::off_(out, 0);
  unsigned uh = static_cast<unsigned>(n_q_heads), uctx = static_cast<unsigned>(ctx);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  unsigned group = static_cast<unsigned>(n_q_heads / n_kv_heads);

  // Split ctx over gridDim.y so grid = heads×S fills the SMs (heads alone is
  // ~32 blocks « 82 SMs). Target ~4 blocks/SM; each split needs enough keys
  // (>=128, multiple of 4 warps) to amortize its fixed cost.
  unsigned S = 1;
  {
    const long target = 328;  // ~4 * 82 SMs
    if (uh > 0 && (long)uh < target && ctx >= 256) {
      unsigned want = static_cast<unsigned>((target + uh - 1) / uh);
      unsigned max_s = static_cast<unsigned>(ctx / 128);  // >=128 keys/split
      if (want > max_s) want = max_s;
      if (want > 1) S = want;
    }
  }
  unsigned uD = static_cast<unsigned>(D);
  if (S == 1) {
    void* args[] = {&pq, &pk, &pv, &po, &uctx, &kv_stride, &group, &scale};
    c.pending = true;
    return c.d.LaunchKernel(c.attn_decode_(D, kv_bf16), uh, 1, 1, uD, 1, 1, 0,
                            nullptr, args, nullptr) == 0;
  }

  unsigned chunk = (uctx + S - 1) / S;
  chunk = (chunk + 3u) & ~3u;  // multiple of the warp count (4 for D=128, 2 for 64)
  if (chunk == 0) chunk = 4;
  S = (uctx + chunk - 1) / chunk;  // recompute after rounding
  // scratch: pm[H*S] , pl[H*S] , pacc[H*S*D]
  size_t hs = (size_t)uh * S;
  size_t bytes = (hs * 2 + hs * (size_t)D) * sizeof(float);
  CUdeviceptr scr = c.attn_scratch_(bytes);
  if (!scr) return false;
  float* pm = reinterpret_cast<float*>(scr);
  float* pl = pm + hs;
  float* pacc = pl + hs;
  c.pending = true;
  void* a1[] = {&pq,     &pk,        &pv,    &pm,    &pl,
                &pacc,   &uctx,      &kv_stride, &group, &chunk,
                &scale};
  if (c.d.LaunchKernel(c.attn_split_(D, kv_bf16), uh, S, 1, uD, 1, 1, 0,
                       c.stream, a1, nullptr) != 0)
    return false;
  void* a2[] = {&pm, &pl, &pacc, &po, &S};
  return c.d.LaunchKernel(c.attn_combine_(), uh, 1, 1, uD, 1, 1, 0, c.stream, a2,
                          nullptr) == 0;
}

// ---- CUDA-graph-capture device-pos launchers (A-min). Each mirrors its
// by-value sibling but sources pos/ctx from a device scalar `d_pos` (a 4-byte
// device buffer the caller owns) and launches on c.stream so the op is captured.
// attn_decode_dpos is single-block-per-head (S=1) only — pos-independent grid.

// RoPE reading pos from *d_pos (else identical to rope()).
inline bool rope_dpos(void* x, void* out, int64_t rows, int64_t T, int64_t D,
                      void* d_pos, float base, void* bias = nullptr) {
  auto& c = context::get();
  if (!c.ready || D <= 0 || (D & 1)) return false;
  c.device_read_(x);
  if (bias) c.device_read_(bias);
  c.device_write_(out);
  c.device_read_(d_pos);
  float* px = context::off_(x, 0);
  float* pbias = bias ? context::off_(bias, 0) : nullptr;
  float* po = context::off_(out, 0);
  float* pp = context::off_(d_pos, 0);
  unsigned uT = (unsigned)T, uD = (unsigned)D;
  void* args[] = {&px, &pbias, &po, &uT, &uD, &pp, &base};
  c.pending = true;
  return c.d.LaunchKernel(c.rope_dpos_(), (unsigned)rows, 1, 1, (unsigned)(D / 2),
                          1, 1, 0, c.stream, args, nullptr) == 0;
}

// One-thread *d_pos += 1 (tail of a captured forward; advances the counter).
inline bool incr_u32(void* d_pos) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_write_(d_pos);
  float* pp = context::off_(d_pos, 0);
  void* args[] = {&pp};
  c.pending = true;
  return c.d.LaunchKernel(c.incr_u32_(), 1, 1, 1, 1, 1, 1, 0, c.stream, args,
                          nullptr) == 0;
}

// KV append with write-row = *d_pos (else identical to kv_append(); f32 KV).
inline bool kv_append_dpos(void* Kc, void* Vc, void* k_new, void* v_new,
                           void* d_pos, int64_t kv_max, int64_t n_kv_heads,
                           int64_t D) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  c.device_read_(k_new);
  c.device_read_(v_new);
  c.device_read_(d_pos);
  c.device_write_(Kc);
  c.device_write_(Vc);
  float* pKc = context::off_(Kc, 0);
  float* pVc = context::off_(Vc, 0);
  float* pk = context::off_(k_new, 0);
  float* pv = context::off_(v_new, 0);
  float* pp = context::off_(d_pos, 0);
  unsigned kv_stride = (unsigned)(kv_max * D);
  void* args[] = {&pKc, &pVc, &pk, &pv, &pp, &kv_stride};
  c.pending = true;
  return c.d.LaunchKernel(c.kv_append_dpos_(), (unsigned)n_kv_heads, 1, 1,
                          (unsigned)D, 1, 1, 0, c.stream, args, nullptr) == 0;
}

// Decode attention with ctx = *d_pos + 1, single-block-per-head (S=1). f32 KV.
inline bool attn_decode_dpos(void* q, void* K, void* V, void* out,
                             int64_t n_q_heads, int64_t n_kv_heads, void* d_pos,
                             int64_t kv_max, int64_t D, float scale) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  c.device_read_(q);
  c.device_read_(K);
  c.device_read_(V);
  c.device_read_(d_pos);
  c.device_write_(out);
  float* pq = context::off_(q, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  float* po = context::off_(out, 0);
  float* pp = context::off_(d_pos, 0);
  unsigned kv_stride = (unsigned)(kv_max * D);
  unsigned group = (unsigned)(n_q_heads / n_kv_heads);
  void* args[] = {&pq, &pk, &pv, &po, &pp, &kv_stride, &group, &scale};
  c.pending = true;
  return c.d.LaunchKernel(c.attn_decode_dpos_(D), (unsigned)n_q_heads, 1, 1,
                          (unsigned)D, 1, 1, 0, c.stream, args, nullptr) == 0;
}

// M9 KV cache append: scatter one decode step's k,v (each [n_kv_heads,D] device
// buffers) into the cache (K,V each [n_kv_heads,kv_max,D]) at row `pos`.
inline bool kv_append(void* Kc, void* Vc, void* k_new, void* v_new, int64_t pos,
                      int64_t kv_max, int64_t n_kv_heads, int64_t D,
                      bool kv_bf16 = false) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  c.device_read_(k_new);
  c.device_read_(v_new);
  c.device_write_(Kc);
  c.device_write_(Vc);
  float* pKc = context::off_(Kc, 0);
  float* pVc = context::off_(Vc, 0);
  float* pk = context::off_(k_new, 0);
  float* pv = context::off_(v_new, 0);
  unsigned upos = static_cast<unsigned>(pos);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  void* args[] = {&pKc, &pVc, &pk, &pv, &upos, &kv_stride};
  c.pending = true;
  return c.d.LaunchKernel(c.kv_append_(kv_bf16), static_cast<unsigned>(n_kv_heads),
                          1, 1, static_cast<unsigned>(D), 1, 1, 0, c.stream, args,
                          nullptr) == 0;
}

// M9 prefill: bulk-copy a prompt's k,v (each [n_kv_heads,T,D] device buffers)
// into the cache (K,V each [n_kv_heads,kv_max,D]) rows [0,T). grid=(n_kv_heads,T).
inline bool kv_fill(void* Kc, void* Vc, void* K, void* V, int64_t T,
                    int64_t kv_max, int64_t n_kv_heads, int64_t D,
                    bool kv_bf16 = false) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  c.device_read_(K);
  c.device_read_(V);
  c.device_write_(Kc);
  c.device_write_(Vc);
  float* pKc = context::off_(Kc, 0);
  float* pVc = context::off_(Vc, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  unsigned uT = static_cast<unsigned>(T);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  void* args[] = {&pKc, &pVc, &pk, &pv, &uT, &kv_stride};
  c.pending = true;
  return c.d.LaunchKernel(c.kv_fill_(kv_bf16), static_cast<unsigned>(n_kv_heads),
                          static_cast<unsigned>(T), 1, static_cast<unsigned>(D),
                          1, 1, 0, c.stream, args, nullptr) == 0;
}

// M9 causal prefill attention: q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D]
// cache read over [0,T). Query p attends keys 0..p. GQA via group. D∈{64,128}.
// One block per (head, query pos); grid=(n_q_heads,T) (T <= 65535 gridDim.y).
inline bool attn_prefill(void* q, void* K, void* V, void* out, int64_t n_q_heads,
                         int64_t n_kv_heads, int64_t T, int64_t kv_max, int64_t D,
                         float scale, bool kv_bf16 = false) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  if (T <= 0 || T > 65535) return false;  // gridDim.y limit (chunk beyond)
  c.device_read_(q);
  c.device_read_(K);
  c.device_read_(V);
  c.device_write_(out);
  float* pq = context::off_(q, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  float* po = context::off_(out, 0);
  unsigned uT = static_cast<unsigned>(T);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  unsigned group = static_cast<unsigned>(n_q_heads / n_kv_heads);
  void* args[] = {&pq, &pk, &pv, &po, &uT, &kv_stride, &group, &scale};
  c.pending = true;
  return c.d.LaunchKernel(c.attn_prefill_(D, kv_bf16),
                          static_cast<unsigned>(n_q_heads), uT, 1,
                          static_cast<unsigned>(D), 1, 1, 0, c.stream, args,
                          nullptr) == 0;
}

// RoPE: rotate a contiguous [rows, D] buffer (rows = H*T). Row r's position is
// pos + (r % T); half-split (GPT-NeoX / HF-llama) convention. D must be even.
inline bool rope(void* x, void* out, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base, void* bias = nullptr) {
  auto& c = context::get();
  if (!c.ready || D <= 0 || (D & 1)) return false;
  c.device_read_(x);
  if (bias) c.device_read_(bias);
  c.device_write_(out);
  float* px = context::off_(x, 0);
  float* pbias = bias ? context::off_(bias, 0) : nullptr;
  float* po = context::off_(out, 0);
  unsigned uT = static_cast<unsigned>(T), uD = static_cast<unsigned>(D),
           upos = static_cast<unsigned>(pos);
  void* args[] = {&px, &pbias, &po, &uT, &uD, &upos, &base};
  c.pending = true;
  return c.d.LaunchKernel(c.rope_(), static_cast<unsigned>(rows), 1, 1,
                          static_cast<unsigned>(D / 2), 1, 1, 0, c.stream, args,
                          nullptr) == 0;
}

// Fused residual-add + RMSNorm: xout = x + delta; hout = rmsnorm(xout)*w.
// Writes both (xout is the next residual base, hout the normalized input).
inline bool rmsnorm_res(void* x, void* delta, void* w, void* xout, void* hout,
                        int64_t n, float eps) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  c.device_read_(x);
  c.device_read_(delta);
  c.device_read_(w);
  c.device_write_(xout);
  c.device_write_(hout);
  float* px = context::off_(x, 0);
  float* pd = context::off_(delta, 0);
  float* pw = context::off_(w, 0);
  float* pxo = context::off_(xout, 0);
  float* pho = context::off_(hout, 0);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&px, &pd, &pw, &pxo, &pho, &un, &eps};
  unsigned block = 256;
  c.pending = true;
  return c.d.LaunchKernel(c.add_rmsnorm_(), 1, 1, 1, block, 1, 1,
                          block * sizeof(float), c.stream, args, nullptr) == 0;
}

// GPU argmax over a length-n device vector (contiguous, offset 0). Reduces on
// device and D2H's only the 4-byte index — replaces the per-token 608KB logits
// copy + host scan that greedy decoding otherwise pays. `in` is a native
// device-buffer handle (e.g. an evaluated logits array's native()); stream
// ordering means the prior gemv that filled it need not be host-synced first.
// Returns the argmax index, tie-broken to the smallest index (matches the
// host `v[i] > v[bi]` loop) so greedy output stays bit-identical.
inline bool argmax(void* in, int64_t n, int64_t* out_idx) {
  auto& c = context::get();
  if (!c.ready || !in || n <= 0 || !out_idx) return false;
  c.device_read_(in);
  float* pin = context::off_(in, 0);
  CUdeviceptr res = c.argmax_res_();
  if (!res) return false;
  int* pres = reinterpret_cast<int*>(res);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pin, &pres, &un};
  unsigned block = 256;
  c.pending = true;
  if (c.d.LaunchKernel(c.argmax_(), 1, 1, 1, block, 1, 1,
                       block * (sizeof(float) + sizeof(int)), c.stream, args,
                       nullptr) != 0)
    return false;
  flush();  // the result index must be ready before the 4-byte D2H
  int h = 0;
  if (c.d.MemcpyDtoH(&h, res, sizeof(int)) != 0) return false;
  *out_idx = h;
  return true;
}

// Fused RMSNorm over one length-n row: out = x * 1/sqrt(mean(x^2)+eps) * w.
// x/w/out are native device handles (offset 0). One block; matches the array
// composition numerically (see tl_rmsnorm). In-place safe (out may alias x).
inline bool rmsnorm(void* x, void* w, void* out, int64_t n, float eps) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  c.device_read_(x);
  c.device_read_(w);
  c.device_write_(out);
  float* px = context::off_(x, 0);
  float* pw = context::off_(w, 0);
  float* po = context::off_(out, 0);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&px, &pw, &po, &un, &eps};
  unsigned block = 256;
  c.pending = true;
  return c.d.LaunchKernel(c.rmsnorm_(), 1, 1, 1, block, 1, 1,
                          block * sizeof(float), c.stream, args, nullptr) == 0;
}

// Fused SwiGLU: out = silu(gate) * up over n elements. Native handles, offset 0.
inline bool swiglu(void* gate, void* up, void* out, int64_t n) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  c.device_read_(gate);
  c.device_read_(up);
  c.device_write_(out);
  float* pg = context::off_(gate, 0);
  float* pu = context::off_(up, 0);
  float* po = context::off_(out, 0);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pg, &pu, &po, &un};
  return c.launch1d_(c.swiglu_(), un, args);
}

// Persistent, device-resident KV cache (roadmap M9, A-surface): K,V buffers
// [n_kv_heads, max_ctx, D] plus a running position. It lives OUTSIDE the lazy
// graph — decode is inference-only and stateful, which the immutable node model
// doesn't fit. append() writes one token and advances; attn() runs GQA-aware
// fused decode attention over the cached prefix [0,pos).
struct kv_cache {
  void* K = nullptr;  // native device-buffer handles (see alloc())
  void* V = nullptr;
  int64_t n_kv_heads = 0, max_ctx = 0, D = 0, pos = 0;
  // KV storage dtype (M9 bf16 KV cache). f32 = the exact baseline; bf16 halves
  // the K,V bytes the attention kernels stream every step (~2x the KV floor) at
  // a small precision cost. q/out/scratch stay f32. init() picks the width;
  // append/attn/prefill route to the matching kernel instantiation.
  bool kv_bf16 = false;

  bool init(int64_t kv_heads, int64_t maxctx, int64_t d, dtype kv_dt = dtype::f32) {
    n_kv_heads = kv_heads;
    max_ctx = maxctx;
    D = d;
    pos = 0;
    kv_bf16 = (kv_dt == dtype::bf16);
    int64_t w = kv_bf16 ? 2 : 4;  // bytes per K/V element
    K = alloc(n_kv_heads * max_ctx * D * w, nullptr);
    V = alloc(n_kv_heads * max_ctx * D * w, nullptr);
    return K && V;
  }
  // k_new/v_new: [n_kv_heads, D] device buffers (this step's projected k,v).
  bool append(void* k_new, void* v_new) {
    if (pos >= max_ctx) return false;
    if (!kv_append(K, V, k_new, v_new, pos, max_ctx, n_kv_heads, D, kv_bf16))
      return false;
    pos++;
    return true;
  }
  // q/out: [n_q_heads, D] device buffers. Attends over the cached prefix.
  bool attn(void* q, void* out, int64_t n_q_heads, float scale) {
    return attn_decode(q, K, V, out, n_q_heads, n_kv_heads, pos, max_ctx, D,
                       scale, kv_bf16);
  }
  // CUDA-graph-capture variants: write row / ctx come from the shared device
  // scalar d_pos (not the host `pos`), so a captured forward replays at the
  // advancing position. The host `pos` is NOT touched here — a tl_incr_u32 at the
  // captured forward's tail advances d_pos, and the orchestrator keeps host `pos`
  // in step out-of-band. f32 KV only (see the dpos launchers).
  bool append_dpos(void* k_new, void* v_new, void* d_pos) {
    return kv_append_dpos(K, V, k_new, v_new, d_pos, max_ctx, n_kv_heads, D);
  }
  bool attn_dpos(void* q, void* out, int64_t n_q_heads, void* d_pos, float scale) {
    return attn_decode_dpos(q, K, V, out, n_q_heads, n_kv_heads, d_pos, max_ctx,
                            D, scale);
  }
  // Prefill a T-token prompt: bulk-fill the cache from k_src/v_src
  // ([n_kv_heads,T,D]) and run causal attention (q/out [n_q_heads,T,D]). Leaves
  // pos=T so decode continues from there. Overwrites the cache from row 0.
  bool prefill(void* q, void* k_src, void* v_src, void* out, int64_t T,
               int64_t n_q_heads, float scale) {
    if (T <= 0 || T > max_ctx) return false;
    if (!kv_fill(K, V, k_src, v_src, T, max_ctx, n_kv_heads, D, kv_bf16))
      return false;
    pos = T;
    return attn_prefill(q, K, V, out, n_q_heads, n_kv_heads, T, max_ctx, D,
                        scale, kv_bf16);
  }
  void destroy() {
    if (K) release(K, 0, nullptr);
    if (V) release(V, 0, nullptr);
    K = V = nullptr;
  }
};

// C(m,n) = (A @ B) * scale + offset. lda/ldb row strides; trans reads a
// transposed view in place. One output per thread (16×16 blocks).
inline bool gemm(void* a, int64_t ao, int64_t lda, bool ta, void* b, int64_t bo,
                 int64_t ldb, bool tb, void* out, int64_t oo, int64_t m,
                 int64_t n, int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(b);
  c.device_write_(out);
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned um = (unsigned)m, un = (unsigned)n, uk = (unsigned)k;

  // Register-blocked fast path (tl_sgemm_rb): NN, contiguous (lda==k, ldb==n),
  // K%8==0 and N%4==0 for the 8-slab / float4 loads, and 16B-aligned bases. M
  // and N block edges are predicated in-kernel, so only the divisibility of the
  // *inner* load dims (K,N) and alignment gate eligibility. Everything else
  // (transpose, strided views, odd K/N, unaligned offset) falls to tl_sgemm.
  bool aligned = (ao % 16 == 0) && (bo % 16 == 0) && (oo % 16 == 0);
  if (!ta && !tb && lda == k && ldb == n && k % 8 == 0 && n % 4 == 0 &&
      aligned && m > 0 && n > 0 && k > 0) {
    unsigned gx = (un + 127) / 128, gy = (um + 127) / 128;

    if (CUfunction f = c.sgemm_rb_()) {
      // Split-K (ladder ②): when the 128² tiling underfills the 82 SMs (base
      // blocks < ~3 waves of 82×2 slots), partition K into S z-slices so S× more
      // blocks run concurrently. Split-K partitions K (not replicates it), so
      // A/B global traffic is unchanged — the only cost is C written S× via
      // atomicAdd into a pre-zeroed buffer, so it applies only when scale/offset
      // are identity (plain GEMM; the host gates on that). The census (RTX 3090)
      // shows S=2 is the robust optimum for the underfilled sizes (1024³
      // 0.63→0.75, 2048³ 0.74→0.76); S≥4 regresses as C-atomic traffic + short-K
      // per-block overhead overtake the occupancy gain, and 4096³ (6 waves) wants
      // S=1. So auto uses S=2 for base<512 with K≥512 (each half ≥256, enough to
      // amortize the smem pipeline). TL_SPLITK forces S for the census.
      unsigned S = 1, ksplit = uk;
      if (scale == 1.0f && offset == 0.0f) {
        long base = (long)gx * gy;
        long want = -1;
        if (const char* e = std::getenv("TL_SPLITK"))
          want = std::atol(e);
        else if (base < 512 && uk >= 512)
          want = 2;
        if (want > 1) {
          unsigned chunk = (uk + (unsigned)want - 1) / (unsigned)want;
          chunk = (chunk + 7u) & ~7u;  // multiple of TL_BK=8
          if (chunk == 0) chunk = 8;
          unsigned s = (uk + chunk - 1) / chunk;
          if (s > 1) { S = s; ksplit = chunk; }
        }
      }

      void* rb[] = {&pa, &pb, &po, &um, &un, &uk, &scale, &offset, &ksplit};
      if (S > 1) c.d.MemsetD8(reinterpret_cast<CUdeviceptr>(po), 0,
                              (size_t)m * n * 4);  // zero C for atomicAdd
      c.pending = true;
      return c.d.LaunchKernel(f, gx, gy, S, 256, 1, 1, 0, c.stream, rb,
                              nullptr) == 0;
    }
  }

  unsigned ula = (unsigned)lda, ulb = (unsigned)ldb;
  unsigned uta = ta ? 1u : 0u, utb = tb ? 1u : 0u;
  void* args[] = {&pa,  &pb,  &po,  &um,  &un,    &uk,
                  &ula, &ulb, &uta, &utb, &scale, &offset};
  CUfunction sg = c.fn_(kop::sgemm32);  // routed to tl_sgemm by kernel_name_
  if (!sg) return false;
  unsigned bx = 16, by = 16;
  unsigned gx = (un + bx - 1) / bx, gy = (um + by - 1) / by;
  if (gx == 0) gx = 1;
  if (gy == 0) gy = 1;
  c.pending = true;
  return c.d.LaunchKernel(sg, gx, gy, 1, bx, by, 1, 0, c.stream, args,
                          nullptr) == 0;
}

// Row op over the last axis: softmax writes rows×cols; row_sum/row_max write
// one value per row, affine epilogue. One block per row, 256 threads.
inline bool row_op(kop op, void* in, int64_t io, void* out, int64_t oo,
                   int64_t rows, int64_t cols, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(in);
  c.device_write_(out);
  float* pin = context::off_(in, io);
  float* po = context::off_(out, oo);
  unsigned ur = (unsigned)rows, uc = (unsigned)cols;
  void* args[] = {&pin, &po, &ur, &uc, &scale, &offset};
  CUfunction f = c.fn_(op);
  if (!f) return false;
  unsigned block = 256;
  c.pending = true;
  return c.d.LaunchKernel(f, ur ? ur : 1, 1, 1, block, 1, 1,
                          block * sizeof(float), c.stream, args, nullptr) == 0;
}

#else  // stubs (Apple, or a build without TENSORLIB_CUDA)

inline bool available() { return false; }
inline bool pending() { return false; }
inline void flush() {}
inline void* alloc(int64_t, float**) { return nullptr; }
inline void release(void*, int64_t, float*) {}
inline bool binary(kop, void*, int64_t, void*, int64_t, void*, int64_t, int64_t,
                   float, float) {
  return false;
}
inline bool binary_bcast(kop, void*, int64_t, int64_t, int64_t, void*, int64_t,
                         int64_t, int64_t, void*, int64_t, int64_t, int64_t,
                         float, float) {
  return false;
}
inline bool unary(kop, void*, int64_t, void*, int64_t, int64_t, float, float) {
  return false;
}
inline bool gemm(void*, int64_t, int64_t, bool, void*, int64_t, int64_t, bool,
                 void*, int64_t, int64_t, int64_t, int64_t, float, float) {
  return false;
}
inline bool row_op(kop, void*, int64_t, void*, int64_t, int64_t, int64_t, float,
                   float) {
  return false;
}
inline bool argmax(void*, int64_t, int64_t*) { return false; }
inline bool rmsnorm(void*, void*, void*, int64_t, float) { return false; }
inline bool rmsnorm_res(void*, void*, void*, void*, void*, int64_t, float) {
  return false;
}
inline bool swiglu(void*, void*, void*, int64_t) { return false; }
inline bool graph_available() { return false; }
inline bool capture_begin() { return false; }
inline void* capture_end() { return nullptr; }
inline bool graph_launch(void*) { return false; }
inline void graph_destroy(void*) {}
inline void upload(void*, const float*, int64_t) {}
inline void sync_to_host(void*, bool) {}

#endif

// CPU-read barrier: sync the GPU before any host read of a managed buffer.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace cuda

// The GPU-backend facade: array.h and storage.h dispatch through tl::gpu, which
// is Metal on Apple and CUDA elsewhere (the CUDA path only does real work when
// TENSORLIB_CUDA is set; otherwise cuda:: is stubs, matching metal:: off-Apple).
// One alias here is the single place the platform choice lives, so the eval
// seam carries no #ifdefs. Both namespaces expose the identical API and share
// tl::metal::kop, so the alias is a drop-in.
#if defined(TENSORLIB_CUDA) && !defined(__APPLE__)
namespace gpu = cuda;
#else
namespace gpu = metal;
#endif

inline bool gpu_available() { return gpu::available(); }

}  // namespace tl
