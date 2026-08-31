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
#include "shape.h"  // tl::contiguous_strides_into (pad/fold meta upload)
#include "types.h"  // tl::dtype (KV cache storage width)

namespace tl {
namespace cuda {

using kop = tl::metal::kop;
using cmp_op = tl::metal::cmp_op;

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
#include <type_traits>
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
    case kop::pow_: return "tl_pow";
    case kop::badd: return "tl_badd";
    case kop::bsub: return "tl_bsub";
    case kop::bmul: return "tl_bmul";
    case kop::bdiv: return "tl_bdiv";
    case kop::bpow: return "tl_bpow";
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

  // Lazy per-symbol kernel lookup: every named-kernel getter below is this one
  // line applied to its slot. The variant getters (D x bf16 etc.) just pick
  // which (slot, name) pair to hand it.
  CUfunction cached_(CUfunction& slot, const char* name) {
    if (!slot) d.ModuleGetFunction(&slot, mod, name);
    return slot;
  }

  // The register-blocked SGEMM fast path (tl_sgemm_rb), cached separately from
  // the kop table since it has no kop of its own.
  CUfunction sgemm_rb_fn = nullptr;
  CUfunction sgemm_rb_() { return cached_(sgemm_rb_fn, "tl_sgemm_rb"); }

  // M7 decode GEMV (f32 and bf16-weight variants), cached like sgemm_rb.
  CUfunction gemv_f32_fn = nullptr, gemv_bf16_fn = nullptr, gemv_bf16v8_fn = nullptr;
  CUfunction gemv_f32_() { return cached_(gemv_f32_fn, "tl_gemv_f32"); }
  CUfunction gemv_bf16_() { return cached_(gemv_bf16_fn, "tl_gemv_bf16"); }
  CUfunction gemv_bf16v8_() { return cached_(gemv_bf16v8_fn, "tl_gemv_bf16v8"); }
  CUfunction gemv_bf16_row_fn = nullptr;
  CUfunction gemv_bf16_row_() {
    return cached_(gemv_bf16_row_fn, "tl_gemv_bf16_row");
  }

  // M8 int4-weight decode GEMV.
  CUfunction gemv_q4_fn = nullptr;
  CUfunction gemv_q4_() { return cached_(gemv_q4_fn, "tl_gemv_q4"); }

  // M9 batched-prefill layout kernels (token-major <-> head-major).
  CUfunction split_heads_fn = nullptr, merge_heads_fn = nullptr;
  CUfunction split_heads_() { return cached_(split_heads_fn, "tl_split_heads"); }
  CUfunction merge_heads_() { return cached_(merge_heads_fn, "tl_merge_heads"); }

  // im2col's pad/fold, cached like split_heads/merge_heads.
  CUfunction pad_fn = nullptr, fold_fn = nullptr;
  CUfunction pad_() { return cached_(pad_fn, "tl_pad"); }
  CUfunction fold_() { return cached_(fold_fn, "tl_fold"); }

  // Embedding-table lookup (index_select/index_add) and pooling-style
  // one-hot scatter (scatter_to_axis), cached the same way.
  CUfunction index_select_fn = nullptr, index_add_fn = nullptr,
             scatter_axis_fn = nullptr;
  CUfunction index_select_() {
    return cached_(index_select_fn, "tl_index_select");
  }
  CUfunction index_add_() { return cached_(index_add_fn, "tl_index_add"); }
  CUfunction scatter_axis_() {
    return cached_(scatter_axis_fn, "tl_scatter_axis");
  }

  // N-D broadcast binary (any rank) and N-D broadcast ternary select
  // (Tensor.where's GPU dispatch) -- new capabilities, one kernel per op
  // like the rank-2 kop/fn_() vocabulary above, but not part of that
  // vocabulary itself (fn_() caches one name per kop; these need a second,
  // different name for the same op), so each gets its own cached
  // CUfunction, dispatched by a small switch on the existing kop value.
  CUfunction badd_nd_fn = nullptr, bsub_nd_fn = nullptr, bmul_nd_fn = nullptr,
             bdiv_nd_fn = nullptr, bpow_nd_fn = nullptr, where_nd_fn = nullptr;
  CUfunction bcast_nd_(kop op) {
    switch (op) {
      case kop::badd: return cached_(badd_nd_fn, "tl_badd_nd");
      case kop::bsub: return cached_(bsub_nd_fn, "tl_bsub_nd");
      case kop::bmul: return cached_(bmul_nd_fn, "tl_bmul_nd");
      case kop::bdiv: return cached_(bdiv_nd_fn, "tl_bdiv_nd");
      case kop::bpow: return cached_(bpow_nd_fn, "tl_bpow_nd");
      default: return nullptr;
    }
  }
  CUfunction where_nd_() { return cached_(where_nd_fn, "tl_where_nd"); }

  // array.h's sum_to (un-broadcast a gradient) -- gather-based, its own
  // meta-buffer layout (not the bcast_nd family's), so its own slot.
  CUfunction sum_to_fn = nullptr;
  CUfunction sum_to_() { return cached_(sum_to_fn, "tl_sum_to"); }

  // Comparisons (gt/lt/ge/le/eq/ne): ReLU/LeakyReLU/Clip's backward gate
  // and Tensor.gt/lt/... generally. Own vocabulary, not the kop table
  // (see metal.h's cmp_op comment for why).
  CUfunction gt_fn = nullptr, lt_fn = nullptr, ge_fn = nullptr,
             le_fn = nullptr, eq_fn = nullptr, ne_fn = nullptr;
  CUfunction compare_(cmp_op op) {
    switch (op) {
      case cmp_op::gt: return cached_(gt_fn, "tl_gt");
      case cmp_op::lt: return cached_(lt_fn, "tl_lt");
      case cmp_op::ge: return cached_(ge_fn, "tl_ge");
      case cmp_op::le: return cached_(le_fn, "tl_le");
      case cmp_op::eq: return cached_(eq_fn, "tl_eq");
      case cmp_op::ne: return cached_(ne_fn, "tl_ne");
      default: return nullptr;
    }
  }

  // M9 batched-prefill GEMM (bf16 [N,K] weights, the decode GEMV's own layout).
  CUfunction gemm_bf16_nt_fn = nullptr, gemm_bf16_nt_s_fn = nullptr,
             gemm_bf16_nt_sk_fn = nullptr;
  CUfunction gemm_bf16_nt_(bool big) {
    return big ? cached_(gemm_bf16_nt_fn, "tl_gemm_bf16_nt")
               : cached_(gemm_bf16_nt_s_fn, "tl_gemm_bf16_nt_s");
  }
  CUfunction gemm_bf16_nt_sk_() {
    return cached_(gemm_bf16_nt_sk_fn, "tl_gemm_bf16_nt_sk");
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
    return bf16 ? (D == 64 ? cached_(attn_decode_bf16_64_fn, "tl_attn_decode_bf16_64")
                           : cached_(attn_decode_bf16_fn, "tl_attn_decode_bf16"))
                : (D == 64 ? cached_(attn_decode_64_fn, "tl_attn_decode_f32_64")
                           : cached_(attn_decode_fn, "tl_attn_decode_f32"));
  }
  CUfunction attn_split_(int64_t D, bool bf16 = false) {
    return bf16 ? (D == 64 ? cached_(attn_split_bf16_64_fn, "tl_attn_decode_split_bf16_64")
                           : cached_(attn_split_bf16_fn, "tl_attn_decode_split_bf16"))
                : (D == 64 ? cached_(attn_split_64_fn, "tl_attn_decode_split_64")
                           : cached_(attn_split_fn, "tl_attn_decode_split"));
  }
  CUfunction attn_combine_() {  // head_dim implicit (blockDim.x) — one symbol
    return cached_(attn_combine_fn, "tl_attn_combine");
  }

  // M9 KV cache append (scatter one token's k,v into the persistent cache).
  CUfunction kv_append_fn = nullptr, kv_append_bf16_fn = nullptr;
  CUfunction kv_append_(bool bf16 = false) {
    return bf16 ? cached_(kv_append_bf16_fn, "tl_kv_append_bf16")
                : cached_(kv_append_fn, "tl_kv_append");
  }

  // RoPE (rotary position embedding) for q/k.
  CUfunction rope_fn = nullptr;
  CUfunction rope_() { return cached_(rope_fn, "tl_rope"); }

  // Device-pos variants (CUDA-graph capture): pos/ctx read from a device scalar
  // so one instantiated graph replays correctly as the decode position advances.
  CUfunction rope_dpos_fn = nullptr, incr_u32_fn = nullptr,
             kv_append_dpos_fn = nullptr, attn_split_dpos_fn = nullptr,
             attn_split_dpos_64_fn = nullptr;
  CUfunction rope_dpos_() { return cached_(rope_dpos_fn, "tl_rope_dpos"); }
  CUfunction incr_u32_() { return cached_(incr_u32_fn, "tl_incr_u32"); }
  CUfunction kv_append_dpos_() {
    return cached_(kv_append_dpos_fn, "tl_kv_append_dpos");
  }
  CUfunction attn_split_dpos_(int64_t D) {
    return D == 64 ? cached_(attn_split_dpos_64_fn, "tl_attn_decode_split_64_dpos")
                   : cached_(attn_split_dpos_fn, "tl_attn_decode_split_dpos");
  }

  // GPU argmax (greedy last-mile): kernel + a persistent 4-byte device result
  // buffer so the per-token result is a 4-byte D2H, not the 608KB logits copy.
  CUfunction argmax_fn = nullptr;
  CUfunction argmax_() { return cached_(argmax_fn, "tl_argmax"); }
  CUdeviceptr argmax_res = 0;
  CUdeviceptr argmax_res_() {
    if (!argmax_res && d.MemAlloc(&argmax_res, 16) != 0) argmax_res = 0;
    return argmax_res;
  }

  // Fused decode-step ops (imperative path): RMSNorm + SwiGLU.
  CUfunction rmsnorm_fn = nullptr, swiglu_fn = nullptr;
  CUfunction rmsnorm_() { return cached_(rmsnorm_fn, "tl_rmsnorm"); }
  CUfunction swiglu_() { return cached_(swiglu_fn, "tl_swiglu"); }
  CUfunction add_rmsnorm_fn = nullptr;
  CUfunction add_rmsnorm_() { return cached_(add_rmsnorm_fn, "tl_add_rmsnorm"); }

  // M9 prefill: bulk cache fill + causal prefill attention.
  CUfunction kv_fill_fn = nullptr, kv_fill_bf16_fn = nullptr;
  CUfunction kv_fill_(bool bf16 = false) {
    return bf16 ? cached_(kv_fill_bf16_fn, "tl_kv_fill_bf16")
                : cached_(kv_fill_fn, "tl_kv_fill");
  }
  CUfunction attn_prefill_tiled_fn = nullptr, attn_prefill_tiled_64_fn = nullptr,
             attn_prefill_tiled_bf16_fn = nullptr,
             attn_prefill_tiled_bf16_64_fn = nullptr;
  CUfunction attn_prefill_tiled_(int64_t D, bool bf16) {
    return bf16 ? (D == 64 ? cached_(attn_prefill_tiled_bf16_64_fn, "tl_attn_prefill_tiled_bf16_64")
                           : cached_(attn_prefill_tiled_bf16_fn, "tl_attn_prefill_tiled_bf16"))
                : (D == 64 ? cached_(attn_prefill_tiled_64_fn, "tl_attn_prefill_tiled_f32_64")
                           : cached_(attn_prefill_tiled_fn, "tl_attn_prefill_tiled_f32"));
  }

  // Grow-once device scratch, shared by every reusable buffer below: on the
  // first call past its current size, free (syncs; fine, this only happens
  // while growing) and reallocate. `buf`/`bytes` are the caller's own
  // persistent slot (attn_scratch/meta_scratch below), so each buffer still
  // grows independently.
  CUdeviceptr grow_scratch_(CUdeviceptr& buf, size_t& cur_bytes, size_t want) {
    if (want > cur_bytes) {
      if (buf) d.MemFree(buf);
      if (d.MemAlloc(&buf, want) != 0) {
        buf = 0;
        cur_bytes = 0;
        return 0;
      }
      cur_bytes = want;
    }
    return buf;
  }

  // Reusable device scratch for split-KV partials, reused across attention
  // calls (sequential on the null stream), freed at teardown.
  CUdeviceptr attn_scratch = 0;
  size_t attn_scratch_bytes = 0;
  CUdeviceptr attn_scratch_(size_t bytes) {
    return grow_scratch_(attn_scratch, attn_scratch_bytes, bytes);
  }

  // Reusable device scratch for pad_/fold_'s per-call shape/stride metadata
  // (small int64 arrays, one upload per call — too varied in size and too
  // short-lived to route through the tracked mirror allocator).
  CUdeviceptr meta_scratch = 0;
  size_t meta_scratch_bytes = 0;
  CUdeviceptr meta_scratch_(size_t bytes) {
    return grow_scratch_(meta_scratch, meta_scratch_bytes, bytes);
  }

  // char* pointer arithmetic to fold a byte offset into a managed pointer.
  static float* off_(void* base, int64_t byte_off) {
    return reinterpret_cast<float*>(static_cast<char*>(base) + byte_off);
  }

  // Grid / block extents for launch_. Brace-initialized with 1-3 values; the
  // omitted trailing dims are 1, so a 1-D launch is just {n}.
  struct dims {
    unsigned x = 1, y = 1, z = 1;
  };

  // Every kernel launch goes through here. cuLaunchKernel takes void** — an
  // array of POINTERS to the arguments — so each one must be an addressable
  // lvalue that outlives the call; taking them BY VALUE makes each a named
  // local of this frame, which is exactly that, and lets call sites pass
  // expressions instead of a ladder of one-use locals.
  //
  // The kernel's parameter list is NOT visible to the compiler — kernels are
  // looked up by name in the PTX at runtime, which is what keeps this backend
  // free of the CUDA runtime — so nothing can check the pack against the kernel
  // signature. What CAN be checked is the failure mode that actually bites:
  // every parameter these kernels declare is a pointer (.u64) or a 4-byte
  // scalar (.u32/.f32), so an int64_t/size_t/double — or a bare `nullptr`,
  // whose type is nullptr_t, not a pointer — would write 8 bytes into a 4-byte
  // slot and silently shift every argument after it. The static_assert below
  // rejects exactly that, which is why call sites may pass pointer expressions
  // inline but always name their scalars as `unsigned`/`float` locals.
  template <typename... Ts>
  bool launch_(CUfunction f, dims grid, dims block, unsigned smem, Ts... args) {
    static_assert(sizeof...(Ts) > 0, "a kernel with no arguments?");
    static_assert(((std::is_pointer_v<Ts> || sizeof(Ts) == 4) && ...),
                  "kernel arg must be a pointer or a 4-byte scalar: an 8-byte "
                  "one (int64_t/size_t/double/nullptr) shifts every arg after "
                  "it. Cast to unsigned/float at the call site.");
    if (!f) return false;
    void* argv[] = {&args...};
    pending = true;
    return d.LaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          smem, stream, argv, nullptr) == 0;
  }

  // The 1-D elementwise shape: 256-thread blocks covering n elements.
  template <typename... Ts>
  bool launch1d_(CUfunction f, unsigned n, Ts... args) {
    unsigned block = 256, grid = (n + block - 1) / block;
    if (grid == 0) grid = 1;
    return launch_(f, {grid}, {block}, 0, args...);
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

// Blocking H2D of the buffer's first `n` floats, marking it BOTH (device
// current). Used to pre-stage inputs (e.g. embedding rows) before a capture, so
// the captured region contains no blocking copy. `src` may be the mirror's own
// host buffer, in which case the memcpy is skipped — a caller that gathers
// straight into the mirror pays only the transfer, and a partly-filled buffer
// (a short prefill chunk) transfers only what it filled.
inline void upload(void* native, const float* src, int64_t n) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m) return;
  size_t bytes = std::min((size_t)n * sizeof(float), m->bytes);
  if (src != m->host) std::memcpy(m->host, src, bytes);
  c.d.MemcpyHtoD(m->dev, m->host, bytes);
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
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(b);
  c.device_write_(out);
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(c.fn_(op), un, pa, pb, po, un, scale, offset);
}

// Rank-2 broadcast binary (bias / row-vector / column-vector / scalar) --
// mirrors metal.h's own kernel; ars/acs/brs/bcs are element strides (0 on a
// broadcast axis), computed host-side by array.h's gpu_binary via the same
// broadcast_strides() the CPU oracle uses.
inline bool binary_bcast(kop op, void* a, int64_t ao, int64_t ars, int64_t acs,
                         void* b, int64_t bo, int64_t brs, int64_t bcs,
                         void* out, int64_t oo, int64_t m, int64_t n,
                         float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a);
  c.device_read_(b);
  c.device_write_(out);
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned um = static_cast<unsigned>(m), un = static_cast<unsigned>(n);
  return c.launch1d_(c.fn_(op), um * un, pa, pb, po, um, un,
                     static_cast<unsigned>(ars), static_cast<unsigned>(acs),
                     static_cast<unsigned>(brs), static_cast<unsigned>(bcs),
                     scale, offset);
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
  return c.launch1d_(c.fn_(op), un, pa, po, un, scale, offset);
}

// Rank cap shared with the kernel side (tensorlib_cuda.cu's
// TL_PAD_FOLD_MAX_RANK) — both the meta buffer layout and each kernel's
// on-stack index array assume it.
inline constexpr int kPadFoldMaxRank = 8;

// Zero an output buffer that a kernel will only partially write (pad's
// border, fold's atomicAdd accumulator) — device_write_ alone only flips the
// mirror's dirty bit, it does not copy the (already-zeroed) host side over.
// Async on the stream, like gemv_run_'s split-K zero above: ordered before
// the launch that follows it on the same stream, capture-safe.
inline void zero_device_(CUdeviceptr dst, int64_t n) {
  auto& c = context::get();
  size_t bytes = static_cast<size_t>(n) * 4;
  if (c.d.MemsetD8Async) c.d.MemsetD8Async(dst, 0, bytes, c.stream);
  else c.d.MemsetD8(dst, 0, bytes);
}

// Shared by pad()/fold() below: builds this call's [a_shape(rank),
// out_strides(stride_len)] meta buffer (out_strides derived from out_shape
// here — pad passes stride_len==rank, fold passes rank-1, since fold's own
// out has one fewer dim than `a`) and uploads it. Async on `c.stream`, right
// after zero_device_'s own async memset on the same stream — both ordered
// before the kernel launch that follows, so nothing here blocks the host
// waiting on the *device* the way the plain (non-Async) MemcpyHtoD would.
// `host_meta` is safe to let go out of scope on return despite being
// pageable, not pinned: a pageable HtoD async copy still stages the source
// into the driver's own DMA buffer synchronously, as part of this call — only
// the destination-side completion is deferred, which is exactly the
// blocking this function needs to avoid. `out_strides` is written out too,
// since pad's shift and fold's axis math both need it.
inline const long long* upload_pad_fold_meta_(context& c, const int64_t* a_shape,
                                              int rank, const int64_t* out_shape,
                                              int stride_len,
                                              int64_t* out_strides) {
  contiguous_strides_into(out_shape, stride_len, out_strides);
  size_t meta_bytes =
      (static_cast<size_t>(rank) + static_cast<size_t>(stride_len)) *
      sizeof(int64_t);
  CUdeviceptr meta = c.meta_scratch_(meta_bytes);
  if (!meta) return nullptr;
  std::vector<int64_t> host_meta(a_shape, a_shape + rank);
  host_meta.insert(host_meta.end(), out_strides, out_strides + stride_len);
  c.d.MemcpyHtoDAsync(meta, host_meta.data(), meta_bytes, c.stream);
  return reinterpret_cast<const long long*>(meta);
}

// Shared by binary_bcast_nd()/where_nd() below: uploads [out_shape(rank),
// strides_0(rank), strides_1(rank), ...] as one int64 buffer -- same async
// reasoning as upload_pad_fold_meta_ above, just a variable operand count
// instead of that one's fixed [a_shape, out_strides] pair.
inline const long long* upload_bcast_meta_(
    context& c, const int64_t* out_shape, int rank,
    std::initializer_list<const int64_t*> stride_arrays) {
  size_t parts = 1 + stride_arrays.size();
  size_t meta_bytes = parts * static_cast<size_t>(rank) * sizeof(int64_t);
  CUdeviceptr meta = c.meta_scratch_(meta_bytes);
  if (!meta) return nullptr;
  std::vector<int64_t> host_meta(out_shape, out_shape + rank);
  for (auto* s : stride_arrays) {
    host_meta.insert(host_meta.end(), s, s + rank);
  }
  c.d.MemcpyHtoDAsync(meta, host_meta.data(), meta_bytes, c.stream);
  return reinterpret_cast<const long long*>(meta);
}

// N-D broadcast binary: generalizes binary_bcast() above to any rank (a
// Transformer's [batch,seq,dim] LayerNorm broadcasts a [N,S,1] mean against
// a [N,S,D] input, rank 3 -- the rank-2 kernel only covers a Linear bias /
// BatchNorm-shaped [N,D] input). `a_strides`/`b_strides` are the broadcast
// strides (0 on a broadcast axis) array.h's gpu_binary_nd_ computes via the
// same broadcast_strides() the CPU oracle uses.
inline bool binary_bcast_nd(kop op, void* a_native, int64_t ao,
                            const int64_t* a_strides, void* b_native,
                            int64_t bo, const int64_t* b_strides,
                            void* out_native, int64_t oo,
                            const int64_t* out_shape, int rank, int64_t n,
                            float scale, float offset) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  CUfunction f = c.bcast_nd_(op);
  if (!f) return false;
  c.device_read_(a_native);
  c.device_read_(b_native);
  c.device_write_(out_native);
  const long long* pmeta =
      upload_bcast_meta_(c, out_shape, rank, {a_strides, b_strides});
  if (!pmeta) return false;
  float* pa = context::off_(a_native, ao);
  float* pb = context::off_(b_native, bo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(f, un, pa, pb, po, pmeta, rank, un, scale, offset);
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch, which existed
// on no backend before this (eval_one's where_ case always ran the CPU
// map_ternary). Same flat-index decode as binary_bcast_nd above, one more
// operand -- masking (attention/padding masks) is the concrete caller.
inline bool where_nd(void* cond_native, int64_t co, const int64_t* c_strides,
                     void* a_native, int64_t ao, const int64_t* a_strides,
                     void* b_native, int64_t bo, const int64_t* b_strides,
                     void* out_native, int64_t oo, const int64_t* out_shape,
                     int rank, int64_t n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(cond_native);
  c.device_read_(a_native);
  c.device_read_(b_native);
  c.device_write_(out_native);
  const long long* pmeta = upload_bcast_meta_(
      c, out_shape, rank, {c_strides, a_strides, b_strides});
  if (!pmeta) return false;
  float* pc = context::off_(cond_native, co);
  float* pa = context::off_(a_native, ao);
  float* pb = context::off_(b_native, bo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(c.where_nd_(), un, pc, pa, pb, po, pmeta, rank, un);
}

// Gather-based GPU dispatch for array.h's sum_to (un-broadcast a gradient).
// See tl_sum_to's own comment for why this needs no atomics, unlike
// index_add. `a_shape`/`a_strides` describe `a` (the caller has already
// checked it's contiguous); `acc` is array.h's own
// broadcast_strides(target, out_strides, a.shape()) -- 0 on every axis
// being summed over. `reduced_n` is the product of a_shape over exactly
// those zero-acc axes (1 if there are none).
inline bool sum_to(void* a_native, int64_t ao, const int64_t* a_shape,
                   const int64_t* a_strides, const int64_t* acc, int rank,
                   int64_t out_n, int64_t reduced_n, void* out_native,
                   int64_t oo) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  CUfunction f = c.sum_to_();
  if (!f) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);
  const long long* pmeta =
      upload_bcast_meta_(c, a_shape, rank, {a_strides, acc});
  if (!pmeta) return false;
  float* pa = context::off_(a_native, ao);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(out_n);
  unsigned ured = static_cast<unsigned>(reduced_n);
  return c.launch1d_(f, un, pa, po, pmeta, rank, un, ured);
}

// Elementwise comparison, same shape only (array.h's gpu_compare_ gates on
// that; ReLU/LeakyReLU/Clip's backward gate and the concrete Tensor.gt/...
// callers never need a broadcast form). Output is a F32 mask (1.0f/0.0f),
// matching the CPU oracle's own comparison ops.
inline bool compare(cmp_op op, void* a_native, int64_t ao, void* b_native,
                    int64_t bo, void* out_native, int64_t oo, int64_t n,
                    int64_t bstride) {
  auto& c = context::get();
  if (!c.ready) return false;
  CUfunction f = c.compare_(op);
  if (!f) return false;
  c.device_read_(a_native);
  c.device_read_(b_native);
  c.device_write_(out_native);
  float* pa = context::off_(a_native, ao);
  float* pb = context::off_(b_native, bo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  unsigned ubs = static_cast<unsigned>(bstride);
  return c.launch1d_(f, un, pa, pb, po, un, ubs);
}

// Places `a` (contiguous) into a zero buffer of out_shape (array.h's
// gpu_pad_ allocates `out` uninitialized via array::empty — this zeros the
// device copy directly, no host round trip), shifted by `before` along
// `axis`. No scale/offset — eval_one's shared epilogue applies those (see
// array.h's op_t::pad_ case).
inline bool pad(void* a_native, int64_t ao, void* out_native, int64_t oo,
                const int64_t* a_shape, const int64_t* out_shape, int rank,
                int axis, int64_t before, int64_t n, int64_t out_n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);
  zero_device_(reinterpret_cast<CUdeviceptr>(out_native), out_n);
  int64_t out_strides[kPadFoldMaxRank];
  const long long* pmeta =
      upload_pad_fold_meta_(c, a_shape, rank, out_shape, rank, out_strides);
  if (!pmeta) return false;
  float* pa = context::off_(a_native, ao);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  unsigned ushift = static_cast<unsigned>(before * out_strides[axis]);
  return c.launch1d_(c.pad_(), un, pa, po, pmeta, rank, ushift, un);
}

// unfold's inverse: scatter-add `a` (contiguous; its last dim is the sliding
// window) into a zero buffer of out_shape (zeroed the same way as pad()
// above) — every overlap accumulates via atomicAdd, so it must start at 0.
inline bool fold(void* a_native, int64_t ao, void* out_native, int64_t oo,
                 const int64_t* a_shape, const int64_t* out_shape, int rank,
                 int axis, int64_t step, int64_t n, int64_t out_n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);
  zero_device_(reinterpret_cast<CUdeviceptr>(out_native), out_n);
  int64_t out_strides[kPadFoldMaxRank];
  const long long* pmeta = upload_pad_fold_meta_(c, a_shape, rank, out_shape,
                                                 rank - 1, out_strides);
  if (!pmeta) return false;
  float* pa = context::off_(a_native, ao);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  unsigned ustep = static_cast<unsigned>(step);
  return c.launch1d_(c.fold_(), un, pa, po, pmeta, rank, axis, ustep, un);
}

// Row gather along axis 0: out[i] = a[indices[i]] (a, indices contiguous;
// indices float-valued, rounded on-device to match argmax's own
// convention). One thread per output element, no write conflicts — no
// zeroing needed (every element is written exactly once).
inline bool index_select(void* a_native, int64_t ao, void* idx_native,
                         int64_t idxo, void* out_native, int64_t oo,
                         int64_t row_size, int64_t k) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a_native);
  c.device_read_(idx_native);
  c.device_write_(out_native);
  float* pa = context::off_(a_native, ao);
  float* pidx = context::off_(idx_native, idxo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(k * row_size);
  unsigned urow = static_cast<unsigned>(row_size);
  return c.launch1d_(c.index_select_(), un, pa, pidx, po, urow, un);
}

// index_select's dual: scatter-add `values` into `out` by row index.
// Repeated indices really do collide (real write conflicts — the kernel
// uses atomicAdd), so `out` must start zeroed, same as pad/fold above.
inline bool index_add(void* idx_native, int64_t idxo, void* values_native,
                      int64_t vo, void* out_native, int64_t oo,
                      int64_t row_size, int64_t k, int64_t out_n) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(idx_native);
  c.device_read_(values_native);
  c.device_write_(out_native);
  zero_device_(reinterpret_cast<CUdeviceptr>(out_native), out_n);
  float* pidx = context::off_(idx_native, idxo);
  float* pv = context::off_(values_native, vo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(k * row_size);
  unsigned urow = static_cast<unsigned>(row_size);
  return c.launch1d_(c.index_add_(), un, pidx, pv, po, urow, un);
}

// One-hot scatter into a new trailing axis of size `size`: out[..., k] =
// values[...] where indices[...] == k, else 0. Every input position maps
// to a distinct output slot (the axis is brand new), so — unlike
// index_add above — there is no accumulation and no atomics; `out` still
// starts zeroed since untouched slots must read back as 0.
inline bool scatter_to_axis(void* idx_native, int64_t idxo,
                            void* values_native, int64_t vo, void* out_native,
                            int64_t oo, int64_t n, int64_t size) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(idx_native);
  c.device_read_(values_native);
  c.device_write_(out_native);
  zero_device_(reinterpret_cast<CUdeviceptr>(out_native), n * size);
  float* pidx = context::off_(idx_native, idxo);
  float* pv = context::off_(values_native, vo);
  float* po = context::off_(out_native, oo);
  unsigned un = static_cast<unsigned>(n);
  unsigned usize = static_cast<unsigned>(size);
  return c.launch1d_(c.scatter_axis_(), un, pidx, pv, po, usize, un);
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
  return c.launch_(f, {bx, gy}, {256}, 0, pa, pB, py, un, uk, ksplit);
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
  unsigned block = gemv_row_block_size(k);
  return c.launch_(c.gemv_bf16_row_(), {uN}, {block}, gemv_row_smem(block), pa,
                   pB, py, uN, uK);
}

// M9 batched-prefill GEMM: C(M,N) = A(M,K) @ W[N,K]^T, W the same row-major
// bf16 weight gemv_bf16_row consumes — so a batched prompt reuses the decode
// weights as-is. Requires K % 8 == 0. See tl_gemm_bf16_nt.
inline bool gemm_bf16_nt(void* a, void* B, void* out, int64_t m, int64_t n,
                         int64_t k) {
  auto& c = context::get();
  if (!c.ready || (k % 8) != 0 || m <= 0 || n <= 0) return false;
  c.device_read_(a);
  c.device_read_(B);  // B reinterpreted as __nv_bfloat16* [N][K] in-kernel
  c.device_write_(out);
  float* pa = context::off_(a, 0);
  float* pB = context::off_(B, 0);
  float* po = context::off_(out, 0);
  unsigned uM = (unsigned)m, uN = (unsigned)n, uK = (unsigned)k;
  // Tile choice = whichever actually fills the GPU. The 128 tile is the more
  // arithmetically efficient one, but a prefill chunk is only a few hundred
  // tokens, so a narrow projection (N=896, M=512) yields 28 blocks against 82
  // SMs; the 64 tile turns that into 112 at half the FMA-per-shared-float. Take
  // the big tile only when it already keeps ~2 blocks per SM busy. It needs
  // K % 8 == 0; the small tile's deeper slab needs K % 16 == 0.
  bool big = ((n + 127) / 128) * ((m + 127) / 128) >= 164 || (k % 16) != 0;
  unsigned t = big ? 128u : 64u;
  unsigned gx = (unsigned)((n + t - 1) / t), gy = (unsigned)((m + t - 1) / t);
  unsigned blocks = gx * gy;

  // Even on the small tile a narrow projection is only ~1.4 waves of the 82 SMs,
  // so the tail runs half-empty; slicing K over gridDim.z fills it. Two bounds
  // decide how far: keep each slice long enough to amortize its own staging
  // (>= 448 K-elements), and stop once the grid is comfortably several waves
  // (~8 blocks/SM). Measured at M=512: wd 506 -> 312 us, wo 92 -> 75, while a
  // grid that already fills (gateup, 1216 blocks) correctly declines to split.
  unsigned z = 1;
  if (blocks < 128 && c.d.MemsetD8Async) {
    unsigned by_k = (unsigned)(k / 448);
    unsigned by_fill = (656 + blocks - 1) / blocks;
    z = by_k < by_fill ? by_k : by_fill;
    if (z < 1) z = 1;
  }
  if (z > 1) {
    constexpr unsigned BK = 16;  // the 64 tile's K slab
    unsigned ksplit = ((uK + z - 1) / z + BK - 1) / BK * BK;
    z = (uK + ksplit - 1) / ksplit;  // recompute after rounding
    // atomicAdd combine needs a zeroed C; async on the stream, so it is ordered
    // before the launch without a host sync (and stays capturable).
    c.d.MemsetD8Async(reinterpret_cast<CUdeviceptr>(po), 0,
                      (size_t)m * n * sizeof(float), c.stream);
    return c.launch_(c.gemm_bf16_nt_sk_(), {gx, gy, z}, {256}, 0, pa, pB, po,
                     uM, uN, uK, ksplit);
  }
  return c.launch_(c.gemm_bf16_nt_(big), {gx, gy}, {256}, 0, pa, pB, po, uM, uN,
                   uK);
}

// ---- M9 batched-prefill layout moves between token-major projections and
// head-major attention.

// [T, ld] token-major -> [H, T, D] head-major, adding an optional [H*D] bias.
// `off` picks a column block of a fused projection output (q|k|v from one GEMM).
inline bool split_heads(void* src, void* bias, void* dst, int64_t T, int64_t ld,
                        int64_t off, int64_t H, int64_t D) {
  auto& c = context::get();
  if (!c.ready || T <= 0 || H <= 0 || D <= 0) return false;
  c.device_read_(src);
  if (bias) c.device_read_(bias);
  c.device_write_(dst);
  float* ps = context::off_(src, 0);
  float* pb = bias ? context::off_(bias, 0) : nullptr;
  float* pd = context::off_(dst, 0);
  unsigned uT = (unsigned)T, uld = (unsigned)ld, uoff = (unsigned)off,
           uD = (unsigned)D;
  return c.launch_(c.split_heads_(), {(unsigned)H, uT}, {uD}, 0, ps, pb, pd, uT,
                   uld, uoff, uD);
}

// [H, T, D] head-major -> [T, H*D] token-major (inverse of split_heads).
inline bool merge_heads(void* src, void* dst, int64_t T, int64_t H, int64_t D) {
  auto& c = context::get();
  if (!c.ready || T <= 0 || H <= 0 || D <= 0) return false;
  c.device_read_(src);
  c.device_write_(dst);
  float* ps = context::off_(src, 0);
  float* pd = context::off_(dst, 0);
  unsigned uT = (unsigned)T, uH = (unsigned)H, uD = (unsigned)D;
  return c.launch_(c.merge_heads_(), {uH, uT}, {uD}, 0, ps, pd, uT, uH, uD);
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
  unsigned block = gemv_row_block_size(K);
  return c.launch_(c.gemv_q4_(), {uN}, {block}, gemv_row_smem(block), pa, pq,
                   ps, py, uN, uK, uG);
}

// Split-KV split-count heuristic: how many ctx-splits make grid = heads×S fill
// the SMs (heads alone is ~32 blocks « 82 SMs; target ~4 blocks/SM, and each
// split needs >=128 keys to amortize its fixed cost). Shared by attn_decode
// (evaluated at the live ctx) and attn_decode_dpos (evaluated at max_ctx, so
// the CUDA-graph grid is pos-independent).
inline unsigned attn_split_count(unsigned n_heads, int64_t ctx) {
  const long target = 328;  // ~4 * 82 SMs
  if (n_heads == 0 || (long)n_heads >= target || ctx < 256) return 1;
  unsigned want = static_cast<unsigned>((target + n_heads - 1) / n_heads);
  unsigned max_s = static_cast<unsigned>(ctx / 128);  // >=128 keys/split
  if (want > max_s) want = max_s;
  return want > 1 ? want : 1;
}

// Launch shape of the tiled prefill attention — the ONE place it lives. It is
// ABI, not a tuning knob: the kernel derives its query tile from the same
// numbers, and a mismatch would silently compute the wrong rows rather than fail
// to launch, so tl_attn_prefill_tiled_* names this as its device twin (cf.
// attn_dpos_chunk and attn_split_chunk below). A block is always 128 threads
// (16 row slots x 8 lanes); how many queries each thread carries is what varies,
// and D=128 carries fewer only because its Q tile would otherwise blow the 48 KB
// static shared limit. See the kernel for the measurements behind both.
inline constexpr unsigned attn_tile_threads = 128;
inline constexpr unsigned attn_tile_queries(int64_t D) { return D == 64 ? 64 : 32; }

// Keys per split for the split-KV kernels: ceil(ctx / attn_split_count), rounded
// up to a multiple of 4 warps. The device-side attn_dpos_chunk in
// tensorlib_cuda.cu is the twin of attn_split_count + this rounding and MUST
// stay in lockstep — the host/dpos bit-identity rests on it. Guarded by the
// attn64 ctest's host-vs-dpos bit-equality sweep.
inline unsigned attn_split_chunk(unsigned n_heads, int64_t ctx) {
  unsigned S = attn_split_count(n_heads, ctx);
  unsigned chunk = (static_cast<unsigned>(ctx) + S - 1) / S;
  chunk = (chunk + 3u) & ~3u;
  return chunk ? chunk : 4u;
}

// Split-KV partials scratch: ONE buffer laid out pm[H*S] | pl[H*S] | pacc[H*S*D]
// (hs = H*S). bytes/the carve constructor/combine() keep that layout contract in
// a single place — attn_decode's split branch and attn_decode_dpos both bake
// these pointers into launches (the dpos ones into a captured graph).
struct attn_partials {
  float *pm, *pl, *pacc;
  attn_partials(float* base, size_t hs) : pm(base), pl(pm + hs), pacc(pl + hs) {}
  static size_t bytes(size_t hs, int64_t D) {
    return (hs * 2 + hs * static_cast<size_t>(D)) * sizeof(float);
  }
  // Merge the S per-head partials into out [H, D] (tl_attn_combine).
  bool combine(context& c, float* out, unsigned uh, unsigned uD, unsigned S) {
    return c.launch_(c.attn_combine_(), {uh}, {uD}, 0, pm, pl, pacc, out, S);
  }
};

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

  // Split ctx over gridDim.y so grid = heads×S fills the SMs (see
  // attn_split_count; chunk below is rounded to a multiple of 4 warps).
  unsigned S = attn_split_count(uh, ctx);
  unsigned uD = static_cast<unsigned>(D);
  if (S == 1) {
    return c.launch_(c.attn_decode_(D, kv_bf16), {uh}, {uD}, 0, pq, pk, pv, po,
                     uctx, kv_stride, group, scale);
  }

  unsigned chunk = attn_split_chunk(uh, ctx);
  S = (uctx + chunk - 1) / chunk;  // recompute after rounding
  size_t hs = (size_t)uh * S;
  CUdeviceptr scr = c.attn_scratch_(attn_partials::bytes(hs, D));
  if (!scr) return false;
  attn_partials p(reinterpret_cast<float*>(scr), hs);
  if (!c.launch_(c.attn_split_(D, kv_bf16), {uh, S}, {uD}, 0, pq, pk, pv, p.pm,
                 p.pl, p.pacc, uctx, kv_stride, group, chunk, scale)) {
    return false;
  }
  return p.combine(c, po, uh, uD, S);
}

// ---- CUDA-graph-capture device-pos launchers (A-min). Each mirrors its
// by-value sibling but sources pos/ctx from a device scalar `d_pos` (a 4-byte
// device buffer the caller owns) and launches on c.stream so the op is captured.
// Grids are pos-independent throughout: attn_decode_dpos gets its split grid
// from the cache CAPACITY (max_ctx) and lets the kernel bound the work by pos.

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
  return c.launch_(c.rope_dpos_(), {(unsigned)rows}, {(unsigned)(D / 2)}, 0, px,
                   pbias, po, uT, uD, pp, base);
}

// One-thread *d_pos += 1 (tail of a captured forward; advances the counter).
inline bool incr_u32(void* d_pos) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_write_(d_pos);
  float* pp = context::off_(d_pos, 0);
  return c.launch_(c.incr_u32_(), {1}, {1}, 0, pp);
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
  return c.launch_(c.kv_append_dpos_(), {(unsigned)n_kv_heads}, {(unsigned)D},
                   0, pKc, pVc, pk, pv, pp, kv_stride);
}

// Decode attention with ctx = *d_pos + 1, split-KV on a capacity-static grid:
// gridDim.y = S_max = attn_split_count(H, max_ctx), a constant for the cache's
// lifetime — so the launch is capturable — while each block derives its live
// split bounds from *d_pos (attn_dpos_chunk, the device twin of
// attn_split_count + attn_split_chunk). Splits past the live count write
// neutral partials that combine as exact zeros, so the output is bit-identical
// to the host attn_decode at every pos, and the captured path stays
// split-KV-flat at long ctx (the S=1 pin it replaces was linear in ctx).
// `partials` is a caller-OWNED device buffer of attn_partials::bytes(H*S_max, D)
// — like d_pos, it is graph-lifetime state (a captured graph bakes its address
// in), so it must not be a shared growable scratch; kv_cache owns one per
// cache. f32 KV.
inline bool attn_decode_dpos(void* q, void* K, void* V, void* out,
                             int64_t n_q_heads, int64_t n_kv_heads, void* d_pos,
                             int64_t kv_max, int64_t D, float scale,
                             void* partials) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64) || !partials) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  c.device_read_(q);
  c.device_read_(K);
  c.device_read_(V);
  c.device_read_(d_pos);
  c.device_write_(out);
  c.device_write_(partials);
  float* pq = context::off_(q, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  float* po = context::off_(out, 0);
  float* pp = context::off_(d_pos, 0);
  unsigned uh = (unsigned)n_q_heads, uD = (unsigned)D;
  unsigned kv_stride = (unsigned)(kv_max * D);
  unsigned group = (unsigned)(n_q_heads / n_kv_heads);
  unsigned S = attn_split_count(uh, kv_max);
  attn_partials p(context::off_(partials, 0), (size_t)uh * S);
  if (!c.launch_(c.attn_split_dpos_(D), {uh, S}, {uD}, 0, pq, pk, pv, p.pm,
                 p.pl, p.pacc, pp, kv_stride, group, scale)) {
    return false;
  }
  return p.combine(c, po, uh, uD, S);
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
  return c.launch_(c.kv_append_(kv_bf16), {static_cast<unsigned>(n_kv_heads)},
                   {static_cast<unsigned>(D)}, 0, pKc, pVc, pk, pv, upos,
                   kv_stride);
}

// M9 prefill: bulk-copy a block of k,v (each [n_kv_heads,T,D] device buffers)
// into the cache (K,V each [n_kv_heads,kv_max,D]) rows [pos0, pos0+T), so a long
// prompt can be filled in chunks. grid=(n_kv_heads,T).
inline bool kv_fill(void* Kc, void* Vc, void* K, void* V, int64_t T,
                    int64_t kv_max, int64_t n_kv_heads, int64_t D,
                    bool kv_bf16 = false, int64_t pos0 = 0) {
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
  unsigned up0 = static_cast<unsigned>(pos0);
  return c.launch_(c.kv_fill_(kv_bf16), {static_cast<unsigned>(n_kv_heads), uT},
                   {static_cast<unsigned>(D)}, 0, pKc, pVc, pk, pv, uT,
                   kv_stride, up0);
}

// M9 causal prefill attention: q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D]
// cache read over [0,pos0+T). Query p is at absolute position pos0+p and attends
// keys 0..pos0+p, so a long prompt can be run in chunks (and a later turn
// appended to a live cache). GQA via group. D∈{64,128}.
// One block per (head, query tile); grid = (n_q_heads, ceil(T/tile)).
inline bool attn_prefill(void* q, void* K, void* V, void* out, int64_t n_q_heads,
                         int64_t n_kv_heads, int64_t T, int64_t kv_max, int64_t D,
                         float scale, bool kv_bf16 = false, int64_t pos0 = 0) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  if (T <= 0 || T > 65535) return false;  // one call is one prompt chunk
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
  unsigned up0 = static_cast<unsigned>(pos0);

  // A block takes a tile of queries and streams K/V through shared memory, so
  // each score is a register dot product instead of a per-key warp-shuffle
  // reduction. Same online softmax, same causal rule. See
  // tl_attn_prefill_tiled_*.
  const unsigned tile = attn_tile_queries(D);
  return c.launch_(c.attn_prefill_tiled_(D, kv_bf16),
                   {static_cast<unsigned>(n_q_heads), (uT + tile - 1) / tile},
                   {attn_tile_threads}, 0, pq, pk, pv, po, uT, kv_stride, group,
                   scale, up0);
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
  return c.launch_(c.rope_(), {static_cast<unsigned>(rows)},
                   {static_cast<unsigned>(D / 2)}, 0, px, pbias, po, uT, uD,
                   upos, base);
}

// ---- Row-wise fused RMSNorm / SwiGLU. `rows` defaults to 1 (a decode step); a
// batched prefill chunk passes its token count. ONE kernel serves both, which is
// what keeps the two paths from drifting numerically. Buffers are [rows, n]
// contiguous; the weight is [n], shared by every row.

// xout = x + delta; hout = rmsnorm(xout) * w, per row. xout may alias x. Folds a
// layer's residual add into the following norm (the o-proj->norm and
// mlp->next-input-norm seams), writing both the residual sum (the next residual
// base) and its normalized form.
inline bool rmsnorm_res(void* x, void* delta, void* w, void* xout, void* hout,
                        int64_t n, float eps, int64_t rows = 1) {
  auto& c = context::get();
  if (!c.ready || n <= 0 || rows <= 0) return false;
  c.device_read_(x);
  c.device_read_(delta);
  c.device_read_(w);
  c.device_write_(xout);
  c.device_write_(hout);
  float* pa = context::off_(x, 0);
  float* pb = context::off_(delta, 0);
  float* pw = context::off_(w, 0);
  float* px = context::off_(xout, 0);
  float* ph = context::off_(hout, 0);
  unsigned un = (unsigned)n, block = 256;
  return c.launch_(c.add_rmsnorm_(), {(unsigned)rows}, {block},
                   block * sizeof(float), pa, pb, pw, px, ph, un, eps);
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
  unsigned block = 256;
  if (!c.launch_(c.argmax_(), {1}, {block},
                 block * (sizeof(float) + sizeof(int)), pin, pres, un)) {
    return false;
  }
  flush();  // the result index must be ready before the 4-byte D2H
  int h = 0;
  if (c.d.MemcpyDtoH(&h, res, sizeof(int)) != 0) return false;
  *out_idx = h;
  return true;
}

// Fused RMSNorm over one length-n row: out = x * 1/sqrt(mean(x^2)+eps) * w.
// x/w/out are native device handles (offset 0). One block; matches the array
// composition numerically (see tl_rmsnorm). In-place safe (out may alias x).
inline bool rmsnorm(void* x, void* w, void* out, int64_t n, float eps,
                    int64_t rows = 1) {
  auto& c = context::get();
  if (!c.ready || n <= 0 || rows <= 0) return false;
  c.device_read_(x);
  c.device_read_(w);
  c.device_write_(out);
  float* px = context::off_(x, 0);
  float* pw = context::off_(w, 0);
  float* po = context::off_(out, 0);
  unsigned un = (unsigned)n, block = 256;
  return c.launch_(c.rmsnorm_(), {(unsigned)rows}, {block},
                   block * sizeof(float), px, pw, po, un, eps);
}

// out[rows, ff] = silu(gate) * up, read out of the FUSED gate|up buffer
// gu[rows, 2*ff] that both paths already produce (up is gate + ff in each row).
inline bool swiglu(void* gu, void* out, int64_t ff, int64_t rows = 1) {
  auto& c = context::get();
  if (!c.ready || ff <= 0 || rows <= 0) return false;
  c.device_read_(gu);
  c.device_write_(out);
  float* pg = context::off_(gu, 0);
  float* po = context::off_(out, 0);
  unsigned uff = (unsigned)ff, block = 256;
  unsigned gx = (uff + block - 1) / block;
  return c.launch_(c.swiglu_(), {gx, (unsigned)rows}, {block}, 0, pg, po, uff);
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
  // The split partials for attn_dpos are graph-lifetime state (a captured graph
  // bakes their address in), so each cache OWNS its buffer rather than sharing
  // the growable attn_scratch_ — an unrelated bigger attn call can then never
  // free memory a live graph still points at. Sized once from the capacity
  // (attn_split_count at max_ctx) on the first — warm, pre-capture — call.
  void* dpos_partials = nullptr;
  bool attn_dpos(void* q, void* out, int64_t n_q_heads, void* d_pos, float scale) {
    if (!dpos_partials) {
      unsigned S = attn_split_count((unsigned)n_q_heads, max_ctx);
      size_t hs = (size_t)n_q_heads * S;
      dpos_partials = alloc((int64_t)attn_partials::bytes(hs, D), nullptr);
    }
    return attn_decode_dpos(q, K, V, out, n_q_heads, n_kv_heads, d_pos, max_ctx,
                            D, scale, dpos_partials);
  }
  // Prefill T tokens: bulk-fill the cache from k_src/v_src ([n_kv_heads,T,D])
  // and run causal attention (q/out [n_q_heads,T,D]). The block is APPENDED at
  // the current pos and leaves pos advanced by T, so a long prompt can be run in
  // chunks and a later turn can extend a live cache; queries attend everything
  // already cached before them. Starting from pos=0 (a fresh cache) this is the
  // original whole-prompt-from-row-0 fill.
  bool prefill(void* q, void* k_src, void* v_src, void* out, int64_t T,
               int64_t n_q_heads, float scale) {
    if (T <= 0 || pos + T > max_ctx) return false;
    if (!kv_fill(K, V, k_src, v_src, T, max_ctx, n_kv_heads, D, kv_bf16, pos))
      return false;
    int64_t p0 = pos;
    pos += T;
    return attn_prefill(q, K, V, out, n_q_heads, n_kv_heads, T, max_ctx, D,
                        scale, kv_bf16, p0);
  }
  void destroy() {
    if (K) release(K, 0, nullptr);
    if (V) release(V, 0, nullptr);
    if (dpos_partials) release(dpos_partials, 0, nullptr);
    K = V = dpos_partials = nullptr;
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

      if (S > 1) c.d.MemsetD8(reinterpret_cast<CUdeviceptr>(po), 0,
                              (size_t)m * n * 4);  // zero C for atomicAdd
      return c.launch_(f, {gx, gy, S}, {256}, 0, pa, pb, po, um, un, uk, scale,
                       offset, ksplit);
    }
  }

  unsigned ula = (unsigned)lda, ulb = (unsigned)ldb;
  unsigned uta = ta ? 1u : 0u, utb = tb ? 1u : 0u;
  unsigned bx = 16, by = 16;
  unsigned gx = (un + bx - 1) / bx, gy = (um + by - 1) / by;
  if (gx == 0) gx = 1;
  if (gy == 0) gy = 1;
  // kop::sgemm32 is routed to tl_sgemm by kernel_name_.
  return c.launch_(c.fn_(kop::sgemm32), {gx, gy}, {bx, by}, 0, pa, pb, po, um,
                   un, uk, ula, ulb, uta, utb, scale, offset);
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
  unsigned block = 256;
  return c.launch_(c.fn_(op), {ur ? ur : 1}, {block}, block * sizeof(float),
                   pin, po, ur, uc, scale, offset);
}

#else  // stubs (Apple, or a build without TENSORLIB_CUDA)

// Only the gpu:: facade surface is stubbed — what array.h/storage.h dispatch
// through, so a consumer can name tl::cuda:: unconditionally and get "no device
// here". The LLM-path entry points (gemv/attention/kv_cache/graph capture/the
// fused decode ops) deliberately have NO stubs: they are reachable only from
// code that is itself CUDA-gated (bench/cuda/*, which needs kv_cache and the
// capture types anyway), so a stub could never be linked — it would just be an
// unreachable `return false` claiming an API that isn't really there.

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
inline bool pad(void*, int64_t, void*, int64_t, const int64_t*,
                const int64_t*, int, int, int64_t, int64_t, int64_t) {
  return false;
}
inline bool fold(void*, int64_t, void*, int64_t, const int64_t*,
                 const int64_t*, int, int, int64_t, int64_t, int64_t) {
  return false;
}
inline bool index_select(void*, int64_t, void*, int64_t, void*, int64_t,
                         int64_t, int64_t) {
  return false;
}
inline bool index_add(void*, int64_t, void*, int64_t, void*, int64_t, int64_t,
                      int64_t, int64_t) {
  return false;
}
inline bool scatter_to_axis(void*, int64_t, void*, int64_t, void*, int64_t,
                            int64_t, int64_t) {
  return false;
}
inline bool binary_bcast_nd(kop, void*, int64_t, const int64_t*, void*,
                            int64_t, const int64_t*, void*, int64_t,
                            const int64_t*, int, int64_t, float, float) {
  return false;
}
inline bool where_nd(void*, int64_t, const int64_t*, void*, int64_t,
                     const int64_t*, void*, int64_t, const int64_t*, void*,
                     int64_t, const int64_t*, int, int64_t) {
  return false;
}
inline bool sum_to(void*, int64_t, const int64_t*, const int64_t*,
                   const int64_t*, int, int64_t, int64_t, void*, int64_t) {
  return false;
}
inline bool compare(cmp_op, void*, int64_t, void*, int64_t, void*, int64_t,
                    int64_t, int64_t) {
  return false;
}
inline void sync_to_host(void*, bool) {}

#endif

// CPU-read barrier: sync the GPU before any host read of a managed buffer.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace cuda

}  // namespace tl
