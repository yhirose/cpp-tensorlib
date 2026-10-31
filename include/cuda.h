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
// The whole header is gated on TENSORLIB_CUDA && !__APPLE__ (Apple uses Metal):
// elsewhere it declares nothing, and gpu.h selects another backend (or
// gpu_null.h). What it provides is the device core gpu.h describes, so the
// eval seam is backend-agnostic and carries no platform #ifdefs.

#include <cstdint>

#include "gpu_abi.h"  // the op vocabulary and the launch contract
#include "profile.h"  // tl::profile (per-launch attribution and timing)
#include "shape.h"  // tl::contiguous_strides_into (pad/fold meta upload)
#include "types.h"  // tl::dtype (KV cache storage width)

#if defined(TENSORLIB_CUDA) && !defined(__APPLE__)

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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tl {
namespace cuda {

using kop = gpu::kop;
using cmp_op = gpu::cmp_op;
using unary_ext_op = gpu::unary_ext_op;
using scalar_op = gpu::scalar_op;

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
struct CUevent_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUstream = CUstream_st*;
using CUgraph = CUgraph_st*;
using CUgraphExec = CUgraphExec_st*;
using CUevent = CUevent_st*;

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
  CUresult (*MemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream) = nullptr;
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
  // Events, for tl::profile's per-launch timing. Optional like the graph
  // set: timing_ok() gates them, and a driver without them still launches.
  CUresult (*EventCreate)(CUevent*, unsigned) = nullptr;
  CUresult (*EventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*EventSynchronize)(CUevent) = nullptr;
  CUresult (*EventElapsedTime)(float*, CUevent, CUevent) = nullptr;
  CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
  // A kernel's resources and the module's kernels, for check_cuda's look over all of
  // them; optional (drivers from CUDA 12.4), nothing launches through them.
  CUresult (*FuncGetAttribute)(int*, int, CUfunction) = nullptr;
  CUresult (*FuncGetName)(const char**, CUfunction) = nullptr;
  CUresult (*ModuleGetFunctionCount)(unsigned*, CUmodule) = nullptr;
  CUresult (*ModuleEnumerateFunctions)(CUfunction*, unsigned, CUmodule) = nullptr;

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
  bool timing_ok() const {
    return EventCreate && EventRecord && EventSynchronize && EventElapsedTime;
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

// The alignment each kernel needs of its pointer arguments, as TL_ALIGN declares
// it beside the kernel (tensorlib_cuda.cu): one byte count per leading pointer
// parameter. nvcc emits each declaration into the PTX as
//   .global .align 1 .b8 <kernel>_ptr_align[n] = {16, 4, 8};
// so the table is read from the module's own text, once, with no driver call.
// A kernel without one takes its pointers on their element's width.
using align_table = std::unordered_map<std::string, std::vector<unsigned char>>;
inline align_table parse_align_decls_(const char* ptx) {
  static constexpr char tag[] = "_ptr_align[";
  align_table out;
  for (const char* p = std::strstr(ptx, tag); p; p = std::strstr(p + 1, tag)) {
    const char* name = p;
    while (name > ptx && name[-1] != ' ' && name[-1] != '\t') --name;
    const char* open = std::strpbrk(p, "{\n");
    if (!open || *open != '{') continue;  // a use, not the definition
    std::vector<unsigned char> bytes;
    for (char* q = const_cast<char*>(open + 1); *q && *q != '}';) {
      bytes.push_back(static_cast<unsigned char>(std::strtoul(q, &q, 10)));
      while (*q == ',' || *q == ' ') ++q;
    }
    out.emplace(std::string(name, p), std::move(bytes));
  }
  return out;
}
inline const align_table& align_decls_() {
  static const align_table table = parse_align_decls_(ptx_source_());
  return table;
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
    case kop::tanh_: return "tl_tanh";
    case kop::sin_: return "tl_sin";
    case kop::cos_: return "tl_cos";
    case kop::softmax: return "tl_softmax";
    case kop::row_sum: return "tl_row_sum";
    case kop::row_max: return "tl_row_max";
    case kop::gt_: return "tl_gt";
    case kop::lt_: return "tl_lt";
    case kop::ge_: return "tl_ge";
    case kop::le_: return "tl_le";
    case kop::eq_: return "tl_eq";
    case kop::ne_: return "tl_ne";
    case kop::clamp_: return "tl_clamp";
    case kop::pow_s_: return "tl_pow_s";
    case kop::gt_s_: return "tl_gt_s";
    case kop::lt_s_: return "tl_lt_s";
    case kop::ge_s_: return "tl_ge_s";
    case kop::le_s_: return "tl_le_s";
    case kop::eq_s_: return "tl_eq_s";
    case kop::ne_s_: return "tl_ne_s";
    case kop::layer_norm_: return "tl_layer_norm";
    case kop::index_select: return "tl_index_select";
    case kop::gather_axis_: return "tl_gather_axis";
    case kop::row_logsumexp_: return "tl_row_logsumexp";
    case kop::xent_bwd_: return "tl_xent_bwd";
    case kop::adam_step_: return "tl_adam_step";
    case kop::rmsnorm_: return "tl_rmsnorm";
    case kop::add_rmsnorm_: return "tl_add_rmsnorm";
    case kop::swiglu_: return "tl_swiglu";
    case kop::ssm_conv_: return "tl_ssm_conv";
    case kop::gated_delta_: return "tl_gated_delta";
    case kop::gemv_bf16_row_: return "tl_gemv_bf16_row";
    case kop::gemv_q4_: return "tl_gemv_q4";
    case kop::kv_append_: return "tl_kv_append";
    case kop::kv_append_f16_: return "tl_kv_append_f16";
    case kop::kv_fill_: return "tl_kv_fill";
    case kop::kv_fill_f16_: return "tl_kv_fill_f16";
    case kop::merge_heads_: return "tl_merge_heads";
    case kop::split_heads_: return "tl_split_heads";
    default: return nullptr;  // no kernel here: dispatch declines
  }
}

// A launch places one block per SM up to this many blocks, two past it.
constexpr long kWaveSingles = 80;

// The f32 gemm's tiles (kernels/tensorlib_cuda.cu): a bm² block tile with a
// bk-deep K slab (K must be a multiple), the kernel per operand layout (NN,
// NT, TN, TT), and how it takes the wave plan (sgemm_wave_chunk_): from how
// many blocks the plan beats the fill heuristics, and how shallow (along K) it
// may cut layers to fill the SMs one block each (0: it does not). The 64²
// takes the plan always; the 128² once it fills the SMs one block each (80
// measured a win over the heuristics, 256×2048×512 +8%; 64 a loss to the 64²
// tile, 256×512×2048 -9%). `id` is the kernel cache slot (context::sgemm_).
struct sgemm_tile {
  int id;
  unsigned bm, bk;
  long wave_min_blocks;
  unsigned single_min_k;
  const char* base;  // kernel name stem; context::sgemm_ appends bias and layout
};
constexpr sgemm_tile sgemm_tiles[] = {
    {0, 64, 16, 0, 96, "tl_sgemm_cp64"},
    {1, 128, 8, kWaveSingles, 0, "tl_sgemm_cp128"},
};

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

  // Each loaded function's record, read back on every launch (which only has
  // the handle): the name it was looked up by, for tl::profile, and the
  // pointer alignment its kernel declares (align_decls_). Then the launches
  // bracketed by events whose elapsed time has not been read back yet. Events
  // are recycled.
  struct kernel_info {
    std::string name;
    std::vector<unsigned char> align;
  };
  std::unordered_map<CUfunction, kernel_info> kernels;
  struct timed_launch {
    profile::row* row;
    CUevent begin, end;
  };
  std::vector<timed_launch> timed;
  std::vector<CUevent> spare_events;

  // Host/device mirror per allocation and its size-keyed free list (shared
  // shape with webgpu.h; `gpu::mirror_table`, gpu_abi.h). Keyed by the device
  // pointer, reinterpreted as the `native` handle stored in storage. Views
  // sharing a storage share the key, so one state serves every view. When to
  // copy is gpu::residency's decision; the copies are made here. Released
  // buffers are recycled, not cuMemFree'd — repeated large alloc/free
  // otherwise fragments the driver allocator (decode benches, training that
  // churns activations); they persist until the (leaked) context tears down.
  using mirror = gpu::mirror_table<CUdeviceptr>::entry;
  gpu::mirror_table<CUdeviceptr> mt;

  mirror* mirror_(void* native) { return mt.find(native); }

  // A kernel is about to touch this buffer as `a`: bring the host copy up if
  // residency says so. Async on the stream like the meta uploads (a blocking
  // copy would wait out every kernel already queued and stall the pipeline
  // mid-graph); the driver stages a pageable source during the call.
  void before_kernel_(void* native, gpu::access a) {
    mirror* m = mirror_(native);
    if (!m) return;
    if (capturing) capture_touches.emplace_back(native, a);
    if (m->live.before_kernel(a)) upload_(*m);
  }
  // A replayed graph runs none of the launchers that keep the mirrors in step,
  // so the capture records every touch its launches made, in order, and each
  // replay makes them again before it runs (graph_launch): a buffer the graph
  // writes is device-current after it, as it is after the launches themselves.
  bool capturing = false;
  std::vector<std::pair<void*, gpu::access>> capture_touches;
  std::unordered_map<CUgraphExec, std::vector<std::pair<void*, gpu::access>>> graph_touches;
  // The three accesses by name, for the ops that state residency themselves.
  void device_read_(void* native) { before_kernel_(native, gpu::access::in); }
  void device_write_(void* native) { before_kernel_(native, gpu::access::out); }
  void device_rmw_(void* native) { before_kernel_(native, gpu::access::inout); }
  // The H2D: async on the stream when the driver has it. Profiled as a
  // transfer either way (the blocking form with its wait).
  void upload_(const mirror& m) {
    if (d.MemcpyHtoDAsync) {
      d.MemcpyHtoDAsync(m.dev, m.host, m.bytes, stream);
      profile::detail::transfer("h2d", m.bytes, 0.0);
      return;
    }
    profile::detail::blocked timing{"h2d", m.bytes};
    d.MemcpyHtoD(m.dev, m.host, m.bytes);
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
    d.MemcpyDtoDAsync = (CUresult(*)(CUdeviceptr, CUdeviceptr, size_t,
                                     CUstream))S("cuMemcpyDtoDAsync_v2");
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
    d.EventCreate = (CUresult(*)(CUevent*, unsigned))S("cuEventCreate");
    d.EventRecord = (CUresult(*)(CUevent, CUstream))S("cuEventRecord");
    d.EventSynchronize = (CUresult(*)(CUevent))S("cuEventSynchronize");
    d.EventElapsedTime =
        (CUresult(*)(float*, CUevent, CUevent))S("cuEventElapsedTime");
    d.FuncSetAttribute = (CUresult(*)(CUfunction, int, int))S("cuFuncSetAttribute");
    d.FuncGetAttribute = (CUresult(*)(int*, int, CUfunction))S("cuFuncGetAttribute");
    d.FuncGetName = (CUresult(*)(const char**, CUfunction))S("cuFuncGetName");
    d.ModuleGetFunctionCount = (CUresult(*)(unsigned*, CUmodule))S("cuModuleGetFunctionCount");
    d.ModuleEnumerateFunctions =
        (CUresult(*)(CUfunction*, unsigned, CUmodule))S("cuModuleEnumerateFunctions");
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
    profile::detail::drain_hook = [] { context::get().resolve_timed_(); };
  }

  // Every function handle comes out of one of the three lookups below, so
  // this is the one place a name is known; launch_ reads it back.
  CUfunction load_(const char* name) {
    CUfunction f = nullptr;
    d.ModuleGetFunction(&f, mod, name);
    if (f) {
      const auto it = align_decls_().find(name);
      kernels.emplace(f, kernel_info{name, it == align_decls_().end()
                                               ? std::vector<unsigned char>{}
                                               : it->second});
    }
    return f;
  }

  // Whether these arguments of `f` fit its kernel: each leading pointer on the
  // alignment the kernel declares (TL_ALIGN), none declared past those
  // pointers (kernels take their pointers first; dispatch_ relies on it too).
  // launch_ and dispatch_ refuse a launch that does not fit, so a misaligned
  // view is declined, never misread; a launcher choosing between kernels asks
  // this of its candidate first.
  bool fits_ptrs_(CUfunction f, const void* const* ptrs, size_t n) const {
    auto it = kernels.find(f);
    if (it == kernels.end()) return true;
    const std::vector<unsigned char>& a = it->second.align;
    if (a.size() > n) return false;
    for (size_t i = 0; i < a.size(); i++)
      if (reinterpret_cast<uintptr_t>(ptrs[i]) % a[i]) return false;
    return true;
  }
  template <typename... Ts>
  bool fits_(CUfunction f, Ts... args) const {
    constexpr bool is_ptr[] = {std::is_pointer_v<Ts>...};
    size_t n = 0;
    while (n < sizeof...(Ts) && is_ptr[n]) n++;
    const void* p[] = {ptr_or_null_(args)...};
    return fits_ptrs_(f, p, n);
  }
  template <typename T>
  static const void* ptr_or_null_(T v) {
    if constexpr (std::is_pointer_v<T>) return static_cast<const void*>(v);
    else return nullptr;
  }
  std::string_view name_(CUfunction f) const {
    auto it = kernels.find(f);
    return it == kernels.end() ? std::string_view("?") : it->second.name;
  }

  CUfunction fn_(kop op) {
    int key = static_cast<int>(op);
    auto it = fns.find(key);
    if (it != fns.end()) return it->second;
    const char* name = kernel_name_(op);
    CUfunction f = name ? load_(name) : nullptr;
    fns[key] = f;
    return f;
  }

  // Lazy per-symbol kernel lookup: every named-kernel getter below is this one
  // line applied to its slot. The variant getters (D x bf16 etc.) just pick
  // which (slot, name) pair to hand it.
  CUfunction cached_(CUfunction& slot, const char* name) {
    if (!slot) slot = load_(name);
    return slot;
  }

  // ---- tl::profile: a launch bracketed by two events on the stream, read
  // back (never mid-pipeline) once the device is known idle ----
  CUevent event_() {
    if (!spare_events.empty()) {
      CUevent e = spare_events.back();
      spare_events.pop_back();
      return e;
    }
    CUevent e = nullptr;
    d.EventCreate(&e, 0 /*CU_EVENT_DEFAULT: timing on*/);
    return e;
  }
  void resolve_timed_() {
    if (timed.empty()) return;
    // One stream completes in order: once the last end event is in, all
    // are, and a flush that already synchronized returns from this at once.
    d.EventSynchronize(timed.back().end);
    for (const timed_launch& t : timed) {
      float ms = 0.0f;
      if (d.EventElapsedTime(&ms, t.begin, t.end) == 0) {
        profile::detail::device_time(t.row, ms * 1000.0);
      }
      spare_events.push_back(t.begin);
      spare_events.push_back(t.end);
    }
    timed.clear();
  }

  // The f32 SGEMM fast path, one kernel per tile (sgemm_tiles) and operand
  // layout, cached separately from the kop table since it has no kop of its
  // own.
  CUfunction sgemm_fn[16] = {};
  CUfunction sgemm_(bool ta, bool tb, bool bias, const sgemm_tile& t) {
    int layout = (bias ? 4 : 0) | (ta ? 2 : 0) | (tb ? 1 : 0);
    CUfunction& slot = sgemm_fn[t.id * 8 | layout];
    if (!slot) {  // the name TL_SGEMM_LAYOUTS gives it
      static constexpr const char* suffix[4] = {"", "_nt", "_tn", "_tt"};
      char name[64];
      std::snprintf(name, sizeof(name), "%s%s%s", t.base, bias ? "_bias" : "",
                    suffix[layout & 3]);
      slot = load_(name);
    }
    return slot;
  }

  // M7 decode GEMV (f32 and bf16-weight variants), cached like sgemm_.
  CUfunction gemv_f32_fn = nullptr, gemv_bf16_fn = nullptr, gemv_bf16v8_fn = nullptr;
  CUfunction gemv_f32_() { return cached_(gemv_f32_fn, "tl_gemv_f32"); }
  CUfunction gemv_bf16_() { return cached_(gemv_bf16_fn, "tl_gemv_bf16"); }
  CUfunction gemv_bf16v8_() { return cached_(gemv_bf16v8_fn, "tl_gemv_bf16v8"); }

  // im2col's pad/fold, cached like split_heads/merge_heads.
  CUfunction pad_fn = nullptr, fold_fn = nullptr;
  CUfunction pad_() { return cached_(pad_fn, "tl_pad"); }
  CUfunction fold_() { return cached_(fold_fn, "tl_fold"); }

  // The f32 gemm's general fallback (the tiled fast path above, sgemm_(), has
  // its own names).
  CUfunction sgemm_fallback_fn = nullptr;
  CUfunction sgemm_fallback_() { return cached_(sgemm_fallback_fn, "tl_sgemm"); }

  // Embedding-table lookup (index_select/index_add) and pooling-style
  // one-hot scatter (scatter_to_axis), cached the same way.
  CUfunction index_add_fn = nullptr, scatter_axis_fn = nullptr;
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
  // clone()'s strided arm (same meta layout as the bcast_nd family, one
  // operand), so its own slot next to where_nd's.
  CUfunction copy_nd_fn = nullptr;
  CUfunction copy_nd_() { return cached_(copy_nd_fn, "tl_copy_nd"); }

  // array.h's sum_to (un-broadcast a gradient) -- gather-based, its own
  // meta-buffer layout (not the bcast_nd family's), so its own slot.
  CUfunction sum_to_fn = nullptr;
  CUfunction sum_to_() { return cached_(sum_to_fn, "tl_sum_to"); }
  // The same reduction with a block per output, for a deep reduced range.
  CUfunction sum_to_blocked_fn = nullptr;
  CUfunction sum_to_blocked_() {
    return cached_(sum_to_blocked_fn, "tl_sum_to_blocked");
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
  CUfunction gemv_q4k_fn = nullptr, gemv_q6k_fn = nullptr,
             gemv_q80_fn = nullptr, quant_act_fn = nullptr, gemv_q4k_q8_fn = nullptr;
  CUfunction rmsnorm_rope_fn[3] = {};  // D 64, 128, 256
  CUfunction gated_rmsnorm_fn[2] = {};
  CUfunction rmsnorm_h16_fn = nullptr, add_rmsnorm_h16_fn = nullptr, swiglu_h16_fn = nullptr,
             merge_heads_h16_fn = nullptr, cvt_h16_fn = nullptr;  // the f16 activation's producers
  CUfunction conv_tok_fn = nullptr, conv_state_fn = nullptr;  // a prefill's conv
  CUfunction conv_tok_() { return cached_(conv_tok_fn, "tl_ssm_conv_tokens"); }
  CUfunction conv_state_() { return cached_(conv_state_fn, "tl_ssm_conv_state"); }
  CUfunction gdc_prep_fn = nullptr, gdc_scan_fn = nullptr;  // the chunked delta rule
  CUfunction gdc_prep_() { return cached_(gdc_prep_fn, "tl_gated_delta_chunk_prep"); }
  // The scan's tile is dynamic shared memory (past the static 48 KB), raised on
  // first lookup. R and the footprint repeat TL_GDC_SCAN_R and the kernel's
  // carve-up in tensorlib_cuda.cu: Buf [C][128], red [128/R][C][32], Dl [C][32],
  // Pc [C][C+1], gc and ed [C], in floats.
  static constexpr unsigned gdc_scan_r = 8;
  static constexpr unsigned gdc_scan_threads = 128 / gdc_scan_r * 32;
  static constexpr unsigned gdc_scan_smem =
      (32 * 128 + 128 / gdc_scan_r * 32 * 32 + 32 * 32 + 32 * 33 + 2 * 32) * 4;
  CUfunction gdc_scan_() {
    const bool fresh = !gdc_scan_fn;
    CUfunction f = cached_(gdc_scan_fn, "tl_gated_delta_chunk_scan");
    if (fresh && f && d.FuncSetAttribute)
      d.FuncSetAttribute(f, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, static_cast<int>(gdc_scan_smem));
    return f;
  }
  // The prefill's tensor-core GEMM: tile (L 128x128, M 64x128, S 64x64) x sums
  // (f32 / f16), each a kernel of its own. The tiles stage through dynamic
  // shared memory (tl_gemm_nt_tc), so a function is raised to its footprint
  // when it is first looked up.
  enum class tc_tile { l, m, s, xl };  // xl: the f16 weight's long-K tile only
  // The form is the operands and sums: bf16 operands with f32 sums, f16 with f16
  // sums, and f16 sums reading the activation already f16 (no rounding stage).
  enum class tc_form { f32, f16, f16h };
  // The weight it reads: bf16 [N][K], or the rows of block-quantized blocks (Q4_K, Q6_K);
  // the quantized ones have the f32 and f16h forms only.
  enum class tc_weight { bf16, q4k, q6k, q80, f16w };
  CUfunction gemm_tc_fn[5][3][4] = {};  // [weight][form][tile]
  CUfunction gemm_tc_(tc_form f, tc_tile t, tc_weight w = tc_weight::bf16) {
    static const char* const forms[3] = {"f32", "f16", "f16h"};
    static const char* const weights[5] = {"", "_q4k", "_q6k", "_q80", "_f16w"};
    static const char* const tiles[4] = {"l", "m", "s", "xl"};
    if (t == tc_tile::xl && (w != tc_weight::f16w || f != tc_form::f16h)) return nullptr;
    if (w != tc_weight::bf16 && f == tc_form::f16) return nullptr;
    if (w == tc_weight::f16w && f != tc_form::f16h) return nullptr;
    CUfunction& slot = gemm_tc_fn[static_cast<int>(w)][static_cast<int>(f)][static_cast<int>(t)];
    if (!slot) {
      const std::string name = std::string("tl_gemm_tc_") + forms[static_cast<int>(f)] +
                               weights[static_cast<int>(w)] + "_" + tiles[static_cast<int>(t)];
      slot = load_(name.c_str());
      if (slot && d.FuncSetAttribute)
        d.FuncSetAttribute(slot, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */,
                           static_cast<int>(gemm_tc_shape(t).smem()));
    }
    return slot;
  }
  // The tiles' shapes: tl_gemm_tc_*_{l,m,s} in tensorlib_cuda.cu spell the same
  // numbers (BM, BN, BK, threads, stages); change them together. smem is the stages of
  // (BM+BN) rows of (BK+8) 2-byte elements.
  struct tc_shape {
    unsigned rows, cols, bk, threads, stages;
    constexpr unsigned smem() const { return (rows + cols) * (bk + 8) * 2 * stages; }
  };
  static constexpr tc_shape gemm_tc_shape(tc_tile t) {
    return t == tc_tile::l ? tc_shape{128, 128, 32, 128, 2}
         : t == tc_tile::m ? tc_shape{64, 128, 64, 256, 2}
         : t == tc_tile::s ? tc_shape{64, 64, 64, 128, 2}
                           : tc_shape{256, 128, 32, 512, 3};
  }

  // The attention kernels, by kind, head width (64, 128, 256) and the KV cache's
  // element type (f32, f16): tl_<kind>_{f32,f16}_<D>, looked up on first use. A
  // width or type a kind has no kernel for comes back null (its launcher says no
  // first). The tensor-core prefill at 256 stages Q in dynamic shared memory too:
  // (64 queries + 2 buffers of K and V of 16 keys) rows of D + 8 halves, past the
  // static 48 KB, so its function is raised to that on lookup (the kernel carves
  // the same layout).
  enum class attn_k { decode, split, split_dpos, tiled, tc };
  CUfunction attn_fns_[5][3][2] = {};
  static constexpr unsigned attn_tc_smem(int64_t D) {
    return D == 256 ? (64 + 4 * 16) * (256 + 8) * 2 : 0;
  }
  CUfunction attn_(attn_k k, int64_t D, bool f16) {
    static const char* const kind[5] = {"tl_attn_decode_", "tl_attn_decode_split_",
                                        "tl_attn_decode_split_dpos_", "tl_attn_prefill_tiled_",
                                        "tl_attn_prefill_tc_"};
    const int di = D == 64 ? 0 : D == 128 ? 1 : D == 256 ? 2 : -1;
    if (di < 0) return nullptr;
    CUfunction& slot = attn_fns_[static_cast<int>(k)][di][f16];
    if (!slot) {
      char name[64];
      std::snprintf(name, sizeof name, "%s%s_%d", kind[static_cast<int>(k)], f16 ? "f16" : "f32", (int)D);
      slot = load_(name);
      if (slot && k == attn_k::tc && D == 256 && d.FuncSetAttribute)
        d.FuncSetAttribute(slot, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, static_cast<int>(attn_tc_smem(D)));
    }
    return slot;
  }
  CUfunction attn_combine_fn = nullptr;
  CUfunction attn_combine_() {  // head_dim implicit (blockDim.x) — one symbol
    return cached_(attn_combine_fn, "tl_attn_combine");
  }

  // RoPE (rotary position embedding) for q/k.
  CUfunction rope_fn = nullptr;
  CUfunction rope_() { return cached_(rope_fn, "tl_rope"); }

  // Device-pos variants (CUDA-graph capture): pos/ctx read from a device scalar
  // so one instantiated graph replays correctly as the decode position advances.
  CUfunction rope_dpos_fn = nullptr, incr_u32_fn = nullptr, kv_append_dpos_fn[2] = {};
  CUfunction rope_dpos_() { return cached_(rope_dpos_fn, "tl_rope_dpos"); }
  CUfunction incr_u32_() { return cached_(incr_u32_fn, "tl_incr_u32"); }
  CUfunction kv_append_dpos_(bool f16) {
    return cached_(kv_append_dpos_fn[f16], f16 ? "tl_kv_append_dpos_f16" : "tl_kv_append_dpos");
  }

  // GPU argmax (greedy last-mile, tl_argmax): a persistent device buffer, so the
  // per-token result is a 4-byte D2H, not the 608KB logits copy. The index at 0,
  // the count of finished blocks at 4 (zero between calls: the last block puts it
  // back), then a (value, index) pair a block, argmax_blocks of each.
  static constexpr unsigned argmax_blocks = 128, argmax_pairs_at = 16;
  CUdeviceptr argmax_res = 0;
  CUfunction argmax_fn = nullptr;
  CUdeviceptr argmax_res_() {
    if (!argmax_res) {
      const size_t bytes = argmax_pairs_at + 8 * argmax_blocks;
      if (d.MemAlloc(&argmax_res, bytes) != 0 || d.MemsetD8(argmax_res, 0, bytes) != 0) argmax_res = 0;
    }
    return argmax_res;
  }

  // The fused layer norm's pullback: three kernels.
  CUfunction layer_norm_bwd_dx_fn = nullptr;
  CUfunction layer_norm_bwd_dx_() {
    return cached_(layer_norm_bwd_dx_fn, "tl_layer_norm_bwd_dx");
  }
  CUfunction layer_norm_bwd_gb_fn = nullptr;
  CUfunction layer_norm_bwd_gb_() {
    return cached_(layer_norm_bwd_gb_fn, "tl_layer_norm_bwd_gb");
  }
  CUfunction layer_norm_bwd_gb_fold_fn = nullptr;
  CUfunction layer_norm_bwd_gb_fold_() {
    return cached_(layer_norm_bwd_gb_fold_fn, "tl_layer_norm_bwd_gb_fold");
  }

  // A fused gemm bias laid under split partials (gemm_batched).
  CUfunction fill_rows_fn = nullptr;
  CUfunction fill_rows_() { return cached_(fill_rows_fn, "tl_fill_rows"); }

  // The two halves of that kernel's pullback. Training only, so f32 only.
  CUfunction attn_bwd_dq_fn = nullptr, attn_bwd_dq_64_fn = nullptr;
  CUfunction attn_bwd_dq_(int64_t D) {
    return D == 64 ? cached_(attn_bwd_dq_64_fn, "tl_attn_bwd_dq_f32_64")
                   : cached_(attn_bwd_dq_fn, "tl_attn_bwd_dq_f32");
  }
  CUfunction attn_bwd_dkv_fn = nullptr, attn_bwd_dkv_64_fn = nullptr;
  CUfunction attn_bwd_dkv_(int64_t D) {
    return D == 64 ? cached_(attn_bwd_dkv_64_fn, "tl_attn_bwd_dkv_f32_64")
                   : cached_(attn_bwd_dkv_fn, "tl_attn_bwd_dkv_f32");
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
  // The chunked delta rule's per-chunk products (q, k, W, U, P, g), reused
  // across layers and calls like the partials above.
  CUdeviceptr delta_scratch = 0;
  size_t delta_scratch_bytes = 0;
  CUdeviceptr delta_scratch_(size_t bytes) {
    return grow_scratch_(delta_scratch, delta_scratch_bytes, bytes);
  }

  // The f16 copy of one slab of a block-quantized weight (the prefill dequantizes a
  // projection ahead of its GEMM, a slab of rows at a time).
  CUdeviceptr dequant_scratch = 0;
  size_t dequant_scratch_bytes = 0;
  CUdeviceptr dequant_scratch_(size_t bytes) {
    return grow_scratch_(dequant_scratch, dequant_scratch_bytes, bytes);
  }
  CUfunction dequant_fn[3] = {};

  // The int8 copy of the prefill's activation for the MMQ GEMM (tl_quant_act_mmq):
  // the scales first, then the int8 rows.
  CUdeviceptr mmq_scratch = 0;
  size_t mmq_scratch_bytes = 0;
  CUdeviceptr mmq_scratch_(size_t bytes) {
    return grow_scratch_(mmq_scratch, mmq_scratch_bytes, bytes);
  }
  CUfunction mmq_quant_fn = nullptr, mmq_fn[2][2] = {}, mmq_reduce_fn = nullptr;
  // The partial products of a split-K MMQ GEMM (tl_mmq_reduce sums them into C).
  CUdeviceptr mmq_part_scratch = 0;
  size_t mmq_part_scratch_bytes = 0;
  CUdeviceptr mmq_part_scratch_(size_t bytes) {
    return grow_scratch_(mmq_part_scratch, mmq_part_scratch_bytes, bytes);
  }
  // The MMQ GEMM (tl_gemm_mmq_q4k, _q80): 128 x 128 tiles of 256 threads, 84 KB of dynamic
  // shared memory (TL_MMQ_SMEM in tensorlib_cuda.cu, change them together), raised on first
  // lookup.
  static constexpr unsigned mmq_tile = 128, mmq_threads = 256, mmq_smem = (128 + 128) * (272 + 64);
  CUfunction mmq_(int type, bool glu) {
    static const char* const names[2][2] = {{"tl_gemm_mmq_q4k", "tl_gemm_mmq_q80"},
                                            {"tl_gemm_mmq_glu_q4k", "tl_gemm_mmq_glu_q80"}};
    CUfunction& slot = mmq_fn[glu][type == 12 ? 0 : 1];
    if (!slot) {
      slot = load_(names[glu][type == 12 ? 0 : 1]);
      if (slot && d.FuncSetAttribute)
        d.FuncSetAttribute(slot, 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */, static_cast<int>(mmq_smem));
    }
    return slot;
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
    if (!fits_(f, args...)) return false;
    void* argv[] = {&args...};
    return launch_argv_(f, grid, block, smem, argv);
  }

  // The launch itself, once argv is built (by launch_ above, or by dispatch_).
  bool launch_argv_(CUfunction f, dims grid, dims block, unsigned smem,
                    void** argv) {
    if (!f) return false;
    pending = true;
    // Profiling: the launch's row, and — outside a graph capture, where an
    // event record would become a graph node — an event on each side of it
    // for the elapsed time.
    profile::row* pr = gpu::launched(name_(f));
    CUevent begin = nullptr;
    if (pr && d.timing_ok() && !stream) {  // null = the default stream
      begin = event_();
      d.EventRecord(begin, stream);
    }
    const bool ok =
        d.LaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                       smem, stream, argv, nullptr) == 0;
    if (begin) {
      CUevent end = event_();
      d.EventRecord(end, stream);
      timed.push_back({pr, begin, end});
    }
    return ok;
  }

  // The shared layer's launch (gpu_abi.h): residency from each view's access,
  // then argv as the views' device addresses followed by the params' 4-byte
  // fields. It is launch_'s contract read the other way round — every kernel
  // takes its pointers first and 4-byte scalars after — so a params struct
  // laid out in the kernel's argument order expands with no per-kernel code.
  bool dispatch_(CUfunction f, const gpu::arg* args, size_t n,
                 const void* params, size_t params_bytes, const gpu::grid& g) {
    constexpr size_t kMaxArgs = 32;
    const size_t words = params_bytes / 4;
    if (!f || params_bytes % 4 || n + words == 0 || n + words > kMaxArgs) {
      return false;
    }
    void* ptrs[kMaxArgs];
    uint32_t scalars[kMaxArgs];
    void* argv[kMaxArgs];
    for (size_t i = 0; i < n; i++) ptrs[i] = off_(args[i].s.buf, args[i].s.off);
    // Declined before any residency change, so a refusal leaves no trace.
    if (!fits_ptrs_(f, ptrs, n)) return false;
    for (size_t i = 0; i < n; i++) {
      before_kernel_(args[i].s.buf, args[i].a);
      argv[i] = &ptrs[i];
    }
    std::memcpy(scalars, params, params_bytes);
    for (size_t j = 0; j < words; j++) argv[n + j] = &scalars[j];
    return launch_argv_(f, {g.gx, g.gy, g.gz}, {g.tx, g.ty, g.tz},
                        g.scratch_bytes, argv);
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

inline bool dispatch(kop k, const gpu::arg* args, size_t n, const void* params,
                     size_t params_bytes, const gpu::grid& g);

// The ops this backend runs its own way: a different algorithm, several
// kernels, or a kernel whose ABI is its own. gpu_ops.h forwards to whichever of
// these exist (TL_GPU_DETECT_OWN) and answers false for the rest, so a backend
// declares what it has and nothing else. Defined below, among their helpers;
// the nn and llm layers' own ops are in cuda_nn.h and cuda_llm.h, and `own` at
// the end of this file joins the three.
struct own_core {
  static bool argmax(gpu::span a, int64_t n, int64_t* out_idx);
  // n floats copied device to device on the launch stream (a copy engine's job,
  // not a kernel's).
  static bool copy_f32(gpu::span from, gpu::span to, int64_t n);
  static bool binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
                              gpu::span b, const int64_t* b_strides,
                              gpu::span out, const int64_t* out_shape, int rank,
                              int64_t n, float scale, float offset);
  static bool where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
                       const int64_t* a_strides, gpu::span b,
                       const int64_t* b_strides, gpu::span out,
                       const int64_t* out_shape, int rank, int64_t n);
  static bool copy_nd(gpu::span a, const int64_t* a_strides, gpu::span out,
                      const int64_t* out_shape, int rank, int64_t n);
  static bool sum_to(gpu::span a, const int64_t* a_shape,
                     const int64_t* a_strides, const int64_t* acc, int rank,
                     int64_t out_n, int64_t reduced_n, gpu::span out);
  static bool pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                  const int64_t* out_shape, int rank, int axis, int64_t before,
                  int64_t n, int64_t out_n);
  static bool fold(gpu::span a, gpu::span out, const int64_t* a_shape,
                   const int64_t* out_shape, int rank, int axis, int64_t step,
                   int64_t n, int64_t out_n);
  static bool concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
                          const int64_t* out_shape, int rank, int axis,
                          int64_t before, int64_t n);
  static bool index_add(gpu::span idx, gpu::span values, gpu::span out,
                        int64_t row_size, int64_t k, int64_t out_n);
  static bool scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                              int64_t n, int64_t size);
  static bool gemm(gpu::span a, int64_t lda, bool ta, gpu::span b, int64_t ldb,
                   bool tb, gpu::span out, int64_t m, int64_t n, int64_t k,
                   float scale, float offset);
  static bool gemm_batched(gpu::span a, int64_t lda, bool ta, int64_t sa,
                           gpu::span b, int64_t ldb, bool tb, int64_t sb,
                           gpu::span out, int64_t m, int64_t n, int64_t k,
                           int64_t batch, float scale, float offset,
                           gpu::span bias = {});
  static bool gemm_bias(gpu::span a, int64_t lda, bool ta, gpu::span b,
                        int64_t ldb, bool tb, gpu::span bias, gpu::span out,
                        int64_t m, int64_t n, int64_t k, float scale,
                        float offset);
};

// What the shared launch policy (gpu_abi.h) may assume of this backend's
// kernels.
struct traits {
  // A [rows, cols] elementwise kernel reads its cell from a flat index.
  static constexpr bool cells_2d = false;
  // Each launch's tl::profile row carries a device time.
  static constexpr bool times_launches = true;
  // Blocks that keep the GPU busy: ~2 per SM on the 82-SM RTX 3090. The
  // target every tile and split choice below measures its grid against. A
  // constant rather than a device query because the decode attention's
  // device twin (attn_dpos_chunk in tensorlib_cuda.cu) bakes it in.
  static constexpr int64_t fill_groups = 164;
};

// The bf16 gemm's tile choice: the 128² tile is the more arithmetically
// efficient, but a few-hundred-row activation against a projection (256×768:
// 12 blocks) leaves most SMs idle, and the 64² tile makes that 48 at half the
// FMA per shared float and about half the registers per thread. Take the big
// tile only when its grid already fills.
// 128²-tile blocks a batch of m×n outputs takes: what the fill rules measure.
inline long blocks128_(int64_t m, int64_t n, int64_t batch) {
  return (long)((n + 127) / 128) * ((m + 127) / 128) * batch;
}
inline bool big_tile_(int64_t m, int64_t n) {
  return blocks128_(m, n, 1) >= traits::fill_groups;
}

// The wave plan. A launch places one block per SM up to kWaveSingles blocks
// and two per SM past that, so a wave is the fill's slots; a block takes time
// in proportion to its slabs, and a launch lasts as long as its busiest slot.
// So split K into `full` equal layers that fit the wave, each at least
// kWaveMinK deep (a block's fixed cost is some twenty 128² slabs), and when
// slots stay spare, into full layers plus a shorter tail layer that cycles
// `rounds` times through the spare slots while the full layers run. A tile
// with a single_min_k then cuts shallower layers, down to that, to fill the
// SMs one block each when the wave plan leaves fewer. Returns the chunk, k when
// unsplit. bench_cuda_gemm (RTX 3090, own GF/s, one shape per process, the fill
// heuristics → this): 512×1024×4096 16.2k → 18.1k, 512×4096×1024 15.8k →
// 19.0k (cuBLAS 18.9k), 256×4096×1024 13.5k → 16.2k, 1024³ 13.6k → 15.9k; and
// with the pipelined 64² over the register-staged one, 256×768×768 7.0k →
// 8.7k, 256×256×768 4.3k → 5.7k, 256×1024×256:nt 5.1k → 6.5k, single fill
// 256×768×256:nt 5.1k → 5.6k and 256×512×256:nt 3.9k → 4.5k.
constexpr unsigned kWaveMinK = 192;
inline unsigned sgemm_wave_chunk_(long tiles, unsigned k, const sgemm_tile& t) {
  // fill_groups is int64_t, tiles/slabs long — the same width everywhere this
  // library builds except Windows (LLP64, long is 32-bit), where std::min/max
  // over the two otherwise deduce to no common type. One cast here keeps the
  // rest of the function in `long`, as it was before the trait.
  const long fill = static_cast<long>(traits::fill_groups);
  if (tiles >= fill) return k;
  const long slabs = k / t.bk, min_slabs = kWaveMinK / t.bk;
  const long full = std::max<long>(1, std::min<long>(fill / tiles, slabs / min_slabs));
  long chunk_slabs = (slabs + full - 1) / full;
  if (const long spare = fill - full * tiles; spare > 0) {
    const long rounds = (tiles + spare - 1) / spare;
    const long parts = full * rounds + 1;  // a full layer is `rounds` tails
    if (slabs >= min_slabs * parts) chunk_slabs = (slabs * rounds + parts - 1) / parts;
  }
  const long layers = (slabs + chunk_slabs - 1) / chunk_slabs;
  if (t.single_min_k && tiles * layers < kWaveSingles) {
    const long singles =
        std::min<long>(kWaveSingles / tiles, slabs / long(t.single_min_k / t.bk));
    if (singles > layers) chunk_slabs = (slabs + singles - 1) / singles;
  }
  return (unsigned)chunk_slabs * t.bk;
}
// A TL_* census knob as a number, -1 when unset; callers keep it in a static.
inline long knob_(const char* name) {
  const char* e = std::getenv(name);
  return e ? std::atol(e) : -1L;
}
inline long sgemm_wave_blocks_(long tiles, unsigned k, const sgemm_tile& t) {
  const unsigned chunk = sgemm_wave_chunk_(tiles, k, t);
  return tiles * (long)((k + chunk - 1) / chunk);
}

// Which f32 tile: the 128² whenever its wave plan reaches its wave_min_blocks,
// or its grid has a wave's worth of
// blocks, or the K is long enough for its pipeline to matter, or M is at
// least 512 (four 128-rows against any N); the 64² for the short-K
// few-hundred-row projections whose grid it quadruples — and always when K
// is not a multiple of the 64² tile's 16-deep slab. From bench_cuda_gemm over
// the transformer block's shapes (own GF/s, register-staged 64² vs 128²): 256×768×768
// 9.6k vs 7.2k, 256×768×3072 13.4k vs 12.1k, 256×1024×4096 13.9k vs 15.4k,
// 512×1024×1024 12.4k vs 13.5k, 256×3072×768 12.3k vs 15.0k, 512×1024×4096
// 15.2k vs 17.9k, 2048³ 14.6k vs 19.2k. TL_TILE forces an index for the
// census (when K allows its slab), read once like the other TL_* knobs.
inline const sgemm_tile& sgemm_tile_(int64_t m, int64_t n, int64_t k,
                                     int64_t batch = 1) {
  static const long forced = knob_("TL_TILE");
  constexpr long ntiles = sizeof(sgemm_tiles) / sizeof(sgemm_tiles[0]);
  if (forced >= 0 && forced < ntiles && k % sgemm_tiles[forced].bk == 0)
    return sgemm_tiles[forced];
  const long tiles = blocks128_(m, n, batch);
  const sgemm_tile& big_tile = sgemm_tiles[1];
  bool big = sgemm_wave_blocks_(tiles, (unsigned)k, big_tile) >= big_tile.wave_min_blocks ||
             tiles >= 64 || k >= 2048 || m >= 512 || k % 16 != 0;
  return sgemm_tiles[big ? 1 : 0];
}

// Diagnostic knob: force gemv to skip split-K (gy=1). See context::no_splitk.
inline void set_no_splitk(bool v) { context::get().no_splitk = v; }
inline bool pending() { return context::get().pending; }

// End the batch: block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  {
    profile::detail::blocked waiting;
    c.d.CtxSynchronize();
  }
  c.pending = false;
  c.resolve_timed_();  // the device is idle: free reads
}

// One int comes back through the device buffer the context keeps (argmax_res_);
// a block of 256 threads for every 256 values, up to argmax_blocks.
inline bool own_core::argmax(gpu::span a, int64_t n, int64_t* out_idx) {
  auto& c = context::get();
  if (!c.ready || n <= 0 || n > INT32_MAX) return false;
  CUdeviceptr res = c.argmax_res_();
  CUfunction f = c.cached_(c.argmax_fn, "tl_argmax");
  if (!res || !f) return false;
  c.device_read_(a.buf);
  const unsigned un = static_cast<unsigned>(n), threads = 256;
  const unsigned blocks = std::min(context::argmax_blocks, (un + threads - 1) / threads);
  char* r = reinterpret_cast<char*>(res);
  float* pv = reinterpret_cast<float*>(r + context::argmax_pairs_at);
  int* pi = reinterpret_cast<int*>(r + context::argmax_pairs_at + 4 * context::argmax_blocks);
  if (!c.launch_(f, {blocks}, {threads}, 0, context::off_(a.buf, a.off), reinterpret_cast<int*>(r), un, pv, pi,
                 reinterpret_cast<unsigned*>(r + 4))) {
    return false;
  }
  flush();  // the result index must be ready before the 4-byte D2H
  int h = 0;
  if (c.d.MemcpyDtoH(&h, res, sizeof(int)) != 0) return false;
  *out_idx = h;
  return true;
}

// What a model may ask of this backend beyond the kernel contract (gpu.h).
struct caps {
  static constexpr bool graph_capture = true;
  static constexpr bool row_gemv = true;   // gemv_bf16_row: weights as [N,K]
  static constexpr bool bf16_gemm = true;  // gemm_bf16_nt: the batched prefill
  static constexpr bool kv_f16 = true;     // an f16 KV cache, every head width and path
};
using graph_exec = CUgraphExec;

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
  m->live.uploaded();
}

// `bytes` of `src` into the buffer at byte offset `off`, straight to the device:
// the host mirror is left alone (a model's quantized weights are gigabytes, and
// copying them through a mirror would hold them in host memory twice). The
// device copy is then the live one.
inline void upload_bytes(void* native, int64_t off, const void* src, int64_t bytes) {
  auto& c = context::get();
  if (!c.ready || !native || bytes <= 0) return;
  context::mirror* m = c.mirror_(native);
  if (!m || off < 0 || (size_t)(off + bytes) > m->bytes) return;
  c.d.MemcpyHtoD(m->dev + off, src, (size_t)bytes);
  m->live.before_kernel(gpu::access::out);  // device-live, host stale
}

// Mirror allocation: a device buffer (returned as `native`) paired with a host
// buffer (returned via `contents`). They are DISTINCT memory — the mirror's
// dirty state copies between them on demand (device_read_/sync_to_host). storage
// keeps native != contents, like Metal (MTLBuffer handle vs .contents pointer).
// `host_fill` (the host writes it first) needs nothing here: the host copy is
// its own allocation, which no queued work writes.
inline void* alloc(int64_t bytes, float** contents, bool host_fill = false) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  size_t nb = bytes > 0 ? (size_t)bytes : 4;
  CUdeviceptr dev = 0;
  float* host = nullptr;
  if (!c.mt.take(nb, dev, host)) {  // no recycled buffer of this exact size
    if (c.d.MemAlloc(&dev, nb) != 0) return nullptr;
    host = static_cast<float*>(std::malloc(nb));
    if (!host) {
      c.d.MemFree(dev);
      return nullptr;
    }
  }
  void* native = reinterpret_cast<void*>(dev);
  c.mt.insert(native, dev, host, nb, host_fill);
  if (contents) *contents = host;
  return native;
}

inline void release(void* buf, int64_t, float*) {
  auto& c = context::get();
  if (!c.ready || !buf) return;
  if (!c.mt.release(buf)) {
    c.d.MemFree(reinterpret_cast<CUdeviceptr>(buf));  // untracked (shouldn't happen); free outright
  }
}

// Reconcile a buffer for a CPU access: when the device holds the live copy,
// wait for the kernels in flight (one may still be writing it) and D2H.
// for_write invalidates the device copy (the host is about to mutate it). A
// host-live buffer needs no wait: kernels only ever write device copies, and
// an upload queued from it staged the pageable host bytes when it was queued.
// No-op for heap storages / unknown pointers.
inline void sync_to_host(void* native, bool for_write) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m) return;
  if (m->live.before_host(for_write)) {
    if (c.pending) flush();
    profile::detail::blocked timing{"d2h", m->bytes};
    c.d.MemcpyDtoH(m->host, m->dev, m->bytes);
  }
}

// The device core's one way to run a kernel for the shared ops (gpu_ops.h).
inline bool dispatch(kop k, const gpu::arg* args, size_t n, const void* params,
                     size_t params_bytes, const gpu::grid& g) {
  auto& c = context::get();
  if (!c.ready) return false;
  return c.dispatch_(c.fn_(k), args, n, params, params_bytes, g);
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
inline bool own_core::binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
                                 gpu::span b, const int64_t* b_strides,
                                 gpu::span out, const int64_t* out_shape,
                                 int rank, int64_t n, float scale,
                                 float offset) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  CUfunction f = c.bcast_nd_(op);
  if (!f) return false;
  c.device_read_(a.buf);
  c.device_read_(b.buf);
  c.device_write_(out.buf);
  const long long* pmeta =
      upload_bcast_meta_(c, out_shape, rank, {a_strides, b_strides});
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* pb = context::off_(b.buf, b.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(f, un, pa, pb, po, pmeta, rank, un, scale, offset);
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch, which existed
// on no backend before this (eval_one's where_ case always ran the CPU
// map_ternary). Same flat-index decode as binary_bcast_nd above, one more
// operand -- masking (attention/padding masks) is the concrete caller.
inline bool own_core::where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
                          const int64_t* a_strides, gpu::span b,
                          const int64_t* b_strides, gpu::span out,
                          const int64_t* out_shape, int rank, int64_t n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(cond.buf);
  c.device_read_(a.buf);
  c.device_read_(b.buf);
  c.device_write_(out.buf);
  const long long* pmeta = upload_bcast_meta_(
      c, out_shape, rank, {c_strides, a_strides, b_strides});
  if (!pmeta) return false;
  float* pc = context::off_(cond.buf, cond.off);
  float* pa = context::off_(a.buf, a.off);
  float* pb = context::off_(b.buf, b.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(c.where_nd_(), un, pc, pa, pb, po, pmeta, rank, un);
}

// N-D strided copy: clone()'s device arm for a view the flat one-input
// kernels cannot read (a permute, a transpose). Same flat-index decode and
// meta upload as where_nd above, one operand.
inline bool own_core::copy_nd(gpu::span a, const int64_t* a_strides, gpu::span out,
                         const int64_t* out_shape, int rank, int64_t n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_write_(out.buf);
  const long long* pmeta = upload_bcast_meta_(c, out_shape, rank, {a_strides});
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  return c.launch1d_(c.copy_nd_(), un, pa, po, pmeta, rank, un);
}

// Gather-based GPU dispatch for array.h's sum_to (un-broadcast a gradient).
// See tl_sum_to's own comment for why this needs no atomics, unlike
// index_add. `a_shape`/`a_strides` describe `a` (the caller has already
// checked it's contiguous); `acc` is array.h's own
// broadcast_strides(target, out_strides, a.shape()) -- 0 on every axis
// being summed over. `reduced_n` is the product of a_shape over exactly
// those zero-acc axes (1 if there are none).
inline bool own_core::sum_to(gpu::span a, const int64_t* a_shape,
                        const int64_t* a_strides, const int64_t* acc, int rank,
                        int64_t out_n, int64_t reduced_n, gpu::span out) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  CUfunction f = c.sum_to_();
  if (!f) return false;
  c.device_read_(a.buf);
  c.device_write_(out.buf);
  const long long* pmeta =
      upload_bcast_meta_(c, a_shape, rank, {a_strides, acc});
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(out_n);
  unsigned ured = static_cast<unsigned>(reduced_n);
  // A deep reduction (a bias gradient sums its column over every row) earns a
  // block per output, whose threads split that range; a shallow one keeps the
  // flat kernel, where one thread per output already has the parallelism.
  if (ured >= 64) {
    if (CUfunction fb = c.sum_to_blocked_()) {
      unsigned block = 256;
      return c.launch_(fb, {un ? un : 1}, {block}, block * sizeof(float), pa, po,
                       pmeta, rank, un, ured);
    }
  }
  return c.launch1d_(f, un, pa, po, pmeta, rank, un, ured);
}

// Places `a` (contiguous) into a zero buffer of out_shape (array.h's
// gpu_pad_ allocates `out` uninitialized via array::empty — this zeros the
// device copy directly, no host round trip), shifted by `before` along
// `axis`. No scale/offset — eval_one's shared epilogue applies those (see
// array.h's op_t::pad_ case).
inline bool own_core::pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                     const int64_t* out_shape, int rank, int axis,
                     int64_t before, int64_t n, int64_t out_n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_write_(out.buf);
  zero_device_(reinterpret_cast<CUdeviceptr>(context::off_(out.buf, out.off)),
               out_n);
  int64_t out_strides[kPadFoldMaxRank];
  const long long* pmeta =
      upload_pad_fold_meta_(c, a_shape, rank, out_shape, rank, out_strides);
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  unsigned ushift = static_cast<unsigned>(before * out_strides[axis]);
  return c.launch1d_(c.pad_(), un, pa, po, pmeta, rank, ushift, un);
}

// unfold's inverse: scatter-add `a` (contiguous; its last dim is the sliding
// window) into a zero buffer of out_shape (zeroed the same way as pad()
// above) — every overlap accumulates via atomicAdd, so it must start at 0.
inline bool own_core::fold(gpu::span a, gpu::span out, const int64_t* a_shape,
                      const int64_t* out_shape, int rank, int axis,
                      int64_t step, int64_t n, int64_t out_n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_write_(out.buf);
  zero_device_(reinterpret_cast<CUdeviceptr>(context::off_(out.buf, out.off)),
               out_n);
  int64_t out_strides[kPadFoldMaxRank];
  const long long* pmeta = upload_pad_fold_meta_(c, a_shape, rank, out_shape,
                                                 rank - 1, out_strides);
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  unsigned ustep = static_cast<unsigned>(step);
  return c.launch1d_(c.fold_(), un, pa, po, pmeta, rank, axis, ustep, un);
}

// Writes `a` (contiguous) into `out_shape` at `before` along `axis`,
// WITHOUT zeroing first — array.h's gpu_concat_ calls this once per part
// into the same `out`, and every part together covers it exactly (no
// padding border to zero, unlike pad() above). Reuses pad's own kernel:
// writing a same-shape source at an axis-shifted offset is exactly what
// tl_pad already does per source element.
inline bool own_core::concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
                             const int64_t* out_shape, int rank, int axis,
                             int64_t before, int64_t n) {
  if (rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_write_(out.buf);
  int64_t out_strides[kPadFoldMaxRank];
  const long long* pmeta =
      upload_pad_fold_meta_(c, a_shape, rank, out_shape, rank, out_strides);
  if (!pmeta) return false;
  float* pa = context::off_(a.buf, a.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  unsigned ushift = static_cast<unsigned>(before * out_strides[axis]);
  return c.launch1d_(c.pad_(), un, pa, po, pmeta, rank, ushift, un);
}

// index_select's dual: scatter-add `values` into `out` by row index.
// Repeated indices really do collide (real write conflicts — the kernel
// uses atomicAdd), so `out` must start zeroed, same as pad/fold above.
inline bool own_core::index_add(gpu::span idx, gpu::span values, gpu::span out,
                           int64_t row_size, int64_t k, int64_t out_n) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(idx.buf);
  c.device_read_(values.buf);
  c.device_write_(out.buf);
  zero_device_(reinterpret_cast<CUdeviceptr>(context::off_(out.buf, out.off)),
               out_n);
  float* pidx = context::off_(idx.buf, idx.off);
  float* pv = context::off_(values.buf, values.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(k * row_size);
  unsigned urow = static_cast<unsigned>(row_size);
  return c.launch1d_(c.index_add_(), un, pidx, pv, po, urow, un);
}

// One-hot scatter into a new trailing axis of size `size`: out[..., k] =
// values[...] where indices[...] == k, else 0. Every input position maps
// to a distinct output slot (the axis is brand new), so — unlike
// index_add above — there is no accumulation and no atomics; `out` still
// starts zeroed since untouched slots must read back as 0.
inline bool own_core::scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                                 int64_t n, int64_t size) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(idx.buf);
  c.device_read_(values.buf);
  c.device_write_(out.buf);
  zero_device_(reinterpret_cast<CUdeviceptr>(context::off_(out.buf, out.off)),
               n * size);
  float* pidx = context::off_(idx.buf, idx.off);
  float* pv = context::off_(values.buf, values.off);
  float* po = context::off_(out.buf, out.off);
  unsigned un = static_cast<unsigned>(n);
  unsigned usize = static_cast<unsigned>(size);
  return c.launch1d_(c.scatter_axis_(), un, pidx, pv, po, usize, un);
}

inline bool own_core::copy_f32(gpu::span from, gpu::span to, int64_t n) {
  auto& c = context::get();
  if (!c.ready || !c.d.MemcpyDtoDAsync || n <= 0) return false;
  c.device_read_(from.buf);
  c.device_write_(to.buf);
  return c.d.MemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(context::off_(to.buf, to.off)),
                             reinterpret_cast<CUdeviceptr>(context::off_(from.buf, from.off)),
                             static_cast<size_t>(n) * sizeof(float), c.stream) == 0;
}

// Split-K (ladder ②) for the f32 gemm: partition K into S z-slices so S× more
// blocks run concurrently. Split-K partitions K (not replicates it), so A/B
// global traffic is unchanged — the only cost is C written S× via atomicAdd
// into a pre-zeroed buffer (the kernel folds scale/offset into the partials,
// so a fused epilogue splits too). The wave plan (sgemm_wave_chunk_) sets S
// once it puts the tile's wave_min_blocks on the grid (the 64² always). The
// 128² grids it does not fill keep the fill heuristics: enough
// splits to reach ~64 blocks with each at least 96 deep, then a long K keeps
// splitting in slices of at least 1024 up to ~128 blocks (own GF/s:
// 512×1024×4096 S=1 17.9k vs S=2 15.4k; 512×1024×1024 S=2 13.5k vs S=4
// 12.3k). `base_blocks` counts the batch too, so a batched product that
// already fills the GPU declines to split. chunk is a whole number of the
// tile's bk-deep K slabs; S == 1 leaves chunk == k (the kernel reads that as
// "no split"). TL_SPLITK forces S and TL_KSPLIT the chunk (a last layer may be
// shorter) for the census, read once like the other TL_* knobs.
struct sgemm_splitk {
  unsigned S, chunk;
};
inline sgemm_splitk sgemm_splitk_(long base_blocks, unsigned k, const sgemm_tile& t) {
  auto by_chunk = [k](unsigned chunk) -> sgemm_splitk {
    if (chunk >= k) return {1, k};
    return {(k + chunk - 1) / chunk, chunk};
  };
  static const long forced_chunk = knob_("TL_KSPLIT");
  static const long forced = knob_("TL_SPLITK");
  auto whole_slabs = [&t](unsigned chunk) { return (chunk + t.bk - 1) / t.bk * t.bk; };
  if (forced_chunk > 0) return by_chunk(whole_slabs((unsigned)forced_chunk));
  if (forced < 0) {
    const unsigned wave = sgemm_wave_chunk_(base_blocks, k, t);
    if (base_blocks * (long)((k + wave - 1) / wave) >= t.wave_min_blocks)
      return by_chunk(wave);
  }
  const long by_fill = std::min((63 + base_blocks) / base_blocks, std::max<long>(1, k / 96));
  const long by_k = std::min<long>(k / 1024, (127 + base_blocks) / base_blocks);
  const long want = forced >= 0 ? forced : std::max(by_fill, by_k);
  if (want <= 1) return {1, k};
  return by_chunk(whole_slabs((k + (unsigned)want - 1) / (unsigned)want));
}

// C[bi](m,n) = (A[bi] @ B[bi]) * scale + offset (+ bias[j]) for bi < batch,
// the batch elements sa/sb floats apart (0 broadcasts) and C's packed at m·n.
// lda/ldb row strides; trans reads a transposed view in place; `bias`, when
// given, is n contiguous floats at biaso added in the kernel's store. One
// launch: the batch rides on gridDim.z next to split-K. batch == 1 is the
// plain GEMM and keeps the one-output-per-thread fallback (tl_sgemm) for the
// layouts the fast path declines; batch > 1 has no fallback here and returns
// false, so the caller loops per slice.
inline bool own_core::gemm_batched(gpu::span a, int64_t lda, bool ta, int64_t sa,
                              gpu::span b, int64_t ldb, bool tb, int64_t sb,
                              gpu::span out, int64_t m, int64_t n, int64_t k,
                              int64_t batch, float scale, float offset,
                              gpu::span bias) {
  auto& c = context::get();
  if (!c.ready || batch < 1) return false;
  c.device_read_(a.buf);
  c.device_read_(b.buf);
  if (bias.buf) c.device_read_(bias.buf);
  c.device_write_(out.buf);
  float* pa = context::off_(a.buf, a.off);
  float* pb = context::off_(b.buf, b.off);
  float* pbias = bias.buf ? context::off_(bias.buf, bias.off) : nullptr;
  float* po = context::off_(out.buf, out.off);
  unsigned um = (unsigned)m, un = (unsigned)n, uk = (unsigned)k;

  // Tiled fast path (tl_sgemm_cp*, one per tile per operand layout): each
  // operand contiguous in its own layout (lda == k, or == m for a transposed
  // view; ldb == n, or == k transposed), K%8==0 for the 8-slab, the dim a
  // float4 load runs along a multiple of 4 (N for NN's B, M for TN's A; K%8
  // covers the K-contiguous operands), and bases where the kernel declares
  // them (TL_ALIGN). The batch strides ride in the kernel's 32-bit
  // arguments and must keep every slice 16B-aligned: multiples of 4 floats (C
  // is stored per element, so only its m·n stride has to fit). M and N block
  // edges are predicated in-kernel. Strided views, odd K and unaligned
  // offsets fall to tl_sgemm.
  bool a_ok = ta ? (lda == m && m % 4 == 0) : (lda == k);
  bool b_ok = tb ? (ldb == k) : (ldb == n && n % 4 == 0);
  bool batch_ok = sa % 4 == 0 && sb % 4 == 0 && sa <= (int64_t)UINT32_MAX &&
                  sb <= (int64_t)UINT32_MAX &&
                  (batch == 1 || m * n <= (int64_t)UINT32_MAX);
  if (a_ok && b_ok && batch_ok && k % 8 == 0 && m > 0 && n > 0 && k > 0) {
    // Tile choice (sgemm_tile_), then split-K fills what the grid leaves.
    const sgemm_tile& t = sgemm_tile_(m, n, k, batch);
    unsigned gx = (un + t.bm - 1) / t.bm, gy = (um + t.bm - 1) / t.bm;
    auto [S, ksplit] = sgemm_splitk_((long)gx * gy * batch, uk, t);
    // A fused bias goes into the store unsplit, and under the partials split:
    // their atomicAdds land on the bias rows in place of a zeroed C, so the
    // split kernel is the plain one.
    const bool need_fill = pbias && S > 1;
    CUfunction f = c.sgemm_(ta, tb, pbias && !need_fill, t);
    // tl_fill_rows puts the rows on gridDim.y, which the driver caps at 65535
    // like z below.
    CUfunction fill = need_fill && m <= 65535 ? c.fill_rows_() : nullptr;
    // gridDim.z carries batch × S; the driver caps it at 65535.
    if (f && c.fits_(f, pa, pb, po, pbias) && (int64_t)S * batch <= 65535 &&
        (!need_fill || fill)) {
      if (fill) {
        if (!c.launch_(fill, {(un + 255) / 256, um}, {256}, 0, po, pbias, um, un))
          return false;
      } else if (S > 1) {
        // atomicAdd needs a zeroed C (async on the stream where the driver has
        // it: no host sync, capturable — as gemm_bf16_nt's is).
        zero_device_(reinterpret_cast<CUdeviceptr>(po), batch * m * n);
      }
      unsigned usa = (unsigned)sa, usb = (unsigned)sb, usc = (unsigned)(m * n),
               uz = (unsigned)(S * batch);
      return c.launch_(f, {gx, gy, uz}, {256}, 0, pa, pb, po, pbias, um, un, uk,
                       scale, offset, ksplit, usa, usb, usc);
    }
  }
  if (batch != 1) return false;

  unsigned ula = (unsigned)lda, ulb = (unsigned)ldb;
  unsigned uta = ta ? 1u : 0u, utb = tb ? 1u : 0u;
  unsigned bx = 16, by = 16;
  unsigned gx = (un + bx - 1) / bx, gy = (um + by - 1) / by;
  if (gx == 0) gx = 1;
  if (gy == 0) gy = 1;
  return c.launch_(c.sgemm_fallback_(), {gx, gy}, {bx, by}, 0, pa, pb, po,
                   pbias, um, un, uk, ula, ulb, uta, utb, scale, offset);
}

// C(m,n) = (A @ B) * scale + offset: the batch == 1 case of gemm_batched.
inline bool own_core::gemm(gpu::span a, int64_t lda, bool ta, gpu::span b,
                      int64_t ldb, bool tb, gpu::span out, int64_t m, int64_t n,
                      int64_t k, float scale, float offset) {
  return gemm_batched(a, lda, ta, 0, b, ldb, tb, 0, out, m, n, k, 1, scale,
                      offset);
}

// C(m,n) = (A @ B) * scale + offset + bias[j]: addmm's shape, the row bias
// added in the gemm's own store rather than by a second pass over C.
inline bool own_core::gemm_bias(gpu::span a, int64_t lda, bool ta, gpu::span b,
                           int64_t ldb, bool tb, gpu::span bias, gpu::span out,
                           int64_t m, int64_t n, int64_t k, float scale,
                           float offset) {
  return gemm_batched(a, lda, ta, 0, b, ldb, tb, 0, out, m, n, k, 1, scale,
                      offset, bias);
}

// CPU-read barrier: sync the GPU before any host read of a managed buffer.
// Nothing to wait for before a CPU access in general: kernels write only
// device copies, so sync_to_host waits per buffer, when that buffer's live copy
// is on the device. Flushing here made every host read of a host-live buffer
// (a rank-0 scalar operand, a freshly filled constant) drain the whole stream.
inline void cpu_barrier() {}

}  // namespace cuda

}  // namespace tl

// The nn and llm layers' own ops, built on the device core above.
#include "cuda_nn.h"
#include "cuda_llm.h"

namespace tl {
namespace cuda {

// What gpu_ops.h forwards to: the own ops of every layer. TL_GPU_DETECT_OWN
// finds a member of any of the bases.
struct own : own_core, own_nn, own_llm {};

}  // namespace cuda
}  // namespace tl

#endif  // TENSORLIB_CUDA && !__APPLE__
