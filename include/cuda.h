#pragma once

// Own CUDA backend (M6) — the non-Apple GPU backend, mirroring metal.h. The
// NVIDIA driver is loaded via dlopen (no link-time CUDA dependency), so a
// binary built with TENSORLIB_CUDA still runs — and falls back to CPU — on a
// machine with no driver. Kernels are AOT-compiled to PTX by nvcc and #embed'd
// (see kernels/tensorlib_cuda.cu), then loaded through the driver API at first
// use. No CUDA runtime, no cuBLAS/CUTLASS.
//
// Memory: cuMemAllocManaged — one allocation visible to both host and device,
// so storage's native==contents (built for Apple unified memory) is reused
// unchanged and CPU ops read/write the same pointer the kernels do (roadmap
// M6 "memory model decided"). View offsets are folded host-side into the
// pointer passed to each kernel.
//
// Real implementation is gated on TENSORLIB_CUDA && !__APPLE__ (Apple uses
// Metal; a plain build gets the stubs below). The API matches metal.h exactly
// — available/pending/flush/alloc/release/binary/unary/gemm/row_op — so the
// eval_one dispatch seam is backend-agnostic and carries no platform #ifdefs.

#include <cstdint>

#include "metal.h"  // reuse tl::metal::kop (platform-independent op enum)

namespace tl {
namespace cuda {

using kop = tl::metal::kop;

#if defined(TENSORLIB_CUDA) && !defined(__APPLE__)

}  // namespace cuda
}  // namespace tl

#include <dlfcn.h>

#include <cstdio>
#include <unordered_map>

namespace tl {
namespace cuda {

// ---- driver API surface (declared by hand; loaded from libcuda via dlopen) ----
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUstream = CUstream_st*;
enum { CU_MEM_ATTACH_GLOBAL = 0x1 };

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
  CUresult (*MemAllocManaged)(CUdeviceptr*, size_t, unsigned) = nullptr;
  CUresult (*MemFree)(CUdeviceptr) = nullptr;

  bool ok() const {
    return Init && DeviceGet && DevicePrimaryCtxRetain && CtxSetCurrent &&
           CtxSynchronize && ModuleLoadData && ModuleGetFunction &&
           LaunchKernel && MemAllocManaged && MemFree;
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
  std::unordered_map<int, CUfunction> fns;

  static context& get() {
    static auto* c = new context();  // leaked: outlives all storage deleters
    return *c;
  }

  context() {
    const char* paths[] = {"/usr/lib/wsl/lib/libcuda.so.1", "libcuda.so.1",
                           "libcuda.so"};
    for (const char* p : paths) {
      lib = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
      if (lib) break;
    }
    if (!lib) return;  // no driver → available()==false → CPU fallback
    auto S = [&](const char* n) { return dlsym(lib, n); };
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
    d.MemAllocManaged =
        (CUresult(*)(CUdeviceptr*, size_t, unsigned))S("cuMemAllocManaged");
    d.MemFree = (CUresult(*)(CUdeviceptr))S("cuMemFree_v2");
    if (!d.MemFree) d.MemFree = (CUresult(*)(CUdeviceptr))S("cuMemFree");
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

  // char* pointer arithmetic to fold a byte offset into a managed pointer.
  static float* off_(void* base, int64_t byte_off) {
    return reinterpret_cast<float*>(static_cast<char*>(base) + byte_off);
  }

  bool launch1d_(CUfunction f, unsigned n, void** args, unsigned shared = 0) {
    if (!f) return false;
    unsigned block = 256, grid = (n + block - 1) / block;
    if (grid == 0) grid = 1;
    pending = true;
    return d.LaunchKernel(f, grid, 1, 1, block, 1, 1, shared, nullptr, args,
                          nullptr) == 0;
  }
};

inline bool available() { return context::get().ready; }
inline bool pending() { return context::get().pending; }

// End the batch: block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  c.d.CtxSynchronize();
  c.pending = false;
}

// Managed allocation: the returned pointer is BOTH the device handle (native)
// and a host-usable pointer (contents), so storage keeps native==contents.
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  CUdeviceptr p = 0;
  if (c.d.MemAllocManaged(&p, bytes > 0 ? (size_t)bytes : 4,
                          CU_MEM_ATTACH_GLOBAL) != 0)
    return nullptr;
  void* ptr = reinterpret_cast<void*>(p);
  if (contents) *contents = reinterpret_cast<float*>(ptr);
  return ptr;
}

inline void release(void* buf, int64_t, float*) {
  auto& c = context::get();
  if (c.ready && buf) c.d.MemFree(reinterpret_cast<CUdeviceptr>(buf));
}

// out = (a OP b) * scale + offset, contiguous; offsets in bytes.
inline bool binary(kop op, void* a, int64_t ao, void* b, int64_t bo, void* out,
                   int64_t oo, int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pa, &pb, &po, &un, &scale, &offset};
  return c.launch1d_(c.fn_(op), un, args);
}

inline bool unary(kop op, void* a, int64_t ao, void* out, int64_t oo, int64_t n,
                  float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  float* pa = context::off_(a, ao);
  float* po = context::off_(out, oo);
  unsigned un = static_cast<unsigned>(n);
  void* args[] = {&pa, &po, &un, &scale, &offset};
  return c.launch1d_(c.fn_(op), un, args);
}

// C(m,n) = (A @ B) * scale + offset. lda/ldb row strides; trans reads a
// transposed view in place. One output per thread (16×16 blocks).
inline bool gemm(void* a, int64_t ao, int64_t lda, bool ta, void* b, int64_t bo,
                 int64_t ldb, bool tb, void* out, int64_t oo, int64_t m,
                 int64_t n, int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  float* pa = context::off_(a, ao);
  float* pb = context::off_(b, bo);
  float* po = context::off_(out, oo);
  unsigned um = (unsigned)m, un = (unsigned)n, uk = (unsigned)k;
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
  return c.d.LaunchKernel(sg, gx, gy, 1, bx, by, 1, 0, nullptr, args,
                          nullptr) == 0;
}

// Row op over the last axis: softmax writes rows×cols; row_sum/row_max write
// one value per row, affine epilogue. One block per row, 256 threads.
inline bool row_op(kop op, void* in, int64_t io, void* out, int64_t oo,
                   int64_t rows, int64_t cols, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  float* pin = context::off_(in, io);
  float* po = context::off_(out, oo);
  unsigned ur = (unsigned)rows, uc = (unsigned)cols;
  void* args[] = {&pin, &po, &ur, &uc, &scale, &offset};
  CUfunction f = c.fn_(op);
  if (!f) return false;
  unsigned block = 256;
  c.pending = true;
  return c.d.LaunchKernel(f, ur ? ur : 1, 1, 1, block, 1, 1,
                          block * sizeof(float), nullptr, args, nullptr) == 0;
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
