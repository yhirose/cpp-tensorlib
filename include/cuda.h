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

namespace tl {
namespace cuda {

using kop = tl::metal::kop;

#if defined(TENSORLIB_CUDA) && !defined(__APPLE__)

}  // namespace cuda
}  // namespace tl

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
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

  bool ok() const {
    return Init && DeviceGet && DevicePrimaryCtxRetain && CtxSetCurrent &&
           CtxSynchronize && ModuleLoadData && ModuleGetFunction &&
           LaunchKernel && MemAlloc && MemFree && MemcpyHtoD && MemcpyDtoH &&
           MemsetD8;
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

  // M9 fused decode attention.
  CUfunction attn_decode_fn = nullptr;
  CUfunction attn_decode_() {
    if (!attn_decode_fn)
      d.ModuleGetFunction(&attn_decode_fn, mod, "tl_attn_decode_f32");
    return attn_decode_fn;
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

// Mirror allocation: a device buffer (returned as `native`) paired with a host
// buffer (returned via `contents`). They are DISTINCT memory — the mirror's
// dirty state copies between them on demand (device_read_/sync_to_host). storage
// keeps native != contents, like Metal (MTLBuffer handle vs .contents pointer).
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  size_t nb = bytes > 0 ? (size_t)bytes : 4;
  CUdeviceptr dev = 0;
  if (c.d.MemAlloc(&dev, nb) != 0) return nullptr;
  float* host = static_cast<float*>(std::malloc(nb));
  if (!host) {
    c.d.MemFree(dev);
    return nullptr;
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
  if (it != c.mirrors.end()) {
    std::free(it->second.host);
    c.mirrors.erase(it);
  }
  c.d.MemFree(dev);
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
  void* args[] = {&pa, &pb, &po, &un, &scale, &offset};
  return c.launch1d_(c.fn_(op), un, args);
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
  if (static_cast<long>(bx) < target && uk >= 512) {
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
  if (gy > 1)
    c.d.MemsetD8(reinterpret_cast<CUdeviceptr>(y_native), 0, (size_t)un * 4);
  void* args[] = {&pa, &pB, &py, &un, &uk, &ksplit};
  c.pending = true;
  return c.d.LaunchKernel(f, bx, gy, 1, 256, 1, 1, 0, nullptr, args, nullptr) ==
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

// M9 fused decode attention: out(h,:) = softmax(scale·q(h,:)·K(h)^T)·V(h) in
// one pass. q [heads,D], K/V [heads,ctx,D], out [heads,D], contiguous, D==128.
// One block per head. No epilogue (attention is its own op).
inline bool attn_decode(void* q, void* K, void* V, void* out, int64_t heads,
                        int64_t ctx, int64_t D, float scale) {
  auto& c = context::get();
  if (!c.ready || D != 128) return false;
  c.device_read_(q);
  c.device_read_(K);
  c.device_read_(V);
  c.device_write_(out);
  float* pq = context::off_(q, 0);
  float* pk = context::off_(K, 0);
  float* pv = context::off_(V, 0);
  float* po = context::off_(out, 0);
  unsigned uctx = static_cast<unsigned>(ctx);
  void* args[] = {&pq, &pk, &pv, &po, &uctx, &scale};
  c.pending = true;
  return c.d.LaunchKernel(c.attn_decode_(), static_cast<unsigned>(heads), 1, 1,
                          128, 1, 1, 0, nullptr, args, nullptr) == 0;
}

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
      return c.d.LaunchKernel(f, gx, gy, S, 256, 1, 1, 0, nullptr, rb,
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
  return c.d.LaunchKernel(sg, gx, gy, 1, bx, by, 1, 0, nullptr, args,
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
