#pragma once

// Metal GPU backend (macOS).
//
// Depends only on storage-free primitives (raw MTLBuffer handles + byte
// offsets), so storage.h can build on it without a cycle. Design points,
// informed by silarray:
//   - Unified memory: when a device exists, every tl::storage buffer is a
//     shared-mode MTLBuffer from a size-keyed pool — CPU and GPU read the
//     same bytes, no residency tracking, no transfers.
//   - One long-lived command buffer/encoder: dispatches accumulate without
//     committing; flush() (end + commit + waitUntilCompleted) runs when the
//     graph evaluation finishes or a CPU-side read needs the data.
//   - Kernels JIT-compile once from the #embed'd MSL source on first GPU
//     dispatch. Editing metal_kernels.metal requires rebuilding the host.
//
// On non-Apple builds everything is an inline stub returning false/null, so
// callers carry no platform conditionals.

#include <cstdint>

#ifdef __APPLE__

#include <objc.h>

#include <stdexcept>
#include <unordered_map>
#include <vector>

extern "C" void* MTLCreateSystemDefaultDevice(void);
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

#endif

namespace tl {
namespace metal {

enum class kop { add, sub, mul, div, exp_, log_, sqrt_, sigmoid, relu, affine };

#ifdef __APPLE__

struct mtl_size {
  unsigned long w, h, d;
};

struct context {
  objc::id device = nullptr;
  objc::id queue = nullptr;
  objc::id library = nullptr;
  objc::id cb = nullptr;   // command buffer (while pending)
  objc::id enc = nullptr;  // compute encoder (while pending)
  void* pool = nullptr;    // autorelease pool for the pending batch
  bool pending = false;
  std::unordered_map<int64_t, std::vector<void*>> free_bufs;  // by byte size
  std::unordered_map<int, objc::id> psos;

  static context& get() {
    static auto* c = new context();  // leaked: outlives all storage deleters
    return *c;
  }

  context() {
    device = MTLCreateSystemDefaultDevice();
    if (device) queue = objc::send(device, "newCommandQueue");
  }

  static const char* msl_source_() {
    static const char src[] = {
#embed "metal_kernels.metal"
        , '\0'};
    return src;
  }

  static const char* kernel_name_(kop op) {
    switch (op) {
      case kop::add: return "add_";
      case kop::sub: return "sub_";
      case kop::mul: return "mul_";
      case kop::div: return "div_";
      case kop::exp_: return "exp_";
      case kop::log_: return "log_";
      case kop::sqrt_: return "sqrt_";
      case kop::sigmoid: return "sigmoid_";
      case kop::relu: return "relu_";
      case kop::affine: return "affine_";
    }
    return "";
  }

  objc::id pso_(kop op) {
    auto it = psos.find(static_cast<int>(op));
    if (it != psos.end()) return it->second;
    if (!library) {
      objc::id err = nullptr;
      auto src = objc::send(objc::cls("NSString"), "stringWithUTF8String:",
                            msl_source_());
      library = objc::send(device, "newLibraryWithSource:options:error:", src,
                           static_cast<objc::id>(nullptr), &err);
      if (!library) {
        throw std::runtime_error("tl::metal: MSL compile failed: " +
                                 objc::error_str(err));
      }
    }
    auto name = objc::send(objc::cls("NSString"), "stringWithUTF8String:",
                           kernel_name_(op));
    auto fn = objc::send(library, "newFunctionWithName:", name);
    objc::id err = nullptr;
    auto pso = objc::send(device, "newComputePipelineStateWithFunction:error:",
                          fn, &err);
    if (!pso) {
      throw std::runtime_error("tl::metal: PSO creation failed for " +
                               std::string(kernel_name_(op)) + ": " +
                               objc::error_str(err));
    }
    psos[static_cast<int>(op)] = pso;
    return pso;
  }

  void ensure_encoder_() {
    if (cb) return;
    pool = objc_autoreleasePoolPush();
    cb = objc::send(queue, "commandBuffer");
    enc = objc::send(cb, "computeCommandEncoder");
    pending = true;
  }
};

inline bool available() { return context::get().device != nullptr; }

inline bool pending() { return context::get().pending; }

// End the batch: commit and block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  objc::send(c.enc, "endEncoding");
  objc::send(c.cb, "commit");
  objc::send(c.cb, "waitUntilCompleted");
  objc_autoreleasePoolPop(c.pool);
  c.cb = c.enc = nullptr;
  c.pool = nullptr;
  c.pending = false;
}

// Pooled shared-mode MTLBuffer. Returns null when no device (caller falls
// back to heap). `bytes` is the pool key — pass the same value to release.
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.device) return nullptr;
  void* buf = nullptr;
  auto it = c.free_bufs.find(bytes);
  if (it != c.free_bufs.end() && !it->second.empty()) {
    buf = it->second.back();
    it->second.pop_back();
  } else {
    // MTLResourceStorageModeShared = 0
    buf = objc::send(c.device, "newBufferWithLength:options:",
                     static_cast<unsigned long>(bytes), 0ul);
    if (!buf) return nullptr;
  }
  *contents = static_cast<float*>(objc::send(buf, "contents"));
  return buf;
}

inline void release(void* buf, int64_t bytes) {
  context::get().free_bufs[bytes].push_back(buf);
}

namespace detail_ {

struct ew_params {
  float scale;
  float offset;
  uint32_t n;
};

inline void dispatch_(objc::id enc, objc::id pso, const ew_params& p,
                      unsigned long params_index) {
  objc::send(enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), params_index);
  unsigned long groups = (p.n + 255ul) / 256ul;
  using dispatch_fn = void (*)(objc::id, SEL, mtl_size, mtl_size);
  reinterpret_cast<dispatch_fn>(objc_msgSend)(
      enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
      mtl_size{groups, 1, 1}, mtl_size{256, 1, 1});
  (void)pso;
}

}  // namespace detail_

// Contiguous elementwise dispatches; offsets in bytes. Epilogue (scale,
// offset) applies inside the kernel. Encodes without committing.
inline bool binary(kop op, void* a, int64_t ao, void* b, int64_t bo, void* out,
                   int64_t oo, int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  objc::send(c.enc, "setBuffer:offset:atIndex:", a,
             static_cast<unsigned long>(ao), 0ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", b,
             static_cast<unsigned long>(bo), 1ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", out,
             static_cast<unsigned long>(oo), 2ul);
  detail_::dispatch_(c.enc, pso,
                     {scale, offset, static_cast<uint32_t>(n)}, 3ul);
  return true;
}

inline bool unary(kop op, void* a, int64_t ao, void* out, int64_t oo,
                  int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  objc::send(c.enc, "setBuffer:offset:atIndex:", a,
             static_cast<unsigned long>(ao), 0ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", out,
             static_cast<unsigned long>(oo), 1ul);
  detail_::dispatch_(c.enc, pso,
                     {scale, offset, static_cast<uint32_t>(n)}, 2ul);
  return true;
}

#else  // !__APPLE__ — stubs so callers carry no platform conditionals

inline bool available() { return false; }
inline bool pending() { return false; }
inline void flush() {}
inline void* alloc(int64_t, float**) { return nullptr; }
inline void release(void*, int64_t) {}
inline bool binary(kop, void*, int64_t, void*, int64_t, void*, int64_t,
                   int64_t, float, float) {
  return false;
}
inline bool unary(kop, void*, int64_t, void*, int64_t, int64_t, float, float) {
  return false;
}

#endif

// Every CPU-side buffer read funnels through array::raw()/data(), which call
// this: one choke point makes mixed CPU/GPU graphs safe.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace metal

inline bool gpu_available() { return metal::available(); }

}  // namespace tl
