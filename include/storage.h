#pragma once

#include <cuda.h>
#include <types.h>

#include <cstdint>
#include <memory>

namespace tl {

// Flat F32 buffer shared between arrays (views share one storage). When a
// Metal device exists every buffer is a pooled shared-mode MTLBuffer
// (unified memory: same bytes visible to CPU and GPU); otherwise heap. CUDA
// residency attaches here in M6.
//
// TL_RUNTIME_HOOKS (opt-in, for embedders like culebra that gate features
// per-binary): allocation, evaluation and the CPU barrier route through
// function-pointer hooks installed by tl::install_runtime_hooks(). Until
// installed, allocation falls back to heap and evaluation throws — an
// embedder's feature loader installs the hooks before any tensor work. The
// point: translation units that only *build* graphs and create arrays
// reference no backend symbol (Metal/Accelerate/kernels), so a linker can
// drop the entire execution engine from binaries that never evaluate.
struct storage {
  std::shared_ptr<void> buf;   // owner: returns to the Metal pool or frees
  float* ptr = nullptr;        // F32 element pointer; for bf16 storage it is
                               // a reinterpreted uint16_t* (see dt) — only the
                               // widen/narrow converters and native GPU kernels
                               // touch bf16 bytes, never generic float derefs.
  void* native = nullptr;      // MTLBuffer handle when Metal-backed
  int64_t size = 0;            // elements
  dtype dt = dtype::f32;

  float* data() const { return ptr; }

  static storage make(int64_t n, dtype dt = dtype::f32);

  // Device-preferred allocation (Metal pool with heap fallback). Referenced
  // directly in the default build; only via the installed hook under
  // TL_RUNTIME_HOOKS.
  static storage make_device_(int64_t n, dtype dt = dtype::f32) {
    storage s;
    s.size = n;
    s.dt = dt;
    int64_t bytes = n > 0 ? n * dtype_size(dt) : 4;  // device length nonzero
    float* contents = nullptr;
    if (void* mb = gpu::alloc(bytes, &contents)) {
      s.native = mb;
      s.ptr = contents;
      s.buf = std::shared_ptr<void>(mb, [bytes, contents](void* p) {
        gpu::release(p, bytes, contents);
      });
    } else {
      return make_heap_(n, dt);
    }
    return s;
  }

  static storage make_heap_(int64_t n, dtype dt = dtype::f32) {
    storage s;
    s.size = n;
    s.dt = dt;
    size_t bytes = static_cast<size_t>(n > 0 ? n * dtype_size(dt) : 4);
    auto* p = new unsigned char[bytes];
    s.ptr = reinterpret_cast<float*>(p);
    s.buf = std::shared_ptr<void>(
        p, [](void* q) { delete[] static_cast<unsigned char*>(q); });
    return s;
  }
};

namespace detail {

// Runtime hooks (see the storage comment). Null until installed.
inline storage (*storage_make_hook)(int64_t) = nullptr;
inline void (*cpu_barrier_hook)() = nullptr;
// Host↔device coherence (CUDA device-mirror). Called with (native, for_write)
// before a CPU read/write of a managed buffer to pull the device copy back
// (D2H) and, on write, invalidate it. Null / no-op on unified backends (Metal).
inline void (*host_sync_hook)(void*, bool) = nullptr;
// GPU-pipeline query for the eager-tiny decision in the graph builders.
// Behind a hook (not a direct gpu::pending() call) so those always-live
// builders reference no Metal symbol in a no-tensor binary — null means no
// GPU backend is installed, i.e. nothing is ever pending.
inline bool (*gpu_pending_hook)() = nullptr;

}  // namespace detail

inline storage storage::make(int64_t n, dtype dt) {
  // The embedder hook predates dtype and is F32-only (its signature is part of
  // the culebra seam); bf16 allocations go straight to the device path, which
  // heap-falls-back exactly like the hookless build.
  if (dt != dtype::f32) return make_device_(n, dt);
#ifdef TL_RUNTIME_HOOKS
  if (detail::storage_make_hook) return detail::storage_make_hook(n);
  return make_heap_(n);
#else
  return make_device_(n);
#endif
}

}  // namespace tl
