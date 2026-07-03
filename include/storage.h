#pragma once

#include <metal.h>

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
  float* ptr = nullptr;
  void* native = nullptr;      // MTLBuffer handle when Metal-backed
  int64_t size = 0;            // elements

  float* data() const { return ptr; }

  static storage make(int64_t n);

  // Device-preferred allocation (Metal pool with heap fallback). Referenced
  // directly in the default build; only via the installed hook under
  // TL_RUNTIME_HOOKS.
  static storage make_device_(int64_t n) {
    storage s;
    s.size = n;
    int64_t bytes = n > 0 ? n * 4 : 4;  // MTLBuffer length must be non-zero
    float* contents = nullptr;
    if (void* mb = metal::alloc(bytes, &contents)) {
      s.native = mb;
      s.ptr = contents;
      s.buf = std::shared_ptr<void>(mb, [bytes, contents](void* p) {
        metal::release(p, bytes, contents);
      });
    } else {
      return make_heap_(n);
    }
    return s;
  }

  static storage make_heap_(int64_t n) {
    storage s;
    s.size = n;
    auto* p = new float[static_cast<size_t>(n > 0 ? n : 1)];
    s.ptr = p;
    s.buf = std::shared_ptr<void>(
        p, [](void* q) { delete[] static_cast<float*>(q); });
    return s;
  }
};

namespace detail {

// Runtime hooks (see the storage comment). Null until installed.
inline storage (*storage_make_hook)(int64_t) = nullptr;
inline void (*cpu_barrier_hook)() = nullptr;
// GPU-pipeline query for the eager-tiny decision in the graph builders.
// Behind a hook (not a direct metal::pending() call) so those always-live
// builders reference no Metal symbol in a no-tensor binary — null means no
// GPU backend is installed, i.e. nothing is ever pending.
inline bool (*gpu_pending_hook)() = nullptr;

}  // namespace detail

inline storage storage::make(int64_t n) {
#ifdef TL_RUNTIME_HOOKS
  if (detail::storage_make_hook) return detail::storage_make_hook(n);
  return make_heap_(n);
#else
  return make_device_(n);
#endif
}

}  // namespace tl
