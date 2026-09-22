#pragma once

// The backend a build gets when no GPU backend's gate holds: no device, so
// every op declines and the evaluator runs on the CPU. It is also the whole of
// what gpu.h asks of a backend, with nothing in it — the list a new backend
// fills in: a device core (lifecycle, memory, one `dispatch`), what it runs its
// own way (`own`), and what the shared layer may assume of it (`traits`,
// `caps`, the graph-capture plumbing).

#include <cstddef>
#include <cstdint>

#include "gpu_abi.h"

namespace tl {
namespace null_gpu {

// ---- lifecycle
inline bool available() { return false; }
inline bool pending() { return false; }  // work encoded but not yet waited on
inline void flush() {}                   // submit it and block until it is done
inline void cpu_barrier() {}             // before any host read of a buffer

// ---- memory. `native` is this backend's handle for a buffer — whatever
// gpu::span::buf should carry — and `contents` the host-readable bytes behind
// it (the same memory where it is unified, a mirror where it is not).
inline void* alloc(int64_t, float** /*contents*/, bool /*host_fill*/ = false) {
  return nullptr;  // null: storage falls back to the heap
}
inline void release(void*, int64_t, float*) {}
inline void sync_to_host(void*, bool /*for_write*/) {}  // before a host access
inline void upload(void*, const float*, int64_t) {}     // stage host floats in

// ---- launch: the one way a shared op (gpu_ops.h) runs a kernel. View i is the
// kernel's i-th buffer at its byte offset, `params` a block of 4-byte fields
// in the kernel's argument order (gpu_abi.h). False for a kernel id this
// backend has no kernel for. Every launch a backend makes, here or in an own
// op, is one gpu::launched(kernel name) call: its row under tl::profile.
inline bool dispatch(gpu::kop, const gpu::arg*, size_t, const void* /*params*/,
                     size_t /*params_bytes*/, const gpu::grid&) {
  return false;
}

// ---- the ops this backend runs its own way, as static members with the
// signature the op has in gpu_ops.h. None.
struct own {};

// ---- what the shared launch policy may assume of this backend's kernels.
struct traits {
  static constexpr bool cells_2d = false;
  // Whether a launch's tl::profile row (gpu::launched, called from the
  // backend's launch primitive) carries a device time.
  static constexpr bool times_launches = false;
};

// ---- what a model may assume, and the graph-capture plumbing behind
// caps::graph_capture.
struct caps {
  static constexpr bool model_path = false;
  static constexpr bool graph_capture = false;
  static constexpr bool row_gemv = false;
  static constexpr bool bf16_gemm = false;
};
using graph_exec = void*;
inline bool graph_available() { return false; }
inline bool capture_begin() { return false; }
inline graph_exec capture_end() { return nullptr; }
inline bool graph_launch(graph_exec) { return false; }
inline void graph_destroy(graph_exec) {}
inline void upload_u32(void*, unsigned) {}
inline int64_t attn_dpos_partials_bytes(int64_t, int64_t, int64_t) { return 0; }

}  // namespace null_gpu
}  // namespace tl
