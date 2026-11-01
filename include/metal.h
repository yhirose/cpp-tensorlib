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
//     graph evaluation finishes or a CPU-side read needs the data. Under a
//     tl::profile each dispatch is committed as its own buffer instead, so
//     its GPU time can be read back.
//   - Kernels JIT-compile once from the #embed'd MSL source on first GPU
//     dispatch. Editing metal_kernels.metal requires rebuilding the host.
//
// The whole header is gated on __APPLE__: elsewhere it declares nothing, and
// gpu.h selects another backend (or gpu_null.h).

#include <cstdint>

#include "gpu_abi.h"

#ifdef __APPLE__

#include <objc.h>
#include <profile.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void* MTLCreateSystemDefaultDevice(void);
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

namespace tl {
namespace metal {

// The op vocabulary is the shared layer's (gpu_abi.h); the names stay reachable
// as metal::kop for the code that spelled them that way.
using kop = gpu::kop;
using cmp_op = gpu::cmp_op;
using unary_ext_op = gpu::unary_ext_op;
using scalar_op = gpu::scalar_op;

// The f32 GEMM's kernel choice: no other backend has these names, so they
// stay out of the shared kop vocabulary. context::pso_id_ caches these
// alongside a kop's pipelines, keyed negative (gtile_key_) so the two id
// spaces never collide.
enum class gtile {
  sgemm32, sgemm32x64, sgemm64x32, sgemm64,
  steel, steel32x64, steel_ta, steel_tb, steel32x64_ta, steel32x64_tb,
};
constexpr int gtile_key_(gtile t) { return -1 - static_cast<int>(t); }
inline const char* gtile_name_(gtile t) {
  switch (t) {
    case gtile::sgemm32: return "sgemm_32_";
    case gtile::sgemm32x64: return "sgemm_32x64_";
    case gtile::sgemm64x32: return "sgemm_64x32_";
    case gtile::sgemm64: return "sgemm_64_";
    case gtile::steel: return "sgemm_steel_";
    case gtile::steel32x64: return "sgemm_steel_32x64_";
    case gtile::steel_ta: return "sgemm_steel_ta_";
    case gtile::steel_tb: return "sgemm_steel_tb_";
    case gtile::steel32x64_ta: return "sgemm_steel_32x64_ta_";
    case gtile::steel32x64_tb: return "sgemm_steel_32x64_tb_";
  }
  return "";
}

// The attention kernels, by kind, head width and the KV cache's element type
// (f32, or f16 with caps::kv_f16): also no other backend's, so also kept out of
// kop. attn_kernel_ names the instantiated ones and is the one place that says
// which exist: an op asks it, and refuses a combination it has no name for.
// Keyed from -101, far enough past gtile_key_'s range that the two never
// collide either.
enum class attn_op { prefill, decode, decode_split, combine, bwd_dq, bwd_dkv, count_ };
struct attn_kernel {
  int key;
  const char* name;  // nullptr: not instantiated
  explicit operator bool() const { return name != nullptr; }
};
// The table's shape, which also sets the keys' range: op, head width (64 / 128
// / 256), cache (f32 / f16).
constexpr int attn_ops_ = static_cast<int>(attn_op::count_), attn_widths_ = 3,
              attn_caches_ = 2;
constexpr int attn_key_last_ = -101 - (attn_ops_ * attn_widths_ * attn_caches_ - 1);
inline attn_kernel attn_kernel_(attn_op op, int64_t D, bool f16) {
  static const char* const names[attn_ops_][attn_widths_][attn_caches_] = {
      {{"attn_prefill_64_", "attn_prefill_64_f16_"},
       {"attn_prefill_128_", "attn_prefill_128_f16_"},
       {nullptr, "attn_prefill_256_f16_"}},
      {{"attn_decode_64_", "attn_decode_64_f16_"},
       {"attn_decode_128_", "attn_decode_128_f16_"},
       {"attn_decode_256_", "attn_decode_256_f16_"}},
      {{"attn_decode_split_64_", "attn_decode_split_64_f16_"},
       {"attn_decode_split_128_", "attn_decode_split_128_f16_"},
       {"attn_decode_split_256_", "attn_decode_split_256_f16_"}},
      {{"attn_combine_64_", nullptr},
       {"attn_combine_128_", nullptr},
       {"attn_combine_256_", nullptr}},
      {{"attn_bwd_dq_64_", nullptr}, {"attn_bwd_dq_128_", nullptr}, {nullptr, nullptr}},
      {{"attn_bwd_dkv_64_", nullptr}, {"attn_bwd_dkv_128_", nullptr}, {nullptr, nullptr}},
  };
  const int w = D == 64 ? 0 : D == 128 ? 1 : D == 256 ? 2 : -1;
  if (w < 0) return {0, nullptr};
  const int o = static_cast<int>(op), c = f16 ? 1 : 0;
  return {-101 - ((o * attn_widths_ + w) * attn_caches_ + c), names[o][w][c]};
}
static_assert(gtile_key_(gtile::steel32x64_tb) > -101,
              "gtile and attn_op pipeline keys must stay in disjoint ranges");

struct mtl_size {
  unsigned long w, h, d;
};

struct context {
  objc::id device = nullptr;
  objc::id queue = nullptr;
  objc::id library = nullptr;
  objc::id cb = nullptr;   // open command buffer (null once committed)
  objc::id enc = nullptr;  // its compute encoder
  void* pool = nullptr;    // autorelease pool for the pending batch
  bool pending = false;
  // Free-list by byte size; contents pointers cached so a pool hit costs no
  // objc round trip (tiny-tensor workloads allocate per op). A buffer
  // released while a batch is pending goes back at once — the next owner's
  // kernels are encoded after the work still queued against it, so the
  // device keeps the order — but that work may yet read or write it, so it
  // carries the batch and a buffer the host fills skips it (alloc's
  // host_fill) until the flush moves `batch` on.
  struct pooled {
    void* buf;
    float* contents;
    uint64_t released_in;  // the pending batch it was released under, or 0
  };
  std::unordered_map<int64_t, std::vector<pooled>> free_bufs;
  uint64_t batch = 1;  // the batch being encoded; the flush advances it
  std::unordered_map<int, objc::id> psos;
  // The kernel the encoder currently has bound (set by bind_): what
  // tl::profile names the next dispatch.
  const char* bound = nullptr;
  // Grow-on-demand scratch (the split-K GEMV's partials, split-KV attention's),
  // kept across calls so a decode loop allocates once. Freed with the context,
  // which is leaked.
  void* scratch = nullptr;
  float* scratch_contents = nullptr;
  int64_t scratch_bytes = 0;
  // argmax's buffer: each group's value, then each group's index, then the
  // final (value, index) pair, whose index argmax reads back after a flush.
  void* argmax_res = nullptr;
  float* argmax_res_contents = nullptr;

  // Command buffers committed but not yet waited on (flush waits the last).
  // Under a profile there is one per dispatch, with the row owed its GPU
  // time; the row is null for an untimed buffer.
  std::vector<std::pair<objc::id, profile::row*>> committed;

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
        , '\n',
#embed "metal_kernels_nn.metal"
        , '\n',
#embed "metal_kernels_llm.metal"
        , '\0'};
    return src;
  }

  static const char* kernel_name_(kop op) {
    switch (op) {
      case kop::add: return "add_";
      case kop::sub: return "sub_";
      case kop::mul: return "mul_";
      case kop::div: return "div_";
      case kop::pow_: return "pow_";
      case kop::badd: return "badd_";
      case kop::bsub: return "bsub_";
      case kop::bmul: return "bmul_";
      case kop::bdiv: return "bdiv_";
      case kop::bpow: return "bpow_";
      case kop::exp_: return "exp_";
      case kop::log_: return "log_";
      case kop::sqrt_: return "sqrt_";
      case kop::sigmoid: return "sigmoid_";
      case kop::relu: return "relu_";
      case kop::affine: return "affine_";
      case kop::softmax: return "softmax_";
      case kop::row_sum: return "row_sum_";
      case kop::row_max: return "row_max_";
      case kop::pad: return "pad_";
      case kop::fold: return "fold_";
      case kop::index_select: return "index_select_";
      case kop::index_add: return "index_add_";
      case kop::scatter_axis: return "scatter_axis_";
      case kop::badd_nd: return "badd_nd_";
      case kop::bsub_nd: return "bsub_nd_";
      case kop::bmul_nd: return "bmul_nd_";
      case kop::bdiv_nd: return "bdiv_nd_";
      case kop::bpow_nd: return "bpow_nd_";
      case kop::where_nd: return "where_nd_";
      case kop::copy_nd: return "copy_nd_";
      case kop::gt_: return "gt_";
      case kop::lt_: return "lt_";
      case kop::ge_: return "ge_";
      case kop::le_: return "le_";
      case kop::eq_: return "eq_";
      case kop::ne_: return "ne_";
      case kop::tanh_: return "tanh_";
      case kop::sin_: return "sin_";
      case kop::cos_: return "cos_";
      case kop::clamp_: return "clamp_";
      case kop::sum_to_: return "sum_to_";
      case kop::sum_to_blocked_: return "sum_to_blocked_";
      case kop::concat_part_: return "concat_part_";
      case kop::rope_: return "rope_";
      case kop::pow_s_: return "pow_s_";
      case kop::gt_s_: return "gt_s_";
      case kop::lt_s_: return "lt_s_";
      case kop::ge_s_: return "ge_s_";
      case kop::le_s_: return "le_s_";
      case kop::eq_s_: return "eq_s_";
      case kop::ne_s_: return "ne_s_";
      case kop::layer_norm_: return "layer_norm_";
      case kop::layer_norm_bwd_dx_: return "layer_norm_bwd_dx_";
      case kop::layer_norm_bwd_gb_: return "layer_norm_bwd_gb_";
      case kop::layer_norm_bwd_gb_fold_: return "layer_norm_bwd_gb_fold_";
      case kop::ssm_conv_: return "ssm_conv_";
      case kop::gated_delta_: return "gated_delta_";
      case kop::kv_append_: return "kv_append_";
      case kop::kv_append_f16_: return "kv_append_f16_";
      case kop::kv_fill_: return "kv_fill_";
      case kop::kv_fill_f16_: return "kv_fill_f16_";
      case kop::argmax_: return "argmax_";
      case kop::rmsnorm_: return "rmsnorm_";
      case kop::add_rmsnorm_: return "add_rmsnorm_";
      case kop::swiglu_: return "swiglu_";
      case kop::split_heads_: return "split_heads_";
      case kop::merge_heads_: return "merge_heads_";
      case kop::gemv_f32_: return "gemv_f32_";
      case kop::gemv_bf16_: return "gemv_bf16_";
      case kop::gemv_q4_: return "gemv_q4_";
      case kop::gemv_combine_: return "gemv_combine_";
      case kop::gemv_bf16_row_: return "gemv_bf16_row_";
      case kop::gemm_bf16_nt_: return "gemm_bf16_nt_";
      case kop::gemm_bf16_nt32_: return "gemm_bf16_nt32_";
      case kop::gather_axis_: return "gather_axis_";
      case kop::row_logsumexp_: return "row_logsumexp_";
      case kop::xent_bwd_: return "xent_bwd_";
      case kop::adam_step_: return "adam_step_";
    }
    return "";
  }

  // Every op binds its pipeline through here: the pipeline for `op` on the
  // pending encoder, opened if there is none.
  void bind_(kop op) { bind_id_(static_cast<int>(op), kernel_name_(op)); }
  void bind_(gtile t) { bind_id_(gtile_key_(t), gtile_name_(t)); }
  void bind_(attn_kernel k) { bind_id_(k.key, k.name); }

  void bind_id_(int id, const char* name) {
    objc::id pso = pso_id_(id, name);
    if (cb && profile::active()) commit_(nullptr);  // untimed dispatches
    ensure_encoder_();
    objc::send(enc, "setComputePipelineState:", pso);
    bound = name;
  }

  // psos is one cache for every family: a kop's id is never negative, gtile's
  // (gtile_key_) counts down from -1, attn_kernel_'s from -101,
  // metal_nn.h's nn_kernel from -201 -- gaps wide enough that no range reaches
  // the next (static_asserts here and in metal_nn.h check them).
  objc::id pso_id_(int id, const char* name) {
    auto it = psos.find(id);
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
    auto oname = objc::send(objc::cls("NSString"), "stringWithUTF8String:",
                            name);
    auto fn = objc::send(library, "newFunctionWithName:", oname);
    objc::id err = nullptr;
    auto pso = objc::send(device, "newComputePipelineStateWithFunction:error:",
                          fn, &err);
    if (!pso) {
      throw std::runtime_error("tl::metal: PSO creation failed for " +
                               std::string(name) + ": " + objc::error_str(err));
    }
    psos[id] = pso;
    return pso;
  }

  void ensure_encoder_() {
    if (cb) return;
    if (!pool) pool = objc_autoreleasePoolPush();
    cb = objc::send(queue, "commandBuffer");
    enc = objc::send(cb, "computeCommandEncoder");
    pending = true;
  }

  // End and commit the open command buffer without waiting, owed `r`'s
  // device time; the batch stays pending until flush() waits for it.
  void commit_(profile::row* r) {
    objc::send(enc, "endEncoding");
    objc::send(cb, "commit");
    committed.emplace_back(cb, r);
    cb = enc = nullptr;
  }
};

inline bool available() { return context::get().device != nullptr; }

// What the shared launch policy (gpu_abi.h) may assume of this backend's
// kernels.
struct traits {
  // A [rows, cols] elementwise kernel reads its cell from a 2-D thread
  // position rather than a flat index.
  static constexpr bool cells_2d = true;
  // Each launch's tl::profile row carries a device time.
  static constexpr bool times_launches = true;
  // Threadgroups that keep the GPU busy: ~4 per core of a 16-core Apple GPU,
  // the target the GEMV and attention splits below measure their grid against.
  static constexpr int64_t fill_groups = 64;
};

// The ops this backend runs its own way: a different algorithm, several
// kernels, or a kernel whose ABI is its own. gpu_ops.h forwards to whichever of
// these exist (TL_GPU_DETECT_OWN) and answers false for the rest, so a backend
// declares what it has and nothing else. Defined below, among their helpers;
// the nn layer's own ops are in metal_nn.h, and `own` at the end of this file
// joins the two.
struct own_core {
  static bool argmax(gpu::span a, int64_t n, int64_t* out_idx);
  static bool argmax_to(gpu::span a, int64_t n, gpu::span idx);
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
};

inline bool pending() { return context::get().pending; }

// End the batch: commit and block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  if (c.cb) c.commit_(nullptr);
  {
    // One queue runs its command buffers in commit order: the last one done
    // is all of them done.
    profile::detail::blocked waiting;
    objc::send(c.committed.back().first, "waitUntilCompleted");
  }
  // A row is owed its time even if the profile stopped before this flush:
  // the rows live until the next start(), which drains first.
  const bool profiling = profile::active();
  for (auto [cb, row] : c.committed) {
    if (!row && !profiling) continue;  // nothing would record it
    const double s = objc::send<double>(cb, "GPUStartTime");
    const double e = objc::send<double>(cb, "GPUEndTime");
    if (e <= s) continue;
    profile::detail::batch_device((e - s) * 1e6);
    if (row) profile::detail::device_time(row, (e - s) * 1e6);
  }
  c.committed.clear();
  c.batch++;
  objc_autoreleasePoolPop(c.pool);
  c.pool = nullptr;
  c.pending = false;
}

// Pooled shared-mode MTLBuffer. Returns null when no device (caller falls
// back to heap). `bytes` is the pool key — pass the same value to release.
// `host_fill`: the host writes the buffer before any kernel does (a tensor
// made from host values), so it must not be one the pending batch may still
// read or write — on unified memory the host's bytes would race that work.
inline void* alloc(int64_t bytes, float** contents, bool host_fill = false) {
  auto& c = context::get();
  if (!c.device) return nullptr;
  auto it = c.free_bufs.find(bytes);
  if (it != c.free_bufs.end()) {
    auto& list = it->second;
    for (auto at = list.rbegin(); at != list.rend(); ++at) {
      if (host_fill && at->released_in == c.batch) continue;
      void* buf = at->buf;
      *contents = at->contents;
      list.erase(std::next(at).base());
      return buf;
    }
  }
  // MTLResourceStorageModeShared = 0
  void* buf = objc::send(c.device, "newBufferWithLength:options:",
                         static_cast<unsigned long>(bytes), 0ul);
  if (!buf) return nullptr;
  *contents = static_cast<float*>(objc::send(buf, "contents"));
  return buf;
}

inline void release(void* buf, int64_t bytes, float* contents) {
  auto& c = context::get();
  c.free_bufs[bytes].push_back({buf, contents, c.pending ? c.batch : 0});
}

// Stage `n` host floats into a device buffer. Unified memory has nothing to
// transfer, but the same hazard `host_fill` covers applies: a queued kernel may
// still be reading these bytes, so the batch drains before the store. Which is
// why `src` must be the caller's own memory here rather than the buffer's
// contents (a shortcut CUDA's mirror allows): writing those first would be the
// race this drain exists to prevent.
inline void upload(void* native, const float* src, int64_t n) {
  if (!native || n <= 0) return;
  if (pending()) flush();
  auto* dst = static_cast<float*>(objc::send(native, "contents"));
  if (dst && src != dst) std::memcpy(dst, src, (size_t)n * sizeof(float));
}

inline void upload_bytes(void* native, int64_t off, const void* src, int64_t bytes) {
  if (!native || bytes <= 0) return;
  if (pending()) flush();
  auto* dst = static_cast<char*>(objc::send(native, "contents"));
  if (dst) std::memcpy(dst + off, src, (size_t)bytes);
}

namespace detail_ {
// The kernel scratch, grown to `bytes` (see context::scratch).
inline void* scratch_(int64_t bytes) {
  auto& c = context::get();
  if (bytes > c.scratch_bytes) {
    if (c.scratch) release(c.scratch, c.scratch_bytes, c.scratch_contents);
    c.scratch = alloc(bytes, &c.scratch_contents);
    c.scratch_bytes = c.scratch ? bytes : 0;
  }
  return c.scratch;
}

// The context's argmax buffer (see context::argmax_res), allocated on first
// use (null without a device or memory).
inline gpu::span argmax_res_() {
  auto& c = context::get();
  constexpr int64_t G = traits::fill_groups;
  if (c.device && !c.argmax_res) c.argmax_res = alloc(8 * (G + 1), &c.argmax_res_contents);
  return {c.argmax_res, 0};
}
}  // namespace detail_

namespace detail_ {

// Every dispatch goes through here: the one place a launch is counted.
inline void dispatch_grid_(objc::id enc, mtl_size grid, mtl_size tg) {
  using fn = void (*)(objc::id, objc::sel_t, mtl_size, mtl_size);
  reinterpret_cast<fn>(objc_msgSend)(
      enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
      grid, tg);
  // Under a profile the launch is its own command buffer, owed the row's
  // GPU time at the flush.
  auto& c = context::get();
  if (profile::row* r = gpu::launched(c.bound ? c.bound : "?")) {
    c.commit_(r);
    profile::detail::drain_hook = &flush;
  }
}

}  // namespace detail_

// The device core's one way to run a kernel (gpu_abi.h has the contract):
// view i goes to buffer index i at its byte offset, the params follow at index
// n, and the grid is threadgroups x threads. Encodes without committing.
// Unified memory, so an arg's access says nothing this backend has to act on.
inline bool dispatch(kop k, const gpu::arg* args, size_t n, const void* params,
                     size_t params_bytes, const gpu::grid& g) {
  auto& c = context::get();
  if (!c.device || !*context::kernel_name_(k)) return false;
  c.bind_(k);
  for (size_t i = 0; i < n; i++) {
    objc::send(c.enc, "setBuffer:offset:atIndex:", args[i].s.buf,
               static_cast<unsigned long>(args[i].s.off),
               static_cast<unsigned long>(i));
  }
  objc::send(c.enc, "setBytes:length:atIndex:", params,
             static_cast<unsigned long>(params_bytes),
             static_cast<unsigned long>(n));
  detail_::dispatch_grid_(c.enc, {g.gx, g.gy, g.gz}, {g.tx, g.ty, g.tz});
  return true;
}

namespace detail_ {
}  // namespace detail_

namespace detail_ {

struct gemm_params {
  uint32_t M, N, K, lda, ldb, trans_a, trans_b;
  uint32_t a_fast, b_fast;  // float4 loader eligibility, verified host-side
  float scale, offset;
  uint32_t sa, sb;  // batch slices' strides in floats; C's is M * N
  uint32_t ldc;     // C's row stride for STEEL (N but for a GEMM into columns)
};

inline void set_buf_(objc::id enc, void* buf, int64_t off, unsigned long idx) {
  objc::send(enc, "setBuffer:offset:atIndex:", buf,
             static_cast<unsigned long>(off), idx);
}
inline void set_buf_(objc::id enc, gpu::span s, unsigned long idx) {
  set_buf_(enc, s.buf, s.off, idx);
}

// Whether a kernel may read `s` more than two bytes at a time. On the M1 Pro
// a load or store of 4 bytes or more is right from any 4-byte-aligned address
// (a float4 off 16 bytes included), but from a 2-byte-aligned one it can
// silently move the wrong bytes, and the compiler may merge two narrower loads
// into such a one. Apple asks only that a buffer's offset be a multiple of the
// bound argument's element, so an f32 buffer always passes; a 1- or 2-byte one
// (bf16, quantized blocks) read wider than its element needs this check.
inline bool word_aligned_(gpu::span s) { return s.off % 4 == 0; }

template <class P>
inline void set_bytes_(objc::id enc, const P& p, unsigned long idx) {
  objc::send(enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), idx);
}

// STEEL's tiling of an m x n product: 64-wide column tiles in bands of 32 rows
// below m 97 (silarray's band, more blocks for a short product) or 64, and a
// threadgroup swizzle for L2 reuse whose log rides in gemm_params' a_fast.
struct steel_grid {
  bool band32;
  uint32_t swizzle_log;
  unsigned long gx, gy;
};
inline steel_grid steel_grid_(int64_t m, int64_t n) {
  const bool band32 = m < 97;
  const unsigned long bm = band32 ? 32 : 64, bn = 64;
  const unsigned long tiles_n = (static_cast<unsigned long>(n) + bn - 1) / bn;
  const unsigned long tiles_m = (static_cast<unsigned long>(m) + bm - 1) / bm;
  uint32_t swizzle_log = 0;
  while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3) swizzle_log++;
  return {band32, swizzle_log, tiles_n << swizzle_log,
          (tiles_m + ((1ul << swizzle_log) - 1)) >> swizzle_log};
}

// The f32 gemm families' launch, STEEL's and the simple tiles': A, B and C at
// buffers 0-2, the params at 3, 128 threads a threadgroup, grid z the batch.
inline void gemm_launch_(context& c, gpu::span a, gpu::span b, gpu::span out,
                         const gemm_params& p, unsigned long gx,
                         unsigned long gy, unsigned long gz) {
  set_buf_(c.enc, a, 0ul);
  set_buf_(c.enc, b, 1ul);
  set_buf_(c.enc, out, 2ul);
  set_bytes_(c.enc, p, 3ul);
  dispatch_grid_(c.enc, {gx, gy, gz}, {128, 1, 1});
}

}  // namespace detail_

// `batch` products C(m,n) = (A @ B) * scale + offset in one launch: slice z
// (grid z) reads A / B z * sa / z * sb floats past slice 0's and writes C
// z * m * n past (a stride of 0 broadcasts one operand slice). lda/ldb are row
// strides; trans flags let a transposed view be read in place. Buffers are
// raw MTLBuffers; byte offsets fold the view offset in. Encodes without
// committing. No row bias yet: one declines.
inline bool own_core::gemm_batched(gpu::span a, int64_t lda, bool ta,
                                   int64_t sa, gpu::span b, int64_t ldb,
                                   bool tb, int64_t sb, gpu::span out,
                                   int64_t m, int64_t n, int64_t k,
                                   int64_t batch, float scale, float offset,
                                   gpu::span bias) {
  auto& c = context::get();
  if (!c.device || bias.buf || batch < 1 || sa < 0 || sb < 0 ||
      sa > UINT32_MAX || sb > UINT32_MAX) {
    return false;
  }
  // Dispatch ladder. STEEL (BN=64 bands) covers NN and single-transposed
  // shapes with enough width (the transposing loader reads the view in
  // place), tiled by steel_grid_. TT or narrow shapes take the simple-tile
  // family, which reads transposed views in place. Gates are provisional
  // pending a full census vs PyTorch-MPS.
  bool steel = !(ta && tb) && m >= 16 && n >= 48 && k >= 16;
  gtile kk_;
  uint32_t fast_a, fast_b;  // STEEL reuses the a_fast slot for swizzle_log
  unsigned long gx, gy;
  if (steel) {
    const detail_::steel_grid g = detail_::steel_grid_(m, n);
    kk_ = g.band32 ? (ta ? gtile::steel32x64_ta
                         : tb ? gtile::steel32x64_tb : gtile::steel32x64)
                   : (ta ? gtile::steel_ta : tb ? gtile::steel_tb : gtile::steel);
    fast_a = g.swizzle_log;
    fast_b = 0;
    gx = g.gx;
    gy = g.gy;
  } else {
    kk_ = m >= 64 ? gtile::sgemm64x32 : gtile::sgemm32;
    const unsigned long bm = m >= 64 ? 64 : 32, bn = 32;
    // float4 loader eligibility: row-major operand only (an f32 operand is
    // always word_aligned_; transposed operands use the strided path).
    fast_a = !ta ? 1u : 0u;
    fast_b = !tb ? 1u : 0u;
    gx = (static_cast<unsigned long>(n) + bn - 1) / bn;
    gy = (static_cast<unsigned long>(m) + bm - 1) / bm;
  }
  c.bind_(kk_);
  const detail_::gemm_params p{static_cast<uint32_t>(m),   static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k),   static_cast<uint32_t>(lda),
                         static_cast<uint32_t>(ldb), ta ? 1u : 0u,
                         tb ? 1u : 0u,               fast_a,
                         fast_b,                     scale,
                         offset,                     static_cast<uint32_t>(sa),
                         static_cast<uint32_t>(sb),  static_cast<uint32_t>(n)};
  detail_::gemm_launch_(c, a, b, out, p, gx, gy,
                        static_cast<unsigned long>(batch));
  return true;
}

// C(m,n) = (A @ B) * scale + offset: the batch == 1 case of gemm_batched.
inline bool own_core::gemm(gpu::span a, int64_t lda, bool ta, gpu::span b,
                      int64_t ldb, bool tb, gpu::span out, int64_t m, int64_t n,
                      int64_t k, float scale, float offset) {
  return gemm_batched(a, lda, ta, 0, b, ldb, tb, 0, out, m, n, k, 1, scale,
                      offset);
}

// Rank cap for pad_/fold_'s GPU dispatch — matches cuda.h's kPadFoldMaxRank
// and metal_kernels.metal's own copy (an MSL kernel can't see a host-side
// C++ constant), and bounds pad_fold_params' fixed-size arrays.
inline constexpr int kPadFoldMaxRank = 8;

namespace detail_ {
struct pad_fold_params {
  uint32_t out_shape[kPadFoldMaxRank];  // pad: length rank; fold: rank-1
  uint32_t a_shape[kPadFoldMaxRank];    // length rank
  uint32_t rank;
  uint32_t axis;
  int32_t shift;  // pad: `before`; fold: `step`
  uint32_t n;     // output element count (dispatch bound)
};

// Shared by pad()/fold() below: the two differ only in which kop to run and
// how many leading dims `out_shape` has (rank for pad, rank-1 for fold,
// since fold's own out has one fewer axis than `a`) — everything else, down
// to the dispatch grid, is identical.
inline bool dispatch_pad_fold_(kop op, void* a_native, int64_t ao,
                               void* out_native, int64_t oo,
                               const int64_t* a_shape,
                               const int64_t* out_shape, int rank,
                               int out_rank, int axis, int64_t shift,
                               int64_t out_n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  c.bind_(op);
  set_buf_(c.enc, a_native, ao, 0ul);
  set_buf_(c.enc, out_native, oo, 1ul);
  pad_fold_params p{};
  for (int d = 0; d < out_rank; d++)
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++)
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
  p.rank = static_cast<uint32_t>(rank);
  p.axis = static_cast<uint32_t>(axis);
  p.shift = static_cast<int32_t>(shift);
  p.n = static_cast<uint32_t>(out_n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(out_n) + 255ul) / 256ul;
  dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
}  // namespace detail_

// Gather-style pad/fold (im2col), mirroring kernels/tensorlib_webgpu.wgsl's
// pad/fold: one invocation per OUTPUT element reads (pad) or sums (fold)
// whatever cells of `a` map to it, so — unlike cuda.h's scatter+atomicAdd —
// no output cell is ever written by two invocations, and no pre-zeroed
// buffer or atomics are needed (Metal's device-memory atomics are int/uint
// only, the same gap WGSL has). `a` is required contiguous by array.h's
// gpu_pad_/gpu_fold_, so the metal_kernels.metal side derives a_strides from
// a_shape rather than have them uploaded. Encodes without committing, like
// every other dispatch above.
inline bool own_core::pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                     const int64_t* out_shape, int rank, int axis,
                     int64_t before, int64_t n, int64_t out_n) {
  (void)n;
  return detail_::dispatch_pad_fold_(kop::pad, a.buf, a.off, out.buf, out.off,
                                     a_shape, out_shape, rank, rank, axis,
                                     before, out_n);
}

// unfold's inverse: the gather twin of fold — see pad() above for why this
// is a gather rather than a scatter+atomicAdd. `out_shape` has rank-1 dims
// (fold's own out has one fewer axis than `a`); `a`'s last dim is the
// sliding window (size a_shape[rank-1]), and `a`'s `axis` dim is the window
// count.
inline bool own_core::fold(gpu::span a, gpu::span out, const int64_t* a_shape,
                      const int64_t* out_shape, int rank, int axis,
                      int64_t step, int64_t n, int64_t out_n) {
  (void)n;
  return detail_::dispatch_pad_fold_(kop::fold, a.buf, a.off, out.buf, out.off,
                                     a_shape, out_shape, rank, rank - 1, axis,
                                     step, out_n);
}

namespace detail_ {
struct index_add_params {
  uint32_t row_size;
  uint32_t k;
};
struct scatter_axis_params {
  uint32_t size;
  uint32_t n;
};

// Shared by index_select()/scatter_to_axis()/gather_from_axis() below: all
// three are a gather into `out` from two source buffers plus a small params
// blob, differing only in which kop/buffers/params struct they use and how
// `n` (the dispatch bound) is derived -- mirrors dispatch_pad_fold_ above,
// which extracts the same kind of shared tail for pad()/fold().
inline bool dispatch_gather3_(kop op, void* buf0, int64_t off0, void* buf1,
                              int64_t off1, void* out_native, int64_t oo,
                              const void* params, unsigned long params_size,
                              uint32_t n) {
  auto& c = context::get();
  if (!c.device) return false;
  c.bind_(op);
  set_buf_(c.enc, buf0, off0, 0ul);
  set_buf_(c.enc, buf1, off1, 1ul);
  set_buf_(c.enc, out_native, oo, 2ul);
  objc::send(c.enc, "setBytes:length:atIndex:", params, params_size, 3ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
}  // namespace detail_

// index_select's dual as a gather (float atomics would need MSL 3): each
// output row sums the source rows whose index matches it, in source order --
// no zeroing needed.
inline bool own_core::index_add(gpu::span idx, gpu::span values, gpu::span out,
                           int64_t row_size, int64_t k, int64_t out_n) {
  auto& c = context::get();
  if (!c.device || row_size <= 0) return false;
  // A threadgroup per output row (see index_add_ in the MSL).
  c.bind_(kop::index_add);
  detail_::set_buf_(c.enc, idx, 0ul);
  detail_::set_buf_(c.enc, values, 1ul);
  detail_::set_buf_(c.enc, out, 2ul);
  detail_::set_bytes_(c.enc,
                      detail_::index_add_params{static_cast<uint32_t>(row_size),
                                                static_cast<uint32_t>(k)},
                      3ul);
  detail_::dispatch_grid_(
      c.enc, {static_cast<unsigned long>(out_n / row_size), 1, 1}, {256, 1, 1});
  return true;
}

// One-hot scatter into a new trailing axis, as a gather: out[pos,k] =
// values[pos] where indices[pos] == k, else 0. Every output element reads,
// never writes twice, so — like index_select above — no zeroing needed.
inline bool own_core::scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                                 int64_t n, int64_t size) {
  detail_::scatter_axis_params p{static_cast<uint32_t>(size),
                                 static_cast<uint32_t>(n * size)};
  return detail_::dispatch_gather3_(kop::scatter_axis, idx.buf, idx.off,
                                    values.buf, values.off, out.buf, out.off, &p,
                                    sizeof(p), p.n);
}

// Cross-entropy's three: the trailing-axis gather, the one-pass row logsumexp
// and the pullback that reads it.

namespace detail_ {
}  // namespace detail_

namespace detail_ {
struct bcast_nd_params {
  uint32_t out_shape[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t b_strides[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t n;
  float scale;
  float offset;
};
struct where_nd_params {
  uint32_t out_shape[kPadFoldMaxRank];
  uint32_t c_strides[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t b_strides[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t n;
};
struct copy_nd_params {
  uint32_t out_shape[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t n;
};

// binary_bcast_nd's incoming `op` is one of the rank-2 kop values (badd etc,
// shared with binary_bcast() above -- array.h's gpu_binary_bcast_nd_ passes
// the same `bk` either kernel would take); map it to its own PSO/kernel name
// here rather than caching the N-D kernel under the rank-2 op's slot in
// context::psos, which pso_id_ keys by this same enum value.
inline kop to_nd_(kop op) {
  switch (op) {
    case kop::badd: return kop::badd_nd;
    case kop::bsub: return kop::bsub_nd;
    case kop::bmul: return kop::bmul_nd;
    case kop::bdiv: return kop::bdiv_nd;
    case kop::bpow: return kop::bpow_nd;
    default: return op;
  }
}
}  // namespace detail_

// N-D broadcast binary: generalizes binary_bcast() above to any rank (a
// Transformer's [N,S,D] LayerNorm broadcasting a [N,S,1] mean, rank 3).
// a_strides/b_strides are the broadcast strides (0 on a broadcast axis)
// array.h computes host-side via the same broadcast_strides() the CPU
// oracle uses -- mirrors cuda.h's own binary_bcast_nd exactly.
inline bool own_core::binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
                                 gpu::span b, const int64_t* b_strides,
                                 gpu::span out, const int64_t* out_shape,
                                 int rank, int64_t n, float scale,
                                 float offset) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  c.bind_(detail_::to_nd_(op));
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, b, 1ul);
  detail_::set_buf_(c.enc, out, 2ul);
  detail_::bcast_nd_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.b_strides[d] = static_cast<uint32_t>(b_strides[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.n = static_cast<uint32_t>(n);
  p.scale = scale;
  p.offset = offset;
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch. Same flat-index
// decode as binary_bcast_nd above, one more operand -- mirrors cuda.h's own
// where_nd exactly.
inline bool own_core::where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
                          const int64_t* a_strides, gpu::span b,
                          const int64_t* b_strides, gpu::span out,
                          const int64_t* out_shape, int rank, int64_t n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  c.bind_(kop::where_nd);
  detail_::set_buf_(c.enc, cond, 0ul);
  detail_::set_buf_(c.enc, a, 1ul);
  detail_::set_buf_(c.enc, b, 2ul);
  detail_::set_buf_(c.enc, out, 3ul);
  detail_::where_nd_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
    p.c_strides[d] = static_cast<uint32_t>(c_strides[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.b_strides[d] = static_cast<uint32_t>(b_strides[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 4ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// clone()'s device arm for a strided view: a gather into a contiguous output,
// where_nd's decode with one operand -- mirrors cuda.h's own copy_nd. The
// .metal banner says why a clone must not go through the host here.
inline bool own_core::copy_nd(gpu::span a, const int64_t* a_strides, gpu::span out,
                         const int64_t* out_shape, int rank, int64_t n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  c.bind_(kop::copy_nd);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::copy_nd_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
namespace detail_ {
struct sum_to_params {
  uint32_t a_shape[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t acc[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t out_n;
  uint32_t reduced_n;
};
}  // namespace detail_

// sum_to (un-broadcast a gradient): gather, mirrors cuda.h's tl_sum_to --
// one thread per OUTPUT element sums every `a` element that broadcasts
// onto it, so no atomics (unlike index_add).
inline bool own_core::sum_to(gpu::span a, const int64_t* a_shape,
                        const int64_t* a_strides, const int64_t* acc, int rank,
                        int64_t out_n, int64_t reduced_n, gpu::span out) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  // A deep reduction (a bias gradient sums its column over every row) earns a
  // threadgroup per output, as on CUDA; a shallow one keeps a thread per output.
  const bool blocked = reduced_n >= 64;
  c.bind_(blocked ? kop::sum_to_blocked_ : kop::sum_to_);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::sum_to_params p{};
  for (int d = 0; d < rank; d++) {
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.acc[d] = static_cast<uint32_t>(acc[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.out_n = static_cast<uint32_t>(out_n);
  p.reduced_n = static_cast<uint32_t>(reduced_n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  const auto un = static_cast<unsigned long>(out_n);
  detail_::dispatch_grid_(c.enc, {blocked ? un : (un + 255ul) / 256ul, 1, 1},
                          {256, 1, 1});
  return true;
}
namespace detail_ {
struct concat_part_params {
  uint32_t out_strides[kPadFoldMaxRank];
  uint32_t a_shape[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t shift;
  uint32_t n;
};
}  // namespace detail_

// concat_part (Tensor.concat along an arbitrary axis, KV-cache append):
// writes `a` (contiguous, this part) into `out` at `shift` (already
// before*out_strides[axis], a flat element offset) -- no zeroing, no
// bounds check, since concat's parts exhaustively cover `out` with no
// border. cuda.h's own concat_part reuses its tl_pad kernel body for this
// (writing a source at an axis-shifted offset is exactly what pad already
// does per element), but this file's pad_ above is a gather dispatched
// over OUTPUT elements (needed for pad's zero border) rather than the much
// smaller SOURCE (this part's own) element count concat wants, so this
// gets its own small kernel instead of reusing pad_'s PSO.
inline bool own_core::concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
                             const int64_t* out_shape, int rank, int axis,
                             int64_t before, int64_t n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  int64_t out_strides[kPadFoldMaxRank];
  int64_t acc = 1;
  for (int d = rank - 1; d >= 0; d--) {
    out_strides[d] = acc;
    acc *= out_shape[d];
  }
  c.bind_(kop::concat_part_);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::concat_part_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_strides[d] = static_cast<uint32_t>(out_strides[d]);
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.shift = static_cast<uint32_t>(before * out_strides[axis]);
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// Two launches of argmax_ (the kernel says how they split the work): the
// groups' pairs and the final value in the buffer the context keeps, the index
// at idx.
inline bool own_core::argmax_to(gpu::span a, int64_t n, gpu::span idx) {
  constexpr int64_t G = traits::fill_groups;
  const gpu::span r = detail_::argmax_res_();
  if (!r) return false;
  struct params {
    uint32_t n, chunk, mapped;
  };
  const gpu::span pv = r, pi = r.at(4 * G), fv = r.at(8 * G);
  // Each group a contiguous run of at least a group's width, G at most.
  constexpr gpu::policy::split_rule rule{G, 0, 256, 1};
  const int64_t chunk = gpu::policy::split_chunk(n, gpu::policy::split_parts(1, n, rule), rule);
  const auto groups = static_cast<uint32_t>((n + chunk - 1) / chunk);
  const gpu::arg pass1[] = {gpu::in(a), gpu::in(a), gpu::out(pv), gpu::out(pi)};
  const params p1{static_cast<uint32_t>(n), static_cast<uint32_t>(chunk), 0};
  const gpu::arg pass2[] = {gpu::in(pv), gpu::in(pi), gpu::out(fv), gpu::out(idx)};
  const params p2{groups, groups, 1};
  return dispatch(kop::argmax_, pass1, 4, &p1, sizeof(p1), {groups, 1, 1, 256, 1, 1, 0}) &&
         dispatch(kop::argmax_, pass2, 4, &p2, sizeof(p2), {1, 1, 1, 256, 1, 1, 0});
}

// argmax_to into the context's buffer, read back after a flush.
inline bool own_core::argmax(gpu::span a, int64_t n, int64_t* out_idx) {
  constexpr int64_t G = traits::fill_groups;
  const gpu::span r = detail_::argmax_res_(), fi = r.at(8 * G + 4);
  if (!r || n > gpu::policy::max_f32_index || !argmax_to(a, n, fi)) return false;
  flush();
  *out_idx = static_cast<int64_t>(context::get().argmax_res_contents[fi.off / 4]);
  return true;
}

// What a model may ask of this backend beyond the kernel contract (gpu.h
// lists every backend's). No graph capture: a Metal command buffer is cheap to
// encode, so a decode step re-encodes each token.
struct caps {
  static constexpr bool graph_capture = false;
  static constexpr bool row_gemv = true;   // gemv_bf16_row: weights as [N,K]
  static constexpr bool bf16_gemm = true;  // gemm_bf16_nt: a bf16-weight GEMM
  static constexpr bool kv_f16 = true;     // an f16 KV cache, every head width and path
};

// The capture group, cuda.h's contracts: absent here, so each answers false
// or does nothing, and a model takes its host-position path. Outside the
// #if/#else like gemm_bias below: the answer is the same either way.
using graph_exec = void*;
inline bool graph_available() { return false; }
inline bool capture_begin() { return false; }
inline graph_exec capture_end() { return nullptr; }
inline bool graph_launch(graph_exec) { return false; }
inline void graph_destroy(graph_exec) {}
inline void upload_u32(void*, unsigned) {}
inline int64_t attn_dpos_partials_bytes(int64_t, int64_t, int64_t) { return 0; }

// Every CPU-side buffer read funnels through array::raw()/data(), which call
// this: one choke point makes mixed CPU/GPU graphs safe.
inline void cpu_barrier() {
  if (pending()) flush();
}

// Host↔device coherence hook (see cuda.h). Metal is genuinely unified memory —
// the CPU and GPU see the same MTLBuffer bytes — so there is nothing to copy;
// cpu_barrier (the flush) is the only synchronization needed. A no-op here keeps
// the eval seam backend-agnostic (the CUDA device-mirror does the real work).
inline void sync_to_host(void*, bool) {}

}  // namespace metal

// tl::gpu_available() and the tl::gpu facade (which backend this build uses)
// live in gpu.h — one place, so array.h's eval seam stays #ifdef-free.

}  // namespace tl

// The nn layer's own ops, built on the device core above.
#include "metal_nn.h"

namespace tl {
namespace metal {

// What gpu_ops.h forwards to: the own ops of every layer. TL_GPU_DETECT_OWN
// finds a member of either base. (No llm-layer own ops here: no graph capture.)
struct own : own_core, own_nn {};

}  // namespace metal
}  // namespace tl

#endif  // __APPLE__
