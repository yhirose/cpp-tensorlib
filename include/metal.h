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
  // argmax's one-int result, read back after a flush.
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
      case kop::sgemm32: return "sgemm_32_";
      case kop::sgemm32x64: return "sgemm_32x64_";
      case kop::sgemm64x32: return "sgemm_64x32_";
      case kop::sgemm64: return "sgemm_64_";
      case kop::steel: return "sgemm_steel_";
      case kop::steel32x64: return "sgemm_steel_32x64_";
      case kop::steel_ta: return "sgemm_steel_ta_";
      case kop::steel_tb: return "sgemm_steel_tb_";
      case kop::steel32x64_ta: return "sgemm_steel_32x64_ta_";
      case kop::steel32x64_tb: return "sgemm_steel_32x64_tb_";
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
      case kop::attn_prefill_64_: return "attn_prefill_64_";
      case kop::attn_prefill_128_: return "attn_prefill_128_";
      case kop::attn_prefill_bf16_64_: return "attn_prefill_bf16_64_";
      case kop::attn_prefill_bf16_128_: return "attn_prefill_bf16_128_";
      case kop::attn_bwd_dq_64_: return "attn_bwd_dq_64_";
      case kop::attn_bwd_dq_128_: return "attn_bwd_dq_128_";
      case kop::attn_bwd_dkv_64_: return "attn_bwd_dkv_64_";
      case kop::attn_bwd_dkv_128_: return "attn_bwd_dkv_128_";
      case kop::attn_decode_64_: return "attn_decode_64_";
      case kop::attn_decode_128_: return "attn_decode_128_";
      case kop::attn_decode_split_64_: return "attn_decode_split_64_";
      case kop::attn_decode_split_128_: return "attn_decode_split_128_";
      case kop::attn_combine_64_: return "attn_combine_64_";
      case kop::attn_combine_128_: return "attn_combine_128_";
      case kop::attn_decode_bf16_64_: return "attn_decode_bf16_64_";
      case kop::attn_decode_bf16_128_: return "attn_decode_bf16_128_";
      case kop::attn_decode_split_bf16_64_: return "attn_decode_split_bf16_64_";
      case kop::attn_decode_split_bf16_128_: return "attn_decode_split_bf16_128_";
      case kop::kv_append_: return "kv_append_";
      case kop::kv_append_bf16_: return "kv_append_bf16_";
      case kop::kv_fill_: return "kv_fill_";
      case kop::kv_fill_bf16_: return "kv_fill_bf16_";
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
  void bind_(kop op) {
    objc::id pso = pso_(op);
    if (cb && profile::active()) commit_(nullptr);  // untimed dispatches
    ensure_encoder_();
    objc::send(enc, "setComputePipelineState:", pso);
    bound = kernel_name_(op);
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

// The ops this backend runs its own way: a different algorithm, several
// kernels, or a kernel whose ABI is its own. gpu_ops.h forwards to whichever of
// these exist (TL_GPU_DETECT_OWN) and answers false for the rest, so a backend
// declares what it has and nothing else. Defined below, among their helpers.
struct own {
  static bool split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                          int64_t T, int64_t ld, int64_t off, int64_t H,
                          int64_t D);
  static bool argmax(gpu::span a, int64_t n, int64_t* out_idx);
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
  static bool gemv_f32(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                       int64_t k);
  static bool gemv_bf16(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                        int64_t k);
  static bool attn_decode(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                          int64_t n_q_heads, int64_t n_kv_heads, int64_t ctx,
                          int64_t kv_max, int64_t D, float scale,
                          bool kv_bf16 = false);
  static bool attn_prefill(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                           int64_t n_q_heads, int64_t n_kv_heads, int64_t T,
                           int64_t kv_max, int64_t D, float scale,
                           bool kv_bf16 = false, int64_t pos0 = 0);
  static bool attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span dO, gpu::span O, gpu::span dq,
                              gpu::span stats, int64_t H, int64_t T, int64_t D,
                              float scale);
  static bool attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                               gpu::span dO, gpu::span stats, gpu::span dK,
                               gpu::span dV, int64_t H, int64_t T, int64_t D,
                               float scale);
  static bool gemm_bf16_nt(gpu::span A, gpu::span B, gpu::span C, int64_t M,
                           int64_t N, int64_t K);
  static bool rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                   int64_t D, int64_t pos, float base, gpu::span bias = {});
  static bool layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                             gpu::span dx, gpu::span dg, gpu::span db,
                             gpu::span stats, gpu::span partials, int64_t rows,
                             int64_t cols, int64_t per_chunk, int64_t chunks,
                             float eps);
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
};

inline void set_buf_(objc::id enc, void* buf, int64_t off, unsigned long idx) {
  objc::send(enc, "setBuffer:offset:atIndex:", buf,
             static_cast<unsigned long>(off), idx);
}
inline void set_buf_(objc::id enc, gpu::span s, unsigned long idx) {
  set_buf_(enc, s.buf, s.off, idx);
}

template <class P>
inline void set_bytes_(objc::id enc, const P& p, unsigned long idx) {
  objc::send(enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), idx);
}

}  // namespace detail_

// C(m,n) = (A @ B) * scale + offset. lda/ldb are row strides; trans flags
// let a transposed view be read in place. Buffers are raw MTLBuffers; byte
// offsets fold the view offset in. Encodes without committing.
inline bool own::gemm(gpu::span a, int64_t lda, bool ta, gpu::span b,
                      int64_t ldb, bool tb, gpu::span out, int64_t m, int64_t n,
                      int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  // Dispatch ladder. STEEL (BN=64 bands) covers NN and single-transposed
  // shapes with enough width (the transposing loader reads the view in
  // place); BM band follows silarray (M < 97 → 32×64 tiles). TT or narrow
  // shapes take the simple-tile family, which reads transposed views in
  // place. Gates are provisional pending a full census vs PyTorch-MPS.
  bool steel = !(ta && tb) && m >= 16 && n >= 48 && k >= 16;
  kop kk_;
  unsigned long bm, bn;
  uint32_t fast_a, fast_b;  // STEEL reuses the a_fast slot for swizzle_log
  unsigned long gx, gy;
  if (steel) {
    bool band32 = m < 97;
    kk_ = band32 ? (ta ? kop::steel32x64_ta
                       : tb ? kop::steel32x64_tb : kop::steel32x64)
                 : (ta ? kop::steel_ta : tb ? kop::steel_tb : kop::steel);
    bm = band32 ? 32 : 64;
    bn = 64;
    unsigned long tiles_n = (static_cast<unsigned long>(n) + bn - 1) / bn;
    unsigned long tiles_m = (static_cast<unsigned long>(m) + bm - 1) / bm;
    uint32_t swizzle_log = 0;  // threadgroup swizzle for L2 reuse
    while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
      swizzle_log++;
    fast_a = swizzle_log;
    fast_b = 0;
    gx = tiles_n << swizzle_log;
    gy = (tiles_m + ((1ul << swizzle_log) - 1)) >> swizzle_log;
  } else {
    kk_ = m >= 64 ? kop::sgemm64x32 : kop::sgemm32;
    bm = m >= 64 ? 64 : 32;
    bn = 32;
    // float4 loader eligibility: row-major operand only (Apple GPUs handle
    // the unaligned vector loads; transposed operands use the strided path).
    fast_a = !ta ? 1u : 0u;
    fast_b = !tb ? 1u : 0u;
    gx = (static_cast<unsigned long>(n) + bn - 1) / bn;
    gy = (static_cast<unsigned long>(m) + bm - 1) / bm;
  }
  c.bind_(kk_);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, b, 1ul);
  detail_::set_buf_(c.enc, out, 2ul);
  detail_::gemm_params p{static_cast<uint32_t>(m),   static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k),   static_cast<uint32_t>(lda),
                         static_cast<uint32_t>(ldb), ta ? 1u : 0u,
                         tb ? 1u : 0u,               fast_a,
                         fast_b,                     scale,
                         offset};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  detail_::dispatch_grid_(c.enc, {gx, gy, 1}, {128, 1, 1});
  return true;
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
inline bool own::pad(gpu::span a, gpu::span out, const int64_t* a_shape,
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
inline bool own::fold(gpu::span a, gpu::span out, const int64_t* a_shape,
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
inline bool own::index_add(gpu::span idx, gpu::span values, gpu::span out,
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
inline bool own::scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
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
struct layer_norm_bwd_params {
  uint32_t rows, cols, rows_per_chunk, chunks;
  float eps;
};
}  // namespace detail_

// Layer norm's pullback: cuda.h's contract and its three launches -- a row
// kernel (dx and the row stats), a column-strip kernel, a fold.
inline bool own::layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                                gpu::span dx, gpu::span dg, gpu::span db,
                                gpu::span stats, gpu::span partials,
                                int64_t rows, int64_t cols, int64_t per_chunk,
                                int64_t chunks, float eps) {
  auto& c = context::get();
  if (!c.device || rows <= 0 || cols <= 0 || chunks <= 0) return false;
  detail_::layer_norm_bwd_params p{
      static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
      static_cast<uint32_t>(per_chunk), static_cast<uint32_t>(chunks), eps};
  const auto ur = static_cast<unsigned long>(rows);
  const auto uc = static_cast<unsigned long>(cols);
  const auto uk = static_cast<unsigned long>(chunks);

  c.bind_(kop::layer_norm_bwd_dx_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, g, 1ul);
  detail_::set_buf_(c.enc, dy, 2ul);
  detail_::set_buf_(c.enc, dx, 3ul);
  detail_::set_buf_(c.enc, stats, 4ul);
  detail_::set_bytes_(c.enc, p, 5ul);
  detail_::dispatch_grid_(c.enc, {ur, 1, 1}, {256, 1, 1});

  c.bind_(kop::layer_norm_bwd_gb_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, dy, 1ul);
  detail_::set_buf_(c.enc, stats, 2ul);
  detail_::set_buf_(c.enc, partials, 3ul);
  detail_::set_bytes_(c.enc, p, 4ul);
  detail_::dispatch_grid_(c.enc, {(uc + 31) / 32, uk, 1}, {32, 8, 1});

  c.bind_(kop::layer_norm_bwd_gb_fold_);
  detail_::set_buf_(c.enc, partials, 0ul);
  detail_::set_buf_(c.enc, dg, 1ul);
  detail_::set_buf_(c.enc, db, 2ul);
  detail_::set_bytes_(c.enc, p, 3ul);
  detail_::dispatch_grid_(c.enc, {(uc + 255) / 256, 1, 1}, {256, 1, 1});
  return true;
}

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
// context::psos, which pso_() keys by this same enum value.
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
inline bool own::binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
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
inline bool own::where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
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
inline bool own::copy_nd(gpu::span a, const int64_t* a_strides, gpu::span out,
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
inline bool own::sum_to(gpu::span a, const int64_t* a_shape,
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
inline bool own::concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
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

namespace detail_ {
struct rope_params {
  uint32_t T, D, pos;
  uint32_t half_;
  float base;
  uint32_t n;
  uint32_t has_bias;
};
}  // namespace detail_

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors tensorlib_cuda.cu's own tl_rope exactly. x is
// [rows, D] contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D] with
// T=1); row r's head-dim vector sits at position `pos + (r % T)`; pairs
// (j, j+D/2) rotate by angle = position * base^(-2j/D). `bias`, when given,
// is [rows, D] added before the rotation (a decode step's q/k bias folded
// in; array.h's gpu_rope_ never passes one). Dispatched flat over rows*(D/2)
// (CUDA instead grids by row, blocks by D/2 -- this file's own kernels are
// all flat-1D, same as sum_to/compare/unary_ext above).
inline bool own::rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                      int64_t D, int64_t pos, float base, gpu::span bias) {
  auto& c = context::get();
  if (!c.device || D <= 0 || (D & 1)) return false;
  int64_t half = D / 2;
  int64_t n = rows * half;
  c.bind_(kop::rope_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::set_buf_(c.enc, bias ? bias : x, 3ul);  // a slot must be bound
  detail_::rope_params p{};
  p.T = static_cast<uint32_t>(T);
  p.D = static_cast<uint32_t>(D);
  p.pos = static_cast<uint32_t>(pos);
  p.half_ = static_cast<uint32_t>(half);
  p.base = base;
  p.n = static_cast<uint32_t>(n);
  p.has_bias = bias.buf ? 1u : 0u;
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// ---- causal prefill attention and its pullback ------------------------------
// cuda.h's contracts. 128 threads per threadgroup, owning 32 rows (queries,
// or keys for the key/value half): the MSL kernels' 4 simdgroups × 8.

namespace detail_ {
struct attn_params {
  uint32_t T, kv_stride, group, pos0;
  float scale;
};

struct gemv_params {
  uint32_t n, k, chunk;
};

struct gemv_combine_params {
  uint32_t n, parts;
};

struct gemm_nt_params {
  uint32_t M, N, K;
};

struct attn_decode_params {
  uint32_t ctx, kv_stride, group, chunk;
  float scale;
};

struct attn_combine_params {
  uint32_t splits;
};

// Keys per split, or 0 for the single-pass kernel: one threadgroup a head
// leaves most of a 16-core GPU idle when a model has few heads, so cut the
// keys until there are enough threadgroups (cuda's attn_split_count).
inline unsigned long attn_split_chunk_(int64_t heads, int64_t ctx) {
  constexpr long kWantGroups = 64, kMinKeys = 128;
  if (heads <= 0 || heads >= kWantGroups || ctx < 2 * kMinKeys) return 0;
  long want = (kWantGroups + heads - 1) / heads;
  const long most = ctx / kMinKeys;
  if (want > most) want = most;
  if (want <= 1) return 0;
  return static_cast<unsigned long>((ctx + want - 1) / want);
}

inline void attn_dispatch_(objc::id enc, const attn_params& p,
                           unsigned long idx, int64_t heads, int64_t T) {
  constexpr unsigned long rows = 32;
  set_bytes_(enc, p, idx);
  dispatch_grid_(enc,
                 {static_cast<unsigned long>(heads),
                  (static_cast<unsigned long>(T) + rows - 1) / rows, 1},
                 {128, 1, 1});
}
}  // namespace detail_

// q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D] cache read over
// [0,pos0+T), query p at absolute position pos0+p. GQA via the head ratio.
// D∈{64,128}; the cache is f32, or bf16 with kv_bf16.
inline bool own::attn_prefill(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span out, int64_t n_q_heads,
                              int64_t n_kv_heads, int64_t T, int64_t kv_max,
                              int64_t D, float scale, bool kv_bf16,
                              int64_t pos0) {
  auto& c = context::get();
  if (!c.device || (D != 64 && D != 128)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 || T <= 0) return false;
  c.bind_(kv_bf16 ? (D == 64 ? kop::attn_prefill_bf16_64_
                             : kop::attn_prefill_bf16_128_)
                  : (D == 64 ? kop::attn_prefill_64_ : kop::attn_prefill_128_));
  detail_::set_buf_(c.enc, q, 0ul);
  detail_::set_buf_(c.enc, K, 1ul);
  detail_::set_buf_(c.enc, V, 2ul);
  detail_::set_buf_(c.enc, out, 3ul);
  detail_::attn_params p{static_cast<uint32_t>(T),
                         static_cast<uint32_t>(kv_max * D),
                         static_cast<uint32_t>(n_q_heads / n_kv_heads),
                         static_cast<uint32_t>(pos0), scale};
  detail_::attn_dispatch_(c.enc, p, 4ul, n_q_heads, T);
  return true;
}

// y[1,N] = a[1,K] · B[K,N] with f32 or bf16 weights, all contiguous: the
// decode projection. A narrow layer's N/256 threadgroups leave the GPU idle,
// so K is split across the grid's y and a combine pass sums the slices.
inline bool gemv_(kop op, gpu::span a, gpu::span B, gpu::span y, int64_t n,
                  int64_t k) {
  auto& c = context::get();
  if (!c.device || n <= 0 || k <= 0) return false;
  constexpr unsigned long NT = 256, kGroupsWanted = 64;
  const unsigned long cols = (static_cast<unsigned long>(n) + NT - 1) / NT;
  unsigned long parts = cols >= kGroupsWanted ? 1 : kGroupsWanted / cols;
  unsigned long chunk = (static_cast<unsigned long>(k) + parts - 1) / parts;
  chunk = (chunk + NT - 1) / NT * NT;  // whole `a` tiles
  parts = (static_cast<unsigned long>(k) + chunk - 1) / chunk;
  gpu::span out = y;
  if (parts > 1) {
    out = {detail_::scratch_(static_cast<int64_t>(parts) * n * 4), 0};
    if (!out) parts = 1, chunk = static_cast<unsigned long>(k), out = y;
  }
  c.bind_(op);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, B, 1ul);
  detail_::set_buf_(c.enc, out, 2ul);
  detail_::gemv_params p{static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                         static_cast<uint32_t>(chunk)};
  detail_::set_bytes_(c.enc, p, 3ul);
  detail_::dispatch_grid_(c.enc, {cols, parts, 1}, {NT, 1, 1});
  if (parts == 1) return true;
  c.bind_(kop::gemv_combine_);
  detail_::set_buf_(c.enc, out, 0ul);
  detail_::set_buf_(c.enc, y, 1ul);
  detail_::gemv_combine_params cp{static_cast<uint32_t>(n),
                                  static_cast<uint32_t>(parts)};
  detail_::set_bytes_(c.enc, cp, 2ul);
  detail_::dispatch_grid_(c.enc, {cols, 1, 1}, {NT, 1, 1});
  return true;
}

inline bool own::gemv_f32(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                          int64_t k) {
  return gemv_(kop::gemv_f32_, a, B, y, n, k);
}

inline bool own::gemv_bf16(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                           int64_t k) {
  return gemv_(kop::gemv_bf16_, a, B, y, n, k);
}

// C[M,N] = A[M,K] · B[N,K]ᵀ, B bf16: the batched prefill's projection, where
// one weight serves a whole chunk of prompt tokens. The 32-row tile for a
// short chunk (more M blocks to fill the GPU), the 64-row one otherwise.
inline bool own::gemm_bf16_nt(gpu::span A, gpu::span B, gpu::span C, int64_t M,
                              int64_t N, int64_t K) {
  auto& c = context::get();
  if (!c.device || M <= 0 || N <= 0 || K <= 0) return false;
  const unsigned long BM = M <= 64 ? 32 : 64;
  c.bind_(M <= 64 ? kop::gemm_bf16_nt32_ : kop::gemm_bf16_nt_);
  detail_::set_buf_(c.enc, A, 0ul);
  detail_::set_buf_(c.enc, B, 1ul);
  detail_::set_buf_(c.enc, C, 2ul);
  detail_::gemm_nt_params p{static_cast<uint32_t>(M), static_cast<uint32_t>(N),
                            static_cast<uint32_t>(K)};
  detail_::set_bytes_(c.enc, p, 3ul);
  const unsigned long gx = (static_cast<unsigned long>(N) + 63) / 64;
  const unsigned long gy = (static_cast<unsigned long>(M) + BM - 1) / BM;
  detail_::dispatch_grid_(c.enc, {gx, gy, 1}, {128, 1, 1});
  return true;
}

// One decode step: q [n_q_heads,D] against a [n_kv_heads,kv_max,D] cache read
// over [0,ctx). GQA via the head ratio; the cache is f32, or bf16 with kv_bf16.
inline bool own::attn_decode(gpu::span q, gpu::span K, gpu::span V,
                             gpu::span out, int64_t n_q_heads,
                             int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                             int64_t D, float scale, bool kv_bf16) {
  auto& c = context::get();
  if (!c.device || (D != 64 && D != 128)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 || ctx <= 0) return false;
  const bool d64 = D == 64;
  unsigned long chunk = detail_::attn_split_chunk_(n_q_heads, ctx);
  const unsigned long splits =
      chunk ? (static_cast<unsigned long>(ctx) + chunk - 1) / chunk : 1;
  gpu::span dst = out;
  if (splits > 1) {
    // pm[H*S] | pl[H*S] | pacc[H*S*D], the layout attn_combine_ reads.
    const int64_t hs = n_q_heads * static_cast<int64_t>(splits);
    dst = {detail_::scratch_((hs * 2 + hs * D) * 4), 0};
    if (!dst) chunk = 0, dst = out;
  }
  const bool split = chunk != 0;
  if (split) {
    c.bind_(kv_bf16 ? (d64 ? kop::attn_decode_split_bf16_64_
                           : kop::attn_decode_split_bf16_128_)
                    : (d64 ? kop::attn_decode_split_64_
                           : kop::attn_decode_split_128_));
  } else {
    c.bind_(kv_bf16 ? (d64 ? kop::attn_decode_bf16_64_
                           : kop::attn_decode_bf16_128_)
                    : (d64 ? kop::attn_decode_64_ : kop::attn_decode_128_));
  }
  detail_::set_buf_(c.enc, q, 0ul);
  detail_::set_buf_(c.enc, K, 1ul);
  detail_::set_buf_(c.enc, V, 2ul);
  detail_::set_buf_(c.enc, dst, 3ul);
  detail_::attn_decode_params p{static_cast<uint32_t>(ctx),
                                static_cast<uint32_t>(kv_max * D),
                                static_cast<uint32_t>(n_q_heads / n_kv_heads),
                                static_cast<uint32_t>(chunk), scale};
  detail_::set_bytes_(c.enc, p, 4ul);
  detail_::dispatch_grid_(c.enc,
                          {static_cast<unsigned long>(n_q_heads),
                           split ? splits : 1ul, 1},
                          {static_cast<unsigned long>(D), 1, 1});
  if (!split) return true;
  c.bind_(d64 ? kop::attn_combine_64_ : kop::attn_combine_128_);
  detail_::set_buf_(c.enc, dst, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::attn_combine_params cp{static_cast<uint32_t>(splits)};
  detail_::set_bytes_(c.enc, cp, 2ul);
  detail_::dispatch_grid_(c.enc, {static_cast<unsigned long>(n_q_heads), 1, 1},
                          {static_cast<unsigned long>(D), 1, 1});
  return true;
}

// ---- the KV cache's writes and the decode step's remaining kernels --------
// cuda.h's contracts: raw device buffers, so a model's step builds no graph.

// The query half of the pullback: q, K, V, dO, O and dq all [H,T,D]
// contiguous, `stats` [2,H,T] the row logsumexp and dO·O.
inline bool own::attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                                 gpu::span dO, gpu::span O, gpu::span dq,
                                 gpu::span stats, int64_t H, int64_t T,
                                 int64_t D, float scale) {
  auto& c = context::get();
  if (!c.device || (D != 64 && D != 128) || H <= 0 || T <= 0) return false;
  c.bind_(D == 64 ? kop::attn_bwd_dq_64_ : kop::attn_bwd_dq_128_);
  const gpu::span views[] = {q, K, V, dO, O, dq, stats};
  for (unsigned long i = 0; i < 7; i++) detail_::set_buf_(c.enc, views[i], i);
  detail_::attn_params p{static_cast<uint32_t>(T), 0, 0, 0, scale};
  detail_::attn_dispatch_(c.enc, p, 7ul, H, T);
  return true;
}

// The key/value half, reading the stats the call above wrote.
inline bool own::attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span dO, gpu::span stats, gpu::span dK,
                                  gpu::span dV, int64_t H, int64_t T, int64_t D,
                                  float scale) {
  auto& c = context::get();
  if (!c.device || (D != 64 && D != 128) || H <= 0 || T <= 0) return false;
  c.bind_(D == 64 ? kop::attn_bwd_dkv_64_ : kop::attn_bwd_dkv_128_);
  const gpu::span views[] = {q, K, V, dO, stats, dK, dV};
  for (unsigned long i = 0; i < 7; i++) detail_::set_buf_(c.enc, views[i], i);
  detail_::attn_params p{static_cast<uint32_t>(T), 0, 0, 0, scale};
  detail_::attn_dispatch_(c.enc, p, 7ul, H, T);
  return true;
}

// What a model may ask of this backend beyond the kernel contract (gpu.h
// lists every backend's). No graph capture: a Metal command buffer is cheap to
// encode, so a decode step re-encodes each token.

// "No bias" is a flag here: a kernel cannot test a buffer for null, so an
// absent bias binds src in its place and is never read.
inline bool own::split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                        int64_t T, int64_t ld, int64_t off, int64_t H,
                        int64_t D) {
  struct {
    uint32_t T, ld, off, has_bias;
  } p{static_cast<uint32_t>(T), static_cast<uint32_t>(ld),
      static_cast<uint32_t>(off), bias ? 1u : 0u};
  const gpu::arg args[] = {gpu::in(src), gpu::in(bias ? bias : src),
                           gpu::out(dst)};
  return dispatch(kop::split_heads_, args, 3, &p, sizeof(p),
                  gpu::policy::per_head(H, T, D));
}

// One int comes back through a 4-byte buffer the context keeps.
inline bool own::argmax(gpu::span a, int64_t n, int64_t* out_idx) {
  auto& c = context::get();
  if (!c.device) return false;
  if (!c.argmax_res) {
    c.argmax_res = alloc(4, &c.argmax_res_contents);
    if (!c.argmax_res) return false;
  }
  const uint32_t p = static_cast<uint32_t>(n);
  const gpu::arg args[] = {gpu::in(a), gpu::out({c.argmax_res, 0})};
  if (!dispatch(kop::argmax_, args, 2, &p, sizeof(p), {1, 1, 1, 256, 1, 1, 0})) {
    return false;
  }
  flush();
  *out_idx = *reinterpret_cast<const int*>(c.argmax_res_contents);
  return true;
}

// What the shared launch policy (gpu_ops.h) may assume of this backend's
// kernels.
struct traits {
  // A [rows, cols] elementwise kernel reads its cell from a 2-D thread
  // position rather than a flat index.
  static constexpr bool cells_2d = true;
  // Each launch's tl::profile row carries a device time.
  static constexpr bool times_launches = true;
};

struct caps {
  // Whether the model-path row is real here, or answers false: a decoder
  // runs on raw buffers only where it is true, and keeps to the array ops
  // otherwise (there is no CPU fallback under that row).
  static constexpr bool model_path = true;
  static constexpr bool graph_capture = false;
  static constexpr bool row_gemv = true;   // gemv_bf16_row: weights as [N,K]
  static constexpr bool bf16_gemm = true;  // gemm_bf16_nt: a bf16-weight GEMM
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

#endif  // __APPLE__
