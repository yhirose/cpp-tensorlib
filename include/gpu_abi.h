#pragma once

// What the shared GPU layer and a backend's device core agree on. No backend is
// named here, and none needs to be: a backend is a device core (memory, a
// kernel table, one `dispatch`) plus kernel source, and everything above it —
// the ops in gpu_ops.h, the launch policy below — is written once against
// these types.
//
// The kernel ABI. An op hands `dispatch` a kernel id, an ordered list of
// buffer views with how each is accessed, a params struct, and a grid. For a
// kernel id, the order of the views and the layout of the params are the same
// on every backend: views in the order the kernel declares its buffers, and
// the params struct a run of 4-byte fields (uint32_t / int32_t / float) in the
// order the kernel takes its scalars. That is what lets each backend realize a
// launch generically — Metal binds view i at buffer index i and the params
// after them; CUDA builds cuLaunchKernel's argv as the view addresses followed
// by the params' fields, four bytes apiece.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "profile.h"

namespace tl {
namespace gpu {

// Kernel ids. The shared ops name the ones every backend may implement; the
// rest are one backend's own (Metal's GEMM tiles, its attention variants),
// kept here only until that backend keys them privately.
enum class kop {
  add, sub, mul, div, pow_, exp_, log_, sqrt_, sigmoid, relu, affine,
  badd, bsub, bmul, bdiv, bpow,  // rank-2 broadcast binary (strided operands)
  sgemm32, sgemm32x64, sgemm64x32, sgemm64,
  steel, steel32x64, steel_ta, steel_tb, steel32x64_ta, steel32x64_tb,
  softmax, row_sum, row_max, pad, fold,
  index_select, index_add, scatter_axis,
  badd_nd, bsub_nd, bmul_nd, bdiv_nd, bpow_nd,  // N-D broadcast binary
  where_nd, copy_nd,             // N-D select / clone()'s strided gather
  gt_, lt_, ge_, le_, eq_, ne_,  // comparisons -- cmp_op maps onto these
  tanh_, sin_, cos_,             // unary_ext_op maps onto these
  clamp_, sum_to_, sum_to_blocked_,  // dedicated ops, mirroring cuda.h's own
  concat_part_, rope_,           // ditto -- Tensor.concat / RoPE's own dispatch
  pow_s_, gt_s_, lt_s_, ge_s_, le_s_, eq_s_, ne_s_,  // scalar_op maps onto these
  layer_norm_,                                       // the fused layer norm
  layer_norm_bwd_dx_, layer_norm_bwd_gb_, layer_norm_bwd_gb_fold_,  // its pullback
  attn_prefill_64_, attn_prefill_128_,  // causal prefill attention, per D
  attn_prefill_bf16_64_, attn_prefill_bf16_128_,  // over a bf16 KV cache
  attn_bwd_dq_64_, attn_bwd_dq_128_, attn_bwd_dkv_64_, attn_bwd_dkv_128_,
  attn_decode_64_, attn_decode_128_,  // fused decode attention, per D
  attn_decode_split_64_, attn_decode_split_128_,  // its split-KV pass
  attn_combine_64_, attn_combine_128_,            // and their partials
  attn_decode_bf16_64_, attn_decode_bf16_128_,    // the same over a bf16 cache
  attn_decode_split_bf16_64_, attn_decode_split_bf16_128_,
  kv_append_, kv_append_bf16_, kv_fill_, kv_fill_bf16_,  // the KV cache's writes
  argmax_, rmsnorm_, add_rmsnorm_, swiglu_, split_heads_, merge_heads_,  // decode's rest
  gemv_f32_, gemv_bf16_, gemv_q4_,   // decode GEMVs, per weight dtype
  gemv_combine_,                     // their split-K partials
  gemv_bf16_row_,                    // ... and the [N,K] weight layout's own
  gemm_bf16_nt_, gemm_bf16_nt32_,    // the prefill's bf16 GEMM, per M tile
  gather_axis_, row_logsumexp_, xent_bwd_, adam_step_  // cross-entropy, Adam
};
inline constexpr size_t kKopCount = static_cast<size_t>(kop::adam_step_) + 1;

// The ids' names, for what reports by kernel (the census, tl::profile).
inline constexpr const char* kKopNames[] = {
    "add", "sub", "mul", "div", "pow_", "exp_", "log_", "sqrt_", "sigmoid",
    "relu", "affine", "badd", "bsub", "bmul", "bdiv", "bpow", "sgemm32",
    "sgemm32x64", "sgemm64x32", "sgemm64", "steel", "steel32x64", "steel_ta",
    "steel_tb", "steel32x64_ta", "steel32x64_tb", "softmax", "row_sum",
    "row_max", "pad", "fold", "index_select", "index_add", "scatter_axis",
    "badd_nd", "bsub_nd", "bmul_nd", "bdiv_nd", "bpow_nd", "where_nd",
    "copy_nd", "gt_", "lt_", "ge_", "le_", "eq_", "ne_", "tanh_", "sin_",
    "cos_", "clamp_", "sum_to_", "sum_to_blocked_", "concat_part_", "rope_",
    "pow_s_", "gt_s_", "lt_s_", "ge_s_", "le_s_", "eq_s_", "ne_s_",
    "layer_norm_", "layer_norm_bwd_dx_", "layer_norm_bwd_gb_",
    "layer_norm_bwd_gb_fold_", "attn_prefill_64_", "attn_prefill_128_",
    "attn_prefill_bf16_64_", "attn_prefill_bf16_128_", "attn_bwd_dq_64_",
    "attn_bwd_dq_128_", "attn_bwd_dkv_64_", "attn_bwd_dkv_128_",
    "attn_decode_64_", "attn_decode_128_", "attn_decode_split_64_",
    "attn_decode_split_128_", "attn_combine_64_", "attn_combine_128_",
    "attn_decode_bf16_64_", "attn_decode_bf16_128_",
    "attn_decode_split_bf16_64_", "attn_decode_split_bf16_128_", "kv_append_",
    "kv_append_bf16_", "kv_fill_", "kv_fill_bf16_", "argmax_", "rmsnorm_",
    "add_rmsnorm_", "swiglu_", "split_heads_", "merge_heads_", "gemv_f32_",
    "gemv_bf16_", "gemv_q4_", "gemv_combine_", "gemv_bf16_row_",
    "gemm_bf16_nt_", "gemm_bf16_nt32_", "gather_axis_", "row_logsumexp_",
    "xent_bwd_", "adam_step_",
};
static_assert(sizeof(kKopNames) / sizeof(kKopNames[0]) == kKopCount,
              "kKopNames lists every kop, in order");
inline const char* kop_name(kop k) { return kKopNames[static_cast<size_t>(k)]; }

// Comparisons (gt/lt/ge/le/eq/ne), the extra unaries and the tensor-scalar
// ops are their own small vocabularies rather than kop values: each arrived
// on one backend ahead of the others, and an op keyed by its own enum can be
// declined (return false, fall back to the CPU) where a kop could not.
enum class cmp_op { gt, lt, ge, le, eq, ne };
enum class unary_ext_op { tanh_, sin_, cos_ };
// pow(x, s) and the comparisons against a scalar, with s a kernel argument
// instead of a rank-0 operand buffer (an allocation and an upload per call).
enum class scalar_op { pow, gt, lt, ge, le, eq, ne };

// A view into a device buffer: the currency every op takes. `buf` is whatever
// the backend's alloc() returned — an MTLBuffer handle, a device address, a
// key into a mirror table — and means nothing outside that backend. `off` is
// bytes. Because the offset travels beside the handle instead of inside it, a
// view is expressible on every backend, and arithmetic on a handle is not
// expressible at all.
struct span {
  void* buf = nullptr;
  int64_t off = 0;

  span at(int64_t bytes) const { return {buf, off + bytes}; }
  explicit operator bool() const { return buf != nullptr; }
};

// How a kernel touches a view. It drives the residency of a mirrored backend
// (`residency` below), so no op says any of that by hand: an `in` is uploaded
// if the host holds the live copy; an `out` makes the device copy the live one,
// uploading first only if the host had filled the buffer (a view may be part
// of it); an `inout` is uploaded and then becomes live.
enum class access : uint8_t { in, out, inout };

struct arg {
  span s;
  access a;
};
inline arg in(span s) { return {s, access::in}; }
inline arg out(span s) { return {s, access::out}; }
inline arg inout(span s) { return {s, access::inout}; }

// Where an allocation's live bytes are, for a backend whose device memory is
// not the host's (a mirrored backend keeps one of these per allocation, next to
// the two copies; a unified one has nothing to track). The backend does the
// copying; when to copy is decided here, once, from how each kernel and each
// host access touches the buffer.
//
// `none` is a fresh allocation nobody has filled. It matters for `out`: a view
// may cover only part of its buffer, so a kernel's output into a buffer whose
// live bytes are the host's has to bring them up first or lose the rest of
// them — but an output into a fresh buffer, which is nearly every output, has
// nothing to bring.
struct residency {
  enum state : uint8_t { none, host, device, both };
  state where = none;

  explicit residency(bool host_filled = false) : where(host_filled ? host : none) {}

  // A kernel is about to touch the buffer as `a`. True: upload the host copy
  // first. (A read of a `none` buffer uploads too: the host may have filled it
  // without saying so, and an unfilled one costs a transfer of garbage that no
  // correct program pays.)
  bool before_kernel(access a) {
    const bool upload = where == host || (where == none && a != access::out);
    if (a == access::in) {
      if (upload) where = both;
    } else {
      where = device;
    }
    return upload;
  }
  // The host is about to read the buffer or, with for_write, overwrite it.
  // True: download the device copy first. (A backend whose download can fail
  // asks needs_download(), and reports downloaded() / host_wrote() itself.)
  bool before_host(bool for_write) {
    const bool download = needs_download();
    if (download) downloaded();
    if (for_write) host_wrote();
    return download;
  }
  bool needs_download() const { return where == device; }
  void downloaded() { where = both; }
  void host_wrote() { where = host; }
  // The backend copied host bytes up outside a kernel (an explicit upload).
  void uploaded() { where = both; }
};

// A launch's extent: how many groups, how many threads in each, and the bytes
// of per-group scratch a kernel's reduction needs where the backend sizes it at
// launch (CUDA's shared memory; Metal and WGSL size theirs in the kernel).
struct grid {
  uint32_t gx = 1, gy = 1, gz = 1;
  uint32_t tx = 1, ty = 1, tz = 1;
  uint32_t scratch_bytes = 0;
};

// Params structs, one per kernel family: the second half of the kernel ABI.
// 4-byte fields only, in the order the kernel takes its scalars (so a backend
// may pass them as one block of bytes or field by field, and they mean the
// same). The MSL and WGSL sources declare the same layouts on their side.
struct ew_params {  // elementwise: out[i] = f(...) * scale + offset, i < n
  uint32_t n;
  float scale, offset;
};
struct bcast_params {  // rank-2 broadcast binary into a contiguous [m, n]
  uint32_t m, n;
  uint32_t ars, acs, brs, bcs;  // each operand's row / column stride, elements
  float scale, offset;
};
struct cmp_params {  // out[i] = a[i] CMP b[i * bstride] (bstride 0: a scalar)
  uint32_t n, bstride;
};
struct clamp_params {
  uint32_t n;
  float lo, hi;
};
struct scalar_params {  // out[i] = (a[i] OP s) * scale + offset
  uint32_t n;
  float s, scale, offset;
};
struct reduce_params {  // a row op over the last axis of [rows, cols]
  uint32_t rows, cols;
  float scale, offset;
};
struct layer_norm_params {
  uint32_t rows, cols;
  float eps, scale, offset;
};
struct gather_params {  // index_select: rows of `row_size`, n output elements
  uint32_t row_size, n;
};
struct gather_axis_params {  // out[i] = src[i * size + idx[i]], i < n
  uint32_t size, n;
};
struct xent_bwd_params {
  uint32_t cols, n;  // n = rows * cols
};
struct adam_params {
  float b1, b2, eps, lr_over_bc1, inv_bc2;
  uint32_t n;
};
struct rmsnorm_params {  // per row of n: x * rsqrt(mean(x^2) + eps) * w
  uint32_t n;
  float eps;
};
struct swiglu_params {
  uint32_t ff;
};
struct gemv_row_params {  // y[n] = a[k] . W[n, k], one group an output row
  uint32_t n, k;
};
struct gemv_q4_params {
  uint32_t n, k, group;
};
struct kv_append_params {
  uint32_t pos, kv_stride;  // kv_stride = kv_max * D, elements a head
};
struct kv_fill_params {
  uint32_t T, kv_stride, pos0;
};
struct merge_heads_params {
  uint32_t T, H, D;
};

// The launch record. Every kernel launch on every backend is one row under
// tl::profile, made here from the kernel's name: a backend's launch primitive
// (the one place its launches funnel through, shared ops and own alike) calls
// this once per launch and stamps the row with a device time where it has one
// (profile::detail::device_time). Under TL_PROFILE=1 the first launch also
// starts the profile — a decoder on the model path never reaches the
// evaluator, so this is where its profile begins.
inline profile::row* launched(std::string_view kernel) {
  profile::detail::env_autostart();
  return profile::active() ? profile::detail::launch(kernel) : nullptr;
}

// Launch policy: the shapes ops launch in, in one place. Host code shared by
// every backend; what differs between devices will come in as traits.
namespace policy {

// One thread an element, in 1-D groups of `threads`.
inline grid flat(int64_t n, uint32_t threads = 256) {
  uint32_t groups = static_cast<uint32_t>((n + threads - 1) / threads);
  return {groups ? groups : 1, 1, 1, threads, 1, 1, 0};
}

// One group a row, its threads reducing the row between them; each thread
// keeps `floats_per_thread` of scratch for the reduction tree.
inline grid one_group_per_row(int64_t rows, uint32_t floats_per_thread = 1,
                              uint32_t threads = 256) {
  uint32_t groups = static_cast<uint32_t>(rows);
  return {groups ? groups : 1, 1, 1, threads, 1, 1,
          threads * floats_per_thread * static_cast<uint32_t>(sizeof(float))};
}

// One thread an element of a row of n, `rows` of them stacked on y.
inline grid flat_rows(int64_t n, int64_t rows, uint32_t threads = 256) {
  grid g = flat(n, threads);
  g.gy = static_cast<uint32_t>(rows);
  return g;
}

// One group an output row, its threads striding that row's k inputs in steps
// of `step` elements and reducing between them: the smallest group (a
// multiple of 32, at most 256) that leaves each thread the fewest steps, so a
// narrow row is not spread over threads with nothing to do. Scratch is one
// float per 32-thread lane set, needed once there is more than one.
inline grid row_reduce(int64_t rows, int64_t k, uint32_t step = 8) {
  uint32_t threads = 32;
  int64_t fewest = INT64_MAX;
  for (uint32_t t = 32; t <= 256; t += 32) {
    const int64_t steps = (k + int64_t(step) * t - 1) / (int64_t(step) * t);
    if (steps < fewest) {
      fewest = steps;
      threads = t;
    }
  }
  const uint32_t groups = static_cast<uint32_t>(rows);
  return {groups ? groups : 1, 1, 1, threads, 1, 1,
          threads > 32 ? (threads >> 5) * static_cast<uint32_t>(sizeof(float)) : 0};
}

// One group per (head, row), a thread per head-dim element: the attention
// and KV-cache kernels' shape.
inline grid per_head(int64_t heads, int64_t rows, int64_t D) {
  return {static_cast<uint32_t>(heads), static_cast<uint32_t>(rows), 1,
          static_cast<uint32_t>(D), 1, 1, 0};
}

// A thread per cell of [rows, cols], for a kernel that reads its cell from a
// 2-D thread position (x the column). 32x8 groups.
inline grid cells_2d(int64_t rows, int64_t cols) {
  return {static_cast<uint32_t>((cols + 31) / 32),
          static_cast<uint32_t>((rows + 7) / 8), 1, 32, 8, 1, 0};
}

// Splitting a reduction over more groups. `groups` groups each walk k units
// (a GEMV's k, attention's keys, a GEMM tile's k); when they alone leave the
// device short of busy groups, each is cut into parts that run as extra groups
// and are combined after. The rule is the kernel's side of it — what its part
// must be a multiple of, and how short a part or a k is still worth the
// combine — and the target is the device's (traits::fill_groups, or a multiple
// of it for a kernel whose groups are small). A backend keeps to this shape
// of decision rather than its own arithmetic, so the split every backend's
// GEMV and attention make is one function of the device.
struct split_rule {
  int64_t target;    // groups that keep the device busy
  int64_t min_k;     // a k shorter than this is not split at all
  int64_t min_part;  // a part is at least this many units
  int64_t granule;   // and a multiple of this (the kernel's step)
};

// The parts the k units are cut into; 1 when not split. Monotone in k.
inline int64_t split_parts(int64_t groups, int64_t k, const split_rule& r) {
  if (groups <= 0 || groups >= r.target || k < r.min_k) return 1;
  int64_t parts = (r.target + groups - 1) / groups;
  if (r.min_part > 0 && parts > k / r.min_part) parts = k / r.min_part;
  return parts > 1 ? parts : 1;
}

// The units each of `parts` parts takes, rounded up to the granule. Rounding
// can leave fewer parts than asked, so the count to launch is (k + chunk - 1)
// / chunk.
inline int64_t split_chunk(int64_t k, int64_t parts, const split_rule& r) {
  const int64_t chunk = ((k + parts - 1) / parts + r.granule - 1) / r.granule * r.granule;
  return chunk > 0 ? chunk : r.granule;
}

}  // namespace policy

}  // namespace gpu
}  // namespace tl
