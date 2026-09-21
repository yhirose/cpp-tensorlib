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
  argmax_, rmsnorm_, swiglu_, split_heads_, merge_heads_,  // the decode step's rest
  gemv_f32_, gemv_bf16_, gemv_q4_,   // decode GEMVs, per weight dtype
  gemv_combine_,                     // their split-K partials
  gemv_bf16_row_,                    // ... and the [N,K] weight layout's own
  gemm_bf16_nt_, gemm_bf16_nt32_,    // the prefill's bf16 GEMM, per M tile
  gather_axis_, row_logsumexp_, xent_bwd_, adam_step_  // cross-entropy, Adam
};
inline constexpr size_t kKopCount = static_cast<size_t>(kop::adam_step_) + 1;

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
// (an `in` is uploaded if the host holds the live copy; an `out` that the
// kernel writes in full makes the device copy the live one; `inout` does
// both), so no op says any of that by hand.
enum class access : uint8_t { in, out, inout };

struct arg {
  span s;
  access a;
};
inline arg in(span s) { return {s, access::in}; }
inline arg out(span s) { return {s, access::out}; }
inline arg inout(span s) { return {s, access::inout}; }

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

// Launch policy: the shapes ops launch in, in one place. Host code shared by
// every backend; what differs between devices will come in as traits.
namespace policy {

// One thread an element, in 1-D groups of `threads`.
inline grid flat(int64_t n, uint32_t threads = 256) {
  uint32_t groups = static_cast<uint32_t>((n + threads - 1) / threads);
  return {groups ? groups : 1, 1, 1, threads, 1, 1, 0};
}

}  // namespace policy

}  // namespace gpu
}  // namespace tl
