#pragma once

// The GPU ops, written once. An op is a kernel id, the views it touches and
// how, a params struct, and a launch shape (gpu_abi.h); `launch` hands those to
// the selected backend's `dispatch`. Nothing here names a backend, so an op
// added here exists on all of them, and a backend without the kernel declines
// (false) and the caller falls back, exactly as before.
//
// Included by gpu.h, after the backend is selected.

#include <array>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

// The census: how a test tells work that ran on the device from an op that
// quietly fell back to the CPU, which no comparison against an oracle can (the
// fallback is right too). census(k) counts the shared ops' launches per kernel
// id, ops_run() every op that ran here, shared or backend-own, since the last
// census_reset().
namespace detail {
inline std::array<uint64_t, kKopCount> census_counts{};
inline uint64_t census_ops_run = 0;
inline bool ran(bool ok) {
  if (ok) census_ops_run++;
  return ok;
}
}  // namespace detail

inline uint64_t census(kop k) {
  return detail::census_counts[static_cast<size_t>(k)];
}
inline uint64_t ops_run() { return detail::census_ops_run; }
inline void census_reset() {
  detail::census_counts.fill(0);
  detail::census_ops_run = 0;
}

// A backend may run an op its own way — a different algorithm, several
// kernels, a host round trip — by declaring it as a static member of its `own`
// struct, with the signature the op has here. An op that allows this asks
// TL_GPU_OWNS whether the selected backend did; one that did not declare it
// simply has no such member, so there are no stubs to keep in step.
#define TL_GPU_DETECT_OWN(name)                                                \
  namespace detail {                                                           \
  template <class B, class = void>                                             \
  struct owns_##name : std::false_type {};                                     \
  template <class B>                                                           \
  struct owns_##name<B, std::void_t<decltype(&B::name)>> : std::true_type {};  \
  }

// Every shared op launches through here.
template <class P>
inline bool launch(kop k, std::initializer_list<arg> args, const P& params,
                   const grid& g) {
  static_assert(std::is_trivially_copyable<P>::value && sizeof(P) % 4 == 0 &&
                    alignof(P) == 4,
                "kernel params are a run of 4-byte fields (gpu_abi.h)");
  if (!dispatch(k, args.begin(), args.size(), &params, sizeof(P), g)) {
    return false;
  }
  detail::census_counts[static_cast<size_t>(k)]++;
  detail::census_ops_run++;
  return true;
}

// out[i] = (a[i] OP b[i]) * scale + offset over n contiguous elements.
inline bool binary(kop op, span a, span b, span o, int64_t n, float scale,
                   float offset) {
  return launch(op, {in(a), in(b), out(o)},
                ew_params{static_cast<uint32_t>(n), scale, offset},
                policy::flat(n));
}

// out[i] = OP(a[i]) * scale + offset over n contiguous elements.
inline bool unary(kop op, span a, span o, int64_t n, float scale,
                  float offset) {
  return launch(op, {in(a), out(o)},
                ew_params{static_cast<uint32_t>(n), scale, offset},
                policy::flat(n));
}

// Rank-2 broadcast binary: out[r,c] = f(a[r*ars + c*acs], b[r*brs + c*bcs])
// into a contiguous [m, n] output, affine epilogue. One stride-parameterized
// kernel covers every rank-2 broadcast (row vector, column vector, per-row
// scalar), which keeps bias/gamma/beta chains on the device. Strides are
// elements and non-negative. A backend's kernel reads its cell either from a
// flat index or from a 2-D thread position; traits::cells_2d says which.
inline bool binary_bcast(kop op, span a, int64_t ars, int64_t acs, span b,
                         int64_t brs, int64_t bcs, span o, int64_t m, int64_t n,
                         float scale, float offset) {
  if (m <= 0 || n <= 0 || ars < 0 || acs < 0 || brs < 0 || bcs < 0) return false;
  auto u = [](int64_t v) { return static_cast<uint32_t>(v); };
  return launch(op, {in(a), in(b), out(o)},
                bcast_params{u(m), u(n), u(ars), u(acs), u(brs), u(bcs), scale,
                             offset},
                traits::cells_2d ? policy::cells_2d(m, n) : policy::flat(m * n));
}

// out[i] = a[i] CMP b[i * bstride] as 1.0 / 0.0 (bstride 0 broadcasts a
// scalar b). No epilogue.
inline bool compare(cmp_op op, span a, span b, span o, int64_t n,
                    int64_t bstride) {
  kop k = kop::gt_;
  switch (op) {
    case cmp_op::gt: k = kop::gt_; break;
    case cmp_op::lt: k = kop::lt_; break;
    case cmp_op::ge: k = kop::ge_; break;
    case cmp_op::le: k = kop::le_; break;
    case cmp_op::eq: k = kop::eq_; break;
    case cmp_op::ne: k = kop::ne_; break;
  }
  return launch(k, {in(a), in(b), out(o)},
                cmp_params{static_cast<uint32_t>(n), static_cast<uint32_t>(bstride)},
                policy::flat(n));
}

// out[i] = min(max(a[i], lo), hi): Clip's forward. No epilogue.
inline bool clamp(span a, span o, int64_t n, float lo, float hi) {
  return launch(kop::clamp_, {in(a), out(o)},
                clamp_params{static_cast<uint32_t>(n), lo, hi}, policy::flat(n));
}

// Tensor-scalar ops: out[i] = (a[i] OP s) * scale + offset, the scalar a
// kernel argument rather than a rank-0 operand buffer.
inline bool scalar_binary(scalar_op op, span a, span o, int64_t n, float s,
                          float scale, float offset) {
  kop k = kop::pow_s_;
  switch (op) {
    case scalar_op::pow: k = kop::pow_s_; break;
    case scalar_op::gt: k = kop::gt_s_; break;
    case scalar_op::lt: k = kop::lt_s_; break;
    case scalar_op::ge: k = kop::ge_s_; break;
    case scalar_op::le: k = kop::le_s_; break;
    case scalar_op::eq: k = kop::eq_s_; break;
    case scalar_op::ne: k = kop::ne_s_; break;
  }
  return launch(k, {in(a), out(o)},
                scalar_params{static_cast<uint32_t>(n), s, scale, offset},
                policy::flat(n));
}

// Row op over the last axis of [rows, cols]: softmax writes rows x cols;
// row_sum / row_max write one value a row. Affine epilogue.
inline bool row_op(kop op, span a, span o, int64_t rows, int64_t cols,
                   float scale, float offset) {
  if (rows <= 0 || cols <= 0) return false;
  return launch(op, {in(a), out(o)},
                reduce_params{static_cast<uint32_t>(rows),
                              static_cast<uint32_t>(cols), scale, offset},
                policy::one_group_per_row(rows));
}

// log(sum exp) per row, affine epilogue: row_op's shape, one pass over the row
// (each thread carries a running max and sum, hence two floats of scratch).
inline bool row_logsumexp(span a, span o, int64_t rows, int64_t cols,
                          float scale, float offset) {
  if (rows <= 0 || cols <= 0) return false;
  return launch(kop::row_logsumexp_, {in(a), out(o)},
                reduce_params{static_cast<uint32_t>(rows),
                              static_cast<uint32_t>(cols), scale, offset},
                policy::one_group_per_row(rows, 2));
}

// out = (x - mean) / sqrt(var + eps) * g + b per row of [rows, cols], affine
// epilogue; g and b are contiguous cols-vectors.
inline bool layer_norm(span x, span g, span b, span o, int64_t rows,
                       int64_t cols, float eps, float scale, float offset) {
  if (rows <= 0 || cols <= 0) return false;
  return launch(kop::layer_norm_, {in(x), in(g), in(b), out(o)},
                layer_norm_params{static_cast<uint32_t>(rows),
                                  static_cast<uint32_t>(cols), eps, scale, offset},
                policy::one_group_per_row(rows));
}

// Row gather along axis 0: out row i = a row idx[i], for k rows of row_size.
inline bool index_select(span a, span idx, span o, int64_t row_size,
                         int64_t k) {
  const int64_t n = k * row_size;
  if (n <= 0) return false;
  return launch(kop::index_select, {in(a), in(idx), out(o)},
                gather_params{static_cast<uint32_t>(row_size),
                              static_cast<uint32_t>(n)},
                policy::flat(n));
}

// out[i] = src[i * size + idx[i]]: the element each of n positions labels
// along a trailing axis of `size`.
inline bool gather_from_axis(span src, span idx, span o, int64_t n,
                             int64_t size) {
  if (n <= 0) return false;
  return launch(kop::gather_axis_, {in(src), in(idx), out(o)},
                gather_axis_params{static_cast<uint32_t>(size),
                                   static_cast<uint32_t>(n)},
                policy::flat(n));
}

// Softmax cross-entropy's pullback from the forward's row logsumexp:
// out[i,j] = g[i] * (exp(x[i,j] - lse[i]) - [j == tgt[i]]). x and out are
// [rows, cols]; lse, tgt and g hold one value a row.
inline bool xent_bwd(span x, span lse, span tgt, span g, span o, int64_t rows,
                     int64_t cols) {
  const int64_t n = rows * cols;
  if (n <= 0) return false;
  return launch(kop::xent_bwd_, {in(x), in(lse), in(tgt), in(g), out(o)},
                xent_bwd_params{static_cast<uint32_t>(cols),
                                static_cast<uint32_t>(n)},
                policy::flat(n));
}

// Adam's update in place over n contiguous elements: m and v advance, p moves
// by the bias-corrected ratio the host folded into lr_over_bc1 and inv_bc2.
inline bool adam_step(span p, span m, span v, span g, int64_t n, float beta1,
                      float beta2, float eps, float lr_over_bc1, float inv_bc2) {
  if (n <= 0) return false;
  return launch(kop::adam_step_, {inout(p), inout(m), inout(v), in(g)},
                adam_params{beta1, beta2, eps, lr_over_bc1, inv_bc2,
                            static_cast<uint32_t>(n)},
                policy::flat(n));
}

// ---- the model path: what a decoder runs on raw device buffers between its
// GEMVs and attention. No CPU fallback sits under these, so a model checks the
// return and keeps to the array ops where it is false.

// out = x * rsqrt(mean(x^2) + eps) * w per row of [rows, n]. out may alias x.
inline bool rmsnorm(span x, span w, span o, int64_t n, float eps,
                    int64_t rows = 1) {
  if (n <= 0 || rows <= 0) return false;
  return launch(kop::rmsnorm_, {in(x), in(w), out(o)},
                rmsnorm_params{static_cast<uint32_t>(n), eps},
                policy::one_group_per_row(rows));
}

// xout = x + delta and hout = rmsnorm(xout) * w: a residual add folded into
// the norm that follows it. xout may alias x.
inline bool rmsnorm_res(span x, span delta, span w, span xout, span hout,
                        int64_t n, float eps, int64_t rows = 1) {
  if (n <= 0 || rows <= 0) return false;
  return launch(kop::add_rmsnorm_,
                {in(x), in(delta), in(w), out(xout), out(hout)},
                rmsnorm_params{static_cast<uint32_t>(n), eps},
                policy::one_group_per_row(rows));
}

// out[rows, ff] = silu(gate) * up out of the fused gate|up buffer [rows, 2ff].
inline bool swiglu(span gu, span o, int64_t ff, int64_t rows = 1) {
  if (ff <= 0 || rows <= 0) return false;
  return launch(kop::swiglu_, {in(gu), out(o)},
                swiglu_params{static_cast<uint32_t>(ff)},
                policy::flat_rows(ff, rows));
}

// y[1,N] = a[1,K] . W[N,K]^T with the bf16 weight row-major (GGML-native): one
// group an output row. Requires k % 8 == 0.
inline bool gemv_bf16_row(span a, span W, span y, int64_t n, int64_t k) {
  if (n <= 0 || k <= 0 || k % 8 != 0) return false;
  return launch(kop::gemv_bf16_row_, {in(a), in(W), out(y)},
                gemv_row_params{static_cast<uint32_t>(n), static_cast<uint32_t>(k)},
                policy::row_reduce(n, k));
}

// The same over int4 weights: qw is [N][K/8] packed words and scales
// [N][K/group] floats, two views that may share a buffer. K % group == 0 and
// group % 8 == 0.
inline bool gemv_q4(span a, span qw, span scales, span y, int64_t N, int64_t K,
                    int64_t group) {
  if (N <= 0 || K <= 0 || group <= 0 || K % group != 0 || group % 8 != 0) {
    return false;
  }
  return launch(kop::gemv_q4_, {in(a), in(qw), in(scales), out(y)},
                gemv_q4_params{static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                               static_cast<uint32_t>(group)},
                policy::row_reduce(N, K));
}

// One decode step's k, v (each [n_kv_heads, D]) into row `pos` of a
// [n_kv_heads, kv_max, D] cache, f32 or bf16.
inline bool kv_append(span Kc, span Vc, span k_new, span v_new, int64_t pos,
                      int64_t kv_max, int64_t n_kv_heads, int64_t D,
                      bool kv_bf16 = false) {
  if ((D != 64 && D != 128) || n_kv_heads <= 0) return false;
  return launch(kv_bf16 ? kop::kv_append_bf16_ : kop::kv_append_,
                {out(Kc), out(Vc), in(k_new), in(v_new)},
                kv_append_params{static_cast<uint32_t>(pos),
                                 static_cast<uint32_t>(kv_max * D)},
                policy::per_head(n_kv_heads, 1, D));
}

// A prefill's k, v (each [n_kv_heads, T, D]) into cache rows [pos0, pos0 + T).
inline bool kv_fill(span Kc, span Vc, span K, span V, int64_t T, int64_t kv_max,
                    int64_t n_kv_heads, int64_t D, bool kv_bf16 = false,
                    int64_t pos0 = 0) {
  if ((D != 64 && D != 128) || n_kv_heads <= 0 || T <= 0) return false;
  return launch(kv_bf16 ? kop::kv_fill_bf16_ : kop::kv_fill_,
                {out(Kc), out(Vc), in(K), in(V)},
                kv_fill_params{static_cast<uint32_t>(T),
                               static_cast<uint32_t>(kv_max * D),
                               static_cast<uint32_t>(pos0)},
                policy::per_head(n_kv_heads, T, D));
}

// Head-major [H, T, D] -> token-major [T, H*D]: split_heads' inverse.
inline bool merge_heads(span src, span dst, int64_t T, int64_t H, int64_t D) {
  if (T <= 0 || H <= 0 || D <= 0) return false;
  return launch(kop::merge_heads_, {in(src), out(dst)},
                merge_heads_params{static_cast<uint32_t>(T),
                                   static_cast<uint32_t>(H),
                                   static_cast<uint32_t>(D)},
                policy::per_head(H, T, D));
}

// Token-major [T, ld] -> head-major [H, T, D] from column block `off`, adding
// the optional per-head bias [H, D] (a null view: none). Backend-own: the
// kernels disagree on how "no bias" is said.
TL_GPU_DETECT_OWN(split_heads)
template <class Own = own>
inline bool split_heads(span src, span bias, span dst, int64_t T, int64_t ld,
                        int64_t off, int64_t H, int64_t D) {
  if (T <= 0 || H <= 0 || D <= 0) return false;
  if constexpr (detail::owns_split_heads<Own>::value) {
    return detail::ran(Own::split_heads(src, bias, dst, T, ld, off, H, D));
  } else {
    return false;
  }
}

// The argmax of a length-n vector, the smallest index on ties: greedy decoding
// reads one int back rather than the logits. Drains the queue. Backend-own:
// the result's staging buffer and its read-back are the backend's.
TL_GPU_DETECT_OWN(argmax)
template <class Own = own>
inline bool argmax(span a, int64_t n, int64_t* out_idx) {
  if (!a || n <= 0 || !out_idx) return false;
  if constexpr (detail::owns_argmax<Own>::value) {
    return detail::ran(Own::argmax(a, n, out_idx));
  } else {
    return false;
  }
}

// tanh / sin / cos: unary's shape under their own vocabulary (gpu_abi.h).
inline bool unary_ext(unary_ext_op op, span a, span o, int64_t n, float scale,
                      float offset) {
  kop k = kop::tanh_;
  switch (op) {
    case unary_ext_op::tanh_: k = kop::tanh_; break;
    case unary_ext_op::sin_: k = kop::sin_; break;
    case unary_ext_op::cos_: k = kop::cos_; break;
  }
  return unary(k, a, o, n, scale, offset);
}

// ---- ops a backend runs its own way. Same contract as the rest — views in,
// false when it declines — but the algorithm under each differs by backend
// (a scatter into a zeroed buffer against a gather, shape metadata in a device
// buffer against a params block, one kernel against a split and a combine), so
// each backend writes its own over its `dispatch`, as a member of `own`.

// N-D broadcast binary into a contiguous out of out_shape: strides are
// elements per axis (0 on a broadcast axis), rank <= 8.
TL_GPU_DETECT_OWN(binary_bcast_nd)
template <class Own = own>
inline bool binary_bcast_nd(kop op, span a, const int64_t* a_strides, span b,
                            const int64_t* b_strides, span o,
                            const int64_t* out_shape, int rank, int64_t n,
                            float scale, float offset) {
  if constexpr (detail::owns_binary_bcast_nd<Own>::value) {
    return detail::ran(Own::binary_bcast_nd(op, a, a_strides, b, b_strides, o, out_shape, rank, n, scale, offset));
  } else {
    return false;
  }
}

// out = cond ? a : b over out_shape, each operand through its own strides.
TL_GPU_DETECT_OWN(where_nd)
template <class Own = own>
inline bool where_nd(span cond, const int64_t* c_strides, span a,
                     const int64_t* a_strides, span b, const int64_t* b_strides,
                     span o, const int64_t* out_shape, int rank, int64_t n) {
  if constexpr (detail::owns_where_nd<Own>::value) {
    return detail::ran(Own::where_nd(cond, c_strides, a, a_strides, b, b_strides, o, out_shape, rank, n));
  } else {
    return false;
  }
}

// clone()'s strided gather: a through a_strides into a contiguous out.
TL_GPU_DETECT_OWN(copy_nd)
template <class Own = own>
inline bool copy_nd(span a, const int64_t* a_strides, span o,
                    const int64_t* out_shape, int rank, int64_t n) {
  if constexpr (detail::owns_copy_nd<Own>::value) {
    return detail::ran(Own::copy_nd(a, a_strides, o, out_shape, rank, n));
  } else {
    return false;
  }
}

// Un-broadcast a gradient: every out element sums the `a` elements that
// broadcast onto it (`acc` the per-axis accumulation strides).
TL_GPU_DETECT_OWN(sum_to)
template <class Own = own>
inline bool sum_to(span a, const int64_t* a_shape, const int64_t* a_strides,
                   const int64_t* acc, int rank, int64_t out_n,
                   int64_t reduced_n, span o) {
  if constexpr (detail::owns_sum_to<Own>::value) {
    return detail::ran(Own::sum_to(a, a_shape, a_strides, acc, rank, out_n, reduced_n, o));
  } else {
    return false;
  }
}

// `a` (contiguous) into a zero buffer of out_shape, shifted by `before` along
// `axis`. No epilogue.
TL_GPU_DETECT_OWN(pad)
template <class Own = own>
inline bool pad(span a, span o, const int64_t* a_shape,
                const int64_t* out_shape, int rank, int axis, int64_t before,
                int64_t n, int64_t out_n) {
  if constexpr (detail::owns_pad<Own>::value) {
    return detail::ran(Own::pad(a, o, a_shape, out_shape, rank, axis, before, n, out_n));
  } else {
    return false;
  }
}

// unfold's inverse: scatter-add a's trailing windows into out_shape.
TL_GPU_DETECT_OWN(fold)
template <class Own = own>
inline bool fold(span a, span o, const int64_t* a_shape,
                 const int64_t* out_shape, int rank, int axis, int64_t step,
                 int64_t n, int64_t out_n) {
  if constexpr (detail::owns_fold<Own>::value) {
    return detail::ran(Own::fold(a, o, a_shape, out_shape, rank, axis, step, n, out_n));
  } else {
    return false;
  }
}

// One part of a concat: `a` into out at `before` along `axis`.
TL_GPU_DETECT_OWN(concat_part)
template <class Own = own>
inline bool concat_part(span a, span o, const int64_t* a_shape,
                        const int64_t* out_shape, int rank, int axis,
                        int64_t before, int64_t n) {
  if constexpr (detail::owns_concat_part<Own>::value) {
    return detail::ran(Own::concat_part(a, o, a_shape, out_shape, rank, axis, before, n));
  } else {
    return false;
  }
}

// index_select's dual: out row idx[i] += values row i, for k rows.
TL_GPU_DETECT_OWN(index_add)
template <class Own = own>
inline bool index_add(span idx, span values, span o, int64_t row_size,
                      int64_t k, int64_t out_n) {
  if constexpr (detail::owns_index_add<Own>::value) {
    return detail::ran(Own::index_add(idx, values, o, row_size, k, out_n));
  } else {
    return false;
  }
}

// One-hot scatter into a new trailing axis: out[i, idx[i]] = values[i].
TL_GPU_DETECT_OWN(scatter_to_axis)
template <class Own = own>
inline bool scatter_to_axis(span idx, span values, span o, int64_t n,
                            int64_t size) {
  if constexpr (detail::owns_scatter_to_axis<Own>::value) {
    return detail::ran(Own::scatter_to_axis(idx, values, o, n, size));
  } else {
    return false;
  }
}

// C(m,n) = (A @ B) * scale + offset. lda/ldb are row strides and the trans
// flags read a transposed view in place.
TL_GPU_DETECT_OWN(gemm)
template <class Own = own>
inline bool gemm(span a, int64_t lda, bool ta, span b, int64_t ldb, bool tb,
                 span o, int64_t m, int64_t n, int64_t k, float scale,
                 float offset) {
  if constexpr (detail::owns_gemm<Own>::value) {
    return detail::ran(Own::gemm(a, lda, ta, b, ldb, tb, o, m, n, k, scale, offset));
  } else {
    return false;
  }
}

// `batch` GEMMs in one launch, slices sa / sb elements apart, with an optional
// row bias added in the store.
TL_GPU_DETECT_OWN(gemm_batched)
template <class Own = own>
inline bool gemm_batched(span a, int64_t lda, bool ta, int64_t sa, span b,
                         int64_t ldb, bool tb, int64_t sb, span o, int64_t m,
                         int64_t n, int64_t k, int64_t batch, float scale,
                         float offset, span bias = {}) {
  if constexpr (detail::owns_gemm_batched<Own>::value) {
    return detail::ran(Own::gemm_batched(a, lda, ta, sa, b, ldb, tb, sb, o, m, n, k, batch, scale, offset, bias));
  } else {
    return false;
  }
}

// addmm's shape: a GEMM with its row bias added in the store.
TL_GPU_DETECT_OWN(gemm_bias)
template <class Own = own>
inline bool gemm_bias(span a, int64_t lda, bool ta, span b, int64_t ldb,
                      bool tb, span bias, span o, int64_t m, int64_t n,
                      int64_t k, float scale, float offset) {
  if constexpr (detail::owns_gemm_bias<Own>::value) {
    return detail::ran(Own::gemm_bias(a, lda, ta, b, ldb, tb, bias, o, m, n, k, scale, offset));
  } else {
    return false;
  }
}

// Rotary embedding over [rows, T, D] at position `pos`, adding the optional
// per-row bias first.
TL_GPU_DETECT_OWN(rope)
template <class Own = own>
inline bool rope(span x, span o, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base, span bias = {}) {
  if constexpr (detail::owns_rope<Own>::value) {
    return detail::ran(Own::rope(x, o, rows, T, D, pos, base, bias));
  } else {
    return false;
  }
}

// Layer norm's pullback: dx, and dg / db reduced over rows in chunks.
TL_GPU_DETECT_OWN(layer_norm_bwd)
template <class Own = own>
inline bool layer_norm_bwd(span x, span g, span dy, span dx, span dg, span db,
                           span stats, span partials, int64_t rows,
                           int64_t cols, int64_t per_chunk, int64_t chunks,
                           float eps) {
  if constexpr (detail::owns_layer_norm_bwd<Own>::value) {
    return detail::ran(Own::layer_norm_bwd(x, g, dy, dx, dg, db, stats, partials, rows, cols, per_chunk, chunks, eps));
  } else {
    return false;
  }
}

// ---- the LLM path.

// y[1,n] = a[1,k] . B[k,n], the weight column-major, f32 or bf16.
TL_GPU_DETECT_OWN(gemv_f32)
template <class Own = own>
inline bool gemv_f32(span a, span B, span y, int64_t n, int64_t k) {
  if constexpr (detail::owns_gemv_f32<Own>::value) {
    return detail::ran(Own::gemv_f32(a, B, y, n, k));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(gemv_bf16)
template <class Own = own>
inline bool gemv_bf16(span a, span B, span y, int64_t n, int64_t k) {
  if constexpr (detail::owns_gemv_bf16<Own>::value) {
    return detail::ran(Own::gemv_bf16(a, B, y, n, k));
  } else {
    return false;
  }
}

// C[M,N] = A[M,K] . B[N,K]^T with B bf16: a prefill chunk's projection.
TL_GPU_DETECT_OWN(gemm_bf16_nt)
template <class Own = own>
inline bool gemm_bf16_nt(span A, span B, span C, int64_t M, int64_t N,
                         int64_t K) {
  if constexpr (detail::owns_gemm_bf16_nt<Own>::value) {
    return detail::ran(Own::gemm_bf16_nt(A, B, C, M, N, K));
  } else {
    return false;
  }
}

// One decode step: q [n_q_heads, D] against a [n_kv_heads, kv_max, D] cache
// read over [0, ctx). GQA by the head ratio; the cache is f32 or bf16.
TL_GPU_DETECT_OWN(attn_decode)
template <class Own = own>
inline bool attn_decode(span q, span K, span V, span o, int64_t n_q_heads,
                        int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                        int64_t D, float scale, bool kv_bf16 = false) {
  if constexpr (detail::owns_attn_decode<Own>::value) {
    return detail::ran(Own::attn_decode(q, K, V, o, n_q_heads, n_kv_heads, ctx, kv_max, D, scale, kv_bf16));
  } else {
    return false;
  }
}

// Causal prefill: q, out [n_q_heads, T, D] against the cache rows
// [0, pos0 + T); query t sits at absolute position pos0 + t.
TL_GPU_DETECT_OWN(attn_prefill)
template <class Own = own>
inline bool attn_prefill(span q, span K, span V, span o, int64_t n_q_heads,
                         int64_t n_kv_heads, int64_t T, int64_t kv_max,
                         int64_t D, float scale, bool kv_bf16 = false,
                         int64_t pos0 = 0) {
  if constexpr (detail::owns_attn_prefill<Own>::value) {
    return detail::ran(Own::attn_prefill(q, K, V, o, n_q_heads, n_kv_heads, T, kv_max, D, scale, kv_bf16, pos0));
  } else {
    return false;
  }
}

// The causal prefill's pullback, query half then key/value half.
TL_GPU_DETECT_OWN(attn_prefill_dq)
template <class Own = own>
inline bool attn_prefill_dq(span q, span K, span V, span dO, span O, span dq,
                            span stats, int64_t H, int64_t T, int64_t D,
                            float scale) {
  if constexpr (detail::owns_attn_prefill_dq<Own>::value) {
    return detail::ran(Own::attn_prefill_dq(q, K, V, dO, O, dq, stats, H, T, D, scale));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(attn_prefill_dkv)
template <class Own = own>
inline bool attn_prefill_dkv(span q, span K, span V, span dO, span stats,
                             span dK, span dV, int64_t H, int64_t T, int64_t D,
                             float scale) {
  if constexpr (detail::owns_attn_prefill_dkv<Own>::value) {
    return detail::ran(Own::attn_prefill_dkv(q, K, V, dO, stats, dK, dV, H, T, D, scale));
  } else {
    return false;
  }
}

// ---- the graph-capture forms (caps::graph_capture): a decode step whose
// position is a device scalar, so one captured step replays as the cache
// advances. `partials` is attn_dpos_partials_bytes() of scratch.
TL_GPU_DETECT_OWN(rope_dpos)
template <class Own = own>
inline bool rope_dpos(span x, span o, int64_t rows, int64_t T, int64_t D,
                      span d_pos, float base, span bias = {}) {
  if constexpr (detail::owns_rope_dpos<Own>::value) {
    return detail::ran(Own::rope_dpos(x, o, rows, T, D, d_pos, base, bias));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(kv_append_dpos)
template <class Own = own>
inline bool kv_append_dpos(span Kc, span Vc, span k_new, span v_new, span d_pos,
                           int64_t kv_max, int64_t n_kv_heads, int64_t D) {
  if constexpr (detail::owns_kv_append_dpos<Own>::value) {
    return detail::ran(Own::kv_append_dpos(Kc, Vc, k_new, v_new, d_pos, kv_max, n_kv_heads, D));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(attn_decode_dpos)
template <class Own = own>
inline bool attn_decode_dpos(span q, span K, span V, span o, int64_t n_q_heads,
                             int64_t n_kv_heads, span d_pos, int64_t kv_max,
                             int64_t D, float scale, span partials) {
  if constexpr (detail::owns_attn_decode_dpos<Own>::value) {
    return detail::ran(Own::attn_decode_dpos(q, K, V, o, n_q_heads, n_kv_heads, d_pos, kv_max, D, scale, partials));
  } else {
    return false;
  }
}

// *d_pos += 1 on the device.
TL_GPU_DETECT_OWN(incr_u32)
template <class Own = own>
inline bool incr_u32(span d_pos) {
  if constexpr (detail::owns_incr_u32<Own>::value) {
    return detail::ran(Own::incr_u32(d_pos));
  } else {
    return false;
  }
}

#undef TL_GPU_DETECT_OWN

}  // namespace gpu
}  // namespace tl
