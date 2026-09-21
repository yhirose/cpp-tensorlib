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

namespace detail {
// Launches per kernel id since the last census_reset(): how a test tells a
// kernel that ran on the device from an op that quietly fell back to the CPU.
inline std::array<uint64_t, kKopCount> census_counts{};
}  // namespace detail

inline uint64_t census(kop k) {
  return detail::census_counts[static_cast<size_t>(k)];
}
inline void census_reset() { detail::census_counts.fill(0); }

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
inline bool unary(kop op, span a, span o, int64_t n, float scale, float offset) {
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
inline bool index_select(span a, span idx, span o, int64_t row_size, int64_t k) {
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

}  // namespace gpu
}  // namespace tl
