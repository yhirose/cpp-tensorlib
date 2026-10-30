#pragma once

// The GPU ops, written once. An op is a kernel id, the views it touches and
// how, a params struct, and a launch shape (gpu_abi.h); `launch` hands those to
// the selected backend's `dispatch`. Nothing here names a backend, so an op
// added here exists on all of them, and a backend without the kernel declines
// (false) and the caller falls back, exactly as before.
//
// Included by gpu.h, after the backend is selected.

#include <array>
#include <cmath>
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
  }                                                                            \
  inline constexpr bool has_##name = detail::owns_##name<own>::value;

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

// ---- what the generic compositions are made of. Each optional op (gpu_ops_nn.h,
// gpu_ops_llm.h) that a backend has no kernel for is composed of the core ops
// above, in gpu::generic: these are its shared pieces, and the composition of
// the one core op a backend may still lack a kernel for (argmax).

namespace generic {

// A device buffer for the extent of one composition. With host_fill the host
// writes `contents` before the first kernel reads it (a table, a mask), and
// the backend uploads it where memory is not unified.
struct scratch {
  void* buf = nullptr;
  float* contents = nullptr;
  int64_t bytes = 0;
  explicit scratch(int64_t floats, bool host_fill = false) : bytes(floats * 4) {
    buf = alloc(bytes, &contents, host_fill);
  }
  ~scratch() {
    if (buf) release(buf, bytes, contents);
  }
  scratch(const scratch&) = delete;
  scratch& operator=(const scratch&) = delete;
  explicit operator bool() const { return buf != nullptr; }
  operator span() const { return {buf, 0}; }
};

// A strided rank-2 view copied into a contiguous [m, n]: the broadcast add of
// a zero, which `zero` holds (one float the host filled).
inline bool copy_2d(span a, int64_t ars, int64_t acs, span zero, span o,
                    int64_t m, int64_t n) {
  return binary_bcast(kop::badd, a, ars, acs, zero, 0, 0, o, m, n, 1.0f, 0.0f);
}

// The vector copied into a buffer of this layer's own, brought to the host
// and scanned: the one host round trip the model path makes.
inline bool argmax(span a, int64_t n, int64_t* out_idx) {
  scratch c(n);
  if (!c || !unary(kop::affine, a, c, n, 1.0f, 0.0f)) return false;
  cpu_barrier();
  sync_to_host(c.buf, false);
  int64_t best = 0;
  for (int64_t i = 1; i < n; i++) {
    if (c.contents[i] > c.contents[best]) best = i;
  }
  *out_idx = best;
  return true;
}

}  // namespace generic

// The argmax of a length-n vector, the smallest index on ties: greedy decoding
// reads one int back rather than the logits. Drains the queue. Backend-own:
// the result's staging buffer and its read-back are the backend's.
TL_GPU_DETECT_OWN(argmax)
template <class Own = own>
inline bool argmax(span a, int64_t n, int64_t* out_idx) {
  if (!a || n <= 0 || !out_idx) return false;
  if constexpr (detail::owns_argmax<Own>::value) {
    if (detail::ran(Own::argmax(a, n, out_idx))) return true;
  }
  return generic::argmax(a, n, out_idx);
}

// n floats copied from one buffer to another: the backend's copy where it has
// one, else an affine pass.
TL_GPU_DETECT_OWN(copy_f32)
template <class Own = own>
inline bool copy_f32(span from, span to, int64_t n) {
  if constexpr (detail::owns_copy_f32<Own>::value) {
    if (detail::ran(Own::copy_f32(from, to, n))) return true;
  }
  return unary(kop::affine, from, to, n, 1.0f, 0.0f);
}

}  // namespace gpu
}  // namespace tl
