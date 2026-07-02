#pragma once

// tl::array — F32 n-d array with zero-copy views, numpy broadcast rules and
// lazy evaluation.
//
// Ops build a graph (detail::node); tl::eval() / any data access topo-sorts
// and evaluates it. Peephole fusion happens at build time: every node
// carries an affine epilogue (result = op(...) * scale + offset), and scalar
// chains fold into the producing node by composing that epilogue — fusion
// never mutates an existing node, it creates a composed copy, so a bypassed
// intermediate stays valid for other consumers.
//
// Evaluation runs through the `ref::` backend: naive strided loops that are
// the correctness oracle every real backend (Accelerate/Metal, CPU
// microkernels, CUDA) is verified against, and the permanent fallback.
// Device dispatch slots in at detail::graph::eval_one (M3+).

#include <storage.h>
#include <types.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tl {

using shape_t = std::vector<int64_t>;

namespace detail {

inline int64_t num_elements(const shape_t& s) {
  int64_t n = 1;
  for (auto d : s) n *= d;
  return n;
}

inline std::vector<int64_t> contiguous_strides(const shape_t& s) {
  std::vector<int64_t> out(s.size(), 1);
  for (size_t i = s.size(); i-- > 1;) out[i - 1] = out[i] * s[i];
  return out;
}

inline std::string shape_str(const shape_t& s) {
  std::string out = "(";
  for (size_t i = 0; i < s.size(); i++) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + ")";
}

// numpy rules: align trailing dims; each pair must match or one must be 1.
inline shape_t broadcast_shape(const shape_t& a, const shape_t& b) {
  size_t rank = std::max(a.size(), b.size());
  shape_t out(rank);
  for (size_t i = 0; i < rank; i++) {
    int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (da != db && da != 1 && db != 1) {
      throw std::invalid_argument("tl: cannot broadcast " + shape_str(a) +
                                  " with " + shape_str(b));
    }
    out[i] = std::max(da, db);
  }
  return out;
}

// Strides of `src` viewed as `out_shape`: missing leading dims and size-1
// dims contribute stride 0.
inline std::vector<int64_t> broadcast_strides(
    const shape_t& src_shape, const std::vector<int64_t>& src_strides,
    const shape_t& out_shape) {
  size_t rank = out_shape.size(), lead = rank - src_shape.size();
  std::vector<int64_t> out(rank, 0);
  for (size_t i = 0; i < src_shape.size(); i++) {
    out[lead + i] = src_shape[i] == 1 ? 0 : src_strides[i];
  }
  return out;
}

// Output shape of an axis reduction; normalizes `axis` in place.
inline shape_t reduce_shape(const shape_t& s, int& axis, bool keepdims) {
  int r = static_cast<int>(s.size());
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::invalid_argument("tl: bad axis");
  shape_t out;
  for (int i = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) out.push_back(1);
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

// Row-major walk over `shape`, calling f(linear_out_index, offsets...) with
// per-source strided offsets. The oracle for every layout: views, broadcast
// and transposed inputs all reduce to strides here.
template <typename F>
void for_each_index(const shape_t& shape,
                    const std::vector<std::vector<int64_t>>& strides, F f) {
  int64_t n = num_elements(shape);
  size_t rank = shape.size(), nsrc = strides.size();
  std::vector<int64_t> idx(rank, 0);
  std::vector<int64_t> off(nsrc, 0);
  for (int64_t i = 0; i < n; i++) {
    f(i, off);
    for (size_t r = rank; r-- > 0;) {
      idx[r]++;
      for (size_t s = 0; s < nsrc; s++) off[s] += strides[s][r];
      if (idx[r] < shape[r]) break;
      for (size_t s = 0; s < nsrc; s++) off[s] -= idx[r] * strides[s][r];
      idx[r] = 0;
    }
  }
}

// Lazy graph node. `constant` wraps a materialized array (possibly a strided
// view); other ops fill stor/strides/soffset when evaluated. Cycles are
// impossible: inputs point input-ward only.
struct node {
  enum class op_t {
    constant,
    add, sub, mul, div, pow_,
    gt, lt, ge, le, eq, ne,  // masks as F32 (0/1)
    affine, recip, exp_, log_, sqrt_, sigmoid, relu,
    softmax,
    where_,
    dot,
    sum_ax, mean_ax, max_ax, argmax_ax, sum_to_,
  };

  op_t op = op_t::constant;
  shape_t shape;  // for sum_to this is the target shape (= the op parameter)
  std::vector<std::shared_ptr<node>> inputs;
  float scale = 1.0f, offset = 0.0f;  // fused epilogue: op(...) * scale + offset
  int axis = 0;
  bool keepdims = false;

  // constant source / evaluated result
  storage stor;
  std::vector<int64_t> strides;
  int64_t soffset = 0;
  bool evaluated = false;
};

using node_ptr = std::shared_ptr<node>;

struct graph;

}  // namespace detail

class array {
 public:
  array() = default;

  // Creation
  static array empty(shape_t shape);
  static array zeros(shape_t shape);
  static array ones(shape_t shape);
  static array full(shape_t shape, float v);
  static array from(std::vector<float> v);  // 1-d
  static array from(std::vector<float> v, shape_t shape);

  // Introspection (valid before evaluation — shapes are known at build time)
  const shape_t& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  size_t rank() const { return shape_.size(); }
  int64_t size() const { return detail::num_elements(shape_); }
  bool contiguous() const;
  bool defined() const { return storage_.buf != nullptr || node_ != nullptr; }

  // Data access — forces evaluation. data() additionally requires a
  // contiguous array; raw() is the strided base pointer kernels consume.
  float* data();
  const float* data() const;
  const float* raw() const;
  float item() const;                              // size() == 1
  float at(std::initializer_list<int64_t>) const;  // strided single read

  // Views (zero-copy on the materialized result) and copies
  array transpose() const;                       // reverse all axes
  array transpose(std::vector<int> axes) const;  // permutation
  array reshape(shape_t shape) const;  // view when contiguous, else copy
  array slice(int64_t start, int64_t count) const;  // axis 0
  array clone() const;                              // contiguous copy

  // Elementwise (lazy)
  array exp() const;
  array log() const;
  array sqrt() const;
  array sigmoid() const;
  array relu() const;
  array softmax() const;  // last axis, numerically stable

  // Linear algebra: (M,K)@(K,N); 1-d operands promote numpy-style. (lazy)
  array dot(const array& b) const;

  // Axis reductions (lazy); argmax yields indices as F32
  array sum(int axis, bool keepdims = false) const;
  array mean(int axis, bool keepdims = false) const;
  array max(int axis, bool keepdims = false) const;
  array argmax(int axis, bool keepdims = false) const;

  // Reduce broadcast dims back to `shape` (the VJP of broadcasting): sums
  // over leading dims and size-1 dims. `shape` must broadcast to shape().
  array sum_to(shape_t shape) const;

  // In-place accumulate (eager): this += b, broadcasting b. Mutates the
  // underlying storage — visible through every view sharing it, and through
  // unevaluated graphs holding it as a constant. Intended for gradient
  // accumulation right after eval; not for arrays still feeding lazy graphs.
  array& add_(const array& b);

  // Scalar reductions (eager — they return host scalars)
  float sum() const;
  float max() const;
  int64_t argmax() const;
  float mean() const { return sum() / static_cast<float>(size()); }

  // Force evaluation of this array's graph.
  const array& eval() const {
    ensure_();
    return *this;
  }

 private:
  shape_t shape_;
  std::vector<int64_t> strides_;  // element units
  int64_t offset_ = 0;
  storage storage_;
  detail::node_ptr node_;  // set while lazy; cleared on adoption

  static array make_(shape_t shape);
  void ensure_() const;  // materialize (evaluate + adopt) if lazy

  friend struct detail::graph;
  friend array make_view_(const array& base, shape_t shape,
                          std::vector<int64_t> strides, int64_t offset);
};

// Operators (broadcasting; scalar forms fuse into affine epilogues)
array operator+(const array& a, const array& b);
array operator-(const array& a, const array& b);
array operator*(const array& a, const array& b);
array operator/(const array& a, const array& b);
array operator+(const array& a, float s);
array operator-(const array& a, float s);
array operator*(const array& a, float s);
array operator/(const array& a, float s);
array operator+(float s, const array& a);
array operator-(float s, const array& a);
array operator*(float s, const array& a);
array operator/(float s, const array& a);
array pow(const array& a, const array& b);
array pow(const array& a, float s);

// Comparisons — F32 masks (1.0 / 0.0), broadcasting. The relu-backward
// pattern is `g * (x > 0.0f)`.
array operator>(const array& a, const array& b);
array operator<(const array& a, const array& b);
array operator>=(const array& a, const array& b);
array operator<=(const array& a, const array& b);
array operator==(const array& a, const array& b);
array operator!=(const array& a, const array& b);
array operator>(const array& a, float s);
array operator<(const array& a, float s);
array operator>=(const array& a, float s);
array operator<=(const array& a, float s);
array operator==(const array& a, float s);
array operator!=(const array& a, float s);

array where(const array& cond, const array& a, const array& b);

array sum_to(const array& a, shape_t shape);

array concat(const std::vector<array>& parts);  // axis 0

// Batch evaluation: one topological pass over all roots.
template <typename... Ts>
void eval(const Ts&... arrays);

// Testing helpers
bool array_equal(const array& a, const array& b);
bool allclose(const array& a, const array& b, float rtol = 1e-5f,
              float atol = 1e-6f);

// Implementation ------------------------------------------------------------

inline array make_view_(const array& base, shape_t shape,
                        std::vector<int64_t> strides, int64_t offset) {
  array v;
  v.shape_ = std::move(shape);
  v.strides_ = std::move(strides);
  v.offset_ = offset;
  v.storage_ = base.storage_;
  return v;
}

inline array array::make_(shape_t shape) {
  array a;
  a.strides_ = detail::contiguous_strides(shape);
  a.storage_ = storage::make(detail::num_elements(shape));
  a.shape_ = std::move(shape);
  return a;
}

inline array array::empty(shape_t shape) { return make_(std::move(shape)); }

inline array array::full(shape_t shape, float v) {
  auto a = make_(std::move(shape));
  auto* p = a.storage_.data();
  for (int64_t i = 0; i < a.size(); i++) p[i] = v;
  return a;
}

inline array array::zeros(shape_t shape) { return full(std::move(shape), 0); }
inline array array::ones(shape_t shape) { return full(std::move(shape), 1); }

inline array array::from(std::vector<float> v) {
  auto n = static_cast<int64_t>(v.size());
  return from(std::move(v), {n});
}

inline array array::from(std::vector<float> v, shape_t shape) {
  if (detail::num_elements(shape) != static_cast<int64_t>(v.size())) {
    throw std::invalid_argument("tl::from: size mismatch with shape " +
                                detail::shape_str(shape));
  }
  auto a = make_(std::move(shape));
  std::copy(v.begin(), v.end(), a.storage_.data());
  return a;
}

inline bool array::contiguous() const {
  if (node_) return true;  // results materialize contiguous
  return strides_ == detail::contiguous_strides(shape_);
}

inline const float* array::raw() const {
  ensure_();
  return storage_.data() + offset_;
}

inline float* array::data() {
  ensure_();
  if (!contiguous()) {
    throw std::logic_error("tl::data: non-contiguous view; use clone()");
  }
  return storage_.data() + offset_;
}

inline const float* array::data() const {
  return const_cast<array*>(this)->data();
}

inline float array::item() const {
  if (size() != 1) throw std::logic_error("tl::item: size != 1");
  return raw()[0];
}

inline float array::at(std::initializer_list<int64_t> idx) const {
  if (idx.size() != rank()) throw std::invalid_argument("tl::at: bad rank");
  const auto* p = raw();
  int64_t off = 0;
  size_t r = 0;
  for (auto i : idx) off += i * strides_[r++];
  return p[off];
}

inline array array::transpose() const {
  std::vector<int> axes(rank());
  for (size_t i = 0; i < rank(); i++) {
    axes[i] = static_cast<int>(rank() - 1 - i);
  }
  return transpose(std::move(axes));
}

inline array array::transpose(std::vector<int> axes) const {
  if (axes.size() != rank()) {
    throw std::invalid_argument("tl::transpose: bad axes");
  }
  ensure_();
  shape_t shape(rank());
  std::vector<int64_t> strides(rank());
  for (size_t i = 0; i < rank(); i++) {
    shape[i] = shape_[axes[i]];
    strides[i] = strides_[axes[i]];
  }
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::reshape(shape_t shape) const {
  if (detail::num_elements(shape) != size()) {
    throw std::invalid_argument("tl::reshape: size mismatch");
  }
  ensure_();
  if (!contiguous()) return clone().reshape(std::move(shape));
  auto strides = detail::contiguous_strides(shape);
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::slice(int64_t start, int64_t count) const {
  if (rank() == 0 || start < 0 || start + count > shape_[0]) {
    throw std::invalid_argument("tl::slice: out of range");
  }
  ensure_();
  auto shape = shape_;
  shape[0] = count;
  return make_view_(*this, std::move(shape), strides_,
                    offset_ + start * strides_[0]);
}

inline array array::clone() const {
  ensure_();
  auto out = make_(shape_);
  auto* po = out.storage_.data();
  const auto* pi = raw();
  detail::for_each_index(
      shape_, {strides_},
      [&](int64_t i, const std::vector<int64_t>& off) { po[i] = pi[off[0]]; });
  return out;
}

// Reference backend ----------------------------------------------------------
//
// Naive strided kernels over materialized arrays. Deliberately simple: the
// correctness oracle and universal fallback. Real backends replace these on
// hot paths via the dispatch in detail::graph::eval_one.

namespace detail {

template <typename F>
array map_unary(const array& a, F f) {
  auto out = array::empty(a.shape());
  auto* po = out.data();
  const auto* pa = a.raw();
  for_each_index(a.shape(), {a.strides()},
                 [&](int64_t i, const std::vector<int64_t>& off) {
                   po[i] = f(pa[off[0]]);
                 });
  return out;
}

template <typename F>
array map_binary(const array& a, const array& b, F f) {
  auto shape = broadcast_shape(a.shape(), b.shape());
  auto out = array::empty(shape);
  auto* po = out.data();
  const auto* pa = a.raw();
  const auto* pb = b.raw();
  for_each_index(shape,
                 {broadcast_strides(a.shape(), a.strides(), shape),
                  broadcast_strides(b.shape(), b.strides(), shape)},
                 [&](int64_t i, const std::vector<int64_t>& off) {
                   po[i] = f(pa[off[0]], pb[off[1]]);
                 });
  return out;
}

template <typename F>
array map_ternary(const array& a, const array& b, const array& c, F f) {
  auto shape = broadcast_shape(broadcast_shape(a.shape(), b.shape()), c.shape());
  auto out = array::empty(shape);
  auto* po = out.data();
  const auto* pa = a.raw();
  const auto* pb = b.raw();
  const auto* pc = c.raw();
  for_each_index(shape,
                 {broadcast_strides(a.shape(), a.strides(), shape),
                  broadcast_strides(b.shape(), b.strides(), shape),
                  broadcast_strides(c.shape(), c.strides(), shape)},
                 [&](int64_t i, const std::vector<int64_t>& off) {
                   po[i] = f(pa[off[0]], pb[off[1]], pc[off[2]]);
                 });
  return out;
}

// Shared axis-reduction driver: for each input element, f(acc_slot, value).
template <typename F>
array reduce_axis(const array& a, int axis, bool keepdims, float init, F f) {
  auto out_shape = reduce_shape(a.shape(), axis, keepdims);
  int r = static_cast<int>(a.rank());
  auto out = array::full(out_shape, init);
  // Map each input index to its accumulator: axis contributes stride 0.
  auto out_strides = contiguous_strides(out_shape);
  std::vector<int64_t> acc_strides(r, 0);
  for (int i = 0, oi = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) oi++;
      continue;
    }
    acc_strides[i] = out_strides[oi++];
  }
  auto* po = out.data();
  const auto* pi = a.raw();
  for_each_index(a.shape(), {a.strides(), acc_strides},
                 [&](int64_t, const std::vector<int64_t>& off) {
                   f(po[off[1]], pi[off[0]]);
                 });
  return out;
}

}  // namespace detail

namespace ref {

inline array softmax(const array& a) {
  auto out = array::empty(a.shape());
  int64_t cols = a.shape().back();
  int64_t rows = a.size() / (cols ? cols : 1);
  int64_t col_stride = a.strides().back();
  // Walk row starts through the outer dims (all but the last).
  shape_t outer(a.shape().begin(), a.shape().end() - 1);
  std::vector<int64_t> outer_strides(a.strides().begin(),
                                     a.strides().end() - 1);
  std::vector<int64_t> row_off(rows);
  detail::for_each_index(outer, {outer_strides},
                         [&](int64_t i, const std::vector<int64_t>& off) {
                           row_off[i] = off[0];
                         });
  const auto* pi = a.raw();
  auto* po = out.data();
  for (int64_t r = 0; r < rows; r++) {
    const float* src = pi + row_off[r];
    float* dst = po + r * cols;
    float m = src[0];
    for (int64_t c = 1; c < cols; c++) m = std::max(m, src[c * col_stride]);
    float denom = 0;
    for (int64_t c = 0; c < cols; c++) {
      dst[c] = std::exp(src[c * col_stride] - m);
      denom += dst[c];
    }
    for (int64_t c = 0; c < cols; c++) dst[c] /= denom;
  }
  return out;
}

inline array dot(const array& a_in, const array& b_in) {
  array a = a_in, b = b_in;
  bool vec_m = a.rank() == 1, vec_n = b.rank() == 1;
  if (vec_m) a = a.reshape({1, a.size()});
  if (vec_n) b = b.reshape({b.size(), 1});
  int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  auto out = array::zeros({m, n});
  const auto* pa = a.raw();
  const auto* pb = b.raw();
  auto* po = out.data();
  int64_t as0 = a.strides()[0], as1 = a.strides()[1];
  int64_t bs0 = b.strides()[0], bs1 = b.strides()[1];
  for (int64_t i = 0; i < m; i++) {
    for (int64_t l = 0; l < k; l++) {
      float av = pa[i * as0 + l * as1];
      for (int64_t j = 0; j < n; j++) {
        po[i * n + j] += av * pb[l * bs0 + j * bs1];
      }
    }
  }
  if (vec_m && vec_n) return out.reshape({});
  if (vec_m) return out.reshape({n});
  if (vec_n) return out.reshape({m});
  return out;
}

inline array sum(const array& a, int axis, bool keepdims) {
  return detail::reduce_axis(a, axis, keepdims, 0.0f,
                             [](float& acc, float v) { acc += v; });
}

inline array mean(const array& a, int axis, bool keepdims) {
  int r = static_cast<int>(a.rank());
  int64_t n = a.shape()[axis < 0 ? axis + r : axis];
  auto s = sum(a, axis, keepdims);
  float inv = 1.0f / static_cast<float>(n);
  return detail::map_unary(s, [inv](float x) { return x * inv; });
}

inline array max(const array& a, int axis, bool keepdims) {
  return detail::reduce_axis(
      a, axis, keepdims, -std::numeric_limits<float>::infinity(),
      [](float& acc, float v) { acc = std::max(acc, v); });
}

inline array argmax(const array& a, int axis, bool keepdims) {
  // Two passes sharing the reduce driver: per-slot max, then the first
  // axis-index attaining it (slots start at -1 = unset).
  int r = static_cast<int>(a.rank());
  if (axis < 0) axis += r;
  auto m = max(a, axis, true);
  auto out = detail::reduce_axis(a, axis, keepdims, -1.0f,
                                 [](float&, float) {});
  auto* po = out.data();
  const auto* pi = a.raw();
  const auto* pm = m.raw();
  auto m_strides =
      detail::broadcast_strides(m.shape(), m.strides(), a.shape());
  auto out_strides = detail::contiguous_strides(out.shape());
  std::vector<int64_t> acc_strides(r, 0);
  for (int i = 0, oi = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) oi++;
      continue;
    }
    acc_strides[i] = out_strides[oi++];
  }
  std::vector<int64_t> pos(r, 0);
  pos[axis] = 1;  // off[3] = index along the reduced axis
  detail::for_each_index(a.shape(),
                         {a.strides(), m_strides, acc_strides, pos},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           if (po[off[2]] < 0 && pi[off[0]] == pm[off[1]]) {
                             po[off[2]] = static_cast<float>(off[3]);
                           }
                         });
  return out;
}

// The VJP of broadcasting: accumulate `a` back down to `target` (leading
// dims and size-1 dims sum away). The accumulator strides are exactly the
// broadcast strides of the target viewed as a's shape — sum_to is the
// transpose of the broadcast read.
inline array sum_to(const array& a, const shape_t& target) {
  auto out = array::zeros(target);
  auto acc = detail::broadcast_strides(target, out.strides(), a.shape());
  auto* po = out.data();
  const auto* pi = a.raw();
  detail::for_each_index(a.shape(), {a.strides(), acc},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           po[off[1]] += pi[off[0]];
                         });
  return out;
}

}  // namespace ref

// Accelerate backend (macOS) --------------------------------------------------
//
// First real backend: vDSP/vForce/CBLAS fast paths tried from eval_one, with
// ref:: as the fallback for shapes they don't cover (broadcast, non-
// contiguous except CBLAS-mappable transposes). Every function returns
// nullopt/false when ineligible or on non-Apple builds, so eval_one carries
// no platform conditionals. Graduates to its own header alongside the Metal
// backend (M3b).

namespace accel {

inline bool enabled_() {
#ifdef __APPLE__
  return use_accelerate_;
#else
  return false;
#endif
}

inline std::optional<array> binary(detail::node::op_t op, const array& a,
                                   const array& b) {
#ifdef __APPLE__
  using op_t = detail::node::op_t;
  if (!enabled_() || !a.contiguous() || !b.contiguous() ||
      a.shape() != b.shape()) {
    return std::nullopt;
  }
  auto n = static_cast<vDSP_Length>(a.size());
  auto out = array::empty(a.shape());
  if (n == 0) return out;
  const float* pa = a.raw();
  const float* pb = b.raw();
  float* po = out.data();
  switch (op) {
    // vDSP argument-order quirk: vsub/vdiv take the subtrahend/divisor FIRST.
    case op_t::add: vDSP_vadd(pa, 1, pb, 1, po, 1, n); break;
    case op_t::sub: vDSP_vsub(pb, 1, pa, 1, po, 1, n); break;  // po = a - b
    case op_t::mul: vDSP_vmul(pa, 1, pb, 1, po, 1, n); break;
    case op_t::div: vDSP_vdiv(pb, 1, pa, 1, po, 1, n); break;  // po = a / b
    default: return std::nullopt;
  }
  return out;
#else
  (void)op; (void)a; (void)b;
  return std::nullopt;
#endif
}

inline std::optional<array> unary(detail::node::op_t op, const array& a) {
#ifdef __APPLE__
  using op_t = detail::node::op_t;
  if (!enabled_() || !a.contiguous()) return std::nullopt;
  auto out = array::empty(a.shape());
  if (a.size() == 0) return out;
  int nn = static_cast<int>(a.size());
  const float* pa = a.raw();
  float* po = out.data();
  switch (op) {
    case op_t::exp_:  vvexpf(po, pa, &nn); break;
    case op_t::log_:  vvlogf(po, pa, &nn); break;
    case op_t::sqrt_: vvsqrtf(po, pa, &nn); break;
    case op_t::relu: {
      float lo = 0.0f;
      vDSP_vthr(pa, 1, &lo, po, 1, static_cast<vDSP_Length>(nn));
      break;
    }
    default: return std::nullopt;
  }
  return out;
#else
  (void)op; (void)a;
  return std::nullopt;
#endif
}

inline std::optional<array> affine(const array& a, float s, float o) {
#ifdef __APPLE__
  if (!enabled_() || !a.contiguous()) return std::nullopt;
  auto out = array::empty(a.shape());
  if (a.size() == 0) return out;
  vDSP_vsmsa(a.raw(), 1, &s, &o, out.data(), 1,
             static_cast<vDSP_Length>(a.size()));
  return out;
#else
  (void)a; (void)s; (void)o;
  return std::nullopt;
#endif
}

// C = alpha * A @ B into a preallocated contiguous (m,n) array. Handles
// operands whose layout maps onto CBLAS: row-major contiguous rows
// (NoTrans, lda = row stride) or their transposed views (Trans) — silarray
// lesson: reading transposed operands in place beats materializing them.
inline bool gemm(const array& a, const array& b, array& out, float alpha) {
#ifdef __APPLE__
  if (!enabled_()) return false;
  struct layout { CBLAS_TRANSPOSE trans; int ld; };
  auto classify = [](const array& x) -> std::optional<layout> {
    int64_t r = x.shape()[0], c = x.shape()[1];
    int64_t s0 = x.strides()[0], s1 = x.strides()[1];
    if (s1 == 1 && s0 >= std::max<int64_t>(c, 1)) {
      return layout{CblasNoTrans, static_cast<int>(s0)};
    }
    if (s0 == 1 && s1 >= std::max<int64_t>(r, 1)) {
      return layout{CblasTrans, static_cast<int>(s1)};
    }
    return std::nullopt;
  };
  auto la = classify(a), lb = classify(b);
  if (!la || !lb) return false;
  int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  if (m == 0 || n == 0) return true;  // out has no elements
  if (k == 0) {
    float* po = out.data();
    for (int64_t i = 0; i < m * n; i++) po[i] = 0.0f;
    return true;
  }
  cblas_sgemm(CblasRowMajor, la->trans, lb->trans, static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), alpha, a.raw(),
              la->ld, b.raw(), lb->ld, 0.0f, out.data(),
              static_cast<int>(n));
  return true;
#else
  (void)a; (void)b; (void)out; (void)alpha;
  return false;
#endif
}

}  // namespace accel

// Graph build + evaluation ----------------------------------------------------

namespace detail {

struct graph {
  using op_t = node::op_t;

  static node_ptr as_node(const array& a) {
    if (!a.defined()) throw std::logic_error("tl: undefined array");
    if (a.node_) return a.node_;
    auto n = std::make_shared<node>();
    n->shape = a.shape_;
    n->stor = a.storage_;
    n->strides = a.strides_;
    n->soffset = a.offset_;
    n->evaluated = true;
    return n;
  }

  static array from_node(node_ptr n) {
    array a;
    a.shape_ = n->shape;
    a.strides_ = contiguous_strides(n->shape);
    a.node_ = std::move(n);
    return a;
  }

  static array binary(op_t op, const array& a, const array& b) {
    auto n = std::make_shared<node>();
    n->op = op;
    n->shape = broadcast_shape(a.shape(), b.shape());  // throws early
    n->inputs = {as_node(a), as_node(b)};
    return from_node(std::move(n));
  }

  static array unary(op_t op, const array& a) {
    auto n = std::make_shared<node>();
    n->op = op;
    n->shape = a.shape();
    n->inputs = {as_node(a)};
    return from_node(std::move(n));
  }

  // y = a * s + o. If `a` is an unevaluated op node, compose into a copy of
  // it (epilogue fusion): (base*S+O)*s+o = base*(S*s) + (O*s+o). The copy
  // shares the original's inputs; the original is left untouched for any
  // other consumer.
  static array affine(const array& a, float s, float o) {
    if (a.node_ && !a.node_->evaluated && a.node_->op != op_t::constant) {
      auto c = std::make_shared<node>(*a.node_);
      c->scale = a.node_->scale * s;
      c->offset = a.node_->offset * s + o;
      return from_node(std::move(c));
    }
    auto v = unary(op_t::affine, a);
    v.node_->scale = s;
    v.node_->offset = o;
    return v;
  }

  static array where(const array& c, const array& a, const array& b) {
    auto n = std::make_shared<node>();
    n->op = op_t::where_;
    n->shape = broadcast_shape(broadcast_shape(c.shape(), a.shape()),
                               b.shape());  // throws early
    n->inputs = {as_node(c), as_node(a), as_node(b)};
    return from_node(std::move(n));
  }

  static array sum_to(const array& a, shape_t target) {
    if (broadcast_shape(target, a.shape()) != a.shape()) {
      throw std::invalid_argument("tl::sum_to: " + shape_str(target) +
                                  " does not broadcast to " +
                                  shape_str(a.shape()));
    }
    auto n = std::make_shared<node>();
    n->op = op_t::sum_to_;
    n->shape = std::move(target);
    n->inputs = {as_node(a)};
    return from_node(std::move(n));
  }

  static array reduce(op_t op, const array& a, int axis, bool keepdims) {
    auto n = std::make_shared<node>();
    n->op = op;
    n->shape = reduce_shape(a.shape(), axis, keepdims);  // normalizes axis
    n->axis = axis;
    n->keepdims = keepdims;
    n->inputs = {as_node(a)};
    return from_node(std::move(n));
  }

  static array dot(const array& a, const array& b) {
    const auto& sa = a.shape();
    const auto& sb = b.shape();
    if (sa.size() < 1 || sa.size() > 2 || sb.size() < 1 || sb.size() > 2 ||
        sa.back() != sb[0]) {
      throw std::invalid_argument("tl::dot: shape mismatch " + shape_str(sa) +
                                  " @ " + shape_str(sb));
    }
    auto n = std::make_shared<node>();
    n->op = op_t::dot;
    if (sa.size() == 2 && sb.size() == 2) {
      n->shape = {sa[0], sb[1]};
    } else if (sa.size() == 2) {
      n->shape = {sa[0]};
    } else if (sb.size() == 2) {
      n->shape = {sb[1]};
    }  // 1-d @ 1-d → rank-0 scalar
    n->inputs = {as_node(a), as_node(b)};
    return from_node(std::move(n));
  }

  // Materialized view of an evaluated node, for kernel consumption.
  static array wrap(const node& n) {
    array a;
    a.shape_ = n.shape;
    a.strides_ = n.strides;
    a.offset_ = n.soffset;
    a.storage_ = n.stor;
    return a;
  }

  static void store(node& n, const array& result) {
    n.stor = result.storage_;
    n.strides = result.strides_;
    n.soffset = result.offset_;
    n.evaluated = true;
  }

  // One topological pass over all roots (MLX-style batch eval), then each
  // node evaluates through eval_one. Iterative DFS: recursion depth must not
  // bound graph depth.
  static void run(const std::vector<node_ptr>& roots) {
    std::vector<node*> order;
    std::unordered_set<node*> visited;
    std::vector<std::pair<node*, size_t>> stack;
    for (const auto& root : roots) {
      if (!root || root->evaluated || visited.count(root.get())) continue;
      visited.insert(root.get());
      stack.emplace_back(root.get(), 0);
      while (!stack.empty()) {
        auto& [n, i] = stack.back();
        if (i < n->inputs.size()) {
          node* in = n->inputs[i++].get();
          if (!in->evaluated && !visited.count(in)) {
            visited.insert(in);
            stack.emplace_back(in, 0);
          }
        } else {
          order.push_back(n);
          stack.pop_back();
        }
      }
    }
    for (auto* n : order) eval_one(*n);
  }

  static void eval_one(node& n) {
    auto in = [&](size_t i) { return wrap(*n.inputs[i]); };
    array r;
    bool epi_done = false;  // epilogue already applied inside the op body
    switch (n.op) {
      case op_t::constant:
        return;
      case op_t::add:
      case op_t::sub:
      case op_t::mul:
      case op_t::div: {
        auto a = in(0), b = in(1);
        if (auto o = accel::binary(n.op, a, b)) {
          r = std::move(*o);
        } else if (n.op == op_t::add) {
          r = map_binary(a, b, std::plus<float>());
        } else if (n.op == op_t::sub) {
          r = map_binary(a, b, std::minus<float>());
        } else if (n.op == op_t::mul) {
          r = map_binary(a, b, std::multiplies<float>());
        } else {
          r = map_binary(a, b, std::divides<float>());
        }
        break;
      }
      case op_t::pow_:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return std::pow(x, y); });
        break;
      case op_t::gt:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x > y ? 1.0f : 0.0f; });
        break;
      case op_t::lt:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x < y ? 1.0f : 0.0f; });
        break;
      case op_t::ge:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x >= y ? 1.0f : 0.0f; });
        break;
      case op_t::le:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x <= y ? 1.0f : 0.0f; });
        break;
      case op_t::eq:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x == y ? 1.0f : 0.0f; });
        break;
      case op_t::ne:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return x != y ? 1.0f : 0.0f; });
        break;
      case op_t::where_:
        r = map_ternary(in(0), in(1), in(2), [](float c, float x, float y) {
          return c != 0.0f ? x : y;
        });
        break;
      case op_t::affine: {
        float s = n.scale, o = n.offset;
        auto a = in(0);
        if (auto out = accel::affine(a, s, o)) {
          r = std::move(*out);
        } else {
          r = map_unary(a, [s, o](float x) { return x * s + o; });
        }
        epi_done = true;
        break;
      }
      case op_t::recip:
        r = map_unary(in(0), [](float x) { return 1.0f / x; });
        break;
      case op_t::exp_:
      case op_t::log_:
      case op_t::sqrt_:
      case op_t::relu: {
        auto a = in(0);
        if (auto o = accel::unary(n.op, a)) {
          r = std::move(*o);
        } else if (n.op == op_t::exp_) {
          r = map_unary(a, [](float x) { return std::exp(x); });
        } else if (n.op == op_t::log_) {
          r = map_unary(a, [](float x) { return std::log(x); });
        } else if (n.op == op_t::sqrt_) {
          r = map_unary(a, [](float x) { return std::sqrt(x); });
        } else {
          r = map_unary(a, [](float x) { return x > 0 ? x : 0.0f; });
        }
        break;
      }
      case op_t::sigmoid:
        r = map_unary(in(0),
                      [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        break;
      case op_t::softmax:
        r = ref::softmax(in(0));
        break;
      case op_t::dot: {
        auto a = in(0), b = in(1);
        array a2 = a.rank() == 1 ? a.reshape({1, a.size()}) : a;
        array b2 = b.rank() == 1 ? b.reshape({b.size(), 1}) : b;
        array out = array::empty({a2.shape()[0], b2.shape()[1]});
        if (accel::gemm(a2, b2, out, n.scale)) {  // epilogue scale = alpha
          if (n.offset != 0.0f) {
            if (auto ofs = accel::affine(out, 1.0f, n.offset)) {
              out = std::move(*ofs);
            } else {
              float o = n.offset;
              out = map_unary(out, [o](float x) { return x + o; });
            }
          }
          r = out.reshape(n.shape);
          epi_done = true;
        } else {
          r = ref::dot(a, b);
        }
        break;
      }
      case op_t::sum_ax:
        r = ref::sum(in(0), n.axis, n.keepdims);
        break;
      case op_t::mean_ax:
        r = ref::mean(in(0), n.axis, n.keepdims);
        break;
      case op_t::max_ax:
        r = ref::max(in(0), n.axis, n.keepdims);
        break;
      case op_t::argmax_ax:
        r = ref::argmax(in(0), n.axis, n.keepdims);
        break;
      case op_t::sum_to_:
        r = ref::sum_to(in(0), n.shape);
        break;
    }
    if (!epi_done && (n.scale != 1.0f || n.offset != 0.0f)) {
      float s = n.scale, o = n.offset;
      if (auto out = accel::affine(r, s, o)) {
        r = std::move(*out);
      } else {
        r = map_unary(r, [s, o](float x) { return x * s + o; });
      }
    }
    store(n, r);
  }

  static void eval_arrays(std::initializer_list<const array*> arrays) {
    std::vector<node_ptr> roots;
    for (const auto* a : arrays) {
      if (a->node_ && !a->node_->evaluated) roots.push_back(a->node_);
    }
    run(roots);
    for (const auto* a : arrays) a->ensure_();
  }
};

}  // namespace detail

inline void array::ensure_() const {
  if (!node_) return;
  if (!node_->evaluated) detail::graph::run({node_});
  auto* self = const_cast<array*>(this);
  self->storage_ = node_->stor;
  self->strides_ = node_->strides;
  self->offset_ = node_->soffset;
  self->node_.reset();
}

template <typename... Ts>
void eval(const Ts&... arrays) {
  detail::graph::eval_arrays({&arrays...});
}

// Operators and lazy methods --------------------------------------------------

inline array operator+(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::add, a, b);
}
inline array operator-(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::sub, a, b);
}
inline array operator*(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::mul, a, b);
}
inline array operator/(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::div, a, b);
}
inline array pow(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::pow_, a, b);
}
inline array pow(const array& a, float s) {
  return pow(a, array::full({}, s));
}

inline array operator+(const array& a, float s) {
  return detail::graph::affine(a, 1.0f, s);
}
inline array operator-(const array& a, float s) {
  return detail::graph::affine(a, 1.0f, -s);
}
inline array operator*(const array& a, float s) {
  return detail::graph::affine(a, s, 0.0f);
}
inline array operator/(const array& a, float s) {
  return detail::graph::affine(a, 1.0f / s, 0.0f);
}
inline array operator+(float s, const array& a) { return a + s; }
inline array operator*(float s, const array& a) { return a * s; }
inline array operator-(float s, const array& a) {
  return detail::graph::affine(a, -1.0f, s);
}
inline array operator/(float s, const array& a) {
  return detail::graph::affine(
      detail::graph::unary(detail::node::op_t::recip, a), s, 0.0f);
}

inline array operator>(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::gt, a, b);
}
inline array operator<(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::lt, a, b);
}
inline array operator>=(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::ge, a, b);
}
inline array operator<=(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::le, a, b);
}
inline array operator==(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::eq, a, b);
}
inline array operator!=(const array& a, const array& b) {
  return detail::graph::binary(detail::node::op_t::ne, a, b);
}
inline array operator>(const array& a, float s) { return a > array::full({}, s); }
inline array operator<(const array& a, float s) { return a < array::full({}, s); }
inline array operator>=(const array& a, float s) { return a >= array::full({}, s); }
inline array operator<=(const array& a, float s) { return a <= array::full({}, s); }
inline array operator==(const array& a, float s) { return a == array::full({}, s); }
inline array operator!=(const array& a, float s) { return a != array::full({}, s); }

inline array where(const array& cond, const array& a, const array& b) {
  return detail::graph::where(cond, a, b);
}

inline array array::sum_to(shape_t shape) const {
  return detail::graph::sum_to(*this, std::move(shape));
}
inline array sum_to(const array& a, shape_t shape) {
  return a.sum_to(std::move(shape));
}

inline array& array::add_(const array& b) {
  ensure_();
  if (detail::broadcast_shape(shape_, b.shape()) != shape_) {
    throw std::invalid_argument("tl::add_: " + detail::shape_str(b.shape()) +
                                " does not broadcast to " +
                                detail::shape_str(shape_));
  }
  const auto* pb = b.raw();
  auto* po = storage_.data() + offset_;
  detail::for_each_index(
      shape_,
      {strides_, detail::broadcast_strides(b.shape(), b.strides(), shape_)},
      [&](int64_t, const std::vector<int64_t>& off) {
        po[off[0]] += pb[off[1]];
      });
  return *this;
}

inline array array::exp() const {
  return detail::graph::unary(detail::node::op_t::exp_, *this);
}
inline array array::log() const {
  return detail::graph::unary(detail::node::op_t::log_, *this);
}
inline array array::sqrt() const {
  return detail::graph::unary(detail::node::op_t::sqrt_, *this);
}
inline array array::sigmoid() const {
  return detail::graph::unary(detail::node::op_t::sigmoid, *this);
}
inline array array::relu() const {
  return detail::graph::unary(detail::node::op_t::relu, *this);
}
inline array array::softmax() const {
  if (rank() == 0) throw std::invalid_argument("tl::softmax: rank 0");
  return detail::graph::unary(detail::node::op_t::softmax, *this);
}

inline array array::dot(const array& b) const {
  return detail::graph::dot(*this, b);
}

inline array array::sum(int axis, bool keepdims) const {
  return detail::graph::reduce(detail::node::op_t::sum_ax, *this, axis,
                               keepdims);
}
inline array array::mean(int axis, bool keepdims) const {
  return detail::graph::reduce(detail::node::op_t::mean_ax, *this, axis,
                               keepdims);
}
inline array array::max(int axis, bool keepdims) const {
  return detail::graph::reduce(detail::node::op_t::max_ax, *this, axis,
                               keepdims);
}
inline array array::argmax(int axis, bool keepdims) const {
  return detail::graph::reduce(detail::node::op_t::argmax_ax, *this, axis,
                               keepdims);
}

// Scalar reductions (eager) ---------------------------------------------------

inline float array::sum() const {
  double acc = 0;  // f64 accumulation: the oracle must not carry order noise
  const auto* pi = raw();
  detail::for_each_index(shape_, {strides_},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           acc += pi[off[0]];
                         });
  return static_cast<float>(acc);
}

inline float array::max() const {
  if (size() == 0) throw std::invalid_argument("tl::max: empty");
  const auto* pi = raw();
  float m = pi[0];
  detail::for_each_index(shape_, {strides_},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           m = std::max(m, pi[off[0]]);
                         });
  return m;
}

inline int64_t array::argmax() const {
  if (size() == 0) throw std::invalid_argument("tl::argmax: empty");
  const auto* pi = raw();
  float m = pi[0];
  int64_t best = 0;
  detail::for_each_index(shape_, {strides_},
                         [&](int64_t i, const std::vector<int64_t>& off) {
                           if (pi[off[0]] > m) {
                             m = pi[off[0]];
                             best = i;
                           }
                         });
  return best;
}

inline array concat(const std::vector<array>& parts) {
  if (parts.empty()) throw std::invalid_argument("tl::concat: empty");
  auto shape = parts[0].shape();
  if (shape.empty()) throw std::invalid_argument("tl::concat: rank 0");
  int64_t rows = 0;
  for (auto& p : parts) {
    auto s = p.shape();
    rows += s[0];
    s[0] = shape[0];
    if (s != shape) throw std::invalid_argument("tl::concat: shape mismatch");
  }
  shape[0] = rows;
  auto out = array::empty(shape);
  auto* po = out.data();
  for (auto& p : parts) {
    const auto* pi = p.raw();
    detail::for_each_index(p.shape(), {p.strides()},
                           [&](int64_t i, const std::vector<int64_t>& off) {
                             po[i] = pi[off[0]];
                           });
    po += p.size();
  }
  return out;
}

inline bool array_equal(const array& a, const array& b) {
  return allclose(a, b, 0.0f, 0.0f);
}

inline bool allclose(const array& a, const array& b, float rtol, float atol) {
  if (a.shape() != b.shape()) return false;
  auto ac = a.clone(), bc = b.clone();
  const auto* pa = ac.data();
  const auto* pb = bc.data();
  for (int64_t i = 0; i < ac.size(); i++) {
    if (std::abs(pa[i] - pb[i]) > atol + rtol * std::abs(pb[i])) return false;
  }
  return true;
}

}  // namespace tl
