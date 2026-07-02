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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
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
    affine, recip, exp_, log_, sqrt_, sigmoid, relu,
    softmax,
    dot,
    sum_ax, mean_ax, max_ax, argmax_ax,
  };

  op_t op = op_t::constant;
  shape_t shape;
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

}  // namespace ref

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
        r = map_binary(in(0), in(1), std::plus<float>());
        break;
      case op_t::sub:
        r = map_binary(in(0), in(1), std::minus<float>());
        break;
      case op_t::mul:
        r = map_binary(in(0), in(1), std::multiplies<float>());
        break;
      case op_t::div:
        r = map_binary(in(0), in(1), std::divides<float>());
        break;
      case op_t::pow_:
        r = map_binary(in(0), in(1),
                       [](float x, float y) { return std::pow(x, y); });
        break;
      case op_t::affine: {
        float s = n.scale, o = n.offset;
        r = map_unary(in(0), [s, o](float x) { return x * s + o; });
        epi_done = true;
        break;
      }
      case op_t::recip:
        r = map_unary(in(0), [](float x) { return 1.0f / x; });
        break;
      case op_t::exp_:
        r = map_unary(in(0), [](float x) { return std::exp(x); });
        break;
      case op_t::log_:
        r = map_unary(in(0), [](float x) { return std::log(x); });
        break;
      case op_t::sqrt_:
        r = map_unary(in(0), [](float x) { return std::sqrt(x); });
        break;
      case op_t::sigmoid:
        r = map_unary(in(0),
                      [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        break;
      case op_t::relu:
        r = map_unary(in(0), [](float x) { return x > 0 ? x : 0.0f; });
        break;
      case op_t::softmax:
        r = ref::softmax(in(0));
        break;
      case op_t::dot:
        r = ref::dot(in(0), in(1));
        break;
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
    }
    if (!epi_done && (n.scale != 1.0f || n.offset != 0.0f)) {
      float s = n.scale, o = n.offset;
      r = map_unary(r, [s, o](float x) { return x * s + o; });
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
