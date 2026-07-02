#pragma once

// tl::array — F32 n-d array with zero-copy views and numpy broadcast rules.
//
// M1: every op runs eagerly through the strided reference implementation
// below (`detail::`). It is deliberately naive — it is the correctness
// oracle every future backend (Accelerate/Metal, CPU microkernels, CUDA) is
// verified against, and the permanent fallback for unsupported targets.
// The lazy graph + fusion layer (M2) slots in between the public API and
// these kernels; `eval()` is already part of the API so user code written
// today survives that switch.

#include <storage.h>
#include <types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tl {

using shape_t = std::vector<int64_t>;

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

  // Introspection
  const shape_t& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  size_t rank() const { return shape_.size(); }
  int64_t size() const;
  bool contiguous() const;
  bool defined() const { return storage_.buf != nullptr; }

  // Data access. data() requires a contiguous array; raw() is the strided
  // base pointer (storage + view offset) and is what kernels consume.
  float* data();
  const float* data() const;
  const float* raw() const { return storage_.data() + offset_; }
  float item() const;                            // size() == 1
  float at(std::initializer_list<int64_t>) const;  // strided single read

  // Views (zero-copy) and copies
  array transpose() const;                       // reverse all axes
  array transpose(std::vector<int> axes) const;  // permutation
  array reshape(shape_t shape) const;  // view when contiguous, else copy
  array slice(int64_t start, int64_t count) const;  // axis 0
  array clone() const;                              // contiguous copy

  // Elementwise
  array exp() const;
  array log() const;
  array sqrt() const;
  array sigmoid() const;
  array relu() const;
  array softmax() const;  // last axis, numerically stable

  // Linear algebra: (M,K)@(K,N); 1-d operands promote numpy-style.
  array dot(const array& b) const;

  // Reductions
  float sum() const;
  float max() const;
  int64_t argmax() const;
  array sum(int axis, bool keepdims = false) const;
  array mean(int axis, bool keepdims = false) const;
  array max(int axis, bool keepdims = false) const;
  array argmax(int axis, bool keepdims = false) const;  // indices as F32
  float mean() const { return sum() / static_cast<float>(size()); }

  // No-op until the lazy graph lands (M2); part of the API contract now so
  // call sites don't change.
  const array& eval() const { return *this; }

 private:
  shape_t shape_;
  std::vector<int64_t> strides_;  // element units
  int64_t offset_ = 0;
  storage storage_;

  static array make_(shape_t shape);
  friend array make_view_(const array& base, shape_t shape,
                          std::vector<int64_t> strides, int64_t offset);
};

// Operators (broadcasting)
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

template <typename... Ts>
void eval(const Ts&... arrays) {}  // no-op until M2

// Testing helpers
bool array_equal(const array& a, const array& b);
bool allclose(const array& a, const array& b, float rtol = 1e-5f,
              float atol = 1e-6f);

// Implementation ------------------------------------------------------------

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

}  // namespace detail

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
  auto* p = a.data();
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
  std::copy(v.begin(), v.end(), a.data());
  return a;
}

inline int64_t array::size() const { return detail::num_elements(shape_); }

inline bool array::contiguous() const {
  return strides_ == detail::contiguous_strides(shape_);
}

inline float* array::data() {
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
  int64_t off = 0;
  size_t r = 0;
  for (auto i : idx) off += i * strides_[r++];
  return raw()[off];
}

inline array array::transpose() const {
  std::vector<int> axes(rank());
  for (size_t i = 0; i < rank(); i++) axes[i] = static_cast<int>(rank() - 1 - i);
  return transpose(std::move(axes));
}

inline array array::transpose(std::vector<int> axes) const {
  if (axes.size() != rank()) {
    throw std::invalid_argument("tl::transpose: bad axes");
  }
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
  if (!contiguous()) return clone().reshape(std::move(shape));
  auto strides = detail::contiguous_strides(shape);
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::slice(int64_t start, int64_t count) const {
  if (rank() == 0 || start < 0 || start + count > shape_[0]) {
    throw std::invalid_argument("tl::slice: out of range");
  }
  auto shape = shape_;
  shape[0] = count;
  return make_view_(*this, std::move(shape), strides_,
                    offset_ + start * strides_[0]);
}

inline array array::clone() const {
  auto out = make_(shape_);
  auto* po = out.storage_.data();
  const auto* pi = raw();
  detail::for_each_index(
      shape_, {strides_},
      [&](int64_t i, const std::vector<int64_t>& off) { po[i] = pi[off[0]]; });
  return out;
}

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

}  // namespace detail

inline array operator+(const array& a, const array& b) {
  return detail::map_binary(a, b, std::plus<float>());
}
inline array operator-(const array& a, const array& b) {
  return detail::map_binary(a, b, std::minus<float>());
}
inline array operator*(const array& a, const array& b) {
  return detail::map_binary(a, b, std::multiplies<float>());
}
inline array operator/(const array& a, const array& b) {
  return detail::map_binary(a, b, std::divides<float>());
}
inline array pow(const array& a, const array& b) {
  return detail::map_binary(a, b, [](float x, float y) { return std::pow(x, y); });
}

inline array operator+(const array& a, float s) {
  return detail::map_unary(a, [s](float x) { return x + s; });
}
inline array operator-(const array& a, float s) { return a + (-s); }
inline array operator*(const array& a, float s) {
  return detail::map_unary(a, [s](float x) { return x * s; });
}
inline array operator/(const array& a, float s) { return a * (1.0f / s); }
inline array operator+(float s, const array& a) { return a + s; }
inline array operator*(float s, const array& a) { return a * s; }
inline array operator-(float s, const array& a) {
  return detail::map_unary(a, [s](float x) { return s - x; });
}
inline array operator/(float s, const array& a) {
  return detail::map_unary(a, [s](float x) { return s / x; });
}
inline array pow(const array& a, float s) {
  return detail::map_unary(a, [s](float x) { return std::pow(x, s); });
}

inline array array::exp() const {
  return detail::map_unary(*this, [](float x) { return std::exp(x); });
}
inline array array::log() const {
  return detail::map_unary(*this, [](float x) { return std::log(x); });
}
inline array array::sqrt() const {
  return detail::map_unary(*this, [](float x) { return std::sqrt(x); });
}
inline array array::sigmoid() const {
  return detail::map_unary(*this,
                           [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}
inline array array::relu() const {
  return detail::map_unary(*this, [](float x) { return x > 0 ? x : 0.0f; });
}

inline array array::softmax() const {
  if (rank() == 0) throw std::invalid_argument("tl::softmax: rank 0");
  auto out = empty(shape_);
  int64_t cols = shape_.back();
  int64_t rows = size() / (cols ? cols : 1);
  int64_t col_stride = strides_.back();
  // Walk row starts through the outer dims (all but the last).
  shape_t outer(shape_.begin(), shape_.end() - 1);
  std::vector<int64_t> outer_strides(strides_.begin(), strides_.end() - 1);
  std::vector<int64_t> row_off(rows);
  detail::for_each_index(outer, {outer_strides},
                         [&](int64_t i, const std::vector<int64_t>& off) {
                           row_off[i] = off[0];
                         });
  const auto* pi = raw();
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

inline array array::dot(const array& b_in) const {
  array a = *this, b = b_in;
  bool vec_m = a.rank() == 1, vec_n = b.rank() == 1;
  if (vec_m) a = a.reshape({1, a.size()});
  if (vec_n) b = b.reshape({b.size(), 1});
  if (a.rank() != 2 || b.rank() != 2 || a.shape()[1] != b.shape()[0]) {
    throw std::invalid_argument("tl::dot: shape mismatch " +
                                detail::shape_str(shape_) + " @ " +
                                detail::shape_str(b_in.shape()));
  }
  int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  auto out = zeros({m, n});
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

namespace detail {

// Shared axis-reduction driver: for each input element, f(acc_slot, value).
template <typename F>
array reduce_axis(const array& a, int axis, bool keepdims, float init, F f) {
  int r = static_cast<int>(a.rank());
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::invalid_argument("tl: bad axis");
  shape_t out_shape;
  for (int i = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) out_shape.push_back(1);
    } else {
      out_shape.push_back(a.shape()[i]);
    }
  }
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
  float m = raw()[0];
  const auto* pi = raw();
  detail::for_each_index(shape_, {strides_},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           m = std::max(m, pi[off[0]]);
                         });
  return m;
}

inline int64_t array::argmax() const {
  if (size() == 0) throw std::invalid_argument("tl::argmax: empty");
  float m = raw()[0];
  int64_t best = 0;
  const auto* pi = raw();
  detail::for_each_index(shape_, {strides_},
                         [&](int64_t i, const std::vector<int64_t>& off) {
                           if (pi[off[0]] > m) {
                             m = pi[off[0]];
                             best = i;
                           }
                         });
  return best;
}

inline array array::sum(int axis, bool keepdims) const {
  return detail::reduce_axis(*this, axis, keepdims, 0.0f,
                             [](float& acc, float v) { acc += v; });
}

inline array array::mean(int axis, bool keepdims) const {
  int r = static_cast<int>(rank());
  int64_t n = shape_[static_cast<size_t>(axis < 0 ? axis + r : axis)];
  return sum(axis, keepdims) / static_cast<float>(n);
}

inline array array::max(int axis, bool keepdims) const {
  return detail::reduce_axis(
      *this, axis, keepdims, -std::numeric_limits<float>::infinity(),
      [](float& acc, float v) { acc = std::max(acc, v); });
}

inline array array::argmax(int axis, bool keepdims) const {
  // Two passes sharing the reduce driver: per-slot max, then the first
  // axis-index attaining it (slots start at -1 = unset).
  int r = static_cast<int>(rank());
  if (axis < 0) axis += r;
  auto m = max(axis, true);
  auto out = detail::reduce_axis(*this, axis, keepdims, -1.0f,
                                 [](float&, float) {});
  auto* po = out.data();
  const auto* pi = raw();
  const auto* pm = m.raw();
  auto m_strides = detail::broadcast_strides(m.shape(), m.strides(), shape_);
  auto out_strides = detail::contiguous_strides(out.shape());
  std::vector<int64_t> acc_strides(static_cast<size_t>(r), 0);
  for (int i = 0, oi = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) oi++;
      continue;
    }
    acc_strides[static_cast<size_t>(i)] = out_strides[static_cast<size_t>(oi++)];
  }
  std::vector<int64_t> pos(static_cast<size_t>(r), 0);
  pos[static_cast<size_t>(axis)] = 1;  // off[3] = index along the axis
  detail::for_each_index(shape_, {strides_, m_strides, acc_strides, pos},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           if (po[off[2]] < 0 && pi[off[0]] == pm[off[1]]) {
                             po[off[2]] = static_cast<float>(off[3]);
                           }
                         });
  return out;
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
