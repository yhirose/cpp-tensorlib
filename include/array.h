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

#include <cpu.h>
#include <storage.h>
#include <types.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// Soft rank cap so index walkers can use stack arrays instead of per-call
// heap vectors (the walker is called per *op*, and tiny-tensor workloads
// live or die on per-op allocation count).
inline constexpr size_t kMaxRank = 16;

// Row-major walk over `shape`, calling f(linear_out_index, offsets...) with
// per-source strided offsets. The oracle for every layout: views, broadcast
// and transposed inputs all reduce to strides here. Hot callers should take
// their contiguous fast path first — this walker is the generic fallback.
template <typename F>
void for_each_index(const shape_t& shape,
                    const std::vector<std::vector<int64_t>>& strides, F f) {
  int64_t n = num_elements(shape);
  size_t rank = shape.size(), nsrc = strides.size();
  if (rank > kMaxRank || nsrc > 4) {
    throw std::invalid_argument("tl: rank/source count over walker limits");
  }
  int64_t idx[kMaxRank] = {};
  std::vector<int64_t> off(nsrc, 0);  // part of f's signature; one alloc
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
    attn_dec,  // fused decode attention: softmax(arg0 · q·Kᵀ)·V
    rope,      // rotary position embedding (arg0=base, axis=position offset)
    sum_ax, mean_ax, max_ax, argmax_ax, sum_to_,
  };

  op_t op = op_t::constant;
  shape_t shape;  // for sum_to this is the target shape (= the op parameter)
  std::vector<std::shared_ptr<node>> inputs;
  float scale = 1.0f, offset = 0.0f;  // fused epilogue: op(...) * scale + offset
  float arg0 = 0.0f;  // op-specific scalar (attn_dec: softmax scale)
  int axis = 0;
  bool keepdims = false;

  // constant source / evaluated result
  storage stor;
  std::vector<int64_t> strides;
  int64_t soffset = 0;
  bool evaluated = false;
  uint64_t visit_mark = 0;  // graph::run visited stamp (see visit_counter)
};

using node_ptr = std::shared_ptr<node>;

struct graph;

// Evaluation hook (TL_RUNTIME_HOOKS; see storage.h). Installed alongside
// the storage/barrier hooks by tl::install_runtime_hooks().
inline void (*run_hook)(const std::vector<node_ptr>&) = nullptr;
// No-sync variant (graph::run_noflush) for view construction: kernels stay
// in flight; a later host read barriers. Null falls back to run_hook.
inline void (*run_noflush_hook)(const std::vector<node_ptr>&) = nullptr;

// Monotonic stamp for graph::run's visited marking (O(1), allocation-free;
// nodes are single-threaded like the rest of evaluation).
inline uint64_t visit_counter = 0;

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

  // True when the data is materialized (no pending lazy graph). Conservative:
  // an evaluated-but-not-yet-adopted node reads as false; any data access
  // adopts it.
  bool materialized() const { return storage_.buf != nullptr && !node_; }

  // Data access — forces evaluation. data() additionally requires a
  // contiguous array; raw() is the strided base pointer kernels consume.
  float* data();
  const float* data() const;
  const float* raw() const;
  float item() const;                              // size() == 1
  float at(std::initializer_list<int64_t>) const;  // strided single read

  // Raw device-buffer handle (CUDA mirror key), or null on host-only builds /
  // unevaluated arrays. The bridge for handing an evaluated array's device
  // buffer to an imperative cuda:: kernel (e.g. the kv_cache decode loop) —
  // eval() first, then pass native() as the q/k/v pointer. Contiguous, offset 0.
  void* native() const { return storage_.native; }

  // Views (zero-copy on the materialized result) and copies. View
  // construction realizes the source without a sync — pending GPU kernels
  // stay in flight, so a mid-graph view costs no pipeline drain.
  array transpose() const;                       // reverse all axes
  array transpose(std::vector<int> axes) const;  // permutation
  array reshape(shape_t shape) const;  // view when contiguous, else copy
  array slice(int64_t start, int64_t count) const;  // axis 0
  array clone() const;                              // contiguous copy

  // Storage dtype (M7). bf16 is a weight-container storage type: create with
  // to_bf16() (materializes, then narrows RNE); the CUDA decode GEMV consumes
  // it natively, every other op transparently widens to an F32 copy at eval.
  // Direct element access (data()/raw()/at()/item()) requires F32 — call
  // to_f32() first. Compute and results are always F32.
  tl::dtype dt() const { return storage_.dt; }
  array to_bf16() const;  // F32 -> bf16 contiguous copy
  array to_f32() const;   // bf16/q4 -> F32 contiguous copy (F32: returns *this)
  // F32 [K,N] weight -> group-symmetric int4 (M8). Logical shape stays [K,N];
  // storage is packed [N,K] int4 + per-group scales. The decode GEMV consumes
  // it natively; other ops dequantize via to_f32(). K % kQ4Group (32) required.
  array to_q4() const;

  // Elementwise (lazy)
  array exp() const;
  array log() const;
  array sqrt() const;
  array sigmoid() const;
  array relu() const;
  array softmax() const;  // last axis, numerically stable

  // Linear algebra: (M,K)@(K,N); 1-d operands promote numpy-style. (lazy)
  array dot(const array& b) const;

  // Fused decode attention (M9): out(h,:) = softmax(scale · q(h,:)·K(h)ᵀ)·V(h),
  // one query row per head. q [H,D], K/V [H,ctx,D], out [H,D]. On CUDA with
  // D==128 this is the fused flash-attention kernel; otherwise a CPU reference.
  static array attn_decode(const array& q, const array& K, const array& V,
                           float scale);

  // Transformer building blocks (M9 model surface). RoPE is a fused op (needs
  // cos/sin); RMSNorm/SiLU/SwiGLU are pure compositions of existing ops (so they
  // ride the tuned kernels and are autograd-ready when VJPs land). RoPE input is
  // [H,D] (decode, T=1) or [H,T,D] (prefill); `pos` is the position of t=0.
  static array rope(const array& x, int64_t pos, float base = 10000.0f);
  static array rmsnorm(const array& x, const array& weight, float eps = 1e-5f);
  static array silu(const array& x);
  static array swiglu(const array& gate, const array& up);

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

  // Launch this array's graph WITHOUT the terminal CtxSynchronize, adopting its
  // storage so native() is valid. The kernels stay in flight on the null
  // stream; a later same-stream kernel (kv_cache append/attn, GPU argmax) sees
  // the writes by stream ordering, and the single sync is deferred to the next
  // host read (raw()/data()) or explicit flush. Use ONLY when the consumer is a
  // GPU kernel — a host read of *this* array still needs eval()/raw(). This is
  // the lever that collapses a decode step's ~98 syncs to ~1; see run_noflush.
  const array& realize() const {
    realize_();
    return *this;
  }

 private:
  shape_t shape_;
  std::vector<int64_t> strides_;  // element units
  int64_t offset_ = 0;
  storage storage_;
  detail::node_ptr node_;  // set while lazy; cleared on adoption
  // Memoized constant wrap (graph::as_node). An array's layout and storage
  // handle never change after construction, so the wrap can never go stale;
  // in-place data writes (add_) flow through, since the node shares the
  // same storage. Saves the per-use node/vector allocations when the same
  // array (a weight, an activation) feeds many ops.
  mutable detail::node_ptr const_node_;

  static array make_(shape_t shape);
  void ensure_() const;    // materialize (evaluate + adopt) if lazy, then sync
  void realize_() const;   // same, but leave kernels in flight (no sync)
  void materialize_(bool do_flush) const;  // shared body

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

namespace detail {

// The one choke point making mixed CPU/GPU graphs safe: every CPU-side
// buffer access flushes pending GPU work first.
inline void barrier_() {
#ifdef TL_RUNTIME_HOOKS
  if (cpu_barrier_hook) cpu_barrier_hook();
#else
  gpu::cpu_barrier();
#endif
}

// Pull a storage's device copy back to host before a CPU access (D2H if the
// device holds the live version), and on a write mark the device copy stale so
// the next GPU op re-uploads. No-op on unified backends (Metal) and for heap
// storages (native==null). Pairs with barrier_(): flush first (kernels done),
// then reconcile this specific buffer.
inline void host_sync_(void* native, bool for_write) {
#ifdef TL_RUNTIME_HOOKS
  if (host_sync_hook) host_sync_hook(native, for_write);
#else
  gpu::sync_to_host(native, for_write);
#endif
}

}  // namespace detail

inline const float* array::raw() const {
  ensure_();
  if (storage_.dt != tl::dtype::f32) {
    throw std::logic_error("tl::raw: bf16 storage; use to_f32()");
  }
  detail::barrier_();
  detail::host_sync_(storage_.native, /*for_write=*/false);
  return storage_.data() + offset_;
}

inline float* array::data() {
  ensure_();
  if (storage_.dt != tl::dtype::f32) {
    throw std::logic_error("tl::data: bf16 storage; use to_f32()");
  }
  detail::barrier_();
  if (!contiguous()) {
    throw std::logic_error("tl::data: non-contiguous view; use clone()");
  }
  // Mutable handle: conservatively treat as a potential host write, so the
  // device mirror is invalidated (correctness over a rare redundant re-upload).
  detail::host_sync_(storage_.native, /*for_write=*/true);
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
  realize_();
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
  realize_();
  if (!contiguous()) return clone().reshape(std::move(shape));
  auto strides = detail::contiguous_strides(shape);
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::slice(int64_t start, int64_t count) const {
  if (rank() == 0 || start < 0 || start + count > shape_[0]) {
    throw std::invalid_argument("tl::slice: out of range");
  }
  realize_();
  auto shape = shape_;
  shape[0] = count;
  return make_view_(*this, std::move(shape), strides_,
                    offset_ + start * strides_[0]);
}

inline array array::clone() const {
  ensure_();
  if (storage_.dt != tl::dtype::f32) {
    // bf16 arrays are contiguous weight leaves (to_bf16 output); byte-copy.
    if (!contiguous() || offset_ != 0) {
      throw std::logic_error("tl::clone: non-contiguous bf16 view");
    }
    array out;
    out.shape_ = shape_;
    out.strides_ = strides_;
    out.storage_ = storage::make(size(), storage_.dt);
    detail::barrier_();
    detail::host_sync_(storage_.native, /*for_write=*/false);
    std::memcpy(out.storage_.data(), storage_.data(),
                static_cast<size_t>(size()) * dtype_size(storage_.dt));
    return out;
  }
  auto out = make_(shape_);
  auto* po = out.storage_.data();
  const auto* pi = raw();
  if (contiguous()) {
    std::memcpy(po, pi, static_cast<size_t>(size()) * sizeof(float));
    return out;
  }
  detail::for_each_index(
      shape_, {strides_},
      [&](int64_t i, const std::vector<int64_t>& off) { po[i] = pi[off[0]]; });
  return out;
}

// F32 -> bf16 contiguous copy (RNE narrow). The result is a materialized
// weight leaf; strided/transposed sources materialize through raw().
inline array array::to_bf16() const {
  ensure_();
  if (storage_.dt == tl::dtype::bf16) return *this;
  array out;
  out.shape_ = shape_;
  out.strides_ = detail::contiguous_strides(shape_);
  out.storage_ = storage::make(size(), tl::dtype::bf16);
  auto* po = reinterpret_cast<uint16_t*>(out.storage_.data());
  const float* pi = raw();
  if (contiguous()) {
    const int64_t n = size();
    for (int64_t i = 0; i < n; i++) po[i] = f32_to_bf16(pi[i]);
  } else {
    detail::for_each_index(shape_, {strides_},
                           [&](int64_t i, const std::vector<int64_t>& off) {
                             po[i] = f32_to_bf16(pi[off[0]]);
                           });
  }
  return out;
}

// bf16/q4 -> F32 contiguous copy.
inline array array::to_f32() const {
  ensure_();
  if (storage_.dt == tl::dtype::f32) return *this;
  detail::barrier_();
  detail::host_sync_(storage_.native, /*for_write=*/false);
  if (storage_.dt == tl::dtype::q4) {
    // Dequantize the packed [N,K] int4 + scales back to logical [K,N] F32.
    const int64_t K = shape_[0], N = shape_[1], G = tl::kQ4Group;
    array out = make_(shape_);  // [K, N]
    const auto* base = reinterpret_cast<const uint8_t*>(storage_.data());
    const uint8_t* qw = base;                       // [N][K/2]
    const float* sc = reinterpret_cast<const float*>(base + N * K / 2);
    float* po = out.storage_.data();
    for (int64_t nn = 0; nn < N; nn++) {
      const uint8_t* qrow = qw + nn * (K / 2);
      const float* srow = sc + nn * (K / G);
      for (int64_t k = 0; k < K; k++) {
        uint8_t byte = qrow[k / 2];
        int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
        po[k * N + nn] = srow[k / G] * (float)(nib - 8);
      }
    }
    return out;
  }
  array out = make_(shape_);
  const auto* pi = reinterpret_cast<const uint16_t*>(storage_.data()) + offset_;
  float* po = out.storage_.data();
  if (contiguous()) {
    const int64_t n = size();
    for (int64_t i = 0; i < n; i++) po[i] = bf16_to_f32(pi[i]);
  } else {
    detail::for_each_index(shape_, {strides_},
                           [&](int64_t i, const std::vector<int64_t>& off) {
                             po[i] = bf16_to_f32(pi[off[0]]);
                           });
  }
  return out;
}

// F32 [K,N] -> group-symmetric int4 weight. Packs the transpose [N,K]: 2
// nibbles/byte contiguous in k (word[n][k/8] slot k%8), scales per group of 32
// appended. Matches the tl_gemv_q4 kernel's layout.
inline array array::to_q4() const {
  ensure_();
  if (storage_.dt == tl::dtype::q4) return *this;
  if (rank() != 2) throw std::logic_error("tl::to_q4: expect [K,N]");
  const int64_t K = shape_[0], N = shape_[1], G = tl::kQ4Group;
  if (K % G != 0)
    throw std::logic_error("tl::to_q4: K must be a multiple of kQ4Group (32)");
  const float* pi = raw();  // [K,N], strided ok
  int64_t s0 = strides_[0], s1 = strides_[1];
  array out;
  out.shape_ = shape_;  // logical [K,N]
  out.strides_ = detail::contiguous_strides(shape_);
  out.storage_ = storage::make_bytes_(N * K, tl::q4_bytes(N, K), tl::dtype::q4);
  auto* base = reinterpret_cast<uint8_t*>(out.storage_.data());
  std::memset(base, 0, static_cast<size_t>(tl::q4_bytes(N, K)));
  uint8_t* qw = base;
  float* sc = reinterpret_cast<float*>(base + N * K / 2);
  for (int64_t nn = 0; nn < N; nn++) {
    uint8_t* qrow = qw + nn * (K / 2);
    float* srow = sc + nn * (K / G);
    for (int64_t g = 0; g < K / G; g++) {
      float maxabs = 1e-8f;
      for (int64_t j = 0; j < G; j++) {
        int64_t k = g * G + j;
        maxabs = std::max(maxabs, std::fabs(pi[k * s0 + nn * s1]));
      }
      float scale = maxabs / 7.0f;
      srow[g] = scale;
      for (int64_t j = 0; j < G; j++) {
        int64_t k = g * G + j;
        int q = (int)std::lround(pi[k * s0 + nn * s1] / scale);
        q = std::max(-8, std::min(7, q));
        uint8_t& byte = qrow[k / 2];
        unsigned nib = (unsigned)(q + 8);
        byte = (k & 1) ? ((byte & 0x0F) | (nib << 4)) : ((byte & 0xF0) | nib);
      }
    }
  }
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
  if (a.contiguous()) {  // flat loop: no walker, autovectorizes
    int64_t n = out.size();
    for (int64_t i = 0; i < n; i++) po[i] = f(pa[i]);
    return out;
  }
  for_each_index(a.shape(), {a.strides()},
                 [&](int64_t i, const std::vector<int64_t>& off) {
                   po[i] = f(pa[off[0]]);
                 });
  return out;
}

template <typename F>
array map_binary(const array& a, const array& b, F f) {
  if (a.shape() == b.shape() && a.contiguous() && b.contiguous()) {
    auto out = array::empty(a.shape());
    auto* po = out.data();
    const auto* pa = a.raw();
    const auto* pb = b.raw();
    int64_t n = out.size();
    for (int64_t i = 0; i < n; i++) po[i] = f(pa[i], pb[i]);
    return out;
  }
  auto shape = broadcast_shape(a.shape(), b.shape());
  auto out = array::empty(shape);
  auto* po = out.data();
  const auto* pa = a.raw();
  const auto* pb = b.raw();

  // Rank-2 contiguous broadcast fast path: covers the common matrix cases —
  // bias/row-vector [1,N], column-vector [M,1], and scalar broadcasts — with
  // flat row/col loops instead of the coordinate walker. broadcast_strides
  // already yields a 0 step on each broadcast axis, so a full operand steps
  // (N,1), a column vector (1,0), a row vector (0,1), a scalar (0,0). When the
  // inner step is 1 the inner loop is contiguous and vectorizes.
  if (shape.size() == 2 && a.contiguous() && b.contiguous()) {
    auto ra = broadcast_strides(a.shape(), a.strides(), shape);
    auto rb = broadcast_strides(b.shape(), b.strides(), shape);
    int64_t M = shape[0], N = shape[1];
    int64_t sa = ra[1], sb = rb[1];
    int64_t o = 0;
    for (int64_t i = 0; i < M; i++) {
      const float* pai = pa + i * ra[0];
      const float* pbi = pb + i * rb[0];
      // Hoist a broadcast operand's per-row value and split on the inner stride
      // so each variant is a stride-1 (or constant) inner loop that vectorizes;
      // the generic j*stride form is an unpredictable gather to the compiler.
      if (sa == 1 && sb == 1) {
        for (int64_t j = 0; j < N; j++, o++) po[o] = f(pai[j], pbi[j]);
      } else if (sa == 1 && sb == 0) {  // b constant within each row ([M,1])
        float bv = pbi[0];
        for (int64_t j = 0; j < N; j++, o++) po[o] = f(pai[j], bv);
      } else if (sa == 0 && sb == 1) {
        float av = pai[0];
        for (int64_t j = 0; j < N; j++, o++) po[o] = f(av, pbi[j]);
      } else {  // (0,0): both per-row scalars
        float av = pai[0], bv = pbi[0];
        for (int64_t j = 0; j < N; j++, o++) po[o] = f(av, bv);
      }
    }
    return out;
  }

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
  auto* po = out.data();
  const auto* pi = a.raw();

  // Contiguous fast path: split the row-major buffer into
  // outer × axis_len × inner and accumulate each axis slab into the matching
  // output slot with flat pointer loops — the accumulator index is just the
  // (outer, inner) position, so no per-element coordinate walk is needed.
  if (a.contiguous()) {
    const auto& sh = a.shape();
    int64_t axis_len = sh[axis];
    int64_t inner = 1, outer = 1;
    for (int i = axis + 1; i < r; i++) inner *= sh[i];
    for (int i = 0; i < axis; i++) outer *= sh[i];
    if (inner == 1) {
      // Last-axis reduction: accumulate each contiguous run into a LOCAL, then
      // store once. Accumulating straight into po[o] instead carries the
      // dependency through memory (store-to-load per element, no vectorize) —
      // ~40x slower here, and this is the common case (softmax denominators,
      // bias/feature-sum gradients, per-row norms).
      for (int64_t o = 0; o < outer; o++) {
        const float* base = pi + o * axis_len;
        float acc = init;
        for (int64_t k = 0; k < axis_len; k++) f(acc, base[k]);
        po[o] = acc;
      }
      return out;
    }
    // inner > 1: each po[j] is an independent accumulator, so the contiguous
    // inner loop vectorizes with no cross-element dependency.
    for (int64_t o = 0; o < outer; o++) {
      const float* base = pi + o * axis_len * inner;
      float* od = po + o * inner;
      for (int64_t k = 0; k < axis_len; k++) {
        const float* src = base + k * inner;
        for (int64_t j = 0; j < inner; j++) f(od[j], src[j]);
      }
    }
    return out;
  }

  // Generic (strided/broadcast/transposed) fallback: map each input index to
  // its accumulator, axis contributing stride 0.
  auto out_strides = contiguous_strides(out_shape);
  std::vector<int64_t> acc_strides(r, 0);
  for (int i = 0, oi = 0; i < r; i++) {
    if (i == axis) {
      if (keepdims) oi++;
      continue;
    }
    acc_strides[i] = out_strides[oi++];
  }
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
    if (a.const_node_) return a.const_node_;
    auto n = std::make_shared<node>();
    n->shape = a.shape_;
    n->stor = a.storage_;
    n->strides = a.strides_;
    n->soffset = a.offset_;
    n->evaluated = true;
    a.const_node_ = n;
    return n;
  }

  static array from_node(node_ptr n) {
    array a;
    a.shape_ = n->shape;
    a.strides_ = contiguous_strides(n->shape);
    a.node_ = std::move(n);
    return a;
  }

  // Eager-tiny: when the inputs are already materialized and the result is
  // small, run the flat loop NOW and skip the node / array-shell / eval
  // machinery entirely (~3x fewer allocations per op). Laziness is an
  // optimization, not a semantic: values are identical, and chains that
  // could fuse stay lazy because their intermediates aren't materialized.
  // Backward passes and optimizer updates — all-materialized by nature —
  // are exactly the tiny-tensor storm this targets.
  static constexpr int64_t kEagerTiny = 4096;

  static bool eager_cpu_ok_() {
    if (device_ == device_type::gpu) return false;
    // Never break a running GPU pipeline. Via the hook under TL_RUNTIME_HOOKS
    // so this builder path (always live in an embedder's core archive) links
    // no Metal symbol; a null hook means no GPU backend, nothing pending.
#ifdef TL_RUNTIME_HOOKS
    if (detail::gpu_pending_hook && detail::gpu_pending_hook()) return false;
#else
    if (gpu::pending()) return false;
#endif
    return true;  // cpu mode; in auto these sizes are below the threshold
  }

  static bool eager_operand_(const array& x) {
    // An evaluated-but-not-adopted node is materialized in all but name
    // (this is the normal state of forward values read by a backward pass);
    // adoption here is a few pointer moves, no evaluation.
    if (x.storage_.dt != tl::dtype::f32) return false;  // bf16: widen at eval
    if (x.node_ && x.node_->evaluated) x.ensure_();
    return x.materialized() && x.contiguous();
  }

  static array binary(op_t op, const array& a, const array& b) {
    if (a.shape() == b.shape() && num_elements(a.shape()) <= kEagerTiny &&
        eager_operand_(a) && eager_operand_(b) && eager_cpu_ok_()) {
      switch (op) {
        case op_t::add: return map_binary(a, b, std::plus<float>());
        case op_t::sub: return map_binary(a, b, std::minus<float>());
        case op_t::mul: return map_binary(a, b, std::multiplies<float>());
        case op_t::div: return map_binary(a, b, std::divides<float>());
        case op_t::pow_:
          return map_binary(a, b,
                            [](float x, float y) { return std::pow(x, y); });
        case op_t::gt:
          return map_binary(
              a, b, [](float x, float y) { return x > y ? 1.0f : 0.0f; });
        case op_t::lt:
          return map_binary(
              a, b, [](float x, float y) { return x < y ? 1.0f : 0.0f; });
        case op_t::ge:
          return map_binary(
              a, b, [](float x, float y) { return x >= y ? 1.0f : 0.0f; });
        case op_t::le:
          return map_binary(
              a, b, [](float x, float y) { return x <= y ? 1.0f : 0.0f; });
        case op_t::eq:
          return map_binary(
              a, b, [](float x, float y) { return x == y ? 1.0f : 0.0f; });
        case op_t::ne:
          return map_binary(
              a, b, [](float x, float y) { return x != y ? 1.0f : 0.0f; });
        default:
          break;  // not an elementwise binary — fall through to lazy
      }
    }
    auto n = std::make_shared<node>();
    n->op = op;
    n->shape = broadcast_shape(a.shape(), b.shape());  // throws early
    n->inputs = {as_node(a), as_node(b)};
    return from_node(std::move(n));
  }

  static array unary(op_t op, const array& a) {
    if (num_elements(a.shape()) <= kEagerTiny && eager_operand_(a) &&
        eager_cpu_ok_()) {
      switch (op) {
        case op_t::recip:
          return map_unary(a, [](float x) { return 1.0f / x; });
        case op_t::exp_:
          return map_unary(a, [](float x) { return std::exp(x); });
        case op_t::log_:
          return map_unary(a, [](float x) { return std::log(x); });
        case op_t::sqrt_:
          return map_unary(a, [](float x) { return std::sqrt(x); });
        case op_t::sigmoid:
          return map_unary(
              a, [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        case op_t::relu:
          return map_unary(a, [](float x) { return x > 0 ? x : 0.0f; });
        default:
          break;  // softmax etc. — fall through to lazy
      }
    }
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
    if (num_elements(a.shape()) <= kEagerTiny && eager_operand_(a) &&
        eager_cpu_ok_()) {
      return map_unary(a, [s, o](float x) { return x * s + o; });
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

  static array attn_decode(const array& q, const array& K, const array& V,
                           float scale) {
    const auto& sq = q.shape();
    const auto& sk = K.shape();
    const auto& sv = V.shape();
    if (sq.size() != 2 || sk.size() != 3 || sv.size() != 3 || sk != sv ||
        sq[0] != sk[0] || sq[1] != sk[2]) {
      throw std::invalid_argument(
          "tl::attn_decode: expect q[H,D], K/V[H,ctx,D] (matching) — got q " +
          shape_str(sq) + ", K " + shape_str(sk) + ", V " + shape_str(sv));
    }
    auto n = std::make_shared<node>();
    n->op = op_t::attn_dec;
    n->shape = sq;  // [H, D]
    n->arg0 = scale;
    n->inputs = {as_node(q), as_node(K), as_node(V)};
    return from_node(std::move(n));
  }

  static array rope(const array& x, int64_t pos, float base) {
    const auto& s = x.shape();
    if (s.size() < 2 || (s.back() & 1))
      throw std::invalid_argument(
          "tl::rope: expect rank>=2 with an even last dim (head_dim) — got " +
          shape_str(s));
    auto n = std::make_shared<node>();
    n->op = op_t::rope;
    n->shape = s;
    n->arg0 = base;
    n->axis = static_cast<int>(pos);
    n->inputs = {as_node(x)};
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

  // Metal eligibility: manual gpu mode dispatches whenever possible; cpu
  // mode never. auto follows silarray's two rules: (1) never break a
  // running GPU pipeline — a CPU op here would force a blocking flush; the
  // choice is asymmetric because picking GPU is sticky (drags downstream
  // ops along) while picking CPU commits nothing. (2) Otherwise start GPU
  // only above the per-kernel-class size threshold (types.h).
  static bool gpu_mode_(int64_t n, kernel_class kc) {
    if (!gpu::available()) return false;
    if (device_ == device_type::gpu) return true;
    if (device_ != device_type::auto_) return false;
    if (gpu::pending()) return true;
    return n >= auto_threshold_(kc);
  }

  static gpu::kop to_kop_(op_t op) {
    switch (op) {
      case op_t::add: return gpu::kop::add;
      case op_t::sub: return gpu::kop::sub;
      case op_t::mul: return gpu::kop::mul;
      case op_t::div: return gpu::kop::div;
      case op_t::pow_: return gpu::kop::pow_;
      case op_t::exp_: return gpu::kop::exp_;
      case op_t::log_: return gpu::kop::log_;
      case op_t::sqrt_: return gpu::kop::sqrt_;
      case op_t::sigmoid: return gpu::kop::sigmoid;
      case op_t::relu: return gpu::kop::relu;
      default: return gpu::kop::affine;
    }
  }

  // Broadcast (strided) variant of a binary kop; nullopt when the op has no
  // broadcast kernel.
  static std::optional<gpu::kop> to_bcast_kop_(op_t op) {
    switch (op) {
      case op_t::add: return gpu::kop::badd;
      case op_t::sub: return gpu::kop::bsub;
      case op_t::mul: return gpu::kop::bmul;
      case op_t::div: return gpu::kop::bdiv;
      case op_t::pow_: return gpu::kop::bpow;
      default: return std::nullopt;
    }
  }

  static std::optional<array> gpu_binary(const node& n, const array& a,
                                           const array& b) {
    if (!gpu_mode_(num_elements(n.shape), kernel_class::elementwise)) {
      return std::nullopt;
    }
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    if (a.contiguous() && b.contiguous() && a.shape() == b.shape()) {
      auto out = array::empty(n.shape);
      if (out.size() == 0) return out;
      if (!out.storage_.native) return std::nullopt;
      if (!gpu::binary(to_kop_(n.op), a.storage_.native, a.offset_ * 4,
                         b.storage_.native, b.offset_ * 4, out.storage_.native,
                         out.offset_ * 4, out.size(), n.scale, n.offset)) {
        return std::nullopt;
      }
      return out;
    }
    // Rank-2 broadcast (bias / row vector / column vector / scalar): one
    // stride-parameterized kernel keeps the op on the GPU — a CPU fallback
    // here drains the whole pending pipeline (commit + wait) mid-graph.
    if (n.shape.size() != 2 || !a.contiguous() || !b.contiguous()) {
      return std::nullopt;
    }
    auto k = to_bcast_kop_(n.op);
    if (!k) return std::nullopt;
    auto ra = broadcast_strides(a.shape(), a.strides(), n.shape);
    auto rb = broadcast_strides(b.shape(), b.strides(), n.shape);
    auto out = array::empty(n.shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::binary_bcast(*k, a.storage_.native, a.offset_ * 4, ra[0], ra[1],
                           b.storage_.native, b.offset_ * 4, rb[0], rb[1],
                           out.storage_.native, out.offset_ * 4, n.shape[0],
                           n.shape[1], n.scale, n.offset)) {
      return std::nullopt;
    }
    return out;
  }

  // Add the fused dot offset to a materialized GEMM result (accel/cpu take
  // the scale as alpha; offset is a cheap post-pass). No-op when offset==0.
  static void apply_dot_offset_(array& out, float offset) {
    if (offset == 0.0f) return;
    if (auto ofs = accel::affine(out, 1.0f, offset)) {
      out = std::move(*ofs);
    } else {
      out = map_unary(out, [offset](float x) { return x + offset; });
    }
  }

  // Own CPU backend (M5): C = scale * A @ B into `out`. Handles any 2-d
  // strided operands in place (packing is stride-aware — no materialization,
  // more general than accel's CBLAS-mappable layouts). Gated by cpu::enabled_
  // so oracle tests can force ref.
  static bool cpu_gemm(const array& a, const array& b, array& out,
                       float scale) {
    if (!cpu::enabled_) return false;
    int64_t m = a.shape()[0], k = a.shape()[1], nn = b.shape()[1];
    cpu::sgemm(a.raw(), a.strides()[0], a.strides()[1], b.raw(),
               b.strides()[0], b.strides()[1], out.data(), m, nn, k, scale);
    return true;
  }

  // Classify a 2-d operand's layout for the GEMM loaders: row-major
  // contiguous rows → (trans=false, ld=row stride); its transpose →
  // (trans=true, ld=col stride). Same test as accel::gemm. nullopt = not
  // GEMM-mappable (needs materialization, handled by falling to ref/accel).
  struct gemm_layout { bool trans; int64_t ld; };
  static std::optional<gemm_layout> gemm_classify_(const array& x) {
    int64_t r = x.shape()[0], c = x.shape()[1];
    int64_t s0 = x.strides()[0], s1 = x.strides()[1];
    if (s1 == 1 && s0 >= std::max<int64_t>(c, 1)) return gemm_layout{false, s0};
    if (s0 == 1 && s1 >= std::max<int64_t>(r, 1)) return gemm_layout{true, s1};
    return std::nullopt;
  }

  static std::optional<array> gpu_gemm(const node& n, const array& a_in,
                                         const array& b_in) {
    int64_t k_dim = a_in.shape().back();
    if (!gpu_mode_(num_elements(n.shape) * (k_dim > 0 ? k_dim : 1),
                     kernel_class::matmul)) {
      return std::nullopt;
    }
    array a = a_in.rank() == 1 ? a_in.reshape({1, a_in.size()}) : a_in;
    array b = b_in.rank() == 1 ? b_in.reshape({b_in.size(), 1}) : b_in;
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    auto la = gemm_classify_(a), lb = gemm_classify_(b);
    if (!la || !lb) return std::nullopt;
    int64_t m = a.shape()[0], k = a.shape()[1], nn = b.shape()[1];
    array out = array::empty({m, nn});
    if (!out.storage_.native) return std::nullopt;
    if (m == 0 || nn == 0) return out.reshape(n.shape);
    if (!gpu::gemm(a.storage_.native, a.offset_ * 4, la->ld, la->trans,
                     b.storage_.native, b.offset_ * 4, lb->ld, lb->trans,
                     out.storage_.native, out.offset_ * 4, m, nn, k, n.scale,
                     n.offset)) {
      return std::nullopt;
    }
    return out.reshape(n.shape);
  }

  // M7 decode GEMV: a(1,K)f32 @ B(K,N) -> (1,N)f32, B either f32 or bf16
  // weights. bf16 is the one op consuming bf16 storage natively; the f32
  // variant matters too — the 128×128-tile gemm wastes 127 rows at M=1, the
  // GEMV is the right kernel for decode on both dtypes (CUDA; Metal returns
  // false). Gated to the exact decode shape — batch row 1, contiguous
  // zero-offset operands. The kernel has no epilogue; scale/offset apply in
  // the generic tail (identity on the plain a.dot(W) decode path).
  static std::optional<array> gpu_gemv(const node& n, const array& a_in,
                                       const array& b) {
    if (b.rank() != 2) return std::nullopt;
    const bool bf16 = b.storage_.dt == tl::dtype::bf16;
    if (!bf16 && b.storage_.dt != tl::dtype::f32) return std::nullopt;
    array a = a_in.rank() == 1 ? a_in.reshape({1, a_in.size()}) : a_in;
    if (a.storage_.dt != tl::dtype::f32 || a.rank() != 2 || a.shape()[0] != 1)
      return std::nullopt;
    int64_t k = a.shape()[1], nn = b.shape()[1];
    if (!gpu_mode_(nn * (k > 0 ? k : 1), kernel_class::matmul))
      return std::nullopt;
    if (!a.contiguous() || a.offset_ != 0 || !b.contiguous() || b.offset_ != 0)
      return std::nullopt;
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    array out = array::empty({int64_t{1}, nn});
    if (!out.storage_.native) return std::nullopt;
    if (nn == 0) return out.reshape(n.shape);
    bool ok = bf16 ? gpu::gemv_bf16(a.storage_.native, b.storage_.native,
                                    out.storage_.native, nn, k)
                   : gpu::gemv_f32(a.storage_.native, b.storage_.native,
                                   out.storage_.native, nn, k);
    if (!ok) return std::nullopt;
    return out.reshape(n.shape);
  }

  // M8 int4-weight decode GEMV: a(1,K)f32 @ Wq(K,N)q4 -> (1,N)f32. Wq's logical
  // shape is [K,N]; its storage is packed [N,K] int4 + appended scales, so the
  // scales pointer is native + N·K/2 bytes (one buffer). Gated to the decode
  // shape; non-decode / non-GPU dequantizes to F32 via the input funnel.
  static std::optional<array> gpu_gemv_q4(const node& n, const array& a_in,
                                          const array& Wq) {
    if (Wq.storage_.dt != tl::dtype::q4 || Wq.rank() != 2) return std::nullopt;
    array a = a_in.rank() == 1 ? a_in.reshape({1, a_in.size()}) : a_in;
    if (a.storage_.dt != tl::dtype::f32 || a.rank() != 2 || a.shape()[0] != 1)
      return std::nullopt;
    int64_t K = Wq.shape()[0], N = Wq.shape()[1];  // logical [K,N]
    if (!gpu_mode_(N * (K > 0 ? K : 1), kernel_class::matmul))
      return std::nullopt;
    if (!a.contiguous() || a.offset_ != 0 || Wq.offset_ != 0) return std::nullopt;
    if (!a.storage_.native || !Wq.storage_.native) return std::nullopt;
    array out = array::empty({int64_t{1}, N});
    if (!out.storage_.native) return std::nullopt;
    if (N == 0) return out.reshape(n.shape);
    void* scales = reinterpret_cast<void*>(
        reinterpret_cast<char*>(Wq.storage_.native) + N * K / 2);
    if (!gpu::gemv_q4(a.storage_.native, Wq.storage_.native, scales,
                      out.storage_.native, N, K, tl::kQ4Group)) {
      return std::nullopt;
    }
    return out.reshape(n.shape);
  }

  // M9 fused decode attention on the GPU. q[H,D], K/V[H,ctx,D] contiguous,
  // D==128. Returns nullopt (→ CPU ref) when the kernel declines.
  static std::optional<array> gpu_attn_(const node& n, const array& q,
                                        const array& K, const array& V) {
    int64_t H = q.shape()[0], D = q.shape()[1], ctx = K.shape()[1];
    if (!gpu_mode_(H * ctx * D, kernel_class::matmul)) return std::nullopt;
    if (!q.contiguous() || q.offset_ != 0 || !K.contiguous() ||
        K.offset_ != 0 || !V.contiguous() || V.offset_ != 0)
      return std::nullopt;
    if (!q.storage_.native || !K.storage_.native || !V.storage_.native)
      return std::nullopt;
    array out = array::empty({H, D});
    if (!out.storage_.native) return std::nullopt;
    // Array path has no persistent cache: K/V are [H,ctx,D], so n_kv_heads==H
    // (no GQA) and kv_max==ctx (kv_stride==ctx*D degenerates to whole-buffer).
    if (!gpu::attn_decode(q.storage_.native, K.storage_.native,
                          V.storage_.native, out.storage_.native, H, H, ctx, ctx,
                          D, n.arg0)) {
      return std::nullopt;
    }
    return out;
  }

  // CPU reference decode attention (fallback / non-GPU builds).
  static array ref_attn_(const array& q, const array& K, const array& V,
                         float scale) {
    int64_t H = q.shape()[0], D = q.shape()[1], ctx = K.shape()[1];
    array out = array::empty({H, D});
    const float* pq = q.raw();
    const float* pk = K.raw();
    const float* pv = V.raw();
    float* po = out.data();
    std::vector<float> s(ctx);
    for (int64_t h = 0; h < H; h++) {
      const float* qh = pq + h * D;
      const float* Kh = pk + h * ctx * D;
      const float* Vh = pv + h * ctx * D;
      float mx = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < ctx; j++) {
        float acc = 0;
        for (int64_t d = 0; d < D; d++) acc += qh[d] * Kh[j * D + d];
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j < ctx; j++) {
        s[j] = std::exp(s[j] - mx);
        sum += s[j];
      }
      float* oh = po + h * D;
      for (int64_t d = 0; d < D; d++) {
        float acc = 0;
        for (int64_t j = 0; j < ctx; j++) acc += s[j] * Vh[j * D + d];
        oh[d] = acc / sum;
      }
    }
    return out;
  }

  // RoPE on the GPU. x [H,D] (T=1) or [H,T,D], contiguous. Row r's position is
  // pos + (r % T). Returns nullopt (→ CPU ref) when the kernel declines.
  static std::optional<array> gpu_rope_(const node& n, const array& x) {
    int64_t D = x.shape().back();
    if (D <= 0 || (D & 1)) return std::nullopt;
    int64_t rows = x.size() / D;
    int64_t T = x.rank() == 3 ? x.shape()[1] : 1;
    if (!gpu_mode_(x.size(), kernel_class::elementwise)) return std::nullopt;
    if (!x.contiguous() || x.offset_ != 0 || !x.storage_.native)
      return std::nullopt;
    array out = array::empty(x.shape());
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::rope(x.storage_.native, out.storage_.native, rows, T, D, n.axis,
                   n.arg0))
      return std::nullopt;
    return out;
  }

  // CPU reference RoPE (fallback / non-GPU builds). Half-split convention,
  // matching tl_rope: pairs (j, j+D/2) rotate by (pos + t)·base^(-2j/D).
  static array ref_rope_(const array& x, int64_t pos, float base) {
    int64_t D = x.shape().back();
    int64_t T = x.rank() == 3 ? x.shape()[1] : 1;
    int64_t rows = D ? x.size() / D : 0;
    int64_t half = D / 2;
    array out = array::empty(x.shape());
    const float* px = x.raw();
    float* po = out.data();
    for (int64_t r = 0; r < rows; r++) {
      int64_t t = T ? r % T : 0;
      double position = static_cast<double>(pos + t);
      const float* xr = px + r * D;
      float* orr = po + r * D;
      for (int64_t j = 0; j < half; j++) {
        double theta = std::pow(static_cast<double>(base),
                                -2.0 * static_cast<double>(j) / D);
        double ang = position * theta;
        float c = static_cast<float>(std::cos(ang));
        float s = static_cast<float>(std::sin(ang));
        float x0 = xr[j], x1 = xr[j + half];
        orr[j] = x0 * c - x1 * s;
        orr[j + half] = x0 * s + x1 * c;
      }
    }
    return out;
  }

  // Row op over the last axis of a contiguous input. `out_cols` is cols for
  // softmax (rows×cols out) or 1 for reductions (rows out).
  static std::optional<array> gpu_row(gpu::kop k, const array& a,
                                        shape_t out_shape, bool reduce,
                                        float scale, float offset) {
    if (!gpu_mode_(a.size(), kernel_class::reduction) || !a.contiguous() ||
        a.rank() == 0) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    int64_t cols = a.shape().back();
    int64_t rows = cols ? a.size() / cols : 0;
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::row_op(k, a.storage_.native, a.offset_ * 4, out.storage_.native,
                       out.offset_ * 4, rows, cols, scale, offset)) {
      return std::nullopt;
    }
    (void)reduce;
    return out;
  }

  static std::optional<array> gpu_unary(gpu::kop k, const array& a,
                                          float scale, float offset) {
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::unary(k, a.storage_.native, a.offset_ * 4, out.storage_.native,
                      out.offset_ * 4, out.size(), scale, offset)) {
      return std::nullopt;
    }
    return out;
  }

  // One topological pass over all roots (MLX-style batch eval), then each
  // node evaluates through eval_one. Iterative DFS: recursion depth must not
  // bound graph depth. do_flush=false leaves the launched kernels in flight on
  // the null stream (no CtxSynchronize) — the caller's realize() path, where a
  // later same-stream kernel consumes the result and a single terminal sync
  // drains the whole batch (collapses a decode step's ~98 syncs to ~1).
  static void run(const std::vector<node_ptr>& roots) { run_(roots, true); }
  static void run_noflush(const std::vector<node_ptr>& roots) {
    run_(roots, false);
  }
  static void run_(const std::vector<node_ptr>& roots, bool do_flush) {
    // Thread-local scratch: run() fires once per eval batch and tiny-graph
    // workloads are per-op-allocation-bound. Nested evaluation cannot happen
    // (kernels never build or evaluate graphs), so reuse is safe. Visited
    // marking is a per-run stamp on the node — O(1), allocation-free.
    thread_local std::vector<node*> order;
    thread_local std::vector<std::pair<node*, size_t>> stack;
    order.clear();
    stack.clear();
    const uint64_t stamp = ++visit_counter;
    for (const auto& root : roots) {
      if (!root || root->evaluated || root->visit_mark == stamp) continue;
      root->visit_mark = stamp;
      stack.emplace_back(root.get(), 0);
      while (!stack.empty()) {
        auto& [n, i] = stack.back();
        if (i < n->inputs.size()) {
          node* in = n->inputs[i++].get();
          if (!in->evaluated && in->visit_mark != stamp) {
            in->visit_mark = stamp;
            stack.emplace_back(in, 0);
          }
        } else {
          order.push_back(n);
          stack.pop_back();
        }
      }
    }
    for (auto* n : order) eval_one(*n);
    if (do_flush) gpu::flush();  // blocking eval: batch done when run() returns
  }

  // Allocation-free contiguity check on node metadata (result/constant
  // strides vs the node's shape).
  static bool node_contig_(const node& in) {
    if (in.strides.size() != in.shape.size()) return false;
    int64_t expected = 1;
    for (size_t r = in.shape.size(); r-- > 0;) {
      if (in.strides[r] != expected) return false;
      expected *= in.shape[r];
    }
    return true;
  }

  static void store_raw_(node& n, storage&& out) {
    n.stor = std::move(out);
    n.strides = contiguous_strides(n.shape);
    n.soffset = 0;
    n.evaluated = true;
  }

  // Tiny-tensor fast path: contiguous same-shape elementwise ops evaluate
  // as flat loops straight on the input nodes' storage — no wrap arrays, no
  // output array shell, no walker. This is where per-op-allocation-bound
  // workloads (microgpt-class, 16–256 element tensors) spend their time;
  // above the cutoff the accel/metal paths win and the cost being shaved
  // here is noise. The epilogue folds into the same loop.
  static bool try_fast_ew_(node& n) {
    constexpr int64_t kCutoff = 4096;
    using op_t = node::op_t;
    int64_t numel = num_elements(n.shape);
    if (numel == 0 || numel > kCutoff) return false;
    if (gpu_mode_(numel, kernel_class::elementwise)) return false;
    if (n.inputs.empty()) return false;
    const node& a = *n.inputs[0];
    if (a.stor.dt != tl::dtype::f32) return false;  // bf16 widens in eval_one
    if (a.shape != n.shape || !node_contig_(a)) return false;
    const float* pa = a.stor.data() + a.soffset;
    const float s = n.scale, o = n.offset;
    const bool epi = s != 1.0f || o != 0.0f;

    auto unary_loop = [&](auto f) {
      detail::barrier_();
      storage out = storage::make(numel);
      float* po = out.data();
      if (epi) {
        for (int64_t i = 0; i < numel; i++) po[i] = f(pa[i]) * s + o;
      } else {
        for (int64_t i = 0; i < numel; i++) po[i] = f(pa[i]);
      }
      store_raw_(n, std::move(out));
      return true;
    };

    switch (n.op) {
      case op_t::add:
      case op_t::sub:
      case op_t::mul:
      case op_t::div:
      case op_t::pow_: {
        const node& b = *n.inputs[1];
        if (b.shape != n.shape || !node_contig_(b)) return false;
        const float* pb = b.stor.data() + b.soffset;
        auto binary_loop = [&](auto f) {
          detail::barrier_();
          storage out = storage::make(numel);
          float* po = out.data();
          if (epi) {
            for (int64_t i = 0; i < numel; i++) po[i] = f(pa[i], pb[i]) * s + o;
          } else {
            for (int64_t i = 0; i < numel; i++) po[i] = f(pa[i], pb[i]);
          }
          store_raw_(n, std::move(out));
          return true;
        };
        switch (n.op) {
          case op_t::add: return binary_loop(std::plus<float>());
          case op_t::sub: return binary_loop(std::minus<float>());
          case op_t::mul: return binary_loop(std::multiplies<float>());
          case op_t::div: return binary_loop(std::divides<float>());
          default:
            return binary_loop(
                [](float x, float y) { return std::pow(x, y); });
        }
      }
      case op_t::affine:
        return unary_loop([](float x) { return x; });
      case op_t::recip:
        return unary_loop([](float x) { return 1.0f / x; });
      case op_t::exp_:
        return unary_loop([](float x) { return std::exp(x); });
      case op_t::log_:
        return unary_loop([](float x) { return std::log(x); });
      case op_t::sqrt_:
        return unary_loop([](float x) { return std::sqrt(x); });
      case op_t::sigmoid:
        return unary_loop(
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
      case op_t::relu:
        return unary_loop([](float x) { return x > 0 ? x : 0.0f; });
      default:
        return false;
    }
  }

  // Tiny matmul fast path: strided triple loop straight on node storage.
  // Below the cutoff the cblas/metal call overhead and the wrap/out array
  // shells cost more than the multiply itself (microgpt-class attention/MLP
  // matmuls are [16,16]@[16,1]). Handles rank-2 × rank-2 with arbitrary
  // strides (transposed views included); everything else falls through.
  static bool try_fast_dot_(node& n) {
    constexpr int64_t kCutoff = 16384;  // M*N*K
    const node& a = *n.inputs[0];
    const node& b = *n.inputs[1];
    if (a.stor.dt != tl::dtype::f32 || b.stor.dt != tl::dtype::f32)
      return false;  // bf16 operands take the eval_one dot path
    if (a.shape.size() != 2 || b.shape.size() != 2) return false;
    int64_t m = a.shape[0], k = a.shape[1], nn = b.shape[1];
    if (m * nn * k > kCutoff || m * nn == 0) return false;
    if (gpu_mode_(m * nn * k, kernel_class::matmul)) return false;
    detail::barrier_();
    storage out = storage::make(m * nn);
    const float* pa = a.stor.data() + a.soffset;
    const float* pb = b.stor.data() + b.soffset;
    float* po = out.data();
    int64_t as0 = a.strides[0], as1 = a.strides[1];
    int64_t bs0 = b.strides[0], bs1 = b.strides[1];
    const float s = n.scale, o = n.offset;
    for (int64_t i = 0; i < m; i++) {
      for (int64_t j = 0; j < nn; j++) {
        float acc = 0.0f;
        for (int64_t l = 0; l < k; l++) {
          acc += pa[i * as0 + l * as1] * pb[l * bs0 + j * bs1];
        }
        po[i * nn + j] = acc * s + o;
      }
    }
    store_raw_(n, std::move(out));
    return true;
  }

  static void eval_one(node& n) {
    if (n.op != node::op_t::constant && try_fast_ew_(n)) return;
    if (n.op == node::op_t::dot && try_fast_dot_(n)) return;
    // Input funnel. bf16 inputs widen to an F32 copy here — the universal
    // fallback that keeps every backend kernel F32-only; the sole native bf16
    // consumer (decode GEMV) intercepts in the dot case before this runs.
    auto in = [&](size_t i) {
      array x = wrap(*n.inputs[i]);
      if (x.storage_.dt != tl::dtype::f32) x = x.to_f32();
      return x;
    };
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
        if (auto g = gpu_binary(n, a, b)) {
          r = std::move(*g);
          epi_done = true;  // kernels apply the epilogue in the store
        } else if (auto o = accel::binary(n.op, a, b)) {
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
      case op_t::pow_: {
        auto a = in(0), b = in(1);
        if (auto g = gpu_binary(n, a, b)) {
          r = std::move(*g);
          epi_done = true;  // kernel applies the epilogue in the store
        } else {
          r = map_binary(a, b,
                         [](float x, float y) { return std::pow(x, y); });
        }
        break;
      }
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
        if (auto g = gpu_unary(gpu::kop::affine, a, s, o)) {
          r = std::move(*g);
        } else if (auto out = accel::affine(a, s, o)) {
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
        if (auto g = gpu_unary(to_kop_(n.op), a, n.scale, n.offset)) {
          r = std::move(*g);
          epi_done = true;
        } else if (auto o = accel::unary(n.op, a)) {
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
      case op_t::sigmoid: {
        auto a = in(0);
        if (auto g = gpu_unary(gpu::kop::sigmoid, a, n.scale, n.offset)) {
          r = std::move(*g);
          epi_done = true;
        } else {
          r = map_unary(a,
                        [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        }
        break;
      }
      case op_t::softmax: {
        auto a = in(0);
        if (auto g = gpu_row(gpu::kop::softmax, a, a.shape(), false, 1.0f,
                               0.0f)) {
          r = std::move(*g);
        } else {
          r = ref::softmax(a);
        }
        break;
      }
      case op_t::dot: {
        // Decode GEMV fast path first (M=1; f32/bf16/q4 weights), on the
        // un-widened inputs.
        if (auto g = gpu_gemv(n, wrap(*n.inputs[0]), wrap(*n.inputs[1]))) {
          r = std::move(*g);
          break;  // kernel is epilogue-free; generic tail applies scale/offset
        }
        if (auto g = gpu_gemv_q4(n, wrap(*n.inputs[0]), wrap(*n.inputs[1]))) {
          r = std::move(*g);
          break;
        }
        auto a = in(0), b = in(1);
        if (auto g = gpu_gemm(n, a, b)) {
          r = std::move(*g);
          epi_done = true;
          break;
        }
        array a2 = a.rank() == 1 ? a.reshape({1, a.size()}) : a;
        array b2 = b.rank() == 1 ? b.reshape({b.size(), 1}) : b;
        array out = array::empty({a2.shape()[0], b2.shape()[1]});
        if (accel::gemm(a2, b2, out, n.scale)) {  // epilogue scale = alpha
          apply_dot_offset_(out, n.offset);
          r = out.reshape(n.shape);
          epi_done = true;
        } else if (cpu_gemm(a2, b2, out, n.scale)) {  // own CPU backend
          apply_dot_offset_(out, n.offset);
          r = out.reshape(n.shape);
          epi_done = true;
        } else {
          r = ref::dot(a, b);
        }
        break;
      }
      case op_t::attn_dec: {
        if (auto g = gpu_attn_(n, wrap(*n.inputs[0]), wrap(*n.inputs[1]),
                               wrap(*n.inputs[2]))) {
          r = std::move(*g);
        } else {
          r = ref_attn_(in(0), in(1), in(2), n.arg0);
        }
        break;
      }
      case op_t::rope: {
        if (auto g = gpu_rope_(n, wrap(*n.inputs[0]))) {
          r = std::move(*g);
        } else {
          r = ref_rope_(in(0), n.axis, n.arg0);
        }
        break;
      }
      case op_t::sum_ax:
      case op_t::max_ax: {
        // GPU row reductions cover the last-axis case (softmax/argmax
        // support shape); other axes fall to the CPU oracle. The epilogue
        // applies in the kernel, so mark it done.
        auto a = in(0);
        gpu::kop k =
            n.op == op_t::sum_ax ? gpu::kop::row_sum : gpu::kop::row_max;
        if (n.axis == static_cast<int>(a.rank()) - 1) {
          if (auto g = gpu_row(k, a, n.shape, true, n.scale, n.offset)) {
            r = std::move(*g);
            epi_done = true;
            break;
          }
        }
        r = n.op == op_t::sum_ax ? ref::sum(a, n.axis, n.keepdims)
                                 : ref::max(a, n.axis, n.keepdims);
        break;
      }
      case op_t::mean_ax: {
        // Last-axis mean lowers to the row_sum kernel with 1/cols folded into
        // the epilogue scale — no dedicated kernel, and no CPU fallback that
        // would drain the GPU pipeline mid-graph (layer-norm's op mix).
        auto a = in(0);
        if (n.axis == static_cast<int>(a.rank()) - 1 &&
            a.shape().back() > 0) {
          float inv = 1.0f / static_cast<float>(a.shape().back());
          if (auto g = gpu_row(gpu::kop::row_sum, a, n.shape, true,
                               n.scale * inv, n.offset)) {
            r = std::move(*g);
            epi_done = true;
            break;
          }
        }
        r = ref::mean(a, n.axis, n.keepdims);
        break;
      }
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

inline void array::materialize_(bool do_flush) const {
  if (!node_) return;
  if (!node_->evaluated) {
#ifdef TL_RUNTIME_HOOKS
    if (!detail::run_hook) {
      throw std::logic_error(
          "tl: evaluation before install_runtime_hooks() (TL_RUNTIME_HOOKS)");
    }
    // Thread-local scratch: single-root evals fire per op in tiny-tensor
    // workloads; an initializer-list vector per call adds up.
    thread_local std::vector<detail::node_ptr> root;
    root.assign(1, node_);
    if (do_flush || !detail::run_noflush_hook)
      detail::run_hook(root);
    else
      detail::run_noflush_hook(root);
    root.clear();
#else
    thread_local std::vector<detail::node_ptr> root;
    root.assign(1, node_);
    if (do_flush)
      detail::graph::run(root);
    else
      detail::graph::run_noflush(root);
    root.clear();
#endif
  }
  auto* self = const_cast<array*>(this);
  self->storage_ = node_->stor;
  self->strides_ = node_->strides;
  self->offset_ = node_->soffset;
  // The evaluated node doubles as the constant wrap for future uses of
  // this array as an input. Drop its input edges first: they are spent
  // (everything is evaluated), and releasing them returns consumed
  // intermediates' buffers to the pool as early as possible.
  node_->inputs.clear();
  self->const_node_ = std::move(self->node_);
}

inline void array::ensure_() const { materialize_(/*do_flush=*/true); }
inline void array::realize_() const { materialize_(/*do_flush=*/false); }

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
  // Fast path first: gradient accumulation is same-shape contiguous +=
  // on tiny tensors, where the generic walker's setup would dominate.
  if (shape_ == b.shape() && contiguous() && b.contiguous()) {
    const auto* pb = b.raw();
    auto* po = storage_.data() + offset_;
    int64_t n = size();
    for (int64_t i = 0; i < n; i++) po[i] += pb[i];
    return *this;
  }
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

inline array array::attn_decode(const array& q, const array& K, const array& V,
                                float scale) {
  return detail::graph::attn_decode(q, K, V, scale);
}

inline array array::rope(const array& x, int64_t pos, float base) {
  return detail::graph::rope(x, pos, base);
}

// RMSNorm: x · rsqrt(mean(x², last) + eps) · weight. Pure composition — the
// mean/sqrt/mul kernels are already tuned, and this is autograd-ready.
inline array array::rmsnorm(const array& x, const array& weight, float eps) {
  array ms = (x * x).mean(static_cast<int>(x.rank()) - 1, /*keepdims=*/true);
  array inv = 1.0f / (ms + eps).sqrt();  // [.,1] broadcasts over the last dim
  return (x * inv) * weight;
}

inline array array::silu(const array& x) { return x * x.sigmoid(); }

inline array array::swiglu(const array& gate, const array& up) {
  return silu(gate) * up;
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

// Install the execution-engine hooks (TL_RUNTIME_HOOKS builds). Call once,
// before any evaluation, from the embedder's tensor feature loader. This is
// the only function referencing the evaluator and device backends by name —
// keep it out of translation units that must stay backend-free.
inline void install_runtime_hooks() {
  detail::storage_make_hook = &storage::make_device_;
  detail::cpu_barrier_hook = &gpu::cpu_barrier;
  detail::host_sync_hook = &gpu::sync_to_host;
  detail::gpu_pending_hook = &gpu::pending;
  detail::run_hook = &detail::graph::run;
  detail::run_noflush_hook = &detail::graph::run_noflush;
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
