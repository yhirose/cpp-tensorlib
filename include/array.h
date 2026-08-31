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
#include <shape.h>
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
  std::vector<int64_t> out(s.size());
  contiguous_strides_into(s.data(), static_cast<int>(s.size()), out.data());
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

// Validates and normalizes a (possibly negative, numpy-style) axis against
// `rank`. Shared by slice/pad/unfold so the bounds check and the negative-
// axis convention `reduce_shape` above already uses for sum/mean/max/argmax
// live in exactly one place.
inline size_t normalize_axis(int axis, size_t rank, const char* who) {
  int r = static_cast<int>(rank);
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) {
    throw std::invalid_argument(std::string("tl::") + who + ": bad axis");
  }
  return static_cast<size_t>(axis);
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
    tanh_, sin_, cos_,
    clamp_,  // clip(x, arg0=min, arg1=max) — the only unary op needing two
             // node-specific scalars, hence `arg1` below.
    softmax,
    where_,
    dot,
    attn_dec,  // fused decode attention: softmax(arg0 · q·Kᵀ)·V
    rope,      // rotary position embedding (arg0=base, axis=position offset)
    sum_ax, mean_ax, max_ax, argmax_ax, sum_to_,
    pad_,   // zero-pad axis `axis` by `arg0` (=before) elements; `shape` is
            // the padded target (`after` is derivable: shape[axis] - before
            // - input.shape()[axis]). A real write, not a view — see pad_.
    fold_,  // unfold's inverse: scatter-add axis `axis` (step=`arg0`) back
            // down to `shape`, accumulating overlaps. Also a real write.
    index_select_,  // row gather along axis 0. inputs={table, indices};
                    // `shape` is the output (indices.size() rows).
    index_add_,     // index_select's dual: scatter-add along axis 0.
                    // inputs={indices, values}; `shape` is the target
                    // (an explicit param, same idea as sum_to_).
    scatter_axis_,  // one-hot scatter into a new trailing axis. inputs=
                    // {indices, values}, same shape; `shape` is theirs
                    // with that axis (size = shape.back()) appended.
    view_,  // zero-copy view (transpose/reshape/slice/unfold) over a still-
            // lazy source: composes strides at eval, no kernel, no flush —
            // keeps the source in the same batch instead of forcing a
            // boundary. unfold reuses view_axes[0]=axis, view_start=step
            // (its window size is just shape.back(), needs no field).
  };

  op_t op = op_t::constant;
  shape_t shape;  // for sum_to this is the target shape (= the op parameter)
  std::vector<std::shared_ptr<node>> inputs;
  float scale = 1.0f, offset = 0.0f;  // fused epilogue: op(...) * scale + offset
  float arg0 = 0.0f;  // op-specific scalar: attn_dec's softmax scale,
                      // pad_'s `before`, fold_'s `step`, or clamp_'s `min`
                      // (whichever the op needs one generic slot for,
                      // rather than growing this struct further — `axis`
                      // below covers another scalar some of them need).
  float arg1 = 0.0f;  // clamp_'s `max` — the one op needing a second
                      // scalar; everything else leaves this at 0.
  int axis = 0;
  bool keepdims = false;

  // view_ parameters (only meaningful when op == view_). view_axes is empty
  // for reshape (no allocation on the common non-view node); it holds the
  // permutation for transpose, or a single axis (one element) for slice/
  // unfold — reused rather than adding a dedicated field for one int.
  enum class vkind : uint8_t { transpose, reshape, slice, unfold };
  vkind view_kind = vkind::reshape;
  std::vector<int> view_axes;  // transpose permutation, or [axis] for slice/unfold
  int64_t view_start = 0;      // slice's start, or unfold's step, along
                               // view_axes[0] (slice: 0 if view_axes empty)

  // constant source / evaluated result
  storage stor;
  std::vector<int64_t> strides;
  int64_t soffset = 0;
  bool evaluated = false;
  uint64_t visit_mark = 0;  // graph::run visited stamp (see visit_counter)
};

using node_ptr = std::shared_ptr<node>;

// The scalar math of each elementwise op, written ONCE — every site that has to
// compute one of these on the CPU (the eager-tiny builder, the tiny-tensor fast
// path, eval_one's fallback) names the functor from here, so no two can drift.
// Comparisons yield F32 masks (1.0 / 0.0). `affine` is deliberately absent: it
// is the epilogue carrier (scale/offset), not a fixed scalar function.
inline constexpr auto ew_pow = [](float x, float y) { return std::pow(x, y); };
inline constexpr auto ew_gt = [](float x, float y) { return x > y ? 1.0f : 0.0f; };
inline constexpr auto ew_lt = [](float x, float y) { return x < y ? 1.0f : 0.0f; };
inline constexpr auto ew_ge = [](float x, float y) { return x >= y ? 1.0f : 0.0f; };
inline constexpr auto ew_le = [](float x, float y) { return x <= y ? 1.0f : 0.0f; };
inline constexpr auto ew_eq = [](float x, float y) { return x == y ? 1.0f : 0.0f; };
inline constexpr auto ew_ne = [](float x, float y) { return x != y ? 1.0f : 0.0f; };
inline constexpr auto ew_recip = [](float x) { return 1.0f / x; };
inline constexpr auto ew_exp = [](float x) { return std::exp(x); };
inline constexpr auto ew_log = [](float x) { return std::log(x); };
inline constexpr auto ew_sqrt = [](float x) { return std::sqrt(x); };
inline constexpr auto ew_sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
inline constexpr auto ew_relu = [](float x) { return x > 0 ? x : 0.0f; };
inline constexpr auto ew_tanh = [](float x) { return std::tanh(x); };
inline constexpr auto ew_sin = [](float x) { return std::sin(x); };
inline constexpr auto ew_cos = [](float x) { return std::cos(x); };

// Dispatch helper for the sites that write their result into an EXISTING array
// (they pay nothing for the callback, unlike the eager builders — see
// graph::binary). Calls f(functor) for a matching op and returns true; false for
// anything else, so the caller can fall through to its own path.
template <typename F>
bool visit_binary_op(node::op_t op, F&& f) {
  using op_t = node::op_t;
  switch (op) {
    case op_t::add: f(std::plus<float>()); return true;
    case op_t::sub: f(std::minus<float>()); return true;
    case op_t::mul: f(std::multiplies<float>()); return true;
    case op_t::div: f(std::divides<float>()); return true;
    case op_t::pow_: f(ew_pow); return true;
    case op_t::gt: f(ew_gt); return true;
    case op_t::lt: f(ew_lt); return true;
    case op_t::ge: f(ew_ge); return true;
    case op_t::le: f(ew_le); return true;
    case op_t::eq: f(ew_eq); return true;
    case op_t::ne: f(ew_ne); return true;
    default: return false;
  }
}

template <typename F>
bool visit_unary_op(node::op_t op, F&& f) {
  using op_t = node::op_t;
  switch (op) {
    case op_t::recip: f(ew_recip); return true;
    case op_t::exp_: f(ew_exp); return true;
    case op_t::log_: f(ew_log); return true;
    case op_t::sqrt_: f(ew_sqrt); return true;
    case op_t::sigmoid: f(ew_sigmoid); return true;
    case op_t::relu: f(ew_relu); return true;
    case op_t::tanh_: f(ew_tanh); return true;
    case op_t::sin_: f(ew_sin); return true;
    case op_t::cos_: f(ew_cos); return true;
    default: return false;
  }
}

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

// Whole-batch GPU bias for auto_ mode: set by graph::run_ for the span of one
// evaluation batch when its total matmul work crosses batch_matmul_bias_
// threshold_ (types.h), read by gpu_mode_. thread_local because each eval
// thread runs its own batch.
inline thread_local bool batch_gpu_bias_ = false;

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
  array slice(int axis, int64_t start, int64_t count) const;  // any axis
  // Zero-pad `axis` by `before`/`after` elements on each side. Not a view —
  // allocates the padded buffer and writes `*this` into it via `add_`
  // through a `slice` (so it needs nothing beyond what already exists:
  // `zeros` + the axis-general `slice` above + `add_`'s already-tested
  // write-through-a-view behavior). Compose two calls for 2-D (H then W)
  // padding, the same way two `unfold` calls give a 2-D window.
  array pad(int axis, int64_t before, int64_t after) const;
  // Sliding-window view over `axis` — shape gains a trailing `size` axis,
  // `axis` itself becomes the window count (PyTorch's `Tensor.unfold`).
  // Zero-copy: reuses the same base storage as transpose/reshape/slice, via
  // a stride smaller than `axis`'s extent (the same trick broadcast's
  // stride-0 already relies on, generalized to a nonzero overlap). Forces
  // materialization first rather than deferring through the lazy graph as
  // a view node — see the TODO at the call site for what a `vkind::unfold`
  // would need.
  array unfold(int axis, int64_t size, int64_t step) const;
  // `unfold`'s inverse (PyTorch's `Tensor.fold`, generalized to any axis):
  // scatter-adds `*this` (shaped like some `x.unfold(axis, size, step)`)
  // back into a fresh `orig_size`-along-`axis` buffer, accumulating every
  // overlap — the gradient a differentiable caller's backward needs. Unlike
  // every view above, this genuinely computes (can't be a stride trick:
  // overlapping windows write to the same destination element more than
  // once), so it's a real O(size(*this)) walk, not a zero-copy view.
  array fold(int axis, int64_t orig_size, int64_t step) const;
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
  array tanh() const;
  array sin() const;
  array cos() const;
  array clamp(float lo, float hi) const;
  array softmax() const;  // last axis, numerically stable

  // Linear algebra: (M,K)@(K,N); 1-d operands promote numpy-style. (lazy)
  array dot(const array& b) const;

  // Row gather along axis 0: out[i] = (*this)[indices[i]] (a 1-D index
  // array; indices are float-valued like argmax's own output, rounded to
  // int). Output shape is indices.size() followed by this array's own
  // trailing dims. The embedding-table lookup PyTorch calls
  // `Tensor.index_select(0, index)` / `nn.Embedding`. Differentiable
  // w.r.t. `*this` -- see index_add below, its exact dual.
  array index_select(const array& indices) const;

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
  // Build a deferred view node over this array's still-lazy source (node_ set).
  // The strides/offset are the caller-computed view layout; eval composes them
  // against the source's evaluated storage without a kernel or a flush.
  array lazy_view_(detail::node::vkind kind, shape_t vshape,
                   std::vector<int64_t> vstrides, int64_t voffset,
                   std::vector<int> axes, int64_t start) const;

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

// index_select's exact dual: scatter-add `values` (shaped like some
// t.index_select(indices)) into a fresh zero buffer of `target_shape` --
// the embedding-table gradient PyTorch computes via
// `grad_weight.index_add_(0, index, grad_output)`. Repeated indices
// accumulate (unlike scatter_to_axis below, this one has real write
// conflicts, so the GPU kernel needs atomics). indices get no gradient.
array index_add(const array& indices, const array& values,
                shape_t target_shape);

// One-hot scatter into a new trailing axis of size `size`: out[..., k] =
// values[...] where indices[...] == k, else 0 -- no accumulation, since
// every input position maps to a distinct output slot (the axis is
// brand new, not shared across inputs). The native counterpart of
// argmax's own host-loop backward: `max(axis)`/`argmax(axis)` pick one
// element out of a window; this scatters a gradient back into that same
// window shape -- e.g. a pooling layer's own hand-derived backward. No
// native VJP yet: that would need this op's own dual (gather one element
// per position back out of the window axis), which no caller needs today.
array scatter_to_axis(const array& indices, const array& values,
                      int64_t size);

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

inline array array::lazy_view_(detail::node::vkind kind, shape_t vshape,
                               std::vector<int64_t> vstrides, int64_t voffset,
                               std::vector<int> axes, int64_t start) const {
  auto n = std::make_shared<detail::node>();
  n->op = detail::node::op_t::view_;
  n->view_kind = kind;
  n->view_axes = std::move(axes);
  n->view_start = start;
  n->shape = vshape;
  n->inputs = {node_};  // the still-lazy source computation
  array v;
  v.shape_ = std::move(vshape);
  v.strides_ = std::move(vstrides);
  v.offset_ = voffset;
  v.node_ = std::move(n);
  return v;
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
  // strides_ tracks the eventual layout even while lazy: a computation node
  // materializes contiguous (from_node sets contiguous strides), while a lazy
  // view node carries its real (possibly strided) layout.
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
    throw std::logic_error(std::string("tl::raw: ") +
                           tl::dtype_name(storage_.dt) +
                           " storage; use to_f32()");
  }
  detail::barrier_();
  detail::host_sync_(storage_.native, /*for_write=*/false);
  return storage_.data() + offset_;
}

inline float* array::data() {
  ensure_();
  if (storage_.dt != tl::dtype::f32) {
    throw std::logic_error(std::string("tl::data: ") +
                           tl::dtype_name(storage_.dt) +
                           " storage; use to_f32()");
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
  shape_t shape(rank());
  std::vector<int64_t> strides(rank());
  for (size_t i = 0; i < rank(); i++) {
    shape[i] = shape_[axes[i]];
    strides[i] = strides_[axes[i]];
  }
  if (node_) {  // lazy source: defer as a view node — no batch boundary
    return lazy_view_(detail::node::vkind::transpose, std::move(shape),
                      std::move(strides), offset_, std::move(axes), 0);
  }
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::reshape(shape_t shape) const {
  if (detail::num_elements(shape) != size()) {
    throw std::invalid_argument("tl::reshape: size mismatch");
  }
  auto strides = detail::contiguous_strides(shape);
  if (node_ && contiguous()) {  // lazy contiguous source: defer, no boundary
    return lazy_view_(detail::node::vkind::reshape, shape, std::move(strides),
                      offset_, {}, 0);
  }
  realize_();
  if (!contiguous()) return clone().reshape(std::move(shape));
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

inline array array::slice(int64_t start, int64_t count) const {
  return slice(0, start, count);
}

inline array array::slice(int axis, int64_t start, int64_t count) const {
  size_t ax = detail::normalize_axis(axis, rank(), "slice");
  if (start < 0 || count < 0 || start + count > shape_[ax]) {
    throw std::invalid_argument("tl::slice: out of range");
  }
  auto shape = shape_;
  shape[ax] = count;
  int64_t voff = offset_ + start * strides_[ax];
  if (node_) {  // lazy source: defer as a view node — no batch boundary
    // `{}` for axis 0 keeps the common case allocation-free, same as before
    // this overload existed; only a non-zero axis needs `view_axes` at all.
    std::vector<int> axes = ax == 0 ? std::vector<int>{} : std::vector<int>{static_cast<int>(ax)};
    return lazy_view_(detail::node::vkind::slice, std::move(shape), strides_,
                      voff, std::move(axes), start);
  }
  realize_();
  return make_view_(*this, std::move(shape), strides_, voff);
}

// `pad`/`fold` are defined out-of-line further down (near `sum_to`), once
// `detail::graph` — which their bodies delegate to — is a complete type;
// `unfold` doesn't need that (it only calls other `array` methods and the
// free `detail::normalize_axis`, both already visible here).
inline array array::unfold(int axis, int64_t win, int64_t step) const {
  size_t ax = detail::normalize_axis(axis, rank(), "unfold");
  int64_t n = shape_[ax];
  if (win <= 0 || win > n || step <= 0) {
    throw std::invalid_argument("tl::unfold: bad size/step");
  }
  int64_t nwin = (n - win) / step + 1;

  shape_t shape = shape_;
  shape[ax] = nwin;
  shape.push_back(win);

  std::vector<int64_t> strides = strides_;
  int64_t base_stride = strides_[ax];
  strides[ax] = base_stride * step;
  strides.push_back(base_stride);

  if (node_) {  // lazy source: defer as a view node — no batch boundary
    return lazy_view_(detail::node::vkind::unfold, std::move(shape),
                      std::move(strides), offset_,
                      {static_cast<int>(ax)}, step);
  }
  realize_();
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
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

// Batched matmul: `a`/`b` share every leading (batch) dim exactly (checked
// by graph::dot before this ever runs), and the trailing two dims of each
// do one 2-D matmul per batch slice. Walks the batch index generically
// (any rank, any strides — `a`/`b` need not be contiguous) rather than
// assuming a flat batch stride, since only the innermost two strides are
// guaranteed contiguous-shaped by anything upstream.
inline array bdot(const array& a, const array& b) {
  size_t r = a.rank();
  const auto& ash = a.shape();
  const auto& as = a.strides();
  const auto& bs = b.strides();
  int64_t m = ash[r - 2], k = ash[r - 1], n = b.shape().back();
  shape_t out_shape(ash.begin(), ash.end() - 2);
  out_shape.push_back(m);
  out_shape.push_back(n);
  auto out = array::zeros(out_shape);
  auto* po = out.data();
  const auto* pa = a.raw();
  const auto* pb = b.raw();
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < r; i++) batch *= ash[i];
  int64_t as0 = as[r - 2], as1 = as[r - 1];
  int64_t bs0 = bs[r - 2], bs1 = bs[r - 1];
  int64_t out_mn = m * n;
  std::vector<int64_t> idx(r - 2, 0);
  for (int64_t bi = 0; bi < batch; bi++) {
    int64_t a_off = 0, b_off = 0;
    for (size_t d = 0; d < r - 2; d++) {
      a_off += idx[d] * as[d];
      b_off += idx[d] * bs[d];
    }
    const float* pa2 = pa + a_off;
    const float* pb2 = pb + b_off;
    float* po2 = po + bi * out_mn;
    for (int64_t i = 0; i < m; i++) {
      for (int64_t l = 0; l < k; l++) {
        float av = pa2[i * as0 + l * as1];
        for (int64_t j = 0; j < n; j++) {
          po2[i * n + j] += av * pb2[l * bs0 + j * bs1];
        }
      }
    }
    for (size_t d = r - 2; d-- > 0;) {
      if (++idx[d] < ash[d]) break;
      idx[d] = 0;
    }
  }
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

// Places `a` into a zero buffer of `out_shape`, shifted by `before` along
// `axis` — a real write (pad_'s non-overlapping placement has no stride
// trick that could make it a view: the destination is strictly bigger than
// the source). `for_each_index` walks `a`'s own shape, computing the
// destination offset with `out_shape`'s strides as if `a` started at
// position 0 along `axis`; adding the constant `before`-sized shift lands
// each element at its real padded position.
inline array pad(const array& a, size_t axis, int64_t before,
                 const shape_t& out_shape) {
  auto out = array::zeros(out_shape);
  auto* po = out.data();
  const auto* pi = a.raw();
  auto dst_strides = out.strides();
  int64_t shift = before * dst_strides[axis];
  detail::for_each_index(a.shape(), {a.strides(), dst_strides},
                         [&](int64_t, const std::vector<int64_t>& off) {
                           po[off[1] + shift] = pi[off[0]];
                         });
  return out;
}

// unfold's inverse: scatter-add `a` (shaped like some `x.unfold(axis, win,
// step)`) back into a zero buffer of `out_shape` (x's original shape),
// accumulating every overlap. Genuinely computes rather than reinterprets
// strides — two different source elements can target the same destination
// slot, which no view can express — so this is its own manual index walk
// rather than `for_each_index` (built for same-shape, different-strides
// sources, not a rank-changing many-to-one map).
inline array fold(const array& a, size_t axis, int64_t step,
                  const shape_t& out_shape) {
  auto out = array::zeros(out_shape);
  auto* po = out.data();
  const auto* pi = a.raw();
  auto out_strides = out.strides();
  size_t r = a.rank(), last = r - 1;
  const auto& a_shape = a.shape();
  const auto& a_strides = a.strides();
  std::vector<int64_t> idx(r, 0);
  int64_t n = a.size();
  for (int64_t i = 0; i < n; i++) {
    int64_t src_off = 0;
    for (size_t d = 0; d < r; d++) src_off += idx[d] * a_strides[d];
    int64_t dst_off = 0;
    for (size_t d = 0; d < last; d++) {
      int64_t di = (d == axis) ? idx[d] * step + idx[last] : idx[d];
      dst_off += di * out_strides[d];
    }
    po[dst_off] += pi[src_off];
    for (size_t d = r; d-- > 0;) {
      if (++idx[d] < a_shape[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

// Row gather along axis 0: out[i] = a[indices[i]] (indices float-valued,
// rounded, matching argmax's own convention). A full odometer over
// `out`'s own shape (same rank as `a`) rather than `for_each_index` --
// `indices[idx[0]]` makes the row 0 stride data-dependent, which no fixed
// reindex could express (the same reason fold above needs a manual walk).
inline array index_select(const array& a, const array& indices,
                          const shape_t& out_shape) {
  auto out = array::empty(out_shape);
  auto* po = out.data();
  const auto* pi = a.raw();
  const auto* pidx = indices.raw();
  size_t r = a.rank();
  const auto& a_strides = a.strides();
  const auto& out_strides = out.strides();
  std::vector<int64_t> idx(r, 0);
  int64_t n = out.size();
  for (int64_t i = 0; i < n; i++) {
    int64_t row = static_cast<int64_t>(std::llround(pidx[idx[0]]));
    int64_t src_off = row * a_strides[0];
    int64_t dst_off = 0;
    for (size_t d = 0; d < r; d++) dst_off += idx[d] * out_strides[d];
    for (size_t d = 1; d < r; d++) src_off += idx[d] * a_strides[d];
    po[dst_off] = pi[src_off];
    for (size_t d = r; d-- > 0;) {
      if (++idx[d] < out_shape[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

// index_select's dual: scatter-add `values` (shaped like some
// t.index_select(indices)) into a zero buffer of `target_shape`. Repeated
// indices really do accumulate here (`+=`), unlike scatter_to_axis below.
inline array index_add(const array& indices, const array& values,
                       const shape_t& target_shape) {
  auto out = array::zeros(target_shape);
  auto* po = out.data();
  const auto* pv = values.raw();
  const auto* pidx = indices.raw();
  size_t r = target_shape.size();
  const auto& v_shape = values.shape();
  const auto& v_strides = values.strides();
  const auto& out_strides = out.strides();
  std::vector<int64_t> idx(r, 0);
  int64_t n = values.size();
  for (int64_t i = 0; i < n; i++) {
    int64_t row = static_cast<int64_t>(std::llround(pidx[idx[0]]));
    int64_t src_off = 0;
    for (size_t d = 0; d < r; d++) src_off += idx[d] * v_strides[d];
    int64_t dst_off = row * out_strides[0];
    for (size_t d = 1; d < r; d++) dst_off += idx[d] * out_strides[d];
    po[dst_off] += pv[src_off];
    for (size_t d = r; d-- > 0;) {
      if (++idx[d] < v_shape[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

// One-hot scatter into a new trailing axis: out[..., k] = values[...] where
// indices[...] == k, else 0. Every input position writes a distinct output
// slot (the axis is brand new), so unlike index_add above there is no
// accumulation and no write conflict -- `out` starting zeroed is enough.
inline array scatter_to_axis(const array& indices, const array& values,
                             const shape_t& out_shape) {
  auto out = array::zeros(out_shape);
  auto* po = out.data();
  const auto* pv = values.raw();
  const auto* pidx = indices.raw();
  size_t r = indices.rank();
  const auto& v_shape = values.shape();
  const auto& v_strides = values.strides();
  const auto& idx_strides = indices.strides();
  const auto& out_strides = out.strides();  // rank r+1, contiguous (fresh)
  std::vector<int64_t> idx(r, 0);
  int64_t n = values.size();
  for (int64_t i = 0; i < n; i++) {
    int64_t v_off = 0, idx_off = 0, out_off = 0;
    for (size_t d = 0; d < r; d++) {
      v_off += idx[d] * v_strides[d];
      idx_off += idx[d] * idx_strides[d];
      out_off += idx[d] * out_strides[d];
    }
    int64_t k = static_cast<int64_t>(std::llround(pidx[idx_off]));
    po[out_off + k] = pv[v_off];
    for (size_t d = r; d-- > 0;) {
      if (++idx[d] < v_shape[d]) break;
      idx[d] = 0;
    }
  }
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

namespace detail {

// Classify a 2-d operand's layout for the GEMM loaders: row-major contiguous
// rows -> (trans=false, ld=row stride); its transpose -> (trans=true, ld=col
// stride). nullopt = not GEMM-mappable (needs materialization; the caller falls
// back to ref::). Shared by accel::gemm (CBLAS) and graph::gpu_gemm — reading
// the same test twice is how the two would drift.
struct gemm_layout {
  bool trans;
  int64_t ld;
};
inline std::optional<gemm_layout> gemm_classify_(const array& x) {
  int64_t r = x.shape()[0], c = x.shape()[1];
  int64_t s0 = x.strides()[0], s1 = x.strides()[1];
  if (s1 == 1 && s0 >= std::max<int64_t>(c, 1)) return gemm_layout{false, s0};
  if (s0 == 1 && s1 >= std::max<int64_t>(r, 1)) return gemm_layout{true, s1};
  return std::nullopt;
}

}  // namespace detail

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
  auto la = detail::gemm_classify_(a), lb = detail::gemm_classify_(b);
  if (!la || !lb) return false;
  auto cb = [](bool t) { return t ? CblasTrans : CblasNoTrans; };
  int64_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
  if (m == 0 || n == 0) return true;  // out has no elements
  if (k == 0) {
    float* po = out.data();
    for (int64_t i = 0; i < m * n; i++) po[i] = 0.0f;
    return true;
  }
  cblas_sgemm(CblasRowMajor, cb(la->trans), cb(lb->trans), static_cast<int>(m),
              static_cast<int>(n), static_cast<int>(k), alpha, a.raw(),
              static_cast<int>(la->ld), b.raw(), static_cast<int>(lb->ld), 0.0f,
              out.data(), static_cast<int>(n));
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
      // A direct switch, NOT visit_binary_op: the visitor's callback returns
      // void, so the result has to land in a named `array` and be moved out,
      // which measures ~7% on this path (16-element operands) — and per-op
      // allocation count is exactly what the eager-tiny path exists to save.
      // The visitor is used at the sites that assign into an existing result.
      switch (op) {
        case op_t::add: return map_binary(a, b, std::plus<float>());
        case op_t::sub: return map_binary(a, b, std::minus<float>());
        case op_t::mul: return map_binary(a, b, std::multiplies<float>());
        case op_t::div: return map_binary(a, b, std::divides<float>());
        case op_t::pow_:
          return map_binary(a, b, ew_pow);
        case op_t::gt: return map_binary(a, b, ew_gt);
        case op_t::lt: return map_binary(a, b, ew_lt);
        case op_t::ge: return map_binary(a, b, ew_ge);
        case op_t::le: return map_binary(a, b, ew_le);
        case op_t::eq: return map_binary(a, b, ew_eq);
        case op_t::ne: return map_binary(a, b, ew_ne);
        default: break;  // not an elementwise binary — fall through to lazy
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
      switch (op) {  // direct switch, see graph::binary
        case op_t::recip: return map_unary(a, ew_recip);
        case op_t::exp_: return map_unary(a, ew_exp);
        case op_t::log_: return map_unary(a, ew_log);
        case op_t::sqrt_: return map_unary(a, ew_sqrt);
        case op_t::sigmoid: return map_unary(a, ew_sigmoid);
        case op_t::relu: return map_unary(a, ew_relu);
        case op_t::tanh_: return map_unary(a, ew_tanh);
        case op_t::sin_: return map_unary(a, ew_sin);
        case op_t::cos_: return map_unary(a, ew_cos);
        default: break;  // softmax / affine / clamp etc. — fall through to lazy
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

  static array index_select(const array& a, const array& indices) {
    if (a.rank() < 1) {
      throw std::invalid_argument("tl::index_select: rank too low");
    }
    if (indices.rank() != 1) {
      throw std::invalid_argument("tl::index_select: indices must be rank-1");
    }
    auto n = std::make_shared<node>();
    n->op = op_t::index_select_;
    n->shape = a.shape();
    n->shape[0] = indices.shape()[0];
    n->inputs = {as_node(a), as_node(indices)};
    return from_node(std::move(n));
  }

  static array index_add(const array& indices, const array& values,
                         shape_t target_shape) {
    if (indices.rank() != 1) {
      throw std::invalid_argument("tl::index_add: indices must be rank-1");
    }
    if (target_shape.empty()) {
      throw std::invalid_argument("tl::index_add: rank too low");
    }
    shape_t expect = target_shape;
    expect[0] = indices.shape()[0];
    if (values.shape() != expect) {
      throw std::invalid_argument("tl::index_add: values shape mismatch");
    }
    auto n = std::make_shared<node>();
    n->op = op_t::index_add_;
    n->shape = std::move(target_shape);
    n->inputs = {as_node(indices), as_node(values)};
    return from_node(std::move(n));
  }

  static array scatter_to_axis(const array& indices, const array& values,
                               int64_t size) {
    if (indices.shape() != values.shape()) {
      throw std::invalid_argument(
          "tl::scatter_to_axis: indices/values shape mismatch");
    }
    if (size < 1) {
      throw std::invalid_argument("tl::scatter_to_axis: bad size");
    }
    auto n = std::make_shared<node>();
    n->op = op_t::scatter_axis_;
    n->shape = indices.shape();
    n->shape.push_back(size);
    n->inputs = {as_node(indices), as_node(values)};
    return from_node(std::move(n));
  }

  // clamp(x, lo, hi): the one unary op needing two node-specific scalars
  // (arg0=lo, arg1=hi), so it can't go through the generic unary() builder
  // above (which has no params to carry them).
  static array clamp(const array& a, float lo, float hi) {
    auto n = std::make_shared<node>();
    n->op = op_t::clamp_;
    n->shape = a.shape();
    n->arg0 = lo;
    n->arg1 = hi;
    n->inputs = {as_node(a)};
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

  static array pad(const array& a, int axis, int64_t before, int64_t after) {
    size_t ax = normalize_axis(axis, a.rank(), "pad");
    if (before < 0 || after < 0) {
      throw std::invalid_argument("tl::pad: negative pad");
    }
    auto n = std::make_shared<node>();
    n->op = op_t::pad_;
    n->shape = a.shape();
    n->shape[ax] = n->shape[ax] + before + after;
    n->axis = static_cast<int>(ax);
    n->arg0 = static_cast<float>(before);
    n->inputs = {as_node(a)};
    return from_node(std::move(n));
  }

  static array fold(const array& a, int axis, int64_t orig_size,
                    int64_t step) {
    if (a.rank() < 1) throw std::invalid_argument("tl::fold: rank too low");
    size_t ax = normalize_axis(axis, a.rank() - 1, "fold");
    if (orig_size < 1 || step < 1) {
      throw std::invalid_argument("tl::fold: bad orig_size/step");
    }
    auto n = std::make_shared<node>();
    n->op = op_t::fold_;
    n->shape = shape_t(a.shape().begin(), a.shape().end() - 1);
    n->shape[ax] = orig_size;
    n->axis = static_cast<int>(ax);
    n->arg0 = static_cast<float>(step);
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
    // Batched: rank >= 3 on both sides, leading (batch) dims matching
    // exactly — no broadcasting yet, kept simple until a real workload
    // needs more (attention's own q/k/v always share their batch/head
    // dims). The last two dims of each do the 2-D matmul every batch slice
    // gets; eval_one's op_t::dot dispatches on rank the same way this
    // validates it.
    if (sa.size() >= 3 || sb.size() >= 3) {
      if (sa.size() != sb.size() || sa.size() < 3 ||
          !std::equal(sa.begin(), sa.end() - 2, sb.begin()) ||
          sa.back() != sb[sb.size() - 2]) {
        throw std::invalid_argument("tl::dot: batched shape mismatch " +
                                    shape_str(sa) + " @ " + shape_str(sb));
      }
      auto n = std::make_shared<node>();
      n->op = op_t::dot;
      n->shape = shape_t(sa.begin(), sa.end() - 2);
      n->shape.push_back(sa[sa.size() - 2]);
      n->shape.push_back(sb.back());
      n->inputs = {as_node(a), as_node(b)};
      return from_node(std::move(n));
    }
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
    if (batch_gpu_bias_) return true;  // batch pinned to GPU (see run_)
    return n >= auto_threshold_(kc);
  }

  // The GPU kernel for an op, or nullopt when the backend has none (masks,
  // recip). nullopt is the honest answer: an "affine" fallback here would run
  // the identity kernel and silently return the input.
  static std::optional<gpu::kop> to_kop_(op_t op) {
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
      default: return std::nullopt;
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
    auto k = to_kop_(n.op);
    if (k && a.contiguous() && b.contiguous() && a.shape() == b.shape()) {
      auto out = array::empty(n.shape);
      if (out.size() == 0) return out;
      if (!out.storage_.native) return std::nullopt;
      if (!gpu::binary(*k, a.storage_.native, a.offset_ * 4,
                         b.storage_.native, b.offset_ * 4, out.storage_.native,
                         out.offset_ * 4, out.size(), n.scale, n.offset)) {
        return std::nullopt;
      }
      return out;
    }
    // Rank-2 broadcast (bias / row vector / column vector / scalar): one
    // stride-parameterized kernel keeps the op on the GPU — a CPU fallback
    // here drains the whole pending pipeline (commit + wait) mid-graph.
    if (!a.contiguous() || !b.contiguous()) {
      return std::nullopt;
    }
    auto bk = to_bcast_kop_(n.op);
    if (!bk) return std::nullopt;
    // Any other rank (a Transformer's [N,S,D] LayerNorm broadcasting a
    // [N,S,1] mean, say): the general N-D kernel this rank-2 one predates.
    if (n.shape.size() != 2) return gpu_binary_bcast_nd_(n, a, b, *bk);
    auto ra = broadcast_strides(a.shape(), a.strides(), n.shape);
    auto rb = broadcast_strides(b.shape(), b.strides(), n.shape);
    auto out = array::empty(n.shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::binary_bcast(*bk, a.storage_.native, a.offset_ * 4, ra[0], ra[1],
                           b.storage_.native, b.offset_ * 4, rb[0], rb[1],
                           out.storage_.native, out.offset_ * 4, n.shape[0],
                           n.shape[1], n.scale, n.offset)) {
      return std::nullopt;
    }
    return out;
  }

  // gpu_binary's own rank-2 path, generalized to any rank -- see
  // gpu::binary_bcast_nd's own comment for the concrete Transformer-shaped
  // caller (LayerNorm's [N,S,D] - [N,S,1] mean subtraction, rank 3).
  static std::optional<array> gpu_binary_bcast_nd_(const node& n,
                                                    const array& a,
                                                    const array& b,
                                                    gpu::kop bk) {
    int rank = static_cast<int>(n.shape.size());
    if (rank <= 0) return std::nullopt;
    auto ra = broadcast_strides(a.shape(), a.strides(), n.shape);
    auto rb = broadcast_strides(b.shape(), b.strides(), n.shape);
    auto out = array::empty(n.shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    std::vector<int64_t> out_shape_v(n.shape.begin(), n.shape.end());
    if (!gpu::binary_bcast_nd(bk, a.storage_.native, a.offset_ * 4, ra.data(),
                              b.storage_.native, b.offset_ * 4, rb.data(),
                              out.storage_.native, out.offset_ * 4,
                              out_shape_v.data(), rank, out.size(), n.scale,
                              n.offset)) {
      return std::nullopt;
    }
    return out;
  }

  // Tensor.where's GPU dispatch (any rank) -- existed on no backend before
  // this (eval_one's where_ case always ran the CPU map_ternary). Masking
  // (attention/padding masks) is the concrete caller; `n.shape` is
  // eval_one's own already-broadcast output shape (where_'s graph builder
  // computes it via a double broadcast_shape at node-construction time),
  // not recomputed here.
  static std::optional<array> gpu_where_(const shape_t& out_shape,
                                         const array& cond, const array& a,
                                         const array& b) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise) ||
        !cond.contiguous() || !a.contiguous() || !b.contiguous()) {
      return std::nullopt;
    }
    if (!cond.storage_.native || !a.storage_.native || !b.storage_.native) {
      return std::nullopt;
    }
    int rank = static_cast<int>(out_shape.size());
    if (rank <= 0) return std::nullopt;
    auto rc = broadcast_strides(cond.shape(), cond.strides(), out_shape);
    auto ra = broadcast_strides(a.shape(), a.strides(), out_shape);
    auto rb = broadcast_strides(b.shape(), b.strides(), out_shape);
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    std::vector<int64_t> out_shape_v(out_shape.begin(), out_shape.end());
    if (!gpu::where_nd(cond.storage_.native, cond.offset_ * 4, rc.data(),
                       a.storage_.native, a.offset_ * 4, ra.data(),
                       b.storage_.native, b.offset_ * 4, rb.data(),
                       out.storage_.native, out.offset_ * 4,
                       out_shape_v.data(), rank, out.size())) {
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

  // Batched matmul, GPU dispatch (v1): a per-slice loop over the same
  // gpu::gemm single-matmul entry point above, one launch per batch element
  // on the same device queue — real backend acceleration per slice, not yet
  // one fused batched kernel (no backend has one; a future optimization if
  // profiling shows the per-slice launch count matters more than the FLOPs
  // do). Requires both operands fully contiguous (batch dims included), not
  // just gemm_classify_'s "2-D contiguous-or-transposed" — gpu::gemm only
  // takes a flat byte offset per call, so a non-contiguous batch axis has no
  // single stride to compute that offset from; falls back to the CPU oracle
  // (ref::bdot) honestly rather than guessing. No scale/offset (matching
  // gpu_gemm's caller): eval_one's shared epilogue applies those afterward.
  static std::optional<array> gpu_bdot_(const array& a, const array& b) {
    size_t r = a.rank();
    int64_t m = a.shape()[r - 2], k = a.shape()[r - 1], nn = b.shape().back();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < r; i++) batch *= a.shape()[i];
    if (!gpu_mode_(m * nn * (k > 0 ? k : 1) * (batch > 0 ? batch : 1),
                   kernel_class::matmul)) {
      return std::nullopt;
    }
    if (!a.contiguous() || !b.contiguous()) return std::nullopt;
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    shape_t out_shape(a.shape().begin(), a.shape().end() - 2);
    out_shape.push_back(m);
    out_shape.push_back(nn);
    auto out = array::empty(out_shape);
    if (!out.storage_.native) return std::nullopt;
    if (m == 0 || nn == 0 || batch == 0) return out;
    int64_t a_stride = m * k, b_stride = k * nn, out_stride = m * nn;
    for (int64_t bi = 0; bi < batch; bi++) {
      int64_t a_off = (a.offset_ + bi * a_stride) * 4;
      int64_t b_off = (b.offset_ + bi * b_stride) * 4;
      int64_t out_off = (out.offset_ + bi * out_stride) * 4;
      if (!gpu::gemm(a.storage_.native, a_off, k, false, b.storage_.native,
                     b_off, nn, false, out.storage_.native, out_off, m, nn,
                     k, 1.0f, 0.0f)) {
        return std::nullopt;
      }
    }
    return out;
  }

  // The decode-shape gate the three weight dtypes share: the activation must be
  // a materialized, GPU-resident, contiguous [1,K] F32 row, and the op must be
  // GPU-eligible at N·K. Returns the promoted [1,K] activation, or nullopt when
  // this isn't a decode GEMV (the caller then falls through to the GEMM path).
  static std::optional<array> gemv_act_(const array& a_in, int64_t K,
                                        int64_t N) {
    array a = a_in.rank() == 1 ? a_in.reshape({1, a_in.size()}) : a_in;
    if (a.storage_.dt != tl::dtype::f32 || a.rank() != 2 || a.shape()[0] != 1)
      return std::nullopt;
    if (!gpu_mode_(N * (K > 0 ? K : 1), kernel_class::matmul))
      return std::nullopt;
    if (!a.contiguous() || a.offset_ != 0 || !a.storage_.native)
      return std::nullopt;
    return a;
  }

  // M7 decode GEMV: a(1,K)f32 @ B(K,N) -> (1,N)f32, B either f32 or bf16
  // weights. bf16 is the one op consuming bf16 storage natively; the f32
  // variant matters too — the 128×128-tile gemm wastes 127 rows at M=1, the
  // GEMV is the right kernel for decode on both dtypes (CUDA; Metal returns
  // false). The kernel has no epilogue; scale/offset apply in the generic tail
  // (identity on the plain a.dot(W) decode path).
  static std::optional<array> gpu_gemv(const node& n, const array& a_in,
                                       const array& b) {
    if (b.rank() != 2) return std::nullopt;
    const bool bf16 = b.storage_.dt == tl::dtype::bf16;
    if (!bf16 && b.storage_.dt != tl::dtype::f32) return std::nullopt;
    int64_t k = b.shape()[0], nn = b.shape()[1];  // b is [K,N] (dot checked it)
    auto a = gemv_act_(a_in, k, nn);
    if (!a || !b.contiguous() || b.offset_ != 0 || !b.storage_.native)
      return std::nullopt;
    array out = array::empty({int64_t{1}, nn});
    if (!out.storage_.native) return std::nullopt;
    if (nn == 0) return out.reshape(n.shape);
    bool ok = bf16 ? gpu::gemv_bf16(a->storage_.native, b.storage_.native,
                                    out.storage_.native, nn, k)
                   : gpu::gemv_f32(a->storage_.native, b.storage_.native,
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
    int64_t K = Wq.shape()[0], N = Wq.shape()[1];  // logical [K,N]
    auto a = gemv_act_(a_in, K, N);
    // No contiguity test on Wq: q4 bytes aren't elems×width, so its strides are
    // nominal — the zero offset is what says the packed buffer starts at base.
    if (!a || Wq.offset_ != 0 || !Wq.storage_.native) return std::nullopt;
    array out = array::empty({int64_t{1}, N});
    if (!out.storage_.native) return std::nullopt;
    if (N == 0) return out.reshape(n.shape);
    void* scales = reinterpret_cast<void*>(
        reinterpret_cast<char*>(Wq.storage_.native) + N * K / 2);
    if (!gpu::gemv_q4(a->storage_.native, Wq.storage_.native, scales,
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

  // Row op over the last axis of a contiguous input. `out_shape` is the input
  // shape for softmax (rows×cols out) or the reduced shape for row_sum/row_max
  // (one value per row) — the kop already says which, so nothing more is needed.
  static std::optional<array> gpu_row(gpu::kop k, const array& a,
                                      shape_t out_shape, float scale,
                                      float offset) {
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
    return out;
  }

  static std::optional<array> gpu_unary(std::optional<gpu::kop> k,
                                        const array& a, float scale,
                                        float offset) {
    if (!k) return std::nullopt;  // no kernel for this op on this backend
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::unary(*k, a.storage_.native, a.offset_ * 4, out.storage_.native,
                      out.offset_ * 4, out.size(), scale, offset)) {
      return std::nullopt;
    }
    return out;
  }

  // Shared setup for gpu_pad_/gpu_fold_ below: validates the gate + `a`'s
  // contiguity (every gpu_* helper above shares that requirement, and it's
  // what lets a backend read `a[i]` at a flat thread id without also
  // uploading strides), allocates `out`, and hands both shapes plus `out` to
  // `dispatch` to make the one backend call pad/fold each need. Ranks above
  // the backend's cap (im2col's own use stays well under it) fall back to
  // the CPU oracle, same as gpu_binary's rank-2-only broadcast path.
  template <typename Dispatch>
  static std::optional<array> gpu_pad_fold_(const array& a,
                                            const shape_t& out_shape,
                                            Dispatch&& dispatch) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise) ||
        !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    std::vector<int64_t> a_shape(a.shape().begin(), a.shape().end());
    std::vector<int64_t> out_shape_v(out_shape.begin(), out_shape.end());
    int rank = static_cast<int>(a_shape.size());
    if (!dispatch(out, a_shape, out_shape_v, rank)) return std::nullopt;
    return out;
  }

  // Places `a` into a zero-initialized `out_shape`, shifted by `before`
  // along `axis` — the GPU-dispatch twin of ref::pad.
  static std::optional<array> gpu_pad_(const array& a, size_t axis,
                                       int64_t before,
                                       const shape_t& out_shape) {
    return gpu_pad_fold_(a, out_shape,
                        [&](array& out, std::vector<int64_t>& a_shape,
                            std::vector<int64_t>& out_shape_v, int rank) {
                          return gpu::pad(a.storage_.native, a.offset_ * 4,
                                          out.storage_.native, out.offset_ * 4,
                                          a_shape.data(), out_shape_v.data(),
                                          rank, static_cast<int>(axis), before,
                                          a.size(), out.size());
                        });
  }

  // unfold's inverse: scatter-add `a` back into a zero-initialized
  // `out_shape`, accumulating every window overlap — the GPU-dispatch twin
  // of ref::fold.
  static std::optional<array> gpu_fold_(const array& a, size_t axis,
                                        int64_t step,
                                        const shape_t& out_shape) {
    return gpu_pad_fold_(a, out_shape,
                        [&](array& out, std::vector<int64_t>& a_shape,
                            std::vector<int64_t>& out_shape_v, int rank) {
                          return gpu::fold(a.storage_.native, a.offset_ * 4,
                                           out.storage_.native, out.offset_ * 4,
                                           a_shape.data(), out_shape_v.data(),
                                           rank, static_cast<int>(axis), step,
                                           a.size(), out.size());
                        });
  }

  // Row gather along axis 0 — the GPU-dispatch twin of ref::index_select.
  // No zeroing: every output element is written exactly once.
  static std::optional<array> gpu_index_select_(const array& a,
                                                const array& indices,
                                                const shape_t& out_shape) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise) ||
        !a.contiguous() || !indices.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native || !indices.storage_.native) return std::nullopt;
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    int64_t row_size = a.size() / a.shape()[0];
    if (!gpu::index_select(a.storage_.native, a.offset_ * 4,
                           indices.storage_.native, indices.offset_ * 4,
                           out.storage_.native, out.offset_ * 4, row_size,
                           out_shape[0])) {
      return std::nullopt;
    }
    return out;
  }

  // index_select's dual — the GPU-dispatch twin of ref::index_add. Repeated
  // indices accumulate (real write conflicts), so the backend kernel needs
  // atomics; the backend also owns zeroing `out` before it runs, the same
  // way gpu::pad/gpu::fold do internally.
  static std::optional<array> gpu_index_add_(const array& indices,
                                             const array& values,
                                             const shape_t& target_shape) {
    if (!gpu_mode_(num_elements(target_shape), kernel_class::elementwise) ||
        !indices.contiguous() || !values.contiguous()) {
      return std::nullopt;
    }
    if (!indices.storage_.native || !values.storage_.native) {
      return std::nullopt;
    }
    auto out = array::empty(target_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    int64_t row_size = out.size() / target_shape[0];
    if (!gpu::index_add(indices.storage_.native, indices.offset_ * 4,
                        values.storage_.native, values.offset_ * 4,
                        out.storage_.native, out.offset_ * 4, row_size,
                        indices.shape()[0], out.size())) {
      return std::nullopt;
    }
    return out;
  }

  // One-hot scatter into a new trailing axis — the GPU-dispatch twin of
  // ref::scatter_to_axis. No accumulation (see the ref:: comment), so no
  // atomics and no zeroing responsibility beyond the backend's own launch.
  static std::optional<array> gpu_scatter_to_axis_(const array& indices,
                                                    const array& values,
                                                    const shape_t& out_shape) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise) ||
        !indices.contiguous() || !values.contiguous()) {
      return std::nullopt;
    }
    if (!indices.storage_.native || !values.storage_.native) {
      return std::nullopt;
    }
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::scatter_to_axis(indices.storage_.native, indices.offset_ * 4,
                              values.storage_.native, values.offset_ * 4,
                              out.storage_.native, out.offset_ * 4,
                              values.size(), out_shape.back())) {
      return std::nullopt;
    }
    return out;
  }

  // GPU dispatch for op_t::sum_to_ (un-broadcast a gradient, the dual of
  // broadcast_to every arithmetic op's backward calls). Gather-based (see
  // gpu::sum_to's own comment) -- no atomics needed, unlike index_add.
  static std::optional<array> gpu_sum_to_(const array& a,
                                          const shape_t& target) {
    if (!gpu_mode_(a.size(), kernel_class::reduction) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(target);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    auto acc = broadcast_strides(target, out.strides(), a.shape());
    int rank = static_cast<int>(a.rank());
    std::vector<int64_t> a_shape_v(a.shape().begin(), a.shape().end());
    auto a_strides_v = a.strides();
    int64_t reduced_n = 1;
    for (int d = 0; d < rank; d++) {
      if (acc[d] == 0) reduced_n *= a_shape_v[d];
    }
    if (!gpu::sum_to(a.storage_.native, a.offset_ * 4, a_shape_v.data(),
                     a_strides_v.data(), acc.data(), rank, out.size(),
                     reduced_n, out.storage_.native, out.offset_ * 4)) {
      return std::nullopt;
    }
    return out;
  }

  // GPU dispatch for the 6 comparison ops (gt/lt/ge/le/eq/ne) -- same shape
  // only (no broadcast form; ReLU/LeakyReLU/Clip's backward gate and the
  // concrete Tensor.gt/... callers never need one). Kept off the shared
  // to_kop_/kop vocabulary deliberately -- see metal.h's cmp_op comment.
  static std::optional<array> gpu_compare_(op_t op, const array& a,
                                           const array& b) {
    // Same-shape (bstride=1) or a scalar b (bstride=0, broadcast-read) --
    // the two shapes array.h's `a > 0.0f`-style scalar overloads and a
    // same-shape mask multiply actually produce. Anything else (a real
    // N-D broadcast b) declines to the CPU oracle, which already handles
    // it generally.
    int64_t bstride;
    if (a.shape() == b.shape()) {
      bstride = 1;
    } else if (b.size() == 1) {
      bstride = 0;
    } else {
      return std::nullopt;
    }
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous() ||
        !b.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    gpu::cmp_op c;
    switch (op) {
      case op_t::gt: c = gpu::cmp_op::gt; break;
      case op_t::lt: c = gpu::cmp_op::lt; break;
      case op_t::ge: c = gpu::cmp_op::ge; break;
      case op_t::le: c = gpu::cmp_op::le; break;
      case op_t::eq: c = gpu::cmp_op::eq; break;
      case op_t::ne: c = gpu::cmp_op::ne; break;
      default: return std::nullopt;
    }
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::compare(c, a.storage_.native, a.offset_ * 4, b.storage_.native,
                      b.offset_ * 4, out.storage_.native, out.offset_ * 4,
                      out.size(), bstride)) {
      return std::nullopt;
    }
    return out;
  }

  // GPU dispatch for tanh_/sin_/cos_ -- a CUDA-only addition (like
  // comparisons above), so its own small vocabulary rather than the
  // shared kop table (see metal.h's cmp_op comment; unary_ext_op is the
  // same reasoning applied to unary ops).
  static std::optional<array> gpu_unary_ext_(op_t op, const array& a,
                                             float scale, float offset) {
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    gpu::unary_ext_op u;
    switch (op) {
      case op_t::tanh_: u = gpu::unary_ext_op::tanh_; break;
      case op_t::sin_: u = gpu::unary_ext_op::sin_; break;
      case op_t::cos_: u = gpu::unary_ext_op::cos_; break;
      default: return std::nullopt;
    }
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::unary_ext(u, a.storage_.native, a.offset_ * 4,
                        out.storage_.native, out.offset_ * 4, out.size(),
                        scale, offset)) {
      return std::nullopt;
    }
    return out;
  }

  // GPU dispatch for clamp_ (Clip's forward -- Culebra's dz.raw_elementwise
  // host loop before this). No epilogue: clamp's own two params occupy the
  // role scale/offset play elsewhere, and nothing composes a further affine
  // onto it today.
  static std::optional<array> gpu_clamp_(const array& a, float lo, float hi) {
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::clamp(a.storage_.native, a.offset_ * 4, out.storage_.native,
                    out.offset_ * 4, out.size(), lo, hi)) {
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
    // Auto-mode whole-batch device bias: sum this batch's matmul work and, if
    // the block as a whole earns the GPU, pin every op in it there so the
    // sub-threshold projection gemms don't strand on the CPU and thrash the
    // pipeline (types.h batch_matmul_bias_threshold_). cpu/gpu modes and a
    // pipeline already in flight short-circuit gpu_mode_, so skip the scan.
    struct BiasGuard {
      ~BiasGuard() { batch_gpu_bias_ = false; }
    } bias_guard;
    if (device_ == device_type::auto_ && gpu::available() && !gpu::pending()) {
      int64_t work = 0;
      for (const node* n : order) {
        if (n->op != node::op_t::dot || n->inputs.size() != 2) continue;
        const auto& sa = n->inputs[0]->shape;
        const auto& sb = n->inputs[1]->shape;
        if (sa.size() == 2 && sb.size() == 2) work += sa[0] * sa[1] * sb[1];
      }
      if (work >= batch_matmul_bias_threshold_()) batch_gpu_bias_ = true;
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
    };

    switch (n.op) {
      case op_t::add:
      case op_t::sub:
      case op_t::mul:
      case op_t::div:
      case op_t::pow_:
      case op_t::gt:  // masks too: same same-shape contiguous flat loop
      case op_t::lt:
      case op_t::ge:
      case op_t::le:
      case op_t::eq:
      case op_t::ne: {
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
        };
        return visit_binary_op(n.op, binary_loop);
      }
      case op_t::affine:  // the epilogue IS the op here (identity f)
        unary_loop([](float x) { return x; });
        return true;
      case op_t::clamp_: {
        // Two node-specific scalars (arg0=min, arg1=max), so this can't go
        // through visit_unary_op's one-constexpr-functor-per-op dispatch —
        // same reason affine above is special-cased.
        float lo = n.arg0, hi = n.arg1;
        unary_loop([lo, hi](float x) { return x < lo ? lo : (x > hi ? hi : x); });
        return true;
      }
      default:  // unary ops the table knows; everything else declines
        return visit_unary_op(n.op, unary_loop);
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
      case op_t::div:
      case op_t::pow_: {
        // ONE chain for every elementwise binary: each stage declines the ops it
        // has no kernel for (gpu_binary via to_kop_, accel::binary via its own
        // switch), so pow_ falls through to the CPU table without needing an
        // arm of its own.
        auto a = in(0), b = in(1);
        if (auto g = gpu_binary(n, a, b)) {
          r = std::move(*g);
          epi_done = true;  // kernels apply the epilogue in the store
        } else if (auto o = accel::binary(n.op, a, b)) {
          r = std::move(*o);
        } else {
          visit_binary_op(n.op, [&](auto f) { r = map_binary(a, b, f); });
        }
        break;
      }
      case op_t::gt:   // comparisons yield F32 masks
      case op_t::lt:
      case op_t::ge:
      case op_t::le:
      case op_t::eq:
      case op_t::ne: {
        // gpu_compare_ instead of gpu_binary/to_kop_: comparisons are a
        // CUDA-only addition and to_kop_'s vocabulary is called
        // unconditionally through Metal's/WebGPU's throwing pso_() path
        // (see metal.h's cmp_op comment), so they can't join it yet.
        auto a = in(0), b = in(1);
        if (auto g = gpu_compare_(n.op, a, b)) {
          r = std::move(*g);
        } else if (auto o = accel::binary(n.op, a, b)) {
          r = std::move(*o);
        } else {
          visit_binary_op(n.op, [&](auto f) { r = map_binary(a, b, f); });
        }
        break;
      }
      case op_t::where_: {
        auto c0 = in(0), a0 = in(1), b0 = in(2);
        if (auto g = gpu_where_(n.shape, c0, a0, b0)) {
          r = std::move(*g);
        } else {
          r = map_ternary(c0, a0, b0, [](float c, float x, float y) {
            return c != 0.0f ? x : y;
          });
        }
        break;
      }
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
      case op_t::exp_:
      case op_t::log_:
      case op_t::sqrt_:
      case op_t::relu:
      case op_t::sigmoid: {
        // Same one-chain shape as the binary arm above: recip has no kernel on
        // any backend and sigmoid none in accel, and both stages say so.
        auto a = in(0);
        if (auto g = gpu_unary(to_kop_(n.op), a, n.scale, n.offset)) {
          r = std::move(*g);
          epi_done = true;
        } else if (auto o = accel::unary(n.op, a)) {
          r = std::move(*o);
        } else {
          visit_unary_op(n.op, [&](auto f) { r = map_unary(a, f); });
        }
        break;
      }
      case op_t::tanh_:
      case op_t::sin_:
      case op_t::cos_: {
        auto a = in(0);
        if (auto g = gpu_unary_ext_(n.op, a, n.scale, n.offset)) {
          r = std::move(*g);
          epi_done = true;
        } else if (auto o = accel::unary(n.op, a)) {
          r = std::move(*o);
        } else {
          visit_unary_op(n.op, [&](auto f) { r = map_unary(a, f); });
        }
        break;
      }
      case op_t::clamp_: {
        auto a = in(0);
        if (auto g = gpu_clamp_(a, n.arg0, n.arg1)) {
          r = std::move(*g);
        } else {
          float lo = n.arg0, hi = n.arg1;
          r = map_unary(a, [lo, hi](float x) {
            return x < lo ? lo : (x > hi ? hi : x);
          });
        }
        break;
      }
      case op_t::softmax: {
        auto a = in(0);
        if (auto g = gpu_row(gpu::kop::softmax, a, a.shape(), 1.0f, 0.0f)) {
          r = std::move(*g);
        } else {
          r = ref::softmax(a);
        }
        break;
      }
      case op_t::dot: {
        // Batched (rank >= 3, graph::dot already required matching batch
        // dims): none of the GEMV/GEMM fast paths below know about a batch
        // axis, so this branches off before them entirely. gpu_bdot_ tries
        // real per-slice GPU dispatch first (see its own comment); ref::bdot
        // is the CPU-correct fallback when it declines.
        if (n.inputs[0]->shape.size() > 2 || n.inputs[1]->shape.size() > 2) {
          auto a = in(0), b = in(1);
          if (auto g = gpu_bdot_(a, b)) {
            r = std::move(*g);
          } else {
            r = ref::bdot(a, b);
          }
          break;
        }
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
          if (auto g = gpu_row(k, a, n.shape, n.scale, n.offset)) {
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
          if (auto g = gpu_row(gpu::kop::row_sum, a, n.shape, n.scale * inv,
                               n.offset)) {
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
      case op_t::sum_to_: {
        auto a = in(0);
        if (auto g = gpu_sum_to_(a, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::sum_to(a, n.shape);
        }
        break;
      }
      case op_t::pad_: {
        auto a = in(0);
        size_t axis = static_cast<size_t>(n.axis);
        auto before = static_cast<int64_t>(n.arg0);
        if (auto g = gpu_pad_(a, axis, before, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::pad(a, axis, before, n.shape);
        }
        break;
      }
      case op_t::fold_: {
        auto a = in(0);
        size_t axis = static_cast<size_t>(n.axis);
        auto step = static_cast<int64_t>(n.arg0);
        if (auto g = gpu_fold_(a, axis, step, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::fold(a, axis, step, n.shape);
        }
        break;
      }
      case op_t::index_select_: {
        auto a = in(0);
        auto indices = in(1);
        if (auto g = gpu_index_select_(a, indices, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::index_select(a, indices, n.shape);
        }
        break;
      }
      case op_t::index_add_: {
        auto indices = in(0);
        auto values = in(1);
        if (auto g = gpu_index_add_(indices, values, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::index_add(indices, values, n.shape);
        }
        break;
      }
      case op_t::scatter_axis_: {
        auto indices = in(0);
        auto values = in(1);
        if (auto g = gpu_scatter_to_axis_(indices, values, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::scatter_to_axis(indices, values, n.shape);
        }
        break;
      }
      case op_t::view_: {
        // Pure layout: the source is evaluated, so re-applying the view on the
        // materialized wrap composes strides only (make_view_, no kernel, no
        // barrier). A reshape of a non-contiguous view is the sole case that
        // copies — inherent, and rare.
        //
        // Falls through to the shared epilogue below instead of an early
        // `store`+`return` — `graph::affine` fuses a scale/offset onto ANY
        // unevaluated, non-constant node (view_ included), so a view that
        // skipped the epilogue here would silently drop it (e.g. `x.slice(
        // ...) * s + o` on a still-lazy `x` returned `x.slice(...)` verbatim).
        array src = wrap(*n.inputs[0]);
        switch (n.view_kind) {
          case node::vkind::transpose: r = src.transpose(n.view_axes); break;
          case node::vkind::reshape: r = src.reshape(n.shape); break;
          case node::vkind::slice: {
            int ax = n.view_axes.empty() ? 0 : n.view_axes[0];
            r = src.slice(ax, n.view_start, n.shape[static_cast<size_t>(ax)]);
            break;
          }
          case node::vkind::unfold: {
            int ax = n.view_axes[0];
            r = src.unfold(ax, n.shape.back(), n.view_start);
            break;
          }
        }
        break;
      }
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

inline array array::index_select(const array& indices) const {
  return detail::graph::index_select(*this, indices);
}
inline array index_add(const array& indices, const array& values,
                       shape_t target_shape) {
  return detail::graph::index_add(indices, values, std::move(target_shape));
}
inline array scatter_to_axis(const array& indices, const array& values,
                             int64_t size) {
  return detail::graph::scatter_to_axis(indices, values, size);
}

inline array array::sum_to(shape_t shape) const {
  return detail::graph::sum_to(*this, std::move(shape));
}
inline array sum_to(const array& a, shape_t shape) {
  return a.sum_to(std::move(shape));
}

inline array array::pad(int axis, int64_t before, int64_t after) const {
  return detail::graph::pad(*this, axis, before, after);
}
inline array array::fold(int axis, int64_t orig_size, int64_t step) const {
  return detail::graph::fold(*this, axis, orig_size, step);
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
inline array array::tanh() const {
  return detail::graph::unary(detail::node::op_t::tanh_, *this);
}
inline array array::sin() const {
  return detail::graph::unary(detail::node::op_t::sin_, *this);
}
inline array array::cos() const {
  return detail::graph::unary(detail::node::op_t::cos_, *this);
}
inline array array::clamp(float lo, float hi) const {
  return detail::graph::clamp(*this, lo, hi);
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
