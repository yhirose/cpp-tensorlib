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
#include <profile.h>
#include <shape.h>
#include <storage.h>
#include <types.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    pow_s, gt_s, lt_s, ge_s, le_s, eq_s, ne_s,  // x OP arg0: the scalar operand
                                                // lives in the node, not an input
    affine, recip, exp_, log_, sqrt_, sigmoid, relu,
    tanh_, sin_, cos_,
    clamp_,  // clip(x, arg0=min, arg1=max) — the only unary op needing two
             // node-specific scalars, hence `arg1` below.
    softmax,
    where_,
    dot,
    attn_dec,  // fused decode attention: softmax(arg0 · q·Kᵀ)·V
    attn_pre,  // fused causal prefill attention: row t is softmax(arg0 ·
               // q_t·K[..t]ᵀ)·V[..t]
    rope,      // rotary position embedding (arg0=base, axis=position offset)
    layer_norm_,  // fused last-axis layer norm: inputs {x, gamma, beta},
                  // arg0=eps
    sum_ax, mean_ax, max_ax, argmax_ax, sum_to_,
    lse_ax,  // log(sum(exp)) along `axis` — softmax's denominator, reduced
             // to one value per row without the probabilities in between.
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
    gather_axis_,   // scatter_axis_'s dual: the one element each position
                    // labels, back out of the trailing axis. inputs=
                    // {src, indices}; `shape` is indices' (one axis fewer
                    // than src's).
    concat_,  // N-ary: inputs are the parts, joined along `axis`. `shape`
              // is the output (every dim but `axis` matches the parts;
              // `axis` is their sum) — same "target shape as op parameter"
              // idea as sum_to_.
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

// The tensor-scalar ops (x OP s, s in the node's arg0) as unary functors over
// the scalar math above, so they cannot drift from their binary forms.
template <typename F>
bool visit_scalar_op(node::op_t op, float s, F&& f) {
  using op_t = node::op_t;
  switch (op) {
    case op_t::pow_s: f([s](float x) { return ew_pow(x, s); }); return true;
    case op_t::gt_s: f([s](float x) { return ew_gt(x, s); }); return true;
    case op_t::lt_s: f([s](float x) { return ew_lt(x, s); }); return true;
    case op_t::ge_s: f([s](float x) { return ew_ge(x, s); }); return true;
    case op_t::le_s: f([s](float x) { return ew_le(x, s); }); return true;
    case op_t::eq_s: f([s](float x) { return ew_eq(x, s); }); return true;
    case op_t::ne_s: f([s](float x) { return ew_ne(x, s); }); return true;
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
// Drains the device stream (gpu::flush); the end of a defer_flush scope.
inline void (*flush_hook)() = nullptr;

// Open defer_flush scopes on this thread. While positive, a blocking
// materialize leaves its kernels in flight like realize(); host reads still
// barrier on their own, so only the sync points move.
inline thread_local int defer_flush_depth = 0;

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
  // This view as the GPU layer names it: the storage's device handle and the
  // view's byte offset (gpu_abi.h). Null `buf` when the storage is heap.
  gpu::span device_span() const { return {storage_.native, offset_ * 4}; }

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

  // Fused causal prefill attention (M9): out(h,t,:) = softmax(scale ·
  // q(h,t,:)·K(h,≤t)ᵀ)·V(h,≤t) — every query row of a prompt at once, each
  // attending the keys up to itself. q, K, V and out all [H,T,D]. On CUDA with
  // D ∈ {64,128} this is the tiled prefill kernel; otherwise a CPU reference.
  static array attn_prefill(const array& q, const array& K, const array& V,
                            float scale);

  // The two halves of attn_prefill's pullback. The scores stay in registers in
  // both — the [T,T] matrix the composed pullback materializes is never
  // anywhere — and the price of that is the row statistics `stats` [2,H,T],
  // the softmax's logsumexp and dO·O per row: the dq half computes both on its
  // way past, and the dK/dV half, which sums over queries rather than keys,
  // can derive neither and reads them instead.
  //
  // Eager, not graph nodes: they run inside a backward pass, and a node per
  // gradient would rebuild the scores once each. The own CPU runs both halves
  // as tiled gemms; nullopt when neither takes it (in GPU mode a head width
  // off {64,128} or an operand not device-resident, a non-contiguous operand,
  // the oracle with cpu::enabled_ off) — the caller then composes the unfused
  // pullback.
  //
  // dq takes q, K, V, the gradient `dout` of the forward's output and that
  // output `out` (all [H,T,D]) and returns {dq, stats}; dkv takes those stats
  // in place of `out` and returns {dK, dV}.
  static std::optional<std::pair<array, array>> attn_prefill_bwd_dq(
      const array& q, const array& K, const array& V, const array& dout,
      const array& out, float scale);
  static std::optional<std::pair<array, array>> attn_prefill_bwd_dkv(
      const array& q, const array& K, const array& V, const array& dout,
      const array& stats, float scale);

  // Softmax cross-entropy's pullback, given the forward's row logsumexp:
  // out[i,j] = g[i] · (exp(logits[i,j] - lse[i]) - [j == targets[i]]) in one
  // pass. logits and the result are [N, C]; lse, targets and g are [N].
  // Eager for the same reason the attention pullback is: the device kernel,
  // or rows across the own CPU's pool, and nullopt when neither takes it —
  // the caller then composes (softmax - onehot) · g, which walks that matrix
  // four more times.
  static std::optional<array> xent_bwd(const array& logits, const array& lse,
                                       const array& targets, const array& g);

  // Adam's per-parameter update, in place: m and v advance on g, then p moves
  // by the bias-corrected ratio. p, m, v and g share one shape; `bc1`/`bc2` are
  // the caller's 1 - beta^t. Written as ops it is a dozen passes over the four
  // and a buffer for every intermediate — the shape of an optimizer step's
  // cost, not of its arithmetic. Total on a contiguous f32 set: the device
  // kernel where there is one, the host loop otherwise. False only for a layout
  // it cannot take, since a caller updating in place cannot compose its way out.
  static bool adam_step(array& p, array& m, array& v, const array& g, float lr,
                        float beta1, float beta2, float eps, float bc1,
                        float bc2);

  // Transformer building blocks (M9 model surface). RoPE is a fused op (needs
  // cos/sin); RMSNorm/SiLU/SwiGLU are pure compositions of existing ops (so they
  // ride the tuned kernels and are autograd-ready when VJPs land). RoPE input is
  // [H,D] (decode, T=1) or [H,T,D] (prefill); `pos` is the position of t=0.
  static array rope(const array& x, int64_t pos, float base = 10000.0f);
  static array rmsnorm(const array& x, const array& weight, float eps = 1e-5f);
  // Layer norm over the last axis, fused: (x - mean) / sqrt(var + eps) · gamma
  // + beta, with gamma/beta holding the last dim's d weights ([d] or [1, d]).
  // One node and one output where the composition takes nine.
  static array layer_norm(const array& x, const array& gamma, const array& beta,
                          float eps = 1e-5f);
  // Its pullback, fused: given the forward's x and gamma and the gradient
  // `dout` of its output, {dx, dgamma, dbeta} — dgamma and dbeta as [d].
  // Eager, like the attention halves below: the device kernels, or chunks of
  // rows across the own CPU's pool; nullopt when neither takes it (an operand
  // not device-resident in GPU mode or not contiguous, the oracle), and the
  // caller composes the unfused form.
  static std::optional<std::array<array, 3>> layer_norm_bwd(
      const array& x, const array& gamma, const array& dout,
      float eps = 1e-5f);
  static array silu(const array& x);
  static array swiglu(const array& gate, const array& up);

  // Axis reductions (lazy); argmax yields indices as F32
  array sum(int axis, bool keepdims = false) const;
  array mean(int axis, bool keepdims = false) const;
  array max(int axis, bool keepdims = false) const;
  array argmax(int axis, bool keepdims = false) const;

  // log(sum(exp(x))) along `axis`, numerically stable — softmax's own
  // denominator, without the probabilities in between. On the last axis one
  // fused pass reads the row once; any other axis composes.
  array logsumexp(int axis, bool keepdims = false) const;

  // Reduce broadcast dims back to `shape` (the VJP of broadcasting): sums
  // over leading dims and size-1 dims. `shape` must broadcast to shape().
  array sum_to(shape_t shape) const;

  // Widen to `shape` (sum_to's dual, and the VJP of a reduction): the added
  // and size-1 axes get stride 0, so this is a view, not a copy. shape() must
  // broadcast to `shape`. An elementwise op on the result walks the whole
  // widened shape, so apply it to the narrow source before widening.
  array broadcast_to(shape_t shape) const;

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

  static array make_(shape_t shape, bool host_fill = false);
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
// native VJP yet, though the dual it would need is now gather_from_axis
// just below.
array scatter_to_axis(const array& indices, const array& values,
                      int64_t size);

// scatter_to_axis's dual: out[...] = src[..., indices[...]] -- the one
// element each position labels, back out of the trailing axis. `src` is
// shaped like some scatter_to_axis output (indices' shape plus that axis)
// and the result is indices' shape. Cross-entropy names a row's target
// logit with this instead of multiplying by a one-hot matrix; a pooling
// backward would use it as max(axis)'s own gather. indices get no gradient.
array gather_from_axis(const array& src, const array& indices);

array sum_to(const array& a, shape_t shape);

array concat(const std::vector<array>& parts, int axis);
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
  // Packed q4 has no per-element strides; its consumers assume to_q4()'s layout.
  if (base.storage_.dt == tl::dtype::q4)
    throw std::logic_error(
        "tl::view: q4 arrays can't be viewed; view the f32 weights before "
        "to_q4(), or to_f32() first");
  array v;
  v.shape_ = std::move(shape);
  v.strides_ = std::move(strides);
  v.offset_ = offset;
  v.storage_ = base.storage_;
  return v;
}

inline array array::make_(shape_t shape, bool host_fill) {
  array a;
  a.strides_ = detail::contiguous_strides(shape);
  a.storage_ = storage::make(detail::num_elements(shape), tl::dtype::f32,
                             host_fill);
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

// Host-born data: filled from the host before any kernel touches the buffer,
// so it takes storage no pending work may still read or write (make_'s
// host_fill) and needs no barrier.
inline array array::full(shape_t shape, float v) {
  auto a = make_(std::move(shape), /*host_fill=*/true);
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
  auto a = make_(std::move(shape), /*host_fill=*/true);
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

// Every CPU-side buffer access passes here first. On unified memory (Metal) a
// pending command buffer may still be writing the very bytes the CPU is about
// to touch, so it flushes; the CUDA device-mirror waits per buffer instead, in
// host_sync_ below, and only when that buffer's live copy is on the device.
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
// storages (native==null). Pairs with barrier_(): that one waits where the
// device can write host-visible bytes (Metal); this one waits for this
// specific buffer, when its live copy is on the device (CUDA).
inline void host_sync_(void* native, bool for_write) {
#ifdef TL_RUNTIME_HOOKS
  if (host_sync_hook) host_sync_hook(native, for_write);
#else
  gpu::sync_to_host(native, for_write);
#endif
}

// The two CPU-side accessors for a buffer that may live on the device: a
// read gets the current bytes, a write also marks the device copy stale.
// Every CPU loop over storage goes through one of these (or array::raw /
// array::data, which do the same) — a bare `stor.data()` on a buffer the
// GPU last wrote reads whatever the host mirror held before.
inline const float* host_read_(const storage& s, int64_t off) {
  barrier_();
  host_sync_(s.native, /*for_write=*/false);
  return s.data() + off;
}
inline float* host_write_(storage& s, int64_t off) {
  barrier_();
  host_sync_(s.native, /*for_write=*/true);
  return s.data() + off;
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

inline array array::broadcast_to(shape_t shape) const {
  if (shape == shape_) return *this;
  if (shape.size() < shape_.size()) {
    throw std::invalid_argument("tl::broadcast_to: target rank is lower");
  }
  size_t lead = shape.size() - shape_.size();
  for (size_t i = 0; i < shape_.size(); i++) {
    if (shape_[i] != 1 && shape_[i] != shape[lead + i]) {
      throw std::invalid_argument("tl::broadcast_to: shape does not broadcast");
    }
  }
  auto strides = detail::broadcast_strides(shape_, strides_, shape);
  // The widened axes step 0, which every consumer that takes per-operand
  // strides handles (the broadcast kernels, the CPU walker, clone's gather).
  // A lazy source materializes first: view nodes carry a vkind and there is
  // none for widening, and the callers (a reduction's VJP) hold an evaluated
  // gradient anyway.
  realize_();
  return make_view_(*this, std::move(shape), std::move(strides), offset_);
}

namespace detail {
// clone()'s device arm, defined once graph is complete.
inline std::optional<array> device_clone_(const array& a);

// ...reached through a hook under TL_RUNTIME_HOOKS, for gpu_pending_hook's
// reason (storage.h): clone() is a builder path, not an evaluation. A strided
// view's reshape clones, so naming the device arm here would leave a Metal
// reference in a translation unit that only builds graphs. Null means no
// backend is installed, and clone() copies on the host as it always did.
inline std::optional<array> (*device_clone_hook)(const array&) = nullptr;

// clone()'s own-CPU arm is the elementwise driver with the identity (defined
// with ew_run below).
template <typename F>
array map_unary(const array& a, F f);
}  // namespace detail

inline array array::clone() const {
  // On a device buffer the host arm below reads the data back and writes it
  // out again (CUDA: D2H, then H2D at the next device use); copy on the device
  // instead and leave the kernel in flight, like any other realized result.
  realize_();
  // Eager, so no evaluator scope names the copy; opened after the source's
  // own evaluation so that lands under its own ops, not here.
  profile::scope ps("clone");
  if (storage_.native && storage_.dt == tl::dtype::f32) {
#ifdef TL_RUNTIME_HOOKS
    if (detail::device_clone_hook) {
      if (auto out = detail::device_clone_hook(*this)) return std::move(*out);
    }
#else
    if (auto out = detail::device_clone_(*this)) return std::move(*out);
#endif
  }
  ensure_();
  if (storage_.dt != tl::dtype::f32) {
    // bf16/q4 arrays are weight leaves (to_bf16/to_q4 output); byte-copy.
    // q4 can't be a view (make_view_ refuses it) and its bytes are packed.
    const bool q4 = storage_.dt == tl::dtype::q4;
    if (!q4 && (!contiguous() || offset_ != 0)) {
      throw std::logic_error("tl::clone: non-contiguous bf16 view");
    }
    const int64_t bytes = q4 ? tl::q4_bytes(shape_[1], shape_[0])
                             : size() * dtype_size(storage_.dt);
    array out;
    out.shape_ = shape_;
    out.strides_ = strides_;
    out.storage_ = storage::make_bytes(size(), bytes, storage_.dt);
    detail::barrier_();
    detail::host_sync_(storage_.native, /*for_write=*/false);
    std::memcpy(out.storage_.data(), storage_.data(), static_cast<size_t>(bytes));
    return out;
  }
  // Own CPU: any layout copied a run at a time across the pool (a permuted or
  // transposed view had been the one-thread element walker below, ~1 ms for
  // 2 MB). The oracle keeps the walker.
  if (cpu::enabled_) return detail::map_unary(*this, [](float x) { return x; });
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
  out.storage_ = storage::make_bytes(N * K, tl::q4_bytes(N, K), tl::dtype::q4);
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

// The own-CPU elementwise driver: the output as `rows` runs of `inner`
// consecutive elements along which each operand steps by a fixed stride —
// contiguous (1), constant (0), or a transposed view's gather (anything
// else) — the runs spread across the thread pool (a lone long run is cut
// into chunks). One loop shape covers same-shape operands, a scalar, a row
// or column vector, a leading broadcast axis (the attention mask over
// [H,T,T]), a slice with a gap between rows and a transposed view. A run's
// operand offsets are derived once per run; the bodies split out the
// stride-1 and constant cases, which vectorize, and keep one gather loop for
// the rest. Everything here is fixed-size — no allocation but the output —
// since an MLP's [10,30] op is a microsecond. Gated by cpu::enabled_ like the
// other own-CPU kernels, so the oracle tests compare against the walker.
// Threads for an own-CPU kernel over `macs` multiply-adds of work; the oracle
// (cpu::enabled_ off) runs one. A streaming pass (elementwise, a reduction)
// costs about kStreamMacs of a gemm's multiply-add per element.
constexpr int64_t kStreamMacs = 8;
inline int own_threads_(int64_t macs) {
  return cpu::enabled_ ? cpu::threads_for_(macs) : 1;
}

constexpr size_t kEwMaxSrc = 4;  // for_each_index's operand limit too
struct ew_strides {
  size_t n = 0;
  int64_t s[kEwMaxSrc][kMaxRank] = {};
  // Add `src` viewed as `out_shape` (broadcast_strides' rule: missing leading
  // dims and size-1 dims step 0).
  void add(const array& src, const shape_t& out_shape) {
    size_t rank = out_shape.size(), lead = rank - src.rank();
    for (size_t i = 0; i < src.rank(); i++)
      s[n][lead + i] = src.shape()[i] == 1 ? 0 : src.strides()[i];
    n++;
  }
  std::vector<std::vector<int64_t>> as_vectors(size_t rank) const {
    std::vector<std::vector<int64_t>> v(n);
    for (size_t k = 0; k < n; k++) v[k].assign(s[k], s[k] + rank);
    return v;
  }
};
struct ew_plan {
  bool ok = false;
  size_t outer_rank = 0;  // axes above the run
  int64_t rows = 0, inner = 0;
  int64_t step[kEwMaxSrc] = {};  // per operand along the run: its innermost stride
};
inline ew_plan ew_plan_for(const shape_t& shape, const ew_strides& st) {
  ew_plan p;
  size_t rank = shape.size();
  if (rank > kMaxRank || st.n > kEwMaxSrc) return p;
  if (rank == 0) {
    p.ok = true;
    p.rows = p.inner = 1;
    return p;
  }
  for (size_t k = 0; k < st.n; k++) p.step[k] = st.s[k][rank - 1];
  int64_t inner = shape[rank - 1];
  size_t d = rank - 1;
  while (d > 0) {
    bool extends = true;
    for (size_t k = 0; k < st.n && extends; k++)
      extends = shape[d - 1] == 1 || st.s[k][d - 1] == p.step[k] * inner;
    if (!extends) break;
    inner *= shape[--d];
  }
  p.ok = true;
  p.outer_rank = d;
  p.inner = inner;
  p.rows = inner ? num_elements(shape) / inner : 0;
  return p;
}

// Runs body(out_offset, operand_offsets, j0, j1) for every chunk of every run;
// the body writes out[out_offset + j0 .. + j1) reading each operand at its
// offset plus j·step.
template <typename Body>
void ew_run(const shape_t& shape, const ew_strides& st, const ew_plan& p,
            Body body) {
  if (p.rows == 0 || p.inner == 0) return;
  // The thread cap sizes the work items too: when there are fewer runs than
  // threads, each run is cut into enough chunks to hand every thread one.
  const int nt = cpu::threads_for_(p.rows * p.inner * kStreamMacs);
  int64_t chunk = p.inner;
  if (p.rows < nt) {
    int64_t want = (nt + p.rows - 1) / p.rows;  // chunks per run
    chunk = (p.inner + want - 1) / want;
  }
  const int64_t cpr = (p.inner + chunk - 1) / chunk;
  cpu::thread_pool::instance().parallel_for(
      p.rows * cpr,
      [&](int64_t i0, int64_t i1) {
        int64_t offs[kEwMaxSrc];
        for (int64_t it = i0; it < i1; it++) {
          int64_t row = it / cpr, c = it % cpr;
          // The run's operand offsets: its index decomposed over the outer
          // axes, once per run (a few divisions against `inner` elements).
          for (size_t k = 0; k < st.n; k++) offs[k] = 0;
          for (int64_t rem = row, d = static_cast<int64_t>(p.outer_rank);
               d-- > 0;) {
            int64_t idx = rem % shape[d];
            rem /= shape[d];
            for (size_t k = 0; k < st.n; k++) offs[k] += idx * st.s[k][d];
          }
          int64_t j0 = c * chunk, j1 = std::min(p.inner, j0 + chunk);
          body(row * p.inner, offs, j0, j1);
        }
      },
      nt);
}

template <typename F>
array map_unary(const array& a, F f) {
  auto out = array::empty(a.shape());
  auto* po = out.data();
  const auto* pa = a.raw();
  ew_strides st;
  st.add(a, a.shape());
  if (cpu::enabled_) {
    if (auto p = ew_plan_for(a.shape(), st); p.ok) {
      ew_run(a.shape(), st, p,
             [&](int64_t o, const int64_t* offs, int64_t j0, int64_t j1) {
               const float* pai = pa + offs[0];
               const int64_t sa = p.step[0];
               if (sa == 1) {
                 for (int64_t j = j0; j < j1; j++) po[o + j] = f(pai[j]);
               } else if (sa == 0) {
                 float v = f(pai[0]);
                 for (int64_t j = j0; j < j1; j++) po[o + j] = v;
               } else {
                 for (int64_t j = j0; j < j1; j++) po[o + j] = f(pai[j * sa]);
               }
             });
      return out;
    }
  }
  for_each_index(a.shape(), st.as_vectors(a.rank()),
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
  // A broadcast axis steps 0, so a scalar, a row vector, a column vector and a
  // leading broadcast axis all read as "constant along the run" or
  // "contiguous along the run" to ew_plan_for.
  ew_strides st;
  st.add(a, shape);
  st.add(b, shape);
  if (cpu::enabled_) {
    if (auto p = ew_plan_for(shape, st); p.ok) {
      ew_run(shape, st, p,
             [&](int64_t o, const int64_t* offs, int64_t j0, int64_t j1) {
               const float* pai = pa + offs[0];
               const float* pbi = pb + offs[1];
               // Hoist a constant operand and split on the steps so each variant
               // is a stride-1 (or constant) inner loop that vectorizes; the
               // generic j*stride form is an unpredictable gather to the compiler,
               // kept for a transposed operand.
               const int64_t sa = p.step[0], sb = p.step[1];
               if ((sa != 0 && sa != 1) || (sb != 0 && sb != 1)) {
                 for (int64_t j = j0; j < j1; j++)
                   po[o + j] = f(pai[j * sa], pbi[j * sb]);
               } else if (sa == 1 && sb == 1) {
                 for (int64_t j = j0; j < j1; j++) po[o + j] = f(pai[j], pbi[j]);
               } else if (sa == 1) {
                 float bv = pbi[0];
                 for (int64_t j = j0; j < j1; j++) po[o + j] = f(pai[j], bv);
               } else if (sb == 1) {
                 float av = pai[0];
                 for (int64_t j = j0; j < j1; j++) po[o + j] = f(av, pbi[j]);
               } else {
                 float v = f(pai[0], pbi[0]);
                 for (int64_t j = j0; j < j1; j++) po[o + j] = v;
               }
             });
      return out;
    }
  }
  for_each_index(shape, st.as_vectors(shape.size()),
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
  ew_strides st;
  st.add(a, shape);
  st.add(b, shape);
  st.add(c, shape);
  if (cpu::enabled_) {
    if (auto p = ew_plan_for(shape, st); p.ok) {
      // Three operands make eight step patterns; the strided inner loop is
      // still a run at a time across the pool, which is where the walker
      // lost (a where over a broadcast mask is the same shape as its add).
      const int64_t sa = p.step[0], sb = p.step[1], sc = p.step[2];
      ew_run(shape, st, p,
             [&](int64_t o, const int64_t* offs, int64_t j0, int64_t j1) {
               const float* pai = pa + offs[0];
               const float* pbi = pb + offs[1];
               const float* pci = pc + offs[2];
               for (int64_t j = j0; j < j1; j++)
                 po[o + j] = f(pai[j * sa], pbi[j * sb], pci[j * sc]);
             });
      return out;
    }
  }
  for_each_index(shape, st.as_vectors(shape.size()),
                 [&](int64_t i, const std::vector<int64_t>& off) {
                   po[i] = f(pa[off[0]], pb[off[1]], pc[off[2]]);
                 });
  return out;
}

// Fold a strided run of `n` floats with f in eight independent lanes, then
// fold the lanes: a single accumulator is a serial chain at the FP add
// latency (a 768-wide row took 0.26 µs, 3 GB/s), eight lanes run at the
// throughput. f must be associative and commutative (sum, max) — the lane
// order is not the element order.
template <typename F>
inline float fold_lanes(const float* src, int64_t stride, int64_t n, float init,
                        F f) {
  constexpr int L = 8;
  float lanes[L];
  for (int l = 0; l < L; l++) lanes[l] = init;
  int64_t k = 0;
  for (; k + L <= n; k += L)
    for (int l = 0; l < L; l++) f(lanes[l], src[(k + l) * stride]);
  for (; k < n; k++) f(lanes[0], src[k * stride]);
  float acc = lanes[0];
  for (int l = 1; l < L; l++) f(acc, lanes[l]);
  return acc;
}

// Shared axis-reduction driver: for each input element, f(acc_slot, value).
// The output slots are independent, so `max_threads` spreads them across the
// thread pool on the contiguous path (the own-CPU dispatch passes its work
// cap; the default runs one thread, the oracle).
template <typename F>
array reduce_axis(const array& a, int axis, bool keepdims, float init, F f,
                  int max_threads = 1) {
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
    auto& pool = cpu::thread_pool::instance();
    if (inner == 1) {
      // Last-axis reduction: each contiguous run folded into locals, stored
      // once. Accumulating straight into po[o] instead carries the dependency
      // through memory (store-to-load per element, no vectorize) — ~40x slower
      // here, and this is the common case (softmax denominators, bias/
      // feature-sum gradients, per-row norms). Rows across the pool.
      pool.parallel_for(outer, [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; o++)
          po[o] = fold_lanes(pi + o * axis_len, 1, axis_len, init, f);
      }, max_threads);
      return out;
    }
    // inner > 1: each po[j] is an independent accumulator, so the contiguous
    // inner loop vectorizes with no cross-element dependency. Split the outer
    // slabs when there are several, else the inner columns (a matrix reduced
    // over axis 0 — the bias gradient — has one slab).
    if (outer > 1) {
      pool.parallel_for(outer, [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; o++) {
          const float* base = pi + o * axis_len * inner;
          float* od = po + o * inner;
          for (int64_t k = 0; k < axis_len; k++) {
            const float* src = base + k * inner;
            for (int64_t j = 0; j < inner; j++) f(od[j], src[j]);
          }
        }
      }, max_threads);
    } else {
      pool.parallel_for(inner, [&](int64_t j0, int64_t j1) {
        for (int64_t k = 0; k < axis_len; k++) {
          const float* src = pi + k * inner;
          for (int64_t j = j0; j < j1; j++) f(po[j], src[j]);
        }
      }, max_threads);
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

// sum_to on the own CPU: adjacent axes that are all summed away (or all kept)
// merge into one — a contiguous buffer reshapes across them for free — and
// each summed group is one reduce_axis pass, whose contiguous paths split
// across the pool. The bias gradient [B·T, C] -> [C] is then a single column
// pass, adding each column in row order as the element walker (ref::sum_to)
// did, which took ~1 ms at GPT-small sizes on one thread.
inline array own_sum_to_(const array& a, const shape_t& target, int max_threads) {
  auto src = a.contiguous() ? a : a.clone();
  const auto& sh = src.shape();
  const size_t lead = sh.size() - target.size();
  shape_t merged;
  std::vector<bool> summed;
  for (size_t i = 0; i < sh.size(); i++) {
    if (sh[i] == 1) continue;  // neither summed nor kept: drops out
    bool s = i < lead || target[i - lead] == 1;
    if (!summed.empty() && summed.back() == s) {
      merged.back() *= sh[i];
    } else {
      merged.push_back(sh[i]);
      summed.push_back(s);
    }
  }
  auto cur = src.reshape(merged);
  for (size_t ax = 0; ax < merged.size(); ax++) {
    if (summed[ax])
      cur = reduce_axis(cur, static_cast<int>(ax), true, 0.0f,
                        [](float& acc, float v) { acc += v; }, max_threads);
  }
  return cur.reshape(target);
}

}  // namespace detail

namespace ref {

// fn(r, off) for each row r of `a`'s last axis, `off` its start walked through
// the outer dims. Rows are independent, so `max_threads` spreads them across
// the thread pool (the own-CPU dispatch passes its work cap; the default runs
// one thread, the oracle, without touching the pool).
template <typename F>
inline void for_last_axis_rows_(const array& a, int max_threads, F&& fn) {
  int64_t cols = a.shape().back();
  int64_t rows = a.size() / (cols ? cols : 1);
  shape_t outer(a.shape().begin(), a.shape().end() - 1);
  std::vector<int64_t> outer_strides(a.strides().begin(), a.strides().end() - 1);
  std::vector<int64_t> row_off(rows);
  detail::for_each_index(outer, {outer_strides},
                         [&](int64_t i, const std::vector<int64_t>& off) {
                           row_off[i] = off[0];
                         });
  auto rows_fn = [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; r++) fn(r, row_off[r]);
  };
  if (max_threads > 1)
    cpu::thread_pool::instance().parallel_for(rows, rows_fn, max_threads);
  else
    rows_fn(0, rows);
}

// Softmax over the last axis, rows across the pool (for_last_axis_rows_). Off
// the GPU this had been one thread of scalar exps whatever the size — the
// whole remaining gap of a hand-written attention once its gemms were even,
// and what causal_attention's backward pays to rebuild its probabilities.
// `own_exp` takes the own-CPU backend's vector exp (cpu::exp_shifted); the
// oracle keeps libm's.
inline array softmax(const array& a, int max_threads = 1, bool own_exp = false) {
  auto out = array::empty(a.shape());
  int64_t cols = a.shape().back();
  int64_t col_stride = a.strides().back();
  const auto* pi = a.raw();
  auto* po = out.data();
  for_last_axis_rows_(a, max_threads, [&](int64_t r, int64_t off) {
    const float* src = pi + off;
    float* dst = po + r * cols;
    float m = detail::fold_lanes(src, col_stride, cols, src[0],
                                 [](float& a, float v) { a = std::max(a, v); });
    float denom;
    if (own_exp) {
      denom = cpu::exp_shifted(dst, src, col_stride, cols, m);
    } else {
      for (int64_t c = 0; c < cols; c++) dst[c] = std::exp(src[c * col_stride] - m);
      denom = detail::fold_lanes(dst, 1, cols, 0.0f, [](float& a, float v) { a += v; });
    }
    for (int64_t c = 0; c < cols; c++) dst[c] /= denom;
  });
  return out;
}

// log(sum(exp)) over the last axis, one value per row. One pass: the row
// carries a running max and the sum of exp below it, rescaling the sum
// whenever a larger element arrives — the same fold the CUDA kernel's threads
// do, so the two backends agree. `out_shape` is the reduced shape (keepdims
// or not); either way there is one output per row. `own_exp` (the own-CPU
// backend) takes two passes instead, the row's max and then the vector exp's
// sum (cpu::exp_shifted); a row whose max is not finite keeps the fold.
inline array logsumexp(const array& a, const shape_t& out_shape,
                       int max_threads = 1, bool own_exp = false) {
  auto out = array::empty(out_shape);
  int64_t cols = a.shape().back();
  int64_t col_stride = a.strides().back();
  const auto* pi = a.raw();
  auto* po = out.data();
  for_last_axis_rows_(a, max_threads, [&](int64_t r, int64_t off) {
    const float* src = pi + off;
    if (own_exp && cols > 0) {
      float m = detail::fold_lanes(src, col_stride, cols, src[0],
                                   [](float& a, float v) { a = std::max(a, v); });
      if (std::isfinite(m)) {
        po[r] = m + std::log(cpu::exp_shifted(nullptr, src, col_stride, cols, m));
        return;
      }
    }
    float m = -3.402823466e+38f, s = 0.0f;
    for (int64_t c = 0; c < cols; c++) {
      float v = src[c * col_stride];
      if (v > m) {
        s *= std::exp(m - v);  // s is 0 on the first step, so -FLT_MAX is safe
        m = v;
      }
      s += std::exp(v - m);
    }
    po[r] = m + std::log(s);
  });
  return out;
}

// Layer norm over the last axis: (x - mu) · 1/sqrt(var + eps) · gamma + beta,
// mu and var the row's mean and biased variance; gamma/beta are d-vectors
// (their last axis holds all d). The arithmetic is the composition's —
// mean(-1, keepdims), sub, square, mean, + eps, 1/sqrt, mul, mul, add — with
// each sum folding the lanes ref::mean folds and 1/d scaling it after, so the
// fused op and the composition agree on this backend. Rows across the pool
// (for_last_axis_rows_).
inline array layer_norm(const array& x, const array& gamma, const array& beta,
                        float eps, int max_threads = 1) {
  auto out = array::empty(x.shape());
  int64_t d = x.shape().back();
  int64_t xs = x.strides().back();
  int64_t gs = gamma.strides().back(), bs = beta.strides().back();
  const auto* pi = x.raw();
  const auto* pg = gamma.raw();
  const auto* pb = beta.raw();
  auto* po = out.data();
  const float inv_d = 1.0f / static_cast<float>(d);
  auto add = [](float& a, float v) { a += v; };
  for_last_axis_rows_(x, max_threads, [&](int64_t r, int64_t off) {
    const float* src = pi + off;
    float* dst = po + r * d;
    float mu = detail::fold_lanes(src, xs, d, 0.0f, add) * inv_d;
    // dst holds the squared deviations for the second sum, then the output.
    for (int64_t c = 0; c < d; c++) {
      float v = src[c * xs] - mu;
      dst[c] = v * v;
    }
    float var = detail::fold_lanes(dst, 1, d, 0.0f, add) * inv_d;
    float inv = 1.0f / std::sqrt(var + eps);
    for (int64_t c = 0; c < d; c++)
      dst[c] = (src[c * xs] - mu) * inv * pg[c * gs] + pb[c * bs];
  });
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

inline array sum(const array& a, int axis, bool keepdims, int max_threads = 1) {
  return detail::reduce_axis(a, axis, keepdims, 0.0f,
                             [](float& acc, float v) { acc += v; }, max_threads);
}

inline array mean(const array& a, int axis, bool keepdims, int max_threads = 1) {
  int r = static_cast<int>(a.rank());
  int64_t n = a.shape()[axis < 0 ? axis + r : axis];
  auto s = sum(a, axis, keepdims, max_threads);
  float inv = 1.0f / static_cast<float>(n);
  return detail::map_unary(s, [inv](float x) { return x * inv; });
}

inline array max(const array& a, int axis, bool keepdims, int max_threads = 1) {
  return detail::reduce_axis(
      a, axis, keepdims, -std::numeric_limits<float>::infinity(),
      [](float& acc, float v) { acc = std::max(acc, v); }, max_threads);
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

// Places every part into its own axis-window of a shared output buffer --
// same for_each_index trick as pad above (each part covers its own shape at
// a shift along `axis`), just run once per part into one `out` instead of
// once into a zeroed buffer. Parts exhaustively cover `out` (no padding
// border), so there's nothing to zero first.
inline array concat(const std::vector<array>& parts, size_t axis,
                    const shape_t& out_shape) {
  auto out = array::empty(out_shape);
  auto* po = out.data();
  auto dst_strides = out.strides();
  int64_t offset = 0;
  for (auto& p : parts) {
    const auto* pi = p.raw();
    int64_t shift = offset * dst_strides[axis];
    detail::for_each_index(p.shape(), {p.strides(), dst_strides},
                           [&](int64_t, const std::vector<int64_t>& off) {
                             po[off[1] + shift] = pi[off[0]];
                           });
    offset += p.shape()[axis];
  }
  return out;
}

// Row gather along axis 0: out[i] = a[indices[i]] (indices float-valued,
// rounded, matching argmax's own convention). Each output row is one source
// row, so the label is read once per row and the row copied whole: a memcpy
// when `a`'s rows are dense, else through offsets walked once for all rows.
// Rows across the pool (`max_threads`, as for_last_axis_rows_). This had been
// an odometer over every element, 19x torch's embedding lookup.
inline array index_select(const array& a, const array& indices,
                          const shape_t& out_shape, int max_threads = 1) {
  auto out = array::empty(out_shape);
  int64_t rows = out_shape[0];
  int64_t row_len = rows ? out.size() / rows : 0;
  if (row_len == 0) return out;
  auto* po = out.data();
  const auto* pi = a.raw();
  const auto* pidx = indices.raw();
  int64_t idx_stride = indices.strides()[0];
  int64_t a_row_stride = a.strides()[0];
  shape_t row_shape(a.shape().begin() + 1, a.shape().end());
  std::vector<int64_t> row_strides(a.strides().begin() + 1, a.strides().end());
  bool dense = row_strides == detail::contiguous_strides(row_shape);
  std::vector<int64_t> offs;
  if (!dense) {
    offs.resize(static_cast<size_t>(row_len));
    detail::for_each_index(row_shape, {row_strides},
                           [&](int64_t j, const std::vector<int64_t>& off) {
                             offs[j] = off[0];
                           });
  }
  auto rows_fn = [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; r++) {
      int64_t row = static_cast<int64_t>(std::llround(pidx[r * idx_stride]));
      const float* src = pi + row * a_row_stride;
      float* dst = po + r * row_len;
      if (dense) {
        std::memcpy(dst, src, static_cast<size_t>(row_len) * sizeof(float));
      } else {
        for (int64_t j = 0; j < row_len; j++) dst[j] = src[offs[j]];
      }
    }
  };
  if (max_threads > 1)
    cpu::thread_pool::instance().parallel_for(rows, rows_fn, max_threads);
  else
    rows_fn(0, rows);
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
    int64_t row = static_cast<int64_t>(
        std::llround(pidx[idx[0] * indices.strides()[0]]));
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
// `k` is taken on trust: graph::check_labels_ has already rejected any label
// outside the new axis before either backend runs.
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

// scatter_to_axis's dual: out[pos] = src[pos, indices[pos]], one element per
// position out of the trailing axis. A manual walk like index_select's --
// the label makes that axis's offset data-dependent, which no fixed reindex
// could express.
inline array gather_from_axis(const array& src, const array& indices,
                              const shape_t& out_shape) {
  auto out = array::empty(out_shape);
  auto* po = out.data();
  const auto* ps = src.raw();
  const auto* pidx = indices.raw();
  size_t r = indices.rank();
  const auto& idx_strides = indices.strides();
  const auto& src_strides = src.strides();
  const auto& out_strides = out.strides();
  std::vector<int64_t> idx(r, 0);
  int64_t n = out.size();
  for (int64_t i = 0; i < n; i++) {
    int64_t idx_off = 0, src_off = 0, out_off = 0;
    for (size_t d = 0; d < r; d++) {
      idx_off += idx[d] * idx_strides[d];
      src_off += idx[d] * src_strides[d];
      out_off += idx[d] * out_strides[d];
    }
    int64_t k = static_cast<int64_t>(std::llround(pidx[idx_off]));
    po[out_off] = ps[src_off + k * src_strides[r]];
    for (size_t d = r; d-- > 0;) {
      if (++idx[d] < out_shape[d]) break;
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
// Reads the trailing two dims of any rank, so a batched matmul classifies each
// slice by the same rule and a permuted view (attention's kᵀ) is "transposed".
inline std::optional<gemm_layout> gemm_classify_(const array& x) {
  size_t n = x.rank();
  int64_t r = x.shape()[n - 2], c = x.shape()[n - 1];
  int64_t s0 = x.strides()[n - 2], s1 = x.strides()[n - 1];
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

  // The row bias fuse_dot_bias_ composed into a dot node, or null.
  static const node* fused_bias_(const node& n) {
    return n.op == op_t::dot && n.inputs.size() == 3 ? n.inputs[2].get() : nullptr;
  }

  // `dot + bias` with a row-vector bias ([n] or [1, n], n the product's
  // columns) composes into a copy of the unevaluated rank-2 dot node as its
  // third input, added after the scale/offset epilogue: addmm's shape, so a
  // backend adds it in the gemm's own store instead of a second pass over the
  // output. The original dot is left for any other consumer, as in affine.
  // Batched dots don't fuse yet (gemm_batched takes the bias pointer for when a
  // batched linear needs it).
  static std::optional<array> fuse_dot_bias_(const array& d, const array& bias) {
    const node* dn = d.node_.get();
    if (!dn || dn->evaluated || dn->op != op_t::dot || dn->inputs.size() != 2 ||
        dn->inputs[0]->shape.size() != 2 || dn->inputs[1]->shape.size() != 2) {
      return std::nullopt;
    }
    const auto& bs = bias.shape();
    if (bs.empty() || bs.size() > 2 || bs.back() != dn->shape[1] ||
        bias.size() != dn->shape[1]) {
      return std::nullopt;
    }
    auto c = std::make_shared<node>(*dn);
    c->inputs.push_back(as_node(bias));
    return from_node(std::move(c));
  }

  static array binary(op_t op, const array& a, const array& b) {
    if (op == op_t::add) {
      if (auto f = fuse_dot_bias_(a, b)) return std::move(*f);
      if (auto f = fuse_dot_bias_(b, a)) return std::move(*f);
    }
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
  // other consumer. A dot carrying a fused bias adds it after its epilogue, so
  // an affine on it stays a node of its own.
  static array affine(const array& a, float s, float o) {
    if (a.node_ && !a.node_->evaluated && a.node_->op != op_t::constant &&
        !fused_bias_(*a.node_)) {
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

  static array gather_from_axis(const array& src, const array& indices) {
    if (src.rank() != indices.rank() + 1 ||
        !std::equal(indices.shape().begin(), indices.shape().end(),
                    src.shape().begin())) {
      throw std::invalid_argument(
          "tl::gather_from_axis: src must be indices' shape plus a trailing "
          "axis -- got src " + shape_str(src.shape()) + ", indices " +
          shape_str(indices.shape()));
    }
    auto n = std::make_shared<node>();
    n->op = op_t::gather_axis_;
    n->shape = indices.shape();
    n->inputs = {as_node(src), as_node(indices)};
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

  // x OP s with the scalar in the node (arg0), not a rank-0 input; an
  // epilogue fuses onto it like any unary's.
  static array scalar_binary(op_t op, const array& a, float s) {
    if (num_elements(a.shape()) <= kEagerTiny && eager_operand_(a) &&
        eager_cpu_ok_()) {
      switch (op) {  // direct switch, see graph::binary
        case op_t::pow_s: return map_unary(a, [s](float x) { return ew_pow(x, s); });
        case op_t::gt_s: return map_unary(a, [s](float x) { return ew_gt(x, s); });
        case op_t::lt_s: return map_unary(a, [s](float x) { return ew_lt(x, s); });
        case op_t::ge_s: return map_unary(a, [s](float x) { return ew_ge(x, s); });
        case op_t::le_s: return map_unary(a, [s](float x) { return ew_le(x, s); });
        case op_t::eq_s: return map_unary(a, [s](float x) { return ew_eq(x, s); });
        case op_t::ne_s: return map_unary(a, [s](float x) { return ew_ne(x, s); });
        default: break;
      }
    }
    auto n = std::make_shared<node>();
    n->op = op;
    n->shape = a.shape();
    n->arg0 = s;
    n->inputs = {as_node(a)};
    return from_node(std::move(n));
  }

  // N-ary: no generic binary()/unary() builder fits, same reason clamp above
  // has its own. Every part must share rank and every dim but `axis`.
  static array concat(const std::vector<array>& parts, int axis) {
    if (parts.empty()) {
      throw std::invalid_argument("tl::concat: empty");
    }
    size_t ax = normalize_axis(axis, parts[0].rank(), "concat");
    auto shape = parts[0].shape();
    int64_t total = 0;
    for (auto& p : parts) {
      auto s = p.shape();
      if (s.size() != shape.size()) {
        throw std::invalid_argument("tl::concat: rank mismatch");
      }
      for (size_t d = 0; d < s.size(); d++) {
        if (d != ax && s[d] != shape[d]) {
          throw std::invalid_argument("tl::concat: shape mismatch");
        }
      }
      total += s[ax];
    }
    shape[ax] = total;
    auto n = std::make_shared<node>();
    n->op = op_t::concat_;
    n->shape = std::move(shape);
    n->axis = static_cast<int>(ax);
    n->inputs.reserve(parts.size());
    for (auto& p : parts) n->inputs.push_back(as_node(p));
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

  static array attn_prefill(const array& q, const array& K, const array& V,
                            float scale) {
    const auto& sq = q.shape();
    if (sq.size() != 3 || K.shape() != sq || V.shape() != sq) {
      throw std::invalid_argument(
          "tl::attn_prefill: expect q, K, V all [H,T,D] — got q " +
          shape_str(sq) + ", K " + shape_str(K.shape()) + ", V " +
          shape_str(V.shape()));
    }
    auto n = std::make_shared<node>();
    n->op = op_t::attn_pre;
    n->shape = sq;  // [H, T, D]
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

  // A layer norm weight: the last axis's d values, as [d] or [1, d].
  static bool holds_last_(const array& w, int64_t d) {
    return w.rank() >= 1 && w.shape().back() == d && w.size() == d;
  }

  static array layer_norm(const array& x, const array& gamma,
                          const array& beta, float eps) {
    const auto& s = x.shape();
    int64_t d = s.empty() ? 0 : s.back();
    if (d <= 0 || !holds_last_(gamma, d) || !holds_last_(beta, d))
      throw std::invalid_argument(
          "tl::layer_norm: expect rank>=1 x and gamma, beta holding its last "
          "dim — got x " + shape_str(s) + ", gamma " + shape_str(gamma.shape()) +
          ", beta " + shape_str(beta.shape()));
    auto n = std::make_shared<node>();
    n->op = op_t::layer_norm_;
    n->shape = s;
    n->arg0 = eps;
    n->inputs = {as_node(x), as_node(gamma), as_node(beta)};
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
    return n >= (kc == kernel_class::matmul ? auto_matmul_threshold_()
                                            : auto_threshold_(kc));
  }

  // The matmul threshold in force: the one derived on this host by the first
  // auto-mode graph eval (calibrate_auto_ below) or pinned by TL_AUTO_MATMUL,
  // else types.h's census value. -1 = not derived yet.
  static inline int64_t auto_matmul_ = -1;
  static int64_t auto_matmul_threshold_() {
    return auto_matmul_ > 0 ? auto_matmul_
                            : auto_threshold_(kernel_class::matmul);
  }

  // The batch-bias threshold in force. TL_BATCH_MATMUL_BIAS pins it exactly
  // (batch_matmul_bias_threshold_ honors the env); otherwise it never sits
  // below the per-op matmul crossover, so "a batch earns the GPU at about the
  // size its largest gemm does" holds on whatever host calibrate_auto_
  // measured — the baked pair assumed that relationship, and calibration can
  // now move the crossover above the baked bias.
  static int64_t batch_bias_threshold_() {
    static const bool pinned = std::getenv("TL_BATCH_MATMUL_BIAS") != nullptr;
    int64_t baked = batch_matmul_bias_threshold_();
    return pinned ? baked : std::max(baked, auto_matmul_threshold_());
  }

  // Derive the matmul crossover on this host, once, at the first graph eval
  // in auto mode (TL_AUTO_TRACE=1 prints the census to stderr). The baked
  // value is a census of one box, and it moved 30x on the
  // same GPU when the CPU gemm stopped paying its pool's wake-up on every
  // call: the crossover is where THIS CPU meets THIS GPU, so measure it.
  // Square gemms at 96/128/192/256 (0.9e6..1.7e7 MAC), 2 warm-ups then 5
  // timings per device, medians; the threshold is the smallest size the GPU
  // wins at, or the baked value when it wins none (Accelerate's AMX holds
  // the CPU past 256^3, so that box keeps its 5e8). A few ms once the
  // context is up; the RTX 3090 lands on 192^3 = 7.1e6 (baked 6e6) run after
  // run. Called from the top of run_ only — evaluation is not re-entrant, so
  // never from an op — under forced cpu/gpu modes, so nothing below consults
  // the threshold being derived.
  static void calibrate_auto_() {
    if (auto_matmul_ > 0) return;
    if (const char* e = std::getenv("TL_AUTO_MATMUL")) {
      char* end = nullptr;
      long long x = std::strtoll(e, &end, 10);
      if (end != e && x > 0) {
        auto_matmul_ = static_cast<int64_t>(x);
        return;
      }
    }
    // The measurement runs outside the caller's evaluation state and leaves
    // none of its own behind — both halves, or the census lies twice over. A
    // caller evaluating inside a defer_flush scope (an autograd walk opens one)
    // lends it to the census unless it is suspended, and that scope suppresses
    // every flush below it: the GPU timings would drop the sync the auto rule
    // compares against the CPU, and win at the smallest size in the sweep. What
    // the census then left in flight is what gpu_mode_ reads as a pipeline not
    // to break, which sends every op of the caller's own graph to the GPU
    // however small.
    struct CensusScope {
      const device_type device = device_;
      const int depth = defer_flush_depth;
      CensusScope() { defer_flush_depth = 0; }
      ~CensusScope() {
        gpu::flush();
        defer_flush_depth = depth;
        device_ = device;
      }
    } census_scope;
    using clk = std::chrono::steady_clock;
    auto median_us = [](auto&& f) {
      f();
      f();
      double ts[5];
      for (double& t : ts) {
        auto t0 = clk::now();
        f();
        t = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
      }
      std::sort(std::begin(ts), std::end(ts));
      return ts[2];
    };
    const bool trace = std::getenv("TL_AUTO_TRACE") != nullptr;
    int64_t chosen = auto_threshold_(kernel_class::matmul);
    for (int64_t m : {96, 128, 192, 256}) {
      auto a = array::full({m, m}, 0.5f), b = array::full({m, m}, 0.25f);
      use_cpu();
      double cpu = median_us([&] { a.dot(b).eval(); });
      use_gpu();
      double gpu = median_us([&] { a.dot(b).eval(); });
      if (trace)
        std::fprintf(stderr, "tl: auto census %lld^3 cpu %.1f us gpu %.1f us\n",
                     (long long)m, cpu, gpu);
      if (gpu < cpu) {
        chosen = m * m * m;
        break;
      }
    }
    auto_matmul_ = chosen;
    if (trace)
      std::fprintf(stderr, "tl: auto matmul threshold %lld\n", (long long)chosen);
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
      if (!gpu::binary(*k, a.device_span(), b.device_span(), out.device_span(),
                       out.size(), n.scale, n.offset)) {
        return std::nullopt;
      }
      return out;
    }
    // Rank-2 broadcast (bias / row vector / column vector / scalar): one
    // stride-parameterized kernel keeps the op on the GPU — a CPU fallback
    // here drains the whole pending pipeline (commit + wait) mid-graph.
    // The kernels below read each operand through its own strides, so a
    // strided operand (a transposed view, or the stride-0 one broadcast_to
    // returns) belongs on this path too rather than on the CPU oracle.
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
    if (!gpu::binary_bcast(*bk, a.device_span(), ra[0], ra[1], b.device_span(),
                           rb[0], rb[1], out.device_span(), n.shape[0],
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
    if (!gpu::binary_bcast_nd(bk, a.device_span(), ra.data(), b.device_span(),
                              rb.data(), out.device_span(), out_shape_v.data(),
                              rank, out.size(), n.scale, n.offset)) {
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
    if (!gpu::where_nd(cond.device_span(), rc.data(), a.device_span(),
                       ra.data(), b.device_span(), rb.data(), out.device_span(),
                       out_shape_v.data(), rank, out.size())) {
      return std::nullopt;
    }
    return out;
  }

  // clone()'s device arm: one gather per output element, so a strided view (a
  // permute, a transpose) copies like a contiguous buffer, and bit-exact
  // (unlike an identity affine, which turns -0.0 into +0.0).
  static std::optional<array> gpu_copy_nd_(const array& a) {
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.storage_.native) {
      return std::nullopt;
    }
    int rank = static_cast<int>(a.rank());
    if (rank <= 0) return std::nullopt;
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    std::vector<int64_t> shape_v(a.shape().begin(), a.shape().end());
    if (!gpu::copy_nd(a.device_span(), a.strides_.data(), out.device_span(),
                      shape_v.data(), rank, out.size())) {
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

  // The elementwise binaries' one chain: the GPU kernel (which applies the
  // epilogue in its store, so it sets epi_done), accel, the CPU table.
  static array eval_binary_(const node& n, const array& a, const array& b,
                            bool& epi_done) {
    if (auto g = gpu_binary(n, a, b)) {
      epi_done = true;
      return std::move(*g);
    }
    if (auto o = accel::binary(n.op, a, b)) return std::move(*o);
    array r;
    visit_binary_op(n.op, [&](auto f) { r = map_binary(a, b, f); });
    return r;
  }

  // r + bias for a fused dot whose backend took no bias: an add node's own
  // chain, so the numbers are the unfused sum's.
  static array add_row_bias_(const array& r, const array& bias) {
    node add;
    add.op = op_t::add;
    add.shape = r.shape();
    bool epi_done = false;  // an identity epilogue either way
    return eval_binary_(add, r, bias, epi_done);
  }

  // A fused row bias still owed goes into the gemm's store when the backend
  // takes it (gpu::gemm_bias), which clears bias_owed.
  static std::optional<array> gpu_gemm(const node& n, const array& a_in,
                                         const array& b_in, bool& bias_owed) {
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
    if (bias_owed) {
      const array bias = wrap(*n.inputs[2]);
      if (bias.storage_.dt == tl::dtype::f32 && bias.contiguous() &&
          bias.storage_.native &&
          gpu::gemm_bias(a.device_span(), la->ld, la->trans, b.device_span(),
                         lb->ld, lb->trans, bias.device_span(),
                         out.device_span(), m, nn, k, n.scale, n.offset)) {
        bias_owed = false;
        return out.reshape(n.shape);
      }
    }
    if (!gpu::gemm(a.device_span(), la->ld, la->trans, b.device_span(), lb->ld,
                   lb->trans, out.device_span(), m, nn, k, n.scale, n.offset)) {
      return std::nullopt;
    }
    return out.reshape(n.shape);
  }

  // The batch walk gpu_bdot_/cpu_bdot_ share with ref::bdot: the batch axes
  // may carry any strides (a permuted view, a slice), so each slice's element
  // offset is the batch index dotted with them, relative to the view's start:
  // a device buffer adds x.offset_, a raw() pointer already has it. `step`
  // advances the odometer.
  struct batch_walk_ {
    std::vector<int64_t> idx;
    explicit batch_walk_(size_t batch_rank) : idx(batch_rank, 0) {}
    int64_t offset(const array& x) const {
      int64_t off = 0;
      for (size_t d = 0; d < idx.size(); d++) off += idx[d] * x.strides()[d];
      return off;
    }
    void step(const array& x) {
      for (size_t d = idx.size(); d-- > 0;) {
        if (++idx[d] < x.shape()[d]) return;
        idx[d] = 0;
      }
    }
  };

  // The batch axes of x as one linear stride, when they walk like a single
  // axis (each stride the product of the inner batch extents — the array's
  // own layout, or a slice of it; size-1 axes don't count): what a backend's
  // one-launch batched gemm can step by. A broadcast batch (stride 0
  // throughout) collapses to 0, which the backend reads as "the same slice
  // every time". nullopt for a permuted batch (a [B,H,…] view transposed to
  // [H,B,…]), which walks the batch_walk_ odometer per slice instead.
  static std::optional<int64_t> batch_stride_(const array& x) {
    size_t r = x.rank();
    int64_t stride = 0, extent = 1;
    for (size_t d = r - 2; d-- > 0;) {
      if (x.shape()[d] == 1) continue;
      if (extent == 1) {
        stride = x.strides()[d];
      } else if (x.strides()[d] != stride * extent) {
        return std::nullopt;
      }
      extent *= x.shape()[d];
    }
    return stride;
  }

  // Batched matmul, GPU dispatch. Each slice's 2-D layout is gemm_classify_'s
  // "row-major or its transpose", so a permuted view goes to the kernel's
  // transposed operand rather than declining — attention's q·kᵀ used to fall
  // all the way to ref::bdot over that (57 ms against 0.1 for the same
  // product plain). When the batch axes collapse to one stride per operand
  // the whole product is one gpu::gemm_batched launch (CUDA folds the batch
  // into its grid; the other backends decline); otherwise, or when the
  // backend declines a layout, it is a per-slice loop over the same gpu::gemm
  // entry point above, one launch per batch element on the same device queue
  // (H launches for attention's probs·v cost 0.43 ms against 0.1 fused).
  // The fused scale/offset go into the launch's epilogue like gpu_gemm's, so
  // `q·kᵀ * scale` never round-trips 2M scores through the host affine tail.
  // bdot_one_launch_ (types.h) off forces the loop, for the census.
  static std::optional<array> gpu_bdot_(const node& n, const array& a,
                                        const array& b) {
    size_t r = a.rank();
    int64_t m = a.shape()[r - 2], k = a.shape()[r - 1], nn = b.shape().back();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < r; i++) batch *= a.shape()[i];
    if (!gpu_mode_(m * nn * (k > 0 ? k : 1) * (batch > 0 ? batch : 1),
                   kernel_class::matmul)) {
      return std::nullopt;
    }
    if (!a.storage_.native || !b.storage_.native) return std::nullopt;
    auto la = detail::gemm_classify_(a), lb = detail::gemm_classify_(b);
    if (!la || !lb) return std::nullopt;
    shape_t out_shape(a.shape().begin(), a.shape().end() - 2);
    out_shape.push_back(m);
    out_shape.push_back(nn);
    auto out = array::empty(out_shape);
    if (!out.storage_.native) return std::nullopt;
    if (m == 0 || nn == 0 || batch == 0) return out;
    auto sa = batch_stride_(a), sb = batch_stride_(b);
    if (bdot_one_launch_ && sa && sb &&
        gpu::gemm_batched(a.device_span(), la->ld, la->trans, *sa,
                          b.device_span(), lb->ld, lb->trans, *sb,
                          out.device_span(), m, nn, k, batch, n.scale,
                          n.offset)) {
      return out;
    }
    batch_walk_ w(r - 2);
    for (int64_t bi = 0; bi < batch; bi++, w.step(a)) {
      if (!gpu::gemm(a.device_span().at(w.offset(a) * 4), la->ld, la->trans,
                     b.device_span().at(w.offset(b) * 4), lb->ld, lb->trans,
                     out.device_span().at(bi * m * nn * 4), m, nn, k, n.scale,
                     n.offset)) {
        return std::nullopt;
      }
    }
    return out;
  }

  // Batched matmul on the own CPU backend: one stride-aware cpu::sgemm per
  // slice (a transposed view passes its strides through, no packing detour),
  // walking the batch like gpu_bdot_; the scale rides as alpha and the offset
  // is the same post-pass cpu_gemm's caller uses. Gated by cpu::enabled_ as
  // cpu_gemm is, so oracle tests can force ref::bdot.
  static std::optional<array> cpu_bdot_(const node& n, const array& a,
                                        const array& b) {
    if (!cpu::enabled_) return std::nullopt;
    size_t r = a.rank();
    int64_t m = a.shape()[r - 2], k = a.shape()[r - 1], nn = b.shape().back();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < r; i++) batch *= a.shape()[i];
    shape_t out_shape(a.shape().begin(), a.shape().end() - 2);
    out_shape.push_back(m);
    out_shape.push_back(nn);
    auto out = array::empty(out_shape);
    if (m == 0 || nn == 0 || batch == 0) return out;
    const float* pa = a.raw();
    const float* pb = b.raw();
    float* po = out.data();
    const auto& as = a.strides();
    const auto& bs = b.strides();
    batch_walk_ w(r - 2);
    for (int64_t bi = 0; bi < batch; bi++, w.step(a)) {
      cpu::sgemm(pa + w.offset(a), as[r - 2], as[r - 1], pb + w.offset(b),
                 bs[r - 2], bs[r - 1], po + bi * m * nn, m, nn, k, n.scale);
    }
    apply_dot_offset_(out, n.offset);
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
    bool ok = bf16 ? gpu::gemv_bf16(a->device_span(), b.device_span(),
                                    out.device_span(), nn, k)
                   : gpu::gemv_f32(a->device_span(), b.device_span(),
                                   out.device_span(), nn, k);
    if (!ok) return std::nullopt;
    return out.reshape(n.shape);
  }

  // M8 int4-weight decode GEMV: a(1,K)f32 @ Wq(K,N)q4 -> (1,N)f32. Wq's logical
  // shape is [K,N]; its storage is packed [N,K] int4 + appended scales, so the
  // scales are the view N·K/2 bytes into the same buffer. Gated to the decode
  // shape; non-decode / non-GPU dequantizes to F32 via the input funnel.
  static std::optional<array> gpu_gemv_q4(const node& n, const array& a_in,
                                          const array& Wq) {
    if (Wq.storage_.dt != tl::dtype::q4 || Wq.rank() != 2) return std::nullopt;
    int64_t K = Wq.shape()[0], N = Wq.shape()[1];  // logical [K,N]
    auto a = gemv_act_(a_in, K, N);
    // No layout checks on Wq: make_view_ refuses q4, so it is always to_q4()'s leaf.
    if (!a || !Wq.storage_.native) return std::nullopt;
    array out = array::empty({int64_t{1}, N});
    if (!out.storage_.native) return std::nullopt;
    if (N == 0) return out.reshape(n.shape);
    const gpu::span qw = Wq.device_span();
    if (!gpu::gemv_q4(a->device_span(), qw, qw.at(N * K / 2), out.device_span(),
                      N, K, tl::kQ4Group)) {
      return std::nullopt;
    }
    return out.reshape(n.shape);
  }

  // M9 fused decode attention on the GPU. q[H,D], K/V[H,ctx,D] contiguous,
  // D==128. Returns nullopt (→ CPU ref) when the kernel declines.
  // q/K/V all contiguous, unoffset and device-resident — the shared
  // eligibility gate for the fused attention kernels (else → CPU ref).
  static bool attn_operands_ready_(const array& q, const array& K,
                                   const array& V) {
    return q.contiguous() && q.offset_ == 0 && K.contiguous() &&
           K.offset_ == 0 && V.contiguous() && V.offset_ == 0 &&
           q.storage_.native && K.storage_.native && V.storage_.native;
  }

  // One output row of reference attention: softmax over keys [0, klen) of
  // scale · q·Kh[j]ᵀ, then that distribution · Vh. q is D long; Kh/Vh are the
  // head's key/value bases; `s` is caller-owned scratch of at least klen. The
  // key bound is the only thing decode (klen=ctx) and causal prefill (klen=
  // t+1) differ by, so both references share this numerically-stable kernel.
  static void ref_attn_row_(const float* q, const float* Kh, const float* Vh,
                            int64_t klen, int64_t D, float scale, float* out,
                            std::vector<float>& s) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < klen; j++) {
      float acc = 0;
      for (int64_t d = 0; d < D; d++) acc += q[d] * Kh[j * D + d];
      s[j] = acc * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0;
    for (int64_t j = 0; j < klen; j++) {
      s[j] = std::exp(s[j] - mx);
      sum += s[j];
    }
    for (int64_t d = 0; d < D; d++) {
      float acc = 0;
      for (int64_t j = 0; j < klen; j++) acc += s[j] * Vh[j * D + d];
      out[d] = acc / sum;
    }
  }

  static std::optional<array> gpu_attn_(const node& n, const array& q,
                                        const array& K, const array& V) {
    int64_t H = q.shape()[0], D = q.shape()[1], ctx = K.shape()[1];
    if (!gpu_mode_(H * ctx * D, kernel_class::matmul)) return std::nullopt;
    if (!attn_operands_ready_(q, K, V)) return std::nullopt;
    array out = array::empty({H, D});
    if (!out.storage_.native) return std::nullopt;
    // Array path has no persistent cache: K/V are [H,ctx,D], so n_kv_heads==H
    // (no GQA) and kv_max==ctx (kv_stride==ctx*D degenerates to whole-buffer).
    if (!gpu::attn_decode(q.device_span(), K.device_span(), V.device_span(),
                          out.device_span(), H, H, ctx, ctx, D, n.arg0)) {
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
      ref_attn_row_(pq + h * D, pk + h * ctx * D, pv + h * ctx * D, ctx, D,
                    scale, po + h * D, s);
    }
    return out;
  }

  // M9 fused causal prefill attention on the GPU. q/K/V [H,T,D] contiguous,
  // D ∈ {64,128}. Returns nullopt (→ CPU ref) when the kernel declines.
  static std::optional<array> gpu_attn_prefill_(const node& n, const array& q,
                                                const array& K,
                                                const array& V) {
    int64_t H = q.shape()[0], T = q.shape()[1], D = q.shape()[2];
    if (!gpu_mode_(H * T * T * D, kernel_class::matmul)) return std::nullopt;
    if (!attn_operands_ready_(q, K, V)) return std::nullopt;
    array out = array::empty({H, T, D});
    if (!out.storage_.native) return std::nullopt;
    // No persistent cache on the array path: K/V are [H,T,D], so n_kv_heads==H
    // (no GQA) and kv_max==T (the whole buffer is the cache, filled from 0).
    if (!gpu::attn_prefill(q.device_span(), K.device_span(), V.device_span(),
                           out.device_span(), H, H, T, T, D, n.arg0)) {
      return std::nullopt;
    }
    return out;
  }

  // The eager ops below open their profile scope once their operands are
  // realized and pass the checks, as clone() does: a pending graph evaluates
  // under its own ops, and a call that declines on its operands leaves no row.
  // A backend without the kernel declines inside the scope, so a row with no
  // launches under it means the caller composed the unfused form.

  // Cross-entropy's pullback from the forward's row logsumexp (see
  // array::xent_bwd). Eager, the same bargain the attention halves below
  // make: one pass here, or the caller's composition. The device kernel in
  // GPU mode; on the own CPU, rows across the pool with the vector exp.
  static std::optional<array> xent_bwd(const array& x, const array& lse,
                                       const array& tgt, const array& g) {
    const auto& s = x.shape();
    if (s.size() != 2) {
      throw std::invalid_argument("tl::xent_bwd: expect rank-2 logits — got " +
                                  shape_str(s));
    }
    int64_t rows = s[0], cols = s[1];
    shape_t per_row{rows};
    if (lse.shape() != per_row || tgt.shape() != per_row ||
        g.shape() != per_row) {
      throw std::invalid_argument(
          "tl::xent_bwd: expect lse, targets and g all [" +
          std::to_string(rows) + "] — got lse " + shape_str(lse.shape()) +
          ", targets " + shape_str(tgt.shape()) + ", g " +
          shape_str(g.shape()));
    }
    const bool gpu = gpu_mode_(rows * cols, kernel_class::elementwise);
    if (!gpu && !cpu::enabled_) return std::nullopt;  // the oracle composes
    x.realize();
    lse.realize();
    tgt.realize();
    g.realize();
    if (!x.contiguous() || !lse.contiguous() || !tgt.contiguous() ||
        !g.contiguous()) {
      return std::nullopt;
    }
    if (!gpu) return cpu_xent_bwd_(x, lse, tgt, g);
    if (!x.storage_.native || !lse.storage_.native || !tgt.storage_.native ||
        !g.storage_.native) {
      return std::nullopt;
    }
    profile::scope ps("xent_bwd");
    auto out = array::empty(s);
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::xent_bwd(x.device_span(), lse.device_span(), tgt.device_span(),
                       g.device_span(), out.device_span(), rows, cols)) {
      return std::nullopt;
    }
    return out;
  }

  // xent_bwd on the own CPU, the kernel's arithmetic: p = exp(x - lse), minus
  // 1 at the target, times g.
  static array cpu_xent_bwd_(const array& x, const array& lse, const array& tgt,
                             const array& g) {
    profile::scope ps("xent_bwd");
    const int64_t rows = x.shape()[0], cols = x.shape()[1];
    auto out = array::empty(x.shape());
    const float *px = x.raw(), *pl = lse.raw(), *pt = tgt.raw(), *pg = g.raw();
    float* po = out.data();
    cpu::thread_pool::instance().parallel_for(
        rows,
        [&](int64_t r0, int64_t r1) {
          for (int64_t r = r0; r < r1; r++) {
            float* dst = po + r * cols;
            cpu::exp_shifted(dst, px + r * cols, 1, cols, pl[r]);
            int64_t k = static_cast<int64_t>(std::llround(pt[r]));
            if (k >= 0 && k < cols) dst[k] -= 1.0f;
            const float gr = pg[r];
            for (int64_t c = 0; c < cols; c++) dst[c] *= gr;
          }
        },
        own_threads_(x.size() * 128));
    return out;
  }

  // Layer norm's fused pullback (see array::layer_norm_bwd): graph::layer_norm's
  // shape rules, with dout on x's shape.
  static std::optional<std::array<array, 3>> layer_norm_bwd(
      const array& x, const array& gamma, const array& dout, float eps) {
    const auto& s = x.shape();
    int64_t d = s.empty() ? 0 : s.back();
    if (d <= 0 || !holds_last_(gamma, d) || dout.shape() != s) {
      throw std::invalid_argument(
          "tl::layer_norm_bwd: expect rank>=1 x, gamma holding its last axis "
          "and dout on x's shape — got x " + shape_str(s) + ", gamma " +
          shape_str(gamma.shape()) + ", dout " + shape_str(dout.shape()));
    }
    int64_t rows = x.size() / d;
    const bool gpu = gpu_mode_(x.size(), kernel_class::reduction);
    if (!gpu && !cpu::enabled_) return std::nullopt;  // the oracle composes
    x.realize();
    gamma.realize();
    dout.realize();
    if (!x.contiguous() || !gamma.contiguous() || !dout.contiguous()) {
      return std::nullopt;
    }
    if (!gpu) return cpu_layer_norm_bwd_(x, gamma, dout, eps);
    if (!x.storage_.native || !gamma.storage_.native ||
        !dout.storage_.native) {
      return std::nullopt;
    }
    profile::scope ps("layer_norm_bwd");
    // The column sums take the rows in chunks of at least 64, at most 64
    // chunks: enough blocks to fill the device on a tall input, one on a short.
    int64_t per_chunk = std::max<int64_t>(64, (rows + 63) / 64);
    int64_t chunks = (rows + per_chunk - 1) / per_chunk;
    array dx = array::empty(s), dg = array::empty({d}), db = array::empty({d});
    array stats = array::empty({2, rows}), partials = array::empty({2, chunks, d});
    if (!dx.storage_.native || !dg.storage_.native || !db.storage_.native ||
        !stats.storage_.native || !partials.storage_.native) {
      return std::nullopt;
    }
    if (!gpu::layer_norm_bwd(x.device_span(), gamma.device_span(),
                             dout.device_span(), dx.device_span(),
                             dg.device_span(), db.device_span(),
                             stats.device_span(), partials.device_span(), rows,
                             d, per_chunk, chunks, eps)) {
      return std::nullopt;
    }
    return std::array<array, 3>{dx, dg, db};
  }

  // layer_norm_bwd on the own CPU, the kernels' arithmetic: each row's mean
  // and rstd recomputed as ref::layer_norm folds them, dx = rstd · (ĝ −
  // mean(ĝ) − x̂ · mean(ĝ ⊙ x̂)) with ĝ = dout ⊙ γ. dγ and dβ sum over rows in
  // fixed chunks folded in chunk order, so the pool's split does not move a
  // bit. Chunks across the pool.
  static std::array<array, 3> cpu_layer_norm_bwd_(const array& x,
                                                  const array& gamma,
                                                  const array& dout, float eps) {
    profile::scope ps("layer_norm_bwd");
    const int64_t d = x.shape().back(), rows = x.size() / d;
    const int64_t per_chunk = std::max<int64_t>(64, (rows + 63) / 64);
    const int64_t chunks = (rows + per_chunk - 1) / per_chunk;
    array dx = array::empty(x.shape()), dg = array::empty({d}),
          db = array::empty({d});
    std::vector<float> partials(static_cast<size_t>(2 * chunks * d), 0.0f);
    const float *px = x.raw(), *pgam = gamma.raw(), *pdy = dout.raw();
    float* pdx = dx.data();
    const float inv_d = 1.0f / static_cast<float>(d);
    auto add = [](float& a, float v) { a += v; };
    cpu::thread_pool::instance().parallel_for(
        chunks,
        [&](int64_t c0, int64_t c1) {
          static thread_local std::vector<float> xh, gh, tmp;
          xh.resize(d);
          gh.resize(d);
          tmp.resize(d);
          for (int64_t c = c0; c < c1; c++) {
            float* sg = partials.data() + c * 2 * d;  // Σ dy ⊙ x̂, then Σ dy
            float* sb = sg + d;
            for (int64_t r = c * per_chunk; r < std::min(rows, (c + 1) * per_chunk); r++) {
              const float* src = px + r * d;
              const float* dy = pdy + r * d;
              float mu = detail::fold_lanes(src, 1, d, 0.0f, add) * inv_d;
              for (int64_t j = 0; j < d; j++) {
                float v = src[j] - mu;
                tmp[j] = v * v;
              }
              float var = detail::fold_lanes(tmp.data(), 1, d, 0.0f, add) * inv_d;
              float rstd = 1.0f / std::sqrt(var + eps);
              for (int64_t j = 0; j < d; j++) {
                xh[j] = (src[j] - mu) * rstd;
                gh[j] = dy[j] * pgam[j];
                tmp[j] = gh[j] * xh[j];
                sg[j] += dy[j] * xh[j];
                sb[j] += dy[j];
              }
              float mg = detail::fold_lanes(gh.data(), 1, d, 0.0f, add) * inv_d;
              float mgx = detail::fold_lanes(tmp.data(), 1, d, 0.0f, add) * inv_d;
              float* out = pdx + r * d;
              for (int64_t j = 0; j < d; j++)
                out[j] = rstd * (gh[j] - mg - xh[j] * mgx);
            }
          }
        },
        own_threads_(x.size() * 4 * kStreamMacs));
    float *pdg = dg.data(), *pdb = db.data();
    for (int64_t j = 0; j < d; j++) pdg[j] = pdb[j] = 0.0f;
    for (int64_t c = 0; c < chunks; c++) {
      const float* sg = partials.data() + c * 2 * d;
      for (int64_t j = 0; j < d; j++) {
        pdg[j] += sg[j];
        pdb[j] += sg[d + j];
      }
    }
    return {dx, dg, db};
  }

  // Adam's fused update (see array::adam_step). Eager, and in place on p, m
  // and v: an optimizer's state is materialized by nature, so there is no graph
  // to build and nothing to fuse into. The device kernel where there is one,
  // the host loop otherwise; false only for a non-contiguous layout.
  static bool adam_step(array& p, array& m, array& v, const array& g, float lr,
                        float beta1, float beta2, float eps, float bc1,
                        float bc2) {
    const auto& s = p.shape();
    if (m.shape() != s || v.shape() != s || g.shape() != s) {
      throw std::invalid_argument(
          "tl::adam_step: expect p, m, v and g to share one shape — got p " +
          shape_str(s) + ", m " + shape_str(m.shape()) + ", v " +
          shape_str(v.shape()) + ", g " + shape_str(g.shape()));
    }
    if (bc1 == 0.0f || bc2 == 0.0f) {
      throw std::invalid_argument(
          "tl::adam_step: bias correction of 0 (beta^t == 1)");
    }
    int64_t n = num_elements(s);
    if (n == 0) return true;
    p.realize();
    m.realize();
    v.realize();
    g.realize();
    if (!p.contiguous() || !m.contiguous() || !v.contiguous() ||
        !g.contiguous()) {
      return false;
    }
    profile::scope ps("adam_step");  // the device kernel or the host loop
    // The composition's own spelling: m · (lr/bc1) over sqrt(v · 1/bc2) + eps.
    const float lr_over_bc1 = lr / bc1, inv_bc2 = 1.0f / bc2;
    if (gpu_mode_(n, kernel_class::elementwise) && p.storage_.native &&
        m.storage_.native && v.storage_.native && g.storage_.native) {
      if (gpu::adam_step(p.device_span(), m.device_span(), v.device_span(),
                         g.device_span(), n, beta1, beta2, eps, lr_over_bc1,
                         inv_bc2)) {
        return true;
      }
      // No kernel for it on this backend: the host loop below. The caller
      // updates in place and cannot compose its way out, so this has to be
      // total -- and Metal's unified memory makes the round trip a flush and a
      // memcpy, the same bargain its other CPU fallbacks make.
    }
    // data() brings any device copy home first, the same as every other host
    // path. A CUDA build hands every buffer a mirror key, so "has a native
    // handle" is not "lives on the device" — only the mode is.
    float* pp = p.data();
    float* pm = m.data();
    float* pv = v.data();
    const float* pg = g.raw();
    for (int64_t i = 0; i < n; i++) {
      float gi = pg[i];
      float mi = beta1 * pm[i] + (1.0f - beta1) * gi;
      float vi = beta2 * pv[i] + (1.0f - beta2) * gi * gi;
      pm[i] = mi;
      pv[i] = vi;
      pp[i] -= (mi * lr_over_bc1) / (std::sqrt(vi * inv_bc2) + eps);
    }
    return true;
  }

  // The query half of the fused pullback (see array::attn_prefill_bwd_dq).
  // Eager: the kernel (or the own CPU's tiles) writes dq and the stats, or this
  // declines and the caller composes the unfused form. The shape rules are the
  // forward's, plus dout and out on the same [H,T,D].
  static std::optional<std::pair<array, array>> attn_prefill_bwd_dq(
      const array& q, const array& K, const array& V, const array& dout,
      const array& out, float scale) {
    const auto& s = q.shape();
    if (s.size() != 3 || K.shape() != s || V.shape() != s ||
        dout.shape() != s || out.shape() != s) {
      throw std::invalid_argument(
          "tl::attn_prefill_bwd_dq: expect q, K, V, dout, out all [H,T,D] — "
          "got q " + shape_str(s) + ", K " + shape_str(K.shape()) + ", V " +
          shape_str(V.shape()) + ", dout " + shape_str(dout.shape()) +
          ", out " + shape_str(out.shape()));
    }
    int64_t H = s[0], T = s[1], D = s[2];
    const bool gpu = gpu_mode_(H * T * T * D, kernel_class::matmul);
    if (!gpu && !cpu::enabled_) return std::nullopt;  // the oracle composes
    q.realize();
    K.realize();
    V.realize();
    dout.realize();
    out.realize();
    if (!gpu) {
      if (H * T * D == 0 || !q.contiguous() || !K.contiguous() ||
          !V.contiguous() || !dout.contiguous() || !out.contiguous()) {
        return std::nullopt;
      }
      return cpu_attn_bwd_dq_(q, K, V, dout, out, scale);
    }
    if (!attn_operands_ready_(q, K, V) ||
        !attn_operands_ready_(dout, out, out)) {
      return std::nullopt;
    }
    profile::scope ps("attn_prefill_bwd_dq");
    array dq = array::empty(s), stats = array::empty({2, H, T});
    if (!dq.storage_.native || !stats.storage_.native) return std::nullopt;
    if (!gpu::attn_prefill_dq(q.device_span(), K.device_span(), V.device_span(),
                              dout.device_span(), out.device_span(),
                              dq.device_span(), stats.device_span(), H, T, D,
                              scale)) {
      return std::nullopt;
    }
    return std::make_pair(dq, stats);
  }

  // The key/value half (see array::attn_prefill_bwd_dkv). Same rules, with the
  // dq half's stats standing in for the forward's output.
  static std::optional<std::pair<array, array>> attn_prefill_bwd_dkv(
      const array& q, const array& K, const array& V, const array& dout,
      const array& stats, float scale) {
    const auto& s = q.shape();
    if (s.size() != 3 || K.shape() != s || V.shape() != s ||
        dout.shape() != s ||
        stats.shape() != shape_t{2, s[0], s[1]}) {
      throw std::invalid_argument(
          "tl::attn_prefill_bwd_dkv: expect q, K, V, dout all [H,T,D] and "
          "stats [2,H,T] — got q " + shape_str(s) + ", K " +
          shape_str(K.shape()) + ", V " + shape_str(V.shape()) + ", dout " +
          shape_str(dout.shape()) + ", stats " + shape_str(stats.shape()));
    }
    int64_t H = s[0], T = s[1], D = s[2];
    const bool gpu = gpu_mode_(H * T * T * D, kernel_class::matmul);
    if (!gpu && !cpu::enabled_) return std::nullopt;  // the oracle composes
    q.realize();
    K.realize();
    V.realize();
    dout.realize();
    stats.realize();
    if (!gpu) {
      if (H * T * D == 0 || !q.contiguous() || !K.contiguous() ||
          !V.contiguous() || !dout.contiguous() || !stats.contiguous()) {
        return std::nullopt;
      }
      return cpu_attn_bwd_dkv_(q, K, V, dout, stats, scale);
    }
    if (!attn_operands_ready_(q, K, V) ||
        !attn_operands_ready_(dout, stats, stats)) {
      return std::nullopt;
    }
    profile::scope ps("attn_prefill_bwd_dkv");
    array dK = array::empty(s), dV = array::empty(s);
    if (!dK.storage_.native || !dV.storage_.native) return std::nullopt;
    if (!gpu::attn_prefill_dkv(q.device_span(), K.device_span(),
                               V.device_span(), dout.device_span(),
                               stats.device_span(), dK.device_span(),
                               dV.device_span(), H, T, D, scale)) {
      return std::nullopt;
    }
    return std::make_pair(dK, dV);
  }

  // Causal prefill attention on the own CPU backend, tiled over (head × a
  // block of BQ query rows) and run across the thread pool. A tile [t0,t1)
  // attends keys [0,t1) only — causality halves the work — as S = scale ·
  // Q_tile · Kᵀ (one stride-aware cpu::sgemm_, Kᵀ a stride pair, no copy), a
  // causal softmax over each row's keys 0..t (the rest set to 0), and the
  // context as a second sgemm_. The tile's gemms run inline on the worker
  // (max_threads = 1 never touches the pool, so nesting is fine) through the
  // same packed microkernel cpu::sgemm uses, and its BQ×T score block is a
  // thread-local scratch that stays in L2 — no online softmax needed at the
  // prefill lengths this serves. Later tiles see more keys, so the work
  // items alternate from both ends (0, n-1, 1, n-2, …) and the pool's static
  // split gets an even load. The single-threaded version of this (one head
  // at a time, T² scalar exps in series) sat at 7 ms for H=8 T=512 D=64,
  // 7x torch's CPU SDPA. Gated by cpu::enabled_ like cpu_gemm.
  static std::optional<array> cpu_attn_prefill_(const array& q, const array& K,
                                                const array& V, float scale) {
    if (!cpu::enabled_) return std::nullopt;
    int64_t H = q.shape()[0], T = q.shape()[1], D = q.shape()[2];
    array out = array::empty({H, T, D});
    if (H == 0 || T == 0 || D == 0) return out;
    const float* pq = q.raw();
    const float* pk = K.raw();
    const float* pv = V.raw();
    float* po = out.data();
    constexpr int64_t BQ = 64;
    const int64_t ntiles = (T + BQ - 1) / BQ, items = H * ntiles;
    // Both gemms over the causal half: H·T²·D multiply-adds.
    const int max_threads = cpu::threads_for_(H * T * T * D);
    cpu::thread_pool::instance().parallel_for(
        items,
        [&](int64_t i0, int64_t i1) {
          static thread_local std::vector<float> s;  // grow-only, like sgemm_'s packs
          if (s.size() < static_cast<size_t>(BQ * T)) s.resize(BQ * T);
          for (int64_t i = i0; i < i1; i++) {
            const int64_t slot = i / H, h = i % H;
            const int64_t tile = slot % 2 == 0 ? slot / 2 : ntiles - 1 - slot / 2;
            const int64_t t0 = tile * BQ, t1 = std::min(T, t0 + BQ), rows = t1 - t0;
            const float* qh = pq + (h * T + t0) * D;
            const float* Kh = pk + h * T * D;
            const float* Vh = pv + h * T * D;
            // S = scale · Q_tile · K[0:t1]ᵀ: B(p, j) = K_h[j*D + p], strides (1, D).
            cpu::sgemm_(qh, D, 1, Kh, 1, D, s.data(), rows, t1, D, scale, 1);
            for (int64_t r = 0; r < rows; r++) {
              const int64_t t = t0 + r;
              float* row = s.data() + r * t1;
              float mx = detail::fold_lanes(row, 1, t + 1, row[0],
                                            [](float& a, float v) { a = std::max(a, v); });
              float inv = 1.0f / cpu::exp_shifted(row, row, 1, t + 1, mx);
              for (int64_t j = 0; j <= t; j++) row[j] *= inv;
              for (int64_t j = t + 1; j < t1; j++) row[j] = 0.0f;
            }
            cpu::sgemm_(s.data(), t1, 1, Vh, D, 1, po + (h * T + t0) * D, rows, D,
                        t1, 1.0f, 1);
          }
        },
        max_threads);
    return out;
  }

  // The fused pullback's query half on the own CPU, tiled like the forward
  // (head × BQ query rows, alternating ends, gemms inline on the worker). A
  // tile rebuilds S = scale · Q Kᵀ over keys [0, t1), its causal softmax P and
  // each row's logsumexp L, then dP = dO Vᵀ, dS = P ⊙ (dP − Δ) with Δ = dO·O,
  // and dq = scale · dS K. stats [2,H,T] carries L and Δ to the key half, as
  // the CUDA kernels' does.
  static std::pair<array, array> cpu_attn_bwd_dq_(const array& q, const array& K,
                                                  const array& V, const array& dout,
                                                  const array& out, float scale) {
    profile::scope ps("attn_prefill_bwd_dq");
    const int64_t H = q.shape()[0], T = q.shape()[1], D = q.shape()[2];
    array dq = array::empty(q.shape()), stats = array::empty({2, H, T});
    const float *pq = q.raw(), *pk = K.raw(), *pv = V.raw(), *pdo = dout.raw(),
                *po = out.raw();
    float *pdq = dq.data(), *pst = stats.data();
    constexpr int64_t BQ = 64;
    const int64_t ntiles = (T + BQ - 1) / BQ, items = H * ntiles;
    cpu::thread_pool::instance().parallel_for(
        items,
        [&](int64_t i0, int64_t i1) {
          static thread_local std::vector<float> s, dp;
          if (s.size() < static_cast<size_t>(BQ * T)) s.resize(BQ * T);
          if (dp.size() < static_cast<size_t>(BQ * T)) dp.resize(BQ * T);
          for (int64_t i = i0; i < i1; i++) {
            const int64_t slot = i / H, h = i % H;
            const int64_t tile = slot % 2 == 0 ? slot / 2 : ntiles - 1 - slot / 2;
            const int64_t t0 = tile * BQ, t1 = std::min(T, t0 + BQ), rows = t1 - t0;
            const float* qh = pq + (h * T + t0) * D;
            const float* doh = pdo + (h * T + t0) * D;
            const float* Kh = pk + h * T * D;
            const float* Vh = pv + h * T * D;
            cpu::sgemm_(qh, D, 1, Kh, 1, D, s.data(), rows, t1, D, scale, 1);
            cpu::sgemm_(doh, D, 1, Vh, 1, D, dp.data(), rows, t1, D, 1.0f, 1);
            for (int64_t r = 0; r < rows; r++) {
              const int64_t t = t0 + r;
              float* srow = s.data() + r * t1;
              float* drow = dp.data() + r * t1;
              float mx = detail::fold_lanes(srow, 1, t + 1, srow[0],
                                            [](float& a, float v) { a = std::max(a, v); });
              float sum = cpu::exp_shifted(srow, srow, 1, t + 1, mx);
              float inv = 1.0f / sum;
              const float* orow = po + (h * T + t) * D;
              const float* dorow = doh + r * D;
              float delta = 0.0f;
              for (int64_t e = 0; e < D; e++) delta += dorow[e] * orow[e];
              for (int64_t j = 0; j <= t; j++) drow[j] = srow[j] * inv * (drow[j] - delta);
              for (int64_t j = t + 1; j < t1; j++) drow[j] = 0.0f;
              pst[h * T + t] = mx + std::log(sum);
              pst[H * T + h * T + t] = delta;
            }
            cpu::sgemm_(dp.data(), t1, 1, Kh, D, 1, pdq + (h * T + t0) * D, rows, D,
                        t1, scale, 1);
          }
        },
        cpu::threads_for_(H * T * T * D));
    return {dq, stats};
  }

  // The key half on the own CPU: a tile of BK keys [k0, k1) is attended by
  // queries [k0, T), so it rebuilds S = scale · Q Kᵀ over those, P = exp(S −
  // L) under the causal mask from the stats, dS = P ⊙ (dO Vᵀ − Δ), then
  // dV = Pᵀ dO and dK = scale · dSᵀ Q. Early key tiles see the most queries,
  // so the items alternate ends as the query half's do.
  static std::pair<array, array> cpu_attn_bwd_dkv_(const array& q, const array& K,
                                                   const array& V, const array& dout,
                                                   const array& stats, float scale) {
    profile::scope ps("attn_prefill_bwd_dkv");
    const int64_t H = q.shape()[0], T = q.shape()[1], D = q.shape()[2];
    array dK = array::empty(q.shape()), dV = array::empty(q.shape());
    const float *pq = q.raw(), *pk = K.raw(), *pv = V.raw(), *pdo = dout.raw(),
                *pst = stats.raw();
    float *pdk = dK.data(), *pdv = dV.data();
    constexpr int64_t BK = 64;
    const int64_t ntiles = (T + BK - 1) / BK, items = H * ntiles;
    cpu::thread_pool::instance().parallel_for(
        items,
        [&](int64_t i0, int64_t i1) {
          static thread_local std::vector<float> s, dp;
          if (s.size() < static_cast<size_t>(T * BK)) s.resize(T * BK);
          if (dp.size() < static_cast<size_t>(T * BK)) dp.resize(T * BK);
          for (int64_t i = i0; i < i1; i++) {
            const int64_t slot = i / H, h = i % H;
            const int64_t tile = slot % 2 == 0 ? slot / 2 : ntiles - 1 - slot / 2;
            const int64_t k0 = tile * BK, k1 = std::min(T, k0 + BK), nk = k1 - k0;
            const int64_t nq = T - k0;
            const float* qh = pq + (h * T + k0) * D;   // queries k0..T-1
            const float* doh = pdo + (h * T + k0) * D;
            const float* Kt = pk + (h * T + k0) * D;   // this tile's keys
            const float* Vt = pv + (h * T + k0) * D;
            const float* L = pst + h * T + k0;
            const float* delta = pst + H * T + h * T + k0;
            cpu::sgemm_(qh, D, 1, Kt, 1, D, s.data(), nq, nk, D, scale, 1);
            cpu::sgemm_(doh, D, 1, Vt, 1, D, dp.data(), nq, nk, D, 1.0f, 1);
            for (int64_t r = 0; r < nq; r++) {
              // Query k0 + r sees this tile's keys k0 .. k0 + r.
              const int64_t len = std::min(nk, r + 1);
              float* srow = s.data() + r * nk;
              float* drow = dp.data() + r * nk;
              cpu::exp_shifted(srow, srow, 1, len, L[r]);
              for (int64_t j = 0; j < len; j++) drow[j] = srow[j] * (drow[j] - delta[r]);
              for (int64_t j = len; j < nk; j++) srow[j] = drow[j] = 0.0f;
            }
            cpu::sgemm_(s.data(), 1, nk, doh, D, 1, pdv + (h * T + k0) * D, nk, D, nq,
                        1.0f, 1);
            cpu::sgemm_(dp.data(), 1, nk, qh, D, 1, pdk + (h * T + k0) * D, nk, D, nq,
                        scale, 1);
          }
        },
        cpu::threads_for_(H * T * T * D));
    return {dK, dV};
  }

  // CPU reference causal prefill attention (the oracle, and the fallback when
  // cpu::enabled_ is off): row t of head h is the decode reference over the
  // keys 0..t.
  static array ref_attn_prefill_(const array& q, const array& K,
                                 const array& V, float scale) {
    int64_t H = q.shape()[0], T = q.shape()[1], D = q.shape()[2];
    array out = array::empty({H, T, D});
    const float* pq = q.raw();
    const float* pk = K.raw();
    const float* pv = V.raw();
    float* po = out.data();
    std::vector<float> s(T);
    for (int64_t h = 0; h < H; h++) {
      const float* Kh = pk + h * T * D;
      const float* Vh = pv + h * T * D;
      for (int64_t t = 0; t < T; t++) {
        ref_attn_row_(pq + (h * T + t) * D, Kh, Vh, t + 1, D, scale,
                      po + (h * T + t) * D, s);
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
    if (!x.contiguous() || !x.storage_.native) return std::nullopt;
    array out = array::empty(x.shape());
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::rope(x.device_span(), out.device_span(), rows, T, D, n.axis,
                   n.arg0))
      return std::nullopt;
    return out;
  }

  // Layer norm on the GPU: x, gamma and beta contiguous (gamma/beta d-vectors,
  // graph::layer_norm checked). The kernels apply the epilogue in the store.
  static std::optional<array> gpu_layer_norm_(const node& n, const array& x,
                                              const array& g, const array& b) {
    if (!gpu_mode_(x.size(), kernel_class::reduction)) return std::nullopt;
    if (!x.contiguous() || !g.contiguous() || !b.contiguous())
      return std::nullopt;
    if (!x.storage_.native || !g.storage_.native || !b.storage_.native)
      return std::nullopt;
    auto out = array::empty(x.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    int64_t d = x.shape().back();
    if (!gpu::layer_norm(x.device_span(), g.device_span(), b.device_span(),
                         out.device_span(), x.size() / d, d, n.arg0, n.scale,
                         n.offset))
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
    if (!gpu::row_op(k, a.device_span(), out.device_span(), rows, cols, scale,
                     offset)) {
      return std::nullopt;
    }
    return out;
  }

  // The one-input elementwise GPU dispatches share this gate and output:
  // `call(a, out, n)` is the backend call, on the two views.
  template <typename Call>
  static std::optional<array> gpu_one_input_(const array& a, Call&& call) {
    if (!gpu_mode_(a.size(), kernel_class::elementwise) || !a.contiguous()) {
      return std::nullopt;
    }
    if (!a.storage_.native) return std::nullopt;
    auto out = array::empty(a.shape());
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!call(a.device_span(), out.device_span(), out.size())) {
      return std::nullopt;
    }
    return out;
  }

  static std::optional<array> gpu_unary(std::optional<gpu::kop> k,
                                        const array& a, float scale,
                                        float offset) {
    if (!k) return std::nullopt;  // no kernel for this op on this backend
    return gpu_one_input_(a, [&](gpu::span in, gpu::span out, int64_t n) {
      return gpu::unary(*k, in, out, n, scale, offset);
    });
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
                           return gpu::pad(a.device_span(), out.device_span(),
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
                           return gpu::fold(a.device_span(), out.device_span(),
                                            a_shape.data(), out_shape_v.data(),
                                            rank, static_cast<int>(axis), step,
                                            a.size(), out.size());
                         });
  }

  // The GPU-dispatch twin of ref::concat: one gpu::concat_part() launch per
  // part into a shared `out`, reusing pad's own kernel (writing a
  // same-shape source at an axis-shifted offset is exactly what pad already
  // does per source element) minus its pre-zero — concat's parts
  // exhaustively cover `out`, unlike pad's bordered write. All parts are
  // validated up front so a mid-loop decline can't leave `out` half
  // written and returned; a nullopt here just re-does the whole thing on
  // the CPU oracle.
  static std::optional<array> gpu_concat_(const std::vector<array>& parts,
                                          size_t axis,
                                          const shape_t& out_shape) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise)) {
      return std::nullopt;
    }
    for (auto& p : parts) {
      if (!p.contiguous() || !p.storage_.native) return std::nullopt;
    }
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    std::vector<int64_t> out_shape_v(out_shape.begin(), out_shape.end());
    int rank = static_cast<int>(out_shape_v.size());
    int64_t offset = 0;
    for (auto& p : parts) {
      std::vector<int64_t> p_shape(p.shape().begin(), p.shape().end());
      if (!gpu::concat_part(p.device_span(), out.device_span(), p_shape.data(),
                            out_shape_v.data(), rank, static_cast<int>(axis),
                            offset, p.size())) {
        return std::nullopt;
      }
      offset += p_shape[static_cast<size_t>(axis)];
    }
    return out;
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
    if (!gpu::index_select(a.device_span(), indices.device_span(),
                           out.device_span(), row_size, out_shape[0])) {
      return std::nullopt;
    }
    return out;
  }

  // The three index ops read their labels out of a tensor, so a label past
  // the end is bad input rather than a bug in this file -- and unchecked it
  // is an out-of-bounds write (index_add, scatter_to_axis) or read
  // (index_select) straight past the buffer, because every kernel indexes
  // bare-handed by design. Checked here, once, ahead of the backend fork:
  // the reference walk and the device kernels then agree on which labels are
  // an error instead of one corrupting memory while the other's gather
  // quietly returns zeros. `raw()` pulls device-resident labels back to the
  // host, which is the cost of saying so.
  //
  // invalid_argument, the same exception the shape checks on these three ops
  // already raise: out_of_range would put a caller's mistake in the same
  // bucket as this library's own logic errors.
  static void check_labels_(const array& indices, int64_t limit,
                            const char* what) {
    const float* p = indices.raw();
    std::optional<int64_t> bad;
    for_each_index(indices.shape(), {indices.strides()},
                   [&](int64_t, const std::vector<int64_t>& off) {
                     if (bad) return;
                     auto k = static_cast<int64_t>(std::llround(p[off[0]]));
                     if (k < 0 || k >= limit) bad = k;
                   });
    if (bad) {
      throw std::invalid_argument(
          std::string("tl::") + what + ": index " + std::to_string(*bad) +
          " out of range [0, " + std::to_string(limit) + ")");
    }
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
    if (!gpu::index_add(indices.device_span(), values.device_span(),
                        out.device_span(), row_size, indices.shape()[0],
                        out.size())) {
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
    if (!gpu::scatter_to_axis(indices.device_span(), values.device_span(),
                              out.device_span(), values.size(),
                              out_shape.back())) {
      return std::nullopt;
    }
    return out;
  }

  // scatter_to_axis's dual — the GPU-dispatch twin of ref::gather_from_axis.
  // A pure gather: nothing to pre-zero, no atomics.
  static std::optional<array> gpu_gather_from_axis_(const array& src,
                                                    const array& indices,
                                                    const shape_t& out_shape) {
    if (!gpu_mode_(num_elements(out_shape), kernel_class::elementwise) ||
        !src.contiguous() || !indices.contiguous()) {
      return std::nullopt;
    }
    if (!src.storage_.native || !indices.storage_.native) return std::nullopt;
    auto out = array::empty(out_shape);
    if (out.size() == 0) return out;
    if (!out.storage_.native) return std::nullopt;
    if (!gpu::gather_from_axis(src.device_span(), indices.device_span(),
                               out.device_span(), out.size(),
                               src.shape().back())) {
      return std::nullopt;
    }
    return out;
  }

  // Row logsumexp over the last axis: gpu_row's shape rules, but its own
  // entry point rather than a kop — the enum is dispatched unconditionally
  // (see metal.h's comment), and this kernel is CUDA's alone for now.
  static std::optional<array> gpu_row_logsumexp_(const array& a,
                                                 const shape_t& out_shape,
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
    if (!gpu::row_logsumexp(a.device_span(), out.device_span(), rows, cols,
                            scale, offset)) {
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
    if (!gpu::sum_to(a.device_span(), a_shape_v.data(), a_strides_v.data(),
                     acc.data(), rank, out.size(), reduced_n,
                     out.device_span())) {
      return std::nullopt;
    }
    return out;
  }

  // An axis reduction that is not over the last axis has no row kernel (those
  // read the last axis contiguously), but it is exactly a sum_to of the
  // keepdims shape -- so the blocked reduction above covers it, and a
  // `x.sum(0)` stops being the one reduction that runs on the host. nullopt
  // when the device declines; the caller then takes the CPU oracle.
  static std::optional<array> gpu_reduce_axis_(const array& a, int axis,
                                               bool keepdims,
                                               const shape_t& out_shape) {
    int rank = static_cast<int>(a.rank());
    if (axis < 0 || axis >= rank) return std::nullopt;
    shape_t kd = a.shape();
    kd[static_cast<size_t>(axis)] = 1;
    auto g = gpu_sum_to_(a, kd);
    if (!g) return std::nullopt;
    if (keepdims) return g;
    return g->reshape(out_shape);
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
    if (!gpu::compare(c, a.device_span(), b.device_span(), out.device_span(),
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
    gpu::unary_ext_op u;
    switch (op) {
      case op_t::tanh_: u = gpu::unary_ext_op::tanh_; break;
      case op_t::sin_: u = gpu::unary_ext_op::sin_; break;
      case op_t::cos_: u = gpu::unary_ext_op::cos_; break;
      default: return std::nullopt;
    }
    return gpu_one_input_(a, [&](gpu::span in, gpu::span out, int64_t n) {
      return gpu::unary_ext(u, in, out, n, scale, offset);
    });
  }

  // GPU dispatch for clamp_ (Clip's forward -- Culebra's dz.raw_elementwise
  // host loop before this). No epilogue: clamp's own two params occupy the
  // role scale/offset play elsewhere, and nothing composes a further affine
  // onto it today.
  static std::optional<array> gpu_clamp_(const array& a, float lo, float hi) {
    return gpu_one_input_(a, [&](gpu::span in, gpu::span out, int64_t n) {
      return gpu::clamp(in, out, n, lo, hi);
    });
  }

  // GPU dispatch for the tensor-scalar ops; the node's epilogue fuses into the
  // store, as in gpu_unary.
  static std::optional<array> gpu_scalar_binary_(const node& n, const array& a) {
    gpu::scalar_op k;
    switch (n.op) {
      case op_t::pow_s: k = gpu::scalar_op::pow; break;
      case op_t::gt_s: k = gpu::scalar_op::gt; break;
      case op_t::lt_s: k = gpu::scalar_op::lt; break;
      case op_t::ge_s: k = gpu::scalar_op::ge; break;
      case op_t::le_s: k = gpu::scalar_op::le; break;
      case op_t::eq_s: k = gpu::scalar_op::eq; break;
      case op_t::ne_s: k = gpu::scalar_op::ne; break;
      default: return std::nullopt;
    }
    return gpu_one_input_(a, [&](gpu::span in, gpu::span out, int64_t len) {
      return gpu::scalar_binary(k, in, out, len, n.arg0, n.scale, n.offset);
    });
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
    profile::detail::env_autostart();
    // First auto-mode eval on a host with a GPU: derive the matmul crossover
    // first. Its census evals each nest a run_, and `roots` may be
    // materialize_'s own thread-local scratch, which those evals reuse — so
    // hold a copy across them and start this eval over with it.
    if (device_ == device_type::auto_ && auto_matmul_ < 0 &&
        gpu::available() && !gpu::pending()) {
      std::vector<node_ptr> held(roots);
      calibrate_auto_();
      run_(held, do_flush);
      return;
    }
    // Thread-local scratch: run() fires once per eval batch and tiny-graph
    // workloads are per-op-allocation-bound. Nested evaluation cannot happen
    // (kernels never build or evaluate graphs; the one-time census above
    // runs before this scratch is touched), so reuse is safe. Visited
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
        if (n->op != node::op_t::dot) continue;
        const auto& sa = n->inputs[0]->shape;
        const auto& sb = n->inputs[1]->shape;
        if (sa.size() == 2 && sb.size() == 2) work += sa[0] * sa[1] * sb[1];
      }
      if (work >= batch_bias_threshold_()) batch_gpu_bias_ = true;
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
    const float* pa = detail::host_read_(a.stor, a.soffset);
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
        const float* pb = detail::host_read_(b.stor, b.soffset);
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
      default:  // unary and tensor-scalar ops the tables know; the rest decline
        return visit_unary_op(n.op, unary_loop) ||
               visit_scalar_op(n.op, n.arg0, unary_loop);
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
    const node* bias = fused_bias_(n);
    if (a.stor.dt != tl::dtype::f32 || b.stor.dt != tl::dtype::f32 ||
        (bias && bias->stor.dt != tl::dtype::f32))
      return false;  // bf16 operands take the eval_one dot path
    if (a.shape.size() != 2 || b.shape.size() != 2) return false;
    int64_t m = a.shape[0], k = a.shape[1], nn = b.shape[1];
    if (m * nn * k > kCutoff || m * nn == 0) return false;
    if (gpu_mode_(m * nn * k, kernel_class::matmul)) return false;
    storage out = storage::make(m * nn);
    const float* pa = detail::host_read_(a.stor, a.soffset);
    const float* pb = detail::host_read_(b.stor, b.soffset);
    const float* pbias = bias ? detail::host_read_(bias->stor, bias->soffset) : nullptr;
    const int64_t bias_stride = bias ? bias->strides.back() : 0;
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
        const float y = acc * s + o;
        po[i * nn + j] = pbias ? y + pbias[j * bias_stride] : y;
      }
    }
    store_raw_(n, std::move(out));
    return true;
  }

  // The op's name as tl::profile labels the evaluator's scope for it.
  static const char* op_name_(node::op_t op) {
    using op_t = node::op_t;
    switch (op) {
      case op_t::constant: return "constant";
      case op_t::add: return "add";
      case op_t::sub: return "sub";
      case op_t::mul: return "mul";
      case op_t::div: return "div";
      case op_t::pow_: return "pow";
      case op_t::gt: return "gt";
      case op_t::lt: return "lt";
      case op_t::ge: return "ge";
      case op_t::le: return "le";
      case op_t::eq: return "eq";
      case op_t::ne: return "ne";
      case op_t::pow_s: return "pow_s";
      case op_t::gt_s: return "gt_s";
      case op_t::lt_s: return "lt_s";
      case op_t::ge_s: return "ge_s";
      case op_t::le_s: return "le_s";
      case op_t::eq_s: return "eq_s";
      case op_t::ne_s: return "ne_s";
      case op_t::affine: return "affine";
      case op_t::recip: return "recip";
      case op_t::exp_: return "exp";
      case op_t::log_: return "log";
      case op_t::sqrt_: return "sqrt";
      case op_t::sigmoid: return "sigmoid";
      case op_t::relu: return "relu";
      case op_t::tanh_: return "tanh";
      case op_t::sin_: return "sin";
      case op_t::cos_: return "cos";
      case op_t::clamp_: return "clamp";
      case op_t::softmax: return "softmax";
      case op_t::where_: return "where";
      case op_t::dot: return "dot";
      case op_t::attn_dec: return "attn_decode";
      case op_t::attn_pre: return "attn_prefill";
      case op_t::rope: return "rope";
      case op_t::layer_norm_: return "layer_norm";
      case op_t::sum_ax: return "sum";
      case op_t::mean_ax: return "mean";
      case op_t::max_ax: return "max";
      case op_t::argmax_ax: return "argmax";
      case op_t::sum_to_: return "sum_to";
      case op_t::lse_ax: return "logsumexp";
      case op_t::pad_: return "pad";
      case op_t::fold_: return "fold";
      case op_t::index_select_: return "index_select";
      case op_t::index_add_: return "index_add";
      case op_t::scatter_axis_: return "scatter_axis";
      case op_t::gather_axis_: return "gather_axis";
      case op_t::concat_: return "concat";
      case op_t::view_: return "view";
    }
    return "?";
  }

  static void eval_one(node& n) {
    if (n.op == node::op_t::constant) return;
    profile::scope ps(op_name_(n.op));
    if (try_fast_ew_(n)) return;
    if (n.op == node::op_t::dot && try_fast_dot_(n)) return;
    // Input funnel. bf16 inputs widen to an F32 copy here — the universal
    // fallback that keeps every backend kernel F32-only; the sole native bf16
    // consumer (decode GEMV) intercepts in the dot case before this runs.
    auto in = [&](size_t i) {
      array x = wrap(*n.inputs[i]);
      if (x.storage_.dt != tl::dtype::f32) x = x.to_f32();
      return x;
    };
    // For the kernels that index rows as `base + r * D` (attention, rope): a
    // strided view (a narrow along T or D) is packed first.
    auto packed = [&](size_t i) {
      array x = in(i);
      return x.contiguous() ? x : x.clone();
    };
    array r;
    bool epi_done = false;  // epilogue already applied inside the op body
    bool bias_owed = false;  // a fused dot bias not yet added (fused_bias_)
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
        r = eval_binary_(n, in(0), in(1), epi_done);
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
      case op_t::affine:
        r = affine_(in(0), n.scale, n.offset);
        epi_done = true;
        break;
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
      case op_t::pow_s:
      case op_t::gt_s:
      case op_t::lt_s:
      case op_t::ge_s:
      case op_t::le_s:
      case op_t::eq_s:
      case op_t::ne_s: {
        auto a = in(0);
        if (auto g = gpu_scalar_binary_(n, a)) {
          r = std::move(*g);
          epi_done = true;  // the kernels apply the epilogue in the store
        } else {
          visit_scalar_op(n.op, n.arg0, [&](auto f) { r = map_unary(a, f); });
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
          // An element costs about a hundred multiply-adds of time (the exp
          // plus the max and normalise passes).
          r = ref::softmax(a, own_threads_(a.size() * 128), cpu::enabled_);
        }
        break;
      }
      case op_t::dot: {
        // Batched (rank >= 3, graph::dot already required matching batch
        // dims): none of the GEMV/GEMM fast paths below know about a batch
        // axis, so this branches off before them entirely. Per-slice GPU
        // dispatch first, then the own CPU gemm per slice; ref::bdot is the
        // scalar oracle when both decline.
        if (n.inputs[0]->shape.size() > 2 || n.inputs[1]->shape.size() > 2) {
          auto a = in(0), b = in(1);
          if (auto g = gpu_bdot_(n, a, b)) {
            r = std::move(*g);
            epi_done = true;
          } else if (auto c = cpu_bdot_(n, a, b)) {
            r = std::move(*c);
            epi_done = true;
          } else {
            r = ref::bdot(a, b);
          }
          break;
        }
        // A fused row bias (fused_bias_) is owed until the CUDA gemm adds it in
        // its store; the tail adds it after the epilogue otherwise.
        bias_owed = fused_bias_(n) != nullptr;
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
        if (auto g = gpu_gemm(n, a, b, bias_owed)) {
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
          r = ref_attn_(packed(0), packed(1), packed(2), n.arg0);
        }
        break;
      }
      case op_t::attn_pre: {
        if (auto g = gpu_attn_prefill_(n, wrap(*n.inputs[0]),
                                       wrap(*n.inputs[1]),
                                       wrap(*n.inputs[2]))) {
          r = std::move(*g);
          break;
        }
        auto q = packed(0), K = packed(1), V = packed(2);
        if (auto c = cpu_attn_prefill_(q, K, V, n.arg0)) {
          r = std::move(*c);
        } else {
          r = ref_attn_prefill_(q, K, V, n.arg0);
        }
        break;
      }
      case op_t::rope: {
        if (auto g = gpu_rope_(n, wrap(*n.inputs[0]))) {
          r = std::move(*g);
        } else {
          r = ref_rope_(packed(0), n.axis, n.arg0);
        }
        break;
      }
      case op_t::layer_norm_: {
        auto x = in(0), g = in(1), b = in(2);
        if (auto gp = gpu_layer_norm_(n, x, g, b)) {
          r = std::move(*gp);
          epi_done = true;
        } else {
          // Four streaming passes per element: sum, square, sum, store.
          r = ref::layer_norm(x, g, b, n.arg0,
                              own_threads_(x.size() * 4 * kStreamMacs));
        }
        break;
      }
      case op_t::lse_ax: {
        // Only ever built for the last axis (array::logsumexp composes any
        // other), so this is the fused row kernel or the one-pass CPU fold.
        auto a = in(0);
        if (auto g = gpu_row_logsumexp_(a, n.shape, n.scale, n.offset)) {
          r = std::move(*g);
          epi_done = true;
          break;
        }
        r = ref::logsumexp(a, n.shape, own_threads_(a.size() * kStreamMacs),
                           cpu::enabled_);
        break;
      }
      case op_t::sum_ax:
      case op_t::max_ax: {
        // GPU row reductions cover the last-axis case (softmax/argmax
        // support shape) and fold the epilogue in, so mark it done. Any other
        // axis goes through sum_to's blocked kernel, which does not fold it;
        // max has no sum_to dual and still takes the CPU oracle.
        auto a = in(0);
        gpu::kop k =
            n.op == op_t::sum_ax ? gpu::kop::row_sum : gpu::kop::row_max;
        if (n.axis == static_cast<int>(a.rank()) - 1) {
          if (auto g = gpu_row(k, a, n.shape, n.scale, n.offset)) {
            r = std::move(*g);
            epi_done = true;
            break;
          }
        } else if (n.op == op_t::sum_ax) {
          if (auto g = gpu_reduce_axis_(a, n.axis, n.keepdims, n.shape)) {
            r = std::move(*g);
            break;
          }
        }
        const int nt = own_threads_(a.size() * kStreamMacs);
        r = n.op == op_t::sum_ax ? ref::sum(a, n.axis, n.keepdims, nt)
                                 : ref::max(a, n.axis, n.keepdims, nt);
        break;
      }
      case op_t::mean_ax: {
        // Last-axis mean lowers to the row_sum kernel with 1/cols folded into
        // the epilogue scale — no dedicated kernel, and no CPU fallback that
        // would drain the GPU pipeline mid-graph (layer-norm's op mix). Any
        // other axis sums through the blocked kernel and takes 1/dim in the
        // affine that follows, which is the epilogue.
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
        } else if (n.axis < static_cast<int>(a.rank()) &&
                   a.shape()[static_cast<size_t>(n.axis)] > 0) {
          float inv =
              1.0f / static_cast<float>(a.shape()[static_cast<size_t>(n.axis)]);
          if (auto g = gpu_reduce_axis_(a, n.axis, n.keepdims, n.shape)) {
            r = affine_(*g, n.scale * inv, n.offset);
            epi_done = true;
            break;
          }
        }
        r = ref::mean(a, n.axis, n.keepdims, own_threads_(a.size() * kStreamMacs));
        break;
      }
      case op_t::argmax_ax:
        r = ref::argmax(in(0), n.axis, n.keepdims);
        break;
      case op_t::sum_to_: {
        auto a = in(0);
        if (auto g = gpu_sum_to_(a, n.shape)) {
          r = std::move(*g);
        } else if (cpu::enabled_) {
          r = own_sum_to_(a, n.shape, own_threads_(a.size() * kStreamMacs));
        } else {
          r = ref::sum_to(a, n.shape);
        }
        break;
      }
      case op_t::concat_: {
        std::vector<array> parts;
        parts.reserve(n.inputs.size());
        for (size_t i = 0; i < n.inputs.size(); i++) parts.push_back(in(i));
        size_t axis = static_cast<size_t>(n.axis);
        if (auto g = gpu_concat_(parts, axis, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::concat(parts, axis, n.shape);
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
        check_labels_(indices, a.shape()[0], "index_select");
        if (auto g = gpu_index_select_(a, indices, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::index_select(a, indices, n.shape,
                                own_threads_(num_elements(n.shape) * kStreamMacs));
        }
        break;
      }
      case op_t::index_add_: {
        auto indices = in(0);
        auto values = in(1);
        check_labels_(indices, n.shape[0], "index_add");
        if (auto g = gpu_index_add_(indices, values, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::index_add(indices, values, n.shape);
        }
        break;
      }
      case op_t::gather_axis_: {
        auto src = in(0);
        auto indices = in(1);
        check_labels_(indices, src.shape().back(), "gather_from_axis");
        if (auto g = gpu_gather_from_axis_(src, indices, n.shape)) {
          r = std::move(*g);
        } else {
          r = ref::gather_from_axis(src, indices, n.shape);
        }
        break;
      }
      case op_t::scatter_axis_: {
        auto indices = in(0);
        auto values = in(1);
        check_labels_(indices, n.shape.back(), "scatter_to_axis");
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
      r = affine_(r, n.scale, n.offset);
    }
    if (bias_owed) r = add_row_bias_(r, in(2));
    store(n, r);
  }

  // y = a * s + o on evaluated data: the op_t::affine body, and the epilogue
  // of every op whose kernel did not fold the fused scale/offset in. The GPU
  // kernel comes first: a view of GPU-resident data with an affine fused onto
  // it (`x.mean(1).reshape({seq, 1}) + eps`) must not drop to the host here —
  // that drains the pipeline and pulls the data back mid-graph, and every
  // consumer after it lands on the CPU too (layer_norm written that way ran
  // 2.5x slower than with keepdims).
  static array affine_(const array& a, float s, float o) {
    if (auto g = gpu_unary(gpu::kop::affine, a, s, o)) return std::move(*g);
    if (auto out = accel::affine(a, s, o)) return std::move(*out);
    return map_unary(a, [s, o](float x) { return x * s + o; });
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
  if (detail::defer_flush_depth > 0) do_flush = false;
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
  return detail::graph::scalar_binary(detail::node::op_t::pow_s, a, s);
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
inline array operator>(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::gt_s, a, s);
}
inline array operator<(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::lt_s, a, s);
}
inline array operator>=(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::ge_s, a, s);
}
inline array operator<=(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::le_s, a, s);
}
inline array operator==(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::eq_s, a, s);
}
inline array operator!=(const array& a, float s) {
  return detail::graph::scalar_binary(detail::node::op_t::ne_s, a, s);
}

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
inline array gather_from_axis(const array& src, const array& indices) {
  return detail::graph::gather_from_axis(src, indices);
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
    auto* po = detail::host_write_(storage_, offset_);
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
  auto* po = detail::host_write_(storage_, offset_);
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

inline array array::attn_prefill(const array& q, const array& K,
                                 const array& V, float scale) {
  return detail::graph::attn_prefill(q, K, V, scale);
}

inline std::optional<std::pair<array, array>> array::attn_prefill_bwd_dq(
    const array& q, const array& K, const array& V, const array& dout,
    const array& out, float scale) {
  return detail::graph::attn_prefill_bwd_dq(q, K, V, dout, out, scale);
}

inline bool array::adam_step(array& p, array& m, array& v, const array& g,
                             float lr, float beta1, float beta2, float eps,
                             float bc1, float bc2) {
  return detail::graph::adam_step(p, m, v, g, lr, beta1, beta2, eps, bc1, bc2);
}

inline std::optional<std::pair<array, array>> array::attn_prefill_bwd_dkv(
    const array& q, const array& K, const array& V, const array& dout,
    const array& stats, float scale) {
  return detail::graph::attn_prefill_bwd_dkv(q, K, V, dout, stats, scale);
}

inline std::optional<array> array::xent_bwd(const array& logits,
                                            const array& lse,
                                            const array& targets,
                                            const array& g) {
  return detail::graph::xent_bwd(logits, lse, targets, g);
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

inline array array::layer_norm(const array& x, const array& gamma,
                               const array& beta, float eps) {
  return detail::graph::layer_norm(x, gamma, beta, eps);
}

inline std::optional<std::array<array, 3>> array::layer_norm_bwd(
    const array& x, const array& gamma, const array& dout, float eps) {
  return detail::graph::layer_norm_bwd(x, gamma, dout, eps);
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
inline array array::logsumexp(int axis, bool keepdims) const {
  int ax = axis;
  auto out_shape = detail::reduce_shape(shape_, ax, keepdims);  // normalizes ax
  if (ax == static_cast<int>(rank()) - 1) {
    return detail::graph::reduce(detail::node::op_t::lse_ax, *this, ax,
                                 keepdims);
  }
  // No fused kernel off the last axis; the stable composition is exact and
  // every op in it already has one.
  auto m = max(ax, /*keepdims=*/true);
  auto r = (*this - m).exp().sum(ax, /*keepdims=*/true).log() + m;
  return keepdims ? r : r.reshape(out_shape);
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

inline array concat(const std::vector<array>& parts, int axis) {
  return detail::graph::concat(parts, axis);
}
inline array concat(const std::vector<array>& parts) { return concat(parts, 0); }

// Install the execution-engine hooks (TL_RUNTIME_HOOKS builds). Call once,
// before any evaluation, from the embedder's tensor feature loader. This is
// the only function referencing the evaluator and device backends by name —
// keep it out of translation units that must stay backend-free.
inline void install_runtime_hooks() {
  detail::storage_make_hook = &storage::make_bytes_;
  detail::cpu_barrier_hook = &gpu::cpu_barrier;
  detail::host_sync_hook = &gpu::sync_to_host;
  detail::gpu_pending_hook = &gpu::pending;
  detail::run_hook = &detail::graph::run;
  detail::run_noflush_hook = &detail::graph::run_noflush;
  detail::flush_hook = &gpu::flush;
  detail::device_clone_hook = &detail::device_clone_;
}

// Waits for the device to finish what is in flight — results realized
// without a sync (clone, realize) included — unless a defer_flush scope is
// open, whose end waits instead.
inline void synchronize() {
  if (detail::defer_flush_depth > 0) return;
#ifdef TL_RUNTIME_HOOKS
  if (detail::flush_hook) detail::flush_hook();
#else
  gpu::flush();
#endif
}

// Leaves every evaluation inside the scope in flight on the device and drains
// the stream once when the outermost scope ends: a caller that evaluates op by
// op (an autograd walk) pays one sync instead of one per evaluation.
class defer_flush {
 public:
  defer_flush() { ++detail::defer_flush_depth; }
  ~defer_flush() {
    if (--detail::defer_flush_depth == 0) synchronize();
  }
  defer_flush(const defer_flush&) = delete;
  defer_flush& operator=(const defer_flush&) = delete;
};

namespace detail {
// clone()'s device arm: a bit-exact gather on the device, contiguous or
// strided, launched straight on evaluated data — no graph, so a view's
// reshape may clone mid-eval without a nested run. nullopt when the device
// declines (no kernel on this backend, small in auto mode); clone() then
// copies on the host as before.
inline std::optional<array> device_clone_(const array& a) {
  return graph::gpu_copy_nd_(a);
}
}  // namespace detail

inline bool array_equal(const array& a, const array& b) {
  return allclose(a, b, 0.0f, 0.0f);
}

// The auto-mode matmul threshold in force (M*N*K): derived on this host by
// the first auto-mode eval, pinned by TL_AUTO_MATMUL, or types.h's census
// value until either happens.
inline int64_t auto_matmul_threshold() {
  return detail::graph::auto_matmul_threshold_();
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
