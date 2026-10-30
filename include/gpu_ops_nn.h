#pragma once

// The nn layer of the GPU ops (layers: gpu_abi.h): the fused ops over plain
// tensors that any network uses -- norms and activations, RoPE, attention, the
// GEMVs and GEMMs over bf16 / f16 / block-quantized weights, Gated DeltaNet,
// cross-entropy and Adam. A backend may have a kernel for each; where it has
// none, the op takes its composition of the core ops (gpu::generic, below) or
// answers false and the caller falls back to the CPU.
//
// Included by gpu.h, after gpu_ops.h.

#include <cmath>
#include <cstdint>

#include "gpu_abi_nn.h"
#include "gpu_ops.h"

namespace tl {
namespace gpu {

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

// ---- the nn ops by composition. Each fused op below is a kernel a backend may
// have; where it does not — the shared launch declines, or `own` has no such
// member — the same result comes from the core ops (gpu_ops.h), written once here.
// Several launches and a scratch buffer or two rather than one kernel, so a
// backend that cares for the decode loop writes the kernel; a new backend has
// the whole model path from its core ops alone. f32 only: a bf16 or int4 operand
// needs a reader the core layer does not have, so those ops still decline. Nothing
// here names a backend or the array layer: spans, core ops, and the device
// core's alloc / release / cpu_barrier / sync_to_host.

namespace generic {

inline bool rmsnorm(span x, span w, span o, int64_t n, float eps,
                    int64_t rows) {
  scratch sq(rows * n), ms(rows), r(rows);
  if (!sq || !ms || !r) return false;
  // mean(x^2) + eps in the reduction's epilogue, then its inverse root, then
  // two broadcasts: by the row's scale and by w. (No launch here writes a
  // buffer it reads — a backend may forbid that — and o, which may alias x,
  // is written last, after x's last read.)
  return binary(kop::mul, x, x, sq, rows * n, 1.0f, 0.0f) &&
         row_op(kop::row_sum, sq, ms, rows, n, 1.0f / static_cast<float>(n), eps) &&
         scalar_binary(scalar_op::pow, ms, r, rows, -0.5f, 1.0f, 0.0f) &&
         binary_bcast(kop::bmul, x, n, 1, r, 1, 0, sq, rows, n, 1.0f, 0.0f) &&
         binary_bcast(kop::bmul, sq, n, 1, w, 0, 1, o, rows, n, 1.0f, 0.0f);
}

inline bool rmsnorm_res(span x, span delta, span w, span xout, span hout,
                        int64_t n, float eps, int64_t rows) {
  if (xout.buf != x.buf) {
    return binary(kop::add, x, delta, xout, rows * n, 1.0f, 0.0f) &&
           rmsnorm(xout, w, hout, n, eps, rows);
  }
  // xout aliases x: the sum goes through a buffer of this layer's own.
  scratch sum(rows * n);
  return sum && binary(kop::add, x, delta, sum, rows * n, 1.0f, 0.0f) &&
         unary(kop::affine, sum, xout, rows * n, 1.0f, 0.0f) &&
         rmsnorm(sum, w, hout, n, eps, rows);
}

inline bool swiglu(span gu, span o, int64_t ff, int64_t rows) {
  scratch zero(1, true), g(rows * ff), s(rows * ff), silu(rows * ff);
  if (!zero || !g || !s || !silu) return false;
  zero.contents[0] = 0.0f;
  // gate out of the fused [rows, 2ff] buffer, silu(gate) = gate * sigmoid(gate),
  // times up read in place.
  return copy_2d(gu, 2 * ff, 1, zero, g, rows, ff) &&
         unary(kop::sigmoid, g, s, rows * ff, 1.0f, 0.0f) &&
         binary(kop::mul, g, s, silu, rows * ff, 1.0f, 0.0f) &&
         binary_bcast(kop::bmul, silu, ff, 1, gu.at(ff * 4), 2 * ff, 1, o, rows,
                      ff, 1.0f, 0.0f);
}

// A GEMV is a GEMM with one row.
inline bool gemv_f32(span a, span B, span y, int64_t n, int64_t k) {
  return gemm(a, k, false, B, n, false, y, 1, n, k, 1.0f, 0.0f);
}

// [H, T, D] -> [T, H*D]: token t's row is the [H, D] gather of the heads'
// row t, one strided copy a token.
inline bool merge_heads(span src, span dst, int64_t T, int64_t H, int64_t D) {
  scratch zero(1, true);
  if (!zero) return false;
  zero.contents[0] = 0.0f;
  for (int64_t t = 0; t < T; t++) {
    if (!copy_2d(src.at(t * D * 4), T * D, 1, zero, dst.at(t * H * D * 4), H, D)) {
      return false;
    }
  }
  return true;
}

// [T, ld] -> [H, T, D] from column block `off`, plus the head's bias row: one
// broadcast add a head (of zero, when there is no bias).
inline bool split_heads(span src, span bias, span dst, int64_t T, int64_t ld,
                        int64_t off, int64_t H, int64_t D) {
  scratch zero(1, true);
  if (!zero) return false;
  zero.contents[0] = 0.0f;
  for (int64_t h = 0; h < H; h++) {
    const span b = bias ? bias.at(h * D * 4) : span(zero);
    if (!binary_bcast(kop::badd, src.at((off + h * D) * 4), ld, 1, b, 0,
                      bias ? 1 : 0, dst.at(h * T * D * 4), T, D, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// Rotary embedding over [rows, D], rows = H*T, row r at position pos + r % T,
// rotating the first `rot` dims of each row (all D when rot == D) and passing
// the rest through: out = x * C + swap(x) * S, with C and S the [T, D] tables
// of cos and of (-sin | +sin) over the two rotated halves (1 and 0 past rot),
// and swap(x) the halves exchanged — a row gather of x seen as rows of rot/2,
// which leaves the chunks past rot in place. Four launches a head over shared
// tables.
inline bool rope(span x, span o, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base, span bias, int64_t rot) {
  if (rows <= 0 || T <= 0 || D <= 0 || rot <= 0 || rot % 2 || rot > D ||
      rows % T || D % (rot / 2)) {
    return false;
  }
  const int64_t half = rot / 2, chunks = D / half, H = rows / T;
  scratch C(T * D, true), S(T * D, true), idx(T * chunks, true),
      xb(bias ? rows * D : 1), sw(T * D), xc(T * D), xs(T * D);
  if (!C || !S || !idx || (bias && !xb) || !sw || !xc || !xs) return false;
  for (int64_t t = 0; t < T; t++) {
    const double position = static_cast<double>(pos + t);
    for (int64_t j = 0; j < half; j++) {
      const double ang = position * std::pow(static_cast<double>(base),
                                             -2.0 * static_cast<double>(j) / rot);
      const float c = static_cast<float>(std::cos(ang)),
                  s = static_cast<float>(std::sin(ang));
      C.contents[t * D + j] = C.contents[t * D + j + half] = c;
      S.contents[t * D + j] = -s;
      S.contents[t * D + j + half] = s;
    }
    for (int64_t j = rot; j < D; j++) {
      C.contents[t * D + j] = 1.0f;
      S.contents[t * D + j] = 0.0f;
    }
    for (int64_t c = 0; c < chunks; c++) {  // swap the two rotated chunks only
      const int64_t from = c < 2 ? 1 - c : c;
      idx.contents[t * chunks + c] = static_cast<float>(t * chunks + from);
    }
  }
  span src = x;
  if (bias) {
    if (!binary(kop::add, x, bias, xb, rows * D, 1.0f, 0.0f)) return false;
    src = xb;
  }
  for (int64_t h = 0; h < H; h++) {
    const span xh = src.at(h * T * D * 4), oh = o.at(h * T * D * 4);
    if (!index_select(xh, idx, sw, half, T * chunks) ||
        !binary(kop::mul, xh, C, xc, T * D, 1.0f, 0.0f) ||
        !binary(kop::mul, sw, S, xs, T * D, 1.0f, 0.0f) ||
        !binary(kop::add, xc, xs, oh, T * D, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// One decode step, a head at a time: scores = q . K^T over the cached
// prefix, softmax, times V. An f32 cache only.
inline bool attn_decode(span q, span K, span V, span o, int64_t n_q_heads,
                        int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                        int64_t D, float scale, bool kv_f16) {
  if (kv_f16 || n_kv_heads <= 0 || n_q_heads % n_kv_heads || ctx <= 0 ||
      D <= 0) {
    return false;
  }
  scratch s(ctx), p(ctx);
  if (!s || !p) return false;
  const int64_t group = n_q_heads / n_kv_heads;
  for (int64_t h = 0; h < n_q_heads; h++) {
    const int64_t kv = (h / group) * kv_max * D * 4;
    if (!gemm(q.at(h * D * 4), D, false, K.at(kv), D, true, s, 1, ctx, D, scale,
              0.0f) ||
        !row_op(kop::softmax, s, p, 1, ctx, 1.0f, 0.0f) ||
        !gemm(p, ctx, false, V.at(kv), D, false, o.at(h * D * 4), 1, D, ctx,
              1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// The causal prefill, a head at a time: scores [T, ctx] over the cache rows
// [0, pos0 + T), a mask that closes the keys past each query's position,
// softmax by row, times V.
inline bool attn_prefill(span q, span K, span V, span o, int64_t n_q_heads,
                         int64_t n_kv_heads, int64_t T, int64_t kv_max,
                         int64_t D, float scale, bool kv_f16, int64_t pos0) {
  if (kv_f16 || n_kv_heads <= 0 || n_q_heads % n_kv_heads || T <= 0 ||
      D <= 0 || pos0 < 0 || pos0 + T > kv_max) {
    return false;
  }
  const int64_t ctx = pos0 + T;
  scratch mask(T * ctx, true), s(T * ctx), p(T * ctx);
  if (!mask || !s || !p) return false;
  for (int64_t t = 0; t < T; t++) {
    for (int64_t j = 0; j < ctx; j++) {
      mask.contents[t * ctx + j] = j <= pos0 + t ? 0.0f : -1e30f;
    }
  }
  const int64_t group = n_q_heads / n_kv_heads;
  for (int64_t h = 0; h < n_q_heads; h++) {
    const int64_t kv = (h / group) * kv_max * D * 4, qh = h * T * D * 4;
    if (!gemm(q.at(qh), D, false, K.at(kv), D, true, s, T, ctx, D, scale, 0.0f) ||
        !binary(kop::add, s, mask, p, T * ctx, 1.0f, 0.0f) ||
        !row_op(kop::softmax, p, s, T, ctx, 1.0f, 0.0f) ||
        !gemm(s, ctx, false, V.at(kv), D, false, o.at(qh), T, D, ctx, 1.0f,
              0.0f)) {
      return false;
    }
  }
  return true;
}

// Softmax cross-entropy's pullback: exp(x - lse) row-broadcast, a one-hot
// scatter of the targets subtracted, g row-broadcast into the result.
inline bool xent_bwd(span x, span lse, span tgt, span g, span o, int64_t rows,
                     int64_t cols) {
  const int64_t n = rows * cols;
  scratch diff(n), p(n), onehot(n), ones(rows, true);
  if (!diff || !p || !onehot || !ones) return false;
  for (int64_t i = 0; i < rows; i++) ones.contents[i] = 1.0f;
  return binary_bcast(kop::bsub, x, cols, 1, lse, 1, 0, diff, rows, cols, 1.0f,
                      0.0f) &&
         unary(kop::exp_, diff, p, n, 1.0f, 0.0f) &&
         scatter_to_axis(tgt, ones, onehot, rows, cols) &&
         binary(kop::sub, p, onehot, diff, n, 1.0f, 0.0f) &&
         binary_bcast(kop::bmul, diff, cols, 1, g, 1, 0, o, rows, cols, 1.0f,
                      0.0f);
}

// Adam's update: m and v each a sum of two affines of their old value and g,
// p moved by their ratio through a scratch of its own (p is read only before
// it is ever the write side, so the update never aliases a launch's input).
inline bool adam_step(span p, span m, span v, span g, int64_t n, float beta1,
                      float beta2, float eps, float lr_over_bc1,
                      float inv_bc2) {
  scratch a(n), b(n), c(n);
  if (!a || !b || !c) return false;
  return unary(kop::affine, m, a, n, beta1, 0.0f) &&
         unary(kop::affine, g, b, n, 1.0f - beta1, 0.0f) &&
         binary(kop::add, a, b, m, n, 1.0f, 0.0f) &&
         binary(kop::mul, g, g, c, n, 1.0f - beta2, 0.0f) &&
         unary(kop::affine, v, b, n, beta2, 0.0f) &&
         binary(kop::add, b, c, v, n, 1.0f, 0.0f) &&
         unary(kop::affine, v, a, n, inv_bc2, 0.0f) &&
         unary(kop::sqrt_, a, c, n, 1.0f, eps) &&
         unary(kop::affine, m, a, n, lr_over_bc1, 0.0f) &&
         binary(kop::div, a, c, b, n, 1.0f, 0.0f) &&
         binary(kop::sub, p, b, a, n, 1.0f, 0.0f) &&
         unary(kop::affine, a, p, n, 1.0f, 0.0f);
}

// Gated DeltaNet's causal conv, a token at a time: win = [state rows, x_t]
// ([K, C], oldest first), o_t = silu(column sums of win * w), and state takes
// win's last K-1 rows. A ones-row GEMM does the column sums.
inline bool ssm_conv(span x, int64_t ld, span state, span w, span o, int64_t C,
                     int64_t K, int64_t T) {
  scratch win(K * C), prod(K * C), ones(K, true), sum(C), sg(C);
  if (!win || !prod || !ones || !sum || !sg) return false;
  for (int64_t k = 0; k < K; k++) ones.contents[k] = 1.0f;
  const span wv = win;
  for (int64_t t = 0; t < T; t++) {
    if (!unary(kop::affine, state, wv, (K - 1) * C, 1.0f, 0.0f) ||
        !unary(kop::affine, x.at(t * ld * 4), wv.at((K - 1) * C * 4), C, 1.0f, 0.0f) ||
        !binary(kop::mul, wv, w, prod, K * C, 1.0f, 0.0f) ||
        !gemm(ones, K, false, prod, C, false, sum, 1, C, K, 1.0f, 0.0f) ||
        !unary(kop::sigmoid, sum, sg, C, 1.0f, 0.0f) ||
        !binary(kop::mul, sum, sg, o.at(t * C * 4), C, 1.0f, 0.0f) ||
        !unary(kop::affine, wv.at(C * 4), state, (K - 1) * C, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// One token of the gated delta rule (see the shared op below): the L2 norms
// as a row reduction and two broadcasts, the gates elementwise, and per head
// three GEMVs (the state read by k, the rank-1 update, the state read by q).
// Hk q / k heads serve the H value heads, head h reading key head h % Hk.
inline bool gated_delta_token_(span qkv, span ab, span adt, span state, span o,
                               int64_t H, int64_t Hk, int64_t dk, int64_t dv, float eps) {
  const int64_t nq = Hk * dk, S = dk * dv;
  scratch sq(2 * nq), ss(2 * Hk), inv(2 * Hk), qk(2 * nq), t(H), e(H), e1(H),
      sp(H), g(H), decay(H), beta(H), S2(H * S), kvm(H * dv), d1(H * dv),
      d2(H * dv), up(S);
  if (!sq || !ss || !inv || !qk || !t || !e || !e1 || !sp || !g || !decay ||
      !beta || !S2 || !kvm || !d1 || !d2 || !up) {
    return false;
  }
  const span qn = qk, kn = qn.at(nq * 4);
  // q and k rows over 1 / sqrt(sum(x^2) + eps), q also over sqrt(dk).
  if (!binary(kop::mul, qkv, qkv, sq, 2 * nq, 1.0f, 0.0f) ||
      !row_op(kop::row_sum, sq, ss, 2 * Hk, dk, 1.0f, eps) ||
      !scalar_binary(scalar_op::pow, ss, inv, 2 * Hk, -0.5f, 1.0f, 0.0f) ||
      !binary_bcast(kop::bmul, qkv, dk, 1, inv, 1, 0, qn, Hk, dk,
                    1.0f / std::sqrt(static_cast<float>(dk)), 0.0f) ||
      !binary_bcast(kop::bmul, qkv.at(nq * 4), dk, 1, span(inv).at(Hk * 4), 1,
                    0, kn, Hk, dk, 1.0f, 0.0f)) {
    return false;
  }
  // decay = exp(A * softplus(alpha + dt)), beta = sigmoid(beta).
  if (!binary(kop::add, ab, adt.at(H * 4), t, H, 1.0f, 0.0f) ||
      !unary(kop::exp_, t, e, H, 1.0f, 0.0f) ||
      !unary(kop::affine, e, e1, H, 1.0f, 1.0f) ||
      !unary(kop::log_, e1, sp, H, 1.0f, 0.0f) ||
      !binary(kop::mul, adt, sp, g, H, 1.0f, 0.0f) ||
      !unary(kop::exp_, g, decay, H, 1.0f, 0.0f) ||
      !unary(kop::sigmoid, ab.at(H * 4), beta, H, 1.0f, 0.0f) ||
      !binary_bcast(kop::bmul, state, S, 1, decay, 1, 0, S2, H, S, 1.0f, 0.0f)) {
    return false;
  }
  const span S2v = S2;
  for (int64_t h = 0; h < H; h++) {  // kv memory = k . S
    if (!gemm(kn.at((h % Hk) * dk * 4), dk, false, S2v.at(h * S * 4), dv, false,
              span(kvm).at(h * dv * 4), 1, dv, dk, 1.0f, 0.0f)) {
      return false;
    }
  }
  if (!binary(kop::sub, qkv.at(2 * nq * 4), kvm, d1, H * dv, 1.0f, 0.0f) ||
      !binary_bcast(kop::bmul, d1, dv, 1, beta, 1, 0, d2, H, dv, 1.0f, 0.0f)) {
    return false;
  }
  for (int64_t h = 0; h < H; h++) {  // S += k (x) delta, then o = q . S
    const span Sh = state.at(h * S * 4);
    if (!gemm(kn.at((h % Hk) * dk * 4), 1, false, span(d2).at(h * dv * 4), dv, false,
              up, dk, dv, 1, 1.0f, 0.0f) ||
        !binary(kop::add, S2v.at(h * S * 4), up, Sh, S, 1.0f, 0.0f) ||
        !gemm(qn.at((h % Hk) * dk * 4), dk, false, Sh, dv, false, o.at(h * dv * 4), 1,
              dv, dk, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// The rule over T tokens is the one-token step T times, in order.
inline bool gated_delta(span qkv, span ab, int64_t ab_ld, span adt, span state,
                        span o, int64_t H, int64_t dk, int64_t dv, int64_t T,
                        float eps, int64_t Hk = 0) {
  if (Hk == 0) Hk = H;
  const int64_t C = 2 * Hk * dk + H * dv;
  for (int64_t t = 0; t < T; t++) {
    if (!gated_delta_token_(qkv.at(t * C * 4), ab.at(t * ab_ld * 4), adt, state,
                            o.at(t * H * dv * 4), H, Hk, dk, dv, eps)) {
      return false;
    }
  }
  return true;
}

}  // namespace generic

// ---- the model path: what a decoder runs on raw device buffers between its
// GEMVs and attention. Each is the backend's kernel where it has one and the
// generic composition where it does not, so every backend has the f32 model
// path; an op still answers false for an operand its backend cannot read (a
// f16 cache, bf16 weight, int4 weights), and a model keeps to the array ops
// there.

// out = x * rsqrt(mean(x^2) + eps) * w per row of [rows, n]. out may alias x.
inline bool rmsnorm(span x, span w, span o, int64_t n, float eps,
                    int64_t rows = 1) {
  if (n <= 0 || rows <= 0) return false;
  return launch(kop::rmsnorm_, {in(x), in(w), out(o)},
                rmsnorm_params{static_cast<uint32_t>(n), eps},
                policy::one_group_per_row(rows)) ||
         generic::rmsnorm(x, w, o, n, eps, rows);
}

// xout = x + delta and hout = rmsnorm(xout) * w: a residual add folded into
// the norm that follows it. xout may alias x.
inline bool rmsnorm_res(span x, span delta, span w, span xout, span hout,
                        int64_t n, float eps, int64_t rows = 1) {
  if (n <= 0 || rows <= 0) return false;
  return launch(kop::add_rmsnorm_,
                {in(x), in(delta), in(w), out(xout), out(hout)},
                rmsnorm_params{static_cast<uint32_t>(n), eps},
                policy::one_group_per_row(rows)) ||
         generic::rmsnorm_res(x, delta, w, xout, hout, n, eps, rows);
}

// out[rows, ff] = silu(gate) * up out of the fused gate|up buffer [rows, 2ff].
inline bool swiglu(span gu, span o, int64_t ff, int64_t rows = 1) {
  if (ff <= 0 || rows <= 0) return false;
  return launch(kop::swiglu_, {in(gu), out(o)},
                swiglu_params{static_cast<uint32_t>(ff)},
                policy::flat_rows(ff, rows)) ||
         generic::swiglu(gu, o, ff, rows);
}

// Gated DeltaNet's causal depthwise conv over T tokens, with its SiLU. Token t
// reads row t of x (rows `ld` apart, its first C channels) and writes row t of
// o [T, C]: over the window of the K-1 inputs before it and itself,
// o_t[c] = silu(sum_k win[k][c] * w[k][c]) for the tap-major weights w [K, C].
// state holds the K-1 inputs before the first token [K-1, C], oldest first,
// and is left holding the K-1 before the next call. The kernels keep the
// window in registers, to K = 8.
TL_GPU_DETECT_OWN(ssm_conv_tokens)
template <class Own = own>
inline bool ssm_conv(span x, int64_t ld, span state, span w, span o, int64_t C,
                     int64_t K, int64_t T = 1) {
  if (C <= 0 || K < 2 || T <= 0 || ld < C) return false;
  // A prefill's many tokens: a backend's token-parallel form, where it has one.
  if constexpr (detail::owns_ssm_conv_tokens<Own>::value) {
    if (T > 1 && detail::ran(Own::ssm_conv_tokens(x, ld, state, w, o, C, K, T))) return true;
  }
  return (K <= 8 &&
          launch(kop::ssm_conv_, {in(x), inout(state), in(w), out(o)},
                 ssm_conv_params{static_cast<uint32_t>(C), static_cast<uint32_t>(K),
                                 static_cast<uint32_t>(T), static_cast<uint32_t>(ld)},
                 policy::flat(C))) ||
         generic::ssm_conv(x, ld, state, w, o, C, K, T);
}

// The gated delta rule (Gated DeltaNet, Qwen3.5) over T tokens in order, per
// head h with a state S_h [dk, dv]. Token t reads row t of qkv [T, C] —
// q [Hk, dk] | k [Hk, dk] | v [H, dv] as the conv left them (Hk = 0 means H;
// fewer key heads than value heads are shared, head h reading key head h % Hk,
// the order GGUF files store the value heads in) — and of ab (rows
// ab_ld apart) — alpha [H] | beta [H] raw; adt = A [H] | dt_bias [H]. q and k
// are L2-normalized (q also over sqrt(dk)), then
//   decay = exp(A * softplus(alpha + dt)), beta = sigmoid(beta),
//   S = S * decay;  S += k (x) ((v - k.S) * beta);  o_t = q.S
// into row t of o [T, H*dv]. State and output are f32.
TL_GPU_DETECT_OWN(gated_delta_chunked)
template <class Own = own>
inline bool gated_delta(span qkv, span ab, int64_t ab_ld, span adt, span state,
                        span o, int64_t H, int64_t dk, int64_t dv, int64_t T,
                        float eps, int64_t Hk = 0) {
  if (Hk == 0) Hk = H;
  if (H <= 0 || Hk <= 0 || H % Hk || dk <= 0 || dv <= 0 || T <= 0) return false;
  // A prefill's many tokens: a backend's chunked form, where it has one for
  // the shape, regroups the same rule into per-chunk matrix products.
  if constexpr (detail::owns_gated_delta_chunked<Own>::value) {
    if (T > 1 && detail::ran(Own::gated_delta_chunked(qkv, ab, ab_ld, adt, state, o, H,
                                                      dk, dv, T, eps, Hk))) {
      return true;
    }
  }
  // The kernels run a block of dk threads over 32 columns of a head's state,
  // each warp holding 32 of its rows: dk a power of two a group can be.
  const bool shape = dk >= 32 && dk <= policy::max_group_threads && (dk & (dk - 1)) == 0;
  const grid g{static_cast<uint32_t>((dv + 31) / 32), static_cast<uint32_t>(H), 1,
               static_cast<uint32_t>(dk), 1, 1,
               static_cast<uint32_t>(6 * dk * sizeof(float))};
  return (shape && launch(kop::gated_delta_,
                {in(qkv), in(ab), in(adt), inout(state), out(o)},
                gated_delta_params{static_cast<uint32_t>(H), static_cast<uint32_t>(dk),
                                   static_cast<uint32_t>(dv), static_cast<uint32_t>(T),
                                   static_cast<uint32_t>(ab_ld), eps,
                                   static_cast<uint32_t>(Hk)},
                g)) ||
         generic::gated_delta(qkv, ab, ab_ld, adt, state, o, H, dk, dv, T, eps, Hk);
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

// Head-major [H, T, D] -> token-major [T, H*D]: split_heads' inverse.
inline bool merge_heads(span src, span dst, int64_t T, int64_t H, int64_t D) {
  if (T <= 0 || H <= 0 || D <= 0) return false;
  return launch(kop::merge_heads_, {in(src), out(dst)},
                merge_heads_params{static_cast<uint32_t>(T),
                                   static_cast<uint32_t>(H),
                                   static_cast<uint32_t>(D)},
                policy::per_head(H, T, D)) ||
         generic::merge_heads(src, dst, T, H, D);
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
    if (detail::ran(Own::split_heads(src, bias, dst, T, ld, off, H, D))) {
      return true;
    }
  }
  return generic::split_heads(src, bias, dst, T, ld, off, H, D);
}

// Rotary embedding over [rows, T, D] at position `pos`, adding the optional
// per-row bias first. `rot` (0: all of D) rotates only the first rot dims of
// each row, the partial RoPE of Qwen3.5: a backend's own `rope` turns the
// whole row, and one that also has `rope_partial` takes the rest.
TL_GPU_DETECT_OWN(rope)
TL_GPU_DETECT_OWN(rope_partial)
template <class Own = own>
inline bool rope(span x, span o, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base, span bias = {}, int64_t rot = 0) {
  if (rot == 0) rot = D;
  if constexpr (detail::owns_rope<Own>::value) {
    if (rot == D &&
        detail::ran(Own::rope(x, o, rows, T, D, pos, base, bias))) {
      return true;
    }
  }
  if constexpr (detail::owns_rope_partial<Own>::value) {
    if (rot != D &&
        detail::ran(Own::rope_partial(x, o, rows, T, D, pos, base, bias, rot))) {
      return true;
    }
  }
  return generic::rope(x, o, rows, T, D, pos, base, bias, rot);
}

// ---- the GEMVs and GEMMs over bf16, f16 and block-quantized weights, the
// decode-path fusions, and attention.

// y[1,n] = a[1,k] . B[k,n], the weight column-major, f32 or bf16.
TL_GPU_DETECT_OWN(gemv_f32)
template <class Own = own>
inline bool gemv_f32(span a, span B, span y, int64_t n, int64_t k) {
  if constexpr (detail::owns_gemv_f32<Own>::value) {
    if (detail::ran(Own::gemv_f32(a, B, y, n, k))) return true;
  }
  return generic::gemv_f32(a, B, y, n, k);
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

// The same product summed in f16, for an inference prefill that takes the
// trade: f16 sums run at twice the f32 rate on tensor cores that have them
// (CUDA's gemm_bf16_nt_f16acc), at a small cost in precision and with f16's
// range — an activation or partial sum past 65504 turns to inf, which the
// caller must check for. A backend without the form sums in f32.
TL_GPU_DETECT_OWN(gemm_bf16_nt_f16acc)
template <class Own = own>
inline bool gemm_bf16_nt_f16acc(span A, span B, span C, int64_t M, int64_t N,
                                int64_t K) {
  if constexpr (detail::owns_gemm_bf16_nt_f16acc<Own>::value) {
    if (detail::ran(Own::gemm_bf16_nt_f16acc(A, B, C, M, N, K))) return true;
  }
  return gemm_bf16_nt<Own>(A, B, C, M, N, K);
}

// The prefill's GEMM over block-quantized rows (see own::gemm_quant in cuda.h): C columns
// [0, n) of rows ldc apart, A f16 (a16) or f32. mmq: Q4_K and Q8_0 on the int8 tensor cores,
// A rounded to 8 bits. False where the backend has none.
TL_GPU_DETECT_OWN(gemm_quant)
template <class Own = own>
inline bool gemm_quant(int type, bool a16, span A, span W, span C, int64_t m, int64_t n,
                        int64_t k, int64_t ldc, bool mmq = false) {
  if constexpr (detail::owns_gemm_quant<Own>::value) {
    return detail::ran(Own::gemm_quant(type, a16, A, W, C, m, n, k, ldc, mmq));
  } else {
    return false;
  }
}

// silu(gate) * up of a fused gate|up block-quantized projection in one MMQ GEMM (see
// own::gemm_quant_glu in cuda.h). False where the backend has none or it does not apply.
TL_GPU_DETECT_OWN(gemm_quant_glu)
template <class Own = own>
inline bool gemm_quant_glu(int type, span A16, span W, span H16, int64_t m, int64_t ff, int64_t k) {
  if constexpr (detail::owns_gemm_quant_glu<Own>::value) {
    return detail::ran(Own::gemm_quant_glu(type, A16, W, H16, m, ff, k));
  } else {
    return false;
  }
}

// Qwen3.5's gated norm, silu(z) * rmsnorm(o) * w, in one pass (see own::gated_rmsnorm in
// cuda.h). False where the backend has none: the caller composes it.
TL_GPU_DETECT_OWN(gated_rmsnorm)
template <class Own = own>
inline bool gated_rmsnorm(span o, span z, int64_t ldz, span w, span out, bool h16, int64_t T,
                          int64_t H, int64_t D, float eps) {
  if constexpr (detail::owns_gated_rmsnorm<Own>::value) {
    return detail::ran(Own::gated_rmsnorm(o, z, ldz, w, out, h16, T, H, D, eps));
  } else {
    return false;
  }
}

// The activation quantized to int8 blocks (see own::quantize_act_q8 in cuda.h) and
// the Q4_K GEMV that reads it; false where the backend has no such pair.
TL_GPU_DETECT_OWN(quantize_act_q8)
TL_GPU_DETECT_OWN(gemv_q4k_q8)
template <class Own = own>
inline bool quantize_act_q8(span a, span qa, int64_t k) {
  if constexpr (detail::owns_quantize_act_q8<Own>::value) {
    return detail::ran(Own::quantize_act_q8(a, qa, k));
  } else {
    return false;
  }
}
template <class Own = own>
inline bool gemv_q4k_q8(span qa, span W, span y, int64_t n, int64_t k) {
  if constexpr (detail::owns_gemv_q4k_q8<Own>::value) {
    return detail::ran(Own::gemv_q4k_q8(qa, W, y, n, k));
  } else {
    return false;
  }
}
inline constexpr bool int8_activations = has_gemv_q4k_q8 && has_quantize_act_q8;

// y[n] = W[n,K] . a with W the rows of a GGUF block-quantized weight (type = ggml_type
// code: 12 Q4_K, 14 Q6_K, 8 Q8_0), read as quantized. False where the backend has no
// such kernel; the caller dequantizes instead.
TL_GPU_DETECT_OWN(gemv_quant)
template <class Own = own>
inline bool gemv_quant(int type, span a, span W, span y, int64_t n, int64_t k) {
  if constexpr (detail::owns_gemv_quant<Own>::value) {
    return detail::ran(Own::gemv_quant(type, a, W, y, n, k));
  } else {
    return false;
  }
}

// x[rows, D] = rope(rmsnorm(x) * w), in place, rows = heads * T: Qwen3's q and k
// rows. One pass where the backend has it, else the two ops.
TL_GPU_DETECT_OWN(rmsnorm_rope)
template <class Own = own>
inline bool rmsnorm_rope(span x, span w, int64_t rows, int64_t T, int64_t D, int64_t pos,
                         float base, int64_t rot, float eps) {
  if constexpr (detail::owns_rmsnorm_rope<Own>::value) {
    if (detail::ran(Own::rmsnorm_rope(x, w, rows, T, D, pos, base, rot, eps))) return true;
  }
  return rmsnorm(x, w, x, D, eps, rows) && rope(x, x, rows, T, D, pos, base, {}, rot);
}

// The f16 prefill's activations, rounded where they are made so the GEMM does
// not round them again in every block that reads them (CUDA only: a backend
// without them answers false, and `f16_activations` says which). The norms
// write the f16 copy beside the f32 output; swiglu and merge_heads write only
// f16, their f32 form being read by nothing else. Spans of an f16 buffer are
// byte offsets like any other; the buffer is sized in floats, half as many.
TL_GPU_DETECT_OWN(gemm_bf16_nt_f16acc_a16)
TL_GPU_DETECT_OWN(rmsnorm_h16)
TL_GPU_DETECT_OWN(rmsnorm_res_h16)
TL_GPU_DETECT_OWN(swiglu_h16)
TL_GPU_DETECT_OWN(merge_heads_h16)
TL_GPU_DETECT_OWN(cvt_h16)
inline constexpr bool f16_activations = has_gemm_bf16_nt_f16acc_a16;
#define TL_GPU_F16_OP(NAME, SIG, CALL)                                  \
  template <class Own = own>                                            \
  inline bool NAME SIG {                                                \
    if constexpr (detail::owns_##NAME<Own>::value) {                    \
      return detail::ran(Own::NAME CALL);                               \
    } else {                                                            \
      return false;                                                     \
    }                                                                   \
  }
TL_GPU_F16_OP(gemm_bf16_nt_f16acc_a16, (span A, span B, span C, int64_t M, int64_t N, int64_t K),
              (A, B, C, M, N, K))
TL_GPU_F16_OP(rmsnorm_h16, (span x, span w, span o, span o16, int64_t n, float eps, int64_t rows),
              (x, w, o, o16, n, eps, rows))
TL_GPU_F16_OP(rmsnorm_res_h16,
              (span x, span delta, span w, span xout, span hout, span hout16, int64_t n, float eps,
               int64_t rows),
              (x, delta, w, xout, hout, hout16, n, eps, rows))
TL_GPU_F16_OP(swiglu_h16, (span gu, span o16, int64_t ff, int64_t rows), (gu, o16, ff, rows))
TL_GPU_F16_OP(merge_heads_h16, (span src, span dst16, int64_t T, int64_t H, int64_t D),
              (src, dst16, T, H, D))
TL_GPU_F16_OP(cvt_h16, (span src, span dst16, int64_t n), (src, dst16, n))
#undef TL_GPU_F16_OP

// One decode step: q [n_q_heads, D] against a [n_kv_heads, kv_max, D] cache
// read over [0, ctx). GQA by the head ratio; the cache is f32, or f16 with kv_f16.
TL_GPU_DETECT_OWN(attn_decode)
template <class Own = own>
inline bool attn_decode(span q, span K, span V, span o, int64_t n_q_heads,
                        int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                        int64_t D, float scale, bool kv_f16 = false) {
  if constexpr (detail::owns_attn_decode<Own>::value) {
    if (detail::ran(Own::attn_decode(q, K, V, o, n_q_heads, n_kv_heads, ctx, kv_max, D, scale, kv_f16))) {
      return true;
    }
  }
  return generic::attn_decode(q, K, V, o, n_q_heads, n_kv_heads, ctx, kv_max, D,
                              scale, kv_f16);
}

// Causal prefill: q, out [n_q_heads, T, D] against the cache rows
// [0, pos0 + T); query t sits at absolute position pos0 + t.
TL_GPU_DETECT_OWN(attn_prefill)
template <class Own = own>
inline bool attn_prefill(span q, span K, span V, span o, int64_t n_q_heads,
                         int64_t n_kv_heads, int64_t T, int64_t kv_max,
                         int64_t D, float scale, bool kv_f16 = false,
                         int64_t pos0 = 0) {
  if constexpr (detail::owns_attn_prefill<Own>::value) {
    if (detail::ran(Own::attn_prefill(q, K, V, o, n_q_heads, n_kv_heads, T, kv_max, D, scale, kv_f16, pos0))) {
      return true;
    }
  }
  return generic::attn_prefill(q, K, V, o, n_q_heads, n_kv_heads, T, kv_max, D,
                               scale, kv_f16, pos0);
}

// The causal prefill for inference: a backend may round its operands to f16
// for tensor cores (CUDA's attn_prefill_f16); one with no such form runs the
// exact attn_prefill. Not for training, whose pullback recomputes the forward
// in f32 and would meet a different softmax.
TL_GPU_DETECT_OWN(attn_prefill_f16)
template <class Own = own>
inline bool attn_prefill_f16(span q, span K, span V, span o, int64_t n_q_heads,
                             int64_t n_kv_heads, int64_t T, int64_t kv_max,
                             int64_t D, float scale, bool kv_f16 = false,
                             int64_t pos0 = 0) {
  if constexpr (detail::owns_attn_prefill_f16<Own>::value) {
    if (detail::ran(Own::attn_prefill_f16(q, K, V, o, n_q_heads, n_kv_heads, T, kv_max, D,
                                          scale, kv_f16, pos0))) {
      return true;
    }
  }
  return attn_prefill<Own>(q, K, V, o, n_q_heads, n_kv_heads, T, kv_max, D, scale,
                           kv_f16, pos0);
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

// ---- the training ops: eager, off the decode path, each the backend's fused
// kernel where it has one and the composition of core ops where it does not.

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
                policy::flat(n)) ||
         generic::xent_bwd(x, lse, tgt, g, o, rows, cols);
}

// Adam's update in place over n contiguous elements: m and v advance, p moves
// by the bias-corrected ratio the host folded into lr_over_bc1 and inv_bc2.
inline bool adam_step(span p, span m, span v, span g, int64_t n, float beta1,
                      float beta2, float eps, float lr_over_bc1, float inv_bc2) {
  if (n <= 0) return false;
  return launch(kop::adam_step_, {inout(p), inout(m), inout(v), in(g)},
                adam_params{beta1, beta2, eps, lr_over_bc1, inv_bc2,
                            static_cast<uint32_t>(n)},
                policy::flat(n)) ||
         generic::adam_step(p, m, v, g, n, beta1, beta2, eps, lr_over_bc1,
                            inv_bc2);
}

}  // namespace gpu
}  // namespace tl
