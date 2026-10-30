#pragma once

// The nn layer of the array API (layers: gpu_abi.h): the definitions behind
// the fused ops over plain tensors that array.h declares -- softmax and
// logsumexp, the norms and activations, RoPE, attention and its pullback,
// cross-entropy's pullback, Adam, and the bf16 / int4 decode GEMVs. array.h
// keeps the declarations, the node kinds and the evaluator's dispatch, and
// includes this file at its end.

namespace tl {

namespace ref {

inline array softmax(const array& a, int max_threads, bool own_exp) {
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

inline array logsumexp(const array& a, const shape_t& out_shape,
                       int max_threads, bool own_exp) {
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

inline array layer_norm(const array& x, const array& gamma, const array& beta,
                        float eps, int max_threads) {
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

}  // namespace ref

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

namespace detail {

inline auto graph::attn_decode(const array& q, const array& K, const array& V,
                               float scale) -> array {
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

inline auto graph::attn_prefill(const array& q, const array& K, const array& V,
                                float scale) -> array {
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

inline auto graph::rope(const array& x, int64_t pos, float base) -> array {
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

inline auto graph::layer_norm(const array& x, const array& gamma,
                              const array& beta, float eps) -> array {
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

inline auto graph::gemv_act_(const array& a_in, int64_t K,
                             int64_t N) -> std::optional<array> {
  array a = a_in.rank() == 1 ? a_in.reshape({1, a_in.size()}) : a_in;
  if (a.storage_.dt != tl::dtype::f32 || a.rank() != 2 || a.shape()[0] != 1)
    return std::nullopt;
  if (!gpu_mode_(N * (K > 0 ? K : 1), kernel_class::matmul))
    return std::nullopt;
  if (!a.contiguous() || a.offset_ != 0 || !a.storage_.native)
    return std::nullopt;
  return a;
}

inline auto graph::gpu_gemv(const node& n, const array& a_in,
                            const array& b) -> std::optional<array> {
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

inline auto graph::gpu_gemv_q4(const node& n, const array& a_in,
                               const array& Wq) -> std::optional<array> {
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

inline auto graph::attn_operands_ready_(const array& q, const array& K,
                                        const array& V) -> bool {
  return q.contiguous() && q.offset_ == 0 && K.contiguous() &&
         K.offset_ == 0 && V.contiguous() && V.offset_ == 0 &&
         q.storage_.native && K.storage_.native && V.storage_.native;
}

inline auto graph::ref_attn_row_(const float* q, const float* Kh, const float* Vh,
                                 int64_t klen, int64_t D, float scale, float* out,
                                 std::vector<float>& s) -> void {
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

inline auto graph::gpu_attn_(const node& n, const array& q,
                             const array& K, const array& V) -> std::optional<array> {
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

inline auto graph::ref_attn_(const array& q, const array& K, const array& V,
                             float scale) -> array {
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

inline auto graph::gpu_attn_prefill_(const node& n, const array& q,
                                     const array& K,
                                     const array& V) -> std::optional<array> {
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

inline auto graph::xent_bwd(const array& x, const array& lse,
                            const array& tgt, const array& g) -> std::optional<array> {
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

inline auto graph::cpu_xent_bwd_(const array& x, const array& lse, const array& tgt,
                                 const array& g) -> array {
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

inline auto graph::layer_norm_bwd(
                                  const array& x, const array& gamma, const array& dout, float eps) -> std::optional<std::array<array, 3>> {
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

inline auto graph::cpu_layer_norm_bwd_(const array& x,
                                       const array& gamma,
                                       const array& dout, float eps) -> std::array<array, 3> {
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

inline auto graph::adam_step(array& p, array& m, array& v, const array& g, float lr,
                             float beta1, float beta2, float eps, float bc1,
                             float bc2) -> bool {
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
    // gpu::adam_step already falls to its tier-0 composition when the
    // backend has no fused kernel, so this is the device itself declining
    // -- a scratch it could not allocate, or a tier-0 launch it refused:
    // the host loop below. The caller updates in place and cannot compose
    // its way out, so this has to be total -- and Metal's unified memory
    // makes the round trip a flush and a memcpy, the same bargain its
    // other CPU fallbacks make.
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

inline auto graph::attn_prefill_bwd_dq(
                                       const array& q, const array& K, const array& V, const array& dout,
                                       const array& out, float scale) -> std::optional<std::pair<array, array>> {
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

inline auto graph::attn_prefill_bwd_dkv(
                                        const array& q, const array& K, const array& V, const array& dout,
                                        const array& stats, float scale) -> std::optional<std::pair<array, array>> {
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

inline auto graph::cpu_attn_prefill_(const array& q, const array& K,
                                     const array& V, float scale) -> std::optional<array> {
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

inline auto graph::cpu_attn_bwd_dq_(const array& q, const array& K,
                                    const array& V, const array& dout,
                                    const array& out, float scale) -> std::pair<array, array> {
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

inline auto graph::cpu_attn_bwd_dkv_(const array& q, const array& K,
                                     const array& V, const array& dout,
                                     const array& stats, float scale) -> std::pair<array, array> {
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

inline auto graph::ref_attn_prefill_(const array& q, const array& K,
                                     const array& V, float scale) -> array {
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

inline auto graph::gpu_rope_(const node& n, const array& x) -> std::optional<array> {
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

inline auto graph::gpu_layer_norm_(const node& n, const array& x,
                                   const array& g, const array& b) -> std::optional<array> {
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

inline auto graph::ref_rope_(const array& x, int64_t pos, float base) -> array {
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

inline auto graph::gpu_row_logsumexp_(const array& a,
                                      const shape_t& out_shape,
                                      float scale, float offset) -> std::optional<array> {
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

}  // namespace detail

inline array array::softmax() const {
  if (rank() == 0) throw std::invalid_argument("tl::softmax: rank 0");
  return detail::graph::unary(detail::node::op_t::softmax, *this);
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

}  // namespace tl
