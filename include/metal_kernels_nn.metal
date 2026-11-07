// The nn layer's MSL kernels (layers: gpu_abi.h): softmax, layer norm and its
// pullback, RoPE, attention and its pullback, the GEMVs, the bf16 GEMM, the
// decode-path norms and activations, cross-entropy and Adam. #embed'd by metal.h
// after metal_kernels.metal, whose helpers these use: one source, compiled once.

kernel void softmax_(device const float* in [[buffer(0)]],
                     device float* out       [[buffer(1)]],
                     constant reduce_params& p [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[T];
  device const float* src = in + row * p.cols;
  device float* dst = out + row * p.cols;

  float m = -INFINITY;
  for (uint c = lid; c < p.cols; c += T) m = max(m, src[c]);
  scratch[lid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] = max(scratch[lid], scratch[lid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float sum = 0.0f;
  for (uint c = lid; c < p.cols; c += T) sum += exp(src[c] - row_max);
  scratch[lid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = 1.0f / scratch[0];
  for (uint c = lid; c < p.cols; c += T) dst[c] = exp(src[c] - row_max) * inv;
}

// Layer norm over the last axis with the affine epilogue: (x - mu) ·
// 1/sqrt(var + eps) · g + b. Two tree sums (x, then the squared deviations),
// each scaled by 1/cols after the tree as row_sum_'s mean is.
struct layer_norm_params {
  uint rows, cols;
  float eps, scale, offset;
};

kernel void layer_norm_(device const float* x [[buffer(0)]],
                        device const float* g [[buffer(1)]],
                        device const float* b [[buffer(2)]],
                        device float* out     [[buffer(3)]],
                        constant layer_norm_params& p [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[T];
  device const float* src = x + row * p.cols;
  device float* dst = out + row * p.cols;
  float inv_n = 1.0f / float(p.cols);

  float sum = 0.0f;
  for (uint c = lid; c < p.cols; c += T) sum += src[c];
  scratch[lid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mu = scratch[0] * inv_n;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float ss = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float v = src[c] - mu;
    ss += v * v;
  }
  scratch[lid] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = 1.0f / sqrt(scratch[0] * inv_n + p.eps);
  for (uint c = lid; c < p.cols; c += T)
    dst[c] = ((src[c] - mu) * inv * g[c] + b[c]) * p.scale + p.offset;
}

// ---- layer norm's pullback over the last axis, in three launches ----
// cuda's tl_layer_norm_bwd_*: with x̂ = (x − μ)·s and ĝ = dy ⊙ γ,
// dx = s · (ĝ − mean(ĝ) − x̂ · mean(ĝ ⊙ x̂)), dγ = Σ_rows dy ⊙ x̂, dβ = Σ_rows dy.
// The dx kernel recomputes (μ, s) as layer_norm_ does and leaves them in
// `stats` [2, rows] for the column sums, which fold in a fixed order.
struct layer_norm_bwd_params {
  uint rows, cols, rows_per_chunk, chunks;
  float eps;
};

// tree_sum_ for two values at once over the threadgroup's T threads (a power
// of two), in the same order as cuda's tl_tree_sum2_; `scratch` holds 2·T.
static inline float2 tree_sum2_(threadgroup float* scratch, uint lid, uint T,
                                float a, float b) {
  scratch[lid] = a;
  scratch[T + lid] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) {
      scratch[lid] += scratch[lid + s];
      scratch[T + lid] += scratch[T + lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float2 total = float2(scratch[0], scratch[T]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// A row's mean and 1/sqrt(biased variance + eps), the sums layer_norm_ takes
// (cuda's tl_row_stats_).
static inline float2 row_stats_(threadgroup float* scratch, uint lid,
                                device const float* src, uint cols, float eps) {
  constexpr uint T = 256;
  float inv_n = 1.0f / float(cols);
  float sum = 0.0f;
  for (uint c = lid; c < cols; c += T) sum += src[c];
  float mu = tree_sum_(scratch, lid, sum) * inv_n;
  float ss = 0.0f;
  for (uint c = lid; c < cols; c += T) {
    float v = src[c] - mu;
    ss += v * v;
  }
  return float2(mu, 1.0f / sqrt(tree_sum_(scratch, lid, ss) * inv_n + eps));
}

kernel void layer_norm_bwd_dx_(device const float* x  [[buffer(0)]],
                               device const float* g  [[buffer(1)]],
                               device const float* dy [[buffer(2)]],
                               device float* dx       [[buffer(3)]],
                               device float* stats    [[buffer(4)]],
                               constant layer_norm_bwd_params& p [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[2 * T];
  device const float* src = x + row * p.cols;
  device const float* gy = dy + row * p.cols;
  device float* dst = dx + row * p.cols;
  float inv_n = 1.0f / float(p.cols);

  const float2 st = row_stats_(scratch, lid, src, p.cols, p.eps);
  const float mu = st.x, inv = st.y;
  float sg = 0.0f, sgx = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float gh = gy[c] * g[c];
    sg += gh;
    sgx += gh * (src[c] - mu) * inv;
  }
  const float2 sums = tree_sum2_(scratch, lid, T, sg, sgx);
  const float mean_g = sums.x * inv_n, mean_gx = sums.y * inv_n;
  for (uint c = lid; c < p.cols; c += T) {
    float xhat = (src[c] - mu) * inv;
    dst[c] = inv * (gy[c] * g[c] - mean_g - xhat * mean_gx);
  }
  if (lid == 0) {
    stats[row] = mu;
    stats[p.rows + row] = inv;
  }
}

// The column sums: a threadgroup per (32-column strip, chunk of rows), 32×8
// threads, into `partials` [2, chunks, cols] (dγ's plane first).
kernel void layer_norm_bwd_gb_(device const float* x      [[buffer(0)]],
                               device const float* dy     [[buffer(1)]],
                               device const float* stats  [[buffer(2)]],
                               device float* partials     [[buffer(3)]],
                               constant layer_norm_bwd_params& p [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float sg[8][32], sgx[8][32];
  uint tx = tid.x, ty = tid.y;
  uint c = tg.x * 32 + tx;
  uint chunk = tg.y;
  uint r0 = chunk * p.rows_per_chunk;
  uint r1 = min(r0 + p.rows_per_chunk, p.rows);
  float ag = 0.0f, agx = 0.0f;
  if (c < p.cols) {
    for (uint r = r0 + ty; r < r1; r += 8) {
      float gy = dy[r * p.cols + c];
      float xhat = (x[r * p.cols + c] - stats[r]) * stats[p.rows + r];
      ag += gy;
      agx += gy * xhat;
    }
  }
  sg[ty][tx] = ag;
  sgx[ty][tx] = agx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ty == 0 && c < p.cols) {
    float tg_ = 0.0f, tgx = 0.0f;
    for (uint l = 0; l < 8; l++) {
      tg_ += sg[l][tx];
      tgx += sgx[l][tx];
    }
    uint at = chunk * p.cols + c;
    partials[at] = tgx;
    partials[p.chunks * p.cols + at] = tg_;
  }
}

// The second half: a thread per column folds its chunks into dγ and dβ.
kernel void layer_norm_bwd_gb_fold_(device const float* partials [[buffer(0)]],
                                    device float* dg             [[buffer(1)]],
                                    device float* db             [[buffer(2)]],
                                    constant layer_norm_bwd_params& p [[buffer(3)]],
                                    uint c [[thread_position_in_grid]]) {
  if (c >= p.cols) return;
  float tg_ = 0.0f, tb = 0.0f;
  for (uint k = 0; k < p.chunks; k++) {
    tg_ += partials[k * p.cols + c];
    tb += partials[(p.chunks + k) * p.cols + c];
  }
  dg[c] = tg_;
  db[c] = tb;
}

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors tensorlib_cuda.cu's own tl_rope. `a` is [rows, D]
// contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D] with T=1);
// row r's head-dim vector sits at position `pos + (r % T)`; pairs
// (j, j+D/2) rotate by angle = position * base^(-2j/D). Only the first `rot`
// dims rotate (rot == D: all of them), as pairs (j, j + rot/2) at angle
// position * base^(-2j/rot), the rest copied through: Qwen3.5's partial RoPE,
// cuda's rope_core. Dispatched flat over rows*(D/2), one invocation per
// (r, j) (CUDA instead grids by row, blocks by D/2 -- every kernel in this
// file is flat-1D); invocation j turns pair j when j < rot/2 and copies dims
// rot + j, rot + j + D/2, ...
struct rope_params {
  uint T, D, pos;
  uint half_;
  float base;
  uint n;
  uint has_bias;
  uint rot;
  float pos_scale;   // a linear scaling of the position (Gemma 3's global layers)
  uint has_factors;  // factors divide each pair's angle (gpu::rope_freq's)
};

// The turn of the pair (x0, x1) at dims (j, j + rot/2) of a row at `position`:
// every rotation in this file goes through it (rope_, rmsnorm_rope_). The
// angle is formed in f32, as the frameworks the models are trained in form it
// (at position p it carries ~p 2^-24 rad of rounding against an exact angle);
// its power, cos and sin are libm's, precise::, since the angle grows with the
// position past where the fast forms keep their range reduction (~40) and
// carries the power's relative error times the position. factors are the
// pairs', each dividing its angle as llama.cpp's freq factors do (precise::, so
// that none, 1, leaves the angle as it was).
static inline void rope_pair_(float position, uint j, uint rot, float base,
                              device const float* factors, uint has_factors,
                              thread float& x0, thread float& x1) {
  const float ff = has_factors ? factors[j] : 1.0f;
  const float ang =
      precise::divide(position * precise::pow(base, -2.0f * float(j) / float(rot)), ff);
  float c;
  const float s = precise::sincos(ang, c);  // one range reduction for both
  const float r0 = x0 * c - x1 * s;
  x1 = x0 * s + x1 * c;
  x0 = r0;
}

// `bias` is [rows, D] like x, added before the rotation when has_bias: the
// decode step's q/k bias folded in (at T == 1 a [H, D] bias is that shape).
kernel void rope_(device const float* x [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant rope_params& p [[buffer(2)]],
                  device const float* bias [[buffer(3)]],
                  device const float* factors [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const uint r = i / p.half_, j = i % p.half_, bi = r * p.D;
  for (uint d = p.rot + j; d < p.D; d += p.half_)
    out[bi + d] = p.has_bias ? x[bi + d] + bias[bi + d] : x[bi + d];
  const uint h = p.rot >> 1;
  if (j >= h) return;
  const uint t = p.T > 0 ? (r % p.T) : 0u;
  float x0 = x[bi + j];
  float x1 = x[bi + j + h];
  if (p.has_bias) {
    x0 += bias[bi + j];
    x1 += bias[bi + j + h];
  }
  rope_pair_(float(p.pos + t) * p.pos_scale, j, p.rot, p.base, factors, p.has_factors, x0, x1);
  out[bi + j] = x0;
  out[bi + j + h] = x1;
}

// ---------------------------------------------------------------------------
// Causal prefill attention and its pullback: cuda's attn_prefill_tiled_core /
// attn_bwd_dq_core / attn_bwd_dkv_core, the same online softmax and causal
// rule, on simdgroup_matrix. A threadgroup is 4 simdgroups of 8 rows each
// (queries, or keys for dK/dV); the products are 8×8 MMAs and the softmax
// runs on each lane's own fragment elements, so scores never leave registers.
// (attn_prefill_wide_, at 512, is the exception: its 4 simdgroups split one
// 8-query block's dims and sum their partial scores through threadgroup memory.)
// Loops over fragments are fully unrolled: a fragment array indexed by a loop
// variable is spilled to memory, which costs more than the MMAs save.
// Tiles are 16 rows at D=128: two tiles of 32 rows would fill all 32 KB of
// threadgroup memory.
// ---------------------------------------------------------------------------

struct attn_params {
  uint T, kv_stride, group, pos0;
  float scale;
  uint window;  // the prefill's sliding window; 0 none
};

// The first key a query attends among the n up to and including it, under a
// window of w (0 none): gpu::attn_window_lo's twin.
static inline uint attn_window_lo_(uint n, uint w) { return w > 0 && n > w ? n - w : 0; }

// Across the four lanes that hold one fragment row (lane bits 0 and 3).
static inline float frag_row_max_(float v) {
  v = max(v, simd_shuffle_xor(v, 1));
  return max(v, simd_shuffle_xor(v, 8));
}
static inline float frag_row_sum_(float v) {
  v += simd_shuffle_xor(v, 1);
  return v + simd_shuffle_xor(v, 8);
}

// Rows [base, end) of row-major [*, AD] sources into ROWS rows of threadgroup
// tiles; rows past `end` read as zero. Two sources share one loop, so both
// loads of an iteration are in flight together.
template <int ROWS, int AD, typename KT>
static inline void attn_stage_rows_(threadgroup float* dst,
                                    device const KT* src, uint base, uint end,
                                    uint tid) {
  constexpr uint A4 = AD / 4;
  threadgroup float4* d4 = (threadgroup float4*)dst;
  for (uint i = tid; i < uint(ROWS) * A4; i += 128) {
    const uint r = i / A4;
    d4[i] = base + r < end ? load4_(src, (base + r) * A4 + i % A4) : float4(0.f);
  }
}
template <int ROWS, int AD, typename KT>
static inline void attn_stage_rows_(threadgroup float* dst0,
                                    device const KT* src0,
                                    threadgroup float* dst1,
                                    device const KT* src1, uint base, uint end,
                                    uint tid) {
  constexpr uint A4 = AD / 4;
  threadgroup float4* d0 = (threadgroup float4*)dst0;
  threadgroup float4* d1 = (threadgroup float4*)dst1;
  for (uint i = tid; i < uint(ROWS) * A4; i += 128) {
    const uint r = i / A4;
    const bool live = base + r < end;
    const uint at = (base + r) * A4 + i % A4;
    d0[i] = live ? load4_(src0, at) : float4(0.f);
    d1[i] = live ? load4_(src1, at) : float4(0.f);
  }
}

// A lane's fragment row (`fn` its first column) written out, scaled by k.
template <int DF>
static inline void attn_store_row_(device float* row,
                                   thread simdgroup_matrix<float, 8, 8> (&f)[DF],
                                   float k) {
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    thread auto& e = f[i].thread_elements();
    *(device float2*)(row + i * 8) = float2(e[0], e[1]) * k;
  }
}

// One key tile of a block's online softmax: the raw scores s (q · K) masked to
// each row's live keys [plo, plo + width], turned into their exp2 weights in
// place, folded into the running max m and sum l, and the output fragments
// rescaled to the new max.
template <int KF, int DF>
static inline void attn_softmax_tile_(thread simdgroup_matrix<float, 8, 8> (&s)[KF],
                                      thread simdgroup_matrix<float, 8, 8> (&of)[DF], uint kt,
                                      uint fn, uint plo, uint width, float sl2, thread float& m,
                                      thread float& l) {
  float mt = -1e30f;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < KF; j++) {
    thread auto& e = s[j].thread_elements();
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 2; k++) {
      const uint col = kt + j * 8 + fn + k;
      // Causal and window mask in one compare: a col below plo wraps past width.
      e[k] = col - plo <= width ? e[k] * sl2 : -1e30f;
      mt = max(mt, e[k]);
    }
  }
  const float m_new = max(m, frag_row_max_(mt));
  const float corr = exp2(m - m_new);
  float ls = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < KF; j++) {
    thread auto& e = s[j].thread_elements();
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 2; k++) {
      e[k] = exp2(e[k] - m_new);
      ls += e[k];
    }
  }
  l = l * corr + frag_row_sum_(ls);
  m = m_new;

  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) of[i].thread_elements() *= corr;
}

// q, out [H,T,D]; K/V a cache of kv_stride elements per kv head read over
// [0, pos0+T); query p sits at pos0+p and attends the keys up to it, or with a
// window its last p.window of them. BK keys per tile. KT is the cache's
// element type: f32, or half for an f16 cache (caps::kv_f16). Width 256 is
// instantiated for the f16 cache only, as cuda's (an f32 cache there takes
// gpu::attn_prefill's GEMM form); its two 16-key tiles fill the whole 32 KB of
// threadgroup memory.
template <int AD, int BK, typename KT>
kernel void attn_prefill_(device const float* q   [[buffer(0)]],
                          device const KT* K      [[buffer(1)]],
                          device const KT* V      [[buffer(2)]],
                          device float* out       [[buffer(3)]],
                          constant attn_params& p [[buffer(4)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 ntg [[threadgroups_per_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 32, DF = AD / 8, KF = BK / 8;
  static_assert(2 * BK >= BQ, "Q is staged through the K/V tiles");
  threadgroup float4 KV4[2 * BK * AD / 4];  // K tile rows, then V tile rows
  threadgroup float* Ks = (threadgroup float*)KV4;
  threadgroup float* Vs = Ks + BK * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint qbase = (ntg.y - 1u - tg.y) * BQ;  // heaviest tiles first
  const uint kv_h = p.group ? h / p.group : h;
  device const KT* Kh = K + kv_h * p.kv_stride;
  device const KT* Vh = V + kv_h * p.kv_stride;

  attn_stage_rows_<BQ, AD>(Ks, q + h * T * AD, qbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> qf[DF], of[DF];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(qf[i], &Ks[sid * 8 * AD + i * 8], AD);
    of[i] = simdgroup_matrix<float, 8, 8>(0);
  }

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  // The row's live keys [plo, pabs]: its causal end and its window's start.
  const uint pabs = p.pos0 + qbase + sid * 8 + fm, plo = attn_window_lo_(pabs + 1, p.window),
             width = pabs - plo;
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const uint kmax = p.pos0 + last;
  // The tile of the block's first key: its first row's window start (the later
  // rows' windows start later).
  const uint kt0 = attn_window_lo_(p.pos0 + qbase + 1, p.window) / BK * BK;
  const float sl2 = p.scale * M_LOG2E_F;
  // The running max starts above the mask's -1e30, so a masked key weighs
  // exp2(-1e30 - m) = 0 even in a tile with no live key (a row's leading tiles
  // under a window), where m stays -1e20 and the tile adds nothing.
  float m = -1e20f, l = 0.0f;

  for (uint kt = kt0; kt <= kmax; kt += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<BK, AD>(Ks, Kh, Vs, Vh, kt, kmax + 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[KF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      simdgroup_matrix<float, 8, 8> kf;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], qf[i], kf, s[j]);
      }
    }

    attn_softmax_tile_(s, of, kt, fn, plo, width, sl2, m, l);
    simdgroup_matrix<float, 8, 8> vf;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(vf, &Vs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(of[i], s[j], vf, of[i]);
      }
    }
  }

  const uint qi = qbase + sid * 8 + fm;
  if (qi < T) attn_store_row_(out + (h * T + qi) * AD + fn, of, 1.0f / l);
}
#define TL_ATTN_PREFILL(name, AD, BK, KT)                                    \
  template [[host_name(name)]] kernel void attn_prefill_<AD, BK, KT>(        \
      device const float*, device const KT*, device const KT*, device float*, \
      constant attn_params&, uint2, uint2, uint, uint, uint);
TL_ATTN_PREFILL("attn_prefill_64_", 64, 32, float)
TL_ATTN_PREFILL("attn_prefill_128_", 128, 16, float)
TL_ATTN_PREFILL("attn_prefill_64_f16_", 64, 32, half)
TL_ATTN_PREFILL("attn_prefill_128_f16_", 128, 16, half)
TL_ATTN_PREFILL("attn_prefill_256_f16_", 256, 16, half)
#undef TL_ATTN_PREFILL

// Rows [r0, r0 + 8) of a row-major source (rows ld apart) as an 8×8 fragment,
// or as its transpose; rows at or past `end` read as zero. The source is read
// where it lies (an f16 cache as half: an MMA takes half and float operands
// together, the products exact in f32).
template <typename T>
static inline void frag_rows_(thread simdgroup_matrix<T, 8, 8>& f, device const T* src, uint ld,
                              uint r0, uint end, bool tr, uint fm, uint fn) {
  if (r0 + 8 <= end) {
    simdgroup_load(f, src + ulong(r0) * ld, ld, ulong2(0, 0), tr);
    return;
  }
  thread auto& e = f.thread_elements();
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 2; k++) {
    const uint r = r0 + (tr ? fn + k : fm), c = tr ? fm : fn + k;
    e[k] = r < end ? src[ulong(r) * ld + c] : T(0);
  }
}

// The prefill at a head width too wide for attn_prefill_'s tiles (512, Gemma
// 4's global layers): llama.cpp's form for its wide heads. A threadgroup owns
// 8 queries (metal_nn.h's attn_prefill_rows_), and its 4 simdgroups split the
// head's dims, each holding its quarter of q and of the output in registers.
// Per tile of BK keys, each simdgroup's partial scores over its dims are
// summed through threadgroup memory (double-buffered, one barrier a tile),
// every simdgroup runs the same online softmax on the sums, and multiplies
// its quarter of V. K and V are read from the cache where they lie, so no
// tile is staged; the last tile's keys past the block read as zero. 8 keys a
// tile hold the fewest registers, and ran fastest of 8, 16, 32 and 64 on the
// M1 Pro.
template <int AD, typename KT>
kernel void attn_prefill_wide_(device const float* q   [[buffer(0)]],
                               device const KT* K      [[buffer(1)]],
                               device const KT* V      [[buffer(2)]],
                               device float* out       [[buffer(3)]],
                               constant attn_params& p [[buffer(4)]],
                               uint2 tg  [[threadgroup_position_in_grid]],
                               uint2 ntg [[threadgroups_per_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sid  [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 8, BK = 8, NSG = 4, DS = AD / NSG, DF = DS / 8, KF = BK / 8;
  static_assert(2 * NSG * BQ * BK <= BQ * AD, "the partial scores fit where q was staged");
  threadgroup float4 buf4[BQ * AD / 4];  // q's rows, then the partial scores
  threadgroup float* buf = (threadgroup float*)buf4;

  const uint T = p.T;
  const uint h = tg.x;
  const uint qbase = (ntg.y - 1u - tg.y) * BQ;  // heaviest tiles first
  const uint kv_h = p.group ? h / p.group : h;
  const uint d0 = sid * DS;  // this simdgroup's dims
  device const KT* Kh = K + kv_h * p.kv_stride + d0;
  device const KT* Vh = V + kv_h * p.kv_stride + d0;

  // q staged whole, each simdgroup then taking its quarter: a coalesced read
  // ran faster on the M1 Pro than each simdgroup's own fragments.
  attn_stage_rows_<BQ, AD>(buf, q + h * T * AD, qbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> qf[DF], of[DF];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(qf[i], &buf[d0 + i * 8], AD);
    of[i] = simdgroup_matrix<float, 8, 8>(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  // The row's live keys [plo, pabs], as attn_prefill_'s.
  const uint pabs = p.pos0 + qbase + fm, plo = attn_window_lo_(pabs + 1, p.window),
             width = pabs - plo;
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const uint kmax = p.pos0 + last;
  const uint kt0 = attn_window_lo_(p.pos0 + qbase + 1, p.window) / BK * BK;
  const float sl2 = p.scale * M_LOG2E_F;
  float m = -1e20f, l = 0.0f;  // as attn_prefill_'s

  uint parity = 0;
  for (uint kt = kt0; kt <= kmax; kt += BK, parity ^= 1u) {
    // This simdgroup's partial scores, into its slot of this tile's buffer.
    threadgroup float* part = buf + parity * (NSG * BQ * BK);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      simdgroup_matrix<float, 8, 8> s = simdgroup_matrix<float, 8, 8>(0);
      simdgroup_matrix<KT, 8, 8> kf;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        frag_rows_(kf, Kh + i * 8, AD, kt + j * 8, kmax + 1, true, fm, fn);
        simdgroup_multiply_accumulate(s, qf[i], kf, s);
      }
      simdgroup_store(s, part + sid * BQ * BK + j * 8, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The whole scores, this lane's elements of them: the same in every simdgroup.
    simdgroup_matrix<float, 8, 8> pf[KF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& pe = pf[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        pe[k] = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < NSG; g++) pe[k] += part[g * BQ * BK + fm * BK + j * 8 + fn + k];
      }
    }
    attn_softmax_tile_(pf, of, kt, fn, plo, width, sl2, m, l);
    simdgroup_matrix<KT, 8, 8> vf;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        frag_rows_(vf, Vh + i * 8, AD, kt + j * 8, kmax + 1, false, fm, fn);
        simdgroup_multiply_accumulate(of[i], pf[j], vf, of[i]);
      }
    }
  }

  const uint qi = qbase + fm;
  if (qi < T) attn_store_row_(out + (h * T + qi) * AD + d0 + fn, of, 1.0f / l);
}
#define TL_ATTN_PREFILL_WIDE(name, AD, KT)                                     \
  template [[host_name(name)]] kernel void attn_prefill_wide_<AD, KT>(         \
      device const float*, device const KT*, device const KT*, device float*, \
      constant attn_params&, uint2, uint2, uint, uint, uint);
TL_ATTN_PREFILL_WIDE("attn_prefill_512_", 512, float)
TL_ATTN_PREFILL_WIDE("attn_prefill_512_f16_", 512, half)
#undef TL_ATTN_PREFILL_WIDE

// The query half of the pullback: dq and `stats` [2,H,T] (the row
// logsumexp, then Δ = dO·O) for the key/value half. dq = scale·C·K / l with
// C = exp(S − m)·(dP − Δ), S = scale·QKᵀ and dP = dO·Vᵀ.
template <int AD, int BK>
kernel void attn_bwd_dq_(device const float* q     [[buffer(0)]],
                         device const float* K     [[buffer(1)]],
                         device const float* V     [[buffer(2)]],
                         device const float* dO    [[buffer(3)]],
                         device const float* O     [[buffer(4)]],
                         device float* dq          [[buffer(5)]],
                         device float* stats       [[buffer(6)]],
                         constant attn_params& p   [[buffer(7)]],
                         uint2 tg  [[threadgroup_position_in_grid]],
                         uint2 ntg [[threadgroups_per_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sid  [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 128, BQ = 32, DF = AD / 8, KF = BK / 8, A4 = AD / 4;
  static_assert(2 * BK >= BQ, "Q and dO are each staged through the K/V tiles");
  static_assert(NT % A4 == 0 && A4 <= 32, "a row's float4s share a simdgroup");
  threadgroup float4 KV4[2 * BK * AD / 4];  // K tile rows, then V tile rows
  threadgroup float Ds[BQ];                 // Δ = dO·O per query row
  threadgroup float* Ks = (threadgroup float*)KV4;
  threadgroup float* Vs = Ks + BK * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint qbase = (ntg.y - 1u - tg.y) * BQ;
  device const float* Kh = K + h * T * AD;
  device const float* Vh = V + h * T * AD;

  // Q, then dO, into registers (each staged through the K tile).
  simdgroup_matrix<float, 8, 8> qf[DF], gf[DF], acc[DF];
  attn_stage_rows_<BQ, AD>(Ks, q + h * T * AD, qbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(qf[i], &Ks[sid * 8 * AD + i * 8], AD);
    acc[i] = simdgroup_matrix<float, 8, 8>(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    // dO staged like attn_stage_rows_, with Δ reduced over the A4 lanes that
    // share a row as it passes.
    device const float4* G4 = (device const float4*)(dO + h * T * AD);
    device const float4* O4 = (device const float4*)(O + h * T * AD);
    threadgroup float4* d4 = KV4;
    for (uint i = tid; i < uint(BQ * A4); i += NT) {
      const uint r = i / A4, c = i % A4;
      const bool live = qbase + r < T;
      const float4 g = live ? G4[(qbase + r) * A4 + c] : float4(0.f);
      d4[i] = g;
      float d = live ? dot(g, O4[(qbase + r) * A4 + c]) : 0.f;
      for (ushort o = A4 / 2; o > 0; o /= 2) d += simd_shuffle_xor(d, o);
      if (c == 0) Ds[r] = d;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++)
    simdgroup_load(gf[i], &Ks[sid * 8 * AD + i * 8], AD);

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  const uint prow = qbase + sid * 8 + fm;
  const float delta = Ds[sid * 8 + fm];
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const float sl2 = p.scale * M_LOG2E_F;
  float m = -1e30f, l = 0.0f;

  for (uint kt = 0; kt <= last; kt += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<BK, AD>(Ks, Kh, Vs, Vh, kt, last + 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[KF], dp[KF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      simdgroup_matrix<float, 8, 8> kf, vf;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      dp[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], qf[i], kf, s[j]);
        simdgroup_load(vf, &Vs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(dp[j], gf[i], vf, dp[j]);
      }
    }

    float mt = -1e30f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        e[k] = kt + j * 8 + fn + k <= prow ? e[k] * sl2 : -1e30f;
        mt = max(mt, e[k]);
      }
    }
    const float m_new = max(m, frag_row_max_(mt));
    const float corr = exp2(m - m_new);
    float ls = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      thread auto& d = dp[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const float x = exp2(e[k] - m_new);
        ls += x;
        e[k] = x * (d[k] - delta);  // C, in place of S
      }
    }
    l = l * corr + frag_row_sum_(ls);
    m = m_new;

    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < DF; i++) acc[i].thread_elements() *= corr;
    simdgroup_matrix<float, 8, 8> kf;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(acc[i], s[j], kf, acc[i]);
      }
    }
  }

  if (prow < T) {
    attn_store_row_(dq + (h * T + prow) * AD + fn, acc, p.scale / l);
    if (fn == 0) {
      // m is in log2 units scaled like the scores; stats keep natural log.
      stats[h * T + prow] = (m + log2(l)) * M_LN2_F;
      stats[ntg.x * T + h * T + prow] = delta;
    }
  }
}
template [[host_name("attn_bwd_dq_64_")]] kernel void
attn_bwd_dq_<64, 32>(device const float*, device const float*,
                     device const float*, device const float*,
                     device const float*, device float*, device float*,
                     constant attn_params&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_bwd_dq_128_")]] kernel void
attn_bwd_dq_<128, 16>(device const float*, device const float*,
                      device const float*, device const float*,
                      device const float*, device float*, device float*,
                      constant attn_params&, uint2, uint2, uint, uint, uint);

// The key/value half: one threadgroup per (head, 32-key block), walking
// the query tiles that can see those keys, P read off the dq half's
// stats: dV = Pᵀ·dO and dK = scale·Cᵀ·Q, C = P·(dP − Δ).
template <int AD, int TQ>
kernel void attn_bwd_dkv_(device const float* q     [[buffer(0)]],
                          device const float* K     [[buffer(1)]],
                          device const float* V     [[buffer(2)]],
                          device const float* dO    [[buffer(3)]],
                          device const float* stats [[buffer(4)]],
                          device float* dK          [[buffer(5)]],
                          device float* dV          [[buffer(6)]],
                          constant attn_params& p   [[buffer(7)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 ntg [[threadgroups_per_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 128, BK = 32, DF = AD / 8, QF = TQ / 8;
  static_assert(2 * TQ >= BK, "K and V are each staged through the Q/dO tiles");
  threadgroup float4 QG4[2 * TQ * AD / 4];  // Q tile rows, then dO tile rows
  threadgroup float Ls[TQ], Ds[TQ];         // log2-scaled logsumexp, Δ
  threadgroup float* Qs = (threadgroup float*)QG4;
  threadgroup float* Gs = Qs + TQ * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint kbase = tg.y * BK;
  device const float* qh = q + h * T * AD;
  device const float* Gh = dO + h * T * AD;
  device const float* Lh = stats + h * T;
  device const float* Dh = stats + ntg.x * T + h * T;

  // This block's K, then V, rows into registers (each staged through Qs).
  simdgroup_matrix<float, 8, 8> kf[DF], vf[DF], dk[DF], dv[DF];
  attn_stage_rows_<BK, AD>(Qs, K + h * T * AD, kbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(kf[i], &Qs[sid * 8 * AD + i * 8], AD);
    dk[i] = simdgroup_matrix<float, 8, 8>(0);
    dv[i] = simdgroup_matrix<float, 8, 8>(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  attn_stage_rows_<BK, AD>(Qs, V + h * T * AD, kbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++)
    simdgroup_load(vf[i], &Qs[sid * 8 * AD + i * 8], AD);

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  const uint krow = kbase + sid * 8 + fm;
  const float sl2 = p.scale * M_LOG2E_F;

  for (uint qt = (kbase / TQ) * TQ; qt < T; qt += TQ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<TQ, AD>(Qs, qh, Gs, Gh, qt, T, tid);
    for (uint r = tid; r < uint(TQ); r += NT) {
      const bool live = qt + r < T;
      Ls[r] = live ? Lh[qt + r] * M_LOG2E_F : 0.f;
      Ds[r] = live ? Dh[qt + r] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[QF], dp[QF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      simdgroup_matrix<float, 8, 8> t;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      dp[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(t, &Qs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], kf[i], t, s[j]);
        simdgroup_load(t, &Gs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(dp[j], vf[i], t, dp[j]);
      }
    }

    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      thread auto& e = s[j].thread_elements();
      thread auto& d = dp[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const uint ql = j * 8 + fn + k, qi = qt + ql;
        const float x =
            (qi < T && krow <= qi) ? exp2(e[k] * sl2 - Ls[ql]) : 0.f;
        e[k] = x;                   // Pᵀ
        d[k] = x * (d[k] - Ds[ql]);  // Cᵀ
      }
    }

    simdgroup_matrix<float, 8, 8> t;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(t, &Gs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(dv[i], s[j], t, dv[i]);
        simdgroup_load(t, &Qs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(dk[i], dp[j], t, dk[i]);
      }
    }
  }

  if (krow < T) {
    attn_store_row_(dK + (h * T + krow) * AD + fn, dk, p.scale);
    attn_store_row_(dV + (h * T + krow) * AD + fn, dv, 1.0f);
  }
}
template [[host_name("attn_bwd_dkv_64_")]] kernel void
attn_bwd_dkv_<64, 32>(device const float*, device const float*,
                      device const float*, device const float*,
                      device const float*, device float*, device float*,
                      constant attn_params&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_bwd_dkv_128_")]] kernel void
attn_bwd_dkv_<128, 16>(device const float*, device const float*,
                       device const float*, device const float*,
                       device const float*, device float*, device float*,
                       constant attn_params&, uint2, uint2, uint, uint, uint);

// ---------------------------------------------------------------------------
// Decode GEMV: y[n] = sum_k a[k]·B[k,n], cuda's tl_gemv_f32/tl_gemv_bf16.
// Batch~1 decode is memory-bandwidth bound on the K×N weight, so one thread
// owns one output column and consecutive threads read consecutive columns —
// the B reads coalesce. `a` is staged a tile at a time, one read per
// threadgroup rather than one per thread.
//
// Split-K over the grid's y: a narrow layer (N/256 threadgroups) leaves most
// of the GPU idle and runs slower than the M=1 GEMM it replaces. Each slice
// sums its own K range into out[s·n + col] and a combine pass adds the slices
// up; with one slice `out` is y itself and the pass is skipped.
// ---------------------------------------------------------------------------

struct gemv_params {
  uint n, k, chunk;
};

// `as` is the caller's threadgroup tile of NT floats (threadgroup memory can
// only be declared in a kernel).
template <typename WT>
static inline void gemv_(device const float* a, device const WT* B,
                         device float* out, constant gemv_params& p,
                         threadgroup float* as, uint2 tgp, uint tid) {
  constexpr uint NT = 256;
  const uint col = tgp.x * NT + tid;
  const uint k0 = tgp.y * p.chunk;
  const uint k1 = min(p.k, k0 + p.chunk);
  float acc = 0.0f;
  for (uint kt = k0; kt < k1; kt += NT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    as[tid] = kt + tid < k1 ? a[kt + tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kn = min(NT, k1 - kt);
    if (col < p.n) {
      device const WT* Bk = B + (ulong)kt * p.n + col;
      for (uint i = 0; i < kn; i++) {
        acc += as[i] * widen_(Bk[(ulong)i * p.n]);
      }
    }
  }
  if (col < p.n) out[(ulong)tgp.y * p.n + col] = acc;
}

kernel void gemv_f32_(device const float* a   [[buffer(0)]],
                      device const float* B   [[buffer(1)]],
                      device float* out       [[buffer(2)]],
                      constant gemv_params& p [[buffer(3)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]]) {
  threadgroup float as[256];
  gemv_<float>(a, B, out, p, as, tgp, tid);
}

kernel void gemv_bf16_(device const float* a   [[buffer(0)]],
                       device const ushort* B  [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       constant gemv_params& p [[buffer(3)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]]) {
  threadgroup float as[256];
  gemv_<ushort>(a, B, out, p, as, tgp, tid);
}

// Sum the split-K slices: parts × n partials down to y[n].
struct gemv_combine_params {
  uint n, parts;
};

kernel void gemv_combine_(device const float* parts        [[buffer(0)]],
                          device float* y                  [[buffer(1)]],
                          constant gemv_combine_params& p  [[buffer(2)]],
                          uint col [[thread_position_in_grid]]) {
  if (col >= p.n) return;
  float acc = 0.0f;
  for (uint s = 0; s < p.parts; s++) acc += parts[(ulong)s * p.n + col];
  y[col] = acc;
}

// The [N,K] weight layouts (K contiguous per output row — GGML's own, so the
// loader drops the transpose) take one threadgroup per output row, its
// threads splitting K. That is the whole difference from the [K,N] GEMV
// above: no split-K grid, because the row IS the parallel axis, and a
// threadgroup reduction instead of a per-thread column.
//
// The reduce is shared by the bf16, f32 and int4 forms, which differ only in how a
// thread reads its slice. `red` is the caller's [8] threadgroup scratch
// (threadgroup memory can only be declared in a kernel); 8 covers the
// 256-thread cap the host dispatches.
static inline void gemv_row_store_(float acc, threadgroup float* red,
                                   device float* y_n, uint tid, uint sgid,
                                   uint nsg, uint lane) {
  const float s = simd_sum(acc);
  if (lane == 0) red[sgid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.0f;
    for (uint w = 0; w < nsg; w++) t += red[w];
    *y_n = t;
  }
}

struct gemv_row_params {
  uint n, k;
};

// bf16 (ushort) or f32 weights in [N,K]: cuda's tl_gemv_bf16_row. Eight
// contiguous weights per thread per step, so consecutive threads read
// consecutive spans of the row. Requires k % 8 == 0 (every transformer dim;
// the host checks).
template <typename WT>
kernel void gemv_row_(device const float* a      [[buffer(0)]],
                      device const WT* B         [[buffer(1)]],
                      device float* y            [[buffer(2)]],
                      constant gemv_row_params& p [[buffer(3)]],
                      uint row  [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint nt   [[threads_per_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const WT* brow = B + (ulong)row * p.k;
  float acc = 0.0f;
  for (uint k0 = tid * 8; k0 < p.k; k0 += nt * 8) {
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; j++) acc += a[k0 + j] * widen_(brow[k0 + j]);
  }
  gemv_row_store_(acc, red, y + row, tid, sgid, nsg, lane);
}
#define TL_GEMV_ROW(name, WT)                                                \
  template [[host_name(name)]] kernel void gemv_row_<WT>(                    \
      device const float*, device const WT*, device float*,                  \
      constant gemv_row_params&, uint, uint, uint, uint, uint, uint);
TL_GEMV_ROW("gemv_bf16_row_", ushort)
TL_GEMV_ROW("gemv_f32_row_", float)
#undef TL_GEMV_ROW

// int4 weights: cuda's tl_gemv_q4. Same one-threadgroup-per-row shape, whose
// quantization groups are contiguous in the [N,K] packing; a thread takes one
// word (8 packed int4) per step.
struct gemv_q4_params {
  uint n, k, group;
};

kernel void gemv_q4_(device const float* a       [[buffer(0)]],
                     device const uint* qw       [[buffer(1)]],
                     device const float* scales  [[buffer(2)]],
                     device float* y             [[buffer(3)]],
                     constant gemv_q4_params& p  [[buffer(4)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]],
                     uint nt   [[threads_per_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint nsg  [[simdgroups_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const uint* qrow = qw + (ulong)row * (p.k >> 3);
  device const float* srow = scales + (ulong)row * (p.k / p.group);
  float acc = 0.0f;
  for (uint k0 = tid * 8; k0 < p.k; k0 += nt * 8) {
    const uint w = qrow[k0 >> 3];
    const float sc = srow[k0 / p.group];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; j++) {
      const int q = int((w >> (j * 4)) & 0xFu) - 8;
      acc += a[k0 + j] * (sc * float(q));
    }
  }
  gemv_row_store_(acc, red, y + row, tid, sgid, nsg, lane);
}

// ---------------------------------------------------------------------------
// bf16-weight NT GEMM: C[M,N] = A[M,K] · B[N,K]ᵀ, cuda's tl_gemm_bf16_nt:
// the batched prefill's projection, where the same weight serves a whole
// chunk of prompt tokens at once. The same over a GGUF block-quantized
// weight (cuda's gemm_quant, its exact form). All run STEEL's body
// (metal_kernels.metal) with W [N,K] as the transposed B: a bf16 W whose K
// is whole 8-element vectors through its vectorized loader, and the rest
// through a reader f(n, k) (below), each weight dequantized to f32 as it is
// staged. C's rows are p.ldc apart, so a run of a weight's rows can write its
// columns of a wider output.
// ---------------------------------------------------------------------------

// A reader gives one weight, f(n, k), and a run of RUN, f.vec(n, k, v): the
// run from k, a multiple of RUN, which never crosses a format's scale group
// (16 or 32 weights), so one scale serves it and its bytes come in a few wide
// loads. STEEL's functor loader takes a tile's whole rows by vec and its
// clipped edges by element.

// B[n,k] of a bf16 [N,K] weight (the reader only where K is not whole vectors).
struct read_bf16_ {
  static constexpr constant short RUN = 8;
  device const ushort* B;
  uint K;
  float operator()(uint n, uint k) const { return widen_(B[(ulong)n * K + k]); }
  void vec(uint n, uint k, thread float* v) const {
    for (uint j = 0; j < 8; j++) v[j] = (*this)(n, k + j);
  }
};

// ---- GGUF block-quantized weights ([N,K] rows of whole blocks, read where
// they lie; include/ggml_quant.h has the formats in words and the CPU
// reference). A Q4_K row is whole 144-byte blocks and a Q5_K row 176-byte ones,
// so from a 4-byte aligned weight (the host checks) their readers take 32-bit
// loads (the decode GEMVs, llama.cpp's, 16-bit ones); Q6_K's (210), Q8_0's (34),
// Q5_0's (22), Q4_0's (18) and Q4_1's (20, read as Q4_0's) only by halves.

// Scale and min of a Q4_K group j (0..7) from the block's 12 scale bytes held
// as three words: ggml's get_scale_min_k4, cuda's q4k_scale_min_.
static inline void q4k_scale_min_(uint3 w, uint j, thread float& sc,
                                  thread float& mn) {
  const uint sh = 8u * (j & 3u), a = w.x >> sh, b = w.y >> sh, c = w.z >> sh;
  sc = float(j < 4 ? a & 63u : (c & 0xFu) | ((a >> 2) & 0x30u));
  mn = float(j < 4 ? b & 63u : ((c >> 4) & 0xFu) | ((b >> 2) & 0x30u));
}
static inline uint3 q4k_scale_words_(device const uchar* b) {
  device const uint* s = reinterpret_cast<device const uint*>(b + 4);
  return uint3(s[0], s[1], s[2]);
}
static inline float half_at_(device const uchar* b) {
  return float(*reinterpret_cast<device const half*>(b));
}
// A 32-bit word from two 16-bit loads (a 2-byte aligned block).
static inline uint ld32_h_(device const uchar* b) {
  device const ushort* s = reinterpret_cast<device const ushort*>(b);
  return uint(s[0]) | (uint(s[1]) << 16);
}

// One weight, k of its row: the dequantization the CPU reference does, op for
// op. Each format's block geometry (weights, bytes) is its reader's, which the
// GEMVs below read too.
struct read_q4k_ {
  static constexpr constant uint ELEMS = 256, BYTES = 144;
  static constexpr constant short RUN = 16;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, g = o >> 5;
    float sc, mn;
    q4k_scale_min_(q4k_scale_words_(b), g, sc, mn);
    const uint q = b[16 + 32 * (g >> 1) + (o & 31u)];
    const float d1 = half_at_(b) * sc, m1 = half_at_(b + 2) * mn;
    return d1 * float((g & 1u) ? q >> 4 : q & 0xFu) - m1;
  }
  // Four 32-bit loads (the block and its quants 4-byte aligned: quant_ok_).
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, g = o >> 5;
    float sc, mn;
    q4k_scale_min_(q4k_scale_words_(b), g, sc, mn);
    device const uint* q =
        reinterpret_cast<device const uint*>(b + 16 + 32 * (g >> 1) + (o & 31u));
    const uint sh = (g & 1u) ? 4u : 0u;
    const float d1 = half_at_(b) * sc, m1 = half_at_(b + 2) * mn;
    for (uint i = 0; i < RUN / 4; i++) {  // four weights a word
      const uint w = q[i] >> sh;
      for (uint j = 0; j < 4; j++) v[4 * i + j] = d1 * float((w >> (8 * j)) & 0xFu) - m1;
    }
  }
};
// Q5_K, 176 bytes a block: Q4_K's d, dmin and scale bytes, then qh (bit g of
// byte l the fifth bit of weight l of group g), then Q4_K's quants at byte 48.
struct read_q5k_ {
  static constexpr constant uint ELEMS = 256, BYTES = 176;
  static constexpr constant short RUN = 16;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, g = o >> 5, l = o & 31u;
    float sc, mn;
    q4k_scale_min_(q4k_scale_words_(b), g, sc, mn);
    const uint q = b[48 + 32 * (g >> 1) + l], h = (b[16 + l] >> g) & 1u;
    const float d1 = half_at_(b) * sc, m1 = half_at_(b + 2) * mn;
    return d1 * float(((g & 1u) ? q >> 4 : q & 0xFu) | (h << 4)) - m1;
  }
  // Eight 32-bit loads (the block 4-byte aligned: quant_ok_).
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, g = o >> 5, l = o & 31u;
    float sc, mn;
    q4k_scale_min_(q4k_scale_words_(b), g, sc, mn);
    device const uint* q = reinterpret_cast<device const uint*>(b + 48 + 32 * (g >> 1) + l);
    device const uint* qh = reinterpret_cast<device const uint*>(b + 16 + l);
    const uint sh = (g & 1u) ? 4u : 0u;
    const float d1 = half_at_(b) * sc, m1 = half_at_(b + 2) * mn;
    for (uint i = 0; i < RUN / 4; i++) {  // four weights a word
      const uint w = q[i] >> sh, h = qh[i] >> g;
      for (uint j = 0; j < 4; j++)
        v[4 * i + j] = d1 * float(((w >> (8 * j)) & 0xFu) | (((h >> (8 * j)) & 1u) << 4)) - m1;
    }
  }
};
struct read_q6k_ {
  static constexpr constant uint ELEMS = 256, BYTES = 210;
  static constexpr constant short RUN = 16;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, h = o >> 7, r = o & 127u, quarter = r >> 5;
    const uint l = r & 31u;
    const uint ql = b[64 * h + l + ((quarter & 1u) ? 32u : 0u)];
    const uint qh = b[128 + 32 * h + l];
    const int q = int(((quarter >= 2 ? ql >> 4 : ql & 0xFu) |
                       (((qh >> (2 * quarter)) & 3u) << 4))) - 32;
    device const char* scales = reinterpret_cast<device const char*>(b + 192);
    const float sc = float(scales[8 * h + l / 16 + 2 * quarter]);
    return half_at_(b + 208) * sc * float(q);
  }
  // 16-bit loads (the block 2-byte aligned).
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 8) * BYTES;
    const uint o = k & 255u, h = o >> 7, r = o & 127u, quarter = r >> 5;
    const uint l = r & 31u;
    device const ushort* ql = reinterpret_cast<device const ushort*>(
        b + 64 * h + l + ((quarter & 1u) ? 32u : 0u));
    device const ushort* qh = reinterpret_cast<device const ushort*>(b + 128 + 32 * h + l);
    device const char* scales = reinterpret_cast<device const char*>(b + 192);
    const float d = half_at_(b + 208) * float(scales[8 * h + l / 16 + 2 * quarter]);
    const uint ls = quarter >= 2 ? 4u : 0u, hs = 2 * quarter;
    for (uint i = 0; i < RUN / 2; i++) {  // two weights a load
      const uint lo = ql[i], hi = qh[i];
      for (uint e = 0; e < 2; e++) {
        const int q = int((((lo >> (8 * e)) >> ls) & 0xFu) |
                          ((((hi >> (8 * e)) >> hs) & 3u) << 4)) - 32;
        v[2 * i + e] = d * float(q);
      }
    }
  }
};
struct read_q80_ {
  static constexpr constant uint ELEMS = 32, BYTES = 34;
  static constexpr constant short RUN = 8;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    device const char* qs = reinterpret_cast<device const char*>(b + 2);
    return float(qs[k & 31u]) * half_at_(b);
  }
  // Four 16-bit loads (the block 2-byte aligned).
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    device const ushort* qs = reinterpret_cast<device const ushort*>(b + 2 + (k & 31u));
    const float d = half_at_(b);
    for (uint i = 0; i < 4; i++) {
      const ushort w = qs[i];
      v[2 * i] = float(char(w & 0xFF)) * d;
      v[2 * i + 1] = float(char(w >> 8)) * d;
    }
  }
};

// The 32-weight formats Q5_0, Q4_0 and Q4_1 hold their four low bits alike: 16
// bytes, byte j element j's in its low nibble and element j + 16's in its high one.
// The nibbles of elements o.. (o % 8 == 0) from those bytes at qs, by 16-bit loads
// (the blocks are 2-byte aligned): the one reading of them.
static inline void nib32_8_(device const uchar* qs, uint o, thread uint* n) {
  device const ushort* q16 = reinterpret_cast<device const ushort*>(qs + (o & 15u));
  const uint sh = (o & 16u) ? 4u : 0u;  // elements 16.. are the high nibbles
  for (uint i = 0; i < 4; i++) {
    const uint w = q16[i];
    n[2 * i] = (w >> sh) & 0xFu;
    n[2 * i + 1] = (w >> (8u + sh)) & 0xFu;
  }
}
// The two whose weight is d * q, a half d at the block's start, as each spells q
// for the eight elements o..: Q5_0, 22 bytes, d, the fifth bits qh (bit e of the
// 32-bit word element e's), then the nibbles; q - 16, as ggml's
// dequantize_row_q5_0 (cuda's q50_vals_). Q4_0, 18 bytes, d and the nibbles; q - 8.
struct q50_ {
  static constexpr constant uint BYTES = 22;
  static void vals8(device const uchar* b, uint o, thread float* q) {
    uint n[8];
    nib32_8_(b + 6, o, n);
    const uint qh = ld32_h_(b + 2);
    for (uint i = 0; i < 8; i++) q[i] = float(int(n[i] | (((qh >> (o + i)) & 1u) << 4)) - 16);
  }
};
struct q40_ {
  static constexpr constant uint BYTES = 18;
  static void vals8(device const uchar* b, uint o, thread float* q) {
    uint n[8];
    nib32_8_(b + 2, o, n);
    for (uint i = 0; i < 8; i++) q[i] = float(int(n[i]) - 8);
  }
};
template <class Vals>
struct read_d32_ {
  using V = Vals;
  static constexpr constant uint ELEMS = 32, BYTES = V::BYTES;
  static constexpr constant short RUN = 8;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    float q[8];
    V::vals8(b, k & 24u, q);
    return half_at_(b) * q[k & 7u];
  }
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    V::vals8(b, k & 31u, v);
    const float d = half_at_(b);
    for (uint i = 0; i < 8; i++) v[i] *= d;
  }
};
using read_q50_ = read_d32_<q50_>;
using read_q40_ = read_d32_<q40_>;
// Q4_1, 20 bytes a block of 32: d, a half min m, then the nibbles; weight d * q + m.
struct read_q41_ {
  static constexpr constant uint ELEMS = 32, BYTES = 20;
  static constexpr constant short RUN = 8;
  device const uchar* W;
  uint row_bytes;
  float operator()(uint n, uint k) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    uint q[8];
    nib32_8_(b + 4, k & 24u, q);
    return float(q[k & 7u]) * half_at_(b) + half_at_(b + 2);
  }
  void vec(uint n, uint k, thread float* v) const {
    device const uchar* b = W + (ulong)n * row_bytes + (k >> 5) * BYTES;
    uint q[8];
    nib32_8_(b + 4, k & 31u, q);
    const float d = half_at_(b), m = half_at_(b + 2);
    for (uint i = 0; i < 8; i++) v[i] = float(q[i]) * d + m;
  }
};

#define TL_READ_BF16 (read_bf16_{B, p.K})
#define TL_READ_Q(L) (L{B, p.K / L::ELEMS * L::BYTES})

// A bf16 W whose K is whole 8-element vectors: the vectorized loader.
TL_STEEL_KERNEL(gemm_bf16_steel_, 64, false, true, ushort)
TL_STEEL_KERNEL(gemm_bf16_steel32_, 32, false, true, ushort)
// The same summed in f16 (cuda's gemm_bf16_nt_f16acc): A and W rounded to
// half as they are staged, and the MMA's sums half.
TL_STEEL_KERNEL_ST(gemm_bf16_f16acc_, 64, false, true, ushort, B, half)
TL_STEEL_KERNEL_ST(gemm_bf16_f16acc32_, 32, false, true, ushort, B, half)
// The rest through a reader (steel_b_'s functor loader); the 32-row band for
// a short chunk (steel_grid_).
TL_STEEL_KERNEL_SRC(gemm_bf16_nt_, 64, false, true, ushort, TL_READ_BF16)
TL_STEEL_KERNEL_SRC(gemm_bf16_nt32_, 32, false, true, ushort, TL_READ_BF16)
// The block formats step K by 32: a thread's 16 weights a step are one run of
// the K-quants' readers (one scale decoded for them) or two of the others',
// and a step's barriers serve twice the weights. Against a step of 16, at
// Gemma 4 E2B's and E4B's MLP shapes (512 rows): Q4_K 2 to 3% faster, Q5_K 7
// to 8%, Q6_K 15%.
#define TL_GEMM_Q(F)                                                         \
  TL_STEEL_KERNEL_BK(gemm_##F##_nt_, 64, false, true, uchar,                 \
                     TL_READ_Q(read_##F##_), float, 32)                      \
  TL_STEEL_KERNEL_BK(gemm_##F##_nt32_, 32, false, true, uchar,               \
                     TL_READ_Q(read_##F##_), float, 32)
TL_GEMM_Q(q4k)
TL_GEMM_Q(q5k)
TL_GEMM_Q(q6k)
TL_GEMM_Q(q80)
TL_GEMM_Q(q50)
TL_GEMM_Q(q40)
TL_GEMM_Q(q41)
#undef TL_GEMM_Q

// ---- one row of a table into f32, the row the f32 index idx[0] names (a
// decode step's input, from the token the last step's argmax_to left; an
// index past the table gives zeros). The strided readers take f32 or bf16 in
// either layout, the block formats their rows as the GEMVs read them.
struct read_f32_s_ {
  device const float* B;
  uint rs, cs;
  float operator()(uint n, uint k) const { return B[(ulong)n * rs + (ulong)k * cs]; }
};
struct read_bf16_s_ {
  device const ushort* B;
  uint rs, cs;
  float operator()(uint n, uint k) const { return widen_(B[(ulong)n * rs + (ulong)k * cs]); }
};
struct embed_row_params {
  uint K, rows, rs, cs;
  float scale;  // each value times it (Gemma's embedding; 1 unchanged)
};
#define TL_EMBED_ROW(name, T, READER)                                        \
  kernel void name(device const T* B [[buffer(0)]],                          \
                   device const float* idx [[buffer(1)]],                    \
                   device float* out [[buffer(2)]],                          \
                   constant embed_row_params& p [[buffer(3)]],               \
                   uint j [[thread_position_in_grid]]) {                     \
    if (j >= p.K) return;                                                    \
    const uint n = uint(idx[0] + 0.5f);                                      \
    out[j] = n < p.rows ? READER(n, j) * p.scale : 0.0f;                     \
  }
TL_EMBED_ROW(embed_row_f32_, float, (read_f32_s_{B, p.rs, p.cs}))
TL_EMBED_ROW(embed_row_bf16_, ushort, (read_bf16_s_{B, p.rs, p.cs}))
TL_EMBED_ROW(embed_row_q4k_, uchar, TL_READ_Q(read_q4k_))
TL_EMBED_ROW(embed_row_q5k_, uchar, TL_READ_Q(read_q5k_))
TL_EMBED_ROW(embed_row_q6k_, uchar, TL_READ_Q(read_q6k_))
TL_EMBED_ROW(embed_row_q80_, uchar, TL_READ_Q(read_q80_))
TL_EMBED_ROW(embed_row_q50_, uchar, TL_READ_Q(read_q50_))
TL_EMBED_ROW(embed_row_q40_, uchar, TL_READ_Q(read_q40_))
TL_EMBED_ROW(embed_row_q41_, uchar, TL_READ_Q(read_q41_))
#undef TL_EMBED_ROW
#undef TL_READ_Q
#undef TL_READ_BF16

// The decode GEMVs over the same rows: cuda's tl_gemv_q4k_row / _q5k_row /
// _q6k_row / _q80_row / _q50_row / _q40_row / _q41_row in what they compute,
// llama.cpp's Metal mul_mv in shape: a
// simdgroup takes QROWS rows (threadgroups of any number of simdgroups, the
// host's gemv_quant_rows_ in all), holds the activations a lane multiplies
// once for all of them, and folds the weights it dequantizes straight into
// the rows' dot products, reduced by simd_sum. One row a threadgroup left
// these short rows (a 1024-wide Q4_K row is four blocks) to the launch and
// the reduction, at under a third of the bf16 GEMV's bandwidth. Each format
// is an accumulate over its blocks (F, with its reader's geometry as F::L);
// the activation is on 16 bytes (the host checks) for the float4 loads of
// Q6_K's and the 32-weight formats'.
constant constexpr uint QROWS = 2;

// A simdgroup's QROWS rows of W: row r's block ib. A ragged last simdgroup
// re-reads row n - 1 for the rows past n, which it does not store.
template <class L>
struct gemv_q_rows_ {
  device const uchar* W;
  uint row0, last;
  ulong rb;
  device const uchar* block(uint r, uint ib) const {
    return W + min(row0 + r, last) * rb + (ulong)ib * L::BYTES;
  }
};

template <class F>
kernel void gemv_q_(device const float* a       [[buffer(0)]],
                    device const uchar* W       [[buffer(1)]],
                    device float* y             [[buffer(2)]],
                    constant gemv_row_params& p [[buffer(3)]],
                    uint tg   [[threadgroup_position_in_grid]],
                    uint sgid [[simdgroup_index_in_threadgroup]],
                    uint nsg  [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  using L = typename F::L;
  const uint row0 = (tg * nsg + sgid) * QROWS;
  if (row0 >= p.n) return;
  const uint nb = p.k / L::ELEMS;
  float acc[QROWS] = {};
  F::accumulate(acc, a, gemv_q_rows_<L>{W, row0, p.n - 1, (ulong)nb * L::BYTES}, nb,
                lane);
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < QROWS; r++) {
    const float t = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < p.n) y[row0 + r] = t;
  }
}

// Q4_K, four blocks at a time, llama.cpp's kernel_mul_mv_q4_K_f32: lane =
// 8 * blk + 4 * iq + ir reads, of pairs iq and iq + 2 (64 weights each), the
// 8 bytes at 8 * ir: their low nibbles are weights 8 * ir.. of the pair's
// first group, the high ones of its second. A nibble is masked in place, not
// shifted down (u & 0x0F00 is 256 q), and the powers of 16 come off with the
// scales. y_n = sum d*sc*q*a - dmin*m*a. Q5_K reads its blocks the same way.
//
// The 32 activations the lane multiplies in block ib: al the weights 8 * ir..
// of groups 2iq and 2iq + 1 (0..7, 8..15), ah those of groups 2iq + 4 and
// 2iq + 5; and their four sums, a group each.
static inline float4 k4_acts_(device const float* a, uint ib, uint iq, uint ir,
                              thread float* al, thread float* ah) {
  device const float* ap = a + (ulong)ib * 256u + 64u * iq + 8u * ir;
  float4 suma = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < 8; i++) {
    al[i] = ap[i];
    al[i + 8] = ap[i + 32];
    ah[i] = ap[i + 128];
    ah[i + 8] = ap[i + 160];
    suma += float4(al[i], al[i + 8], ah[i], ah[i + 8]);
  }
  return suma;
}
// The 6-bit scales and mins of those four groups from the 12 scale bytes at
// b + 4, a byte each as sc16 holds them: the scales of groups 2iq, 2iq + 1 at
// 0, 1 and their mins at 2, 3; groups 2iq + 4, 2iq + 5's at 4, 5 and 6, 7.
// ggml's get_scale_min_k4, two groups a ushort.
static inline void k4_scales_(device const uchar* b, uint iq, thread ushort* sc16) {
  device const ushort* sc = reinterpret_cast<device const ushort*>(b + 4) + iq;
  sc16[0] = sc[0] & 0x3f3f;
  sc16[1] = sc[2] & 0x3f3f;
  sc16[2] = (sc[4] & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
  sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
}
// The lane's quants (Q4_K's at b + 16, Q5_K's at b + 48) times those
// activations, a nibble masked in place: acc1 over groups 2iq and 2iq + 1,
// acc2 over 2iq + 4 and 2iq + 5, their components 1, 256, 16 and 4096 times
// the sums of q a.
static inline void k4_nibbles_(device const uchar* qs, uint iq, uint ir, thread const float* al,
                               thread const float* ah, thread float4& acc1, thread float4& acc2) {
  device const ushort* q1 = reinterpret_cast<device const ushort*>(qs) + 16 * iq + 4 * ir;
  device const ushort* q2 = q1 + 32;
  acc1 = 0.0f;
  acc2 = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (uint i = 0; i < 4; i++) {
    const ushort u = q1[i], v = q2[i];
    acc1 += float4(al[2 * i], al[2 * i + 1], al[2 * i + 8], al[2 * i + 9]) *
            float4(u & 0x000F, u & 0x0F00, u & 0x00F0, u & 0xF000);
    acc2 += float4(ah[2 * i], ah[2 * i + 1], ah[2 * i + 8], ah[2 * i + 9]) *
            float4(v & 0x000F, v & 0x0F00, v & 0x00F0, v & 0xF000);
  }
}
struct gemv_q4k_dot_ {
  using L = read_q4k_;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint it = lane & 7u, iq = it >> 2, ir = it & 3u;
    float al[16], ah[16];
    for (uint ib = lane >> 3; ib < nb; ib += 4) {
      const float4 suma = k4_acts_(a, ib, iq, ir, al, ah);
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        ushort sc16[4];
        k4_scales_(b, iq, sc16);
        thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
        float4 acc1, acc2;
        k4_nibbles_(b + 16, iq, ir, al, ah, acc1, acc2);
        acc[r] += half_at_(b) * ((acc1[0] + acc1[1] / 256.0f) * sc8[0] +
                                 (acc1[2] + acc1[3] / 256.0f) * sc8[1] / 16.0f +
                                 (acc2[0] + acc2[1] / 256.0f) * sc8[4] +
                                 (acc2[2] + acc2[3] / 256.0f) * sc8[5] / 16.0f) -
                  half_at_(b + 2) * dot(suma, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
      }
    }
  }
};

// Q5_K, as Q4_K's (its quants at byte 48), and the fifth bits, llama.cpp's
// kernel_mul_mv_q5_K_f32: bits 2iq, 2iq + 1, 2iq + 4 and 2iq + 5 of qh byte
// 8 * ir + i are weight 8 * ir + i's in the four groups. A set bit is 16 more
// q, so a group adds 16 times the activations its set bits meet.
struct gemv_q5k_dot_ {
  using L = read_q5k_;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint it = lane & 7u, iq = it >> 2, ir = it & 3u;
    float al[16], ah[16];
    for (uint ib = lane >> 3; ib < nb; ib += 4) {
      const float4 suma = k4_acts_(a, ib, iq, ir, al, ah);
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        ushort sc16[4];
        k4_scales_(b, iq, sc16);
        thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
        float4 acc1, acc2, hi = 0.0f;
        k4_nibbles_(b + 48, iq, ir, al, ah, acc1, acc2);
        // The qh bytes 8 * ir.. as two words (the block 4-byte aligned:
        // quant_ok_), shifted so bits 0, 1, 4 and 5 of each byte are its
        // four groups'.
        device const uint* qh = reinterpret_cast<device const uint*>(b + 16 + 8 * ir);
        const uint h0 = qh[0] >> (2u * iq), h1 = qh[1] >> (2u * iq);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 8; i++) {
          const uint h = (i < 4 ? h0 : h1) >> (8u * (i & 3u));
          hi += float4((h & 1u) ? al[i] : 0.0f, (h & 2u) ? al[i + 8] : 0.0f,
                       (h & 16u) ? ah[i] : 0.0f, (h & 32u) ? ah[i + 8] : 0.0f);
        }
        acc[r] += half_at_(b) * ((acc1[0] + acc1[1] / 256.0f + 16.0f * hi[0]) * sc8[0] +
                                 ((acc1[2] + acc1[3] / 256.0f) / 16.0f + 16.0f * hi[1]) * sc8[1] +
                                 (acc2[0] + acc2[1] / 256.0f + 16.0f * hi[2]) * sc8[4] +
                                 ((acc2[2] + acc2[3] / 256.0f) / 16.0f + 16.0f * hi[3]) * sc8[5]) -
                  half_at_(b + 2) * dot(suma, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
      }
    }
  }
};

// Q6_K, two blocks at a time: lane = 16 * which + 8 * half + j reads the 4
// consecutive positions 4j.. of its 128-weight half in each of the four
// 32-weight quarters (4 + 4 bytes of ql, 4 of qh, 16 activations).
struct gemv_q6k_dot_ {
  using L = read_q6k_;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint h = (lane >> 3) & 1u, j = lane & 7u;
    for (uint ib = lane >> 4; ib < nb; ib += 2) {
      device const float4* ap = reinterpret_cast<device const float4*>(
          a + (ulong)ib * 256u + 128u * h + 4u * j);
      const float4 a0 = ap[0], a1 = ap[8], a2 = ap[16], a3 = ap[24];
      // The -32 offset, once for both rows: sum (q - 32) a = q.a - 32 sum a.
      const float4 o = 32.0f * float4(dot(a0, 1.0f), dot(a1, 1.0f), dot(a2, 1.0f),
                                      dot(a3, 1.0f));
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        const uint qa = ld32_h_(b + 64 * h + 4 * j);
        const uint qb = ld32_h_(b + 64 * h + 32 + 4 * j);
        const uint qh = ld32_h_(b + 128 + 32 * h + 4 * j);
        // Each quarter's four 6-bit values, a byte each, decoded at once.
        const uint lo = 0x0F0F0F0Fu;
        const float4 q0 = float4(as_type<uchar4>((qa & lo) | ((qh & 0x03030303u) << 4)));
        const float4 q1 = float4(as_type<uchar4>((qb & lo) | ((qh & 0x0C0C0C0Cu) << 2)));
        const float4 q2 = float4(as_type<uchar4>(((qa >> 4) & lo) | (qh & 0x30303030u)));
        const float4 q3 = float4(as_type<uchar4>(((qb >> 4) & lo) | ((qh >> 2) & 0x30303030u)));
        device const char* sc =
            reinterpret_cast<device const char*>(b + 192) + 8 * h + (j >> 2);
        acc[r] += half_at_(b + 208) *
                  (float(sc[0]) * (dot(q0, a0) - o[0]) + float(sc[2]) * (dot(q1, a1) - o[1]) +
                   float(sc[4]) * (dot(q2, a2) - o[2]) + float(sc[6]) * (dot(q3, a3) - o[3]));
      }
    }
  }
};

// Q8_0, eight blocks at a time: lane = 4 * which + j reads the 8 bytes 8j..
// of its block and the 8 activations they multiply. y_n = sum d*q*a.
struct gemv_q80_dot_ {
  using L = read_q80_;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint j = lane & 3u;
    for (uint ib = lane >> 2; ib < nb; ib += 8) {
      device const float4* ap =
          reinterpret_cast<device const float4*>(a + (ulong)ib * 32u + 8u * j);
      const float4 a0 = ap[0], a1 = ap[1];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        device const ushort* q16 = reinterpret_cast<device const ushort*>(b + 2 + 8 * j);
        float4 q0, q1;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 2; i++) {
          const uint w0 = q16[i], w1 = q16[2 + i];
          q0[2 * i] = float(char(w0 & 0xFFu));
          q0[2 * i + 1] = float(char(w0 >> 8));
          q1[2 * i] = float(char(w1 & 0xFFu));
          q1[2 * i + 1] = float(char(w1 >> 8));
        }
        acc[r] += half_at_(b) * (dot(q0, a0) + dot(q1, a1));
      }
    }
  }
};

// Q5_0, as Q8_0's: eight blocks at a time, lane = 4 * which + j taking the
// elements 8j.. of its block (q50_::vals8) and the 8 activations they
// multiply. y_n = sum d*(q - 16)*a.
struct gemv_q50_dot_ {
  using L = read_q50_;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint j = lane & 3u;
    for (uint ib = lane >> 2; ib < nb; ib += 8) {
      device const float4* ap =
          reinterpret_cast<device const float4*>(a + (ulong)ib * 32u + 8u * j);
      const float4 a0 = ap[0], a1 = ap[1];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        float q[8];
        q50_::vals8(b, 8u * j, q);
        acc[r] += half_at_(b) * (dot(float4(q[0], q[1], q[2], q[3]), a0) +
                                 dot(float4(q[4], q[5], q[6], q[7]), a1));
      }
    }
  }
};
// Q4_0 and Q4_1, sixteen blocks at a time, llama.cpp's kernel_mul_mv_q4_0_f32:
// lane = 2 * which + h reads the 8 nibble bytes 8h.. of its block, whose low
// nibbles are elements 8h.. and high ones 8h + 16... A nibble is masked in place,
// not shifted down (u & 0x0F00 is 256 q), and the powers of 16 come off the
// activations. y_n = sum d*q*a + m * sum a: m is Q4_1's min (Min: the half after
// d, the nibbles after it) or Q4_0's -8d.
template <class Reader, bool Min>
struct gemv_n32_dot_ {
  using L = Reader;
  static void accumulate(thread float* acc, device const float* a, gemv_q_rows_<L> w,
                         uint nb, uint lane) {
    const uint h = lane & 1u;
    const float4 lo = float4(1.0f, 1.0f / 256.0f, 1.0f, 1.0f / 256.0f), hi = lo / 16.0f;
    for (uint ib = lane >> 1; ib < nb; ib += 16) {
      device const float4* ap =
          reinterpret_cast<device const float4*>(a + (ulong)ib * 32u + 8u * h);
      const float4 a0 = ap[0], a1 = ap[1], a2 = ap[4], a3 = ap[5];
      const float suma = dot(a0 + a1 + a2 + a3, 1.0f);
      const float4 l0 = a0 * lo, l1 = a1 * lo, h0 = a2 * hi, h1 = a3 * hi;
      _Pragma("clang loop unroll(full)")
      for (uint r = 0; r < QROWS; r++) {
        device const uchar* b = w.block(r, ib);
        device const ushort* q16 =
            reinterpret_cast<device const ushort*>(b + (Min ? 4 : 2) + 8 * h);
        const uint u0 = q16[0], u1 = q16[1], u2 = q16[2], u3 = q16[3];
        const float s = dot(float4(u0 & 0x000Fu, u0 & 0x0F00u, u1 & 0x000Fu, u1 & 0x0F00u), l0) +
                        dot(float4(u2 & 0x000Fu, u2 & 0x0F00u, u3 & 0x000Fu, u3 & 0x0F00u), l1) +
                        dot(float4(u0 & 0x00F0u, u0 & 0xF000u, u1 & 0x00F0u, u1 & 0xF000u), h0) +
                        dot(float4(u2 & 0x00F0u, u2 & 0xF000u, u3 & 0x00F0u, u3 & 0xF000u), h1);
        const float d = half_at_(b);
        acc[r] += d * s + (Min ? half_at_(b + 2) : -8.0f * d) * suma;
      }
    }
  }
};
using gemv_q40_dot_ = gemv_n32_dot_<read_q40_, false>;
using gemv_q41_dot_ = gemv_n32_dot_<read_q41_, true>;

#define TL_GEMV_Q(name, F)                                                   \
  template [[host_name(name)]] kernel void gemv_q_<F>(                       \
      device const float*, device const uchar*, device float*,               \
      constant gemv_row_params&, uint, uint, uint, uint);
TL_GEMV_Q("gemv_q4k_", gemv_q4k_dot_)
TL_GEMV_Q("gemv_q5k_", gemv_q5k_dot_)
TL_GEMV_Q("gemv_q6k_", gemv_q6k_dot_)
TL_GEMV_Q("gemv_q80_", gemv_q80_dot_)
TL_GEMV_Q("gemv_q50_", gemv_q50_dot_)
TL_GEMV_Q("gemv_q40_", gemv_q40_dot_)
TL_GEMV_Q("gemv_q41_", gemv_q41_dot_)
#undef TL_GEMV_Q

// ---------------------------------------------------------------------------
// Fused decode attention: one query row a head against the cache, the online
// softmax of cuda's attn_dec_core in llama.cpp's Metal form
// (kernel_flash_attn_ext_vec). A threadgroup takes a kv head and G of the q
// heads that share it (GQA), and reads each of its keys once for all of them.
// Its simdgroups take the keys C at a time in turn. In a simdgroup NL lanes
// share a key, lane tx holding its float4s j * NL + tx, so NE = 32 / NL keys
// go at once, P passes a chunk; K and V rows are read together. A chunk's
// scores take one max and an exp each, and a lane keeps its weighted sums of
// V in registers across chunks: the lanes holding the same dims of different
// keys are summed once at the end, and the simdgroups merge through
// threadgroup memory. A slice's heads past the group's last run on zero
// queries and store nothing.
//
// Split-KV: a model with few heads leaves most of the GPU idle at one
// threadgroup a slice, so the keys are cut into chunks over the grid's y, each
// writing its own (m, l, acc) partial for attn_combine_ to merge.
//
// The shape, by head width (attn_dec_heads_, attn_dec_pass_ and the host's
// attn_decode_sg_), measured as the kernel's time a call at Llama 3.2 1B's,
// Qwen2.5 0.5B's, PLaMo 3 2B's, Gemma 3 1B's and Gemma 4 E2B's heads, 64 to
// 4096 cached keys of f16, against 1, 2, 4 and 8 heads, 2, 4 and 8 passes and
// 1 to 8 simdgroups: two heads, two passes and 4 simdgroups at 64, 128 and
// 512; one head, four passes and 8 simdgroups at 256, where a short window's
// few keys (Gemma's 512) want the threadgroups more than the reads. The form
// that took one q head a threadgroup and a key a step took Llama 3.2 1B's
// layer 160 us at 2048 keys; this one takes 28.
// ---------------------------------------------------------------------------

struct attn_decode_params {
  uint ctx, kv_stride, group, chunk;
  float scale;
  uint heads;  // q heads: the partials' layout
};

// q heads a threadgroup (the host's attn_dec_heads_, the same table), lanes a
// key, passes a chunk.
constexpr int attn_dec_heads_(int AD) { return AD == 256 ? 1 : 2; }
constexpr int attn_dec_nl_(int AD) { return AD >= 256 ? 32 : AD / 8; }
constexpr int attn_dec_pass_(int AD) { return AD == 256 ? 4 : 2; }

// Threadgroup `row` over keys [0, ctx) (FINAL: stores the output) or over
// split s's chunk (stores its partials). sh is sacc [nsg][G][AD] | sm, sl
// [nsg][G], sacc first so its float4s are 16-byte aligned at any shape.
template <int AD, typename KT, bool FINAL>
static inline void attn_dec_core_(device const float* q, device const KT* K,
                                  device const KT* V, device float* out,
                                  constant attn_decode_params& p,
                                  threadgroup float* sh, uint row, uint s,
                                  uint S, uint sgid, uint nsg, uint lane) {
  constexpr int G = attn_dec_heads_(AD), NL = attn_dec_nl_(AD), NE = 32 / NL;
  constexpr int NV = AD / (4 * NL), P = attn_dec_pass_(AD), C = NE * P;
  using KT4 = vec<KT, 4>;
  const uint slices = (p.group + G - 1) / G;
  const uint kv_h = row / slices, g0 = (row % slices) * G;  // g0: in the group
  const uint h0 = kv_h * p.group + g0, gn = min((uint)G, p.group - g0);  // gn: real heads
  const uint k0 = s * p.chunk, k1 = FINAL ? p.ctx : min(p.ctx, k0 + p.chunk);
  device const KT4* Kh = (device const KT4*)(K + (ulong)kv_h * p.kv_stride);
  device const KT4* Vh = (device const KT4*)(V + (ulong)kv_h * p.kv_stride);
  const uint tx = lane % NL, ty = lane / NL;

  float4 qr[G][NV], acc[G][NV];
  float m[G], l[G];
  _Pragma("clang loop unroll(full)")
  for (int g = 0; g < G; g++) {
    m[g] = -1e30f;
    l[g] = 0.0f;
    device const float4* qh = (device const float4*)(q + (ulong)(h0 + min((uint)g, gn - 1)) * AD);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NV; j++) {
      qr[g][j] = g < (int)gn ? qh[j * NL + tx] : 0.0f;
      acc[g][j] = 0.0f;
    }
  }

  for (uint c0 = k0 + sgid * C; c0 < k1; c0 += nsg * C) {
    // Key c0 + u * NE + ty, a key past the range reading the last one's row.
    float sc[P][G];
    KT4 vr[P][NV];
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < P; u++) {
      // A row within one head: its index fits 32 bits (a ulong one cost 10-17%).
      const uint i = min(c0 + (uint)(u * NE) + ty, k1 - 1) * (AD / 4);
      KT4 kr[NV];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NV; j++) {
        kr[j] = Kh[i + j * NL + tx];
        vr[u][j] = Vh[i + j * NL + tx];
      }
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < G; g++) {
        float d = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NV; j++) d += dot(qr[g][j], float4(kr[j]));
        sc[u][g] = d;
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int off = NL / 2; off > 0; off >>= 1)
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < P; u++)
        _Pragma("clang loop unroll(full)")
        for (int g = 0; g < G; g++) sc[u][g] += simd_shuffle_xor(sc[u][g], (ushort)off);
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < G; g++) {
      float mx = -INFINITY;
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < P; u++) {
        sc[u][g] = c0 + (uint)(u * NE) + ty < k1 ? sc[u][g] * p.scale : -INFINITY;
        mx = max(mx, sc[u][g]);
      }
      const float m_new = max(m[g], simd_max(mx)), corr = exp(m[g] - m_new);
      float ls = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NV; j++) acc[g][j] *= corr;
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < P; u++) {
        const float pr = exp(sc[u][g] - m_new);
        ls += pr;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < NV; j++) acc[g][j] += pr * float4(vr[u][j]);
      }
      l[g] = l[g] * corr + ls;
      m[g] = m_new;
    }
  }

  // The lanes holding the same dims (one tx) hold different keys' sums.
  _Pragma("clang loop unroll(full)")
  for (int off = NL; off < 32; off <<= 1)
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < G; g++) {
      l[g] += simd_shuffle_xor(l[g], (ushort)off);
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NV; j++) acc[g][j] += simd_shuffle_xor(acc[g][j], (ushort)off);
    }
  // Merge the simdgroups' states (one whose keys were none carries m = -1e30
  // and l = 0, so it adds nothing).
  threadgroup float4* sacc = (threadgroup float4*)sh;
  threadgroup float* sm = sh + nsg * G * AD;
  threadgroup float* sl = sm + nsg * G;
  if (lane == 0) {
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < G; g++) {
      sm[sgid * G + g] = m[g];
      sl[sgid * G + g] = l[g];
    }
  }
  if (ty == 0) {
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < G; g++)
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < NV; j++) sacc[(sgid * G + g) * (AD / 4) + j * NL + tx] = acc[g][j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup const float* sa = (threadgroup const float*)sacc;
  const uint hs = p.heads * S;
  for (uint e = sgid * 32 + lane; e < gn * AD; e += nsg * 32) {
    const uint g = e / AD, d = e % AD, h = h0 + g;
    float gm = -1e30f;
    for (uint w = 0; w < nsg; w++) gm = max(gm, sm[w * G + g]);
    float gl = 0.0f, o = 0.0f;
    for (uint w = 0; w < nsg; w++) {
      const float x = exp(sm[w * G + g] - gm);
      gl += sl[w * G + g] * x;
      o += sa[(w * G + g) * AD + d] * x;
    }
    if (FINAL) {
      out[(ulong)h * AD + d] = o / gl;
    } else {
      const uint at = h * S + s;
      if (d == 0) {
        out[at] = gm;
        out[hs + at] = gl;
      }
      out[2 * hs + (ulong)at * AD + d] = o;
    }
  }
}

template <int AD, typename KT>
kernel void attn_decode_(device const float* q          [[buffer(0)]],
                         device const KT* K             [[buffer(1)]],
                         device const KT* V             [[buffer(2)]],
                         device float* out              [[buffer(3)]],
                         constant attn_decode_params& p [[buffer(4)]],
                         threadgroup float* sh          [[threadgroup(0)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint nsg  [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  attn_dec_core_<AD, KT, true>(q, K, V, out, p, sh, row, 0, 1, sgid, nsg, lane);
}
// Every head width (64, 128, 256, 512) over either cache.
#define TL_ATTN_DECODE(name, AD, KT)                                          \
  template [[host_name(name)]] kernel void attn_decode_<AD, KT>(              \
      device const float*, device const KT*, device const KT*, device float*, \
      constant attn_decode_params&, threadgroup float*, uint, uint, uint, uint);
TL_ATTN_DECODE("attn_decode_64_", 64, float)
TL_ATTN_DECODE("attn_decode_128_", 128, float)
TL_ATTN_DECODE("attn_decode_256_", 256, float)
TL_ATTN_DECODE("attn_decode_512_", 512, float)
TL_ATTN_DECODE("attn_decode_64_f16_", 64, half)
TL_ATTN_DECODE("attn_decode_128_f16_", 128, half)
TL_ATTN_DECODE("attn_decode_256_f16_", 256, half)
TL_ATTN_DECODE("attn_decode_512_f16_", 512, half)
#undef TL_ATTN_DECODE

// One key chunk (grid y) of one row: partials pm[H*S] | pl[H*S] | pacc[H*S*AD]
// over the q heads, the layout attn_combine_ reads (cuda's attn_partials). The
// host cuts no chunk past the live count, so every chunk has keys.
template <int AD, typename KT>
kernel void attn_decode_split_(device const float* q          [[buffer(0)]],
                               device const KT* K             [[buffer(1)]],
                               device const KT* V             [[buffer(2)]],
                               device float* parts            [[buffer(3)]],
                               constant attn_decode_params& p [[buffer(4)]],
                               threadgroup float* sh          [[threadgroup(0)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               uint2 ntg [[threadgroups_per_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint nsg  [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  attn_dec_core_<AD, KT, false>(q, K, V, parts, p, sh, tgp.x, tgp.y, ntg.y, sgid, nsg, lane);
}
#define TL_ATTN_SPLIT(name, AD, KT)                                           \
  template [[host_name(name)]] kernel void attn_decode_split_<AD, KT>(        \
      device const float*, device const KT*, device const KT*, device float*, \
      constant attn_decode_params&, threadgroup float*, uint2, uint2, uint,   \
      uint, uint);
TL_ATTN_SPLIT("attn_decode_split_64_", 64, float)
TL_ATTN_SPLIT("attn_decode_split_128_", 128, float)
TL_ATTN_SPLIT("attn_decode_split_256_", 256, float)
TL_ATTN_SPLIT("attn_decode_split_512_", 512, float)
TL_ATTN_SPLIT("attn_decode_split_64_f16_", 64, half)
TL_ATTN_SPLIT("attn_decode_split_128_f16_", 128, half)
TL_ATTN_SPLIT("attn_decode_split_256_f16_", 256, half)
TL_ATTN_SPLIT("attn_decode_split_512_f16_", 512, half)
#undef TL_ATTN_SPLIT

// Merge one head's S partials, rescaling each by exp(m_s − max m).
struct attn_combine_params {
  uint splits;
};

template <int AD>
kernel void attn_combine_(device const float* parts        [[buffer(0)]],
                          device float* out                [[buffer(1)]],
                          constant attn_combine_params& p  [[buffer(2)]],
                          uint h   [[threadgroup_position_in_grid]],
                          uint ntg [[threadgroups_per_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const uint S = p.splits, hs = ntg * S;
  device const float* pm = parts + h * S;
  device const float* pl = parts + hs + h * S;
  device const float* pacc = parts + 2 * hs + (ulong)h * S * AD;
  float gm = -1e30f;
  for (uint s = 0; s < S; s++) gm = max(gm, pm[s]);
  float gl = 0.0f, o = 0.0f;
  for (uint s = 0; s < S; s++) {
    const float e = exp(pm[s] - gm);
    gl += pl[s] * e;
    o += pacc[(ulong)s * AD + tid] * e;
  }
  out[h * AD + tid] = o / gl;
}
#define TL_ATTN_COMBINE(name, AD)                                         \
  template [[host_name(name)]] kernel void attn_combine_<AD>(             \
      device const float*, device float*, constant attn_combine_params&, \
      uint, uint, uint);
TL_ATTN_COMBINE("attn_combine_64_", 64)
TL_ATTN_COMBINE("attn_combine_128_", 128)
TL_ATTN_COMBINE("attn_combine_256_", 256)
TL_ATTN_COMBINE("attn_combine_512_", 512)
#undef TL_ATTN_COMBINE

// ---------------------------------------------------------------------------
// Gated DeltaNet (Qwen3.5): its conv and its delta rule, cuda's tl_ssm_conv
// and tl_gated_delta (gpu_ops_nn.h has the math of both).
// ---------------------------------------------------------------------------

struct ssm_conv_params {
  uint C, K, T, ld;
};

// The causal depthwise conv over T tokens, then SiLU: token t reads row t of
// x (rows ld apart) and writes row t of out [T, C]; state holds the K-1 inputs
// before the first token [K-1, C] (oldest first), w the tap-major weights
// [K, C]. A thread a channel walks the tokens with the window in registers
// (K <= 8, which gpu::ssm_conv checks) and leaves the last K-1 inputs in its
// column of the state.
kernel void ssm_conv_(device const float* x      [[buffer(0)]],
                      device float* state        [[buffer(1)]],
                      device const float* w      [[buffer(2)]],
                      device float* out          [[buffer(3)]],
                      constant ssm_conv_params& p [[buffer(4)]],
                      uint c [[thread_position_in_grid]]) {
  if (c >= p.C) return;
  float win[7], wk[8];
  for (uint k = 0; k + 1 < p.K; k++) win[k] = state[(ulong)k * p.C + c];
  for (uint k = 0; k < p.K; k++) wk[k] = w[(ulong)k * p.C + c];
  for (uint t = 0; t < p.T; t++) {
    const float xc = x[(ulong)t * p.ld + c];
    float acc = xc * wk[p.K - 1];
    for (uint k = 0; k + 1 < p.K; k++) acc += win[k] * wk[k];
    out[(ulong)t * p.C + c] = silu_(acc);
    for (uint k = 0; k + 2 < p.K; k++) win[k] = win[k + 1];
    win[p.K - 2] = xc;
  }
  for (uint k = 0; k + 1 < p.K; k++) state[(ulong)k * p.C + c] = win[k];
}

struct gated_delta_params {
  uint H, dk, dv, T, ab_ld;
  float eps;
  uint Hk;
};

// The delta rule over T tokens: grid (ceil(dv/32), H), dk threads (a power of
// two, 32 to 1024). A threadgroup owns 32 columns of its head's state S
// [dk, dv] and 32-thread slice w the 32 rows from 32w of them, so each thread
// holds a 32-row slice of one column in registers for the whole call. A token
// stages its q and k (a thread an element), L2-normalizes them through one
// tree, and reduces k.S and q.S over the slices in threadgroup memory: 6 dk
// floats, sized here for the largest dk (sized for Qwen3.5's 128 instead, its
// decode on an M1 Pro moved by under 1%).
kernel void gated_delta_(device const float* qkv          [[buffer(0)]],
                         device const float* ab           [[buffer(1)]],
                         device const float* adt          [[buffer(2)]],
                         device float* state              [[buffer(3)]],
                         device float* out                [[buffer(4)]],
                         constant gated_delta_params& p   [[buffer(5)]],
                         uint2 tg   [[threadgroup_position_in_grid]],
                         uint t     [[thread_index_in_threadgroup]]) {
  constexpr uint R = 32;  // rows a 32-thread slice holds
  threadgroup float sm[6 * 1024];
  const uint dk = p.dk;
  // Slices of 32 threads, as cuda's warps: no simd intrinsic is used, so the
  // mapping does not depend on the simdgroup's width.
  const uint lane = t & 31u, w = t >> 5, nsg = dk >> 5;
  threadgroup float *qs = sm, *ks = sm + dk, *red = sm + 2 * dk;
  threadgroup float *pk = sm + 4 * dk, *po = sm + 5 * dk;
  const uint h = tg.y, j = tg.x * 32u + lane, i0 = w * R;
  const bool live = j < p.dv;
  const uint hk = h % p.Hk;  // the key head this value head reads
  const ulong nq = (ulong)p.Hk * dk, C = 2 * nq + (ulong)p.H * p.dv;
  const float A = adt[h], dt = adt[p.H + h];
  device float* S = state + (ulong)h * dk * p.dv;
  float Sr[R];
  for (uint r = 0; r < R; r++) Sr[r] = live ? S[(ulong)(i0 + r) * p.dv + j] : 0.0f;
  for (uint tok = 0; tok < p.T; tok++) {
    device const float* row = qkv + tok * C;
    const float a = row[(ulong)hk * dk + t], b = row[nq + (ulong)hk * dk + t];
    // The last token's reads of qs and ks all came before its final barrier,
    // and the tree writes only `red`.
    const float2 sums = tree_sum2_(red, t, dk, a * a, b * b);
    qs[t] = a * (1.0f / sqrt(sums.x + p.eps) / sqrt((float)dk));
    ks[t] = b * (1.0f / sqrt(sums.y + p.eps));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* abt = ab + (ulong)tok * p.ab_ld;
    const float x = abt[h] + dt;
    const float decay = exp(A * (x > 20.0f ? x : log(1.0f + exp(x))));
    const float beta = 1.0f / (1.0f + exp(-abt[p.H + h]));
    float kv = 0.0f;
    for (uint r = 0; r < R; r++) kv += Sr[r] * ks[i0 + r];
    pk[w * 32u + lane] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint ww = 0; ww < nsg; ww++) kv += pk[ww * 32u + lane];
    const float delta = live ? (row[2 * nq + (ulong)h * p.dv + j] - decay * kv) * beta : 0.0f;
    float o = 0.0f;
    for (uint r = 0; r < R; r++) {
      Sr[r] = Sr[r] * decay + ks[i0 + r] * delta;
      o += Sr[r] * qs[i0 + r];
    }
    po[w * 32u + lane] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0 && live) {
      o = 0.0f;
      for (uint ww = 0; ww < nsg; ww++) o += po[ww * 32u + lane];
      out[(ulong)tok * p.H * p.dv + (ulong)h * p.dv + j] = o;
    }
  }
  if (live) {
    for (uint r = 0; r < R; r++) S[(ulong)(i0 + r) * p.dv + j] = Sr[r];
  }
}

// ---------------------------------------------------------------------------
// The decode step's other kernels, cuda's tl_kv_append / tl_kv_fill /
// tl_argmax / tl_rmsnorm / tl_add_rmsnorm / tl_swiglu / tl_split_heads /
// tl_merge_heads: what a model runs between its GEMVs and its attention,
// on raw buffers so a step builds no graph. A row-wise kernel takes its row
// from the grid's y and serves a decode step (one row) and a batched prefill
// chunk (a row per token) alike.
// ---------------------------------------------------------------------------
// hout = xout · rsqrt(mean(xout²) + eps) · w per row, where xout is x itself
// or, with `add`, the residual sum x + delta written back on the way (a
// layer's residual add folded into the norm that follows it; xout may alias
// x). One threadgroup a row, 256 threads; 1/sqrt, as the array composition.
struct rmsnorm_params {
  uint n;
  float eps;
};

// hout = v * rsqrt(mean(v^2) + eps) * w per row, where v is x, or with ADD
// x + delta (also stored to xout): the residual add folded into the norm that
// follows it. Without ADD, delta and xout are never touched.
template <bool ADD>
static inline void rmsnorm_core_(device const float* x, device const float* delta,
                                 device const float* w, device float* xout,
                                 device float* hout, constant rmsnorm_params& p,
                                 threadgroup float* red, uint row, uint tid,
                                 uint nt, uint sgid, uint lane) {
  const uint base = row * p.n;
  float acc = 0.0f;
  for (uint i = tid; i < p.n; i += nt) {
    float v = x[base + i];
    if (ADD) {
      v += delta[base + i];
      xout[base + i] = v;
    }
    acc += v * v;
  }
  acc = simd_sum(acc);
  if (lane == 0) red[sgid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ss = 0.0f;
  for (uint s = 0; s < nt / 32; s++) ss += red[s];
  const float inv = 1.0f / sqrt(ss / float(p.n) + p.eps);
  device const float* src = ADD ? xout : x;
  for (uint i = tid; i < p.n; i += nt) hout[base + i] = src[base + i] * inv * w[i];
}

kernel void rmsnorm_(device const float* x         [[buffer(0)]],
                     device const float* w         [[buffer(1)]],
                     device float* out             [[buffer(2)]],
                     constant rmsnorm_params& p    [[buffer(3)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]],
                     uint nt   [[threads_per_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  rmsnorm_core_<false>(x, x, w, out, out, p, red, row, tid, nt, sgid, lane);
}

kernel void add_rmsnorm_(device const float* x         [[buffer(0)]],
                         device const float* delta     [[buffer(1)]],
                         device const float* w         [[buffer(2)]],
                         device float* xout            [[buffer(3)]],
                         device float* hout            [[buffer(4)]],
                         constant rmsnorm_params& p    [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint nt   [[threads_per_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  rmsnorm_core_<true>(x, delta, w, xout, hout, p, red, row, tid, nt, sgid, lane);
}

// rmsnorm_core_'s 1/sqrt(mean(x^2) + eps) of a row of n, taken by one
// simdgroup in the same order: acc[k] is what its thread lane + 32 k summed
// over elements lane + 32 k, + 256, ..., simd_sum gives simdgroup k's
// partial, and the eight add in order (cuda's rmsnorm_sum_warp_, which keeps
// its fusions bit for bit with the ops they replace). Fast math lets each
// kernel order its products its own way, so here the fused forms agree with
// rmsnorm_ and the ops after it to an ulp or so, not bit for bit.
static inline float rmsnorm_inv_lane_(device const float* row, uint n, float eps,
                                      uint lane) {
  float ss = 0.0f;
  for (uint k = 0; k < 8; k++) {
    float acc = 0.0f;
    for (uint i = lane + 32 * k; i < n; i += 256) acc += row[i] * row[i];
    ss += simd_sum(acc);
  }
  return 1.0f / sqrt(ss / float(n) + eps);
}

// Qwen3's q / k head rows RMS-normalized and then rotated, in place: rmsnorm_
// then rope_ in one pass, a simdgroup a row and eight a threadgroup, row r =
// head * T + token. D = 64, 128 or 256 and rot a multiple of 64, so a lane's
// values x[lane + 32 k] hold both halves of its rotation pairs (the host
// refuses other shapes, and gpu::rmsnorm_rope runs the two ops).
struct rmsnorm_rope_params {
  uint rows, T, pos;
  float base;
  uint rot;
  float eps;
  float pos_scale;   // rope_params's
  uint has_factors;
};
template <uint D>
kernel void rmsnorm_rope_(device float* x                  [[buffer(0)]],
                          device const float* w            [[buffer(1)]],
                          constant rmsnorm_rope_params& p  [[buffer(2)]],
                          device const float* factors      [[buffer(3)]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint sgid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint E = D / 32;
  const uint row = tg * 8 + sgid;
  if (row >= p.rows) return;
  device float* xr = x + (ulong)row * D;
  const float inv = rmsnorm_inv_lane_(xr, D, p.eps, lane);
  float v[E];
  for (uint k = 0; k < E; k++) {
    const uint d = lane + 32 * k;
    v[k] = xr[d] * inv * w[d];
  }
  const uint h = p.rot >> 1, P = h / 32;  // rotation pairs a lane
  const float position = float(p.pos + row % p.T) * p.pos_scale;  // rope_'s
  for (uint k = 0; k < E; k++) {
    const uint d = lane + 32 * k;
    if (d >= p.rot) xr[d] = v[k];
  }
  for (uint k = 0; k < P; k++) {
    const uint t = lane + 32 * k;
    float x0 = v[k], x1 = v[k + P];
    rope_pair_(position, t, p.rot, p.base, factors, p.has_factors, x0, x1);
    xr[t] = x0;
    xr[t + h] = x1;
  }
}
template [[host_name("rmsnorm_rope_64_")]] kernel void rmsnorm_rope_<64>(
    device float*, device const float*, constant rmsnorm_rope_params&, device const float*,
    uint, uint, uint);
template [[host_name("rmsnorm_rope_128_")]] kernel void rmsnorm_rope_<128>(
    device float*, device const float*, constant rmsnorm_rope_params&, device const float*,
    uint, uint, uint);
template [[host_name("rmsnorm_rope_256_")]] kernel void rmsnorm_rope_<256>(
    device float*, device const float*, constant rmsnorm_rope_params&, device const float*,
    uint, uint, uint);

// Qwen3.5's gated norm, silu(z) * (rmsnorm(o) * w), a simdgroup a row and
// eight a threadgroup: rmsnorm_ and swiglu_ in one pass, with their sum and
// their silu_. Row r = token * H + head of o [T*H, D]; its
// gate is row `token` of z (rows ldz apart), at head * D.
struct gated_rmsnorm_params {
  uint rows, H, D, ldz;
  float eps;
};
kernel void gated_rmsnorm_(device const float* o               [[buffer(0)]],
                           device const float* z               [[buffer(1)]],
                           device const float* w               [[buffer(2)]],
                           device float* out                   [[buffer(3)]],
                           constant gated_rmsnorm_params& p    [[buffer(4)]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * 8 + sgid;
  if (r >= p.rows) return;
  const uint t = r / p.H, h = r % p.H;
  device const float* orow = o + (ulong)r * p.D;
  device const float* zrow = z + (ulong)t * p.ldz + (ulong)h * p.D;
  const float inv = rmsnorm_inv_lane_(orow, p.D, p.eps, lane);
  for (uint d = lane; d < p.D; d += 32) {
    const float y = orow[d] * inv * w[d], g = zrow[d];
    out[(ulong)r * p.D + d] = silu_(g) * y;
  }
}

// out[row, f] = act(gate) · up out of the fused gate|up projection
// gu[row, 2·ff], act SiLU (swiglu_) or GELU's tanh form (geglu_): cuda's
// glu_core_. Grid (ceil(ff / 256), rows).
struct glu_params {
  uint ff;
};
struct silu_act_ {
  float operator()(float g) const { return silu_(g); }
};
struct gelu_tanh_act_ {
  float operator()(float g) const { return gelu_tanh_(g); }
};
template <class Act>
kernel void glu_(device const float* gu     [[buffer(0)]],
                 device float* out          [[buffer(1)]],
                 constant glu_params& p  [[buffer(2)]],
                 uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x, row = gid.y;
  if (f >= p.ff) return;
  const uint src = row * 2 * p.ff + f;
  out[row * p.ff + f] = Act{}(gu[src]) * gu[src + p.ff];
}
template [[host_name("swiglu_")]] kernel void glu_<silu_act_>(
    device const float*, device float*, constant glu_params&, uint2);
template [[host_name("geglu_")]] kernel void glu_<gelu_tanh_act_>(
    device const float*, device float*, constant glu_params&, uint2);

// Token-major [T, ld] -> head-major [H, T, D] with an optional per-head bias
// (`off` picks a column block of a fused projection, so q/k/v come from one
// GEMM); grid (H, T), D threads. merge_heads_ is its inverse.
struct heads_params {
  uint T, ld, off, has_bias;
};

kernel void split_heads_(device const float* src   [[buffer(0)]],
                         device const float* bias  [[buffer(1)]],
                         device float* dst         [[buffer(2)]],
                         constant heads_params& p  [[buffer(3)]],
                         uint2 g  [[threadgroup_position_in_grid]],
                         uint d   [[thread_index_in_threadgroup]],
                         uint2 nt [[threads_per_threadgroup]]) {
  const uint h = g.x, t = g.y, D = nt.x;
  float v = src[t * p.ld + p.off + h * D + d];
  if (p.has_bias) v += bias[h * D + d];
  dst[(h * p.T + t) * D + d] = v;
}

struct merge_heads_params {
  uint T, H, D;
};

kernel void merge_heads_(device const float* src   [[buffer(0)]],
                         device float* dst         [[buffer(1)]],
                         constant merge_heads_params& p [[buffer(2)]],
                         uint2 g  [[threadgroup_position_in_grid]],
                         uint2 ng [[threadgroups_per_grid]],
                         uint d   [[thread_index_in_threadgroup]],
                         uint2 nt [[threads_per_threadgroup]]) {
  const uint h = g.x, t = g.y, H = ng.x, D = nt.x;
  dst[t * H * D + h * D + d] = src[(h * p.T + t) * D + d];
}

// ---------------------------------------------------------------------------
// Cross-entropy and the optimizer: cuda's tl_gather_axis, tl_row_logsumexp,
// tl_xent_bwd and tl_adam_step, the same arithmetic.
// ---------------------------------------------------------------------------

// Row logsumexp in one pass: each thread carries a running (max, sum of exp
// below that max) over its columns and the tree merges the pairs.
kernel void row_logsumexp_(device const float* in [[buffer(0)]],
                           device float* out      [[buffer(1)]],
                           constant reduce_params& p [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float sm[T], ss[T];
  device const float* src = in + row * p.cols;
  float m = -FLT_MAX, s = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float v = src[c];
    if (v > m) {
      s *= exp(m - v);  // s is 0 on the first step, so -FLT_MAX is safe here
      m = v;
    }
    s += exp(v - m);
  }
  sm[lid] = m;
  ss[lid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = T / 2; h > 0; h >>= 1) {
    if (lid < h) {
      float m1 = sm[lid], s1 = ss[lid], m2 = sm[lid + h], s2 = ss[lid + h];
      float mm = max(m1, m2);
      ss[lid] = s1 * exp(m1 - mm) + s2 * exp(m2 - mm);
      sm[lid] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[row] = (sm[0] + log(ss[0])) * p.scale + p.offset;
}

struct xent_bwd_params {
  uint cols;
  uint n;
};

// dx[i,j] = g[i] · (exp(x[i,j] - lse[i]) - [j == target[i]]).
kernel void xent_bwd_(device const float* x   [[buffer(0)]],
                      device const float* lse [[buffer(1)]],
                      device const float* tgt [[buffer(2)]],
                      device const float* g   [[buffer(3)]],
                      device float* out       [[buffer(4)]],
                      constant xent_bwd_params& p [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint row = i / p.cols, col = i % p.cols;
  float e = exp(x[i] - lse[row]);
  if (col == uint(tgt[row] + 0.5f)) e -= 1.0f;
  out[i] = e * g[row];
}

struct adam_params {
  float b1, b2, eps, lr_over_bc1, inv_bc2;
  uint n;
};

// Adam's update in one pass: m and v advance, p moves by the bias-corrected
// ratio the host folded into lr_over_bc1 and inv_bc2.
kernel void adam_step_(device float* p_      [[buffer(0)]],
                       device float* m       [[buffer(1)]],
                       device float* v       [[buffer(2)]],
                       device const float* g [[buffer(3)]],
                       constant adam_params& p [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float gi = g[i];
  float mi = p.b1 * m[i] + (1.0f - p.b1) * gi;
  float vi = p.b2 * v[i] + (1.0f - p.b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p_[i] -= (mi * p.lr_over_bc1) / (sqrt(vi * p.inv_bc2) + p.eps);
}
