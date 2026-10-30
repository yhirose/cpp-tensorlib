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

// tree_sum_ for two values at once, in the same order; `scratch` holds 2·256.
static inline float2 tree_sum2_(threadgroup float* scratch, uint lid, float a,
                                float b) {
  constexpr uint T = 256;
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
  const float2 sums = tree_sum2_(scratch, lid, sg, sgx);
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
// (j, j+D/2) rotate by angle = position * base^(-2j/D). Dispatched flat
// over rows*(D/2), one invocation per (r, j) pair (CUDA instead grids by
// row, blocks by D/2 -- every kernel in this file is flat-1D).
struct rope_params {
  uint T, D, pos;
  uint half_;
  float base;
  uint n;
  uint has_bias;
};

// `bias` is [rows, D] like x, added before the rotation when has_bias: the
// decode step's q/k bias folded in (at T == 1 a [H, D] bias is that shape).
kernel void rope_(device const float* x [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant rope_params& p [[buffer(2)]],
                  device const float* bias [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint r = i / p.half_;
  uint j = i % p.half_;
  uint t = p.T > 0 ? (r % p.T) : 0u;
  float position = float(p.pos + t);
  float theta = pow(p.base, -2.0f * float(j) / float(p.D));
  float ang = position * theta;
  float c = cos(ang);
  float s = sin(ang);
  uint bi = r * p.D;
  float x0 = x[bi + j];
  float x1 = x[bi + j + p.half_];
  if (p.has_bias) {
    x0 += bias[bi + j];
    x1 += bias[bi + j + p.half_];
  }
  out[bi + j] = x0 * c - x1 * s;
  out[bi + j + p.half_] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// Causal prefill attention and its pullback: cuda's attn_prefill_tiled_core /
// attn_bwd_dq_core / attn_bwd_dkv_core, the same online softmax and causal
// rule, on simdgroup_matrix. A threadgroup is 4 simdgroups of 8 rows each
// (queries, or keys for dK/dV); the products are 8×8 MMAs and the softmax
// runs on each lane's own fragment elements, so scores never leave registers.
// Loops over fragments are fully unrolled: a fragment array indexed by a loop
// variable is spilled to memory, which costs more than the MMAs save.
// Tiles are 16 rows at D=128: two tiles of 32 rows would fill all 32 KB of
// threadgroup memory.
// ---------------------------------------------------------------------------

struct attn_params {
  uint T, kv_stride, group, pos0;
  float scale;
};

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

// q, out [H,T,D]; K/V a cache of kv_stride elements per kv head read over
// [0, pos0+T); query p sits at pos0+p. BK keys per tile. KT is the cache's
// element type (f32 is the one instantiated).
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
  const uint pabs = p.pos0 + qbase + sid * 8 + fm;
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const uint kmax = p.pos0 + last;
  const float sl2 = p.scale * M_LOG2E_F;
  float m = -1e30f, l = 0.0f;

  for (uint kt = 0; kt <= kmax; kt += BK) {
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

    float mt = -1e30f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const uint col = kt + j * 8 + fn + k;
        e[k] = col <= pabs ? e[k] * sl2 : -1e30f;  // causal mask
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
template [[host_name("attn_prefill_64_")]] kernel void
attn_prefill_<64, 32, float>(device const float*, device const float*,
                             device const float*, device float*,
                             constant attn_params&, uint2, uint2, uint, uint,
                             uint);
template [[host_name("attn_prefill_128_")]] kernel void
attn_prefill_<128, 16, float>(device const float*, device const float*,
                              device const float*, device float*,
                              constant attn_params&, uint2, uint2, uint, uint,
                              uint);

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
// The reduce is shared by the bf16 and int4 forms, which differ only in how a
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

// bf16 weights in [N,K]: cuda's tl_gemv_bf16_row. Eight contiguous halves per
// thread per step, so consecutive threads read consecutive 16-byte spans of
// the row. Requires k % 8 == 0 (every transformer dim; the host checks).
kernel void gemv_bf16_row_(device const float* a      [[buffer(0)]],
                           device const ushort* B     [[buffer(1)]],
                           device float* y            [[buffer(2)]],
                           constant gemv_row_params& p [[buffer(3)]],
                           uint row  [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint nt   [[threads_per_threadgroup]],
                           uint sgid [[simdgroup_index_in_threadgroup]],
                           uint nsg  [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const ushort* brow = B + (ulong)row * p.k;
  float acc = 0.0f;
  for (uint k0 = tid * 8; k0 < p.k; k0 += nt * 8) {
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; j++) acc += a[k0 + j] * widen_(brow[k0 + j]);
  }
  gemv_row_store_(acc, red, y + row, tid, sgid, nsg, lane);
}

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
// bf16-weight NT GEMM: C[M,N] = A[M,K] · B[N,K]ᵀ, cuda's tl_gemm_bf16_nt —
// the batched prefill's projection, where the same weight serves a whole
// chunk of prompt tokens at once.
//
// Its own body rather than a dtype parameter of sgemm_body_ because the
// operand layout is what differs: that one stages B as [k][n], and reading
// this [N,K] weight into it costs one strided device read per element. Here
// both operands stage in their own row-major order, so every device read is
// contiguous, and the B fragment is transposed on the way OUT of threadgroup
// memory, which simdgroup_load does for free. Tiles, MMA shape and the
// edge-tile store otherwise follow sgemm_body_.
// ---------------------------------------------------------------------------

struct gemm_nt_params {
  uint M, N, K;
};

template <uint BM, uint BN, uint BK>
void gemm_nt_body_(device const float* A, device const ushort* B,
                   device float* C, constant gemm_nt_params& p,
                   threadgroup float* As, threadgroup float* Bs, uint3 tgid,
                   uint tid, uint sid, uint lane) {
  constexpr uint N_SM = 2, N_SN = 2, TM = BM / N_SM, TN = BN / N_SN;
  constexpr uint FM = TM / 8, FN = TN / 8, THREADS = N_SM * N_SN * 32;
  constexpr uint S = BK + 4;  // both staged row-major in K; padding kills
                              // the bank conflicts of the fragment reads

  const uint wm = sid / N_SN, wn = sid % N_SN;
  const uint row0 = tgid.y * BM, col0 = tgid.x * BN;
  simdgroup_matrix<float, 8, 8> acc[FM][FN];
  for (uint i = 0; i < FM; i++)
    for (uint j = 0; j < FN; j++) acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    for (uint i = tid; i < BM * BK; i += THREADS) {
      const uint r = i / BK, c = i % BK, gr = row0 + r, gc = k0 + c;
      As[r * S + c] = (gr < p.M && gc < p.K) ? A[(ulong)gr * p.K + gc] : 0.0f;
    }
    for (uint i = tid; i < BN * BK; i += THREADS) {
      const uint r = i / BK, c = i % BK, gr = col0 + r, gc = k0 + c;
      Bs[r * S + c] =
          (gr < p.N && gc < p.K) ? widen_(B[(ulong)gr * p.K + gc]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<float, 8, 8> af[FM], bf[FN];
      for (uint i = 0; i < FM; i++)
        simdgroup_load(af[i], &As[(wm * TM + i * 8) * S + kk], S);
      // Bs holds [n][k]; the MMA wants [k][n], which the transposing load
      // produces without a second staging pass.
      for (uint j = 0; j < FN; j++)
        simdgroup_load(bf[j], &Bs[(wn * TN + j * 8) * S + kk], S, ulong2(0, 0),
                       true);
      for (uint i = 0; i < FM; i++)
        for (uint j = 0; j < FN; j++)
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    // Also guards the store below, which reuses As as scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = 0; i < FM; i++) {
    for (uint j = 0; j < FN; j++) {
      const uint r = row0 + wm * TM + i * 8, c = col0 + wn * TN + j * 8;
      if (r + 8 <= p.M && c + 8 <= p.N) {
        simdgroup_store(acc[i][j], C + (ulong)r * p.N + c, p.N);
      } else if (r < p.M && c < p.N) {
        threadgroup float* sc = As;  // per-simdgroup region, 4 × 64 floats
        simdgroup_store(acc[i][j], &sc[sid * 64], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
          const uint er = r + e / 8, ec = c + e % 8;
          if (er < p.M && ec < p.N) C[(ulong)er * p.N + ec] = sc[sid * 64 + e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

#define TL_GEMM_NT_KERNEL(name, BM, BN, BK)                                  \
  kernel void name(device const float* A [[buffer(0)]],                      \
                   device const ushort* B [[buffer(1)]],                     \
                   device float* C [[buffer(2)]],                            \
                   constant gemm_nt_params& p [[buffer(3)]],                 \
                   uint3 tgid [[threadgroup_position_in_grid]],              \
                   uint tid [[thread_index_in_threadgroup]],                 \
                   uint sid [[simdgroup_index_in_threadgroup]],              \
                   uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float As[BM * (BK + 4)];                                     \
    threadgroup float Bs[BN * (BK + 4)];                                     \
    gemm_nt_body_<BM, BN, BK>(A, B, C, p, As, Bs, tgid, tid, sid, lane);     \
  }

TL_GEMM_NT_KERNEL(gemm_bf16_nt_, 64, 64, 16)
// A prompt chunk is a few hundred rows, so a 64-row tile leaves most of the
// grid on one M block; the 32-row one doubles the M blocks for short chunks.
TL_GEMM_NT_KERNEL(gemm_bf16_nt32_, 32, 64, 16)
#undef TL_GEMM_NT_KERNEL

// ---------------------------------------------------------------------------
// Fused decode attention: one query row against the whole cache. cuda's
// attn_decode_core, the same online softmax and merge. A threadgroup owns one
// query head as AD threads, so a lane holds the output dims lane, lane + 32,
// …; each simdgroup walks every NW-th key, reduces the score with simd_sum,
// and the simdgroups' softmax states are merged through threadgroup memory.
//
// Split-KV: a model with few heads leaves most of the GPU idle at one
// threadgroup a head, so the keys are cut into chunks over the grid's y, each
// chunk writing its own (m, l, acc) partial for attn_combine_ to merge.
// ---------------------------------------------------------------------------

struct attn_decode_params {
  uint ctx, kv_stride, group, chunk;
  float scale;
};

// The softmax state of keys [k0, k1) for this threadgroup's head: `gm`, `gl`
// and this thread's output dim `o`. The threadgroup arrays are the caller's
// (threadgroup memory can only be declared in a kernel): qs [AD], sm/sl [NW]
// and sacc [NW][AD].
template <int AD, typename KT>
static inline void attn_span_(device const float* qh, device const KT* Kh,
                             device const KT* Vh, uint k0, uint k1,
                             float scale, threadgroup float* qs,
                             threadgroup float* sm, threadgroup float* sl,
                             threadgroup float* sacc, uint tid, uint sgid,
                             uint lane, thread float& gm, thread float& gl,
                             thread float& o) {
  constexpr int NW = AD / 32;
  qs[tid] = qh[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m = -1e30f, l = 0.0f, acc[NW];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NW; r++) acc[r] = 0.0f;
  for (uint i = k0 + sgid; i < k1; i += NW) {
    device const KT* Ki = Kh + i * AD;
    float dot = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NW; r++)
      dot += qs[lane + r * 32] * widen_(Ki[lane + r * 32]);
    const float sc = simd_sum(dot) * scale;
    const float m_new = max(m, sc);
    const float corr = exp(m - m_new), pr = exp(sc - m_new);
    l = l * corr + pr;
    device const KT* Vi = Vh + i * AD;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NW; r++)
      acc[r] = acc[r] * corr + pr * widen_(Vi[lane + r * 32]);
    m = m_new;
  }

  // Merge the simdgroups' states (one whose key range was empty carries
  // m = -1e30 and l = 0, so it contributes nothing).
  if (lane == 0) {
    sm[sgid] = m;
    sl[sgid] = l;
  }
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NW; r++) sacc[sgid * AD + lane + r * 32] = acc[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  gm = -1e30f;
  _Pragma("clang loop unroll(full)")
  for (int w = 0; w < NW; w++) gm = max(gm, sm[w]);
  gl = 0.0f;
  o = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int w = 0; w < NW; w++) {
    const float e = exp(sm[w] - gm);
    gl += sl[w] * e;
    o += sacc[w * AD + tid] * e;
  }
}

template <int AD, typename KT>
kernel void attn_decode_(device const float* q          [[buffer(0)]],
                         device const KT* K             [[buffer(1)]],
                         device const KT* V             [[buffer(2)]],
                         device float* out              [[buffer(3)]],
                         constant attn_decode_params& p [[buffer(4)]],
                         uint h    [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NW = AD / 32;
  threadgroup float qs[AD], sm[NW], sl[NW], sacc[NW * AD];
  const uint kv_h = p.group ? h / p.group : h;
  float gm, gl, o;
  attn_span_<AD>(q + h * AD, K + kv_h * p.kv_stride, V + kv_h * p.kv_stride, 0,
                 p.ctx, p.scale, qs, sm, sl, sacc, tid, sgid, lane, gm, gl, o);
  out[h * AD + tid] = o / gl;
}
template [[host_name("attn_decode_64_")]] kernel void
attn_decode_<64, float>(device const float*, device const float*,
                        device const float*, device float*,
                        constant attn_decode_params&, uint, uint, uint, uint);
template [[host_name("attn_decode_128_")]] kernel void
attn_decode_<128, float>(device const float*, device const float*,
                         device const float*, device float*,
                         constant attn_decode_params&, uint, uint, uint, uint);

// One key chunk of one head: partials pm[H*S] | pl[H*S] | pacc[H*S*AD], the
// layout attn_combine_ reads (cuda's attn_partials).
template <int AD, typename KT>
kernel void attn_decode_split_(device const float* q          [[buffer(0)]],
                               device const KT* K             [[buffer(1)]],
                               device const KT* V             [[buffer(2)]],
                               device float* parts            [[buffer(3)]],
                               constant attn_decode_params& p [[buffer(4)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               uint2 ntg [[threadgroups_per_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NW = AD / 32;
  threadgroup float qs[AD], sm[NW], sl[NW], sacc[NW * AD];
  const uint h = tgp.x, s = tgp.y;
  const uint kv_h = p.group ? h / p.group : h;
  const uint k0 = s * p.chunk, k1 = min(p.ctx, k0 + p.chunk);
  float gm = -1e30f, gl = 0.0f, o = 0.0f;
  if (k0 < k1) {
    attn_span_<AD>(q + h * AD, K + kv_h * p.kv_stride, V + kv_h * p.kv_stride,
                   k0, k1, p.scale, qs, sm, sl, sacc, tid, sgid, lane, gm, gl,
                   o);
  }
  const uint hs = ntg.x * ntg.y, at = h * ntg.y + s;
  if (tid == 0) {
    parts[at] = gm;
    parts[hs + at] = gl;
  }
  parts[2 * hs + at * AD + tid] = o;
}
template [[host_name("attn_decode_split_64_")]] kernel void
attn_decode_split_<64, float>(device const float*, device const float*,
                              device const float*, device float*,
                              constant attn_decode_params&, uint2, uint2, uint,
                              uint, uint);
template [[host_name("attn_decode_split_128_")]] kernel void
attn_decode_split_<128, float>(device const float*, device const float*,
                               device const float*, device float*,
                               constant attn_decode_params&, uint2, uint2, uint,
                               uint, uint);

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
template [[host_name("attn_combine_64_")]] kernel void
attn_combine_<64>(device const float*, device float*,
                  constant attn_combine_params&, uint, uint, uint);
template [[host_name("attn_combine_128_")]] kernel void
attn_combine_<128>(device const float*, device float*,
                   constant attn_combine_params&, uint, uint, uint);

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

// out[row, f] = silu(gate) · up out of the fused gate|up projection
// gu[row, 2·ff]; grid (ceil(ff / 256), rows).
struct swiglu_params {
  uint ff;
};

kernel void swiglu_(device const float* gu     [[buffer(0)]],
                    device float* out          [[buffer(1)]],
                    constant swiglu_params& p  [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x, row = gid.y;
  if (f >= p.ff) return;
  const uint src = row * 2 * p.ff + f;
  const float g = gu[src];
  out[row * p.ff + f] = (g / (1.0f + exp(-g))) * gu[src + p.ff];
}

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
