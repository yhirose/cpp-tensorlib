#pragma once

// The nn layer's kernels (layers: gpu_abi.h): the fused ops over plain tensors
// -- norms and activations, the f16 activation, Gated DeltaNet, the head
// layouts, softmax and cross-entropy, Adam, layer norm, the GEMVs and GEMMs over
// bf16 / f16 / block-quantized weights, attention and its pullback, RoPE.
// Included by tensorlib_cuda.cu, after cuda_core.cuh.

extern "C" {
// ---- Row-wise fused RMSNorm / SwiGLU. One block per row (grid.x = rows), so
// the SAME kernel serves a decode step (rows == 1) and a batched prefill chunk
// (rows == prompt tokens). That identity is the point: prefill agreeing
// numerically with decode is what bench_qwen_prefill's check 0 asserts, and it
// would be a hand-maintained coincidence if these were two kernels. ----

// hout = xout * rsqrt(mean(xout^2)+eps) * w, per row, where xout is either x
// itself (tl_rmsnorm) or the residual sum a+b written back on the way through
// (tl_add_rmsnorm). Matches the array composition (x*x).mean(-1) ->
// 1/sqrt(ms+eps) -> *x *w exactly (f32, 1/sqrtf not the approximate rsqrtf) so
// greedy output is unchanged; kills the ~7 array-op launches per RMSNorm.
//
// The fused-add form folds a layer's residual add into the following norm (the
// o-proj->norm and mlp->next-input-norm seams), writing BOTH the residual sum
// (the next residual base) and its normalized form; xout may alias a. It is the
// same reduction either way, so ONE core serves both, with ADD a template
// parameter so the fused-add path compiles out of the plain kernel by
// construction rather than by trusting constant propagation.
}  // extern "C"
template <bool ADD>
__device__ __forceinline__ void rmsnorm_core(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ w, float* __restrict__ xout,
    float* __restrict__ hout, unsigned n, float eps, __half* __restrict__ h16 = nullptr) {
  extern __shared__ float sm[];
  const size_t base = (size_t)blockIdx.x * n;
  unsigned t = threadIdx.x, T = blockDim.x;
  float acc = 0.0f;
  for (unsigned i = t; i < n; i += T) {
    float v = a[base + i];
    if constexpr (ADD) {
      v += b[base + i];
      xout[base + i] = v;
    }
    acc += v * v;
  }
  sm[t] = acc;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) sm[t] += sm[t + s];
    __syncthreads();
  }
  float inv = 1.0f / sqrtf(sm[0] / (float)n + eps);
  for (unsigned i = t; i < n; i += T) {
    const float h = (ADD ? xout[base + i] : a[base + i]) * inv * w[i];
    if (hout) hout[base + i] = h;  // null: only the f16 copy is read
    if (h16) h16[base + i] = __float2half(h);  // the f16 GEMM's operand
  }
}
extern "C" __global__ void tl_rmsnorm(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ out, unsigned n,
                                      float eps) {
  rmsnorm_core<false>(x, nullptr, w, nullptr, out, n, eps);
}
extern "C" __global__ void tl_add_rmsnorm(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ w,
                                          float* __restrict__ xout,
                                          float* __restrict__ hout, unsigned n,
                                          float eps) {
  rmsnorm_core<true>(a, b, w, xout, hout, n, eps);
}
// The same norms writing an f16 copy of their output beside it (or instead of
// it, when the f32 pointer is null), for the GEMM that reads the activation as
// f16 (own::gemm_bf16_nt_f16acc_a16).
extern "C" __global__ void tl_rmsnorm_h16(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ out,
                                          __half* __restrict__ out16, unsigned n,
                                          float eps) {
  rmsnorm_core<false>(x, nullptr, w, nullptr, out, n, eps, out16);
}
extern "C" __global__ void tl_add_rmsnorm_h16(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              const float* __restrict__ w,
                                              float* __restrict__ xout,
                                              float* __restrict__ hout,
                                              __half* __restrict__ hout16,
                                              unsigned n, float eps) {
  rmsnorm_core<true>(a, b, w, xout, hout, n, eps, hout16);
}

extern "C" {
// SwiGLU straight out of the FUSED gate|up projection: gu is [rows, 2*ff] (row r
// holds gate(ff) then up(ff)), out is [rows, ff]. Both paths already produce
// that fused layout, so neither has to split it into two projections.
// out = silu(gate) * up; grid = (ceil(ff/256), rows).
extern "C" __global__ void tl_swiglu(const float* __restrict__ gu,
                                     float* __restrict__ out, unsigned ff) {
  unsigned f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= ff) return;
  const size_t src = (size_t)blockIdx.y * 2u * ff + f;
  float g = gu[src];
  out[(size_t)blockIdx.y * ff + f] = (g / (1.0f + expf(-g))) * gu[src + ff];
}
// tl_swiglu writing only f16: the activation only the down GEMM reads.
extern "C" __global__ void tl_swiglu_h16(const float* __restrict__ gu,
                                         __half* __restrict__ out, unsigned ff) {
  unsigned f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= ff) return;
  const size_t src = (size_t)blockIdx.y * 2u * ff + f;
  float g = gu[src];
  out[(size_t)blockIdx.y * ff + f] = __float2half((g / (1.0f + expf(-g))) * gu[src + ff]);
}
// Qwen3.5's gated norm of the DeltaNet output: out[t, h D + d] = silu(z) * o * w[d] /
// sqrt(mean_d o^2 + eps), o's rows (t H + h) D wide and contiguous, z's at t ldz + h D.
// One warp a row, so a short row is not a block of mostly idle threads, and bit for bit
// the composition it replaces (tl_rmsnorm, then tl_swiglu / tl_swiglu_h16): the squares
// are summed in rmsnorm_core's order on its 256 threads (thread t the elements t, t+256,
// ..., then a tree halving 256 -> 1), lane l standing in for threads l + 32 k, and the
// norm and the gate are their formulas. A chunked prefill of a repetitive prompt
// amplifies even a reordering of that sum (check_qwen's 96-token prompt moved 6x), so
// the fusion keeps the numbers rather than approximating them. Written as f32, or (h16,
// a constant at each call) f16 for the GEMM that reads it.
//
// rmsnorm_core's sum of squares on its 256 threads, taken by one warp: acc[k] is what
// thread lane + 32 k accumulated over its elements t, t + 256, ... (as rounded values:
// a square left to fuse into the tree's first add would round differently), the tree's
// levels 128, 64 and 32 pair a lane's own values and its levels 16 .. 1 cross the warp.
// An xor step leaves both lanes the same sum (an add commutes), so every lane ends
// with the tree's thread 0's.
__device__ __forceinline__ float rmsnorm_sum_warp_(float (&acc)[8]) {
#pragma unroll
  for (int k = 0; k < 4; k++) acc[k] += acc[k + 4];  // s = 128
#pragma unroll
  for (int k = 0; k < 2; k++) acc[k] += acc[k + 2];  // s = 64
  float ss = acc[0] + acc[1];                          // s = 32
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, s);  // s = 16 .. 1
  return ss;
}
__device__ __forceinline__ void gated_rmsnorm_core(const float* __restrict__ o,
                                                   const float* __restrict__ z,
                                                   const float* __restrict__ w, void* __restrict__ out,
                                                   unsigned rows, unsigned H, unsigned D,
                                                   unsigned ldz, float eps, bool h16) {
  const unsigned lane = threadIdx.x & 31u, r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
  const unsigned t = r / H, h = r % H;
  const float* orow = o + (size_t)r * D;
  const float* zrow = z + (size_t)t * ldz + (size_t)h * D;
  float acc[8];  // rmsnorm_core's thread lane + 32 k
#pragma unroll
  for (int k = 0; k < 8; k++) {
    acc[k] = 0.0f;
    for (unsigned i = lane + 32u * k; i < D; i += 256) acc[k] += orow[i] * orow[i];
  }
  const float inv = 1.0f / sqrtf(rmsnorm_sum_warp_(acc) / (float)D + eps);
  for (unsigned d = lane; d < D; d += 32) {
    const float y = orow[d] * inv * w[d], g = zrow[d];
    const float v = (g / (1.0f + expf(-g))) * y;
    if (h16) static_cast<__half*>(out)[(size_t)r * D + d] = __float2half(v);
    else static_cast<float*>(out)[(size_t)r * D + d] = v;
  }
}
extern "C" __global__ void tl_gated_rmsnorm(const float* __restrict__ o, const float* __restrict__ z,
                                            const float* __restrict__ w, float* __restrict__ out,
                                            unsigned rows, unsigned H, unsigned D, unsigned ldz,
                                            float eps) {
  gated_rmsnorm_core(o, z, w, out, rows, H, D, ldz, eps, false);
}
extern "C" __global__ void tl_gated_rmsnorm_h16(const float* __restrict__ o, const float* __restrict__ z,
                                                const float* __restrict__ w, __half* __restrict__ out,
                                                unsigned rows, unsigned H, unsigned D, unsigned ldz,
                                                float eps) {
  gated_rmsnorm_core(o, z, w, out, rows, H, D, ldz, eps, true);
}
// f32 -> f16, for an activation no fused producer rounds.
extern "C" __global__ void tl_cvt_h16(const float* __restrict__ src,
                                      __half* __restrict__ dst, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// Tree-sum one value per thread through the block's shared `s`, handing every
// thread the total; the trailing barrier lets a second sum reuse `s`.
__device__ __forceinline__ float tl_tree_sum_(float* s, unsigned t, unsigned T, float v) {
  s[t] = v;
  __syncthreads();
  for (unsigned h = T >> 1; h > 0; h >>= 1) {
    if (t < h) s[t] += s[t + h];
    __syncthreads();
  }
  const float total = s[0];
  __syncthreads();
  return total;
}

// tl_tree_sum_ for two values at once, in the same order; `s` holds 2·T.
__device__ __forceinline__ float2 tl_tree_sum2_(float* s, unsigned t, unsigned T,
                                                float a, float b) {
  s[t] = a;
  s[T + t] = b;
  __syncthreads();
  for (unsigned h = T >> 1; h > 0; h >>= 1) {
    if (t < h) {
      s[t] += s[t + h];
      s[T + t] += s[T + t + h];
    }
    __syncthreads();
  }
  const float2 total = make_float2(s[0], s[T]);
  __syncthreads();
  return total;
}

// Gated DeltaNet's causal depthwise conv over T tokens, then SiLU. Token t
// reads row t of x (rows ld apart) and writes row t of out [T, C]; state is
// the K-1 inputs before the first token [K-1, C] (oldest first), w the tap-
// major weights [K, C]. A thread per channel walks the tokens with the window
// in registers (K <= 8, which the launcher checks) and leaves the last K-1
// inputs in its own column of the state. grid = ceil(C/256).
extern "C" __global__ void tl_ssm_conv(const float* __restrict__ x,
                                       float* __restrict__ state,
                                       const float* __restrict__ w,
                                       float* __restrict__ out, unsigned C,
                                       unsigned K, unsigned T, unsigned ld) {
  const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float win[7], wk[8];
  for (unsigned k = 0; k + 1 < K; k++) win[k] = state[(size_t)k * C + c];
  for (unsigned k = 0; k < K; k++) wk[k] = w[(size_t)k * C + c];
  for (unsigned t = 0; t < T; t++) {
    const float xc = x[(size_t)t * ld + c];
    float acc = xc * wk[K - 1];
    for (unsigned k = 0; k + 1 < K; k++) acc += win[k] * wk[k];
    out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));
    for (unsigned k = 0; k + 2 < K; k++) win[k] = win[k + 1];
    win[K - 2] = xc;
  }
  for (unsigned k = 0; k + 1 < K; k++) state[(size_t)k * C + c] = win[k];
}

// The same conv over a prefill's tokens with a thread a (channel, token): each
// output reads its window from x and, for the first K-1 tokens, the state, in
// the order tl_ssm_conv sums it (so the outputs are the same bits). The state
// moves on in tl_ssm_conv_state, once every window has read the old one.
extern "C" __global__ void tl_ssm_conv_tokens(const float* __restrict__ x,
                                              const float* __restrict__ state,
                                              const float* __restrict__ w,
                                              float* __restrict__ out, unsigned C,
                                              unsigned K, unsigned T, unsigned ld) {
  const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (c >= C) return;
  // in(u): input u of the window, u < 0 reaching back into the state.
  auto in = [&](int u) {
    return u >= 0 ? x[(size_t)u * ld + c] : state[(size_t)((int)K - 1 + u) * C + c];
  };
  float acc = in((int)t) * w[(size_t)(K - 1) * C + c];
  for (unsigned k = 0; k + 1 < K; k++) acc += in((int)t - (int)(K - 1) + (int)k) * w[(size_t)k * C + c];
  out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));
}
// The window after the last token: the K-1 inputs before the next call, read
// whole before any is written (a short prompt keeps some of the old state).
extern "C" __global__ void tl_ssm_conv_state(const float* __restrict__ x,
                                             float* __restrict__ state, unsigned C,
                                             unsigned K, unsigned T, unsigned ld) {
  const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float keep[7];  // K <= 8, which the launcher checks
  for (unsigned k = 0; k + 1 < K; k++) {
    const int u = (int)T - (int)(K - 1) + (int)k;
    keep[k] = u >= 0 ? x[(size_t)u * ld + c] : state[(size_t)((int)K - 1 + u) * C + c];
  }
  for (unsigned k = 0; k + 1 < K; k++) state[(size_t)k * C + c] = keep[k];
}

// The gated delta rule over T tokens (gpu_ops.h gated_delta has the math).
// grid = (ceil(dv/32), H), blockDim = dk (a power of two, 32 to 1024): a block
// owns 32 columns of its head's state S [dk, dv], and warp w the 32 rows from
// 32w of them, so each thread holds a 32-row slice of one column in registers
// for the whole call — read once before the first token, written once after
// the last. A token stages its q and k (a thread an element), L2-normalizes
// them through one tree, and reduces k.S and q.S over the warps' slices in
// shared memory. Dynamic shared memory: 6*dk floats.
extern "C" __global__ void tl_gated_delta(const float* __restrict__ qkv,
                                          const float* __restrict__ ab,
                                          const float* __restrict__ adt,
                                          float* __restrict__ state,
                                          float* __restrict__ out, unsigned H,
                                          unsigned dk, unsigned dv, unsigned T,
                                          unsigned ab_ld, float eps, unsigned Hk) {
  constexpr int R = 32;  // rows a warp holds
  extern __shared__ float sm[];
  float *qs = sm, *ks = sm + dk, *red = sm + 2 * dk;  // red: 2*dk, the tree's
  float *pk = sm + 4 * dk, *po = sm + 5 * dk;         // [warps][32] partials
  const unsigned t = threadIdx.x, lane = t & 31u, w = t >> 5, NW = dk >> 5;
  const unsigned h = blockIdx.y, j = blockIdx.x * 32u + lane, i0 = w * R;
  const bool live = j < dv;
  const unsigned hk = h % Hk;  // the key head this value head reads
  const size_t nq = (size_t)Hk * dk, C = 2 * nq + (size_t)H * dv;
  const float A = adt[h], dt = adt[H + h];
  float* S = state + (size_t)h * dk * dv;
  float Sr[R];
#pragma unroll
  for (int r = 0; r < R; r++) Sr[r] = live ? S[(size_t)(i0 + r) * dv + j] : 0.0f;
  for (unsigned tok = 0; tok < T; tok++) {
    const float* row = qkv + tok * C;
    const float a = row[(size_t)hk * dk + t], b = row[nq + (size_t)hk * dk + t];
    __syncthreads();  // the previous token is done reading qs / ks
    qs[t] = a;
    ks[t] = b;
    const float2 sums = tl_tree_sum2_(red, t, dk, a * a, b * b);
    qs[t] = a * (1.0f / sqrtf(sums.x + eps) / sqrtf((float)dk));
    ks[t] = b * (1.0f / sqrtf(sums.y + eps));
    __syncthreads();
    const float* abt = ab + (size_t)tok * ab_ld;
    const float x = abt[h] + dt;
    const float decay = expf(A * (x > 20.0f ? x : log1pf(expf(x))));
    const float beta = 1.0f / (1.0f + expf(-abt[H + h]));
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++) kv += Sr[r] * ks[i0 + r];
    pk[w * 32u + lane] = kv;
    __syncthreads();
    kv = 0.0f;
    for (unsigned ww = 0; ww < NW; ww++) kv += pk[ww * 32u + lane];
    const float delta = live ? (row[2 * nq + (size_t)h * dv + j] - decay * kv) * beta : 0.0f;
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++) {
      Sr[r] = Sr[r] * decay + ks[i0 + r] * delta;
      o += Sr[r] * qs[i0 + r];
    }
    po[w * 32u + lane] = o;
    __syncthreads();
    if (w == 0 && live) {
      o = 0.0f;
      for (unsigned ww = 0; ww < NW; ww++) o += po[ww * 32u + lane];
      out[(size_t)tok * H * dv + (size_t)h * dv + j] = o;
    }
  }
  if (live) {
#pragma unroll
    for (int r = 0; r < R; r++) S[(size_t)(i0 + r) * dv + j] = Sr[r];
  }
}

// ---- The gated delta rule in chunks (a prefill's T tokens; the math of
// gpu_ops.h gated_delta, regrouped). Within a chunk of C tokens with g_t the
// cumulative log-decay from its start, each token's correction
// d_t = beta_t (v_t - k_t . S'_t) solves (I + A) D = beta V - beta e^g K h, h
// the state entering the chunk and A_ts = beta_t e^(g_t - g_s) k_t . k_s for
// s < t. With T = (I + A)^-1, W = T diag(beta e^g) K and U = T diag(beta) V:
//   D = U - W h,  o_t = e^(g_t) q_t h + sum_{s<=t} (q_t . k_s) e^(g_t - g_s) d_s,
//   h' = e^(g_C) h + sum_s e^(g_C - g_s) k_s d_s^T.
// Everything but h is the chunk's own, so a first kernel builds it for every
// (chunk, head) at once, and a second walks the chunks in order, a block a
// (head, 32 columns of the state): a few small matrix products a chunk where
// the per-token kernel had a dependent step a token. All f32; prep's products
// (K K^T, Q K^T, T K, T V) run on the tensor cores as 3xTF32, f32's accuracy.
#define TL_GDC 32  // tokens a chunk
#define TL_GDC_SCAN_R 8  // state rows a lane holds in the scan (cuda.h repeats it)
// An f32 split into two tf32s whose sum is the f32 (to ~2^-21), and the product
// of two such pairs on the tensor cores: three m16n8k8 mmas (the small terms
// first, the lo*lo one dropped), which keep f32's accuracy where one tf32 mma
// would not. A and B fragments as the f16 m16n8k16's, in 8-deep steps.
__device__ __forceinline__ void split_tf32_(float x, unsigned& hi, unsigned& lo) {
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(hi) : "f"(x));
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(lo) : "f"(x - __uint_as_float(hi)));
}
__device__ __forceinline__ void mma_tf32_(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma3_tf32_(float* c, const unsigned* ah, const unsigned* al, const unsigned* bh, const unsigned* bl) {
  mma_tf32_(c, al, bh[0], bh[1]);
  mma_tf32_(c, ah, bl[0], bl[1]);
  mma_tf32_(c, ah, bh[0], bh[1]);
}
TL_ALIGN(tl_gated_delta_chunk_prep, 4, 4, 4, 4, 4, 8, 8)  // W, U by float2
extern "C" __global__ void tl_gated_delta_chunk_prep(
    const float* __restrict__ qkv, const float* __restrict__ ab,
    const float* __restrict__ adt, float* __restrict__ Qn, float* __restrict__ Kn,
    float* __restrict__ W, float* __restrict__ U, float* __restrict__ P,
    float* __restrict__ G, unsigned H, unsigned dv, unsigned T, unsigned ab_ld,
    float eps, unsigned Hk) {
  constexpr int C = TL_GDC, DK = 128;
  // 256 threads: the K K^T / Q K^T tiles below are one per warp of eight. The rows
  // are padded by 4 so a fragment's 8 rows x 4 columns fall in 32 distinct banks.
  __shared__ float Qs[C][DK + 4], Ks[C][DK + 4];  // Qs later holds V
  __shared__ float Am[C][C + 1], Tm[C][C + 1];
  __shared__ float gs[C], bs[C], bg[C];  // bg = beta e^g
  const unsigned c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, NTH = blockDim.x;
  const unsigned warp = tid >> 5, lane = tid & 31, NW = NTH >> 5;
  const unsigned t0 = c * C, n = min((unsigned)C, T - t0);
  const size_t nq = (size_t)Hk * DK, Cq = 2 * nq + (size_t)H * dv;
  const size_t hk = h % Hk;         // the key head this value head reads
  const size_t hb = (size_t)h * T;  // this head's first row in the scratch
  // q and k, L2-normalized (q also over sqrt(DK)); rows past the chunk zero.
  for (unsigned t = warp; t < C; t += NW) {
    float a[DK / 32], b[DK / 32], sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int r = 0; r < DK / 32; r++) {
      const unsigned d = lane + 32 * r;
      a[r] = t < n ? qkv[(t0 + t) * Cq + hk * DK + d] : 0.0f;
      b[r] = t < n ? qkv[(t0 + t) * Cq + nq + hk * DK + d] : 0.0f;
      sq += a[r] * a[r];
      sk += b[r] * b[r];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, off);
      sk += __shfl_xor_sync(0xffffffffu, sk, off);
    }
    const float qn = 1.0f / sqrtf(sq + eps) / sqrtf((float)DK), kn = 1.0f / sqrtf(sk + eps);
#pragma unroll
    for (int r = 0; r < DK / 32; r++) {
      const unsigned d = lane + 32 * r;
      Qs[t][d] = a[r] * qn;
      Ks[t][d] = b[r] * kn;
      if (t < n) {
        Qn[(hb + t0 + t) * DK + d] = a[r] * qn;
        Kn[(hb + t0 + t) * DK + d] = b[r] * kn;
      }
    }
  }
  // The gates, then g's running sum over the chunk.
  if (tid < C) {
    float la = 0.0f, be = 0.0f;
    if (tid < n) {
      const float* abt = ab + (size_t)(t0 + tid) * ab_ld;
      const float x = abt[h] + adt[H + h];
      la = adt[h] * (x > 20.0f ? x : log1pf(expf(x)));
      be = 1.0f / (1.0f + expf(-abt[H + h]));
    }
    gs[tid] = la;
    bs[tid] = be;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.0f;
    for (int t = 0; t < C; t++) gs[t] = (acc += gs[t]);
  }
  __syncthreads();
  if (tid < n) G[hb + t0 + tid] = gs[tid];
  if (tid < C) bg[tid] = bs[tid] * expf(gs[tid]);
  // A (strictly lower) and P (lower with the diagonal): K K^T and Q K^T on the
  // tensor cores (3xTF32), a warp an m16n8 tile of both.
  {
    const unsigned g = lane >> 2, tq = lane & 3, i = warp >> 2, j = warp & 3;
    const unsigned r0 = 16 * i + g, r1 = r0 + 8, nn = 8 * j + g;
    float ckk[4] = {0, 0, 0, 0}, cqk[4] = {0, 0, 0, 0};
    for (unsigned k0 = 0; k0 < DK; k0 += 8) {
      unsigned bh[2], bl[2];
      split_tf32_(Ks[nn][k0 + tq], bh[0], bl[0]);
      split_tf32_(Ks[nn][k0 + tq + 4], bh[1], bl[1]);
      unsigned ah[4], al[4];
      split_tf32_(Ks[r0][k0 + tq], ah[0], al[0]);
      split_tf32_(Ks[r1][k0 + tq], ah[1], al[1]);
      split_tf32_(Ks[r0][k0 + tq + 4], ah[2], al[2]);
      split_tf32_(Ks[r1][k0 + tq + 4], ah[3], al[3]);
      mma3_tf32_(ckk, ah, al, bh, bl);
      split_tf32_(Qs[r0][k0 + tq], ah[0], al[0]);
      split_tf32_(Qs[r1][k0 + tq], ah[1], al[1]);
      split_tf32_(Qs[r0][k0 + tq + 4], ah[2], al[2]);
      split_tf32_(Qs[r1][k0 + tq + 4], ah[3], al[3]);
      mma3_tf32_(cqk, ah, al, bh, bl);
    }
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const unsigned t = (e < 2 ? r0 : r1), sc = 8 * j + 2 * tq + (e & 1);
      const float dec = sc <= t && t < n ? expf(gs[t] - gs[sc]) : 0.0f;
      const float kk = sc <= t && t < n ? ckk[e] : 0.0f, qk = sc <= t && t < n ? cqk[e] : 0.0f;
      Am[t][sc] = sc < t ? bs[t] * dec * kk : 0.0f;
      P[((size_t)h * gridDim.x + c) * C * C + t * C + sc] = qk * dec;
    }
  }
  __syncthreads();
  // T = (I + A)^-1 by forward substitution, a warp a pass: row t is e_t minus
  // A[t][s] times row s for every s < t.
  if (warp == 0) {
    for (unsigned t = 0; t < C; t++) {
      float v = lane == t ? 1.0f : 0.0f;
      for (unsigned sc = 0; sc < t; sc++) v -= Am[t][sc] * Tm[sc][lane];
      Tm[t][lane] = v;
      __syncwarp();
    }
  }
  // V into Qs, whose q is spent.
  for (unsigned e = tid; e < C * dv; e += NTH) {
    const unsigned t = e / dv, d = e % dv;
    Qs[t][d] = t < n ? qkv[(t0 + t) * Cq + 2 * nq + (size_t)h * dv + d] : 0.0f;
  }
  __syncthreads();
  // W = T diag(beta e^g) K and U = T diag(beta) V on the tensor cores.
  {
    const unsigned g = lane >> 2, tq = lane & 3;
    const unsigned nt = dv / 8;  // n8 tiles a matrix (dk = 128 >= dv)
    for (unsigned job = warp; job < 2 * 2 * nt; job += NW) {
      const unsigned mat = job / (2 * nt), i = (job / nt) & 1, nj = job % nt;
      const float* sc = mat ? bs : bg;
      const unsigned r0 = 16 * i + g, r1 = r0 + 8;
      float cc[4] = {0, 0, 0, 0};
      for (unsigned k0 = 0; k0 < C; k0 += 8) {
        unsigned ah[4], al[4], bh[2], bl[2];
        split_tf32_(Tm[r0][k0 + tq] * sc[k0 + tq], ah[0], al[0]);
        split_tf32_(Tm[r1][k0 + tq] * sc[k0 + tq], ah[1], al[1]);
        split_tf32_(Tm[r0][k0 + tq + 4] * sc[k0 + tq + 4], ah[2], al[2]);
        split_tf32_(Tm[r1][k0 + tq + 4] * sc[k0 + tq + 4], ah[3], al[3]);
        const float(*X)[DK + 4] = mat ? Qs : Ks;
        split_tf32_(X[k0 + tq][8 * nj + g], bh[0], bl[0]);
        split_tf32_(X[k0 + tq + 4][8 * nj + g], bh[1], bl[1]);
        mma3_tf32_(cc, ah, al, bh, bl);
      }
      float* dst = mat ? U : W;
      const unsigned ld = mat ? dv : DK, col = 8 * nj + 2 * tq;
      if (r0 < n) *reinterpret_cast<float2*>(dst + (hb + t0 + r0) * ld + col) = make_float2(cc[0], cc[1]);
      if (r1 < n) *reinterpret_cast<float2*>(dst + (hb + t0 + r1) * ld + col) = make_float2(cc[2], cc[3]);
    }
  }
}
// The chunks in order, a block a (head, 32 state columns): DK/R warps, warp w
// holding rows R*w.. of its lane's column in registers as the decode kernel
// does, and each chunk's W, q and k rows staged in turn through one buffer.
// Each chunk is a chain of dependent steps with a barrier between, so the time
// is the chain's latency, not the arithmetic: R = 8 puts 16 warps on the SM's
// four schedulers where R = 32 left one each (the scan took 260 us against 170
// at T = 512). The tile is dynamic shared memory, TL_GDC_SCAN_SMEM bytes; the
// launcher (own::gated_delta_chunked) must raise the limit and match R.
TL_ALIGN(tl_gated_delta_chunk_scan, 16, 16, 16)  // Qn, Kn, W by float4
extern "C" __global__ void tl_gated_delta_chunk_scan(
    const float* __restrict__ Qn, const float* __restrict__ Kn,
    const float* __restrict__ W, const float* __restrict__ U,
    const float* __restrict__ P, const float* __restrict__ G,
    float* __restrict__ state, float* __restrict__ out, unsigned H, unsigned dv,
    unsigned T) {
  constexpr int C = TL_GDC, DK = 128, R = TL_GDC_SCAN_R, NW = DK / R, NT = NW * 32;
  static_assert(C * C % NT == 0, "the P tile splits evenly over the block");
  extern __shared__ __align__(16) float smx[];
  float(*Buf)[DK] = reinterpret_cast<float(*)[DK]>(smx);
  float(*red)[C][32] = reinterpret_cast<float(*)[C][32]>(smx + C * DK);
  float(*Dl)[32] = reinterpret_cast<float(*)[32]>(smx + C * DK + NW * C * 32);
  float(*Pc)[C + 1] = reinterpret_cast<float(*)[C + 1]>(smx + C * DK + NW * C * 32 + C * 32);
  float* gc = smx + C * DK + NW * C * 32 + C * 32 + C * (C + 1);
  float* ed = gc + C;
  const unsigned h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const unsigned j = blockIdx.x * 32 + lane, i0 = w * R;
  const size_t hb = (size_t)h * T;
  const unsigned nch = (T + C - 1) / C;
  float* S = state + (size_t)h * DK * dv;
  float Sr[R];
#pragma unroll
  for (int r = 0; r < R; r++) Sr[r] = S[(size_t)(i0 + r) * dv + j];
  // A chunk's rows of W, q and k reach Buf in turn, but are all read into
  // registers at the chunk's start, 16 bytes a load, so the three latencies
  // overlap.
  constexpr int NV = C * DK / 4 / NT;  // float4s a thread a matrix
  auto load = [&](const float* src, unsigned t0, unsigned n, float4* v) {
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const unsigned e = (tid + k * NT) * 4, t = e / DK, d = e % DK;
      v[k] = t < n ? *reinterpret_cast<const float4*>(src + (hb + t0 + t) * DK + d)
                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  };
  auto put = [&](const float4* v) {
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const unsigned e = (tid + k * NT) * 4, t = e / DK, d = e % DK;
      *reinterpret_cast<float4*>(&Buf[t][d]) = v[k];
    }
  };
  // red[w][t][lane]: warp w's part of sum_i Buf[t][i] h[i][j], over its rows,
  // four tokens at a time so four sums are in flight (one warp a scheduler
  // has nothing else to hide a dependent chain behind).
  auto partial = [&]() {
    for (unsigned t = 0; t < C; t += 4) {
      float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int u = 0; u < 4; u++) p[u] += Buf[t + u][i0 + r] * Sr[r];
#pragma unroll
      for (int u = 0; u < 4; u++) red[w][t + u][lane] = p[u];
    }
  };
  // The next chunk's inputs are read into registers while this one runs its
  // update, so no chunk starts by waiting on global memory.
  constexpr int NP = C * C / NT;  // P cells a thread
  float4 vw[NV], vq[NV], vk[NV];
  float vp[NP], vg = 0.0f;
  auto fetch = [&](unsigned c) {
    const unsigned t0 = c * C, n = min((unsigned)C, T - t0);
    load(W, t0, n, vw);
    load(Qn, t0, n, vq);
    load(Kn, t0, n, vk);
#pragma unroll
    for (int k = 0; k < NP; k++) vp[k] = P[((size_t)h * nch + c) * C * C + tid + k * NT];
    if (tid < C) vg = tid < n ? G[hb + t0 + tid] : 0.0f;
  };
  fetch(0);
  for (unsigned c = 0; c < nch; c++) {
    const unsigned t0 = c * C, n = min((unsigned)C, T - t0);
    __syncthreads();  // the previous chunk is done with every buffer
#pragma unroll
    for (int k = 0; k < NP; k++) Pc[(tid + k * NT) / C][(tid + k * NT) % C] = vp[k];
    if (tid < C) gc[tid] = vg;
    put(vw);
    __syncthreads();
    if (tid < C) ed[tid] = tid < n ? expf(gc[n - 1] - gc[tid]) : 0.0f;
    partial();
    __syncthreads();
    // D = U - W h; each warp finishes C/4 of the chunk's tokens.
    for (unsigned t = w * (C / NW); t < (w + 1) * (C / NW); t++) {
      float wh = 0.0f;
#pragma unroll
      for (int ww = 0; ww < DK / R; ww++) wh += red[ww][t][lane];
      Dl[t][lane] = t < n ? U[(hb + t0 + t) * dv + j] - wh : 0.0f;
    }
    __syncthreads();
    put(vq);
    __syncthreads();
    partial();
    __syncthreads();
    // o_t = e^(g_t) q_t h + sum_{s<=t} P[t][s] d_s.
    for (unsigned t = w * (C / NW); t < (w + 1) * (C / NW) && t < n; t++) {
      float qh = 0.0f;
#pragma unroll
      for (int ww = 0; ww < DK / R; ww++) qh += red[ww][t][lane];
      float o = expf(gc[t]) * qh;
      for (unsigned sc = 0; sc <= t; sc++) o += Pc[t][sc] * Dl[sc][lane];
      out[(size_t)(t0 + t) * H * dv + (size_t)h * dv + j] = o;
    }
    __syncthreads();  // every warp is done with q's rows
    put(vk);
    __syncthreads();
    if (c + 1 < nch) fetch(c + 1);  // in flight through the update below
    // h' = e^(g_C) h + sum_t e^(g_C - g_t) k_t d_t^T.
    // The token loop outside: the 32 rows are 32 independent sums.
    const float eC = expf(gc[n - 1]);
#pragma unroll
    for (int r = 0; r < R; r++) Sr[r] *= eC;
    for (unsigned t = 0; t < n; t++) {
      const float cf = ed[t] * Dl[t][lane];
#pragma unroll
      for (int r = 0; r < R; r++) Sr[r] += cf * Buf[t][i0 + r];
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) S[(size_t)(i0 + r) * dv + j] = Sr[r];
}

// Token-major [T, ld] -> head-major [H, T, D], adding an optional per-head bias.
// The batched projections produce all heads of a token contiguously, while the
// attention kernels want each head's token sequence contiguous; `off` selects a
// column block of the fused QKV output, so q/k/v come from one GEMM. Folding the
// bias in here is what lets the following rope run bias-free (its fused-bias form
// only indexes correctly at T==1). grid = (H, T), block = D.
extern "C" __global__ void tl_split_heads(const float* __restrict__ src,
                                          const float* __restrict__ bias,
                                          float* __restrict__ dst, unsigned T,
                                          unsigned ld, unsigned off,
                                          unsigned D) {
  const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  float v = src[(size_t)t * ld + off + h * D + d];
  if (bias) v += bias[(size_t)h * D + d];
  dst[((size_t)h * T + t) * D + d] = v;
}

// Head-major [H, T, D] -> token-major [T, H*D], the inverse of tl_split_heads
// (attention output back into the shape the o-projection GEMM reads).
extern "C" __global__ void tl_merge_heads(const float* __restrict__ src,
                                          float* __restrict__ dst, unsigned T,
                                          unsigned H, unsigned D) {
  const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  dst[(size_t)t * H * D + h * D + d] = src[((size_t)h * T + t) * D + d];
}
extern "C" __global__ void tl_merge_heads_h16(const float* __restrict__ src,
                                              __half* __restrict__ dst, unsigned T,
                                              unsigned H, unsigned D) {
  const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  dst[(size_t)t * H * D + h * D + d] = __float2half(src[((size_t)h * T + t) * D + d]);
}

// ---- softmax over the last axis (rows×cols out); scale/offset ignored ----
// Numerically stable (subtract row max). Two shared reductions (max, sum).
__global__ void tl_softmax(const float* in, float* out, unsigned rows,
                           unsigned cols, float scale, float offset) {
  (void)scale;
  (void)offset;
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const float* src = in + (size_t)row * cols;
  float* dst = out + (size_t)row * cols;
  extern __shared__ float sdata[];
  unsigned t = threadIdx.x, T = blockDim.x;

  float m = -3.402823466e+38f;
  for (unsigned c = t; c < cols; c += T) m = fmaxf(m, src[c]);
  sdata[t] = m;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) sdata[t] = fmaxf(sdata[t], sdata[t + s]);
    __syncthreads();
  }
  float row_max = sdata[0];
  __syncthreads();

  float sum = 0.0f;
  for (unsigned c = t; c < cols; c += T) sum += expf(src[c] - row_max);
  sdata[t] = sum;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) sdata[t] += sdata[t + s];
    __syncthreads();
  }
  float inv = 1.0f / sdata[0];
  for (unsigned c = t; c < cols; c += T) dst[c] = expf(src[c] - row_max) * inv;
}

// ---- row logsumexp over the last axis (one value per row); affine epilogue.
// One block per row, ONE pass: each thread carries a running (max, sum-of-
// exp-below-that-max) pair for its own columns and the tree merges pairs, so
// the row is read once where tl_softmax above reads it three times and writes
// a whole row back. Cross-entropy's forward is this plus a gather.
__global__ void tl_row_logsumexp(const float* in, float* out, unsigned rows,
                                 unsigned cols, float scale, float offset) {
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const float* src = in + (size_t)row * cols;
  extern __shared__ float sdata[];
  unsigned t = threadIdx.x, T = blockDim.x;
  float* sm = sdata;      // each thread's running max
  float* ss = sdata + T;  // and its sum of exp(x - that max)

  float m = -3.402823466e+38f, s = 0.0f;
  for (unsigned c = t; c < cols; c += T) {
    float v = src[c];
    if (v > m) {
      s *= expf(m - v);  // s is 0 on the first step, so -FLT_MAX is safe here
      m = v;
    }
    s += expf(v - m);
  }
  sm[t] = m;
  ss[t] = s;
  __syncthreads();
  for (unsigned h = T >> 1; h > 0; h >>= 1) {
    if (t < h) {
      float m1 = sm[t], s1 = ss[t], m2 = sm[t + h], s2 = ss[t + h];
      float mm = fmaxf(m1, m2);
      ss[t] = s1 * expf(m1 - mm) + s2 * expf(m2 - mm);
      sm[t] = mm;
    }
    __syncthreads();
  }
  if (t == 0) out[row] = (sm[0] + logf(ss[0])) * scale + offset;
}

// ---- softmax cross-entropy's pullback, from the forward's row logsumexp:
// dx[i,j] = g[i] · (exp(x[i,j] - lse[i]) - [j == target[i]]). One read and one
// write of [rows, cols], where composing it re-derives the probabilities, then
// builds a one-hot, subtracts and scales -- four more passes over the same
// matrix. The forward keeps lse (one float per row) instead of those probs.
extern "C" __global__ void tl_xent_bwd(const float* __restrict__ x,
                                       const float* __restrict__ lse,
                                       const float* __restrict__ tgt,
                                       const float* __restrict__ g,
                                       float* __restrict__ out, unsigned cols,
                                       unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned row = i / cols, col = i % cols;
  float p = expf(x[i] - lse[row]);
  long long k = (long long)(tgt[row] + 0.5f);
  if ((long long)col == k) p -= 1.0f;
  out[i] = p * g[row];
}

// Adam's per-parameter update, fused: m and v advance, then p moves by the
// bias-corrected ratio. One read of g, one read-modify-write of m, v and p,
// where the composed form walks all four a dozen times and allocates a buffer
// for every intermediate. `lr_over_bc1` and `inv_bc2` fold the bias correction
// the host already computed, so the kernel spells the composition's arithmetic.
extern "C" __global__ void tl_adam_step(float* __restrict__ p,
                                        float* __restrict__ m,
                                        float* __restrict__ v,
                                        const float* __restrict__ g, float b1,
                                        float b2, float eps, float lr_over_bc1,
                                        float inv_bc2, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i];
  float mi = b1 * m[i] + (1.0f - b1) * gi;
  float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p[i] -= (mi * lr_over_bc1) / (sqrtf(vi * inv_bc2) + eps);
}

// A row's mean and 1/sqrt(biased variance + eps) for layer norm and its
// pullback: two shared sums (x, then the squared deviations), each scaled by
// 1/cols after the tree the way tl_row_sum's mean is, so both read the numbers
// the mean(-1, keepdims) composition does on this backend.
__device__ __forceinline__ float2 tl_row_stats_(float* s, unsigned t, unsigned T,
                                                const float* src, unsigned cols,
                                                float eps) {
  float inv_n = 1.0f / (float)cols;
  float sum = 0.0f;
  for (unsigned c = t; c < cols; c += T) sum += src[c];
  const float mu = tl_tree_sum_(s, t, T, sum) * inv_n;
  float ss = 0.0f;
  for (unsigned c = t; c < cols; c += T) {
    float v = src[c] - mu;
    ss += v * v;
  }
  return make_float2(mu, 1.0f / sqrtf(tl_tree_sum_(s, t, T, ss) * inv_n + eps));
}

// ---- layer norm over the last axis (rows×cols out), affine epilogue ----
// out = (x - mu) * 1/sqrt(var + eps) * g + b per row.
__global__ void tl_layer_norm(const float* x, const float* g, const float* b,
                              float* out, unsigned rows, unsigned cols,
                              float eps, float scale, float offset) {
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const float* src = x + (size_t)row * cols;
  float* dst = out + (size_t)row * cols;
  extern __shared__ float sdata[];
  unsigned t = threadIdx.x, T = blockDim.x;
  const float2 st = tl_row_stats_(sdata, t, T, src, cols, eps);
  const float mu = st.x, inv = st.y;
  for (unsigned c = t; c < cols; c += T)
    dst[c] = ((src[c] - mu) * inv * g[c] + b[c]) * scale + offset;
}

// ---- layer norm's pullback over the last axis, in three launches ----
// With x̂ = (x − μ)·s and ĝ = dy ⊙ γ:  dx = s · (ĝ − mean(ĝ) − x̂ · mean(ĝ ⊙ x̂)),
// dγ = Σ_rows dy ⊙ x̂, dβ = Σ_rows dy. The dx kernel recomputes (μ, s) like the
// forward and leaves them in `stats` [2, rows] for the column sums. One block
// per row; shared memory holds 2·blockDim floats.
__global__ void tl_layer_norm_bwd_dx(const float* x, const float* g,
                                     const float* dy, float* dx, float* stats,
                                     unsigned rows, unsigned cols, float eps) {
  unsigned row = blockIdx.x;
  if (row >= rows) return;
  const float* src = x + (size_t)row * cols;
  const float* gy = dy + (size_t)row * cols;
  float* dst = dx + (size_t)row * cols;
  extern __shared__ float sdata[];
  unsigned t = threadIdx.x, T = blockDim.x;
  float inv_n = 1.0f / (float)cols;
  const float2 st = tl_row_stats_(sdata, t, T, src, cols, eps);
  const float mu = st.x, inv = st.y;
  float sg = 0.0f, sgx = 0.0f;
  for (unsigned c = t; c < cols; c += T) {
    float gh = gy[c] * g[c];
    sg += gh;
    sgx += gh * (src[c] - mu) * inv;
  }
  const float2 sums = tl_tree_sum2_(sdata, t, T, sg, sgx);
  const float mean_g = sums.x * inv_n, mean_gx = sums.y * inv_n;
  for (unsigned c = t; c < cols; c += T) {
    float xhat = (src[c] - mu) * inv;
    dst[c] = inv * (gy[c] * g[c] - mean_g - xhat * mean_gx);
  }
  if (t == 0) {
    stats[row] = mu;
    stats[rows + row] = inv;
  }
}

// The column sums: a block per (32-column strip, chunk of rows), a warp reading
// 32 consecutive columns of one row, into `partials` [2, chunks, cols] (dγ's
// plane first). Fixed lane and chunk order, so deterministic unlike atomics.
__global__ void tl_layer_norm_bwd_gb(const float* x, const float* dy,
                                     const float* stats, float* partials,
                                     unsigned rows, unsigned cols,
                                     unsigned rows_per_chunk) {
  __shared__ float sg[8][32], sgx[8][32];
  unsigned tx = threadIdx.x, ty = threadIdx.y;
  unsigned c = blockIdx.x * 32 + tx;
  unsigned chunk = blockIdx.y;
  unsigned r0 = chunk * rows_per_chunk;
  unsigned r1 = r0 + rows_per_chunk;
  if (r1 > rows) r1 = rows;
  float ag = 0.0f, agx = 0.0f;
  if (c < cols) {
    for (unsigned r = r0 + ty; r < r1; r += 8) {
      float gy = dy[(size_t)r * cols + c];
      float xhat = (x[(size_t)r * cols + c] - stats[r]) * stats[rows + r];
      ag += gy;
      agx += gy * xhat;
    }
  }
  sg[ty][tx] = ag;
  sgx[ty][tx] = agx;
  __syncthreads();
  if (ty == 0 && c < cols) {
    float tg = 0.0f, tgx = 0.0f;
    for (unsigned l = 0; l < 8; l++) {
      tg += sg[l][tx];
      tgx += sgx[l][tx];
    }
    size_t at = (size_t)chunk * cols + c;
    partials[at] = tgx;
    partials[(size_t)gridDim.y * cols + at] = tg;
  }
}

// The second half: a thread per column folds its chunks into dγ and dβ.
__global__ void tl_layer_norm_bwd_gb_fold(const float* partials, float* dg,
                                          float* db, unsigned chunks,
                                          unsigned cols) {
  unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float tg = 0.0f, tb = 0.0f;
  for (unsigned k = 0; k < chunks; k++) {
    tg += partials[(size_t)k * cols + c];
    tb += partials[((size_t)chunks + k) * cols + c];
  }
  dg[c] = tg;
  db[c] = tb;
}

// ---- M9 batched-prefill GEMM: C[M,N] f32 = A[M,K] f32 @ B[N,K]^T bf16 ----
// Decode is a GEMV (M=1) and bandwidth-bound; PREFILL has M = the prompt chunk,
// which makes the same weights compute-bound (at M=256, N=9728, K=896 the
// arithmetic intensity is ~159 FLOP/byte vs the ~37 the HBM can feed). Feeding
// the prompt one token at a time therefore leaves ~25x on the table — measured
// in bench_qwen_prefill before this kernel existed.
//
// "NT": B is the SAME [N,K] row-major bf16 weight the decode GEMV already uses
// (K contiguous per output row, GGML-native), so batching costs no extra weight
// memory and no repacking — and both operands stream K contiguously, which is
// the friendliest layout for tiling.
//
// A and B tiles are staged transposed in shared memory so the inner loop reads
// each operand as float4s. Any K and any operand alignment: the loads take
// vectors where they can and predicate the K tail as they do the M/N edges
// (the launcher sends the aligned, K % 8 == 0 case to the tensor cores).
// The block tile is templated: 256 threads always, tile BM x BM, K-slab
// BK = 1024/BM (so each thread stages exactly one 4-wide chunk of each operand),
// microtile (BM/16) x (BM/16) held as G = BM/64 float4 GROUPS per axis, 64 apart.
// Grouping matters at BM=128: a contiguous 8-wide microtile makes each warp's
// LDS.128 span 8-float strides and serialize 2-way, while two 4-wide groups read
// consecutive float4s. At BM=64 there is one group, i.e. the plain map.
// Two instantiations, picked host-side by which one actually FILLS the GPU at
// the shape in hand:
//   128 -> 8x8 microtile, 64 FMA per 16 shared floats — the most efficient, but
//          a [512, 896] projection is only 4x7 = 28 blocks on 82 SMs.
//   64  -> 4x4, half the arithmetic intensity, 4x the blocks.
// Prefill chunks are a few hundred tokens, so the small tile is what the narrow
// attention projections (N=896) need; the wide MLP one (N=9728) fills either way
// and keeps the efficient tile.
//
// SPLITK slices K over gridDim.z and combines with atomicAdd (the caller zeroes
// C first) — the lever for the narrow shapes, whose 112-block grid is 1.4 waves
// of the 82 SMs and leaves most of the tail idle. Measured at M=512: wd
// (N=896,K=4864) 506 -> 312 us, wo 92 -> 75. It costs exact reproducibility
// (float adds land in arrival order), so it is opt-in per launch, not the
// default.
//
// The K slab is prefetched into registers while the current one computes:
// gateup 547 -> 461 us.
}  // extern "C"
template <int BM, bool SPLITK>
__device__ __forceinline__ void gemm_bf16_nt_core(const float* __restrict__ A,
                                                  const __nv_bfloat16* __restrict__ B,
                                                  float* __restrict__ C,
                                                  unsigned M, unsigned N,
                                                  unsigned K, unsigned ksplit) {
  constexpr int BK = 1024 / BM;   // BM*BK == 1024 == 256 threads x 4 elements
  constexpr int G = BM / 64;      // float4 microtile groups per axis
  // +4 floats of row padding: the staging stores walk a whole tile row per
  // thread, which at stride BM lands every warp on 8-16 banks (4- resp. 2-way
  // conflicts). Padding by 4 breaks that while keeping the stride a multiple of
  // 4 floats, so the microtile's float4 reads stay 16-byte aligned.
  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BM + 4];
  const unsigned blockRow = blockIdx.y * BM;
  const unsigned blockCol = blockIdx.x * BM;
  const unsigned tid = threadIdx.x;

  const unsigned k0 = SPLITK ? blockIdx.z * ksplit : 0u;
  if (SPLITK && k0 >= K) return;
  const unsigned k1 = SPLITK ? (k0 + ksplit < K ? k0 + ksplit : K) : K;

  // Global-load map: each thread stages 4 contiguous K-elements of one row —
  // of A (one float4) and of B (4 bf16 = one 8-byte uint2).
  const unsigned ldRow = tid / (BK / 4);         // M index for A, N index for B
  const unsigned ldK = (tid % (BK / 4)) * 4u;
  const unsigned gA = blockRow + ldRow;
  const unsigned gB = blockCol + ldRow;
  const unsigned tr4 = (tid >> 4) * 4u;   // this thread's group-local row base
  const unsigned tc4 = (tid & 15u) * 4u;  // ... and column base

  float acc[G][4][G][4] = {};
  float4 pa;
  float2 p0, p1;
  // A thread's 4 elements of each operand load as one vector (A a float4, B
  // 8 bytes of bf16) where they lie whole on that width; otherwise one at a
  // time, zero past k1. So any K and any operand alignment run, and the zeros
  // a ragged last slab adds leave the sums as they were.
  const bool vecA = K % 4 == 0 && (size_t)A % 16 == 0;
  const bool vecB = K % 4 == 0 && (size_t)B % 8 == 0;
  // Stage the K slab at `kt` into registers (global -> reg), then into shared.
  auto load = [&](unsigned kt) {
    const unsigned k = kt + ldK;
    const bool whole = k + 4 <= k1;
    float a[4] = {}, b[4] = {};
    if (gA < M) {
      const float* src = &A[(size_t)gA * K + k];
      if (vecA && whole) {
        *reinterpret_cast<float4*>(a) = *reinterpret_cast<const float4*>(src);
      } else {
#pragma unroll
        for (int e = 0; e < 4; e++)
          if (k + e < k1) a[e] = src[e];
      }
    }
    if (gB < N) {
      const __nv_bfloat16* src = &B[(size_t)gB * K + k];
      if (vecB && whole) {
        const uint2 raw = *reinterpret_cast<const uint2*>(src);
        *reinterpret_cast<float2*>(b) =
            __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(raw.x));
        *reinterpret_cast<float2*>(b + 2) =
            __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(raw.y));
      } else {
#pragma unroll
        for (int e = 0; e < 4; e++)
          if (k + e < k1) b[e] = __bfloat162float(src[e]);
      }
    }
    pa = make_float4(a[0], a[1], a[2], a[3]);
    p0 = make_float2(b[0], b[1]);
    p1 = make_float2(b[2], b[3]);
  };
  auto store = [&]() {
    As[ldK + 0][ldRow] = pa.x;
    As[ldK + 1][ldRow] = pa.y;
    As[ldK + 2][ldRow] = pa.z;
    As[ldK + 3][ldRow] = pa.w;
    Bs[ldK + 0][ldRow] = p0.x;
    Bs[ldK + 1][ldRow] = p0.y;
    Bs[ldK + 2][ldRow] = p1.x;
    Bs[ldK + 3][ldRow] = p1.y;
  };

  load(k0);
  store();
  __syncthreads();
  for (unsigned kt = k0; kt < k1; kt += BK) {
    const bool more = kt + BK < k1;
    if (more) load(kt + BK);  // prefetch: overlaps the global read with compute
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float regM[G][4], regN[G][4];
#pragma unroll
      for (int g = 0; g < G; g++) {
        *reinterpret_cast<float4*>(regM[g]) =
            *reinterpret_cast<const float4*>(&As[kk][g * 64 + tr4]);
        *reinterpret_cast<float4*>(regN[g]) =
            *reinterpret_cast<const float4*>(&Bs[kk][g * 64 + tc4]);
      }
#pragma unroll
      for (int gi = 0; gi < G; gi++)
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int gj = 0; gj < G; gj++)
#pragma unroll
            for (int j = 0; j < 4; j++)
              acc[gi][i][gj][j] += regM[gi][i] * regN[gj][j];
    }
    __syncthreads();
    if (more) {
      store();
      __syncthreads();
    }
  }

#pragma unroll
  for (int gi = 0; gi < G; gi++)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned r = blockRow + gi * 64 + tr4 + i;
      if (r >= M) continue;
#pragma unroll
      for (int gj = 0; gj < G; gj++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const unsigned c = blockCol + gj * 64 + tc4 + j;
          if (c >= N) continue;
          if (SPLITK) atomicAdd(&C[(size_t)r * N + c], acc[gi][i][gj][j]);
          else C[(size_t)r * N + c] = acc[gi][i][gj][j];
        }
    }
}
// Elements only: the core picks its vector loads by address (vecA, vecB).
TL_ALIGN(tl_gemm_bf16_nt, sizeof(float), sizeof(__nv_bfloat16), sizeof(float))
TL_ALIGN(tl_gemm_bf16_nt_s, sizeof(float), sizeof(__nv_bfloat16), sizeof(float))
TL_ALIGN(tl_gemm_bf16_nt_sk, sizeof(float), sizeof(__nv_bfloat16), sizeof(float))
extern "C" __global__ void tl_gemm_bf16_nt(const float* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, float* __restrict__ C, unsigned M,
    unsigned N, unsigned K) {
  gemm_bf16_nt_core<128, false>(A, B, C, M, N, K, 0);
}
extern "C" __global__ void tl_gemm_bf16_nt_s(const float* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, float* __restrict__ C, unsigned M,
    unsigned N, unsigned K) {
  gemm_bf16_nt_core<64, false>(A, B, C, M, N, K, 0);
}
extern "C" __global__ void tl_gemm_bf16_nt_sk(const float* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, float* __restrict__ C, unsigned M,
    unsigned N, unsigned K, unsigned ksplit) {
  gemm_bf16_nt_core<64, true>(A, B, C, M, N, K, ksplit);
}

// The same product on the tensor cores: C[M,N] = A[M,K] . B[N,K]^T over
// mma.sync.m16n8k16, the operands read from shared memory with ldmatrix.
//
// A's f32 activations are rounded as they are staged, B is the bf16 weights.
// F16 is the fast form: both operands rounded to f16 (a bf16 weight that came
// from an F16 file round-trips exactly) and the sums f16, which the tensor
// cores here run at twice the f32 rate; f16's range is the price: an
// activation or a partial sum past 65504 becomes inf (own::gemm_bf16_nt_f16acc).
// Otherwise the operands are bf16 (the range an outlier needs) and the sums f32.
//
// A BM x BN block tile in BK-deep slabs, a warp to each WM x WN of it. Two
// slabs live in shared memory, and the next one is read into registers (16-byte
// loads, so K % 8 == 0 and 16-byte aligned rows) while the tensor cores run on
// the current one: the conversion happens once per block as the slab is stored,
// not once per warp as it is read. Two stages rather than three: the smaller
// footprint lets two blocks share an SM, which was worth more than the depth
// (QKV at M=512: 58 us against 83 for four stages of the same tile). Tails in
// M, N or K are zero-filled as staged and masked as stored; N is even, as a
// thread stores its two adjacent columns together.
__device__ __forceinline__ void tc_ldsm4_(unsigned& r0, unsigned& r1, unsigned& r2,
                                          unsigned& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"((unsigned)__cvta_generic_to_shared(p)));
}
// One m16n8k16 step, per operand type and accumulator.
__device__ __forceinline__ void tc_mma_(unsigned* c, const unsigned* a, unsigned b0,
                                        unsigned b1, __half) {  // f16 operands, f16 sums
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
      "{%0,%1};\n"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void tc_mma_(float* c, const unsigned* a, unsigned b0,
                                        unsigned b1, __nv_bfloat16) {  // bf16 operands, f32 sums
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// An accumulator's two adjacent columns of one row, as f32.
__device__ __forceinline__ float2 tc_pair_(const unsigned* c, int row_half) {
  return __half22float2(*reinterpret_cast<const __half2*>(&c[row_half]));
}
__device__ __forceinline__ float2 tc_pair_(const float* c, int row_half) {
  return make_float2(c[2 * row_half], c[2 * row_half + 1]);
}
// Eight consecutive weights of one block-quantized row (k = gk .. gk+7, gk % 8 == 0),
// dequantized as ggml does and rounded to the GEMM's operand type, four pairs
// packed in a uint4: what the GEMM stages for a B row chunk of a quantized weight.
template <bool F16>
__device__ __forceinline__ unsigned tc_pack2_(float a, float b) {
  if constexpr (F16) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
  } else {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h);
  }
}
// A Q4_K block's 6-bit scale and min of group j (0..7), ggml's get_scale_min_k4,
// from its 12 scale bytes (block bytes 4..15) held as three words. Indexing a
// 12-byte copy of them instead put the copy in local memory (a stack frame) in
// every kernel that did it: the Q4_K decode GEMV ran at 650-720 GB/s rather than
// 830, and the MMQ, dequantization and GEMM staging carried the same frame.
__device__ __forceinline__ void q4k_scale_min_(uint3 w, int j, int& sc, int& mn) {
  // Bytes j & 3 of the three words: j < 4 reads q[j] and q[j + 4], j >= 4 the low or high
  // nibble of q[j + 4] and the top two bits of q[j - 4] / q[j].
  const unsigned sh = 8u * (unsigned)(j & 3), a = w.x >> sh, b = w.y >> sh, c = w.z >> sh;
  sc = (int)(j < 4 ? a & 63u : (c & 0xFu) | ((a >> 2) & 0x30u));
  mn = (int)(j < 4 ? b & 63u : ((c >> 4) & 0xFu) | ((b >> 2) & 0x30u));
}
template <bool F16>
__device__ __forceinline__ uint4 tc_dq8_q4k_(const unsigned char* row, unsigned gk) {
  const unsigned char* b = row + (size_t)(gk >> 8) * 144u;
  const unsigned o = gk & 255u, g = o >> 5, p = g >> 1, l0 = o & 31u;
  const float2 dd = __half22float2(*reinterpret_cast<const __half2*>(b));  // d, dmin
  int sc, mn;
  q4k_scale_min_(*reinterpret_cast<const uint3*>(b + 4), (int)g, sc, mn);
  const float d1 = dd.x * (float)sc, m1 = dd.y * (float)mn;
  const uint2 w = *reinterpret_cast<const uint2*>(b + 16 + 32 * p + l0);
  const unsigned sh = (g & 1u) ? 4u : 0u;
  float v[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = d1 * (float)((w.x >> (8 * i + sh)) & 0xFu) - m1;
    v[4 + i] = d1 * (float)((w.y >> (8 * i + sh)) & 0xFu) - m1;
  }
  return make_uint4(tc_pack2_<F16>(v[0], v[1]), tc_pack2_<F16>(v[2], v[3]),
                    tc_pack2_<F16>(v[4], v[5]), tc_pack2_<F16>(v[6], v[7]));
}
template <bool F16>
__device__ __forceinline__ uint4 tc_dq8_q6k_(const unsigned char* row, unsigned gk) {
  const unsigned char* b = row + (size_t)(gk >> 8) * 210u;
  const unsigned o = gk & 255u, h = o >> 7, qd = (o >> 5) & 3u, l0 = o & 31u;
  const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
  const unsigned char* qlp = b + 64 * h + (qd & 1u) * 32 + l0;
  const unsigned char* qhp = b + 128 + 32 * h + l0;
  const signed char sc = reinterpret_cast<const signed char*>(b + 192)[8 * h + (l0 >> 4) + 2 * qd];
  const float ds = d * (float)sc;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const unsigned ql = qlp[i], qh = qhp[i];
    const int lo = qd < 2 ? (int)(ql & 0xF) : (int)(ql >> 4);
    v[i] = ds * (float)(((lo | (int)(((qh >> (2 * qd)) & 3u) << 4))) - 32);
  }
  return make_uint4(tc_pack2_<F16>(v[0], v[1]), tc_pack2_<F16>(v[2], v[3]),
                    tc_pack2_<F16>(v[4], v[5]), tc_pack2_<F16>(v[6], v[7]));
}

template <bool F16>
__device__ __forceinline__ uint4 tc_dq8_q80_(const unsigned char* row, unsigned gk) {
  const unsigned char* b = row + (size_t)(gk >> 5) * 34u;  // 2-byte aligned only
  const float d = __half2float(*reinterpret_cast<const __half*>(b));
  const unsigned short* q16 = reinterpret_cast<const unsigned short*>(b + 2 + (gk & 31u));
  float v[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned w = q16[i];
    v[2 * i] = (float)(signed char)(w & 0xFFu) * d;
    v[2 * i + 1] = (float)(signed char)(w >> 8) * d;
  }
  return make_uint4(tc_pack2_<F16>(v[0], v[1]), tc_pack2_<F16>(v[2], v[3]),
                    tc_pack2_<F16>(v[4], v[5]), tc_pack2_<F16>(v[6], v[7]));
}

// WF: the weight format B holds: 0 bf16 [N][K], 1 Q4_K, 2 Q6_K blocks (rows of
// K/256 blocks, K % 256 == 0), 3 Q8_0 (rows of K/32 blocks), 4 f16 [N][K] (a
// block-quantized weight dequantized once ahead, tl_dequant_f16). C rows are `ldc` apart, so a part of a fused
// projection writes its own columns of the shared output.
template <int BM, int BN, int BK, int WM, int WN, bool F16, bool AH = false, int WF = 0>
__device__ __forceinline__ void gemm_nt_tc_core(const void* __restrict__ Av,
                                                const void* __restrict__ Bv,
                                                float* __restrict__ C, unsigned M,
                                                unsigned N, unsigned K, unsigned ldc) {
  static_assert(!AH || F16, "an f16 activation feeds the f16 form");
  using T = typename std::conditional<F16, __half, __nv_bfloat16>::type;     // operands
  using ACC = typename std::conditional<F16, unsigned, float>::type;         // sums
  const float* A = static_cast<const float*>(Av);          // AH: really __half
  const __half* Ah = static_cast<const __half*>(Av);
  const __nv_bfloat16* B = static_cast<const __nv_bfloat16*>(Bv);
  const unsigned char* Bq = static_cast<const unsigned char*>(Bv);
  const size_t qrow = WF == 1 ? (size_t)(K >> 8) * 144u
                    : WF == 2 ? (size_t)(K >> 8) * 210u
                              : (size_t)(K >> 5) * 34u;  // bytes a row
  constexpr int LD = BK + 8;                    // +8: rows stay 16-byte aligned and
                                                // ldmatrix reads hit distinct banks
  constexpr int NT = (BM / WM) * (BN / WN) * 32;
  constexpr int FM = WM / 16, FN = WN / 8, NA = F16 ? 2 : 4;
  constexpr int AV = BM * BK / 8 / NT;          // 8-element vectors a thread stages
  constexpr int BV = BN * BK / 8 / NT;
  static_assert(AV >= 1 && BV >= 1 && FN % 2 == 0, "tile too small for its threads");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  T* sm = reinterpret_cast<T*>(tc_smem);
  auto As = [&](int s) { return sm + s * (BM + BN) * LD; };
  auto Bs = [&](int s) { return sm + s * (BM + BN) * LD + BM * LD; };
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned wm = (warp / (BN / WN)) * WM, wn = (warp % (BN / WN)) * WN;
  const unsigned row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  ACC acc[FM][FN][NA];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++)
#pragma unroll
      for (int e = 0; e < NA; e++) acc[i][j][e] = ACC(0);
  float4 ra[AV][AH ? 1 : 2];  // 8 values a vector: 2 float4s, or 8 halves in one
  uint4 rb[BV];
  // Vector id of a thread: row id / (BK/8), column (id % (BK/8)) * 8.
  auto load = [&](unsigned k0) {
#pragma unroll
    for (int v = 0; v < AV; v++) {
      const unsigned id = tid + v * NT, gr = row0 + id / (BK / 8), gk = k0 + (id % (BK / 8)) * 8;
      const bool ok = gr < M && gk < K;
      if constexpr (AH) {  // 8 halves in the first 16 bytes
        ra[v][0] = ok ? *reinterpret_cast<const float4*>(Ah + (size_t)gr * K + gk)
                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      } else {
        const float4* p = reinterpret_cast<const float4*>(A + (size_t)gr * K + gk);
        ra[v][0] = ok ? p[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        ra[v][1] = ok ? p[1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
#pragma unroll
    for (int v = 0; v < BV; v++) {
      const unsigned id = tid + v * NT, gn = col0 + id / (BK / 8), gk = k0 + (id % (BK / 8)) * 8;
      if constexpr (WF == 0) {
        rb[v] = gn < N && gk < K ? *reinterpret_cast<const uint4*>(B + (size_t)gn * K + gk)
                                 : make_uint4(0u, 0u, 0u, 0u);
      } else if constexpr (WF == 1) {
        rb[v] = gn < N && gk < K ? tc_dq8_q4k_<F16>(Bq + gn * qrow, gk) : make_uint4(0u, 0u, 0u, 0u);
      } else if constexpr (WF == 2) {
        rb[v] = gn < N && gk < K ? tc_dq8_q6k_<F16>(Bq + gn * qrow, gk) : make_uint4(0u, 0u, 0u, 0u);
      } else if constexpr (WF == 4) {
        rb[v] = gn < N && gk < K ? *reinterpret_cast<const uint4*>(Bq + (gn * (size_t)K + gk) * 2u)
                                 : make_uint4(0u, 0u, 0u, 0u);
      } else {
        rb[v] = gn < N && gk < K ? tc_dq8_q80_<F16>(Bq + gn * qrow, gk) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto stage = [&](int s) {
#pragma unroll
    for (int v = 0; v < AV; v++) {
      const unsigned id = tid + v * NT;
      T* d = As(s) + (id / (BK / 8)) * LD + (id % (BK / 8)) * 8;
      if constexpr (AH) {
        *reinterpret_cast<float4*>(d) = ra[v][0];
      } else {
        const float* f = reinterpret_cast<const float*>(&ra[v][0]);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          if constexpr (F16) *reinterpret_cast<__half2*>(d + 2 * q) = __floats2half2_rn(f[2 * q], f[2 * q + 1]);
          else *reinterpret_cast<__nv_bfloat162*>(d + 2 * q) = __floats2bfloat162_rn(f[2 * q], f[2 * q + 1]);
        }
      }
    }
#pragma unroll
    for (int v = 0; v < BV; v++) {
      const unsigned id = tid + v * NT;
      T* d = Bs(s) + (id / (BK / 8)) * LD + (id % (BK / 8)) * 8;
      if constexpr (WF != 0) {  // dequantized to T pairs already
        *reinterpret_cast<uint4*>(d) = rb[v];
      } else if constexpr (F16) {
        const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(&rb[v]);
#pragma unroll
        for (int q = 0; q < 4; q++)
          *reinterpret_cast<__half2*>(d + 2 * q) = __float22half2_rn(__bfloat1622float2(w[q]));
      } else {
        *reinterpret_cast<uint4*>(d) = rb[v];
      }
    }
  };
  // ldmatrix lanes: A is a 16x16 tile (lane -> row lane%16, column half lane/16);
  // B ([n][k]) two n8 blocks by two k halves.
  const unsigned la_r = lane & 15, la_c = (lane >> 4) * 8;
  const unsigned lb_r = (lane & 7) + ((lane >> 4) << 3), lb_c = ((lane >> 3) & 1) * 8;
  const unsigned nk = (K + BK - 1) / BK;
  load(0);
  stage(0);
  __syncthreads();
  for (unsigned kt = 0; kt < nk; kt++) {
    if (kt + 1 < nk) load((kt + 1) * BK);  // in flight while this slab multiplies
    const T* a = As(kt & 1);
    const T* b = Bs(kt & 1);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      unsigned fa[FM][4], fb[FN][2];
#pragma unroll
      for (int i = 0; i < FM; i++)
        tc_ldsm4_(fa[i][0], fa[i][1], fa[i][2], fa[i][3], a + (wm + 16 * i + la_r) * LD + kk + la_c);
#pragma unroll
      for (int j = 0; j < FN; j += 2)
        tc_ldsm4_(fb[j][0], fb[j][1], fb[j + 1][0], fb[j + 1][1], b + (wn + 8 * j + lb_r) * LD + kk + lb_c);
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) tc_mma_(acc[i][j], fa[i], fb[j][0], fb[j][1], T());
    }
    if (kt + 1 < nk) stage((kt + 1) & 1);  // the other buffer: nobody reads it now
    __syncthreads();
  }
  // m16n8 accumulator: row lane/4 (+8), columns 2*(lane%4) + 0/1.
  const unsigned g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      const unsigned gr = row0 + wm + 16 * i + g, gc = col0 + wn + 8 * j + 2 * t;
      if (gc >= N) continue;
      if (gr < M) *reinterpret_cast<float2*>(C + (size_t)gr * ldc + gc) = tc_pair_(acc[i][j], 0);
      if (gr + 8 < M) *reinterpret_cast<float2*>(C + (size_t)(gr + 8) * ldc + gc) = tc_pair_(acc[i][j], 1);
    }
}
// Three tiles, picked by how well the grid fills the GPU (gemm_tc_run_ in cuda.h):
// L 128x128 for wide outputs, M 64x128 and S 64x64 for narrow ones. Dynamic
// shared memory, (BM+BN)*(BK+8)*2 stages*2 bytes: 40/55/37 KB.
// A and a bf16 / f16 B by 16-byte vectors, Q4_K's nibbles by 8, Q6_K's and Q8_0's
// halves by 2; C by float2 stores.
#define TL_GEMM_TC(NAME, BM, BN, BK, WM, WN, F16, AH, WF)                            \
  TL_ALIGN(NAME, 16, (WF) == 1 ? 8 : (WF) == 2 || (WF) == 3 ? 2 : 16, 8)              \
  extern "C" __global__ void NAME(const void* __restrict__ A,                        \
      const void* __restrict__ B, float* __restrict__ C, unsigned M, unsigned N,     \
      unsigned K, unsigned ldc) {                                                    \
    gemm_nt_tc_core<BM, BN, BK, WM, WN, F16, AH, WF>(A, B, C, M, N, K, ldc);         \
  }
// Per weight format: bf16 and the block-quantized ones (q4k, q6k, q80); per tile (l, m, s);
// per form: f32 (bf16 operands, f32 sums), f16 (f16 operands and sums, A f32), f16h
// (as f16 with A already f16). The quantized weights have the exact form (f32) and
// the f16h one: the prefill's two, as a backend that rounds activations where they
// are made always has the second.
#define TL_GEMM_TC_TILES(PFX, F16, AH, WF)                                           \
  TL_GEMM_TC(PFX##_l, 128, 128, 32, 64, 64, F16, AH, WF)                             \
  TL_GEMM_TC(PFX##_m, 64, 128, 64, 32, 32, F16, AH, WF)                              \
  TL_GEMM_TC(PFX##_s, 64, 64, 64, 32, 32, F16, AH, WF)
TL_GEMM_TC_TILES(tl_gemm_tc_f32, false, false, 0)
TL_GEMM_TC_TILES(tl_gemm_tc_f16, true, false, 0)
TL_GEMM_TC_TILES(tl_gemm_tc_f16h, true, true, 0)
TL_GEMM_TC_TILES(tl_gemm_tc_f32_q4k, false, false, 1)
TL_GEMM_TC_TILES(tl_gemm_tc_f16h_q4k, true, true, 1)
TL_GEMM_TC_TILES(tl_gemm_tc_f32_q6k, false, false, 2)
TL_GEMM_TC_TILES(tl_gemm_tc_f16h_q6k, true, true, 2)
TL_GEMM_TC_TILES(tl_gemm_tc_f32_q80, false, false, 3)
TL_GEMM_TC_TILES(tl_gemm_tc_f16h_q80, true, true, 3)
TL_GEMM_TC_TILES(tl_gemm_tc_f16h_f16w, true, true, 4)
#undef TL_GEMM_TC_TILES

// The f16 weight's fourth tile, for the long-K GEMMs of a big model (a 27B's
// projections, K >= 3072): 256x128, BK 32, three cp.async stages, 16 warps of
// 64x32. A 256-row block reads each weight tile from L2 half as often as the
// 128-row ones, and the copies go straight from global to shared memory with
// no registers in between: 86 TFLOPS at the 27B's shapes against 63 for the
// tile above (K = 1024 shapes lose a little, which is why gemm_tc_run_ picks it).
// Both operands are f16 already, so there is no conversion to stage.
__device__ __forceinline__ void tc_cp16_(void* s, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"((unsigned)__cvta_generic_to_shared(s)), "l"(g), "r"(ok ? 16 : 0));
}
template <int BM, int BN, int BK, int WM, int WN, int STG>
__device__ __forceinline__ void gemm_nt_tc_cp_core(const __half* __restrict__ A,
                                                   const __half* __restrict__ B,
                                                   float* __restrict__ C, unsigned M,
                                                   unsigned N, unsigned K, unsigned ldc) {
  constexpr int LD = BK + 8;
  constexpr int NT = (BM / WM) * (BN / WN) * 32;
  constexpr int FM = WM / 16, FN = WN / 8;
  constexpr int AV = BM * BK / 8 / NT, BV = BN * BK / 8 / NT;
  static_assert(AV >= 1 && BV >= 1 && FN % 2 == 0, "tile too small for its threads");
  extern __shared__ __align__(16) unsigned char tc_smem[];
  __half* sm = reinterpret_cast<__half*>(tc_smem);
  auto As = [&](int s) { return sm + s * (BM + BN) * LD; };
  auto Bs = [&](int s) { return sm + s * (BM + BN) * LD + BM * LD; };
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned wm = (warp / (BN / WN)) * WM, wn = (warp % (BN / WN)) * WN;
  const unsigned row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  unsigned acc[FM][FN][2];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) acc[i][j][0] = acc[i][j][1] = 0;
  // A slab to stage s; out-of-range vectors copy zero bytes (src-size 0).
  auto issue = [&](int s, unsigned k0) {
#pragma unroll
    for (int v = 0; v < AV; v++) {
      const unsigned id = tid + v * NT, r = id / (BK / 8), c = (id % (BK / 8)) * 8;
      const unsigned gr = row0 + r, gk = k0 + c;
      const bool ok = gr < M && gk < K;
      tc_cp16_(As(s) + r * LD + c, A + (size_t)(ok ? gr : 0) * K + (ok ? gk : 0), ok);
    }
#pragma unroll
    for (int v = 0; v < BV; v++) {
      const unsigned id = tid + v * NT, r = id / (BK / 8), c = (id % (BK / 8)) * 8;
      const unsigned gn = col0 + r, gk = k0 + c;
      const bool ok = gn < N && gk < K;
      tc_cp16_(Bs(s) + r * LD + c, B + (size_t)(ok ? gn : 0) * K + (ok ? gk : 0), ok);
    }
  };
  const unsigned la_r = lane & 15, la_c = (lane >> 4) * 8;
  const unsigned lb_r = (lane & 7) + ((lane >> 4) << 3), lb_c = ((lane >> 3) & 1) * 8;
  const unsigned nk = (K + BK - 1) / BK;
#pragma unroll
  for (int s = 0; s < STG - 1; s++) {
    if (s < (int)nk) issue(s, s * BK);
    asm volatile("cp.async.commit_group;\n");
  }
  for (unsigned kt = 0; kt < nk; kt++) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(STG - 2));
    __syncthreads();  // slab kt landed for everyone, and slab kt-1's readers are done
    const unsigned nx = kt + STG - 1;
    if (nx < nk) issue(nx % STG, nx * BK);
    asm volatile("cp.async.commit_group;\n");
    const __half* a = As(kt % STG);
    const __half* b = Bs(kt % STG);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      unsigned fa[FM][4], fb[FN][2];
#pragma unroll
      for (int i = 0; i < FM; i++)
        tc_ldsm4_(fa[i][0], fa[i][1], fa[i][2], fa[i][3], a + (wm + 16 * i + la_r) * LD + kk + la_c);
#pragma unroll
      for (int j = 0; j < FN; j += 2)
        tc_ldsm4_(fb[j][0], fb[j][1], fb[j + 1][0], fb[j + 1][1], b + (wn + 8 * j + lb_r) * LD + kk + lb_c);
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) tc_mma_(acc[i][j], fa[i], fb[j][0], fb[j][1], __half());
    }
  }
  const unsigned g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      const unsigned gr = row0 + wm + 16 * i + g, gc = col0 + wn + 8 * j + 2 * t;
      if (gc >= N) continue;
      if (gr < M) *reinterpret_cast<float2*>(C + (size_t)gr * ldc + gc) = tc_pair_(acc[i][j], 0);
      if (gr + 8 < M) *reinterpret_cast<float2*>(C + (size_t)(gr + 8) * ldc + gc) = tc_pair_(acc[i][j], 1);
    }
}
TL_ALIGN(tl_gemm_tc_f16h_f16w_xl, 16, 16, 8)
extern "C" __global__ void tl_gemm_tc_f16h_f16w_xl(const void* __restrict__ A,
    const void* __restrict__ B, float* __restrict__ C, unsigned M, unsigned N, unsigned K,
    unsigned ldc) {
  gemm_nt_tc_cp_core<256, 128, 32, 64, 32, 3>(static_cast<const __half*>(A),
                                              static_cast<const __half*>(B), C, M, N, K, ldc);
}

// A block-quantized weight run (rows x K) dequantized to f16 [rows][K], a thread eight
// weights: the prefill does it once per projection per chunk instead of in every
// row block's staging of the GEMM, which then reads plain f16 (72 against 45 TFLOPS).
// The values are the staging's own (tc_dq8_*_<true>), so the GEMM's result is too.
template <int WF>
__device__ __forceinline__ void dequant_f16_core(const unsigned char* __restrict__ W,
                                                 __half* __restrict__ out, unsigned rows, unsigned K) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned per_row = K >> 3;
  if (t >= (size_t)rows * per_row) return;
  const unsigned row = (unsigned)(t / per_row), gk = (unsigned)(t % per_row) << 3;
  const size_t qrow = WF == 1 ? (size_t)(K >> 8) * 144u : WF == 2 ? (size_t)(K >> 8) * 210u : (size_t)(K >> 5) * 34u;
  const unsigned char* r = W + row * qrow;
  const uint4 v = WF == 1 ? tc_dq8_q4k_<true>(r, gk) : WF == 2 ? tc_dq8_q6k_<true>(r, gk) : tc_dq8_q80_<true>(r, gk);
  *reinterpret_cast<uint4*>(out + (size_t)row * K + gk) = v;
}
TL_ALIGN(tl_dequant_f16_q4k, 8, 16)
TL_ALIGN(tl_dequant_f16_q6k, 2, 16)
TL_ALIGN(tl_dequant_f16_q80, 2, 16)
extern "C" __global__ void tl_dequant_f16_q4k(const unsigned char* W, __half* out, unsigned rows, unsigned K) {
  dequant_f16_core<1>(W, out, rows, K);
}
extern "C" __global__ void tl_dequant_f16_q6k(const unsigned char* W, __half* out, unsigned rows, unsigned K) {
  dequant_f16_core<2>(W, out, rows, K);
}
extern "C" __global__ void tl_dequant_f16_q80(const unsigned char* W, __half* out, unsigned rows, unsigned K) {
  dequant_f16_core<3>(W, out, rows, K);
}

// ---- The prefill GEMM over block-quantized weights on the int8 tensor cores (llama.cpp's MMQ).
// C[M,N] = A[M,K] . W[N,K]^T with W kept as GGUF blocks and A quantized to int8, a
// scale d_a = max|a| / 127 per 32 values. Every format here is int8-like a group of 32
// weights at a time, w = d_w q - m_w, so the group's dot with the activation is
//   d_a (d_w sum q a  -  m_w sum a)
// and sum q a is one m16n8k32 s8 mma:
//   Q4_K: q the nibbles 0..15, d_w = d sc and m_w = dmin mn of the group's 6-bit scales;
//   Q8_0: q the int8s, d_w the block's d, m_w = 0.
// Nothing is dequantized to f16: each tile of the weight is unpacked to int8 (and its
// group scales to f32) once, as it is staged, and both operands are then read with
// ldmatrix by the same loop for every format. The price is the 8-bit rounding of the
// activation, as llama.cpp's MMQ and the decode's Q8 GEMV round it; it is the default
// prefill GEMM for these formats, and TL_NO_MMQ=1 takes the f16 path (the weight
// dequantized to f16, the activation f16) instead.
//
// tl_quant_act_mmq lays the activation out for it: int8 [Mpad][K], and the
// scales S[group][Mpad] as float2 (d, d * sum of the group's int8), so a tile's
// scales for one group are contiguous (bank-free reads, 16-byte loads). Rows
// from M to Mpad (a multiple of the tile) are zero.
// A thread rounds 8 values (16 bytes of the f16 row), four threads a group of 32, so a
// warp reads a whole super-block of a row at once.
TL_ALIGN(tl_quant_act_mmq, 16, 8, 8)
extern "C" __global__ void tl_quant_act_mmq(const __half* __restrict__ a, signed char* __restrict__ q,
                                            float2* __restrict__ S, unsigned M, unsigned Mpad,
                                            unsigned K) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x, per_row = K / 8;
  if (t >= Mpad * per_row) return;  // whole groups of 4 lanes: K % 256 == 0
  const unsigned m = t / per_row, k0 = (t % per_row) * 8, g = k0 / 32;
  float x[8];
  if (m < M) {
    const uint4 v = *reinterpret_cast<const uint4*>(a + (size_t)m * K + k0);
    const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float2 f = __half22float2(h[i]);
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = 0.0f;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const float d = amax / 127.0f;
  int sum = 0;
  unsigned packed[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int qi = d == 0.0f ? 0 : __float2int_rn(x[i] / d);
    sum += qi;
    packed[i / 4] |= (unsigned)(qi & 0xFF) << (8 * (i % 4));
  }
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  *reinterpret_cast<uint2*>(q + (size_t)m * K + k0) = make_uint2(packed[0], packed[1]);
  if ((t & 3) == 0) S[(size_t)g * Mpad + m] = make_float2(d, d * (float)sum);
}
__device__ __forceinline__ void tc_mma_s8_(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
// 128 activation rows x 128 weight rows a block, 8 warps of 32 x 64, one super-block (256
// of K) a step, as llama.cpp's Ampere MMQ. The next step's activation, its scales and the
// raw weight bytes are loaded into registers while this step multiplies; the step then
// unpacks them into shared memory (one stage, 84 KB: one block an SM). Unpacking at the
// stage rather than in the multiply loop does it once a block instead of once a warp.
// N is even; K % 256 == 0; the A side is padded to Mpad rows (a multiple of 128). A grid too
// small to fill the GPU splits K over blockIdx.z, kchunk super-blocks a part, part z writing
// its own copy of C zstride floats on (tl_mmq_reduce sums them, in order, so the result
// does not depend on scheduling).
// The activation's 128-row tiles run along blockIdx.x and the weight's along blockIdx.y,
// so the blocks the GPU runs at once hold every activation tile of a prompt chunk (four
// at 512 tokens) for ~20 weight tiles: each weight tile comes from memory once and its
// other blocks find it in L2, where the weight-major order read it once per activation
// tile: Qwen3.8-27B's prefill GEMMs take 2.6% (Q4_K) to 4.2% (Q8_0) less time.
//
// GLU: W is a fused gate|up projection (rows [0, N) the gate, [N, 2N) the up) and the
// output is silu(gate) * up in f16, [M][N] ldc apart, as tl_swiglu_h16 rounds it: a
// block's 128 weight rows are the gate and up rows of the same 64 outputs, a warp's
// 64 columns 32 of each, so a thread holds both halves of its outputs. Not split.
#define TL_MMQ_BM 128
#define TL_MMQ_BN 128
#define TL_MMQ_THREADS 256
#define TL_MMQ_LDK 272  // a staged row: 256 int8 + 16, so ldmatrix's 8 rows hit distinct banks
#define TL_MMQ_SMEM ((TL_MMQ_BM + TL_MMQ_BN) * (TL_MMQ_LDK + 64))  // int8 rows and 8 float2 scales a row
template <int WF, bool GLU = false>  // WF: 1 Q4_K, 3 Q8_0
__device__ __forceinline__ void gemm_mmq_core(const signed char* __restrict__ Aq,
                                              const float2* __restrict__ As_g,
                                              const unsigned char* __restrict__ W,
                                              void* __restrict__ Cv, unsigned M, unsigned N,
                                              unsigned K, unsigned Mpad, unsigned ldc,
                                              unsigned kchunk, unsigned zstride) {
  constexpr int BM = TL_MMQ_BM, BN = TL_MMQ_BN, NT = TL_MMQ_THREADS, LDK = TL_MMQ_LDK;
  constexpr int WM = 32, WN = 64, FM = WM / 16, FN = WN / 8;
  constexpr int NB = BN * 8 / NT;  // weight items a thread: (row, 16 B of nibbles) or (row, Q8_0 block)
  static_assert((BM / WM) * (BN / WN) * 32 == NT, "a warp a 32 x 64 tile");
  extern __shared__ __align__(16) unsigned char mmq_sm[];
  signed char* As = reinterpret_cast<signed char*>(mmq_sm);                       // [BM][LDK]
  float2* ASs = reinterpret_cast<float2*>(mmq_sm + BM * LDK);                     // [group][BM]
  signed char* Bs = reinterpret_cast<signed char*>(mmq_sm + BM * LDK + BM * 64);  // [BN][LDK]
  float2* BSs = reinterpret_cast<float2*>(mmq_sm + (BM + BN) * LDK + BM * 64);    // [group][BN]
  const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned wm = (warp / (BN / WN)) * WM, wn = (warp % (BN / WN)) * WN;
  const unsigned tm = blockIdx.x, tn = blockIdx.y;  // the activation's tile, the weight's
  const unsigned row0 = tm * BM, col0 = tn * BN;
  const unsigned nk = K / 256;
  const size_t wrow = (size_t)nk * (WF == 1 ? 144 : 272);  // bytes a weight row
  const unsigned kt0 = blockIdx.z * kchunk, kt1 = min(nk, kt0 + kchunk);
  float* C = static_cast<float*>(Cv) + (size_t)blockIdx.z * zstride;
  // The weight row a tile row stages: the next 128, or (GLU) gate rows of 32 outputs then
  // the same outputs' up rows, twice over 64 outputs.
  auto wrow_of = [&](unsigned r) {
    if constexpr (GLU) {
      const unsigned o = min(tn * (BN / 2) + (r / 64) * 32 + (r & 31), N - 1);
      return (r & 32) ? o + N : o;
    } else {
      return min(col0 + r, N - 1);
    }
  };
  float acc[FM][FN][4];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
  uint4 ra[BM * 16 / NT], rs[BM * 4 / NT];  // the activation's int8 rows and scales
  uint4 rb[WF == 1 ? NB : 1], rh[WF == 1 ? NB : 1];  // Q4_K: 16 B of nibbles, the header
  unsigned rw[WF == 3 ? NB : 1][9];                  // Q8_0: the 9 words a block lies in
  auto fetch = [&](unsigned kt) {
#pragma unroll
    for (int v = 0; v < BM * 16 / NT; v++) {
      const unsigned id = tid + v * NT, r = id >> 4, c = id & 15;
      ra[v] = *reinterpret_cast<const uint4*>(Aq + (size_t)(row0 + r) * K + kt * 256 + c * 16);
    }
#pragma unroll
    for (int v = 0; v < BM * 4 / NT; v++) {
      const unsigned id = tid + v * NT, g = id / (BM / 2), c = id % (BM / 2);
      rs[v] = *reinterpret_cast<const uint4*>(As_g + (size_t)(kt * 8 + g) * Mpad + row0 + c * 2);
    }
#pragma unroll
    for (int v = 0; v < NB; v++) {
      const unsigned id = tid + v * NT, r = id >> 3, c = id & 7, gn = wrow_of(r);
      if constexpr (WF == 1) {  // chunk c of the nibbles; the header for group c's scales
        const unsigned char* b = W + gn * wrow + (size_t)kt * 144;
        rb[v] = *reinterpret_cast<const uint4*>(b + 16 + c * 16);
        rh[v] = *reinterpret_cast<const uint4*>(b);
      } else {  // block c: rows are whole words and W 16-aligned, so an odd block starts 2 B into one
        const unsigned* w = reinterpret_cast<const unsigned*>(W + gn * wrow + (size_t)(kt * 8 + c) * 34 - 2 * (c & 1));
#pragma unroll
        for (int i = 0; i < 9; i++) rw[v][i] = w[i];
      }
    }
  };
  auto stage = [&]() {
#pragma unroll
    for (int v = 0; v < BM * 16 / NT; v++) {
      const unsigned id = tid + v * NT, r = id >> 4, c = id & 15;
      *reinterpret_cast<uint4*>(As + r * LDK + c * 16) = ra[v];
    }
#pragma unroll
    for (int v = 0; v < BM * 4 / NT; v++) {
      const unsigned id = tid + v * NT, g = id / (BM / 2), c = id % (BM / 2);
      *reinterpret_cast<uint4*>(&ASs[g * BM + c * 2]) = rs[v];
    }
#pragma unroll
    for (int v = 0; v < NB; v++) {
      const unsigned id = tid + v * NT, r = id >> 3, c = id & 7;
      if constexpr (WF == 1) {
        // 16 bytes of pair c / 2: low nibbles are group 2 (c / 2), high the next one.
        const unsigned p = c >> 1, h = c & 1;
        const uint4 x = rb[v];
        *reinterpret_cast<uint4*>(Bs + r * LDK + 64 * p + 16 * h) =
            make_uint4(x.x & 0x0F0F0F0Fu, x.y & 0x0F0F0F0Fu, x.z & 0x0F0F0F0Fu, x.w & 0x0F0F0F0Fu);
        *reinterpret_cast<uint4*>(Bs + r * LDK + 64 * p + 32 + 16 * h) =
            make_uint4((x.x >> 4) & 0x0F0F0F0Fu, (x.y >> 4) & 0x0F0F0F0Fu, (x.z >> 4) & 0x0F0F0F0Fu,
                       (x.w >> 4) & 0x0F0F0F0Fu);
        // Group c's d sc and dmin mn, from d, dmin and the 12 bytes of 6-bit scales.
        const uint4 hw = rh[v];
        const float2 dd = __half22float2(*reinterpret_cast<const __half2*>(&hw.x));
        int sc, mn;
        q4k_scale_min_(make_uint3(hw.y, hw.z, hw.w), (int)c, sc, mn);
        BSs[c * BN + r] = make_float2(dd.x * (float)sc, dd.y * (float)mn);
      } else {
        const unsigned* w = rw[v];
        const bool odd = c & 1;
        const float d = __half2float(__ushort_as_half((unsigned short)(odd ? w[0] >> 16 : w[0] & 0xFFFFu)));
        unsigned q[8];
#pragma unroll
        for (int i = 0; i < 8; i++) q[i] = odd ? w[i + 1] : __funnelshift_r(w[i], w[i + 1], 16);
        *reinterpret_cast<uint4*>(Bs + r * LDK + c * 32) = make_uint4(q[0], q[1], q[2], q[3]);
        *reinterpret_cast<uint4*>(Bs + r * LDK + c * 32 + 16) = make_uint4(q[4], q[5], q[6], q[7]);
        BSs[c * BN + r] = make_float2(d, 0.0f);
      }
    }
  };
  const unsigned la_r = lane & 15, la_c = (lane >> 4) * 16;  // A: 16 rows x two 16-byte halves
  const unsigned lb_r = (lane & 7) + ((lane >> 4) << 3), lb_c = ((lane >> 3) & 1) * 16;  // B: two n8 x two halves
  const unsigned g8 = lane >> 2, t2 = (lane & 3) * 2;
  fetch(kt0);
  for (unsigned kt = kt0; kt < kt1; kt++) {
    stage();
    __syncthreads();
    if (kt + 1 < kt1) fetch(kt + 1);  // in flight while this step multiplies
#pragma unroll
    for (int g = 0; g < 8; g++) {
      unsigned fa[FM][4], fb[FN][2];
#pragma unroll
      for (int i = 0; i < FM; i++)
        tc_ldsm4_(fa[i][0], fa[i][1], fa[i][2], fa[i][3], As + (wm + 16 * i + la_r) * LDK + g * 32 + la_c);
#pragma unroll
      for (int j = 0; j < FN; j += 2)
        tc_ldsm4_(fb[j][0], fb[j][1], fb[j + 1][0], fb[j + 1][1], Bs + (wn + 8 * j + lb_r) * LDK + g * 32 + lb_c);
      float2 as[FM][2];  // (d_a, d_a sum a) of rows g8 and g8 + 8
#pragma unroll
      for (int i = 0; i < FM; i++) {
        as[i][0] = ASs[g * BM + wm + 16 * i + g8];
        as[i][1] = ASs[g * BM + wm + 16 * i + g8 + 8];
      }
#pragma unroll
      for (int j = 0; j < FN; j++) {
        // (d_w, m_w) of columns t2 and t2 + 1.
        const float4 bs = *reinterpret_cast<const float4*>(&BSs[g * BN + wn + 8 * j + t2]);
#pragma unroll
        for (int i = 0; i < FM; i++) {
          int c[4];
          tc_mma_s8_(c, fa[i], fb[j][0], fb[j][1]);
          // c0, c1: row g8, columns t2, t2 + 1; c2, c3: row g8 + 8.
          acc[i][j][0] = fmaf((float)c[0], as[i][0].x * bs.x, acc[i][j][0]);
          acc[i][j][1] = fmaf((float)c[1], as[i][0].x * bs.z, acc[i][j][1]);
          acc[i][j][2] = fmaf((float)c[2], as[i][1].x * bs.x, acc[i][j][2]);
          acc[i][j][3] = fmaf((float)c[3], as[i][1].x * bs.z, acc[i][j][3]);
          if constexpr (WF == 1) {  // the min term
            acc[i][j][0] = fmaf(-as[i][0].y, bs.y, acc[i][j][0]);
            acc[i][j][1] = fmaf(-as[i][0].y, bs.w, acc[i][j][1]);
            acc[i][j][2] = fmaf(-as[i][1].y, bs.y, acc[i][j][2]);
            acc[i][j][3] = fmaf(-as[i][1].y, bs.w, acc[i][j][3]);
          }
        }
      }
    }
    __syncthreads();  // everyone is done with the stage before the next one overwrites it
  }
  if constexpr (GLU) {  // columns j < FN / 2 the gate, j + FN / 2 the up of the same outputs
    __half* H = static_cast<__half*>(Cv);
    auto glu = [](float g, float u) { return (g / (1.0f + expf(-g))) * u; };
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN / 2; j++) {
        const unsigned gr = row0 + wm + 16 * i + g8;
        const unsigned o = tn * (BN / 2) + (wn / WN) * 32 + 8 * j + t2;
        if (o >= N) continue;
        const float* gt = acc[i][j];
        const float* up = acc[i][j + FN / 2];
        if (gr < M)
          *reinterpret_cast<__half2*>(H + (size_t)gr * ldc + o) =
              __floats2half2_rn(glu(gt[0], up[0]), glu(gt[1], up[1]));
        if (gr + 8 < M)
          *reinterpret_cast<__half2*>(H + (size_t)(gr + 8) * ldc + o) =
              __floats2half2_rn(glu(gt[2], up[2]), glu(gt[3], up[3]));
      }
  } else {
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
      for (int j = 0; j < FN; j++) {
        const unsigned gr = row0 + wm + 16 * i + g8, gc = col0 + wn + 8 * j + t2;
        if (gc >= N) continue;
        if (gr < M) *reinterpret_cast<float2*>(C + (size_t)gr * ldc + gc) = make_float2(acc[i][j][0], acc[i][j][1]);
        if (gr + 8 < M) *reinterpret_cast<float2*>(C + (size_t)(gr + 8) * ldc + gc) = make_float2(acc[i][j][2], acc[i][j][3]);
      }
  }
}
// The int8 rows and their scales by 16 bytes, Q4_K by uint4 and Q8_0 by words; C by
// float2 (GLU: __half2).
#define TL_GEMM_MMQ(NAME, WF, GLU)                                                                   \
  TL_ALIGN(NAME, 16, 16, (WF) == 1 ? 16 : 4, (GLU) ? 4 : 8)                                   \
  extern "C" __global__ void __launch_bounds__(TL_MMQ_THREADS, 1)                              \
  NAME(const signed char* __restrict__ Aq, const float2* __restrict__ As_g,                    \
       const unsigned char* __restrict__ W, void* __restrict__ C, unsigned M, unsigned N,      \
       unsigned K, unsigned Mpad, unsigned ldc, unsigned kchunk, unsigned zstride) {            \
    gemm_mmq_core<WF, GLU>(Aq, As_g, W, C, M, N, K, Mpad, ldc, kchunk, zstride);                \
  }
TL_GEMM_MMQ(tl_gemm_mmq_q4k, 1, false)
TL_GEMM_MMQ(tl_gemm_mmq_q80, 3, false)
TL_GEMM_MMQ(tl_gemm_mmq_glu_q4k, 1, true)
TL_GEMM_MMQ(tl_gemm_mmq_glu_q80, 3, true)
#undef TL_GEMM_MMQ
// C[M, N columns ldc apart] = the sum of z partial products, each M x N, stride apart.
TL_ALIGN(tl_mmq_reduce, 8, 8)
extern "C" __global__ void tl_mmq_reduce(const float* __restrict__ part, float* __restrict__ C,
                                         unsigned M, unsigned N, unsigned ldc, unsigned z,
                                         unsigned stride) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x, half = N / 2;
  if (i >= M * half) return;
  const unsigned r = i / half, c = (i % half) * 2;
  float2 s = *reinterpret_cast<const float2*>(part + (size_t)r * N + c);
  for (unsigned p = 1; p < z; p++) {
    const float2 v = *reinterpret_cast<const float2*>(part + (size_t)p * stride + (size_t)r * N + c);
    s.x += v.x;
    s.y += v.y;
  }
  *reinterpret_cast<float2*>(C + (size_t)r * ldc + c) = s;
}
#undef TL_GEMM_TC

extern "C" {
// ---- M7 decode GEMV: y[n] = sum_k a[k] * B[k,n], F32 accumulate ----
// The decode regime is batch~1, so the matmul is a GEMV and MEMORY-BANDWIDTH-
// bound: the K×N weight B dominates traffic. One thread per output column;
// consecutive threads read consecutive columns B[k*n+col], so the B reads
// coalesce. a[k] is a warp-uniform broadcast. The f32 and bf16 variants are
// structurally identical — only B's dtype (and thus its byte traffic) differs,
// so their timing ratio isolates the storage-width win. bf16 widens on load
// with no precision loss beyond bf16's 8-bit mantissa; the accumulator is F32.
//
// Split-K over blockIdx.y: small-N layers (N/256 blocks) can't fill the SMs and
// would go occupancy-bound instead of bandwidth-bound, hiding the bf16 win. Each
// z-slice sums its K-range and atomicAdds into a pre-zeroed y (host memsets when
// gridDim.y>1); gridDim.y==1 stores directly and is bit-identical to no split.
// The f32 and bf16 kernels differ ONLY in how one weight element is loaded, so
// one macro emits both — their timing ratio then isolates the storage width and
// nothing else. LOAD is that expression, in terms of B and kk, the same way the
// elementwise macros at the top of this file take `a[i] + b[i]`.
#define TL_GEMV_SPLITK(NAME, BT, LOAD)                                        \
  __global__ void NAME(const float* __restrict__ a,                           \
                       const BT* __restrict__ B, float* __restrict__ y,       \
                       unsigned n, unsigned k, unsigned ksplit) {             \
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (col >= n) return;                                                     \
    unsigned k0 = blockIdx.y * ksplit;                                        \
    if (k0 >= k) return;                                                      \
    unsigned k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;                      \
    float acc = 0.0f;                                                         \
    for (unsigned kk = k0; kk < k1; kk++) acc += a[kk] * (LOAD);              \
    if (gridDim.y > 1)                                                        \
      atomicAdd(&y[col], acc);                                                \
    else                                                                      \
      y[col] = acc;                                                           \
  }
TL_GEMV_SPLITK(tl_gemv_f32, float, B[(size_t)kk * n + col])
TL_GEMV_SPLITK(tl_gemv_bf16, __nv_bfloat16,
               __bfloat162float(B[(size_t)kk * n + col]))
#undef TL_GEMV_SPLITK
// Vectorized bf16 GEMV: 8 columns/thread via one 16-byte (uint4 = 8×bf16) load,
// so each thread issues f32-width memory transactions instead of scalar 2-byte
// loads — closes the bandwidth gap to the f32 kernel. Requires n % 8 == 0 (all
// transformer dims are; the host gates on it and falls back to the scalar kernel
// otherwise), which keeps every row's load on 16 bytes when B's base is (TL_ALIGN).
TL_ALIGN(tl_gemv_bf16v8, 4, 16)  // B: 8 bf16 a uint4
__global__ void tl_gemv_bf16v8(const float* __restrict__ a,
                               const __nv_bfloat16* __restrict__ B,
                               float* __restrict__ y, unsigned n, unsigned k,
                               unsigned ksplit) {
  unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 8u;
  if (col >= n) return;
  unsigned k0 = blockIdx.y * ksplit;
  if (k0 >= k) return;
  unsigned k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;
  float acc[8] = {};
  for (unsigned kk = k0; kk < k1; kk++) {
    float av = a[kk];
    uint4 raw = *reinterpret_cast<const uint4*>(&B[(size_t)kk * n + col]);
    float2 f0 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.x));
    float2 f1 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.y));
    float2 f2 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.z));
    float2 f3 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.w));
    acc[0] += av * f0.x;
    acc[1] += av * f0.y;
    acc[2] += av * f1.x;
    acc[3] += av * f1.y;
    acc[4] += av * f2.x;
    acc[5] += av * f2.y;
    acc[6] += av * f3.x;
    acc[7] += av * f3.y;
  }
  bool split = gridDim.y > 1;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    if (split)
      atomicAdd(&y[col + j], acc[j]);
    else
      y[col + j] = acc[j];
  }
}

// Block-wide sum-reduce + store for the one-block-per-row GEMVs (tl_gemv_bf16_row,
// tl_gemv_q4): every lane brings its partial `acc` (already summed over its own
// K-slice); the block's total lands in *y_n. Warp-shuffle within each warp, then
// (only when nwarps>1) one round through dynamic shared memory. The launcher
// sizes the shared region (block>32 ? nwarps*4 : 0) and single-warp blocks return
// before ever dereferencing it. Assumes nwarps<=32 (block<=1024; the launcher
// caps block at 256, so 8). Shared here rather than templated on the fetch so the
// two extern "C" __global__ kernels — which differ ONLY in how they load each
// element (bf16 uint4 vs int4 dequant) — don't duplicate the reduction.
static __device__ __forceinline__ void gemv_row_block_store(
    float acc, unsigned lane, unsigned warp_id, unsigned nwarps, float* y_n) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (nwarps == 1) {
    if (lane == 0) *y_n = acc;
    return;
  }
  extern __shared__ float gemv_row_sdata[];
  if (lane == 0) gemv_row_sdata[warp_id] = acc;
  __syncthreads();
  if (warp_id == 0) {
    acc = (lane < nwarps) ? gemv_row_sdata[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) *y_n = acc;
  }
}

// ---- GEMVs over GGUF block-quantized weights ([N,K] rows of K/256 blocks, read where
// they lie: the decode is bound by the bytes it reads, and a Q4_K weight is
// 0.56 byte). One block of 4 warps a row, as tl_gemv_bf16_row; the dequantized
// value is never stored, each lane folds it into the dot product. The formats
// are ggml's (include/ggml_quant.h has them in words); a block's d / scales
// are read by every lane that needs them, from L1.

// Q4_K, 144 bytes a block, a warp a block: lane = 8 * pair + j reads 4 bytes of
// the pair's 32 (8 weights: 4 low nibbles of group 2*pair, 4 high of 2*pair+1)
// and the 4 + 4 activations they multiply.
// y_n = sum d*sc*q*a - dmin*m*a.
TL_ALIGN(tl_gemv_q4k_row, 16, 4)  // a by float4; Q4_K by words
__global__ void tl_gemv_q4k_row(const float* __restrict__ a, const unsigned char* __restrict__ W,
                                float* __restrict__ y, unsigned N, unsigned K) {
  const unsigned n = blockIdx.x;
  if (n >= N) return;
  const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const unsigned nb = K / 256u, p = lane >> 3, j = lane & 7u;
  const unsigned char* row = W + (size_t)n * nb * 144u;
  float acc = 0.0f;
  for (unsigned ib = warp; ib < nb; ib += nwarps) {
    const unsigned char* b = row + (size_t)ib * 144u;
    const float2 dd = __half22float2(*reinterpret_cast<const __half2*>(b));  // d, dmin
    const uint3 sw = *reinterpret_cast<const uint3*>(b + 4);
    const unsigned w = *reinterpret_cast<const unsigned*>(b + 16 + 32 * p + 4 * j);
    const float* ap = a + (size_t)ib * 256u + 64u * p + 4u * j;
    const float4 alo = *reinterpret_cast<const float4*>(ap), ahi = *reinterpret_cast<const float4*>(ap + 32);
    const float lo[4] = {(float)(w & 0xFu), (float)((w >> 8) & 0xFu), (float)((w >> 16) & 0xFu), (float)((w >> 24) & 0xFu)};
    const float hi[4] = {(float)((w >> 4) & 0xFu), (float)((w >> 12) & 0xFu), (float)((w >> 20) & 0xFu), (float)((w >> 28) & 0xFu)};
    const float sql = lo[0] * alo.x + lo[1] * alo.y + lo[2] * alo.z + lo[3] * alo.w;
    const float sqh = hi[0] * ahi.x + hi[1] * ahi.y + hi[2] * ahi.z + hi[3] * ahi.w;
    int s0, m0, s1, m1;  // scale / min of groups 2p and 2p+1
    q4k_scale_min_(sw, 2 * (int)p, s0, m0);
    q4k_scale_min_(sw, 2 * (int)p + 1, s1, m1);
    const float suml = alo.x + alo.y + alo.z + alo.w, sumh = ahi.x + ahi.y + ahi.z + ahi.w;
    acc += dd.x * (s0 * sql + s1 * sqh) - dd.y * (m0 * suml + m1 * sumh);
  }
  gemv_row_block_store(acc, lane, warp, nwarps, y + n);
}

// ---- Q4_K . int8 activations (llama.cpp's q8_1 scheme, and what its Q4_K GEMV does).
// The float kernel above spends its time on the activation loads: 8 float4 for
// every 32 weights, each instruction spread over several cache lines (a Q4_K
// GEMV at ~450 GB/s where the weights alone stream at ~850). Quantizing the
// activation once per call to int8 (a scale d = max|x| / 127 per 32 values, and
// the sum of the original 32, which the K-quants' min terms multiply) cuts those
// bytes four-fold and turns the multiplies into dp4a, 4 a lane-instruction:
// ~750 GB/s. The price is the rounding of the activation to 8 bits, as in
// llama.cpp: ~0.5% relative error on a random matvec.
// Layout of the quantized activation: K int8, then K/32 float2 (d, sum).
TL_ALIGN(tl_quant_act_q8, 4, 1, 8)
__global__ void tl_quant_act_q8(const float* __restrict__ a, signed char* __restrict__ qs,
                                float2* __restrict__ ds, unsigned nblk) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (blk >= nblk) return;
  const float x = a[(size_t)blk * 32u + lane];
  float amax = fabsf(x), sum = x;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    sum += __shfl_xor_sync(0xffffffffu, sum, off);
  }
  const float d = amax / 127.0f;
  qs[(size_t)blk * 32u + lane] = (signed char)(d == 0.0f ? 0 : __float2int_rn(x / d));
  if (lane == 0) ds[blk] = make_float2(d, sum);
}

// A warp a row, four rows a block; lane = 8 * superblock + 2 * pair + half reads 16
// bytes of its superblock (16 low-nibble weights of group 2*pair and 16 high of
// group 2*pair+1, positions 16*half..) and the 16 + 16 int8 activations they meet.
// The group's min term (dmin * m * sum of the group) is taken by the half-0 lane.
TL_ALIGN(tl_gemv_q4k_q8_row, 16, 8, 16)
__global__ void tl_gemv_q4k_q8_row(const signed char* __restrict__ qa, const float2* __restrict__ ds,
                                   const unsigned char* __restrict__ W, float* __restrict__ y,
                                   unsigned N, unsigned K) {
  const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const unsigned n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const unsigned nb = K / 256u, sbl = lane >> 3, p = (lane & 7u) >> 1, hf = lane & 1u;
  const unsigned char* row = W + (size_t)n * nb * 144u;
  float acc = 0.0f;
  for (unsigned ib = sbl; ib < nb; ib += 4) {
    const unsigned char* b = row + (size_t)ib * 144u;
    const float2 dd = __half22float2(*reinterpret_cast<const __half2*>(b));  // d, dmin
    const uint3 sw = *reinterpret_cast<const uint3*>(b + 4);
    const uint4 q = *reinterpret_cast<const uint4*>(b + 16 + 32 * p + 16 * hf);
    const signed char* ap = qa + (size_t)ib * 256u + 64u * p + 16u * hf;
    const uint4 al = *reinterpret_cast<const uint4*>(ap), ah = *reinterpret_cast<const uint4*>(ap + 32);
    int sl = 0, sh = 0;
    sl = __dp4a((int)(q.x & 0x0F0F0F0Fu), (int)al.x, sl);
    sl = __dp4a((int)(q.y & 0x0F0F0F0Fu), (int)al.y, sl);
    sl = __dp4a((int)(q.z & 0x0F0F0F0Fu), (int)al.z, sl);
    sl = __dp4a((int)(q.w & 0x0F0F0F0Fu), (int)al.w, sl);
    sh = __dp4a((int)((q.x >> 4) & 0x0F0F0F0Fu), (int)ah.x, sh);
    sh = __dp4a((int)((q.y >> 4) & 0x0F0F0F0Fu), (int)ah.y, sh);
    sh = __dp4a((int)((q.z >> 4) & 0x0F0F0F0Fu), (int)ah.z, sh);
    sh = __dp4a((int)((q.w >> 4) & 0x0F0F0F0Fu), (int)ah.w, sh);
    const int g0 = 2 * (int)p, g1 = g0 + 1;  // scale / min of groups 2p, 2p+1
    int s0, m0, s1, m1;
    q4k_scale_min_(sw, g0, s0, m0);
    q4k_scale_min_(sw, g1, s1, m1);
    const float2 d0 = ds[ib * 8u + g0], d1 = ds[ib * 8u + g1];
    float t = dd.x * (d0.x * (float)(s0 * sl) + d1.x * (float)(s1 * sh));
    if (hf == 0) t -= dd.y * (d0.y * (float)m0 + d1.y * (float)m1);
    acc += t;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[n] = acc;
}

// Q8_0, 34 bytes a block of 32 (2-byte aligned only): a warp takes eight blocks
// at a time, lane = 4 * which + j reading the 8 bytes 8j.. of its block and the
// 8 activations they multiply. y_n = sum d * q * a.
TL_ALIGN(tl_gemv_q80_row, 16, 2)  // a by float4; Q8_0 by halves
__global__ void tl_gemv_q80_row(const float* __restrict__ a, const unsigned char* __restrict__ W,
                                float* __restrict__ y, unsigned N, unsigned K) {
  const unsigned n = blockIdx.x;
  if (n >= N) return;
  const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const unsigned nb = K / 32u, j = lane & 3u;
  const unsigned char* row = W + (size_t)n * nb * 34u;
  float acc = 0.0f;
  for (unsigned ib = warp * 8u + (lane >> 2); ib < nb; ib += nwarps * 8u) {
    const unsigned char* b = row + (size_t)ib * 34u;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const unsigned short* q16 = reinterpret_cast<const unsigned short*>(b + 2 + 8 * j);
    const float* ap = a + (size_t)ib * 32u + 8u * j;
    const float4 a0 = *reinterpret_cast<const float4*>(ap), a1 = *reinterpret_cast<const float4*>(ap + 4);
    const float v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned w = q16[i];
      s += (float)(signed char)(w & 0xFFu) * v[2 * i] + (float)(signed char)(w >> 8) * v[2 * i + 1];
    }
    acc += d * s;
  }
  gemv_row_block_store(acc, lane, warp, nwarps, y + n);
}

// Q6_K, 210 bytes a block (2-byte aligned only, hence the 16-bit loads): a warp
// takes two blocks at a time, lane = 16 * which + 8 * half + j reading the 4
// consecutive positions 4j.. of its 128-weight half, in each of the four
// 32-weight quarters (4 + 4 bytes of ql, 4 of qh, 16 activations).
__device__ __forceinline__ unsigned ld32_h(const unsigned char* p) {
  return (unsigned)*reinterpret_cast<const unsigned short*>(p) |
         ((unsigned)*reinterpret_cast<const unsigned short*>(p + 2) << 16);
}
TL_ALIGN(tl_gemv_q6k_row, 16, 2)  // a by float4; Q6_K by halves
__global__ void tl_gemv_q6k_row(const float* __restrict__ a, const unsigned char* __restrict__ W,
                                float* __restrict__ y, unsigned N, unsigned K) {
  const unsigned n = blockIdx.x;
  if (n >= N) return;
  const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const unsigned nb = K / 256u, r = lane & 15u, h = r >> 3, j = r & 7u;
  const unsigned char* row = W + (size_t)n * nb * 210u;
  float acc = 0.0f;
  for (unsigned ib = warp * 2u + (lane >> 4); ib < nb; ib += nwarps * 2u) {
    const unsigned char* b = row + (size_t)ib * 210u;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const unsigned qa = ld32_h(b + 64 * h + 4 * j), qb = ld32_h(b + 64 * h + 32 + 4 * j),
                   qh = ld32_h(b + 128 + 32 * h + 4 * j);
    const signed char* sc = reinterpret_cast<const signed char*>(b + 192) + 8 * h + (j >> 2);
    const float* ap = a + (size_t)ib * 256u + 128u * h + 4u * j;
    const float4 a0 = *reinterpret_cast<const float4*>(ap), a1 = *reinterpret_cast<const float4*>(ap + 32),
                 a2 = *reinterpret_cast<const float4*>(ap + 64), a3 = *reinterpret_cast<const float4*>(ap + 96);
    const float v0[4] = {a0.x, a0.y, a0.z, a0.w}, v1[4] = {a1.x, a1.y, a1.z, a1.w},
                v2[4] = {a2.x, a2.y, a2.z, a2.w}, v3[4] = {a3.x, a3.y, a3.z, a3.w};
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned la = (qa >> (8 * i)) & 0xFFu, lb = (qb >> (8 * i)) & 0xFFu, hq = (qh >> (8 * i)) & 0xFFu;
      s0 += (float)((int)((la & 0xF) | ((hq & 3u) << 4)) - 32) * v0[i];
      s1 += (float)((int)((lb & 0xF) | (((hq >> 2) & 3u) << 4)) - 32) * v1[i];
      s2 += (float)((int)((la >> 4) | (((hq >> 4) & 3u) << 4)) - 32) * v2[i];
      s3 += (float)((int)((lb >> 4) | (((hq >> 6) & 3u) << 4)) - 32) * v3[i];
    }
    acc += d * ((float)sc[0] * s0 + (float)sc[2] * s1 + (float)sc[4] * s2 + (float)sc[6] * s3);
  }
  gemv_row_block_store(acc, lane, warp, nwarps, y + n);
}

// Warp-per-row bf16 GEMV (M9 decode lever A): weights in [N,K] (K contiguous per
// output row = GGML-native, so the loader drops the transpose). ONE BLOCK per
// output row n (grid.x == N); its threads split K, each lane consuming 8 bf16
// per step via one 16-byte uint4 load — consecutive lanes read consecutive 16B,
// so each warp issues one coalesced 512-byte transaction — MAC into an F32
// accumulator, warp-shuffle-reduce, then (when the block has >1 warp) a shared-
// mem cross-warp reduce. NO split-K, so NO MemsetD8Async prezero and NO
// atomicAdd combine: one launch, one grid.
//
// Block size is chosen host-side per K (gemv_row_block_size, 32..256 threads —
// llama.cpp's mul_mat_vec_f strategy) so a wide row (large K) gets more warps
// collaborating on the reduction, while grid.x == N (not N/8 as the earlier
// 8-rows/block packing did) so small-N shapes (e.g. Qwen wk/wv, N=128) still
// get one block per SM instead of leaving most of the GPU idle. Requires
// K % 8 == 0 (every transformer dim); the last partial block is handled by the
// per-thread k0 < K guard (tail threads simply skip — no K % 256 requirement).
TL_ALIGN(tl_gemv_bf16_row, 4, 16)  // B: 8 bf16 a uint4
__global__ void tl_gemv_bf16_row(const float* __restrict__ a,
                                 const __nv_bfloat16* __restrict__ B,
                                 float* __restrict__ y, unsigned N, unsigned K) {
  const unsigned n = blockIdx.x;
  if (n >= N) return;
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u;
  const unsigned warp_id = tid >> 5;
  const unsigned nwarps = blockDim.x >> 5;
  const __nv_bfloat16* row = B + (size_t)n * K;
  float acc = 0.0f;
  for (unsigned k0 = tid * 8u; k0 < K; k0 += blockDim.x * 8u) {
    uint4 raw = *reinterpret_cast<const uint4*>(&row[k0]);
    float2 f0 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.x));
    float2 f1 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.y));
    float2 f2 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.z));
    float2 f3 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.w));
    acc += a[k0 + 0] * f0.x + a[k0 + 1] * f0.y + a[k0 + 2] * f1.x +
           a[k0 + 3] * f1.y + a[k0 + 4] * f2.x + a[k0 + 5] * f2.y +
           a[k0 + 6] * f3.x + a[k0 + 7] * f3.y;
  }
  gemv_row_block_store(acc, lane, warp_id, nwarps, &y[n]);
}

// ---- M8 int4-weight decode GEMV: y[n] = sum_k a[k] * dequant(Wq[n,k]) ----
// The quantized-inference heart: weights are group-symmetric int4 in [N,K]
// (out×in) layout so the K-axis quantization groups are contiguous (GGUF/GPTQ
// convention). Decode is bandwidth-bound, and int4 reads ~0.625 bytes/weight
// (0.5 packed + one f32 scale per group) vs bf16's 2 — the biggest remaining
// decode lever.
//
// ONE BLOCK per output row n (grid.x == N), block size K-adaptive (same
// gemv_row_block_size search as tl_gemv_bf16_row — llama.cpp's mul_mat_vec_f
// strategy): each thread dequantizes one uint32 (8 packed int4) per step,
// strided by blockDim.x across K, MACs into an F32 accumulator, then warp-
// shuffle-reduce + (when >1 warp) a shared-mem cross-warp reduce. Every thread
// in the block reads a DISJOINT slice of K, so — unlike the old fixed-32-lane/
// 8-rows-per-block layout — there's no redundant re-read of `a` for a shared-
// memory staging pass to amortize; one kernel replaces the former global-a/
// shared-a pair. Requires K % group == 0 and group % 8 == 0 (every transformer
// dim; group=32); the per-thread k0 < K guard handles the tail (e.g. Qwen's
// K=896) — no K % 256 requirement.
__global__ void tl_gemv_q4(const float* __restrict__ a,
                           const unsigned* __restrict__ qw,   // [N][K/8] words
                           const float* __restrict__ scales,  // [N][K/G]
                           float* __restrict__ y, unsigned N, unsigned K,
                           unsigned G) {
  const unsigned n = blockIdx.x;
  if (n >= N) return;
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u;
  const unsigned warp_id = tid >> 5;
  const unsigned nwarps = blockDim.x >> 5;
  const unsigned* qrow = qw + (size_t)n * (K >> 3);
  const float* srow = scales + (size_t)n * (K / G);
  float acc = 0.0f;
  for (unsigned k0 = tid * 8u; k0 < K; k0 += blockDim.x * 8u) {
    unsigned w = qrow[k0 >> 3];
    float sc = srow[k0 / G];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int q = (int)((w >> (j * 4)) & 0xFu) - 8;
      acc += a[k0 + j] * (sc * (float)q);
    }
  }
  gemv_row_block_store(acc, lane, warp_id, nwarps, &y[n]);
}

// ---- M9 fused decode attention (flash-attention, one query row per head) ----
// out(h,:) = softmax(scale * q(h,:) · K(h)^T) · V(h), computed in ONE pass with
// the online-softmax recurrence so the ctx-long scores are never materialized
// and K,V are each read exactly once (the decode floor is KV bandwidth). This
// replaces the array path's 3 launches × 2 materializations × (heads×layers)
// — which is launch/materialize-bound, not compute-bound — for the regime that
// dominates decode-token time.
//
// One block per head, blockDim = head_dim AD (NW = AD/32 warps). Each warp owns
// keys i = warp, warp+NW, ...; its 32 lanes cooperatively reduce the AD-long
// score (coalesced K/V row reads, warp-shuffle reduction, no block sync in the
// loop), keeping per-warp (m,l,acc) running state. A final shared-memory step
// merges the NW warps' partial softmax states (flash rescale-by-exp(m_w-m)).
//
// head_dim generalization (M9): the kernels are templated on AD (compile-time,
// because acc[]/sacc[][] are register/smem arrays sized AD/32) and instantiated
// for {64,128} — the head dims of the target models (Qwen2 = 64, llama-7B = 128)
// — and 256 (Qwen3.5's full-attention layers), each for an f32 and an f16 cache.
// Each lane holds NW = AD/32 dims (d = lane, lane+32, ..., lane+(NW-1)*32) and
// there are exactly NW warps, so the warp count and the per-lane dim count
// coincide. extern "C" wrappers below give each instantiation a stable symbol
// (the driver loads kernels by name): tl_attn_decode_{f32,f16}_{64,128,256}, etc.

}  // extern "C"
// The KV cache's element type: the cores below are templated on KT ∈ {float, __half}.
// kv_ld widens a cached element to f32 for the dot and the accumulation, kv_st rounds
// an f32 projection to nearest as it is stored. KT = float is the identity; KT = __half
// halves the cache's bytes (the decode streams them every step) and holds what the
// tensor-core prefill rounds K and V to anyway. q, out and scratch stay f32;
// kv_stride and every index count elements, so only the element width changes.
__device__ __forceinline__ float kv_ld(float x) { return x; }
__device__ __forceinline__ float kv_ld(__half x) { return __half2float(x); }
__device__ __forceinline__ void kv_st(float* p, float x) { *p = x; }
__device__ __forceinline__ void kv_st(__half* p, float x) { *p = __float2half_rn(x); }

// Shared middle of the decode-attention family. Stages the head's query into
// shared memory, runs the online-softmax score loop over keys [k0, k1) (each
// warp owns keys i = k0+warp, +NW, ...; its 32 lanes cooperatively reduce the
// AD-long dot), then merges the NW warps' partial states. Leaves every thread
// holding the block-level (gm, gl) and its dim's UN-normalized accumulator `o`:
// the single-pass kernels normalize and store o/gl, the split kernel writes all
// three as partials for tl_attn_combine. An empty range (k0 >= k1 — a dpos
// split past the live count) yields the neutral partial (gm=-1e30, gl=0, o=0)
// that combines as exact zeros.
template <int AD, typename KT>
__device__ __forceinline__ void attn_span_core(
    const float* __restrict__ qh, const KT* __restrict__ Kh,
    const KT* __restrict__ Vh, unsigned k0, unsigned k1, float scale,
    float& gm_out, float& gl_out, float& o_out) {
  constexpr int NW = AD / 32;                        // warps == dims-per-lane
  const unsigned tid = threadIdx.x;                  // 0..AD-1
  const unsigned lane = tid & 31u, warp = tid >> 5;  // NW warps of 32
  __shared__ float q_sh[AD];
  q_sh[tid] = qh[tid];
  __syncthreads();

  float m = -1e30f, l = 0.0f;
  float acc[NW] = {};  // lane holds dims d = lane, lane+32, ..., lane+(NW-1)*32
  // A warp's keys, U at a time: the U K rows are read and their scores reduced
  // together, so each key's load latency overlaps the others' instead of
  // standing alone at the head of a serial chain; the softmax update and the V
  // accumulation then take the keys one by one, in the order they would have
  // gone alone, so the sums are the same. (A tail of fewer than U keys masks
  // the rest out.)
  constexpr int U = 4;
  for (unsigned i0 = k0 + warp; i0 < k1; i0 += (unsigned)(NW * U)) {
    float sc[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned i = i0 + (unsigned)(u * NW);
      float partial = 0.0f;
      if (i < k1) {
        const KT* Ki = Kh + (size_t)i * AD;
#pragma unroll
        for (int r = 0; r < NW; r++)
          partial += q_sh[lane + r * 32] * kv_ld(Ki[lane + r * 32]);
      }
      sc[u] = partial;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
      for (int u = 0; u < U; u++)
        sc[u] += __shfl_down_sync(0xffffffffu, sc[u], off);
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned i = i0 + (unsigned)(u * NW);
      if (i >= k1) break;
      float s = __shfl_sync(0xffffffffu, sc[u], 0) * scale;  // lane0's sum
      float m_new = fmaxf(m, s);
      float corr = __expf(m - m_new);
      float p = __expf(s - m_new);
      l = l * corr + p;
      const KT* Vi = Vh + (size_t)i * AD;
#pragma unroll
      for (int r = 0; r < NW; r++)
        acc[r] = acc[r] * corr + p * kv_ld(Vi[lane + r * 32]);
      m = m_new;
    }
  }

  // Merge the NW warps' partial softmax states through shared memory (warps
  // whose key range was empty carry m=-1e30, l=0 and contribute exp(-inf)=0).
  __shared__ float sm[NW], sl[NW], sacc[NW][AD];
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
#pragma unroll
  for (int r = 0; r < NW; r++) sacc[warp][lane + r * 32] = acc[r];
  __syncthreads();

  float gm = -1e30f;
#pragma unroll
  for (int w = 0; w < NW; w++) gm = fmaxf(gm, sm[w]);
  float gl = 0.0f, o = 0.0f;
#pragma unroll
  for (int w = 0; w < NW; w++) {
    float e = __expf(sm[w] - gm);
    gl += sl[w] * e;
    o += sacc[w][tid] * e;
  }
  gm_out = gm;
  gl_out = gl;
  o_out = o;
}

template <int AD, typename KT = float>
__device__ void attn_decode_core(const float* __restrict__ q,
                                  const KT* __restrict__ K,
                                  const KT* __restrict__ V,
                                  float* __restrict__ out, unsigned ctx,
                                  unsigned kv_stride, unsigned group,
                                  float scale) {
  const unsigned h = blockIdx.x;                // query head
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  // K,V are [H_kv, max_ctx, D]; kv_stride = max_ctx*D lets a persistent cache be
  // read as its valid prefix [0,ctx) (kv_stride==ctx*AD is the no-cache case).
  float gm, gl, o;
  attn_span_core<AD, KT>(q + (size_t)h * AD, K + (size_t)kv_h * kv_stride,
                         V + (size_t)kv_h * kv_stride, 0, ctx, scale, gm, gl, o);
  out[(size_t)h * AD + threadIdx.x] = o / gl;
}
// tl_attn_decode_{f32,f16}_{64,128,256}
#define TL_ATTN_DECODE(AD, KT, TN)                                                             \
  extern "C" __global__ void tl_attn_decode_##TN##_##AD(const float* q, const KT* K, const KT* V, \
                                                        float* out, unsigned ctx,                \
                                                        unsigned kv_stride, unsigned group,      \
                                                        float scale) {                           \
    attn_decode_core<AD, KT>(q, K, V, out, ctx, kv_stride, group, scale);                       \
  }
TL_ATTN_DECODE(64, float, f32)
TL_ATTN_DECODE(128, float, f32)
TL_ATTN_DECODE(256, float, f32)
TL_ATTN_DECODE(64, __half, f16)
TL_ATTN_DECODE(128, __half, f16)
TL_ATTN_DECODE(256, __half, f16)
#undef TL_ATTN_DECODE

// Split-KV (flash-decoding): the one-block-per-head kernel above launches only
// H blocks — a few % of the 82 SMs, so it's occupancy-bound. Partition ctx over
// gridDim.y so grid = H×S fills the SMs; each (head,split) block writes its
// partial softmax state (m,l,acc at its local max) to scratch, and tl_attn_
// combine merges the S partials per head. K,V are still each read exactly once.
template <int AD, typename KT = float>
__device__ void attn_decode_split_core(const float* __restrict__ q,
                                        const KT* __restrict__ K,
                                        const KT* __restrict__ V,
                                        float* __restrict__ pm,
                                        float* __restrict__ pl,
                                        float* __restrict__ pacc, unsigned ctx,
                                        unsigned kv_stride, unsigned group,
                                        unsigned chunk, float scale) {
  const unsigned h = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  const unsigned k0 = sp * chunk;
  const unsigned k1 = (k0 + chunk < ctx) ? (k0 + chunk) : ctx;
  float gm, gl, o;
  attn_span_core<AD, KT>(q + (size_t)h * AD, K + (size_t)kv_h * kv_stride,
                         V + (size_t)kv_h * kv_stride, k0, k1, scale, gm, gl, o);
  const size_t sidx = (size_t)h * S + sp;
  if (threadIdx.x == 0) {
    pm[sidx] = gm;
    pl[sidx] = gl;
  }
  pacc[sidx * AD + threadIdx.x] = o;  // un-normalized accumulator at local max
}
// tl_attn_decode_split_{f32,f16}_{64,128,256}
#define TL_ATTN_SPLIT(AD, KT, TN)                                                                    \
  extern "C" __global__ void tl_attn_decode_split_##TN##_##AD(                                       \
      const float* q, const KT* K, const KT* V, float* pm, float* pl, float* pacc, unsigned ctx,     \
      unsigned kv_stride, unsigned group, unsigned chunk, float scale) {                             \
    attn_decode_split_core<AD, KT>(q, K, V, pm, pl, pacc, ctx, kv_stride, group, chunk, scale);     \
  }
TL_ATTN_SPLIT(64, float, f32)
TL_ATTN_SPLIT(128, float, f32)
TL_ATTN_SPLIT(256, float, f32)
TL_ATTN_SPLIT(64, __half, f16)
TL_ATTN_SPLIT(128, __half, f16)
TL_ATTN_SPLIT(256, __half, f16)
#undef TL_ATTN_SPLIT

// Merge the S per-head partials (each already at its local max) into out. Not
// templated: the per-dim thread stride D == blockDim.x, so head_dim is implicit.
extern "C" __global__ void tl_attn_combine(const float* __restrict__ pm,
                                           const float* __restrict__ pl,
                                           const float* __restrict__ pacc,
                                           float* __restrict__ out, unsigned S) {
  const unsigned h = blockIdx.x, tid = threadIdx.x, D = blockDim.x;
  const size_t base = (size_t)h * S;
  float gm = -1e30f;
  for (unsigned s = 0; s < S; s++) gm = fmaxf(gm, pm[base + s]);
  float gl = 0.0f, o = 0.0f;
  for (unsigned s = 0; s < S; s++) {
    float e = __expf(pm[base + s] - gm);
    gl += pl[base + s] * e;
    o += pacc[(base + s) * D + tid] * e;
  }
  out[(size_t)h * D + tid] = o / gl;
}

// ---- Tiled causal prefill attention (flash-attention shape) ----
// Causal prefill attention: all T query positions of a prompt at once. Query at
// position pos0+p attends keys 0..pos0+p, so a long prompt runs in chunks and a
// later turn extends a live cache. q,out are [H_q, T, D]; K,V are the
// [H_kv, kv_max, D] cache read over [0,pos0+p] via kv_stride.
//
// The first cut ran one block per (head, query) and reused the decode kernels'
// online softmax directly — correct, but it warp-shuffle-reduced a dot product
// for EVERY key, so at Qwen's D=64 each key cost 2 useful FMAs and ~7 reduction
// ops: 1.67 TFLOP/s at T=2048, 5% of peak, while the KV it streamed (2 MB for 2
// KV heads) sat in L2 the whole time — never a bandwidth problem. Tiling fixes
// the arithmetic instead: a block owns a tile of queries, streams K/V through
// shared memory in TK-key tiles, and holds each score as a plain register dot
// product with no cross-lane reduction at all.
//
// Layout: 128 threads = ROWS(16) row slots x LANES(8). Lane `dg` of row slot
// `qr0` owns QPT queries (qr0, qr0+ROWS, ...) and, of each, dims {d*LANES+dg}
// of the accumulator plus SPT of its scores. Two consequences:
//   - a row's 8 lanes are 8 consecutive lanes of one warp, so its online-softmax
//     max/sum are __shfl_xor reductions over 3 steps once per TILE, not per key;
//   - the accumulator dims are INTERLEAVED across lanes, so the PV loop's
//     Vs[kk][d*LANES+dg] and the final store are consecutive across lanes
//     (a lane-blocked dg*DG+d ownership makes both 2-way conflicted).
// Online softmax (running max/sum, rescale in flight) is the same recurrence the
// decode kernels use, so no T x T score matrix ever exists.
//
// QPT — queries per thread — is the lever that makes the tiling pay. At QPT=1
// every shared word feeds exactly one FMA and the kernel is shared-load-issue
// bound (measured ~5x oversubscribed against its own math, 9-11% of FP32 peak);
// each extra query reuses the same K/V word for another FMA. Measured at T=512:
// QPT=2 is 1.6-1.8x and QPT=4 is 2.2-2.5x QPT=1, bit-identical output. D=64
// takes QPT=4; D=128 takes 2 because Qs alone would otherwise exceed the 48 KB
// static shared limit. Occupancy is a red herring here — QPT=4 runs at 16.7%
// and is twice as fast as a 58% variant.
//
// Causal: row slot qr0's q-th query is at absolute position pos0 + qbase +
// qr0 + q*ROWS and attends keys 0..that. Whole tiles past the block's last row
// are never loaded. Blocks are also mapped to query tiles in REVERSE, because
// work grows with the tile index and blocks dispatch in order: launching the
// expensive tiles first leaves a cheap tail instead of a maximally expensive one
// (+9% at pos0=0, neutral once pos0 dominates).

// Widen four consecutive cached elements. The rows are AD-contiguous and AD % 4
// == 0, so the address is 16-byte aligned for f32 and 8-byte for f16 when the
// base is — one vector load per four elements instead of four scalar ones (+9%
// on the staging).
__device__ __forceinline__ void kv_ld4(const float* p, float* o) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
__device__ __forceinline__ void kv_ld4(const __half* p, float* o) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const float2 f0 = __half22float2(reinterpret_cast<const __half2&>(raw.x));
  const float2 f1 = __half22float2(reinterpret_cast<const __half2&>(raw.y));
  o[0] = f0.x; o[1] = f0.y; o[2] = f1.x; o[3] = f1.y;
}
// The same four by vector when `vec`, else one element at a time.
template <typename KT>
__device__ __forceinline__ void kv_ld4(const KT* p, float* o, bool vec) {
  if (vec) {
    kv_ld4(p, o);
  } else {
#pragma unroll
    for (int e = 0; e < 4; e++) o[e] = kv_ld(p[e]);
  }
}

template <int AD, int QPT, typename KT = float>
__device__ void attn_prefill_tiled_core(const float* __restrict__ q,
                                        const KT* __restrict__ K,
                                        const KT* __restrict__ V,
                                        float* __restrict__ out, unsigned T,
                                        unsigned kv_stride, unsigned group,
                                        float scale, unsigned pos0) {
  constexpr int NT = 128, LANES = 8, ROWS = NT / LANES;
  constexpr int BQ = ROWS * QPT;            // queries per block
  constexpr int TK = AD == 64 ? 32 : AD == 128 ? 16 : 8;  // keys per tile (shared budget)
  constexpr int DG = AD / LANES;            // accumulator dims per lane
  constexpr int SPT = TK / LANES;           // scores per lane per tile
  // +1 of row padding on Qs, Ks and Ps: each is read down a column by lanes that
  // differ only in the ROW index, and the row strides (AD, TK) are multiples of
  // 32, so without it all those lanes land in one bank — 8-way on the score
  // loop's Ks, 4-way on Qs and Ps. (Vs is read across its row, and the
  // interleaved dim ownership already spreads that.)
  __shared__ float Qs[BQ][AD + 1];
  __shared__ float Ks[TK][AD + 1];
  __shared__ float Vs[TK][AD];
  __shared__ float Ps[BQ][TK + 1];

  const unsigned h = blockIdx.x;
  const unsigned qbase = (gridDim.y - 1u - blockIdx.y) * BQ;  // reversed order
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  const KT* Kh = K + (size_t)kv_h * kv_stride;
  const KT* Vh = V + (size_t)kv_h * kv_stride;
  // kv_stride is a multiple of AD, so the heads keep the base's alignment.
  const bool vecK = (size_t)K % (4 * sizeof(KT)) == 0;
  const bool vecV = (size_t)V % (4 * sizeof(KT)) == 0;
  const unsigned tid = threadIdx.x;
  const unsigned qr0 = tid >> 3, dg = tid & 7u;

  for (unsigned i = tid; i < BQ * AD; i += NT) {
    const unsigned r = i / AD, c = i % AD;
    Qs[r][c] = (qbase + r < T) ? q[((size_t)h * T + qbase + r) * AD + c] : 0.f;
  }
  // Last key any row of this block can need.
  const unsigned last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const unsigned kmax = pos0 + last;

  float acc[QPT][DG] = {};
  float m[QPT], l[QPT];
#pragma unroll
  for (int u = 0; u < QPT; u++) { m[u] = -1e30f; l[u] = 0.0f; }

  for (unsigned kt = 0; kt <= kmax; kt += TK) {
    __syncthreads();  // previous tile's Ks/Vs are done being read
    for (unsigned i = tid; i < TK * AD / 4; i += NT) {
      const unsigned r = i / (AD / 4), c4 = (i % (AD / 4)) * 4;
      const unsigned kk = kt + r;
      float kb[4] = {0, 0, 0, 0}, vb[4] = {0, 0, 0, 0};
      if (kk <= kmax) {
        kv_ld4(Kh + (size_t)kk * AD + c4, kb, vecK);
        kv_ld4(Vh + (size_t)kk * AD + c4, vb, vecV);
      }
#pragma unroll
      for (int e = 0; e < 4; e++) {
        Ks[r][c4 + e] = kb[e];
        Vs[r][c4 + e] = vb[e];
      }
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < QPT; u++) {
      const unsigned row = qr0 + u * ROWS;
      const unsigned pabs = pos0 + qbase + row;
      float sc[SPT], mt = -1e30f;
#pragma unroll
      for (int t2 = 0; t2 < SPT; t2++) {
        const unsigned kk = kt + dg * SPT + t2;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < AD; d++) dot += Qs[row][d] * Ks[dg * SPT + t2][d];
        sc[t2] = (kk <= pabs) ? dot * scale : -1e30f;  // causal mask
        mt = fmaxf(mt, sc[t2]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
      const float m_new = fmaxf(m[u], mt);
      const float corr = __expf(m[u] - m_new);
      float ls = 0.0f;
#pragma unroll
      for (int t2 = 0; t2 < SPT; t2++) {
        const float p = __expf(sc[t2] - m_new);
        Ps[row][dg * SPT + t2] = p;
        ls += p;
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
      l[u] = l[u] * corr + ls;
      m[u] = m_new;
#pragma unroll
      for (int d = 0; d < DG; d++) acc[u][d] *= corr;
    }
    __syncthreads();  // Ps complete
#pragma unroll
    for (int kk = 0; kk < TK; kk++) {
#pragma unroll
      for (int u = 0; u < QPT; u++) {
        const float pv = Ps[qr0 + u * ROWS][kk];
#pragma unroll
        for (int d = 0; d < DG; d++)
          acc[u][d] += pv * Vs[kk][d * LANES + dg];
      }
    }
  }

#pragma unroll
  for (int u = 0; u < QPT; u++) {
    const unsigned qi = qbase + qr0 + u * ROWS;
    if (qi >= T) continue;
    const float inv = 1.0f / l[u];
#pragma unroll
    for (int d = 0; d < DG; d++)
      out[((size_t)h * T + qi) * AD + d * LANES + dg] = acc[u][d] * inv;
  }
}
// tl_attn_prefill_tiled_{f32,f16}_{64,128}, and f16 at 256 (QPT 1, 8-key tiles): the
// exact prefill of an f16 cache at that width (an f32 one takes gpu::attn_prefill's
// GEMM form there, as it always has).
// K and V by four elements at once (f32: float4, f16: uint2) when their base
// allows, else one at a time (kv_ld4), so they need only their element's
// alignment: an f16 cache has no composition to fall back to, so this kernel
// must read it at whatever offset it is handed.
#define TL_ATTN_TILED(AD, QPT, KT, TN)                                                             \
  TL_ALIGN(tl_attn_prefill_tiled_##TN##_##AD, 4, sizeof(KT), sizeof(KT))                        \
  extern "C" __global__ void tl_attn_prefill_tiled_##TN##_##AD(                                    \
      const float* q, const KT* K, const KT* V, float* out, unsigned T, unsigned kv_stride,         \
      unsigned group, float scale, unsigned pos0) {                                                \
    attn_prefill_tiled_core<AD, QPT, KT>(q, K, V, out, T, kv_stride, group, scale, pos0);          \
  }
TL_ATTN_TILED(64, 4, float, f32)
TL_ATTN_TILED(128, 2, float, f32)
TL_ATTN_TILED(64, 4, __half, f16)
TL_ATTN_TILED(128, 2, __half, f16)
TL_ATTN_TILED(256, 1, __half, f16)
#undef TL_ATTN_TILED

// ---- The causal prefill on the tensor cores: flash attention over
// mma.sync.m16n8k16 with f16 operands and f32 accumulators. f16 rather than
// the GEMM's bf16 because a score is an exponent: its 3 more mantissa bits
// keep softmax weights ~8x closer, and q and k, normalized, sit far inside
// f16's range. An f32 cache is rounded as a block stages it; an f16 one is not.
//
// One block per (q head, 16*NW queries), a warp per 16 of them. A warp's Q
// fragments are read once, from global, and stay in registers; a BK-key tile of
// K and V at a time is converted to f16 into one of two shared buffers while
// the tensor cores work on the other (the next tile's f32 is already in flight
// in registers), and read back with ldmatrix: K as it is stored (the B operand
// of Q.K^T), V transposed (the B operand of P.V). Each warp computes its
// 16 x BK scores into registers, scales and causally masks them, and keeps the
// online softmax per row in registers — the m16n8 accumulator's layout is
// documented (row = lane/4 and lane/4 + 8, cols 2*(lane%4) + 0/1), so a row's
// max and sum are a reduction across a lane quad and the output's rescale a
// multiply per register. The probabilities then go straight into P.V as the A
// operand: an m16n8 accumulator pair is an m16k16 A fragment. Query tiles run
// last to first: the later ones see more keys, and blocks start in order.
// Against the version below (Q through shared memory, fragments built from
// 2-byte loads) this is 1.3-1.5x at 2048 cached tokens, bit for bit.
__device__ __forceinline__ unsigned h2_bits_(__half a, __half b) {
  __half2 h = __halves2half2(a, b);
  return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ void mma_f16_16816_(float* c, const unsigned* a,
                                               const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void tc_ldsm4t_(unsigned& r0,unsigned& r1,unsigned& r2,unsigned& r3,const void* p){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"((unsigned)__cvta_generic_to_shared(p)));}
__device__ __forceinline__ unsigned tc_pack_(float a,float b){__half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<unsigned*>(&h);}
// A K/V tile of the tensor-core prefill: BK cached rows of K and V read into registers
// ahead (the next tile's while this one multiplies), then staged to shared memory as
// f16 rows LD apart. An f32 cache is rounded to nearest as it is staged; an f16 one
// holds those same values already and is copied. The two address the rows in the forms
// the compiler schedules best for each (measured at Qwen3's 16 heads, 2048 cached
// tokens: an offset from the tile's start takes the f32 tile 243 -> 221 us, while the
// f16 copy runs 230 there and 220 indexed by row).
template <int AD, int NTH, int NV, typename KT>
struct attn_kv_tile_ {
  using RT = typename std::conditional<std::is_same<KT, float>::value, float4, uint2>::type;
  RT rk[NV], rv[NV];
  __device__ __forceinline__ void load(const KT* Kh, const KT* Vh, unsigned tid, unsigned k0,
                                       unsigned kv_end) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const unsigned e = (tid + i * NTH) * 4, r = e / AD, c = e % AD, row = k0 + r;
      const bool ok = row < kv_end;
      rk[i] = ok ? *reinterpret_cast<const RT*>(Kh + (size_t)row * AD + c) : RT{};
      rv[i] = ok ? *reinterpret_cast<const RT*>(Vh + (size_t)row * AD + c) : RT{};
    }
  }
  template <int LD>
  __device__ __forceinline__ void stage(__half (*Ks)[LD], __half (*Vs)[LD], unsigned tid) const {
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const unsigned e = (tid + i * NTH) * 4, r = e / AD, c = e % AD;
      if constexpr (std::is_same<KT, float>::value) {
        __half* k = &Ks[0][0] + r * LD + c;
        __half* v = &Vs[0][0] + r * LD + c;
        *reinterpret_cast<__half2*>(k) = __floats2half2_rn(rk[i].x, rk[i].y);
        *reinterpret_cast<__half2*>(k + 2) = __floats2half2_rn(rk[i].z, rk[i].w);
        *reinterpret_cast<__half2*>(v) = __floats2half2_rn(rv[i].x, rv[i].y);
        *reinterpret_cast<__half2*>(v + 2) = __floats2half2_rn(rv[i].z, rv[i].w);
      } else {
        *reinterpret_cast<uint2*>(&Ks[r][c]) = rk[i];
        *reinterpret_cast<uint2*>(&Vs[r][c]) = rv[i];
      }
    }
  }
};
template <int AD, int NW, int BK, typename KT>
__device__ void attn_prefill_tc_core(const float* __restrict__ q, const KT* __restrict__ K,
    const KT* __restrict__ V, float* __restrict__ out, unsigned T, unsigned kv_stride,
    unsigned group, float scale, unsigned pos0) {
  constexpr int BQ = 16 * NW, LD = AD + 8, NTH = 32 * NW;
  constexpr int NS = BK / 8, NO = AD / 8, NV = BK * AD / 4 / NTH;
  static_assert(NV >= 1 && NS % 2 == 0 && NO % 2 == 0, "tile");
  __shared__ __align__(16) __half Ks[2][BK][LD];
  __shared__ __align__(16) __half Vs[2][BK][LD];
  const unsigned h = blockIdx.x, t0 = (gridDim.y - 1 - blockIdx.y) * BQ, tid = threadIdx.x;  // heavy tiles first
  const unsigned warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const float* qh = q + (size_t)h * T * AD;
  const KT* Kh = K + (size_t)(h / group) * kv_stride;
  const KT* Vh = V + (size_t)(h / group) * kv_stride;
  const unsigned kv_end = pos0 + min(T, t0 + BQ);
  const unsigned qr = warp * 16 + g;
  const unsigned pa = pos0 + t0 + qr, pb = pa + 8;
  // Q fragments, once, straight from global.
  unsigned qf[AD / 16][4];
#pragma unroll
  for (int kc = 0; kc < AD / 16; kc++) {
    const unsigned c = kc * 16 + tq * 2;
    const unsigned r0 = t0 + qr, r1 = r0 + 8;
    float2 x0 = r0 < T ? *reinterpret_cast<const float2*>(qh + (size_t)r0 * AD + c) : make_float2(0, 0);
    float2 x1 = r1 < T ? *reinterpret_cast<const float2*>(qh + (size_t)r1 * AD + c) : make_float2(0, 0);
    float2 x2 = r0 < T ? *reinterpret_cast<const float2*>(qh + (size_t)r0 * AD + c + 8) : make_float2(0, 0);
    float2 x3 = r1 < T ? *reinterpret_cast<const float2*>(qh + (size_t)r1 * AD + c + 8) : make_float2(0, 0);
    qf[kc][0] = tc_pack_(x0.x, x0.y); qf[kc][1] = tc_pack_(x1.x, x1.y);
    qf[kc][2] = tc_pack_(x2.x, x2.y); qf[kc][3] = tc_pack_(x3.x, x3.y);
  }
  attn_kv_tile_<AD, NTH, NV, KT> kv;
  auto load = [&](unsigned k0) { kv.load(Kh, Vh, tid, k0, kv_end); };
  auto stage = [&](int b) { kv.template stage<LD>(Ks[b], Vs[b], tid); };
  const unsigned lb_r = (lane & 7) + ((lane >> 4) << 3), lb_c = ((lane >> 3) & 1) * 8;   // K (non-trans)
  const unsigned lv_r = (lane & 7) + ((lane >> 3 & 1) << 3), lv_c = (lane >> 4) * 8;      // V (trans)
  float o[NO][4];
#pragma unroll
  for (int d = 0; d < NO; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  load(0);
  stage(0);
  __syncthreads();
  for (unsigned k0 = 0, it = 0; k0 < kv_end; k0 += BK, it++) {
    const int cur = it & 1;
    const bool more = k0 + BK < kv_end;
    if (more) load(k0 + BK);
    float s[NS][4];
#pragma unroll
    for (int n = 0; n < NS; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
    for (int kc = 0; kc < AD / 16; kc++) {
#pragma unroll
      for (int j = 0; j < NS / 2; j++) {
        unsigned b0, b1, b2, b3;
        tc_ldsm4_(b0, b1, b2, b3, &Ks[cur][j * 16 + lb_r][kc * 16 + lb_c]);
        const unsigned bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
        mma_f16_16816_(s[2 * j], qf[kc], bb0);
        mma_f16_16816_(s[2 * j + 1], qf[kc], bb1);
      }
    }
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int n = 0; n < NS; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const unsigned j = k0 + n * 8 + tq * 2 + (e & 1), pos = e < 2 ? pa : pb;
        const float v = j <= pos && j < kv_end ? s[n][e] * scale : -INFINITY;
        s[n][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    }
    float corr[2], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) corr[r] = mx[r] == -INFINITY ? 1.0f : __expf(m[r] - mx[r]);
#pragma unroll
    for (int n = 0; n < NS; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = e >> 1;
        const float p = mx[r] == -INFINITY ? 0.0f : __expf(s[n][e] - mx[r]);
        s[n][e] = p;
        rs[r] += p;
      }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
      rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
      l[r] = l[r] * corr[r] + rs[r];
      m[r] = mx[r];
    }
#pragma unroll
    for (int d = 0; d < NO; d++) { o[d][0] *= corr[0]; o[d][1] *= corr[0]; o[d][2] *= corr[1]; o[d][3] *= corr[1]; }
#pragma unroll
    for (int kc = 0; kc < BK / 16; kc++) {
      const unsigned a[4] = {tc_pack_(s[2 * kc][0], s[2 * kc][1]), tc_pack_(s[2 * kc][2], s[2 * kc][3]),
                             tc_pack_(s[2 * kc + 1][0], s[2 * kc + 1][1]), tc_pack_(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
      for (int dj = 0; dj < NO / 2; dj++) {
        unsigned b0, b1, b2, b3;
        tc_ldsm4t_(b0, b1, b2, b3, &Vs[cur][kc * 16 + lv_r][dj * 16 + lv_c]);
        const unsigned bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
        mma_f16_16816_(o[2 * dj], a, bb0);
        mma_f16_16816_(o[2 * dj + 1], a, bb1);
      }
    }
    if (more) stage(cur ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const unsigned t = t0 + qr + 8 * r;
    if (t >= T) continue;
    const float inv = 1.0f / l[r];
    float* dst = out + ((size_t)h * T + t) * AD;
#pragma unroll
    for (int d = 0; d < NO; d++) {
      *reinterpret_cast<float2*>(dst + d * 8 + tq * 2) = make_float2(o[d][2 * r] * inv, o[d][2 * r + 1] * inv);
    }
  }
}

// The same kernel with Q staged through shared memory (once, as f16) and read
// back with ldmatrix, for a head width whose Q fragments (AD/4 registers) do
// not fit beside the output (AD/2) and the prefetch: AD = 256. Dynamic shared
// memory (context::attn_tc_smem): Q, then K and V in two buffers each. The Qwen3.5
// attention layers' prefill is 3.4x faster on this than on the first form of
// the kernel (Q fragments from 2-byte loads, K and V by scalar loads), bit for bit.
template <int AD, int NW, int BK, typename KT>
__device__ void attn_prefill_tc_core_smem(const float* __restrict__ q, const KT* __restrict__ K,
    const KT* __restrict__ V, float* __restrict__ out, unsigned T, unsigned kv_stride,
    unsigned group, float scale, unsigned pos0) {
  constexpr int BQ = 16 * NW, LD = AD + 8, NTH = 32 * NW;
  constexpr int NS = BK / 8, NO = AD / 8, NV = BK * AD / 4 / NTH;
  static_assert(NV >= 1 && NS % 2 == 0 && NO % 2 == 0, "tile");
  extern __shared__ __align__(16) unsigned char attn_smem[];
  __half (*Qs)[LD] = reinterpret_cast<__half(*)[LD]>(attn_smem);
  __half (*Ks)[BK][LD] = reinterpret_cast<__half(*)[BK][LD]>(attn_smem + BQ * LD * 2);
  __half (*Vs)[BK][LD] = reinterpret_cast<__half(*)[BK][LD]>(attn_smem + BQ * LD * 2 + 2 * BK * LD * 2);
  const unsigned h = blockIdx.x, t0 = (gridDim.y - 1 - blockIdx.y) * BQ, tid = threadIdx.x;  // heavy tiles first
  const unsigned warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const float* qh = q + (size_t)h * T * AD;
  const KT* Kh = K + (size_t)(h / group) * kv_stride;
  const KT* Vh = V + (size_t)(h / group) * kv_stride;
  const unsigned kv_end = pos0 + min(T, t0 + BQ);
  const unsigned qr = warp * 16 + g;
  const unsigned pa = pos0 + t0 + qr, pb = pa + 8;
  for (unsigned e = tid * 4; e < BQ * AD; e += NTH * 4) {
    const unsigned r = e / AD, c = e % AD, row = t0 + r;
    const float4 v = row < T ? *reinterpret_cast<const float4*>(qh + (size_t)row * AD + c) : make_float4(0, 0, 0, 0);
    *reinterpret_cast<__half2*>(&Qs[r][c]) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(&Qs[r][c + 2]) = __floats2half2_rn(v.z, v.w);
  }
  attn_kv_tile_<AD, NTH, NV, KT> kv;
  auto load = [&](unsigned k0) { kv.load(Kh, Vh, tid, k0, kv_end); };
  auto stage = [&](int b) { kv.template stage<LD>(Ks[b], Vs[b], tid); };
  const unsigned lb_r = (lane & 7) + ((lane >> 4) << 3), lb_c = ((lane >> 3) & 1) * 8;   // K (non-trans)
  const unsigned lv_r = (lane & 7) + ((lane >> 3 & 1) << 3), lv_c = (lane >> 4) * 8;      // V (trans)
  float o[NO][4];
#pragma unroll
  for (int d = 0; d < NO; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  load(0);
  stage(0);
  __syncthreads();
  for (unsigned k0 = 0, it = 0; k0 < kv_end; k0 += BK, it++) {
    const int cur = it & 1;
    const bool more = k0 + BK < kv_end;
    if (more) load(k0 + BK);
    float s[NS][4];
#pragma unroll
    for (int n = 0; n < NS; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
    for (int kc = 0; kc < AD / 16; kc++) {
      unsigned qa[4];
      tc_ldsm4_(qa[0], qa[1], qa[2], qa[3], &Qs[warp * 16 + (lane & 15)][kc * 16 + (lane >> 4) * 8]);
#pragma unroll
      for (int j = 0; j < NS / 2; j++) {
        unsigned b0, b1, b2, b3;
        tc_ldsm4_(b0, b1, b2, b3, &Ks[cur][j * 16 + lb_r][kc * 16 + lb_c]);
        const unsigned bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
        mma_f16_16816_(s[2 * j], qa, bb0);
        mma_f16_16816_(s[2 * j + 1], qa, bb1);
      }
    }
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int n = 0; n < NS; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const unsigned j = k0 + n * 8 + tq * 2 + (e & 1), pos = e < 2 ? pa : pb;
        const float v = j <= pos && j < kv_end ? s[n][e] * scale : -INFINITY;
        s[n][e] = v;
        mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    }
    float corr[2], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; r++) corr[r] = mx[r] == -INFINITY ? 1.0f : __expf(m[r] - mx[r]);
#pragma unroll
    for (int n = 0; n < NS; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = e >> 1;
        const float p = mx[r] == -INFINITY ? 0.0f : __expf(s[n][e] - mx[r]);
        s[n][e] = p;
        rs[r] += p;
      }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 1);
      rs[r] += __shfl_xor_sync(0xffffffffu, rs[r], 2);
      l[r] = l[r] * corr[r] + rs[r];
      m[r] = mx[r];
    }
#pragma unroll
    for (int d = 0; d < NO; d++) { o[d][0] *= corr[0]; o[d][1] *= corr[0]; o[d][2] *= corr[1]; o[d][3] *= corr[1]; }
#pragma unroll
    for (int kc = 0; kc < BK / 16; kc++) {
      const unsigned a[4] = {tc_pack_(s[2 * kc][0], s[2 * kc][1]), tc_pack_(s[2 * kc][2], s[2 * kc][3]),
                             tc_pack_(s[2 * kc + 1][0], s[2 * kc + 1][1]), tc_pack_(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
      for (int dj = 0; dj < NO / 2; dj++) {
        unsigned b0, b1, b2, b3;
        tc_ldsm4t_(b0, b1, b2, b3, &Vs[cur][kc * 16 + lv_r][dj * 16 + lv_c]);
        const unsigned bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
        mma_f16_16816_(o[2 * dj], a, bb0);
        mma_f16_16816_(o[2 * dj + 1], a, bb1);
      }
    }
    if (more) stage(cur ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const unsigned t = t0 + qr + 8 * r;
    if (t >= T) continue;
    const float inv = 1.0f / l[r];
    float* dst = out + ((size_t)h * T + t) * AD;
#pragma unroll
    for (int d = 0; d < NO; d++) {
      *reinterpret_cast<float2*>(dst + d * 8 + tq * 2) = make_float2(o[d][2 * r] * inv, o[d][2 * r + 1] * inv);
    }
  }
}
// tl_attn_prefill_tc_{f32,f16}_{64,128,256}
// q by float2 (D = 256: float4), K and V by four elements, out by float2.
#define TL_ATTN_TC(AD, CORE, BK, KT, TN)                                                           \
  TL_ALIGN(tl_attn_prefill_tc_##TN##_##AD, (AD) == 256 ? 16 : 8, 4 * sizeof(KT),            \
           4 * sizeof(KT), 8)                                                               \
  extern "C" __global__ void tl_attn_prefill_tc_##TN##_##AD(                                       \
      const float* q, const KT* K, const KT* V, float* out, unsigned T, unsigned kv_stride,        \
      unsigned group, float scale, unsigned pos0) {                                                \
    CORE<AD, 4, BK, KT>(q, K, V, out, T, kv_stride, group, scale, pos0);                           \
  }
TL_ATTN_TC(64, attn_prefill_tc_core, 32, float, f32)
TL_ATTN_TC(128, attn_prefill_tc_core, 32, float, f32)
TL_ATTN_TC(256, attn_prefill_tc_core_smem, 16, float, f32)
TL_ATTN_TC(64, attn_prefill_tc_core, 32, __half, f16)
TL_ATTN_TC(128, attn_prefill_tc_core, 32, __half, f16)
TL_ATTN_TC(256, attn_prefill_tc_core_smem, 16, __half, f16)
#undef TL_ATTN_TC

// The query half of attn_prefill's pullback, and the row statistics the key/
// value half needs. Same streaming shape as the forward — one block per (head,
// query tile), K/V walked in tiles through shared memory — over a training
// shape: q, K, V, dO, O, dq all [H,T,D] contiguous, no cache, no GQA.
//
// Writing Δ_i = dO_i·O_i, row i's gradient is
//   dq_i = scale · Σ_j P_ij (dP_ij − Δ_i) k_j,   dP_ij = dO_i·v_j,
// with P_ij = exp(s_ij − m_i)/l_i the same softmax the forward built. Δ_i needs
// no P of its own: O_i = Σ_j P_ij v_j makes Σ_j P_ij dP_ij exactly dO_i·O_i. A
// per-row constant is what lets the whole sum run under one online softmax —
// accumulate against the running max, rescale the accumulator when the max
// grows, divide by l once at the end — so the [T,T] scores are never anywhere
// but in registers, and one pass over the keys is enough.
//
// Both of this pass's by-products are written to `stats` [2,H,T] — plane 0 the
// row logsumexp L_i = m_i + log l_i, plane 1 the Δ_i above — because the dK/dV
// kernel needs exactly those two and can compute neither cheaply: it sums over
// i, where no rescaling can stand in for a normalizer, and it never holds an
// output row next to its gradient. Handing them over is free here.
//
// A tile carries half the queries the forward's does (BQ = ROWS·QPT with QPT
// halved): dO's rows take a second Qs-sized block of shared memory, and the
// forward's tile would put the total past the 48 KB static limit.
template <int AD, int QPT>
__device__ void attn_bwd_dq_core(const float* __restrict__ q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 const float* __restrict__ dO,
                                 const float* __restrict__ O,
                                 float* __restrict__ dq,
                                 float* __restrict__ stats, unsigned T,
                                 float scale) {
  constexpr int NT = 128, LANES = 8, ROWS = NT / LANES;
  constexpr int BQ = ROWS * QPT;            // queries per block
  constexpr int TK = (AD == 64) ? 32 : 16;  // keys per tile
  constexpr int DG = AD / LANES;            // accumulator dims per lane
  constexpr int SPT = TK / LANES;           // scores per lane per tile
  // Same +1 row padding as the forward, and for the same reason: every one of
  // these is read down a column by lanes that differ only in the row index.
  // Vs is padded here too — the dP dot reads it the way the score loop reads
  // Ks, not the way the forward's context loop does.
  __shared__ float Qs[BQ][AD + 1];
  __shared__ float Gs[BQ][AD + 1];  // dO's rows
  __shared__ float Ks[TK][AD + 1];
  __shared__ float Vs[TK][AD + 1];
  __shared__ float Cs[BQ][TK + 1];  // exp(s − m)·(dP − Δ), the dq coefficient
  __shared__ float Ds[BQ];          // Δ per row

  const unsigned h = blockIdx.x;
  const unsigned qbase = (gridDim.y - 1u - blockIdx.y) * BQ;  // reversed order
  const unsigned tid = threadIdx.x;
  const unsigned qr0 = tid >> 3, dg = tid & 7u;
  const float* qh = q + (size_t)h * T * AD;
  const float* Kh = K + (size_t)h * T * AD;
  const float* Vh = V + (size_t)h * T * AD;
  const float* Gh = dO + (size_t)h * T * AD;
  const float* Oh = O + (size_t)h * T * AD;

  for (unsigned i = tid; i < BQ * AD; i += NT) {
    const unsigned r = i / AD, c = i % AD;
    const bool live = qbase + r < T;
    Qs[r][c] = live ? qh[(size_t)(qbase + r) * AD + c] : 0.f;
    Gs[r][c] = live ? Gh[(size_t)(qbase + r) * AD + c] : 0.f;
  }
  __syncthreads();
  for (unsigned r = tid; r < BQ; r += NT) {
    float d = 0.f;
    if (qbase + r < T) {
      const float* op = Oh + (size_t)(qbase + r) * AD;
      for (int c = 0; c < AD; c++) d += Gs[r][c] * op[c];
    }
    Ds[r] = d;
  }
  const unsigned last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;

  float acc[QPT][DG] = {};
  float m[QPT], l[QPT];
#pragma unroll
  for (int u = 0; u < QPT; u++) { m[u] = -1e30f; l[u] = 0.0f; }

  for (unsigned kt = 0; kt <= last; kt += TK) {
    __syncthreads();  // previous tile's Ks/Vs are done being read
    for (unsigned i = tid; i < TK * AD / 4; i += NT) {
      const unsigned r = i / (AD / 4), c4 = (i % (AD / 4)) * 4;
      const unsigned kk = kt + r;
      float kb[4] = {0, 0, 0, 0}, vb[4] = {0, 0, 0, 0};
      if (kk <= last) {
        kv_ld4(Kh + (size_t)kk * AD + c4, kb);
        kv_ld4(Vh + (size_t)kk * AD + c4, vb);
      }
#pragma unroll
      for (int e = 0; e < 4; e++) {
        Ks[r][c4 + e] = kb[e];
        Vs[r][c4 + e] = vb[e];
      }
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < QPT; u++) {
      const unsigned row = qr0 + u * ROWS;
      const unsigned pabs = qbase + row;
      // A dot per key, each re-reading this query's row: the dK/dV half is
      // written the other way round (the operand it holds fixed read once,
      // used against every query) because there it spills otherwise, but here
      // the same rewrite only costs — 62 registers instead of 196, and 9%
      // slower for having less to overlap.
      float sc[SPT], dp[SPT], mt = -1e30f;
#pragma unroll
      for (int t2 = 0; t2 < SPT; t2++) {
        const unsigned kk = kt + dg * SPT + t2;
        float dot = 0.0f, dpd = 0.0f;
#pragma unroll
        for (int d = 0; d < AD; d++) {
          dot += Qs[row][d] * Ks[dg * SPT + t2][d];
          dpd += Gs[row][d] * Vs[dg * SPT + t2][d];
        }
        sc[t2] = (kk <= pabs) ? dot * scale : -1e30f;  // causal mask
        dp[t2] = dpd;
        mt = fmaxf(mt, sc[t2]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
      const float m_new = fmaxf(m[u], mt);
      const float corr = __expf(m[u] - m_new);
      const float delta = Ds[row];
      float ls = 0.0f;
#pragma unroll
      for (int t2 = 0; t2 < SPT; t2++) {
        const float p = __expf(sc[t2] - m_new);
        Cs[row][dg * SPT + t2] = p * (dp[t2] - delta);
        ls += p;
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1)
        ls += __shfl_xor_sync(0xffffffffu, ls, off);
      l[u] = l[u] * corr + ls;
      m[u] = m_new;
#pragma unroll
      for (int d = 0; d < DG; d++) acc[u][d] *= corr;
    }
    __syncthreads();  // Cs complete
#pragma unroll
    for (int kk = 0; kk < TK; kk++) {
#pragma unroll
      for (int u = 0; u < QPT; u++) {
        const float c = Cs[qr0 + u * ROWS][kk];
#pragma unroll
        for (int d = 0; d < DG; d++)
          acc[u][d] += c * Ks[kk][d * LANES + dg];
      }
    }
  }

#pragma unroll
  for (int u = 0; u < QPT; u++) {
    const unsigned qi = qbase + qr0 + u * ROWS;
    if (qi >= T) continue;
    const float f = scale / l[u];
#pragma unroll
    for (int d = 0; d < DG; d++)
      dq[((size_t)h * T + qi) * AD + d * LANES + dg] = acc[u][d] * f;
    if (dg == 0) {
      const size_t row = (size_t)h * T + qi;
      stats[row] = m[u] + __logf(l[u]);
      stats[(size_t)gridDim.x * T + row] = Ds[qr0 + u * ROWS];
    }
  }
}
TL_ALIGN(tl_attn_bwd_dq_f32, 4, 16, 16)  // K and V by float4
TL_ALIGN(tl_attn_bwd_dq_f32_64, 4, 16, 16)
extern "C" __global__ void tl_attn_bwd_dq_f32(const float* q, const float* K,
    const float* V, const float* dO, const float* O, float* dq, float* stats,
    unsigned T, float scale) {
  attn_bwd_dq_core<128, 1>(q, K, V, dO, O, dq, stats, T, scale);
}
extern "C" __global__ void tl_attn_bwd_dq_f32_64(const float* q, const float* K,
    const float* V, const float* dO, const float* O, float* dq, float* stats,
    unsigned T, float scale) {
  attn_bwd_dq_core<64, 2>(q, K, V, dO, O, dq, stats, T, scale);
}

// The key/value half, the same streaming shape with the roles swapped: one
// block per (head, key tile), walking the query tiles that can see those keys.
//   dv_j = Σ_{i≥j} P_ij dO_i,   dk_j = scale · Σ_{i≥j} P_ij (dP_ij − Δ_i) q_i.
// Summing over queries is what makes this half different: their normalizers
// differ, so no running rescale can stand in for one, and P_ij = exp(s_ij − L_i)
// is read off the stats the dq kernel already wrote. Block 0 holds the earliest
// keys, which the most queries see — the heaviest blocks are the first
// launched, so the natural order is the balanced one here (the forward, whose
// last queries are its heaviest, reverses for the same reason).
template <int AD, int KPT>
__device__ void attn_bwd_dkv_core(const float* __restrict__ q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  const float* __restrict__ dO,
                                  const float* __restrict__ stats,
                                  float* __restrict__ dK,
                                  float* __restrict__ dV, unsigned T,
                                  float scale) {
  constexpr int NT = 128, LANES = 8, ROWS = NT / LANES;
  constexpr int BK = ROWS * KPT;            // keys per block
  constexpr int TQ = (AD == 64) ? 32 : 16;  // queries per tile
  constexpr int DG = AD / LANES;            // accumulator dims per lane
  constexpr int SPT = TQ / LANES;           // scores per lane per tile
  __shared__ float Ks[BK][AD + 1];
  __shared__ float Vs[BK][AD + 1];
  __shared__ float Qs[TQ][AD + 1];
  __shared__ float Gs[TQ][AD + 1];
  __shared__ float Ps[BK][TQ + 1];  // the softmax probability
  __shared__ float Cs[BK][TQ + 1];  // P·(dP − Δ), the dk coefficient
  __shared__ float Ls[TQ], Ds[TQ];

  const unsigned h = blockIdx.x;
  const unsigned kbase = blockIdx.y * BK;
  const unsigned tid = threadIdx.x;
  const unsigned kr0 = tid >> 3, dg = tid & 7u;
  const float* qh = q + (size_t)h * T * AD;
  const float* Kh = K + (size_t)h * T * AD;
  const float* Vh = V + (size_t)h * T * AD;
  const float* Gh = dO + (size_t)h * T * AD;
  const float* Lh = stats + (size_t)h * T;
  const float* Dh = stats + (size_t)gridDim.x * T + (size_t)h * T;

  for (unsigned i = tid; i < BK * AD; i += NT) {
    const unsigned r = i / AD, c = i % AD;
    const bool live = kbase + r < T;
    Ks[r][c] = live ? Kh[(size_t)(kbase + r) * AD + c] : 0.f;
    Vs[r][c] = live ? Vh[(size_t)(kbase + r) * AD + c] : 0.f;
  }

  float dk[KPT][DG] = {}, dv[KPT][DG] = {};

  // Queries before this tile's first key see none of it.
  for (unsigned qt = (kbase / TQ) * TQ; qt < T; qt += TQ) {
    __syncthreads();  // previous tile's Qs/Gs are done being read
    for (unsigned i = tid; i < TQ * AD; i += NT) {
      const unsigned r = i / AD, c = i % AD;
      const bool live = qt + r < T;
      Qs[r][c] = live ? qh[(size_t)(qt + r) * AD + c] : 0.f;
      Gs[r][c] = live ? Gh[(size_t)(qt + r) * AD + c] : 0.f;
    }
    for (unsigned r = tid; r < TQ; r += NT) {
      const bool live = qt + r < T;
      Ls[r] = live ? Lh[qt + r] : 0.f;
      Ds[r] = live ? Dh[qt + r] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < KPT; u++) {
      const unsigned row = kr0 + u * ROWS;
      const unsigned kabs = kbase + row;
      // The key's own dimension is read once and used against every query this
      // lane carries. Written the other way round — a dot per query, each
      // re-reading Ks and Vs — the 64-deep loop unrolls into so many loads in
      // flight that ptxas spills, and the kernel runs 4x slower.
      float sc[SPT] = {}, dpd[SPT] = {};
#pragma unroll 8
      for (int d = 0; d < AD; d++) {
        const float kd = Ks[row][d], vd = Vs[row][d];
#pragma unroll
        for (int t2 = 0; t2 < SPT; t2++) {
          sc[t2] += Qs[dg * SPT + t2][d] * kd;
          dpd[t2] += Gs[dg * SPT + t2][d] * vd;
        }
      }
#pragma unroll
      for (int t2 = 0; t2 < SPT; t2++) {
        const unsigned ql = dg * SPT + t2, qi = qt + ql;
        // Off the causal half, or past the last query: contributes nothing.
        const float p =
            (qi < T && kabs <= qi) ? __expf(sc[t2] * scale - Ls[ql]) : 0.f;
        Ps[row][ql] = p;
        Cs[row][ql] = p * (dpd[t2] - Ds[ql]);
      }
    }
    __syncthreads();  // Ps/Cs complete
#pragma unroll
    for (int qq = 0; qq < TQ; qq++) {
#pragma unroll
      for (int u = 0; u < KPT; u++) {
        const float p = Ps[kr0 + u * ROWS][qq], c = Cs[kr0 + u * ROWS][qq];
#pragma unroll
        for (int d = 0; d < DG; d++) {
          dv[u][d] += p * Gs[qq][d * LANES + dg];
          dk[u][d] += c * Qs[qq][d * LANES + dg];
        }
      }
    }
  }

#pragma unroll
  for (int u = 0; u < KPT; u++) {
    const unsigned kj = kbase + kr0 + u * ROWS;
    if (kj >= T) continue;
#pragma unroll
    for (int d = 0; d < DG; d++) {
      dK[((size_t)h * T + kj) * AD + d * LANES + dg] = dk[u][d] * scale;
      dV[((size_t)h * T + kj) * AD + d * LANES + dg] = dv[u][d];
    }
  }
}
extern "C" __global__ void tl_attn_bwd_dkv_f32(const float* q, const float* K,
    const float* V, const float* dO, const float* stats, float* dK, float* dV,
    unsigned T, float scale) {
  attn_bwd_dkv_core<128, 1>(q, K, V, dO, stats, dK, dV, T, scale);
}
extern "C" __global__ void tl_attn_bwd_dkv_f32_64(const float* q, const float* K,
    const float* V, const float* dO, const float* stats, float* dK, float* dV,
    unsigned T, float scale) {
  attn_bwd_dkv_core<64, 2>(q, K, V, dO, stats, dK, dV, T, scale);
}

// RoPE's turn of the pair (x0, x1) at dims (j, j + rot/2) of a row at `position`
// (rope_core below; every rotation here goes through it, so they agree bit for bit).
__device__ __forceinline__ void rope_pair_(float position, unsigned j, unsigned rot, float base,
                                           float& x0, float& x1) {
  const float ang = position * __powf(base, -2.0f * (float)j / (float)rot);
  const float c = __cosf(ang), s = __sinf(ang);
  // r1's fused multiply-add named (x0 s fused, x1 c rounded first, the form rope_core
  // had): left to contraction, the compiler fused whichever product suited the code
  // around each copy, and the copies stopped agreeing bit for bit. r0 stays as written,
  // the PTX it always produced (the JIT's own contraction then sees the same code).
  const float r0 = x0 * c - x1 * s;
  const float r1 = __fmaf_rn(x0, s, x1 * c);
  x0 = r0;
  x1 = r1;
}
// Qwen3's q / k head rows RMS-normalized and then rotated, in place, a warp a row (eight
// a block); a row is r = head * T + token, as for rope. D = 64, 128 or 256 and rot a
// multiple of 64, so that a lane's values x[lane + 32 k] hold both halves of its
// rotation pairs; own::rmsnorm_rope refuses other shapes (the caller then runs rmsnorm
// and rope). The sum is rmsnorm_core's (rmsnorm_sum_warp_, the squares rounded) and the
// turn rope_core's, so the result is tl_rmsnorm then tl_rope's, bit for bit. A block a
// row, 256 threads for 128 values, took 36 us for Qwen3-0.6B's 8192 q rows; this 8.6.
template <unsigned D>
__device__ __forceinline__ void rmsnorm_rope_core_(float* __restrict__ x, const float* __restrict__ w,
                                                   unsigned rows, unsigned T, unsigned pos, float base,
                                                   unsigned rot, float eps) {
  constexpr int E = D / 32;
  const unsigned lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  float* xr = x + (size_t)row * D;
  float v[E], acc[8];
#pragma unroll
  for (int k = 0; k < E; k++) v[k] = xr[lane + 32 * k];
#pragma unroll
  for (int k = 0; k < 8; k++) acc[k] = k < E ? __fmul_rn(v[k], v[k]) : 0.0f;
  const float inv = 1.0f / sqrtf(rmsnorm_sum_warp_(acc) / (float)D + eps);
  const unsigned half = rot >> 1, P = half / 32;  // rotation pairs a lane
#pragma unroll
  for (int k = 0; k < E; k++) {
    const unsigned d = lane + 32 * k;
    if (d >= rot) xr[d] = v[k] * inv * w[d];
  }
  const float position = (float)(pos + row % T);
#pragma unroll
  for (int k = 0; k < E / 2; k++) {
    if ((unsigned)k >= P) break;
    const unsigned t = lane + 32 * k;
    float x0 = v[k] * inv * w[t], x1 = v[k + P] * inv * w[t + half];
    rope_pair_(position, t, rot, base, x0, x1);
    xr[t] = x0;
    xr[t + half] = x1;
  }
}
#define TL_RMSNORM_ROPE(D)                                                                       \
  extern "C" __global__ void tl_rmsnorm_rope_##D(float* __restrict__ x, const float* __restrict__ w, \
                                                 unsigned rows, unsigned T, unsigned pos, float base, \
                                                 unsigned rot, float eps) {                          \
    rmsnorm_rope_core_<D>(x, w, rows, T, pos, base, rot, eps);                                     \
  }
TL_RMSNORM_ROPE(64)
TL_RMSNORM_ROPE(128)
TL_RMSNORM_ROPE(256)
#undef TL_RMSNORM_ROPE

extern "C" {
// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama) convention.
// x is [rows, D] contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D]
// with T=1). Row r's head-dim vector is at position pos + (r % T). Pairs
// (j, j+D/2) rotate by angle = position · base^(-2j/D). grid = rows, block = D/2.
// An optional fused bias is added before the rotation: folding q/k's bias-add
// into rope removes 2 elementwise launches per decode layer.
// Rotates the first `rot` dims of each row (rot == D: all of it, the math
// unchanged from before rot existed) and copies the rest through, bias added.
// blockDim = D/2: thread j turns the pair (j, j + rot/2) when j < rot/2, and
// strides the pass-through dims from rot + j.
static __device__ __forceinline__ void rope_core(
    const float* __restrict__ x, const float* __restrict__ bias,
    float* __restrict__ out, unsigned T, unsigned D, unsigned pos, float base,
    unsigned rot) {
  const unsigned r = blockIdx.x, j = threadIdx.x;  // j in 0..D/2-1
  const size_t bi = (size_t)r * D;
  for (unsigned d = rot + j; d < D; d += blockDim.x)
    out[bi + d] = bias ? x[bi + d] + bias[bi + d] : x[bi + d];
  const unsigned half = rot >> 1;
  if (j >= half) return;
  const unsigned t = T ? (r % T) : 0u;
  float x0 = x[bi + j];
  float x1 = x[bi + j + half];
  if (bias) {
    x0 += bias[bi + j];
    x1 += bias[bi + j + half];
  }
  rope_pair_((float)(pos + t), j, rot, base, x0, x1);
  out[bi + j] = x0;
  out[bi + j + half] = x1;
}
__global__ void tl_rope(const float* __restrict__ x,
                        const float* __restrict__ bias, float* __restrict__ out,
                        unsigned T, unsigned D, unsigned pos, float base,
                        unsigned rot) {
  rope_core(x, bias, out, T, D, pos, base, rot);
}
}  // extern "C"
