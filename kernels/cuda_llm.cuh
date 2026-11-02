#pragma once

// The llm layer's kernels (layers: gpu_abi.h): the KV cache's writes and the
// graph-capture forms that read the decode position from the device.
// Included by tensorlib_cuda.cu, after cuda_nn.cuh.

// Device-pos split-KV decode attention (CUDA-graph capture): ctx = *d_pos + 1
// (the decode step attends the cached prefix [0, pos] after this step's k,v
// were appended at row pos). Grid dims come from the STATIC cache capacity
// (the launcher sets gridDim.y = S_max from max_ctx), work bounds from the
// DYNAMIC position: each block recomputes the host launcher's split heuristic
// from *d_pos, so one instantiated graph replays correctly at every pos AND
// stays split-KV-flat at long ctx (the earlier S=1 pin was linear in ctx).
// Splits past ceil(ctx/chunk) see k0 >= ctx, fall through the core's loop
// empty, and write the neutral partial (m=-1e30, l=0, acc=0) that contributes
// exact zeros in tl_attn_combine — so the output is bit-identical to the host
// path (single-block or split) at the same pos. Body is attn_decode_split_core,
// byte-identical to the by-value kernels.
//
// attn_dpos_chunk is the device twin of cuda.h's attn_split_count +
// attn_split_chunk and MUST stay in lockstep with them — that equality is what
// the host/dpos bit-identity rests on (guarded by the attn64 ctest's
// host-vs-dpos bit-equality sweep). ctx <= max_ctx is a cache invariant, so the
// live split count never exceeds the launched gridDim.y (attn_split_count is
// monotone in ctx).
__device__ __forceinline__ unsigned attn_dpos_chunk(unsigned ctx, unsigned H) {
  unsigned S = 1;
  if (H < 328u && ctx >= 256u) {  // target ~4 blocks/SM (328 = 4 * 82 SMs)
    unsigned want = (328u + H - 1u) / H;
    unsigned max_s = ctx / 64u;  // >=64 keys per split
    if (want > max_s) want = max_s;
    if (want > 1u) S = want;
  }
  unsigned chunk = (ctx + S - 1u) / S;
  chunk = (chunk + 3u) & ~3u;  // multiple of the warp count
  return chunk ? chunk : 4u;
}
// tl_attn_decode_split_dpos_{f32,f16}_{64,128,256}
#define TL_ATTN_SPLIT_DPOS(AD, KT, TN)                                                               \
  extern "C" __global__ void tl_attn_decode_split_dpos_##TN##_##AD(                                  \
      const float* q, const KT* K, const KT* V, float* pm, float* pl, float* pacc,                   \
      const unsigned* __restrict__ d_pos, unsigned kv_stride, unsigned group, float scale) {         \
    const unsigned ctx = *d_pos + 1u;                                                                \
    attn_decode_split_core<AD, KT>(q, K, V, pm, pl, pacc, ctx, kv_stride, group,                     \
                                   attn_dpos_chunk(ctx, gridDim.x), scale);                          \
  }
TL_ATTN_SPLIT_DPOS(64, float, f32)
TL_ATTN_SPLIT_DPOS(128, float, f32)
TL_ATTN_SPLIT_DPOS(256, float, f32)
TL_ATTN_SPLIT_DPOS(64, __half, f16)
TL_ATTN_SPLIT_DPOS(128, __half, f16)
TL_ATTN_SPLIT_DPOS(256, __half, f16)
#undef TL_ATTN_SPLIT_DPOS

// Append one decode step's k,v (each [H_kv, D] contiguous) into the persistent
// cache at row `pos`. The cache is [H_kv, max_ctx, D], so head h's slot for the
// new token starts at h*kv_stride + pos*D (kv_stride = max_ctx*D). Heads are
// non-contiguous in the cache, so this is a scatter, not a plain copy. Not
// templated: D == blockDim.x. grid.x = H_kv, blockDim = D.
template <typename KT>
__device__ void kv_append_core(KT* __restrict__ Kc, KT* __restrict__ Vc,
                               const float* __restrict__ k_new,
                               const float* __restrict__ v_new, unsigned pos,
                               unsigned kv_stride) {
  const unsigned h = blockIdx.x, d = threadIdx.x, D = blockDim.x;
  const size_t dst = (size_t)h * kv_stride + (size_t)pos * D + d;
  const size_t src = (size_t)h * D + d;
  kv_st(&Kc[dst], k_new[src]);  // narrow to KT (identity for f32)
  kv_st(&Vc[dst], v_new[src]);
}
extern "C" __global__ void tl_kv_append(float* Kc, float* Vc,
                                        const float* k_new, const float* v_new,
                                        unsigned pos, unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, pos, kv_stride);
}
extern "C" __global__ void tl_kv_append_f16(__half* Kc, __half* Vc, const float* k_new,
                                            const float* v_new, unsigned pos, unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, pos, kv_stride);
}
// Device-pos variants (CUDA-graph capture): write row = *d_pos, read from a device
// scalar so replay targets the advancing cache row. Body is kv_append_core,
// byte-identical to tl_kv_append — only the pos source differs.
extern "C" __global__ void tl_kv_append_dpos(float* Kc, float* Vc,
                                             const float* k_new,
                                             const float* v_new,
                                             const unsigned* __restrict__ d_pos,
                                             unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, *d_pos, kv_stride);
}
extern "C" __global__ void tl_kv_append_dpos_f16(__half* Kc, __half* Vc, const float* k_new,
                                                 const float* v_new, const unsigned* __restrict__ d_pos,
                                                 unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, *d_pos, kv_stride);
}

// Bulk-fill the cache from a prefill's k,v (each [H_kv, T, D] contiguous) into
// the cache [H_kv, max_ctx, D] rows [pos0, pos0+T) — pos0 lets a long prompt be
// filled in chunks, and a later turn extend a live cache. The two layouts differ
// only in the row stride (T vs max_ctx), so this is a strided copy.
// D == blockDim.x. grid = (H_kv, T).
template <typename KT>
__device__ void kv_fill_core(KT* __restrict__ Kc, KT* __restrict__ Vc,
                             const float* __restrict__ K,
                             const float* __restrict__ V, unsigned T,
                             unsigned kv_stride, unsigned pos0) {
  const unsigned h = blockIdx.x, p = blockIdx.y, d = threadIdx.x, D = blockDim.x;
  const size_t dst = (size_t)h * kv_stride + (size_t)(pos0 + p) * D + d;
  const size_t src = ((size_t)h * T + p) * D + d;
  kv_st(&Kc[dst], K[src]);  // narrow to KT (identity for f32)
  kv_st(&Vc[dst], V[src]);
}
extern "C" __global__ void tl_kv_fill(float* Kc, float* Vc, const float* K,
                                      const float* V, unsigned T,
                                      unsigned kv_stride, unsigned pos0) {
  kv_fill_core(Kc, Vc, K, V, T, kv_stride, pos0);
}
extern "C" __global__ void tl_kv_fill_f16(__half* Kc, __half* Vc, const float* K, const float* V,
                                          unsigned T, unsigned kv_stride, unsigned pos0) {
  kv_fill_core(Kc, Vc, K, V, T, kv_stride, pos0);
}
extern "C" {
// Device-pos variant (CUDA-graph capture): reads the sequence position from a
// device scalar `*d_pos` instead of a by-value host arg, so one instantiated
// graph replays correctly as pos advances. The math is rope_core, byte-identical
// to tl_rope — only the pos SOURCE differs (register arg -> device load).
__global__ void tl_rope_dpos(const float* __restrict__ x,
                             const float* __restrict__ bias,
                             float* __restrict__ out, unsigned T, unsigned D,
                             const unsigned* __restrict__ d_pos, float base,
                             unsigned rot) {
  rope_core(x, bias, out, T, D, *d_pos, base, rot);
}

// Device-scalar increment: *p += 1. Bumps the shared decode pos counter at the
// tail of a captured forward so each replay advances it. Launched <<<1,1>>>.
__global__ void tl_incr_u32(unsigned* p) { ++*p; }

}  // extern "C"

// ---- One row of a table into f32, the row the f32 index idx[0] names: a decode
// step run ahead gathers its input from the token the last step's argmax_to
// left on the device (an index past the table gives zeros). f32 and bf16 are
// read by element strides, so a [rows, K] table and its [K, rows] transpose
// both serve; the block formats a thread eight weights, as their GEMMs read
// them (dq8_), but kept f32.
template <class T>
__device__ __forceinline__ void embed_row_strided_core(const T* __restrict__ W, const float* __restrict__ idx,
                                                       float* __restrict__ out, unsigned K, unsigned rows,
                                                       unsigned rs, unsigned cs) {
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= K) return;
  const unsigned n = (unsigned)(idx[0] + 0.5f);
  float v = 0.0f;
  if (n < rows) {
    const T w = W[(size_t)n * rs + (size_t)j * cs];
    if constexpr (std::is_same<T, float>::value) v = w;
    else v = __bfloat162float(w);
  }
  out[j] = v;
}
template <int WF>
__device__ __forceinline__ void embed_row_quant_core(const unsigned char* __restrict__ W,
                                                     const float* __restrict__ idx, float* __restrict__ out,
                                                     unsigned K, unsigned rows) {
  const unsigned gk = (blockIdx.x * blockDim.x + threadIdx.x) * 8u;
  if (gk >= K) return;
  const unsigned n = (unsigned)(idx[0] + 0.5f);
  float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (n < rows) dq8_<WF>(W + (size_t)n * quant_row_bytes_<WF>(K), gk, v);
#pragma unroll
  for (int i = 0; i < 8; i++) out[gk + i] = v[i];
}
TL_ALIGN(tl_embed_row_q4k, 8)
TL_ALIGN(tl_embed_row_q6k, 2)
TL_ALIGN(tl_embed_row_q80, 2)
extern "C" __global__ void tl_embed_row_f32(const float* W, const float* idx, float* out, unsigned K,
                                            unsigned rows, unsigned rs, unsigned cs) {
  embed_row_strided_core(W, idx, out, K, rows, rs, cs);
}
extern "C" __global__ void tl_embed_row_bf16(const __nv_bfloat16* W, const float* idx, float* out,
                                             unsigned K, unsigned rows, unsigned rs, unsigned cs) {
  embed_row_strided_core(W, idx, out, K, rows, rs, cs);
}
extern "C" __global__ void tl_embed_row_q4k(const unsigned char* W, const float* idx, float* out,
                                            unsigned K, unsigned rows) {
  embed_row_quant_core<1>(W, idx, out, K, rows);
}
extern "C" __global__ void tl_embed_row_q6k(const unsigned char* W, const float* idx, float* out,
                                            unsigned K, unsigned rows) {
  embed_row_quant_core<2>(W, idx, out, K, rows);
}
extern "C" __global__ void tl_embed_row_q80(const unsigned char* W, const float* idx, float* out,
                                            unsigned K, unsigned rows) {
  embed_row_quant_core<3>(W, idx, out, K, rows);
}

// ---- gpu_abi_llm.h's sample_ref on the device: metal_kernels_llm.metal's
// sample_ (its comment says how the launches split the work), a warp for a
// simdgroup. The arithmetic is the host's, rounded as it rounds (no fused
// multiply-add).

// Philox4x32-10 of (c0, c1, 0, 0) under (k0, 0): the first word.
__device__ __forceinline__ unsigned sample_philox_(unsigned c0, unsigned c1, unsigned k0) {
  unsigned c2 = 0, c3 = 0, k1 = 0;
#pragma unroll
  for (int r = 0; r < 10; r++) {
    const unsigned n0 = __umulhi(0xCD9E8D57u, c2) ^ c1 ^ k0, n2 = __umulhi(0xD2511F53u, c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * c2;
    c3 = 0xD2511F53u * c0;
    c0 = n0;
    c2 = n2;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}
// gpu_abi_llm.h's sample_uniform: 24 bits in [0, 1), or 23 centred in (0, 1).
__device__ __forceinline__ float sample_uniform_(unsigned bits, bool open) {
  return open ? (float(bits >> 9) + 0.5f) * 0x1p-23f : float(bits >> 8) * 0x1p-24f;
}
constexpr unsigned sample_pick_element_ = 0xFFFFFFFFu;  // gpu_abi_llm.h's sample_pick_element
constexpr int sample_thread_keys_ = 16;                         // a thread's keys

// The first of a thread's keys after (last, lasti): argmax_take_'s rank.
__device__ __forceinline__ void sample_lane_next_(const float (&x)[sample_thread_keys_],
                                                  const int (&xi)[sample_thread_keys_], float last,
                                                  int lasti, float& v, int& vi) {
  v = -INFINITY;
  vi = -1;
#pragma unroll
  for (int e = 0; e < sample_thread_keys_; e++) {
    if (x[e] < last || (x[e] == last && xi[e] > lasti)) argmax_take_(v, vi, x[e], xi[e]);
  }
}

extern "C" __global__ void __launch_bounds__(256)
    tl_sample(const float* __restrict__ in, const float* __restrict__ idx, unsigned* counter,
              float* out_v, float* out_i, unsigned n, unsigned chunk, unsigned k, unsigned mapped,
              unsigned gumbel, unsigned draw, unsigned seed, float inv_t, float top_p) {
  __shared__ float lv[8][64];  // each warp's k first (policy::max_top_k)
  __shared__ int li[8][64];
  __shared__ float tv[64];     // the drawing block's k first
  __shared__ int ti[64];
  const unsigned tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const unsigned ctr = counter[0];
  const unsigned begin = blockIdx.x * chunk, end = min(n, begin + chunk);
  float x[sample_thread_keys_];
  int xi[sample_thread_keys_];
#pragma unroll
  for (int e = 0; e < sample_thread_keys_; e++) {
    const unsigned i = begin + tid + e * 256u;
    x[e] = -INFINITY;
    xi[e] = -1;
    if (i >= end) continue;
    xi[e] = mapped ? (int)idx[i] : (int)i;
    x[e] = in[i];
    if (gumbel && !mapped)
      x[e] = __fadd_rn(__fmul_rn(x[e], inv_t),
                       -logf(-logf(sample_uniform_(sample_philox_(ctr, i, seed), true))));
  }
  float hv;
  int hi;
  sample_lane_next_(x, xi, INFINITY, -1, hv, hi);
  for (unsigned r = 0; r < k; r++) {
    float v = hv;
    int vi = hi;
    argmax_warp_(v, vi);
    if (lane == 0) lv[wid][r] = v, li[wid][r] = vi;
    if (hi == vi && vi >= 0) sample_lane_next_(x, xi, hv, hi, hv, hi);
  }
  __syncthreads();
  if (wid != 0) return;
  // The eight lists, a lane each.
  unsigned at = 0;
  hv = lane < 8 ? lv[lane][0] : -INFINITY;
  hi = lane < 8 ? li[lane][0] : -1;
  for (unsigned r = 0; r < k; r++) {
    float v = hv;
    int vi = hi;
    argmax_warp_(v, vi);
    if (lane == 0) {
      if (draw) {
        tv[r] = v;
        ti[r] = vi;
      } else {
        out_v[blockIdx.x * k + r] = v;
        out_i[blockIdx.x * k + r] = (float)vi;
      }
    }
    if (hi == vi && vi >= 0) {
      at++;
      hv = at < k ? lv[lane][at] : -INFINITY;
      hi = at < k ? li[lane][at] : -1;
    }
  }
  if (!draw || tid != 0) return;
  unsigned kk = 0;
  while (kk < k && ti[kk] >= 0) kk++;
  int tok = 0;
  if (kk > 0 && gumbel) {
    tok = ti[0];
  } else if (kk > 0) {
    // Each key's weight over the first's, in order.
    const float top = tv[0];
    float z = 0.0f;
    for (unsigned j = 0; j < kk; j++) z += tv[j] = expf(__fmul_rn(tv[j] - top, inv_t));
    unsigned kept = 1;
    float acc = tv[0];
    while (kept < kk && acc < __fmul_rn(top_p, z)) acc += tv[kept++];
    const float target =
        __fmul_rn(sample_uniform_(sample_philox_(ctr, sample_pick_element_, seed), false), acc);
    float run = 0.0f;
    tok = ti[kept - 1];
    for (unsigned j = 0; j < kept; j++) {
      if (target < (run += tv[j])) {
        tok = ti[j];
        break;
      }
    }
  }
  out_i[0] = (float)tok;
  counter[0] = ctr + 1;
}
