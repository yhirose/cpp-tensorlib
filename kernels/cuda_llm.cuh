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
