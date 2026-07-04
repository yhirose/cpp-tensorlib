// CUDA kernels for the M6 own-CUDA backend — the non-Apple analogue of
// metal_kernels.metal. AOT-compiled to PTX by nvcc and #embed'd into cuda.h,
// then loaded at runtime via the dlopen'd driver API (cuModuleLoadData +
// cuLaunchKernel). No CUDA runtime, no cuBLAS/CUTLASS — own kernels only.
//
// Conventions mirror the Metal kernels exactly so the graph's fused affine
// epilogue works identically on both backends:
//   * every elementwise/gemm kernel computes  out = expr * scale + offset
//     (fma), so fused scalar chains and dot-then-scale cost zero extra passes;
//   * softmax ignores scale/offset (a pre-softmax affine is not meaningful),
//     matching softmax_ in the MSL source;
//   * row_sum/row_max write one value per row with the affine epilogue.
// View offsets are folded host-side (pointer arithmetic on the managed
// allocation), so kernels take already-offset pointers — simpler than Metal's
// setBuffer:offset:. Kernels are extern "C" for stable, unmangled PTX names.
//
// Correctness-first (M6 stage 1). The SGEMM here is a one-output-per-thread
// tiled loop — the tuned cuBLAS-90% ladder is the stage-2 sprint (roadmap M6).

#include <cuda_bf16.h>  // __nv_bfloat16 + __bfloat162float (M7 storage widen)

extern "C" {

// ---- elementwise binary: out = (a OP b) * scale + offset ----
#define TL_EW_BINARY(NAME, EXPR)                                            \
  __global__ void NAME(const float* a, const float* b, float* out,         \
                       unsigned n, float scale, float offset) {             \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) out[i] = (EXPR) * scale + offset;                           \
  }
TL_EW_BINARY(tl_add, a[i] + b[i])
TL_EW_BINARY(tl_sub, a[i] - b[i])
TL_EW_BINARY(tl_mul, a[i] * b[i])
TL_EW_BINARY(tl_div, a[i] / b[i])
#undef TL_EW_BINARY

// ---- unary: out = f(a) * scale + offset (affine = identity f) ----
#define TL_EW_UNARY(NAME, EXPR)                                             \
  __global__ void NAME(const float* a, float* out, unsigned n, float scale, \
                       float offset) {                                      \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) out[i] = (EXPR) * scale + offset;                           \
  }
TL_EW_UNARY(tl_exp, expf(a[i]))
TL_EW_UNARY(tl_log, logf(a[i]))
TL_EW_UNARY(tl_sqrt, sqrtf(a[i]))
TL_EW_UNARY(tl_sigmoid, 1.0f / (1.0f + expf(-a[i])))
TL_EW_UNARY(tl_relu, a[i] > 0.0f ? a[i] : 0.0f)
TL_EW_UNARY(tl_affine, a[i])
#undef TL_EW_UNARY

// ---- row reductions over the last axis (cols); one block per row ----
// Block-wide tree reduction in shared memory; cols may exceed blockDim, so
// grid-stride accumulate first. One output per row, affine epilogue.
#define TL_ROW_REDUCE(NAME, INIT, COMBINE)                                  \
  __global__ void NAME(const float* in, float* out, unsigned rows,          \
                       unsigned cols, float scale, float offset) {          \
    unsigned row = blockIdx.x;                                              \
    if (row >= rows) return;                                                \
    const float* src = in + (size_t)row * cols;                            \
    extern __shared__ float sdata[];                                        \
    unsigned t = threadIdx.x, T = blockDim.x;                              \
    float acc = (INIT);                                                     \
    for (unsigned c = t; c < cols; c += T) {                               \
      float v = src[c];                                                     \
      acc = (COMBINE);                                                      \
    }                                                                       \
    sdata[t] = acc;                                                         \
    __syncthreads();                                                        \
    for (unsigned s = T >> 1; s > 0; s >>= 1) {                            \
      if (t < s) {                                                          \
        float v = sdata[t + s];                                            \
        acc = sdata[t];                                                     \
        sdata[t] = (COMBINE);                                               \
      }                                                                     \
      __syncthreads();                                                      \
    }                                                                       \
    if (t == 0) out[row] = sdata[0] * scale + offset;                      \
  }
TL_ROW_REDUCE(tl_row_sum, 0.0f, acc + v)
TL_ROW_REDUCE(tl_row_max, -3.402823466e+38f, (acc > v ? acc : v))
#undef TL_ROW_REDUCE

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

// ---- SGEMM: C(m,n) = (A @ B) * scale + offset ----
// lda/ldb are row strides; trans flags let a transposed (col-major) view be
// read in place — same layout contract as metal::gemm / accel::gemm:
//   A(i,k) = trans_a ? A[k*lda + i] : A[i*lda + k]
//   B(k,j) = trans_b ? B[j*ldb + k] : B[k*ldb + j]
// One output element per thread. Correctness-first; stage 2 tiles/shared-mem.
__global__ void tl_sgemm(const float* A, const float* B, float* C, unsigned m,
                         unsigned n, unsigned k, unsigned lda, unsigned ldb,
                         unsigned trans_a, unsigned trans_b, float scale,
                         float offset) {
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
  float acc = 0.0f;
  for (unsigned p = 0; p < k; p++) {
    float av = trans_a ? A[(size_t)p * lda + i] : A[(size_t)i * lda + p];
    float bv = trans_b ? B[(size_t)j * ldb + p] : B[(size_t)p * ldb + j];
    acc += av * bv;
  }
  C[(size_t)i * n + j] = acc * scale + offset;
}

// ---- warp-tiled SGEMM fast path (NN, contiguous) ----
// C(m,n) = (A(m,k) @ B(k,n)) * scale + offset. A 128×128×8 blocktile split into
// 64×32 warp tiles (8 warps / 256 threads), each warp iterating WNITER=2 column
// sub-tiles so every thread accumulates an 8×8 register microtile (64 outputs).
// A is staged transposed (As[BK][BM]) so both fragment reads are a contiguous
// float4 per thread. The warp-tiled thread→output map makes the shared-memory
// fragment reads BROADCAST within a warp (a whole warp shares each As/Bs
// address), which is what removes the bank conflicts a plain tid/16,tid%16 map
// hits (its stride-8 B reads serialize 4-way). This is the step that lifts the
// kernel toward cuBLAS; global loads are float4, one __syncthreads per K-slab.
//
// Contract for eligibility (checked host-side in cuda::gemm, else tl_sgemm runs):
// no transpose, lda==k, ldb==n, K%8==0, N%4==0, all base offsets folded in and
// 16B-aligned. M and N block edges are predicated (zero-filled loads, guarded
// stores), so arbitrary M and any N%4==0 are correct.
//
// Split-K (ladder ②): gridDim.z = S partitions the K axis so S× more blocks
// fill the SMs at mid sizes (1024³/2048³ underfill 82 SMs with 128² tiles).
// blockIdx.z picks the split; ksplit is the per-split K chunk (a multiple of
// TL_BK, so slab boundaries stay aligned). When S>1 each split atomicAdds its
// partial into a pre-zeroed C (scale/offset must be identity — the host gates
// on that); when S==1 (ksplit>=k) the epilogue is the normal fused store, so
// the non-split path is bit-identical to before.
#define TL_BM 128
#define TL_BN 128
#define TL_BK 8
#define TL_WM 64   // warp-tile rows
#define TL_WN 32   // warp-tile cols
#define TL_WNI 2   // column sub-iterations per warp
#define TL_TM 8    // thread microtile rows
#define TL_TN 8    // thread microtile cols (TL_WNI * (TN/TL_WNI)=4 per sub-tile)
#define TL_TNSUB 4  // = TL_TN / TL_WNI, cols per warp sub-iteration
__global__ void tl_sgemm_rb(const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C,
                            unsigned m, unsigned n, unsigned k, float scale,
                            float offset, unsigned ksplit) {
  __shared__ float As[2][TL_BK][TL_BM];  // double-buffered, transposed
  __shared__ float Bs[2][TL_BK][TL_BN];

  const unsigned blockRow = blockIdx.y * TL_BM;
  const unsigned blockCol = blockIdx.x * TL_BN;
  const unsigned tid = threadIdx.x;  // 0..255

  // split-K K-range for this z-slice (multiple of TL_BK; identity when S==1)
  const unsigned k0 = blockIdx.z * ksplit;
  if (k0 >= k) return;
  const unsigned k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;

  // warp placement in the block: 8 warps as 2 rows × 4 cols of 64×32 tiles
  const unsigned warp = tid / 32;
  const unsigned lane = tid % 32;
  const unsigned warpRow = warp / (TL_BN / TL_WN);  // 0..1
  const unsigned warpCol = warp % (TL_BN / TL_WN);  // 0..3
  // thread placement in the 64×32 warp tile: (WM/TM)=8 rows × (WN/WNI/TNSUB)=4 cols
  const unsigned threadRowInWarp = lane / (TL_WN / TL_WNI / TL_TNSUB);  // 0..7
  const unsigned threadColInWarp = lane % (TL_WN / TL_WNI / TL_TNSUB);  // 0..3

  // global-load index maps (one float4 per thread per stage: BM*BK/256 = 4)
  const unsigned aRow = tid / (TL_BK / 4);          // 0..127
  const unsigned aColx4 = (tid % (TL_BK / 4)) * 4;  // 0 or 4
  const unsigned bRow = tid / (TL_BN / 4);          // 0..7
  const unsigned bColx4 = (tid % (TL_BN / 4)) * 4;  // 0,4,..,124
  const unsigned gArow = blockRow + aRow;
  const unsigned gBcol = blockCol + bColx4;

  float acc[TL_TM][TL_TN] = {};
  float regM[TL_TM];
  float regN[TL_TN];

  // global loads staged in registers → overlap with compute (double buffer)
  float4 ldgA, ldgB;
  auto load_regs = [&](unsigned kt) {
    ldgA = (gArow < m)
        ? *reinterpret_cast<const float4*>(&A[(size_t)gArow * k + kt + aColx4])
        : make_float4(0, 0, 0, 0);
    ldgB = (gBcol < n)
        ? *reinterpret_cast<const float4*>(&B[(size_t)(kt + bRow) * n + gBcol])
        : make_float4(0, 0, 0, 0);
  };
  auto store_smem = [&](int buf) {
    As[buf][aColx4 + 0][aRow] = ldgA.x;
    As[buf][aColx4 + 1][aRow] = ldgA.y;
    As[buf][aColx4 + 2][aRow] = ldgA.z;
    As[buf][aColx4 + 3][aRow] = ldgA.w;
    *reinterpret_cast<float4*>(&Bs[buf][bRow][bColx4]) = ldgB;
  };

  load_regs(k0);
  store_smem(0);
  __syncthreads();
  int buf = 0;
  for (unsigned kt = k0; kt < k1; kt += TL_BK) {
    bool has_next = kt + TL_BK < k1;
    if (has_next) load_regs(kt + TL_BK);  // prefetch next slab into registers
#pragma unroll
    for (unsigned kk = 0; kk < TL_BK; kk++) {
      unsigned aBase = warpRow * TL_WM + threadRowInWarp * TL_TM;
      reinterpret_cast<float4*>(regM)[0] =
          *reinterpret_cast<float4*>(&As[buf][kk][aBase]);
      reinterpret_cast<float4*>(regM)[1] =
          *reinterpret_cast<float4*>(&As[buf][kk][aBase + 4]);
#pragma unroll
      for (unsigned wn = 0; wn < TL_WNI; wn++) {
        unsigned bBase = warpCol * TL_WN + wn * (TL_WN / TL_WNI) +
                         threadColInWarp * TL_TNSUB;
        reinterpret_cast<float4*>(regN)[wn] =
            *reinterpret_cast<float4*>(&Bs[buf][kk][bBase]);
      }
#pragma unroll
      for (unsigned i = 0; i < TL_TM; i++)
#pragma unroll
        for (unsigned j = 0; j < TL_TN; j++) acc[i][j] += regM[i] * regN[j];
    }
    if (has_next) store_smem(buf ^ 1);  // regs → other buffer after compute
    __syncthreads();
    buf ^= 1;
  }

  // --- epilogue: guarded store (mirror the load map) ---
  // S==1: fused affine store. S>1: atomicAdd the raw partial into a pre-zeroed
  // C (scale/offset are identity on this path, gated host-side). gridDim.z is
  // uniform across the block, so the branch never diverges.
  const bool split = gridDim.z > 1;
#pragma unroll
  for (unsigned i = 0; i < TL_TM; i++) {
    unsigned gRow = blockRow + warpRow * TL_WM + threadRowInWarp * TL_TM + i;
    if (gRow >= m) continue;
#pragma unroll
    for (unsigned wn = 0; wn < TL_WNI; wn++) {
#pragma unroll
      for (unsigned js = 0; js < TL_TNSUB; js++) {
        unsigned j = wn * TL_TNSUB + js;
        unsigned gCol = blockCol + warpCol * TL_WN + wn * (TL_WN / TL_WNI) +
                        threadColInWarp * TL_TNSUB + js;
        if (gCol >= n) continue;
        size_t idx = (size_t)gRow * n + gCol;
        if (split)
          atomicAdd(&C[idx], acc[i][j]);
        else
          C[idx] = acc[i][j] * scale + offset;
      }
    }
  }
}
#undef TL_BM
#undef TL_BN
#undef TL_BK
#undef TL_WM
#undef TL_WN
#undef TL_WNI
#undef TL_TM
#undef TL_TN
#undef TL_TNSUB

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
__global__ void tl_gemv_f32(const float* __restrict__ a,
                            const float* __restrict__ B, float* __restrict__ y,
                            unsigned n, unsigned k, unsigned ksplit) {
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  unsigned k0 = blockIdx.y * ksplit;
  if (k0 >= k) return;
  unsigned k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;
  float acc = 0.0f;
  for (unsigned kk = k0; kk < k1; kk++) acc += a[kk] * B[(size_t)kk * n + col];
  if (gridDim.y > 1)
    atomicAdd(&y[col], acc);
  else
    y[col] = acc;
}
__global__ void tl_gemv_bf16(const float* __restrict__ a,
                             const __nv_bfloat16* __restrict__ B,
                             float* __restrict__ y, unsigned n, unsigned k,
                             unsigned ksplit) {
  unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  unsigned k0 = blockIdx.y * ksplit;
  if (k0 >= k) return;
  unsigned k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;
  float acc = 0.0f;
  for (unsigned kk = k0; kk < k1; kk++)
    acc += a[kk] * __bfloat162float(B[(size_t)kk * n + col]);
  if (gridDim.y > 1)
    atomicAdd(&y[col], acc);
  else
    y[col] = acc;
}
// Vectorized bf16 GEMV: 8 columns/thread via one 16-byte (uint4 = 8×bf16) load,
// so each thread issues f32-width memory transactions instead of scalar 2-byte
// loads — closes the bandwidth gap to the f32 kernel. Requires n % 8 == 0 (all
// transformer dims are; the host gates on it and falls back to the scalar kernel
// otherwise), which also guarantees the 16-byte alignment of every row's load.
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

// ---- M9 fused decode attention (flash-attention, one query row per head) ----
// out(h,:) = softmax(scale * q(h,:) · K(h)^T) · V(h), computed in ONE pass with
// the online-softmax recurrence so the ctx-long scores are never materialized
// and K,V are each read exactly once (the decode floor is KV bandwidth). This
// replaces the array path's 3 launches × 2 materializations × (heads×layers)
// — which is launch/materialize-bound, not compute-bound — for the regime that
// dominates decode-token time.
//
// One block per head, blockDim = head_dim = 128 (4 warps). Each warp owns keys
// i = warp, warp+4, ...; its 32 lanes cooperatively reduce the D=128 score
// (coalesced K/V row reads, warp-shuffle reduction, no block sync in the loop),
// keeping per-warp (m,l,acc) running state. A final shared-memory step merges
// the 4 warps' partial softmax states (flash-attention rescale-by-exp(m_w-m)).
#define TL_AD 128    // head dim
#define TL_ADPT 4    // TL_AD / 32 = dims per lane
__global__ void tl_attn_decode_f32(const float* __restrict__ q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ out, unsigned ctx,
                                   float scale) {
  const unsigned h = blockIdx.x;
  const unsigned tid = threadIdx.x;  // 0..127
  const unsigned lane = tid & 31u, warp = tid >> 5;  // 4 warps of 32
  const float* qh = q + (size_t)h * TL_AD;
  const float* Kh = K + (size_t)h * ctx * TL_AD;
  const float* Vh = V + (size_t)h * ctx * TL_AD;

  __shared__ float q_sh[TL_AD];
  q_sh[tid] = qh[tid];
  __syncthreads();

  float m = -1e30f, l = 0.0f;
  float acc[TL_ADPT] = {};  // lane holds dims d = lane, lane+32, lane+64, +96

  for (unsigned i = warp; i < ctx; i += 4u) {
    const float* Ki = Kh + (size_t)i * TL_AD;
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < TL_ADPT; r++) partial += q_sh[lane + r * 32] * Ki[lane + r * 32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    float s = __shfl_sync(0xffffffffu, partial, 0) * scale;  // lane0 has the sum

    float m_new = fmaxf(m, s);
    float corr = __expf(m - m_new);
    float p = __expf(s - m_new);
    l = l * corr + p;
    const float* Vi = Vh + (size_t)i * TL_AD;
#pragma unroll
    for (int r = 0; r < TL_ADPT; r++) acc[r] = acc[r] * corr + p * Vi[lane + r * 32];
    m = m_new;
  }

  // merge the 4 warps' partial softmax states through shared memory
  __shared__ float sm[4], sl[4], sacc[4][TL_AD];
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
#pragma unroll
  for (int r = 0; r < TL_ADPT; r++) sacc[warp][lane + r * 32] = acc[r];
  __syncthreads();

  float gm = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
  float gl = 0.0f, o = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    float e = __expf(sm[w] - gm);
    gl += sl[w] * e;
    o += sacc[w][tid] * e;
  }
  out[(size_t)h * TL_AD + tid] = o / gl;
}
#undef TL_AD
#undef TL_ADPT

}  // extern "C"
