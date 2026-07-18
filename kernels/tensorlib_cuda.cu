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

// ---- argmax over a single length-n vector -> one int index (out[0]) ----
// Greedy decoding's last mile: reduce logits[VOCAB] on-device so only a 4-byte
// index crosses PCIe (vs the 608KB logits D2H the host argmax needed). One
// block, grid-stride load, tree reduction carrying (value,index). Tie-break
// matches the host `v[i] > v[bi]` loop exactly — the SMALLEST index wins on
// ties — so greedy output stays bit-identical to the CPU reference.
extern "C" __global__ void tl_argmax(const float* __restrict__ in, int* out,
                                     unsigned n) {
  extern __shared__ float smem[];
  float* sval = smem;
  int* sidx = reinterpret_cast<int*>(smem + blockDim.x);
  unsigned t = threadIdx.x, T = blockDim.x;
  float best = -3.402823466e+38f;
  int besti = 0;
  // Strict '>' so within a thread's stripe the first (smallest) index wins.
  for (unsigned i = t; i < n; i += T) {
    float v = in[i];
    if (v > best) { best = v; besti = (int)i; }
  }
  sval[t] = best;
  sidx[t] = besti;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) {
      float ov = sval[t + s];
      int oi = sidx[t + s];
      // Take the other half if strictly greater, or equal with a smaller index.
      if (ov > sval[t] || (ov == sval[t] && oi < sidx[t])) {
        sval[t] = ov;
        sidx[t] = oi;
      }
    }
    __syncthreads();
  }
  if (t == 0) out[0] = sidx[0];
}

// ---- fused RMSNorm over a single length-n row: out = x * rsqrt(mean(x^2)+eps) * w
// One block, grid-stride sum-of-squares reduction, then a normalize+scale pass.
// Matches the array composition (x*x).mean(-1) -> 1/sqrt(ms+eps) -> *x *w exactly
// (f32, 1/sqrtf not the approximate rsqrtf) so greedy output is unchanged. Kills
// the ~7 array-op launches per RMSNorm (2/layer + 1 final) in the decode step.
extern "C" __global__ void tl_rmsnorm(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ out, unsigned n,
                                      float eps) {
  extern __shared__ float sm[];
  unsigned t = threadIdx.x, T = blockDim.x;
  float acc = 0.0f;
  for (unsigned i = t; i < n; i += T) { float v = x[i]; acc += v * v; }
  sm[t] = acc;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) sm[t] += sm[t + s];
    __syncthreads();
  }
  float inv = 1.0f / sqrtf(sm[0] / (float)n + eps);
  for (unsigned i = t; i < n; i += T) out[i] = x[i] * inv * w[i];
}

// ---- fused residual-add + RMSNorm: xout = a+b; hout = rmsnorm(xout,eps)*w ----
// Folds a decode layer's residual add into the following norm (the o-proj->norm
// and mlp->next-input-norm seams), writing BOTH the residual sum (needed
// downstream as the next residual base) and the normalized+scaled output. One
// block. xout may alias a (in-place residual). Removes 2 elementwise launches
// per layer — pure per-kernel-latency wins on the launch-bound decode step.
extern "C" __global__ void tl_add_rmsnorm(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ w,
                                          float* __restrict__ xout,
                                          float* __restrict__ hout, unsigned n,
                                          float eps) {
  extern __shared__ float sm[];
  unsigned t = threadIdx.x, T = blockDim.x;
  float acc = 0.0f;
  for (unsigned i = t; i < n; i += T) {
    float v = a[i] + b[i];
    xout[i] = v;
    acc += v * v;
  }
  sm[t] = acc;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (t < s) sm[t] += sm[t + s];
    __syncthreads();
  }
  float inv = 1.0f / sqrtf(sm[0] / (float)n + eps);
  for (unsigned i = t; i < n; i += T) hout[i] = xout[i] * inv * w[i];
}

// ---- fused SwiGLU: out = silu(gate) * up = (gate * sigmoid(gate)) * up ----
// Elementwise over n (the FF dim). Matches array silu(gate)*up; replaces the
// sigmoid+mul+mul launch chain with one kernel.
extern "C" __global__ void tl_swiglu(const float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     float* __restrict__ out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  out[i] = (g / (1.0f + expf(-g))) * up[i];
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
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (nwarps == 1) {
    if (lane == 0) y[n] = acc;
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
    if (lane == 0) y[n] = acc;
  }
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
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (nwarps == 1) {
    if (lane == 0) y[n] = acc;
    return;
  }
  extern __shared__ float gemv_q4_sdata[];
  if (lane == 0) gemv_q4_sdata[warp_id] = acc;
  __syncthreads();
  if (warp_id == 0) {
    acc = (lane < nwarps) ? gemv_q4_sdata[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = acc;
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
// One block per head, blockDim = head_dim AD (NW = AD/32 warps). Each warp owns
// keys i = warp, warp+NW, ...; its 32 lanes cooperatively reduce the AD-long
// score (coalesced K/V row reads, warp-shuffle reduction, no block sync in the
// loop), keeping per-warp (m,l,acc) running state. A final shared-memory step
// merges the NW warps' partial softmax states (flash rescale-by-exp(m_w-m)).
//
// head_dim generalization (M9): the kernels are templated on AD (compile-time,
// because acc[]/sacc[][] are register/smem arrays sized AD/32) and instantiated
// for {64,128} — the head dims of the target models (Qwen2 = 64, llama-7B = 128).
// Each lane holds NW = AD/32 dims (d = lane, lane+32, ..., lane+(NW-1)*32) and
// there are exactly NW warps, so the warp count and the per-lane dim count
// coincide. extern "C" wrappers below give each instantiation a stable symbol
// (the driver loads kernels by name): tl_attn_decode_f32[_64], etc.
}  // close extern "C": the __device__ core templates below can't have C linkage;
   // each __global__ wrapper re-declares its own extern "C" for a stable symbol.

// KV-cache storage-dtype seam (bf16 KV, M9). The cores below are templated on the
// K/V element type KT ∈ {float, __nv_bfloat16}: kv_ld widens a cached element to
// f32 for the dot/accumulate, kv_st narrows an f32 projection on store. KT=float
// is the identity (the original f32 path, byte-for-byte unchanged); KT=bf16 halves
// the cache bytes the attention kernels stream every step. q/out/scratch stay f32;
// kv_stride and all indices are ELEMENT counts, so only the element width changes.
__device__ __forceinline__ float kv_ld(float x) { return x; }
__device__ __forceinline__ float kv_ld(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void kv_st(float* p, float x) { *p = x; }
__device__ __forceinline__ void kv_st(__nv_bfloat16* p, float x) { *p = __float2bfloat16(x); }

template <int AD, typename KT = float>
__device__ void attn_decode_core(const float* __restrict__ q,
                                  const KT* __restrict__ K,
                                  const KT* __restrict__ V,
                                  float* __restrict__ out, unsigned ctx,
                                  unsigned kv_stride, unsigned group,
                                  float scale) {
  constexpr int NW = AD / 32;                   // warps == dims-per-lane
  const unsigned h = blockIdx.x;                // query head
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  const unsigned tid = threadIdx.x;             // 0..AD-1
  const unsigned lane = tid & 31u, warp = tid >> 5;  // NW warps of 32
  const float* qh = q + (size_t)h * AD;
  // K,V are [H_kv, max_ctx, D]; kv_stride = max_ctx*D lets a persistent cache be
  // read as its valid prefix [0,ctx) (kv_stride==ctx*AD is the no-cache case).
  const KT* Kh = K + (size_t)kv_h * kv_stride;
  const KT* Vh = V + (size_t)kv_h * kv_stride;

  __shared__ float q_sh[AD];
  q_sh[tid] = qh[tid];
  __syncthreads();

  float m = -1e30f, l = 0.0f;
  float acc[NW] = {};  // lane holds dims d = lane, lane+32, ..., lane+(NW-1)*32

  for (unsigned i = warp; i < ctx; i += (unsigned)NW) {
    const KT* Ki = Kh + (size_t)i * AD;
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < NW; r++)
      partial += q_sh[lane + r * 32] * kv_ld(Ki[lane + r * 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    float s = __shfl_sync(0xffffffffu, partial, 0) * scale;  // lane0 has the sum

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

  // merge the NW warps' partial softmax states through shared memory
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
  out[(size_t)h * AD + tid] = o / gl;
}
extern "C" __global__ void tl_attn_decode_f32(const float* q, const float* K,
    const float* V, float* out, unsigned ctx, unsigned kv_stride,
    unsigned group, float scale) {
  attn_decode_core<128>(q, K, V, out, ctx, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_decode_f32_64(const float* q, const float* K,
    const float* V, float* out, unsigned ctx, unsigned kv_stride,
    unsigned group, float scale) {
  attn_decode_core<64>(q, K, V, out, ctx, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_decode_bf16(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned ctx,
    unsigned kv_stride, unsigned group, float scale) {
  attn_decode_core<128, __nv_bfloat16>(q, K, V, out, ctx, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_decode_bf16_64(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned ctx,
    unsigned kv_stride, unsigned group, float scale) {
  attn_decode_core<64, __nv_bfloat16>(q, K, V, out, ctx, kv_stride, group, scale);
}

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
  constexpr int NW = AD / 32;
  const unsigned h = blockIdx.x, s = blockIdx.y, S = gridDim.y;
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u, warp = tid >> 5;
  const float* qh = q + (size_t)h * AD;
  const KT* Kh = K + (size_t)kv_h * kv_stride;
  const KT* Vh = V + (size_t)kv_h * kv_stride;
  const unsigned k0 = s * chunk;
  const unsigned k1 = (k0 + chunk < ctx) ? (k0 + chunk) : ctx;

  __shared__ float q_sh[AD];
  q_sh[tid] = qh[tid];
  __syncthreads();

  float m = -1e30f, l = 0.0f;
  float acc[NW] = {};
  for (unsigned i = k0 + warp; i < k1; i += (unsigned)NW) {
    const KT* Ki = Kh + (size_t)i * AD;
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < NW; r++)
      partial += q_sh[lane + r * 32] * kv_ld(Ki[lane + r * 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    float sc = __shfl_sync(0xffffffffu, partial, 0) * scale;
    float m_new = fmaxf(m, sc);
    float corr = __expf(m - m_new);
    float p = __expf(sc - m_new);
    l = l * corr + p;
    const KT* Vi = Vh + (size_t)i * AD;
#pragma unroll
    for (int r = 0; r < NW; r++)
      acc[r] = acc[r] * corr + p * kv_ld(Vi[lane + r * 32]);
    m = m_new;
  }

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
  const size_t sidx = (size_t)h * S + s;
  if (tid == 0) {
    pm[sidx] = gm;
    pl[sidx] = gl;
  }
  pacc[sidx * AD + tid] = o;  // un-normalized accumulator at local max gm
}
extern "C" __global__ void tl_attn_decode_split(const float* q, const float* K,
    const float* V, float* pm, float* pl, float* pacc, unsigned ctx,
    unsigned kv_stride, unsigned group, unsigned chunk, float scale) {
  attn_decode_split_core<128>(q, K, V, pm, pl, pacc, ctx, kv_stride, group,
                              chunk, scale);
}
extern "C" __global__ void tl_attn_decode_split_64(const float* q,
    const float* K, const float* V, float* pm, float* pl, float* pacc,
    unsigned ctx, unsigned kv_stride, unsigned group, unsigned chunk,
    float scale) {
  attn_decode_split_core<64>(q, K, V, pm, pl, pacc, ctx, kv_stride, group,
                             chunk, scale);
}
extern "C" __global__ void tl_attn_decode_split_bf16(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* pm, float* pl,
    float* pacc, unsigned ctx, unsigned kv_stride, unsigned group,
    unsigned chunk, float scale) {
  attn_decode_split_core<128, __nv_bfloat16>(q, K, V, pm, pl, pacc, ctx,
                                             kv_stride, group, chunk, scale);
}
extern "C" __global__ void tl_attn_decode_split_bf16_64(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* pm, float* pl,
    float* pacc, unsigned ctx, unsigned kv_stride, unsigned group,
    unsigned chunk, float scale) {
  attn_decode_split_core<64, __nv_bfloat16>(q, K, V, pm, pl, pacc, ctx,
                                            kv_stride, group, chunk, scale);
}

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
extern "C" __global__ void tl_kv_append_bf16(__nv_bfloat16* Kc,
                                             __nv_bfloat16* Vc,
                                             const float* k_new,
                                             const float* v_new, unsigned pos,
                                             unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, pos, kv_stride);
}

// Bulk-fill the cache from a prefill's k,v (each [H_kv, T, D] contiguous) into
// the cache [H_kv, max_ctx, D] rows [0,T). The two differ only in the row stride
// (T vs max_ctx), so this is a strided copy. D == blockDim.x. grid = (H_kv, T).
template <typename KT>
__device__ void kv_fill_core(KT* __restrict__ Kc, KT* __restrict__ Vc,
                             const float* __restrict__ K,
                             const float* __restrict__ V, unsigned T,
                             unsigned kv_stride) {
  const unsigned h = blockIdx.x, p = blockIdx.y, d = threadIdx.x, D = blockDim.x;
  const size_t dst = (size_t)h * kv_stride + (size_t)p * D + d;
  const size_t src = ((size_t)h * T + p) * D + d;
  kv_st(&Kc[dst], K[src]);  // narrow to KT (identity for f32)
  kv_st(&Vc[dst], V[src]);
}
extern "C" __global__ void tl_kv_fill(float* Kc, float* Vc, const float* K,
                                      const float* V, unsigned T,
                                      unsigned kv_stride) {
  kv_fill_core(Kc, Vc, K, V, T, kv_stride);
}
extern "C" __global__ void tl_kv_fill_bf16(__nv_bfloat16* Kc, __nv_bfloat16* Vc,
                                           const float* K, const float* V,
                                           unsigned T, unsigned kv_stride) {
  kv_fill_core(Kc, Vc, K, V, T, kv_stride);
}

// Causal prefill attention: process all T query positions of a prompt at once.
// Query at position p attends to keys 0..p (the causal mask). q,out are
// [H_q, T, D]; K,V are the [H_kv, kv_max, D] cache read over [0,p] via kv_stride.
// Reuses the decode online-softmax (no T×T scores materialized), one block per
// (head, query pos) so the grid = H_q×T fills the SMs (no split-KV needed). This
// is the correctness-first baseline — each query re-streams its keys from DRAM
// (O(T²) traffic); a query×key tiled flash-attention is the deferred tuning pass.
// grid = (H_q, T), blockDim = AD = head_dim (NW warps).
template <int AD, typename KT = float>
__device__ void attn_prefill_core(const float* __restrict__ q,
                                  const KT* __restrict__ K,
                                  const KT* __restrict__ V,
                                  float* __restrict__ out, unsigned T,
                                  unsigned kv_stride, unsigned group,
                                  float scale) {
  constexpr int NW = AD / 32;
  const unsigned h = blockIdx.x;                // query head
  const unsigned p = blockIdx.y;                // query pos, attends keys 0..p
  const unsigned kv_h = group ? h / group : h;  // GQA: q head -> shared kv head
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u, warp = tid >> 5;
  const float* qh = q + ((size_t)h * T + p) * AD;  // q is [H_q, T, D]
  const KT* Kh = K + (size_t)kv_h * kv_stride;
  const KT* Vh = V + (size_t)kv_h * kv_stride;

  __shared__ float q_sh[AD];
  q_sh[tid] = qh[tid];
  __syncthreads();

  float m = -1e30f, l = 0.0f;
  float acc[NW] = {};
  for (unsigned i = warp; i <= p; i += (unsigned)NW) {  // causal: keys 0..p only
    const KT* Ki = Kh + (size_t)i * AD;
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < NW; r++)
      partial += q_sh[lane + r * 32] * kv_ld(Ki[lane + r * 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    float s = __shfl_sync(0xffffffffu, partial, 0) * scale;
    float m_new = fmaxf(m, s);
    float corr = __expf(m - m_new);
    float pp = __expf(s - m_new);
    l = l * corr + pp;
    const KT* Vi = Vh + (size_t)i * AD;
#pragma unroll
    for (int r = 0; r < NW; r++)
      acc[r] = acc[r] * corr + pp * kv_ld(Vi[lane + r * 32]);
    m = m_new;
  }

  // merge the NW warps' partial softmax states (empty warps carry m=-1e30,l=0 →
  // contribute exp(-inf)=0, correct for the short causal prefixes at small p)
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
  out[((size_t)h * T + p) * AD + tid] = o / gl;
}
extern "C" __global__ void tl_attn_prefill_f32(const float* q, const float* K,
    const float* V, float* out, unsigned T, unsigned kv_stride, unsigned group,
    float scale) {
  attn_prefill_core<128>(q, K, V, out, T, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_prefill_f32_64(const float* q,
    const float* K, const float* V, float* out, unsigned T, unsigned kv_stride,
    unsigned group, float scale) {
  attn_prefill_core<64>(q, K, V, out, T, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_prefill_bf16(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned T,
    unsigned kv_stride, unsigned group, float scale) {
  attn_prefill_core<128, __nv_bfloat16>(q, K, V, out, T, kv_stride, group, scale);
}
extern "C" __global__ void tl_attn_prefill_bf16_64(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned T,
    unsigned kv_stride, unsigned group, float scale) {
  attn_prefill_core<64, __nv_bfloat16>(q, K, V, out, T, kv_stride, group, scale);
}
extern "C" {  // reopen: the remaining kernels rely on the file-level C linkage

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama) convention.
// x is [rows, D] contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D]
// with T=1). Row r's head-dim vector is at position pos + (r % T). Pairs
// (j, j+D/2) rotate by angle = position · base^(-2j/D). grid = rows, block = D/2.
// RoPE with an optional fused bias: rotates x (+bias if non-null). Folding q/k's
// bias-add into rope removes 2 elementwise launches per decode layer.
__global__ void tl_rope(const float* __restrict__ x,
                        const float* __restrict__ bias, float* __restrict__ out,
                        unsigned T, unsigned D, unsigned pos, float base) {
  const unsigned r = blockIdx.x, j = threadIdx.x;  // j in 0..D/2-1
  const unsigned half = D >> 1;
  if (j >= half) return;
  const unsigned t = T ? (r % T) : 0u;
  const float position = (float)(pos + t);
  const float theta = __powf(base, -2.0f * (float)j / (float)D);
  const float ang = position * theta;
  const float c = __cosf(ang), s = __sinf(ang);
  const size_t bi = (size_t)r * D;
  float x0 = x[bi + j];
  float x1 = x[bi + j + half];
  if (bias) {
    x0 += bias[bi + j];
    x1 += bias[bi + j + half];
  }
  out[bi + j] = x0 * c - x1 * s;
  out[bi + j + half] = x0 * s + x1 * c;
}

}  // extern "C"
