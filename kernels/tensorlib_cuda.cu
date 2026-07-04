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
                            float offset) {
  __shared__ float As[2][TL_BK][TL_BM];  // double-buffered, transposed
  __shared__ float Bs[2][TL_BK][TL_BN];

  const unsigned blockRow = blockIdx.y * TL_BM;
  const unsigned blockCol = blockIdx.x * TL_BN;
  const unsigned tid = threadIdx.x;  // 0..255

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

  load_regs(0);
  store_smem(0);
  __syncthreads();
  int buf = 0;
  for (unsigned kt = 0; kt < k; kt += TL_BK) {
    bool has_next = kt + TL_BK < k;
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

  // --- epilogue: scale/offset + guarded store (mirror the load map) ---
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
        if (gCol < n) C[(size_t)gRow * n + gCol] = acc[i][j] * scale + offset;
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

}  // extern "C"
