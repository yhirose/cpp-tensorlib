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
}  // close extern "C": the __device__ core template below can't have C linkage;
   // each __global__ wrapper re-declares its own for a stable symbol.
template <bool ADD>
__device__ __forceinline__ void rmsnorm_core(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ w, float* __restrict__ xout,
    float* __restrict__ hout, unsigned n, float eps) {
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
  for (unsigned i = t; i < n; i += T)
    hout[base + i] = (ADD ? xout[base + i] : a[base + i]) * inv * w[i];
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
extern "C" {  // reopen: the remaining kernels rely on the file-level C linkage

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

// tl_pad/tl_fold's per-call shape/stride metadata is uploaded to a scratch
// device buffer by cuda.h's pad()/fold(); this caps how many int64 slots that
// buffer (and each kernel's on-stack index array) needs to hold. im2col's
// chained unfold/pad calls on a conv input (N,C,H,W -> ...,winH,winW) stay
// well under it.
#define TL_PAD_FOLD_MAX_RANK 8

// ---- pad: copy a (contiguous) into a zero-initialized larger buffer,
// shifted by `before` elements along one axis. The caller pre-zeros `out` on
// the device (MemsetD8) since this kernel only visits a's own n elements —
// the border cells pad introduces are never written here. meta packs
// [a_shape(rank), out_strides(rank)] as int64; i is already a's own flat
// contiguous index (a is required contiguous — see cuda.h's pad()), so no
// a_strides are needed to read a[i].
extern "C" __global__ void tl_pad(const float* __restrict__ a,
                                  float* __restrict__ out,
                                  const long long* __restrict__ meta, int rank,
                                  unsigned shift, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long* a_shape = meta;
  const long long* out_strides = meta + rank;
  long long rem = i, dst = 0;
  for (int d = rank - 1; d >= 0; --d) {
    long long dim = a_shape[d];
    long long coord = rem % dim;
    rem /= dim;
    dst += coord * out_strides[d];
  }
  out[dst + shift] = a[i];
}

// ---- fold: unfold's inverse. Scatter-add a (contiguous; its last dim is the
// sliding window) into out (one rank smaller, pre-zeroed by the caller like
// pad's out above), accumulating every window overlap via atomicAdd. meta
// packs [a_shape(rank), out_strides(rank-1)] as int64 — mirrors tl::ref::fold
// exactly, just with the index walk unrolled from a flat thread id instead of
// a host-side nested loop.
extern "C" __global__ void tl_fold(const float* __restrict__ a,
                                   float* __restrict__ out,
                                   const long long* __restrict__ meta,
                                   int rank, int axis, unsigned step,
                                   unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long* a_shape = meta;
  const long long* out_strides = meta + rank;
  long long idx[TL_PAD_FOLD_MAX_RANK];
  long long rem = i;
  for (int d = rank - 1; d >= 0; --d) {
    long long dim = a_shape[d];
    idx[d] = rem % dim;
    rem /= dim;
  }
  int last = rank - 1;
  long long dst = 0;
  for (int d = 0; d < last; ++d) {
    long long di = (d == axis) ? idx[d] * (long long)step + idx[last] : idx[d];
    dst += di * out_strides[d];
  }
  atomicAdd(&out[dst], a[i]);
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
// each operand as float4s. Requires K % 8 == 0 (every transformer dim; the
// launcher checks) — so there is no K tail, only M/N edges, which the loads and
// the epilogue predicate.
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
// The K slab is prefetched into registers while the current one computes
// (tl_sgemm_rb's trick): gateup 547 -> 461 us.
}  // close extern "C": the __device__ core template below can't have C linkage;
   // each __global__ wrapper re-declares its own for a stable symbol.
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
  // Stage the K slab at `kt` into registers (global -> reg), then into shared.
  auto load = [&](unsigned kt) {
    pa = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gA < M) pa = *reinterpret_cast<const float4*>(&A[(size_t)gA * K + kt + ldK]);
    p0 = make_float2(0.f, 0.f);
    p1 = make_float2(0.f, 0.f);
    if (gB < N) {
      uint2 raw = *reinterpret_cast<const uint2*>(&B[(size_t)gB * K + kt + ldK]);
      p0 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.x));
      p1 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162&>(raw.y));
    }
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
extern "C" {  // reopen: the remaining kernels rely on the file-level C linkage

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
  for (unsigned i = k0 + warp; i < k1; i += (unsigned)NW) {
    const KT* Ki = Kh + (size_t)i * AD;
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < NW; r++)
      partial += q_sh[lane + r * 32] * kv_ld(Ki[lane + r * 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    float sc = __shfl_sync(0xffffffffu, partial, 0) * scale;  // lane0's sum
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
// byte-identical to the by-value kernels. f32 KV (Qwen decode).
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
    unsigned max_s = ctx / 128u;  // >=128 keys per split
    if (want > max_s) want = max_s;
    if (want > 1u) S = want;
  }
  unsigned chunk = (ctx + S - 1u) / S;
  chunk = (chunk + 3u) & ~3u;  // multiple of the warp count
  return chunk ? chunk : 4u;
}
extern "C" __global__ void tl_attn_decode_split_dpos(const float* q,
    const float* K, const float* V, float* pm, float* pl, float* pacc,
    const unsigned* __restrict__ d_pos, unsigned kv_stride, unsigned group,
    float scale) {
  const unsigned ctx = *d_pos + 1u;
  attn_decode_split_core<128>(q, K, V, pm, pl, pacc, ctx, kv_stride, group,
                              attn_dpos_chunk(ctx, gridDim.x), scale);
}
extern "C" __global__ void tl_attn_decode_split_64_dpos(const float* q,
    const float* K, const float* V, float* pm, float* pl, float* pacc,
    const unsigned* __restrict__ d_pos, unsigned kv_stride, unsigned group,
    float scale) {
  const unsigned ctx = *d_pos + 1u;
  attn_decode_split_core<64>(q, K, V, pm, pl, pacc, ctx, kv_stride, group,
                             attn_dpos_chunk(ctx, gridDim.x), scale);
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
// Device-pos variant (CUDA-graph capture): write row = *d_pos, read from a device
// scalar so replay targets the advancing cache row. Body is kv_append_core,
// byte-identical to tl_kv_append — only the pos source differs. f32 KV only
// (Qwen's decode path); a bf16 sibling would mirror this if a bf16-KV model is
// captured.
extern "C" __global__ void tl_kv_append_dpos(float* Kc, float* Vc,
                                             const float* k_new,
                                             const float* v_new,
                                             const unsigned* __restrict__ d_pos,
                                             unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, *d_pos, kv_stride);
}
extern "C" __global__ void tl_kv_append_bf16(__nv_bfloat16* Kc,
                                             __nv_bfloat16* Vc,
                                             const float* k_new,
                                             const float* v_new, unsigned pos,
                                             unsigned kv_stride) {
  kv_append_core(Kc, Vc, k_new, v_new, pos, kv_stride);
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
extern "C" __global__ void tl_kv_fill_bf16(__nv_bfloat16* Kc, __nv_bfloat16* Vc,
                                           const float* K, const float* V,
                                           unsigned T, unsigned kv_stride,
                                           unsigned pos0) {
  kv_fill_core(Kc, Vc, K, V, T, kv_stride, pos0);
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
// == 0, so the address is 16-byte aligned for f32 and 8-byte for bf16 — one
// vector load per four elements instead of four scalar ones (+9% on the staging).
__device__ __forceinline__ void kv_ld4(const float* p, float* o) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
__device__ __forceinline__ void kv_ld4(const __nv_bfloat16* p, float* o) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const float2 f0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(raw.x));
  const float2 f1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(raw.y));
  o[0] = f0.x; o[1] = f0.y; o[2] = f1.x; o[3] = f1.y;
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
  constexpr int TK = (AD == 64) ? 32 : 16;  // keys per tile (shared budget)
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
extern "C" __global__ void tl_attn_prefill_tiled_f32(const float* q,
    const float* K, const float* V, float* out, unsigned T, unsigned kv_stride,
    unsigned group, float scale, unsigned pos0) {
  attn_prefill_tiled_core<128, 2>(q, K, V, out, T, kv_stride, group, scale, pos0);
}
extern "C" __global__ void tl_attn_prefill_tiled_f32_64(const float* q,
    const float* K, const float* V, float* out, unsigned T, unsigned kv_stride,
    unsigned group, float scale, unsigned pos0) {
  attn_prefill_tiled_core<64, 4>(q, K, V, out, T, kv_stride, group, scale, pos0);
}
extern "C" __global__ void tl_attn_prefill_tiled_bf16(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned T,
    unsigned kv_stride, unsigned group, float scale, unsigned pos0) {
  attn_prefill_tiled_core<128, 2, __nv_bfloat16>(q, K, V, out, T, kv_stride,
                                                 group, scale, pos0);
}
extern "C" __global__ void tl_attn_prefill_tiled_bf16_64(const float* q,
    const __nv_bfloat16* K, const __nv_bfloat16* V, float* out, unsigned T,
    unsigned kv_stride, unsigned group, float scale, unsigned pos0) {
  attn_prefill_tiled_core<64, 4, __nv_bfloat16>(q, K, V, out, T, kv_stride,
                                                group, scale, pos0);
}

extern "C" {  // reopen: the remaining kernels rely on the file-level C linkage

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama) convention.
// x is [rows, D] contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D]
// with T=1). Row r's head-dim vector is at position pos + (r % T). Pairs
// (j, j+D/2) rotate by angle = position · base^(-2j/D). grid = rows, block = D/2.
// An optional fused bias is added before the rotation: folding q/k's bias-add
// into rope removes 2 elementwise launches per decode layer.
static __device__ __forceinline__ void rope_core(
    const float* __restrict__ x, const float* __restrict__ bias,
    float* __restrict__ out, unsigned T, unsigned D, unsigned pos, float base) {
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
__global__ void tl_rope(const float* __restrict__ x,
                        const float* __restrict__ bias, float* __restrict__ out,
                        unsigned T, unsigned D, unsigned pos, float base) {
  rope_core(x, bias, out, T, D, pos, base);
}
// Device-pos variant (CUDA-graph capture): reads the sequence position from a
// device scalar `*d_pos` instead of a by-value host arg, so one instantiated
// graph replays correctly as pos advances. The math is rope_core, byte-identical
// to tl_rope — only the pos SOURCE differs (register arg -> device load).
__global__ void tl_rope_dpos(const float* __restrict__ x,
                             const float* __restrict__ bias,
                             float* __restrict__ out, unsigned T, unsigned D,
                             const unsigned* __restrict__ d_pos, float base) {
  rope_core(x, bias, out, T, D, *d_pos, base);
}

// Device-scalar increment: *p += 1. Bumps the shared decode pos counter at the
// tail of a captured forward so each replay advances it. Launched <<<1,1>>>.
__global__ void tl_incr_u32(unsigned* p) { ++*p; }

}  // extern "C"
