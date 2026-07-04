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

}  // extern "C"
