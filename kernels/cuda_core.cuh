#pragma once

// The core layer's kernels (layers: gpu_abi.h): the elementwise, broadcast,
// comparison and scalar ops, the row reductions, argmax, the N-D and index ops,
// pad / fold, and the f32 GEMMs. Included by tensorlib_cuda.cu.

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
TL_EW_BINARY(tl_pow, powf(a[i], b[i]))
#undef TL_EW_BINARY

// ---- elementwise comparison: out = (a OP b) ? 1.0f : 0.0f (no scale/offset
// -- masks don't compose with the affine epilogue). `bstride` is 1 for a
// same-shape `b` and 0 for an explicit size-1 `b` (`x > s` itself is
// tl_gt_s below) -- one kernel covers both, the only two shapes array.h's
// gpu_compare_ dispatches.
#define TL_EW_CMP(NAME, EXPR)                                               \
  __global__ void NAME(const float* a, const float* b, float* out,         \
                       unsigned n, unsigned bstride) {                     \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= n) return;                                                    \
    float bv = b[i * bstride];                                             \
    out[i] = (EXPR) ? 1.0f : 0.0f;                                         \
  }
TL_EW_CMP(tl_gt, a[i] > bv)
TL_EW_CMP(tl_lt, a[i] < bv)
TL_EW_CMP(tl_ge, a[i] >= bv)
TL_EW_CMP(tl_le, a[i] <= bv)
TL_EW_CMP(tl_eq, a[i] == bv)
TL_EW_CMP(tl_ne, a[i] != bv)
#undef TL_EW_CMP

// ---- rank-2 broadcast binary: out[r,c] = f(a[r*ars+c*acs], b[r*brs+c*bcs])
// * scale + offset, into a contiguous [m,n] output. Mirrors metal.h's own
// rank-2 broadcast kernel (bias/row-vector/column-vector/scalar) -- CUDA
// had no kernel for this at all before (binary_bcast() unconditionally
// returned false, sending every broadcasting op through the CPU fallback,
// draining the pending pipeline every time).
#define TL_EW_BINARY_BCAST(NAME, EXPR)                                      \
  __global__ void NAME(const float* a, const float* b, float* out,         \
                       unsigned m, unsigned n, unsigned ars, unsigned acs,  \
                       unsigned brs, unsigned bcs, float scale,             \
                       float offset) {                                     \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                    \
    if (i >= m * n) return;                                                \
    unsigned r = i / n, c = i % n;                                         \
    float av = a[r * ars + c * acs];                                       \
    float bv = b[r * brs + c * bcs];                                       \
    out[i] = (EXPR) * scale + offset;                                      \
  }
TL_EW_BINARY_BCAST(tl_badd, av + bv)
TL_EW_BINARY_BCAST(tl_bsub, av - bv)
TL_EW_BINARY_BCAST(tl_bmul, av * bv)
TL_EW_BINARY_BCAST(tl_bdiv, av / bv)
TL_EW_BINARY_BCAST(tl_bpow, powf(av, bv))
#undef TL_EW_BINARY_BCAST

// ---- N-D broadcast binary: generalizes tl_b*/binary_bcast above to any
// rank (the rank-2 kernel only covers a Linear bias / BatchNorm-shaped
// [N,D] input; a Transformer's [batch,seq,dim] LayerNorm broadcasts a
// [N,S,1] mean against a [N,S,D] input, rank 3). meta packs [out_shape
// (rank), a_strides(rank), b_strides(rank)] -- a_strides/b_strides are the
// *broadcast* strides (0 on a broadcast axis), computed host-side by the
// same detail::broadcast_strides() the CPU oracle already uses, so this
// kernel needs no broadcast logic of its own, just the flat-index decode
// pad/fold's own kernels above already establish.
#define TL_EW_BINARY_BCAST_ND(NAME, EXPR)                                    \
  __global__ void NAME(const float* a, const float* b, float* out,          \
                       const long long* meta, int rank, unsigned n,         \
                       float scale, float offset) {                         \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= n) return;                                                     \
    const long long* out_shape = meta;                                      \
    const long long* a_strides = meta + rank;                               \
    const long long* b_strides = meta + 2 * rank;                           \
    long long rem = i;                                                      \
    long long a_off = 0, b_off = 0;                                         \
    for (int d = rank - 1; d >= 0; --d) {                                   \
      long long dim = out_shape[d];                                         \
      long long coord = rem % dim;                                          \
      rem /= dim;                                                           \
      a_off += coord * a_strides[d];                                        \
      b_off += coord * b_strides[d];                                        \
    }                                                                       \
    float av = a[a_off], bv = b[b_off];                                     \
    out[i] = (EXPR) * scale + offset;                                       \
  }
TL_EW_BINARY_BCAST_ND(tl_badd_nd, av + bv)
TL_EW_BINARY_BCAST_ND(tl_bsub_nd, av - bv)
TL_EW_BINARY_BCAST_ND(tl_bmul_nd, av * bv)
TL_EW_BINARY_BCAST_ND(tl_bdiv_nd, av / bv)
TL_EW_BINARY_BCAST_ND(tl_bpow_nd, powf(av, bv))
#undef TL_EW_BINARY_BCAST_ND

// ---- N-D broadcast ternary select (Tensor.where's GPU dispatch, which
// didn't exist on any backend before this): same flat-index decode as
// tl_badd_nd above, one more operand. meta packs [out_shape(rank),
// cond_strides(rank), a_strides(rank), b_strides(rank)].
__global__ void tl_where_nd(const float* cond, const float* a,
                            const float* b, float* out,
                            const long long* meta, int rank, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long* out_shape = meta;
  const long long* cond_strides = meta + rank;
  const long long* a_strides = meta + 2 * rank;
  const long long* b_strides = meta + 3 * rank;
  long long rem = i;
  long long c_off = 0, a_off = 0, b_off = 0;
  for (int d = rank - 1; d >= 0; --d) {
    long long dim = out_shape[d];
    long long coord = rem % dim;
    rem /= dim;
    c_off += coord * cond_strides[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[i] = cond[c_off] != 0.0f ? a[a_off] : b[b_off];
}

// ---- N-D strided copy (clone() of a permuted/transposed view, which the flat
// one-input kernels cannot read): same flat-index decode as tl_where_nd, one
// operand. meta packs [out_shape(rank), a_strides(rank)].
__global__ void tl_copy_nd(const float* a, float* out, const long long* meta,
                           int rank, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long* out_shape = meta;
  const long long* a_strides = meta + rank;
  long long rem = i;
  long long a_off = 0;
  for (int d = rank - 1; d >= 0; --d) {
    long long dim = out_shape[d];
    a_off += (rem % dim) * a_strides[d];
    rem /= dim;
  }
  out[i] = a[a_off];
}

// ---- sum_to: sum `a` down to a smaller broadcast-target shape (array.h's
// sum_to, the dual of broadcast_to that every arithmetic op's backward uses
// to un-broadcast a gradient). Gather, not scatter: one thread per OUTPUT
// element sums every `a` element that broadcasts onto it, so unlike
// index_add there is no write conflict and no atomics. meta packs
// [a_shape(rank), a_strides(rank), acc(rank)] -- acc is a's shape
// broadcast-aligned against the *output*'s own strides (array.h's
// broadcast_strides(target, out.strides(), a.shape())), 0 on every axis
// being summed over. `reduced_n` is the host-computed product of a_shape
// over exactly those zero-acc axes (1 if there are none, i.e. sum_to is
// really just a shape-preserving copy).
__global__ void tl_sum_to(const float* a, float* out, const long long* meta,
                          int rank, unsigned out_n, unsigned reduced_n) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= out_n) return;
  const long long* a_shape = meta;
  const long long* a_strides = meta + rank;
  const long long* acc = meta + 2 * rank;
  long long base = 0;
  int red_axis[8];
  int red_count = 0;
  for (int d = 0; d < rank; d++) {
    if (acc[d] != 0) {
      long long idx = (t / acc[d]) % a_shape[d];
      base += idx * a_strides[d];
    } else {
      red_axis[red_count++] = d;
    }
  }
  float sum = 0.0f;
  for (unsigned r = 0; r < reduced_n; r++) {
    unsigned rem = r;
    long long off = base;
    for (int k = red_count - 1; k >= 0; --k) {
      int d = red_axis[k];
      unsigned dim = static_cast<unsigned>(a_shape[d]);
      unsigned coord = rem % dim;
      rem /= dim;
      off += static_cast<long long>(coord) * a_strides[d];
    }
    sum += a[off];
  }
  out[t] = sum;
}

// ---- sum_to, one block per output element ------------------------------
// The flat kernel above gives each output a single thread that walks the whole
// reduced range, so a [N, C] -> [C] bias gradient does N adds in sequence per
// column and the reduced axis carries no parallelism at all. Here the block's
// threads stride over that range and finish in shared memory, the same shape
// as the row reductions below. Same meta layout, so the host picks between the
// two on depth alone.
__global__ void tl_sum_to_blocked(const float* a, float* out,
                                  const long long* meta, int rank,
                                  unsigned out_n, unsigned reduced_n) {
  unsigned t = blockIdx.x;
  if (t >= out_n) return;
  const long long* a_shape = meta;
  const long long* a_strides = meta + rank;
  const long long* acc = meta + 2 * rank;
  long long base = 0;
  int red_axis[8];
  int red_count = 0;
  for (int d = 0; d < rank; d++) {
    if (acc[d] != 0) {
      long long idx = (t / acc[d]) % a_shape[d];
      base += idx * a_strides[d];
    } else {
      red_axis[red_count++] = d;
    }
  }
  extern __shared__ float sdata[];
  unsigned tid = threadIdx.x, T = blockDim.x;
  float sum = 0.0f;
  for (unsigned r = tid; r < reduced_n; r += T) {
    unsigned rem = r;
    long long off = base;
    for (int k = red_count - 1; k >= 0; --k) {
      int d = red_axis[k];
      unsigned dim = static_cast<unsigned>(a_shape[d]);
      unsigned coord = rem % dim;
      rem /= dim;
      off += static_cast<long long>(coord) * a_strides[d];
    }
    sum += a[off];
  }
  sdata[tid] = sum;
  __syncthreads();
  for (unsigned s = T >> 1; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[t] = sdata[0];
}

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
TL_EW_UNARY(tl_tanh, tanhf(a[i]))
TL_EW_UNARY(tl_sin, sinf(a[i]))
TL_EW_UNARY(tl_cos, cosf(a[i]))
#undef TL_EW_UNARY

// ---- clamp: out = min(max(a, lo), hi). No scale/offset -- lo/hi occupy
// that role (Clip's forward: culebra's dz.raw_elementwise host loop before
// this got a native op).
__global__ void tl_clamp(const float* a, float* out, unsigned n, float lo,
                         float hi) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = a[i];
  out[i] = v < lo ? lo : (v > hi ? hi : v);
}

// ---- tensor-scalar: out = f(a, s) * scale + offset, s a kernel argument
// (metal.h's scalar_op). Masks take the epilogue like the unaries.
#define TL_EW_SCALAR(NAME, EXPR)                                            \
  __global__ void NAME(const float* a, float* out, unsigned n, float s,     \
                       float scale, float offset) {                         \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) out[i] = (EXPR) * scale + offset;                           \
  }
TL_EW_SCALAR(tl_pow_s, powf(a[i], s))
TL_EW_SCALAR(tl_gt_s, a[i] > s ? 1.0f : 0.0f)
TL_EW_SCALAR(tl_lt_s, a[i] < s ? 1.0f : 0.0f)
TL_EW_SCALAR(tl_ge_s, a[i] >= s ? 1.0f : 0.0f)
TL_EW_SCALAR(tl_le_s, a[i] <= s ? 1.0f : 0.0f)
TL_EW_SCALAR(tl_eq_s, a[i] == s ? 1.0f : 0.0f)
TL_EW_SCALAR(tl_ne_s, a[i] != s ? 1.0f : 0.0f)
#undef TL_EW_SCALAR

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
// index crosses PCIe (vs the 608KB logits D2H the host argmax needed). The
// largest value wins and the SMALLEST index on ties, the host `v[i] > v[bi]`
// loop's choice among numbers, so greedy output stays bit-identical to the CPU
// reference; NaN never wins, and with nothing above -FLT_MAX the index is 0 (every
// pair starts at (-FLT_MAX, 0), and only a larger value replaces it).
// One launch over the GPU: each block reduces a grid-stride stripe to a (value,
// index) pair in pv / pi, and the last block to finish (`done` counts them)
// reduces the blocks' pairs, writes out[0] and puts the counter back to 0 for
// the next call. One block alone took ~110 us over a 152k vocabulary, waiting on
// one load at a time, every decode step; this takes ~9.
__device__ __forceinline__ void argmax_take_(float& v, int& i, float ov, int oi) {
  if (ov > v || (ov == v && oi < i)) {
    v = ov;
    i = oi;
  }
}
__device__ __forceinline__ void argmax_warp_(float& v, int& i) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1)
    argmax_take_(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
}
// The block's pair, in thread 0 (blockDim.x a multiple of 32, at most 1024).
__device__ __forceinline__ void argmax_block_(float& v, int& i) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const unsigned lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  argmax_warp_(v, i);
  if (lane == 0) sv[wid] = v, si[wid] = i;
  __syncthreads();
  if (wid == 0) {
    const bool live = lane < (blockDim.x >> 5);
    v = live ? sv[lane] : -3.402823466e+38f;
    i = live ? si[lane] : 0;
    argmax_warp_(v, i);
  }
}
extern "C" __global__ void tl_argmax(const float* __restrict__ in, int* out, unsigned n,
                                     float* pv, int* pi, unsigned* done) {
  __shared__ bool last;
  float v = -3.402823466e+38f;
  int i = 0;
  // Strict '>' within a thread's ascending stripe: its first maximum.
  for (unsigned k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
    const float x = in[k];
    if (x > v) {
      v = x;
      i = (int)k;
    }
  }
  argmax_block_(v, i);
  if (threadIdx.x == 0) {
    pv[blockIdx.x] = v;
    pi[blockIdx.x] = i;
    __threadfence();  // the pair is visible before the count says so
    last = atomicAdd(done, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  v = -3.402823466e+38f;
  i = 0;
  for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x)
    argmax_take_(v, i, reinterpret_cast<volatile float*>(pv)[b], reinterpret_cast<volatile int*>(pi)[b]);
  argmax_block_(v, i);
  if (threadIdx.x == 0) {
    out[0] = i;
    *done = 0;
  }
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

// ---- index_select: row gather along axis 0. out[i] = a[indices[row(i)]]
// (indices float-valued, rounded here to match argmax's own convention).
// One thread per output element, flat index i decomposed into (row, col)
// by row_size -- no write conflicts, so no pre-zero needed.
extern "C" __global__ void tl_index_select(const float* __restrict__ a,
                                           const float* __restrict__ idx,
                                           float* __restrict__ out,
                                           unsigned row_size, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned row = i / row_size, col = i % row_size;
  long long src_row = (long long)(idx[row] + 0.5f);
  out[i] = a[(size_t)src_row * row_size + col];
}

// ---- index_add: index_select's dual. Scatter-adds `values` into `out` by
// row index -- repeated indices really do collide (real write conflicts),
// so this needs atomicAdd; the caller pre-zeros `out` (cuda.h's index_add)
// since untouched rows must read back as 0.
extern "C" __global__ void tl_index_add(const float* __restrict__ idx,
                                        const float* __restrict__ values,
                                        float* __restrict__ out,
                                        unsigned row_size, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned row = i / row_size, col = i % row_size;
  long long dst_row = (long long)(idx[row] + 0.5f);
  atomicAdd(&out[(size_t)dst_row * row_size + col], values[i]);
}

// ---- scatter_to_axis: one-hot scatter into a new trailing axis of size
// `size`. out[i*size + indices[i]] = values[i], every other slot stays the
// 0 the caller pre-zeroed. Every input position writes a distinct output
// slot (the axis is brand new), so — unlike index_add above — no atomics.
extern "C" __global__ void tl_scatter_axis(const float* __restrict__ idx,
                                           const float* __restrict__ values,
                                           float* __restrict__ out,
                                           unsigned size, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long k = (long long)(idx[i] + 0.5f);
  out[(size_t)i * size + k] = values[i];
}

// ---- gather_from_axis: scatter_to_axis's dual. Takes the one element each
// position labels out of the trailing axis: out[i] = src[i*size + idx[i]].
// One thread per output, no conflicts -- the read-side twin of the write
// above, so cross-entropy can name a row's target logit without building the
// one-hot [rows, size] matrix the scatter would.
extern "C" __global__ void tl_gather_axis(const float* __restrict__ src,
                                          const float* __restrict__ idx,
                                          float* __restrict__ out,
                                          unsigned size, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long k = (long long)(idx[i] + 0.5f);
  out[i] = src[(size_t)i * size + k];
}
// ---- C(i, j) = bias[j]: the row bias laid under a split gemm's partials ----
// One element per thread, a block per 256 columns of a row (grid y = rows): a
// thread per row streaming its row ran 512×4096 in 1 ms on two blocks.
__global__ void tl_fill_rows(float* __restrict__ C, const float* __restrict__ bias,
                             unsigned m, unsigned n) {
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i = blockIdx.y;
  if (i >= m || j >= n) return;
  C[(size_t)i * n + j] = bias[j];
}

// ---- SGEMM: C(m,n) = (A @ B) * scale + offset ----
// lda/ldb are row strides; trans flags let a transposed (col-major) view be
// read in place — same layout contract as metal::gemm / accel::gemm:
//   A(i,k) = trans_a ? A[k*lda + i] : A[i*lda + k]
//   B(k,j) = trans_b ? B[j*ldb + k] : B[k*ldb + j]
// One output element per thread. Correctness-first; stage 2 tiles/shared-mem.
// The one-output-per-thread fallback: its runtime bias test is cheap next to
// the K-long loop (the tiled kernels template it — see sgemm_store_).
__global__ void tl_sgemm(const float* A, const float* B, float* C,
                         const float* bias, unsigned m, unsigned n, unsigned k,
                         unsigned lda, unsigned ldb, unsigned trans_a,
                         unsigned trans_b, float scale, float offset) {
  unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
  float acc = 0.0f;
  for (unsigned p = 0; p < k; p++) {
    float av = trans_a ? A[(size_t)p * lda + i] : A[(size_t)i * lda + p];
    float bv = trans_b ? B[(size_t)j * ldb + p] : B[(size_t)p * ldb + j];
    acc += av * bv;
  }
  const float y = acc * scale + offset;
  C[(size_t)i * n + j] = bias ? y + bias[j] : y;
}

// ---- warp-tiled SGEMM fast path (contiguous operands, any transpose) ----
// C(m,n) = (A(m,k) @ B(k,n)) * scale + offset. The 128×128×8 blocktile is split
// into 64×32 warp tiles (8 warps / 256 threads), each warp iterating WNITER=2
// column sub-tiles so every thread accumulates an 8×8 register microtile (64
// outputs); a 64×64×16 tile with the same structure at half scale serves the
// grids the big tile leaves at a dozen blocks (host picks by fill).
// A is staged transposed (As[BK][BM]) so both fragment reads are a contiguous
// float4 per thread. The warp-tiled thread→output map makes the shared-memory
// fragment reads BROADCAST within a warp (a whole warp shares each As/Bs
// address), which is what removes the bank conflicts a plain tid/16,tid%16 map
// hits (its stride-8 B reads serialize 4-way). This is the step that lifts the
// kernel toward cuBLAS; global loads are float4, one __syncthreads per K-slab.
//
// One core templated on the operand layouts <TA, TB>, so NT (attention's
// Q·Kᵀ) and TN (the backward gemms) run at NN speed instead of falling to
// tl_sgemm (512×1024×512 NT on the RTX 3090: 1128 → ~110 µs). A K-contiguous
// operand (A when !TA, B when TB) is read as a float4 along K and scattered
// into the K-major shared tile; an M/N-contiguous one (A when TA, B when !TB)
// is read as a float4 along M/N and stored straight. The tile, the inner loop
// and the epilogue are shared, so the four instantiations differ only in the
// global loads.
//
// Contract for eligibility (checked host-side in cuda::gemm, else tl_sgemm runs):
// each operand contiguous in its own layout (A: lda==k, or lda==m when TA;
// B: ldb==n, or ldb==k when TB), K%8==0 for the 8-slab, the dim a float4 load
// runs along a multiple of 4 (N for NN's B, M for TN's A — K%8 covers the
// K-contiguous ones), all base offsets folded in and 16B-aligned. M and N
// block edges are predicated (zero-filled loads, guarded stores).
//
// Split-K (ladder ②): S = ceil(k/ksplit) partitions the K axis so S× more
// blocks fill the SMs when the tiling leaves most of them idle (a 1024³
// product is 64 big-tile blocks on 82 SMs). ksplit is the per-split K chunk
// (a multiple of the tile's BK, so slab boundaries stay aligned). When S>1
// each split atomicAdds its partial into a pre-zeroed C: the affine epilogue
// is linear in the partials, so every split adds `acc·scale` and split 0
// alone adds `offset` — a fused `q·kᵀ·scale` splits like a plain GEMM. When
// S==1 (ksplit>=k) the epilogue is the normal fused store, so the non-split
// path is bit-identical to before.
//
// Batching: gridDim.z = batch × S, blockIdx.z / S picks the batch element and
// blockIdx.z % S its K split. sa/sb/sc are the per-batch element strides of
// A/B/C (0 for a broadcast operand), so a batched matmul over [batch,m,k] ·
// [batch,k,n] is one launch instead of `batch` of them; batch == 1 with zero
// strides is the plain GEMM.
}  // extern "C"
// One kernel at two tiles: a BM×BM output per block of 256 threads, BK =
// 1024/BM deep K slabs in shared memory (one float4 per thread per operand
// per slab), 8 warps as 2×4 tiles of (BM/2)×(BM/4), each thread a (BM/16)²
// microtile in two column sub-iterations, slabs streamed into shared memory
// by cp.async through a 4-deep ring (sgemm_cp_core). The 128² tile (BK 8,
// 8×8) is for everything that fills the GPU, the 64² (BK 16, 4×4) for the
// grids a 128² tile leaves at a dozen blocks (the host picks, cuda.h
// sgemm_tile_). Register-staged cores measured below it at both tiles: 0.53-
// 0.83 of cuBLAS against 0.59-0.90 at 128² over the transformer block's
// shapes, and at 64² 9.2k against 11.9k GF/s on an 80-block grid
// (256×1024×1280). (A 256×128 16×8 tile at 215 registers and one block per
// SM, 16-deep slabs, and a 3-deep ring were also tried and measured at or
// below the 128² one.) The pieces — placement, the split-K range, the
// global-load index map, the slab's fragment loads + FMAs, and the epilogue —
// live in the helpers below.

// Where this block and thread sit: block tile, warp tile in the block, thread
// microtile in the warp.
template <int BM>
struct sgemm_geom {
  static constexpr int BK = 1024 / BM;
  static constexpr int WM = BM / 2, WN = BM / 4;  // warp tile: 8 warps as 2 × 4
  static constexpr int TM = BM / 16, TN = BM / 16;  // thread microtile
  static constexpr int WNI = 2, TNSUB = TN / WNI;   // column sub-iterations per warp
  static_assert(BM == 128 || BM == 64, "the two tiles the host knows");
  static_assert((WM / TM) * (WN / WNI / TNSUB) == 32, "32 threads per warp");
  unsigned blockRow, blockCol, warpRow, warpCol, threadRowInWarp, threadColInWarp;
  __device__ __forceinline__ sgemm_geom() {
    blockRow = blockIdx.y * BM;
    blockCol = blockIdx.x * BM;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    warpRow = warp / (BM / WN);
    warpCol = warp % (BM / WN);
    threadRowInWarp = lane / (WN / WNI / TNSUB);
    threadColInWarp = lane % (WN / WNI / TNSUB);
  }
  __device__ __forceinline__ unsigned row0() const {
    return blockRow + warpRow * WM + threadRowInWarp * TM;
  }
  __device__ __forceinline__ unsigned col0(unsigned wn) const {
    return blockCol + warpCol * WN + wn * (WN / WNI) + threadColInWarp * TNSUB;
  }
};

// z = batch element × K split: bi is this block's batch element (the cores
// step A/B/C by it), [k0, k1) its K-range (a multiple of BK; the whole K when
// S==1). sz < S = ceil(k/ksplit) keeps k0 < k, so every split has work.
struct sgemm_split {
  unsigned bi, sz, k0, k1;
  __device__ __forceinline__ sgemm_split(unsigned k, unsigned ksplit) {
    const unsigned S = (k + ksplit - 1) / ksplit;
    bi = blockIdx.z / S;
    sz = blockIdx.z % S;
    k0 = sz * ksplit;
    k1 = (k0 + ksplit < k) ? (k0 + ksplit) : k;
  }
};

// This thread's float4 of each operand per slab. A K-contiguous operand: one
// row (A) / column (B), 4 along K. An M/N-contiguous one: one k, 4 along M/N.
// gArow/gBcol are the M/N index the float4 sits at (the load's guard).
template <bool TA, bool TB, int BM>
struct sgemm_ldmap {
  static constexpr int BK = 1024 / BM;
  unsigned aRow, aCol, bRow, bCol, gArow, gBcol;
  __device__ __forceinline__ sgemm_ldmap(unsigned blockRow, unsigned blockCol) {
    const unsigned tid = threadIdx.x;
    aRow = TA ? tid / (BM / 4) : tid / (BK / 4);        // k | m
    aCol = (TA ? tid % (BM / 4) : tid % (BK / 4)) * 4;  // m | k
    bRow = TB ? tid / (BK / 4) : tid / (BM / 4);        // n | k
    bCol = (TB ? tid % (BK / 4) : tid % (BM / 4)) * 4;  // k | n
    gArow = blockRow + (TA ? aCol : aRow);
    gBcol = blockCol + (TB ? bRow : bCol);
  }
  __device__ __forceinline__ const float* a_src(const float* A, unsigned m, unsigned k,
                                                unsigned kt) const {
    return TA ? &A[(size_t)(kt + aRow) * m + gArow] : &A[(size_t)gArow * k + kt + aCol];
  }
  __device__ __forceinline__ const float* b_src(const float* B, unsigned n, unsigned k,
                                                unsigned kt) const {
    return TB ? &B[(size_t)gBcol * k + kt + bCol] : &B[(size_t)(kt + bRow) * n + gBcol];
  }
};

// Scatter a K-contiguous operand's float4 (4 along k) into the [k][m|n]
// compute layout. 2-way bank conflict inherited from the original: the two
// lanes sharing a row write columns 4 apart.
template <int BM>
__device__ __forceinline__ void sgemm_scatter_(float* dst, unsigned row, unsigned col,
                                               float4 v) {
  dst[(col + 0) * BM + row] = v.x;
  dst[(col + 1) * BM + row] = v.y;
  dst[(col + 2) * BM + row] = v.z;
  dst[(col + 3) * BM + row] = v.w;
}

// One slab's FMAs from compute-layout tiles As/Bs ([BK][BM], flat).
template <int BM>
__device__ __forceinline__ void sgemm_slab_fma_(
    const float* As, const float* Bs, const sgemm_geom<BM>& g,
    float (&acc)[sgemm_geom<BM>::TM][sgemm_geom<BM>::TN]) {
  using G = sgemm_geom<BM>;
  float regM[G::TM];
  float regN[G::TN];
  const unsigned aBase = g.warpRow * G::WM + g.threadRowInWarp * G::TM;
#pragma unroll
  for (unsigned kk = 0; kk < G::BK; kk++) {
#pragma unroll
    for (int q = 0; q < G::TM / 4; q++)
      reinterpret_cast<float4*>(regM)[q] =
          *reinterpret_cast<const float4*>(&As[kk * BM + aBase + 4 * q]);
#pragma unroll
    for (unsigned wn = 0; wn < G::WNI; wn++) {
      const unsigned bBase =
          g.warpCol * G::WN + wn * (G::WN / G::WNI) + g.threadColInWarp * G::TNSUB;
      if constexpr (G::TNSUB == 4)
        reinterpret_cast<float4*>(regN)[wn] =
            *reinterpret_cast<const float4*>(&Bs[kk * BM + bBase]);
      else
        reinterpret_cast<float2*>(regN)[wn] =
            *reinterpret_cast<const float2*>(&Bs[kk * BM + bBase]);
    }
#pragma unroll
    for (unsigned i = 0; i < G::TM; i++)
#pragma unroll
      for (unsigned j = 0; j < G::TN; j++) acc[i][j] += regM[i] * regN[j];
  }
}

// Epilogue: guarded store (mirroring the load map's edges). S==1: fused
// affine store, then the row bias (BIAS). S>1: atomicAdd this split's share of
// the result into a pre-set C — scale on every partial, offset from split 0
// only; a fused bias is already in C (the host lays the bias rows down with
// tl_fill_rows in place of the zeroing), so a split never reads it here. ksplit
// is uniform across the block, so the branch never diverges. BIAS is a template
// parameter so the plain kernels compile without it: a runtime bias test in
// this store cost 512×1024×4096 14% unbiased, and a bias read in the split
// store 16% biased.
template <int BM, bool BIAS>
__device__ __forceinline__ void sgemm_store_(
    float* __restrict__ C, const float* __restrict__ bias, unsigned m,
    unsigned n, float scale, float offset, bool split, unsigned sz,
    const sgemm_geom<BM>& g,
    const float (&acc)[sgemm_geom<BM>::TM][sgemm_geom<BM>::TN]) {
  using G = sgemm_geom<BM>;
  const float part_offset = sz == 0 ? offset : 0.0f;
#pragma unroll
  for (unsigned i = 0; i < G::TM; i++) {
    const unsigned gRow = g.row0() + i;
    if (gRow >= m) continue;
#pragma unroll
    for (unsigned wn = 0; wn < G::WNI; wn++) {
#pragma unroll
      for (unsigned js = 0; js < G::TNSUB; js++) {
        const unsigned gCol = g.col0(wn) + js;
        if (gCol >= n) continue;
        const size_t idx = (size_t)gRow * n + gCol;
        const float v = acc[i][wn * G::TNSUB + js];
        if (split)
          atomicAdd(&C[idx], v * scale + part_offset);
        else if constexpr (BIAS)
          C[idx] = v * scale + offset + bias[gCol];
        else
          C[idx] = v * scale + offset;
      }
    }
  }
}

// ---- cp.async-pipelined core ----
// Global slabs stream straight into a STAGES-deep shared ring with cp.async
// (no registers in the path, STAGES-1 slabs in flight), and an operand that
// arrives K-contiguous is turned to the [k][m] compute layout by a
// shared-to-shared step (float4 in, four scalars out) into a double-buffered
// compute tile — transposing on the way in, through registers, would tie the
// prefetch depth to registers. The transpose runs one slab ahead of the
// compute, so a slab costs one barrier.
__device__ __forceinline__ void cp_async16_(void* smem, const void* gmem, bool valid) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int bytes = valid ? 16 : 0;  // src-size 0: zero-fill, gmem still a valid address
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
               "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit_() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool TA, bool TB, int BM, bool BIAS>
__device__ __forceinline__ void sgemm_cp_core(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const float* __restrict__ bias,
                                              unsigned m,
                                              unsigned n, unsigned k, float scale,
                                              float offset, unsigned ksplit,
                                              unsigned sa, unsigned sb,
                                              unsigned sc) {
  using G = sgemm_geom<BM>;
  constexpr int BK = G::BK;
  // Ring depth: at least 3, so slab i+1 has landed while slab i computes (a
  // 3-deep ring measured at or below this one).
  constexpr int STAGES = 4;
  // Ring stages hold an M/N-contiguous operand in compute layout [k][m|n]
  // and a K-contiguous one raw as [m|n][k]; the raw ones get a double-
  // buffered compute-layout tile of their own.
  __shared__ __align__(16) float ringA[STAGES][BK * BM];
  __shared__ __align__(16) float ringB[STAGES][BK * BM];
  __shared__ __align__(16) float compA[TA ? 4 : 2 * BK * BM];
  __shared__ __align__(16) float compB[TB ? 2 * BK * BM : 4];
  const G g;
  const sgemm_split sp(k, ksplit);
  A += (size_t)sp.bi * sa;
  B += (size_t)sp.bi * sb;
  C += (size_t)sp.bi * sc;
  const sgemm_ldmap<TA, TB, BM> ld(g.blockRow, g.blockCol);
  const unsigned nslabs = (sp.k1 - sp.k0) / BK;

  // Issue slab `slab` (k offset k0 + slab·BK) into ring stage `st`.
  auto issue = [&](unsigned slab, int st) {
    const unsigned kt = sp.k0 + slab * BK;
    const bool a_ok = ld.gArow < m, b_ok = ld.gBcol < n;
    cp_async16_(&ringA[st][TA ? ld.aRow * BM + ld.aCol : ld.aRow * BK + ld.aCol],
                a_ok ? ld.a_src(A, m, k, kt) : A, a_ok);
    cp_async16_(&ringB[st][TB ? ld.bRow * BK + ld.bCol : ld.bRow * BM + ld.bCol],
                b_ok ? ld.b_src(B, n, k, kt) : B, b_ok);
  };
  // Turn the raw [m|n][k] stage into compute-layout buffer `cb`.
  auto transpose = [&](int st, int cb) {
    if constexpr (!TA)
      sgemm_scatter_<BM>(compA + cb * (BK * BM), ld.aRow, ld.aCol,
                         *reinterpret_cast<const float4*>(&ringA[st][ld.aRow * BK + ld.aCol]));
    if constexpr (TB)
      sgemm_scatter_<BM>(compB + cb * (BK * BM), ld.bRow, ld.bCol,
                         *reinterpret_cast<const float4*>(&ringB[st][ld.bRow * BK + ld.bCol]));
  };

  float acc[G::TM][G::TN] = {};
  // Prologue: the first STAGES-1 slabs in flight (one commit group each, empty
  // groups past the end keep the wait arithmetic uniform), slab 0 transposed.
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if ((unsigned)s < nslabs) issue(s, s);
    cp_async_commit_();
  }
  cp_async_wait_<STAGES - 2>();  // slab 0 landed
  __syncthreads();
  transpose(0, 0);
  // Each iteration: slab i+1 has landed → one barrier → issue slab i+STAGES-1
  // (into the stage slab i-1 used, consumed before this barrier), transpose
  // slab i+1 into the other compute buffer, compute slab i. Compute and the
  // next slab's transpose share the phase, so one barrier per slab.
  for (unsigned i = 0; i < nslabs; i++) {
    cp_async_wait_<STAGES - 3>();  // slab i+1 landed (this thread's copies)
    __syncthreads();               // ...everyone's; and slab i-1 fully consumed
    const unsigned nx = i + STAGES - 1;
    if (nx < nslabs) issue(nx, nx % STAGES);
    cp_async_commit_();
    if (i + 1 < nslabs) transpose((i + 1) % STAGES, (i + 1) & 1);
    const int st = i % STAGES;
    sgemm_slab_fma_<BM>(TA ? ringA[st] : compA + (i & 1) * (BK * BM),
                        TB ? compB + (i & 1) * (BK * BM) : ringB[st], g, acc);
  }
  cp_async_wait_<0>();  // drain the trailing empty groups
  sgemm_store_<BM, BIAS>(C, bias, m, n, scale, offset, ksplit < k, sp.sz, g, acc);
}

// One __global__ per operand layout per tile, plain and with a fused row bias
// (extern "C" for a stable symbol the host looks up by name — cuda.h
// context::sgemm_).
#define TL_SGEMM_KERNEL(NAME, TA, TB, BM, BIAS)                                  \
  TL_ALIGN(NAME, 16, 16) /* A and B by cp.async's 16 bytes */                  \
  extern "C" __global__ void NAME(                                             \
      const float* __restrict__ A, const float* __restrict__ B,                \
      float* __restrict__ C, const float* __restrict__ bias, unsigned m,       \
      unsigned n, unsigned k, float scale, float offset, unsigned ksplit,      \
      unsigned sa, unsigned sb, unsigned sc) {                                 \
    sgemm_cp_core<TA, TB, BM, BIAS>(A, B, C, bias, m, n, k, scale, offset,     \
                                    ksplit, sa, sb, sc);                       \
  }
#define TL_SGEMM_LAYOUTS(NAME, BM, BIAS)           \
  TL_SGEMM_KERNEL(NAME, false, false, BM, BIAS)    \
  TL_SGEMM_KERNEL(NAME##_nt, false, true, BM, BIAS) \
  TL_SGEMM_KERNEL(NAME##_tn, true, false, BM, BIAS) \
  TL_SGEMM_KERNEL(NAME##_tt, true, true, BM, BIAS)
TL_SGEMM_LAYOUTS(tl_sgemm_cp64, 64, false)
TL_SGEMM_LAYOUTS(tl_sgemm_cp128, 128, false)
TL_SGEMM_LAYOUTS(tl_sgemm_cp64_bias, 64, true)
TL_SGEMM_LAYOUTS(tl_sgemm_cp128_bias, 128, true)
#undef TL_SGEMM_LAYOUTS
#undef TL_SGEMM_KERNEL
