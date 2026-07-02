// MSL kernel source, #embed'd into metal.h and JIT-compiled on first GPU
// use. Editing this file changes nothing until the host binary is rebuilt.
//
// Every kernel applies the graph's affine epilogue (out = op * scale +
// offset) so fused scalar chains cost zero extra dispatches. The host
// dispatches ceil(n/256) threadgroups; the `i >= p.n` bound uses the same
// ceiling arithmetic (shared-template rule from silarray's edge-tile bugs:
// op bodies live in one macro, never copy-pasted per variant).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ew_params {
  float scale;
  float offset;
  uint n;
};

#define EW_BINARY(name, expr)                                  \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device const float* b [[buffer(1)]],        \
                   device float* out [[buffer(2)]],            \
                   constant ew_params& p [[buffer(3)]],        \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    out[i] = fma(expr, p.scale, p.offset);                     \
  }

EW_BINARY(add_, a[i] + b[i])
EW_BINARY(sub_, a[i] - b[i])
EW_BINARY(mul_, a[i] * b[i])
EW_BINARY(div_, a[i] / b[i])

#define EW_UNARY(name, expr)                                   \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device float* out [[buffer(1)]],            \
                   constant ew_params& p [[buffer(2)]],        \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    out[i] = fma(expr, p.scale, p.offset);                     \
  }

EW_UNARY(exp_, exp(a[i]))
EW_UNARY(log_, log(a[i]))
EW_UNARY(sqrt_, sqrt(a[i]))
EW_UNARY(sigmoid_, 1.0f / (1.0f + exp(-a[i])))
EW_UNARY(relu_, max(a[i], 0.0f))
EW_UNARY(affine_, a[i])

// ---------------------------------------------------------------------------
// Tiled SGEMM — 32×32×16, simdgroup_matrix 8×8 MMA, float4 vectorized loads.
// trans_a/trans_b let strided/transposed operands be read in place (silarray:
// materializing a transpose gives back the win). The affine epilogue
// (out = A@B * scale + offset) is fused into the store so a fused
// dot-then-scalar chain is one dispatch. Edge tiles bounce the accumulator
// through threadgroup scratch — one store path, ceiling-rounded bounds check
// (silarray edge-tile bug class).
// ---------------------------------------------------------------------------

struct gemm_params {
  uint M, N, K, lda, ldb, trans_a, trans_b;
  float scale, offset;
};

kernel void sgemm_(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C       [[buffer(2)]],
                   constant gemm_params& p [[buffer(3)]],
                   uint3 tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_index_in_threadgroup]],
                   uint sid  [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM = 32, BN = 32, BK = 16;
  constexpr uint N_SM = 2, N_SN = 2, TM = BM / N_SM, TN = BN / N_SN;
  constexpr uint FM = TM / 8, FN = TN / 8, THREADS = N_SM * N_SN * 32;
  constexpr uint aS = BK + 4, bS = BN + 4;  // float4-aligned padding

  threadgroup float As[BM * aS];
  threadgroup float Bs[BK * bS];

  uint wm = sid / N_SN, wn = sid % N_SN;
  simdgroup_matrix<float, 8, 8> acc[FM][FN];
  for (uint i = 0; i < FM; i++)
    for (uint j = 0; j < FN; j++)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

  uint row0 = tgid.y * BM, col0 = tgid.x * BN;
  uint a_rs = p.trans_a ? 1u : p.lda, a_cs = p.trans_a ? p.lda : 1u;
  uint b_rs = p.trans_b ? 1u : p.ldb, b_cs = p.trans_b ? p.ldb : 1u;

  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    for (uint i = tid; i < BM * BK; i += THREADS) {
      uint r = i / BK, c = i % BK, gr = row0 + r, gc = k0 + c;
      As[r * aS + c] = (gr < p.M && gc < p.K) ? A[gr * a_rs + gc * a_cs] : 0.0f;
    }
    for (uint i = tid; i < BK * BN; i += THREADS) {
      uint r = i / BN, c = i % BN, gr = k0 + r, gc = col0 + c;
      Bs[r * bS + c] = (gr < p.K && gc < p.N) ? B[gr * b_rs + gc * b_cs] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<float, 8, 8> af[FM], bf[FN];
      for (uint i = 0; i < FM; i++)
        simdgroup_load(af[i], &As[(wm * TM + i * 8) * aS + kk], aS);
      for (uint j = 0; j < FN; j++)
        simdgroup_load(bf[j], &Bs[kk * bS + wn * TN + j * 8], bS);
      for (uint i = 0; i < FM; i++)
        for (uint j = 0; j < FN; j++)
          simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store: bounce every fragment through As scratch for scalar epilogue +
  // edge masking (one path, no aligned/edge divergence).
  float scale = p.scale, offset = p.offset;
  for (uint i = 0; i < FM; i++) {
    for (uint j = 0; j < FN; j++) {
      uint r = row0 + wm * TM + i * 8, c = col0 + wn * TN + j * 8;
      threadgroup float* sc = As;
      simdgroup_store(acc[i][j], &sc[sid * 64], 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        uint er = r + e / 8, ec = c + e % 8;
        if (er < p.M && ec < p.N)
          C[er * p.N + ec] = sc[sid * 64 + e] * scale + offset;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// ---------------------------------------------------------------------------
// Row reductions over the last axis: one threadgroup per row, 256 threads,
// threadgroup-scratch tree reduction. `cols` may exceed the thread count
// (grid-stride accumulate first). softmax_ is numerically stable (subtract
// row max) and applies the affine epilogue to the input pre-softmax is not
// meaningful, so scale/offset are ignored for softmax.
// ---------------------------------------------------------------------------

struct reduce_params {
  uint rows, cols;
  float scale, offset;
};

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

#define ROW_REDUCE(name, init, combine, finish)                              \
  kernel void name(device const float* in [[buffer(0)]],                     \
                   device float* out       [[buffer(1)]],                    \
                   constant reduce_params& p [[buffer(2)]],                  \
                   uint row [[threadgroup_position_in_grid]],                \
                   uint lid [[thread_index_in_threadgroup]]) {               \
    constexpr uint T = 256;                                                  \
    threadgroup float scratch[T];                                            \
    device const float* src = in + row * p.cols;                            \
    float acc = init;                                                        \
    for (uint c = lid; c < p.cols; c += T) { float v = src[c]; acc = combine; } \
    scratch[lid] = acc;                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint s = T / 2; s > 0; s >>= 1) {                                   \
      if (lid < s) { float v = scratch[lid + s]; acc = scratch[lid];        \
                     scratch[lid] = combine; }                              \
      threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                        \
    if (lid == 0) { float acc_final = scratch[0];                           \
                    out[row] = (finish) * p.scale + p.offset; }             \
  }

ROW_REDUCE(row_sum_, 0.0f, acc + v, acc_final)
ROW_REDUCE(row_max_, -INFINITY, max(acc, v), acc_final)
