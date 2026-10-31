#pragma once

// The CUDA backend's nn layer (layers: gpu_abi.h): its own ops for the fused ops over plain
// tensors (the GEMVs and GEMMs over bf16, f16 and block-quantized weights, the
// f16 activation, attention, RoPE, DeltaNet, layer norm's pullback) and their
// launch helpers.
//
// Included by cuda.h, after the device core.

#include "gpu_abi_nn.h"

namespace tl {
namespace cuda {

struct own_nn {
  static bool split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                          int64_t T, int64_t ld, int64_t off, int64_t H,
                          int64_t D);
  // x[rows, D] RMS-normalized by w and then rotated, in place (rows = heads * T); D 64,
  // 128 or 256 and rot a multiple of 64, else false.
  static bool rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows, int64_t T, int64_t D,
                           int64_t pos, float base, int64_t rot, float eps);
  // The activation a[K] quantized to int8 blocks of 32 (K int8, then K/32 float2 of
  // scale and sum) into qa, and Q4_K rows dotted with it (dp4a): the Q4_K GEMV
  // with the activation's bytes cut four-fold. K % 256 == 0.
  static bool quantize_act_q8(gpu::span a, gpu::span qa, int64_t k);
  static bool gemv_q4k_q8(gpu::span qa, gpu::span W, gpu::span y, int64_t n, int64_t k);
  // C[m, ldc-strided n columns] = A[m,K] . W[n,K]^T over GGUF block-quantized rows (type 12
  // Q4_K, 14 Q6_K, 8 Q8_0). a16: A is f16 and the sums f16, else A f32, the operands
  // rounded to bf16 and the sums f32 (the exact form: no f16 range to overflow). K % 256 == 0
  // (Q8_0: % 32), n even. mmq (with a16, Q4_K, or Q8_0 with
  // K % 256 == 0):
  // A is quantized to int8 and the product runs on the int8 tensor cores, nothing
  // dequantized (tl_gemm_mmq_q4k, _q80): faster, and the activation rounded to 8 bits.
  static bool gemm_quant(int type, bool a16, gpu::span A, gpu::span W, gpu::span C, int64_t m,
                          int64_t n, int64_t k, int64_t ldc, bool mmq = false);
  // H16[m, ff] = silu(A16 . Wg^T) * (A16 . Wu^T) in f16, W the fused [Wg | Wu] (2 ff rows of
  // Q4_K or Q8_0, K % 256 == 0) on the int8 tensor cores, as gemm_quant(mmq) then
  // swiglu_h16 would make it. False where it does not apply (the caller does those two).
  static bool gemm_quant_glu(int type, gpu::span A16, gpu::span W, gpu::span H16, int64_t m,
                             int64_t ff, int64_t k);
  // y[n] = W[n,K] . a over GGUF block-quantized rows (type: the ggml_type code, 12 Q4_K
  // 14 Q6_K or 8 Q8_0), the blocks read where they lie. K a whole number of blocks.
  static bool gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y, int64_t n, int64_t k);
  // The f16 prefill GEMM reading an activation that is already f16, and the
  // producers that round it where it is made (see their definitions).
  static bool gemm_bf16_nt_f16acc_a16(gpu::span a, gpu::span B, gpu::span C,
                                      int64_t M, int64_t N, int64_t K);
  static bool rmsnorm_h16(gpu::span x, gpu::span w, gpu::span o, gpu::span o16,
                          int64_t n, float eps, int64_t rows);
  static bool rmsnorm_res_h16(gpu::span x, gpu::span delta, gpu::span w,
                              gpu::span xout, gpu::span hout, gpu::span hout16,
                              int64_t n, float eps, int64_t rows);
  static bool swiglu_h16(gpu::span gu, gpu::span o16, int64_t ff, int64_t rows);
  // out = silu(z) * rmsnorm(o) * w over T x H rows of D (tl_gated_rmsnorm): o contiguous,
  // z's row t ldz apart, out f32 or (h16) f16 [T, H D]; bit for bit rmsnorm then swiglu.
  static bool gated_rmsnorm(gpu::span o, gpu::span z, int64_t ldz, gpu::span w, gpu::span out,
                            bool h16, int64_t T, int64_t H, int64_t D, float eps);
  static bool merge_heads_h16(gpu::span src, gpu::span dst16, int64_t T, int64_t H,
                              int64_t D);
  static bool cvt_h16(gpu::span src, gpu::span dst16, int64_t n);
  static bool gemv_f32(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                       int64_t k);
  static bool gemv_bf16(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                        int64_t k);
  static bool attn_decode(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                          int64_t n_q_heads, int64_t n_kv_heads, int64_t ctx,
                          int64_t kv_max, int64_t D, float scale,
                          bool kv_f16 = false);
  static bool attn_prefill(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                           int64_t n_q_heads, int64_t n_kv_heads, int64_t T,
                           int64_t kv_max, int64_t D, float scale,
                           bool kv_f16 = false, int64_t pos0 = 0);
  static bool attn_prefill_f16(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                               int64_t n_q_heads, int64_t n_kv_heads, int64_t T,
                               int64_t kv_max, int64_t D, float scale,
                               bool kv_f16 = false, int64_t pos0 = 0);
  // Gated DeltaNet's conv over a prefill's T tokens, a thread a (channel, token).
  static bool ssm_conv_tokens(gpu::span x, int64_t ld, gpu::span state, gpu::span w,
                              gpu::span o, int64_t C, int64_t K, int64_t T);
  // The gated delta rule over a prefill's T tokens, in chunks (gpu::gated_delta).
  static bool gated_delta_chunked(gpu::span qkv, gpu::span ab, int64_t ab_ld,
                                  gpu::span adt, gpu::span state, gpu::span o,
                                  int64_t H, int64_t dk, int64_t dv, int64_t T,
                                  float eps, int64_t Hk);
  static bool attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span dO, gpu::span O, gpu::span dq,
                              gpu::span stats, int64_t H, int64_t T, int64_t D,
                              float scale);
  static bool attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                               gpu::span dO, gpu::span stats, gpu::span dK,
                               gpu::span dV, int64_t H, int64_t T, int64_t D,
                               float scale);
  static bool gemm_bf16_nt(gpu::span a, gpu::span B, gpu::span out, int64_t m,
                           int64_t n, int64_t k);
  static bool gemm_bf16_nt_f16acc(gpu::span a, gpu::span B, gpu::span out, int64_t m,
                                  int64_t n, int64_t k);
  static bool rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                   int64_t D, int64_t pos, float base, gpu::span bias = {});
  // rope over the first `rot` dims of each row only (Qwen3.5's partial RoPE).
  static bool rope_partial(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                           int64_t D, int64_t pos, float base, gpu::span bias,
                           int64_t rot);
  static bool layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                             gpu::span dx, gpu::span dg, gpu::span db,
                             gpu::span stats, gpu::span partials, int64_t rows,
                             int64_t cols, int64_t per_chunk, int64_t chunks,
                             float eps);
};

// "No bias" is a null pointer here, which the kernel tests.
inline bool own_nn::split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                        int64_t T, int64_t ld, int64_t off, int64_t H,
                        int64_t D) {
  struct {
    uint32_t T, ld, off, D;
  } p{static_cast<uint32_t>(T), static_cast<uint32_t>(ld),
      static_cast<uint32_t>(off), static_cast<uint32_t>(D)};
  const gpu::arg args[] = {gpu::in(src), gpu::in(bias), gpu::out(dst)};
  return dispatch(kop::split_heads_, args, 3, &p, sizeof(p),
                  gpu::policy::per_head(H, T, D));
}

// M7 decode GEMV: y(n) = a(1,k) @ B(k,n), F32 accumulate. B is either f32 or
// bf16 weights (bf16 halves the dominant K×N weight traffic — the decode
// bandwidth lever). Buffers are opaque device pointers; the kernel interprets
// B's dtype. Offset 0 (contiguous weight/activation operands). Separate from
// gemm() so the M=1 path skips the 128×128 tile that wastes 127 rows.
//
// Split-K when the N/256 column-blocks alone underfill the SMs (small-N layers):
// partition K over gridDim.y, atomicAdd into a pre-zeroed y, so the kernel stays
// bandwidth-bound rather than occupancy-bound. gridDim.y==1 stores directly.
// The kernel's rule: no split under K=512, and a part is a multiple of its
// 32-wide K step.
inline bool gemv_run_(CUfunction f, void* pa, float* pB, float* py,
                      unsigned un, unsigned uk, unsigned vcols = 1) {
  auto& c = context::get();
  unsigned per = 256u * vcols;  // output columns covered by one block
  unsigned bx = (un + per - 1) / per;
  if (bx == 0) bx = 1;
  unsigned gy = 1, ksplit = uk;
  constexpr gpu::policy::split_rule rule{traits::fill_groups, 512, 0, 32};
  const int64_t parts =
      c.no_splitk ? 1 : gpu::policy::split_parts(bx, uk, rule);
  if (parts > 1) {
    ksplit = static_cast<unsigned>(gpu::policy::split_chunk(uk, parts, rule));
    gy = (uk + ksplit - 1) / ksplit;
  }
  if (gy > 1) {
    // Zero y for the split-K atomicAdd. Async on the stream (ordered before the
    // gemv on the same stream) so this stays capturable — a blocking MemsetD8
    // is illegal mid CUDA-graph capture.
    CUdeviceptr yd = reinterpret_cast<CUdeviceptr>(py);
    if (c.d.MemsetD8Async) c.d.MemsetD8Async(yd, 0, (size_t)un * 4, c.stream);
    else c.d.MemsetD8(yd, 0, (size_t)un * 4);
  }
  return c.launch_(f, {bx, gy}, {256}, 0, pa, pB, py, un, uk, ksplit);
}
inline bool own_nn::gemv_f32(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                          int64_t k) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_read_(B.buf);
  c.device_write_(y.buf);
  return gemv_run_(c.gemv_f32_(), context::off_(a.buf, a.off),
                   context::off_(B.buf, B.off), context::off_(y.buf, y.off),
                   static_cast<unsigned>(n), static_cast<unsigned>(k));
}
inline bool own_nn::gemv_bf16(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                           int64_t k) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_read_(a.buf);
  c.device_read_(B.buf);  // B reinterpreted as __nv_bfloat16* in-kernel
  c.device_write_(y.buf);
  // Vectorized 8-cols/thread path when n%8==0 (all transformer dims) and B fits
  // its 16-byte loads — they close the bandwidth gap to f32; scalar otherwise.
  float* pa = context::off_(a.buf, a.off);
  float* pB = context::off_(B.buf, B.off);
  CUfunction f = c.gemv_bf16v8_();
  const bool v8 = n % 8 == 0 && c.fits_(f, pa, pB);
  if (!v8) f = c.gemv_bf16_();
  return gemv_run_(f, pa, pB, context::off_(y.buf, y.off), static_cast<unsigned>(n),
                   static_cast<unsigned>(k), v8 ? 8u : 1u);
}

// Block size (32..256 threads) for the one-block-per-row GEMVs (tl_gemv_bf16_row,
// tl_gemv_q4), chosen per K to minimize the per-thread iteration count (llama.cpp's
// mul_mat_vec_f strategy) — a wide row (large K) gets more warps collaborating on
// the reduction. Ties keep the smaller block size (loop only replaces on strictly
// fewer iters). Always a multiple of 32 and <=256, as both kernels require.
inline unsigned gemv_row_block_size(int64_t k) {
  unsigned best = 32;
  int64_t niter_best = INT64_MAX;
  for (unsigned bs = 32; bs <= 256; bs += 32) {
    int64_t niter = (k + 8 * bs - 1) / (8 * bs);
    if (niter < niter_best) { niter_best = niter; best = bs; }
  }
  return best;
}
// Companion dynamic-shared size for a gemv_row_block_size block: one float per
// warp for the cross-warp reduce, or 0 for a single-warp block (which never
// touches shared). Kept beside the block-size policy so the two can't drift.
inline unsigned gemv_row_smem(unsigned block) {
  return block > 32 ? (block >> 5) * (unsigned)sizeof(float) : 0u;
}

// The tensor-core GEMM's launch: the widest tile whose grid still keeps most of
// the SMs busy. A prefill chunk is a few hundred rows against a projection that
// is often narrow (wo, down: N ~ 1K), so the 128x128 tile is 32 blocks of work
// there; the 64x128 doubles that and the 64x64 doubles it again. "Busy" is
// two thirds of a block per SM (a third of fill_groups, which counts two):
// the tiles share an SM two at a time, and the threshold was swept at M=512.
struct tc_plan_ {
  CUfunction f;
  unsigned gx, gy, threads, smem;
};
inline tc_plan_ gemm_tc_plan_(context& c, context::tc_form form, unsigned m, unsigned n,
                              unsigned k, context::tc_weight w = context::tc_weight::bf16) {
  using tile = context::tc_tile;
  constexpr int64_t busy = traits::fill_groups / 3;
  auto grid = [&](tile t) {
    const auto sh = context::gemm_tc_shape(t);
    return std::pair<unsigned, unsigned>{(n + sh.cols - 1) / sh.cols, (m + sh.rows - 1) / sh.rows};
  };
  auto blocks = [&](tile t) { return (int64_t)grid(t).first * grid(t).second; };
  // The f16 weight's long-K tile (a big model's projections): one block a SM, so it
  // wants a grid that is nearly whole waves (the 6144 x 5120 at 96 blocks is 59% of
  // two waves and loses to the 128-row tile), and K = 1024 shapes lose to the others.
  constexpr int64_t sms = traits::fill_groups / 2;
  const int64_t bx = blocks(tile::xl), waves = (bx + sms - 1) / sms;
  const bool xl = w == context::tc_weight::f16w && form == context::tc_form::f16h && k >= 3072 &&
                  bx >= sms - 2 && bx * 4 >= waves * sms * 3;
  const tile t = xl                        ? tile::xl
                 : blocks(tile::l) >= busy ? tile::l
                 : blocks(tile::m) >= busy ? tile::m
                                           : tile::s;
  const auto sh = context::gemm_tc_shape(t);
  return {c.gemm_tc_(form, t, w), grid(t).first, grid(t).second, sh.threads, sh.smem()};
}
inline bool gemm_tc_launch_(context& c, const tc_plan_& p, void* pa, void* pB, float* po,
                            unsigned m, unsigned n, unsigned k, unsigned ldc) {
  return c.launch_(p.f, {p.gx, p.gy}, {p.threads}, p.smem, pa, pB, po, m, n, k, ldc);
}
inline bool gemm_tc_run_(context& c, context::tc_form form, void* pa, void* pB, float* po,
                         unsigned m, unsigned n, unsigned k, unsigned ldc,
                         context::tc_weight w = context::tc_weight::bf16) {
  return gemm_tc_launch_(c, gemm_tc_plan_(c, form, m, n, k, w), pa, pB, po, m, n, k, ldc);
}

// M9 batched-prefill GEMM: C(M,N) = A(M,K) @ W[N,K]^T, W the same row-major
// bf16 weight gemv_bf16_row consumes — so a batched prompt reuses the decode
// weights as-is. Any K and alignment (see tl_gemm_bf16_nt).
inline bool own_nn::gemm_bf16_nt(gpu::span a, gpu::span B, gpu::span out,
                              int64_t m, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready || m <= 0 || n <= 0 || k <= 0) return false;
  c.device_read_(a.buf);
  c.device_read_(B.buf);  // B reinterpreted as __nv_bfloat16* [N][K] in-kernel
  c.device_write_(out.buf);
  float* pa = context::off_(a.buf, a.off);
  float* pB = context::off_(B.buf, B.off);
  float* po = context::off_(out.buf, out.off);
  unsigned uM = (unsigned)m, uN = (unsigned)n, uK = (unsigned)k;
  // The tensor cores read 16-byte vectors of both operands' rows (K % 8 == 0)
  // and store two columns at once (N even), from bases their kernel declares;
  // the rest runs on the SIMT tiles below. The plan is asked first, not the
  // launch: a launch's false is also a driver failure, which no fallback fixes.
  if (k % 8 == 0 && n % 2 == 0) {
    const tc_plan_ tc = gemm_tc_plan_(c, context::tc_form::f32, uM, uN, uK);
    if (c.fits_(tc.f, pa, pB, po)) return gemm_tc_launch_(c, tc, pa, pB, po, uM, uN, uK, uN);
  }
  // Tile choice by fill (big_tile_): a prefill chunk is only a few hundred
  // tokens, so a narrow projection (N=896, M=512) is 28 big blocks against 82
  // SMs and 112 small ones.
  bool big = big_tile_(m, n);
  unsigned t = big ? 128u : 64u;
  unsigned gx = (unsigned)((n + t - 1) / t), gy = (unsigned)((m + t - 1) / t);
  unsigned blocks = gx * gy;

  // Even on the small tile a narrow projection is only ~1.4 waves of the 82 SMs,
  // so the tail runs half-empty; slicing K over gridDim.z fills it. Two bounds
  // decide how far: keep each slice long enough to amortize its own staging
  // (>= 448 K-elements), and stop once the grid is comfortably several waves
  // (~8 blocks/SM). Measured at M=512: wd 506 -> 312 us, wo 92 -> 75, while a
  // grid that already fills (gateup, 1216 blocks) correctly declines to split.
  // As a rule: four times the fill, slices in the 64 tile's 16-deep slabs.
  constexpr gpu::policy::split_rule rule{4 * traits::fill_groups, 0, 448, 16};
  unsigned z = 1;
  if (blocks < 128 && c.d.MemsetD8Async) {
    z = (unsigned)gpu::policy::split_parts(blocks, k, rule);
  }
  if (z > 1) {
    unsigned ksplit = (unsigned)gpu::policy::split_chunk(k, z, rule);
    z = (uK + ksplit - 1) / ksplit;  // recount after rounding
    // atomicAdd combine needs a zeroed C; async on the stream, so it is ordered
    // before the launch without a host sync (and stays capturable).
    c.d.MemsetD8Async(reinterpret_cast<CUdeviceptr>(po), 0,
                      (size_t)m * n * sizeof(float), c.stream);
    return c.launch_(c.gemm_bf16_nt_sk_(), {gx, gy, z}, {256}, 0, pa, pB, po,
                     uM, uN, uK, ksplit);
  }
  return c.launch_(c.gemm_bf16_nt_(big), {gx, gy}, {256}, 0, pa, pB, po, uM, uN,
                   uK);
}

// The prefill GEMM summed in f16 (tl_gemm_tc_f16*, the tensor-core tiles'
// f16 form), in both its forms: K % 8 == 0, N even and operands where the
// kernel declares them, else false (the caller's f32 GEMM, or for an f16
// activation nothing: it must be sized for these, see prefill_chunk_).
inline bool f16acc_run_(context::tc_form form, gpu::span a, gpu::span B, gpu::span out,
                        int64_t m, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready || (k % 8) != 0 || m <= 0 || n <= 0 || n % 2) return false;
  float* pa = context::off_(a.buf, a.off);
  float* pB = context::off_(B.buf, B.off);
  float* po = context::off_(out.buf, out.off);
  c.device_read_(a.buf);
  c.device_read_(B.buf);
  c.device_write_(out.buf);
  return gemm_tc_run_(c, form, pa, pB, po, (unsigned)m, (unsigned)n, (unsigned)k, (unsigned)n);
}
inline bool own_nn::gemm_bf16_nt_f16acc(gpu::span a, gpu::span B, gpu::span out, int64_t m,
                                     int64_t n, int64_t k) {
  return f16acc_run_(context::tc_form::f16, a, B, out, m, n, k);
}
// The same with the activation already f16 (rounded by the producer that wrote
// it: own::rmsnorm_h16 and its kin), which `a` holds as 2-byte elements.
inline bool own_nn::gemm_bf16_nt_f16acc_a16(gpu::span a, gpu::span B, gpu::span out, int64_t m,
                                         int64_t n, int64_t k) {
  return f16acc_run_(context::tc_form::f16h, a, B, out, m, n, k);
}

// The MMQ GEMM (tl_gemm_mmq_*): A (f16, M x K) rounded to int8 into a scratch, then the
// product with the Q4_K / Q8_0 rows at W into out (f32 C columns, or with glu f16
// silu(gate) * up of a fused gate|up W, n the outputs). False where glu would need K
// split (the caller then takes the unfused path).
inline bool mmq_run_(context& c, int type, bool glu, void* pa, void* pw, void* out, unsigned uM,
                     unsigned uN, unsigned uK, unsigned ldc) {
  constexpr unsigned tile = context::mmq_tile;
  // A tile a block and one block an SM: a grid short of the SMs (a narrow projection,
  // N ~ 1K) leaves some idle, so K is split into as many parts as still fit in one
  // wave (rounding down: a part past it would start a second, nearly empty wave), each
  // at least two super-blocks. The parts' sums go through a scratch and are added in order.
  const unsigned mpad = (uM + tile - 1) / tile * tile, nk = uK / 256;
  // Weight tiles tn, activation tiles tm (the grid's y and x: tl_gemm_mmq_*'s order).
  const unsigned tn = glu ? (uN + tile / 2 - 1) / (tile / 2) : (uN + tile - 1) / tile, tm = mpad / tile;
  constexpr unsigned sms = traits::fill_groups / 2;
  const unsigned parts = std::max(1u, std::min(sms / (tn * tm), nk / 2));
  if (glu && parts > 1) return false;
  const unsigned kchunk = (nk + parts - 1) / parts;
  const unsigned z = (nk + kchunk - 1) / kchunk;
  const size_t scales = (size_t)(uK / 32) * mpad * 8;  // float2 (d, d * sum) a group and row
  const CUdeviceptr scr = c.mmq_scratch_(scales + (size_t)mpad * uK);
  CUfunction quant = c.cached_(c.mmq_quant_fn, "tl_quant_act_mmq");
  CUfunction gemm = c.mmq_(type, glu);
  if (!scr || !quant || !gemm) return false;
  void* ps = reinterpret_cast<void*>(scr);
  void* pq = reinterpret_cast<char*>(ps) + scales;
  if (!c.launch_(quant, {(unsigned)(((size_t)mpad * (uK / 8) + 255) / 256)}, {256}, 0, pa, pq, ps, uM,
                 mpad, uK))
    return false;
  if (z == 1)
    return c.launch_(gemm, {tm, tn}, {context::mmq_threads}, context::mmq_smem, pq, ps, pw, out, uM, uN,
                     uK, mpad, ldc, kchunk, 0u);
  const unsigned stride = uM * uN;
  const CUdeviceptr part = c.mmq_part_scratch_((size_t)z * stride * sizeof(float));
  CUfunction reduce = c.cached_(c.mmq_reduce_fn, "tl_mmq_reduce");
  if (!part || !reduce) return false;
  float* pp = reinterpret_cast<float*>(part);
  return c.launch_(gemm, {tm, tn, z}, {context::mmq_threads}, context::mmq_smem, pq, ps, pw, pp, uM, uN,
                   uK, mpad, uN, kchunk, stride) &&
         c.launch_(reduce, {(uM * (uN / 2) + 255) / 256}, {256}, 0, pp, out, uM, uN, ldc, z, stride);
}

inline bool own_nn::gemm_quant(int type, bool a16, gpu::span A, gpu::span W, gpu::span C, int64_t m,
                             int64_t n, int64_t k, int64_t ldc, bool mmq) {
  auto& c = context::get();
  if (!c.ready || m <= 0 || n <= 0 || k % (type == 8 ? 32 : 256) || n % 2 || ldc % 2 || ldc < n ||
      (type != 12 && type != 14 && type != 8)) {
    return false;
  }
  float* pa = context::off_(A.buf, A.off);
  float* pw = context::off_(W.buf, W.off);
  float* pc = context::off_(C.buf, C.off);
  c.device_read_(A.buf);
  c.device_read_(W.buf);
  c.device_rmw_(C.buf);  // a part writes some columns of a shared output
  const context::tc_weight qw = type == 12 ? context::tc_weight::q4k
                                : type == 14 ? context::tc_weight::q6k : context::tc_weight::q80;
  if (mmq && a16 && (type == 12 || type == 8) && k % 256 == 0)
    return mmq_run_(c, type, false, pa, pw, pc, (unsigned)m, (unsigned)n, (unsigned)k, (unsigned)ldc);
  if (!a16) {  // the exact form dequantizes in the GEMM's staging
    return gemm_tc_run_(c, context::tc_form::f32, pa, pw, pc, (unsigned)m, (unsigned)n, (unsigned)k,
                        (unsigned)ldc, qw);
  }
  // The fast form: the weight to f16 first, a slab of rows at a time (the GEMM then
  // streams plain f16 and runs at 72 TFLOPS rather than the 45 of dequantizing in
  // every row block of the activation). The values are the staging's, rounded the
  // same, so the product is the same bits.
  // Bytes a row of k weights takes: Q4_K 144 per 256, Q6_K 210 per 256, Q8_0 34 per 32.
  const size_t row_bytes = type == 12 ? (size_t)k / 256 * 144 : type == 14 ? (size_t)k / 256 * 210 : (size_t)k / 32 * 34;
  // 192 MB: a 27B's largest projection (down, 5120 x 17408) in one slab. Split in two it left
  // each GEMM too few blocks for the long-K tile (pp512 1027 -> 1140 tok/s on the 27B).
  constexpr size_t slab_bytes = 192u << 20;
  const int64_t slab = std::max<int64_t>(128, (int64_t)(slab_bytes / ((size_t)k * 2)) / 128 * 128);
  CUfunction& fn = c.dequant_fn[type == 12 ? 0 : type == 14 ? 1 : 2];
  if (!fn) fn = c.load_(type == 12 ? "tl_dequant_f16_q4k" : type == 14 ? "tl_dequant_f16_q6k" : "tl_dequant_f16_q80");
  const CUdeviceptr scr = c.dequant_scratch_((size_t)std::min<int64_t>(n, slab) * k * 2);
  if (!fn || !scr) return false;
  for (int64_t r0 = 0; r0 < n; r0 += slab) {
    const unsigned rows = (unsigned)std::min<int64_t>(slab, n - r0);
    const size_t threads = (size_t)rows * (k / 8);
    if (!c.launch_(fn, {(unsigned)((threads + 255) / 256)}, {256}, 0,
                   reinterpret_cast<unsigned char*>(pw) + (size_t)r0 * row_bytes,
                   reinterpret_cast<void*>(scr), rows, (unsigned)k))
      return false;
    if (!gemm_tc_run_(c, context::tc_form::f16h, pa, reinterpret_cast<void*>(scr), pc + r0,
                      (unsigned)m, rows, (unsigned)k, (unsigned)ldc, context::tc_weight::f16w))
      return false;
  }
  return true;
}

inline bool own_nn::gemm_quant_glu(int type, gpu::span A16, gpu::span W, gpu::span H16, int64_t m,
                                 int64_t ff, int64_t k) {
  auto& c = context::get();
  if (!c.ready || m <= 0 || ff <= 0 || ff % 2 || k % 256 || (type != 12 && type != 8)) return false;
  void* pa = context::off_(A16.buf, A16.off);
  void* pw = context::off_(W.buf, W.off);
  void* ph = context::off_(H16.buf, H16.off);
  c.device_read_(A16.buf);
  c.device_read_(W.buf);
  c.device_write_(H16.buf);
  return mmq_run_(c, type, true, pa, pw, ph, (unsigned)m, (unsigned)ff, (unsigned)k, (unsigned)ff);
}

inline bool own_nn::quantize_act_q8(gpu::span a, gpu::span qa, int64_t k) {
  auto& c = context::get();
  if (!c.ready || k <= 0 || k % 256) return false;
  float* pa = context::off_(a.buf, a.off);
  float* pq = context::off_(qa.buf, qa.off);
  c.device_read_(a.buf);
  c.device_write_(qa.buf);
  const unsigned nblk = (unsigned)(k / 32);
  return c.launch_(c.cached_(c.quant_act_fn, "tl_quant_act_q8"), {(nblk + 7) / 8}, {256}, 0, pa,
                   reinterpret_cast<signed char*>(pq),
                   reinterpret_cast<void*>(reinterpret_cast<char*>(pq) + k), nblk);
}

inline bool own_nn::gemv_q4k_q8(gpu::span qa, gpu::span W, gpu::span y, int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.ready || n <= 0 || k <= 0 || k % 256) return false;
  float* pq = context::off_(qa.buf, qa.off);
  float* pw = context::off_(W.buf, W.off);
  c.device_read_(qa.buf);
  c.device_read_(W.buf);
  c.device_write_(y.buf);
  constexpr unsigned rows_per_block = 4;
  return c.launch_(c.cached_(c.gemv_q4k_q8_fn, "tl_gemv_q4k_q8_row"),
                   {((unsigned)n + rows_per_block - 1) / rows_per_block}, {rows_per_block * 32}, 0,
                   reinterpret_cast<signed char*>(pq),
                   reinterpret_cast<void*>(reinterpret_cast<char*>(pq) + k),
                   reinterpret_cast<unsigned char*>(pw), context::off_(y.buf, y.off), (unsigned)n,
                   (unsigned)k);
}

inline bool own_nn::gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y, int64_t n,
                             int64_t k) {
  auto& c = context::get();
  if (!c.ready || n <= 0 || k <= 0 || k % (type == 8 ? 32 : 256) || (type != 12 && type != 14 && type != 8))
    return false;
  float* pa = context::off_(a.buf, a.off);
  float* pw = context::off_(W.buf, W.off);
  c.device_read_(a.buf);
  c.device_read_(W.buf);
  c.device_write_(y.buf);
  constexpr unsigned threads = 128;  // 4 warps a row
  return c.launch_(type == 12   ? c.cached_(c.gemv_q4k_fn, "tl_gemv_q4k_row")
                   : type == 14 ? c.cached_(c.gemv_q6k_fn, "tl_gemv_q6k_row")
                                : c.cached_(c.gemv_q80_fn, "tl_gemv_q80_row"),
                   {(unsigned)n}, {threads}, threads / 32 * sizeof(float), pa,
                   reinterpret_cast<unsigned char*>(pw),
                   context::off_(y.buf, y.off), (unsigned)n, (unsigned)k);
}

inline bool own_nn::rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows, int64_t T, int64_t D,
                              int64_t pos, float base, int64_t rot, float eps) {
  auto& c = context::get();
  // tl_rmsnorm_rope_<D>: a warp a row, for the head widths whose rotation pairs a lane
  // holds whole; other shapes are refused (gpu::rmsnorm_rope then runs rmsnorm and rope).
  const int e = D == 64 ? 0 : D == 128 ? 1 : D == 256 ? 2 : -1;
  if (!c.ready || e < 0 || rows <= 0 || T <= 0 || rot <= 0 || rot % 64 || rot > D || rows % T) return false;
  c.device_rmw_(x.buf);
  c.device_read_(w.buf);
  static const char* const names[3] = {"tl_rmsnorm_rope_64", "tl_rmsnorm_rope_128", "tl_rmsnorm_rope_256"};
  return c.launch_(c.cached_(c.rmsnorm_rope_fn[e], names[e]), {(unsigned)((rows + 7) / 8)}, {256u}, 0,
                   context::off_(x.buf, x.off), context::off_(w.buf, w.off), (unsigned)rows, (unsigned)T,
                   (unsigned)pos, base, (unsigned)rot, eps);
}

// ---- The f16 activation: producers that write what the f16 GEMM reads, so it
// is rounded once where it is made rather than by every block that stages it.
// Each is the plain op's kernel with an f16 output (the norms write it beside
// the f32 one: the residual and the logits still read that). ----
inline bool own_nn::rmsnorm_h16(gpu::span x, gpu::span w, gpu::span o, gpu::span o16,
                             int64_t n, float eps, int64_t rows) {
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::rmsnorm_params p{static_cast<uint32_t>(n), eps};
  const gpu::arg args[] = {gpu::in(x), gpu::in(w), gpu::out(o), gpu::out(o16)};
  return c.dispatch_(c.cached_(c.rmsnorm_h16_fn, "tl_rmsnorm_h16"), args, 4, &p, sizeof(p),
                     gpu::policy::one_group_per_row(rows));
}
inline bool own_nn::rmsnorm_res_h16(gpu::span x, gpu::span delta, gpu::span w,
                                 gpu::span xout, gpu::span hout, gpu::span hout16,
                                 int64_t n, float eps, int64_t rows) {
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::rmsnorm_params p{static_cast<uint32_t>(n), eps};
  const gpu::arg args[] = {gpu::in(x),    gpu::in(delta), gpu::in(w),
                           gpu::out(xout), gpu::out(hout), gpu::out(hout16)};
  return c.dispatch_(c.cached_(c.add_rmsnorm_h16_fn, "tl_add_rmsnorm_h16"), args, 6, &p,
                     sizeof(p), gpu::policy::one_group_per_row(rows));
}
inline bool own_nn::swiglu_h16(gpu::span gu, gpu::span o16, int64_t ff, int64_t rows) {
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::swiglu_params p{static_cast<uint32_t>(ff)};
  const gpu::arg args[] = {gpu::in(gu), gpu::out(o16)};
  return c.dispatch_(c.cached_(c.swiglu_h16_fn, "tl_swiglu_h16"), args, 2, &p, sizeof(p),
                     gpu::policy::flat_rows(ff, rows));
}
inline bool own_nn::gated_rmsnorm(gpu::span o, gpu::span z, int64_t ldz, gpu::span w, gpu::span out,
                               bool h16, int64_t T, int64_t H, int64_t D, float eps) {
  auto& c = context::get();
  if (!c.ready || T <= 0 || H <= 0 || D <= 0) return false;
  c.device_read_(o.buf);
  c.device_read_(z.buf);
  c.device_read_(w.buf);
  c.device_write_(out.buf);
  CUfunction f = h16 ? c.cached_(c.gated_rmsnorm_fn[1], "tl_gated_rmsnorm_h16")
                     : c.cached_(c.gated_rmsnorm_fn[0], "tl_gated_rmsnorm");
  const unsigned rows = (unsigned)(T * H);
  return f && c.launch_(f, {(rows + 7) / 8}, {256}, 0, context::off_(o.buf, o.off),
                        context::off_(z.buf, z.off), context::off_(w.buf, w.off),
                        context::off_(out.buf, out.off), rows, (unsigned)H, (unsigned)D,
                        (unsigned)ldz, eps);
}
inline bool own_nn::merge_heads_h16(gpu::span src, gpu::span dst16, int64_t T, int64_t H,
                                 int64_t D) {
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::merge_heads_params p{static_cast<uint32_t>(T), static_cast<uint32_t>(H),
                                  static_cast<uint32_t>(D)};
  const gpu::arg args[] = {gpu::in(src), gpu::out(dst16)};
  return c.dispatch_(c.cached_(c.merge_heads_h16_fn, "tl_merge_heads_h16"), args, 2, &p,
                     sizeof(p), gpu::policy::per_head(H, T, D));
}
inline bool own_nn::cvt_h16(gpu::span src, gpu::span dst16, int64_t n) {
  auto& c = context::get();
  if (!c.ready) return false;
  const uint32_t p = static_cast<uint32_t>(n);
  const gpu::arg args[] = {gpu::in(src), gpu::out(dst16)};
  return c.dispatch_(c.cached_(c.cvt_h16_fn, "tl_cvt_h16"), args, 2, &p, sizeof(p),
                     gpu::policy::flat(n));
}

// ---- M9 batched-prefill layout moves between token-major projections and
// head-major attention.

// Split-KV split-count heuristic: how many ctx-splits make grid = heads×S fill
// the SMs (heads alone is ~32 blocks « 82 SMs; target ~4 blocks/SM, and each
// split needs >=64 keys to amortize its fixed cost). Shared by attn_decode
// (evaluated at the live ctx) and attn_decode_dpos (evaluated at max_ctx, so
// the CUDA-graph grid is pos-independent).
// The kernel's rule: 128-thread blocks, so twice the fill (~4 per SM); no
// split under 256 keys, a split at least 64 keys and a multiple of its 4
// warps. A block's warps walk their keys in series, so it is the length of a
// split, not the bytes, that sets the time at depth: 64 against 128 took
// Qwen2.5-0.5B at 2048 cached tokens from 396 to 426 tok/s, and 32 bought
// nothing more.
inline constexpr gpu::policy::split_rule attn_split_rule{
    2 * traits::fill_groups, 256, 64, 4};
inline unsigned attn_split_count(unsigned n_heads, int64_t ctx) {
  return static_cast<unsigned>(
      gpu::policy::split_parts(n_heads, ctx, attn_split_rule));
}

// Launch shape of the tiled prefill attention — the ONE place it lives. It is
// ABI, not a tuning knob: the kernel derives its query tile from the same
// numbers, and a mismatch would silently compute the wrong rows rather than fail
// to launch, so tl_attn_prefill_tiled_* names this as its device twin (cf.
// attn_dpos_chunk and attn_split_chunk below). A block is always 128 threads
// (16 row slots x 8 lanes); how many queries each thread carries is what varies,
// and D=128 carries fewer only because its Q tile would otherwise blow the 48 KB
// static shared limit. See the kernel for the measurements behind both.
inline constexpr unsigned attn_tile_threads = 128;
inline constexpr unsigned attn_tile_queries(int64_t D) { return D == 64 ? 64 : D == 128 ? 32 : 16; }

// The same ABI for the two backward kernels — rows of the sequence a block
// owns, queries in the dq half and keys in the dK/dV half. Half the forward's
// tile: both halves hold a second [tile, D] block of shared memory (the output
// gradient's rows, or the probabilities), and the forward's tile would put them
// past the 48 KB of static shared memory.
inline constexpr unsigned attn_bwd_tile(int64_t D) {
  return attn_tile_queries(D) / 2;
}

// Keys per split for the split-KV kernels: ceil(ctx / attn_split_count), rounded
// up to a multiple of 4 warps. The device-side attn_dpos_chunk in
// tensorlib_cuda.cu is the twin of attn_split_count + this rounding and MUST
// stay in lockstep — the host/dpos bit-identity rests on it. Guarded by the
// attn64 ctest's host-vs-dpos bit-equality sweep.
inline unsigned attn_split_chunk(unsigned n_heads, int64_t ctx) {
  return static_cast<unsigned>(gpu::policy::split_chunk(
      ctx, attn_split_count(n_heads, ctx), attn_split_rule));
}

// Split-KV partials scratch: ONE buffer laid out pm[H*S] | pl[H*S] | pacc[H*S*D]
// (hs = H*S). bytes/the carve constructor/combine() keep that layout contract in
// a single place — attn_decode's split branch and attn_decode_dpos both bake
// these pointers into launches (the dpos ones into a captured graph).
struct attn_partials {
  float *pm, *pl, *pacc;
  attn_partials(float* base, size_t hs) : pm(base), pl(pm + hs), pacc(pl + hs) {}
  static size_t bytes(size_t hs, int64_t D) {
    return (hs * 2 + hs * static_cast<size_t>(D)) * sizeof(float);
  }
  // Merge the S per-head partials into out [H, D] (tl_attn_combine).
  bool combine(context& c, float* out, unsigned uh, unsigned uD, unsigned S) {
    return c.launch_(c.attn_combine_(), {uh}, {uD}, 0, pm, pl, pacc, out, S);
  }
};

// The head widths the decode-attention kernels are instantiated for: 64 and
// 128 with either cache dtype, 256 with an f32 cache.
inline bool attn_decode_head_ok(int64_t D) { return D == 64 || D == 128 || D == 256; }

// M9 fused decode attention: out(h,:) = softmax(scale·q(h,:)·K(kv,:)^T)·V(kv) in
// one pass. q [n_q_heads,D], out [n_q_heads,D]; K/V are a [n_kv_heads,kv_max,D]
// cache read over its valid prefix [0,ctx) (kv_max==ctx is the no-cache case).
// GQA: q head h reads kv head h/(n_q_heads/n_kv_heads). Contiguous, D as
// attn_decode_head_ok allows; the cache f32, or f16 with kv_f16.
inline bool own_nn::attn_decode(gpu::span q, gpu::span K, gpu::span V,
                             gpu::span out, int64_t n_q_heads,
                             int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                             int64_t D, float scale, bool kv_f16) {
  auto& c = context::get();
  if (!c.ready || !attn_decode_head_ok(D)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_write_(out.buf);
  float* pq = context::off_(q.buf, q.off);
  float* pk = context::off_(K.buf, K.off);
  float* pv = context::off_(V.buf, V.off);
  float* po = context::off_(out.buf, out.off);
  unsigned uh = static_cast<unsigned>(n_q_heads), uctx = static_cast<unsigned>(ctx);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  unsigned group = static_cast<unsigned>(n_q_heads / n_kv_heads);

  // Split ctx over gridDim.y so grid = heads×S fills the SMs (see
  // attn_split_count; chunk below is rounded to a multiple of 4 warps).
  unsigned S = attn_split_count(uh, ctx);
  unsigned uD = static_cast<unsigned>(D);
  if (S == 1) {
    return c.launch_(c.attn_(context::attn_k::decode, D, kv_f16), {uh}, {uD}, 0, pq, pk, pv, po,
                     uctx, kv_stride, group, scale);
  }

  unsigned chunk = attn_split_chunk(uh, ctx);
  S = (uctx + chunk - 1) / chunk;  // recompute after rounding
  size_t hs = (size_t)uh * S;
  CUdeviceptr scr = c.attn_scratch_(attn_partials::bytes(hs, D));
  if (!scr) return false;
  attn_partials p(reinterpret_cast<float*>(scr), hs);
  if (!c.launch_(c.attn_(context::attn_k::split, D, kv_f16), {uh, S}, {uD}, 0, pq, pk, pv, p.pm,
                 p.pl, p.pacc, uctx, kv_stride, group, chunk, scale)) {
    return false;
  }
  return p.combine(c, po, uh, uD, S);
}

// M9 causal prefill attention: q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D]
// cache read over [0,pos0+T). Query p is at absolute position pos0+p and attends
// keys 0..pos0+p, so a long prompt can be run in chunks (and a later turn
// appended to a live cache). GQA via group. D∈{64,128}.
// One block per (head, query tile); grid = (n_q_heads, ceil(T/tile)).
inline bool own_nn::attn_prefill(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span out, int64_t n_q_heads,
                              int64_t n_kv_heads, int64_t T, int64_t kv_max,
                              int64_t D, float scale, bool kv_f16,
                              int64_t pos0) {
  auto& c = context::get();
  // D 64 or 128; 256 for an f16 cache (an f32 one there takes gpu::attn_prefill's
  // GEMM form, which reads only f32).
  if (!c.ready || (D != 128 && D != 64 && !(D == 256 && kv_f16))) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  if (T <= 0 || T > 65535) return false;  // one call is one prompt chunk
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_write_(out.buf);
  float* pq = context::off_(q.buf, q.off);
  float* pk = context::off_(K.buf, K.off);
  float* pv = context::off_(V.buf, V.off);
  float* po = context::off_(out.buf, out.off);
  unsigned uT = static_cast<unsigned>(T);
  unsigned kv_stride = static_cast<unsigned>(kv_max * D);
  unsigned group = static_cast<unsigned>(n_q_heads / n_kv_heads);
  unsigned up0 = static_cast<unsigned>(pos0);

  // A block takes a tile of queries and streams K/V through shared memory, so
  // each score is a register dot product instead of a per-key warp-shuffle
  // reduction. Same online softmax, same causal rule. See
  // tl_attn_prefill_tiled_*.
  const unsigned tile = attn_tile_queries(D);
  return c.launch_(c.attn_(context::attn_k::tiled, D, kv_f16),
                   {static_cast<unsigned>(n_q_heads), (uT + tile - 1) / tile},
                   {attn_tile_threads}, 0, pq, pk, pv, po, uT, kv_stride, group,
                   scale, up0);
}

// The same causal prefill on the tensor cores (tl_attn_prefill_tc_*), its
// operands rounded to f16 as staged (an f16 cache's K and V already are): head
// width 64, 128 or 256, operands where the kernel declares them. Inference only
// (gpu::attn_prefill_f16).
inline bool own_nn::attn_prefill_f16(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span out, int64_t n_q_heads,
                                  int64_t n_kv_heads, int64_t T, int64_t kv_max,
                                  int64_t D, float scale, bool kv_f16,
                                  int64_t pos0) {
  auto& c = context::get();
  if (!c.ready || (D != 64 && D != 128 && D != 256)) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  if (T <= 0 || T > 65535) return false;
  float* pq = context::off_(q.buf, q.off);
  float* pk = context::off_(K.buf, K.off);
  float* pv = context::off_(V.buf, V.off);
  float* po = context::off_(out.buf, out.off);
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_write_(out.buf);
  const unsigned uT = static_cast<unsigned>(T), bq = 64u;  // 4 warps, 16 queries each
  return c.launch_(c.attn_(context::attn_k::tc, D, kv_f16),
                   {static_cast<unsigned>(n_q_heads), (uT + bq - 1) / bq}, {bq * 2},
                   context::attn_tc_smem(D), pq,
                   pk, pv, po, uT, static_cast<unsigned>(kv_max * D),
                   static_cast<unsigned>(n_q_heads / n_kv_heads), scale,
                   static_cast<unsigned>(pos0));
}

// A prefill's conv (tl_ssm_conv_tokens, then tl_ssm_conv_state for the window
// the next call starts from).
inline bool own_nn::ssm_conv_tokens(gpu::span x, int64_t ld, gpu::span state,
                                 gpu::span w, gpu::span o, int64_t C, int64_t K,
                                 int64_t T) {
  auto& c = context::get();
  if (!c.ready || K < 2 || K > 8 || T < 2 || T > 65535) return false;
  c.device_read_(x.buf);
  c.device_rmw_(state.buf);
  c.device_read_(w.buf);
  c.device_write_(o.buf);
  float* px = context::off_(x.buf, x.off);
  float* ps = context::off_(state.buf, state.off);
  float* pw = context::off_(w.buf, w.off);
  float* po = context::off_(o.buf, o.off);
  const unsigned uC = (unsigned)C, uK = (unsigned)K, uT = (unsigned)T, uld = (unsigned)ld;
  const unsigned gx = (uC + 255) / 256;
  return c.launch_(c.conv_tok_(), {gx, uT}, {256}, 0, px, ps, pw, po, uC, uK, uT, uld) &&
         c.launch_(c.conv_state_(), {gx}, {256}, 0, px, ps, uC, uK, uT, uld);
}

// The chunked delta rule (tl_gated_delta_chunk_prep, then _scan): dk 128, dv a
// multiple of 32 up to 128, and T past one chunk. The scratch holds, per head,
// T rows each of q, k and W (dk) and U (dv), g, and a C x C P per chunk.
inline bool own_nn::gated_delta_chunked(gpu::span qkv, gpu::span ab, int64_t ab_ld,
                                     gpu::span adt, gpu::span state, gpu::span o,
                                     int64_t H, int64_t dk, int64_t dv, int64_t T,
                                     float eps, int64_t Hk) {
  constexpr int64_t C = 32;
  auto& c = context::get();
  if (!c.ready || dk != 128 || dv % 32 || dv > 128 || T < C || Hk <= 0 || H % Hk) return false;
  const int64_t nch = (T + C - 1) / C;
  const size_t rows = (size_t)H * T;
  const size_t floats = rows * (3 * dk + dv + 1) + (size_t)H * nch * C * C;
  CUdeviceptr scr = c.delta_scratch_(floats * sizeof(float));
  if (!scr) return false;
  float* Qn = reinterpret_cast<float*>(scr);
  float* Kn = Qn + rows * dk;
  float* W = Kn + rows * dk;
  float* U = W + rows * dk;
  float* G = U + rows * dv;
  float* P = G + rows;
  c.device_read_(qkv.buf);
  c.device_read_(ab.buf);
  c.device_read_(adt.buf);
  c.device_rmw_(state.buf);
  c.device_write_(o.buf);
  float* pqkv = context::off_(qkv.buf, qkv.off);
  float* pab = context::off_(ab.buf, ab.off);
  float* padt = context::off_(adt.buf, adt.off);
  float* pst = context::off_(state.buf, state.off);
  float* po = context::off_(o.buf, o.off);
  const unsigned uH = (unsigned)H, udv = (unsigned)dv, uT = (unsigned)T,
                 uld = (unsigned)ab_ld;
  // prep is laid out for 256 threads (tl_gated_delta_chunk_prep).
  return c.launch_(c.gdc_prep_(), {(unsigned)nch, uH}, {256}, 0, pqkv, pab, padt, Qn,
                   Kn, W, U, P, G, uH, udv, uT, uld, eps, (unsigned)Hk) &&
         c.launch_(c.gdc_scan_(), {udv / 32, uH}, {context::gdc_scan_threads},
                   context::gdc_scan_smem, Qn, Kn, W, U, P, G, pst, po, uH, udv, uT);
}

// The query half of causal prefill attention's pullback: q, K, V, dO, O and dq
// all [H,T,D] contiguous — a training shape, so no KV cache, no GQA and no
// chunked positions — and `stats` [2,H,T] the row logsumexp and dO·O the dK/dV
// half reads. D∈{64,128}. One block per (head, query tile).
inline bool own_nn::attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                                 gpu::span dO, gpu::span O, gpu::span dq,
                                 gpu::span stats, int64_t H, int64_t T,
                                 int64_t D, float scale) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (H <= 0 || T <= 0 || T > 65535) return false;
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_read_(dO.buf);
  c.device_read_(O.buf);
  c.device_write_(dq.buf);
  c.device_write_(stats.buf);
  unsigned uT = static_cast<unsigned>(T);
  const unsigned tile = attn_bwd_tile(D);
  return c.launch_(c.attn_bwd_dq_(D),
                   {static_cast<unsigned>(H), (uT + tile - 1) / tile},
                   {attn_tile_threads}, 0, context::off_(q.buf, q.off),
                   context::off_(K.buf, K.off), context::off_(V.buf, V.off),
                   context::off_(dO.buf, dO.off), context::off_(O.buf, O.off),
                   context::off_(dq.buf, dq.off), context::off_(stats.buf, stats.off), uT, scale);
}

// The key/value half, reading the stats the call above wrote: q, K, V, dO, dK
// and dV all [H,T,D] contiguous, `stats` [2,H,T]. One block per (head, key
// tile), and the head count reaches the kernel as gridDim.x — it is what the
// stats' plane stride is made of.
inline bool own_nn::attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span dO, gpu::span stats, gpu::span dK,
                                  gpu::span dV, int64_t H, int64_t T, int64_t D,
                                  float scale) {
  auto& c = context::get();
  if (!c.ready || (D != 128 && D != 64)) return false;
  if (H <= 0 || T <= 0 || T > 65535) return false;
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_read_(dO.buf);
  c.device_read_(stats.buf);
  c.device_write_(dK.buf);
  c.device_write_(dV.buf);
  unsigned uT = static_cast<unsigned>(T);
  const unsigned tile = attn_bwd_tile(D);
  return c.launch_(c.attn_bwd_dkv_(D),
                   {static_cast<unsigned>(H), (uT + tile - 1) / tile},
                   {attn_tile_threads}, 0, context::off_(q.buf, q.off),
                   context::off_(K.buf, K.off), context::off_(V.buf, V.off),
                   context::off_(dO.buf, dO.off), context::off_(stats.buf, stats.off),
                   context::off_(dK.buf, dK.off), context::off_(dV.buf, dV.off), uT, scale);
}

// RoPE: rotate a contiguous [rows, D] buffer (rows = H*T). Row r's position is
// pos + (r % T); half-split (GPT-NeoX / HF-llama) convention. D must be even.
inline bool own_nn::rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                      int64_t D, int64_t pos, float base, gpu::span bias) {
  return rope_partial(x, out, rows, T, D, pos, base, bias, D);
}
inline bool own_nn::rope_partial(gpu::span x, gpu::span out, int64_t rows,
                              int64_t T, int64_t D, int64_t pos, float base,
                              gpu::span bias, int64_t rot) {
  auto& c = context::get();
  if (!c.ready || D <= 0 || (D & 1) || rot <= 0 || (rot & 1) || rot > D) return false;
  c.device_read_(x.buf);
  if (bias.buf) c.device_read_(bias.buf);
  c.device_write_(out.buf);
  float* px = context::off_(x.buf, x.off);
  float* pbias = bias.buf ? context::off_(bias.buf, bias.off) : nullptr;
  float* po = context::off_(out.buf, out.off);
  unsigned uT = static_cast<unsigned>(T), uD = static_cast<unsigned>(D),
           upos = static_cast<unsigned>(pos), urot = static_cast<unsigned>(rot);
  return c.launch_(c.rope_(), {static_cast<unsigned>(rows)},
                   {static_cast<unsigned>(D / 2)}, 0, px, pbias, po, uT, uD,
                   upos, base, urot);
}

// ---- Row-wise fused RMSNorm / SwiGLU. `rows` defaults to 1 (a decode step); a
// batched prefill chunk passes its token count. ONE kernel serves both, which is
// what keeps the two paths from drifting numerically. Buffers are [rows, n]
// contiguous; the weight is [n], shared by every row.

// Layer norm's pullback: dx [rows, cols], dg and db [cols] from x and dy
// [rows, cols] and the d-vector g, all contiguous; dx/dg/db, `stats` [2, rows]
// and `partials` [2, chunks, cols] are fresh buffers of the caller's, the rows
// taken `per_chunk` at a time. A row kernel, a column-strip kernel, a fold.
inline bool own_nn::layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                                gpu::span dx, gpu::span dg, gpu::span db,
                                gpu::span stats, gpu::span partials,
                                int64_t rows, int64_t cols, int64_t per_chunk,
                                int64_t chunks, float eps) {
  auto& c = context::get();
  if (!c.ready || rows <= 0 || cols <= 0 || chunks <= 0) return false;
  c.device_read_(x.buf);
  c.device_read_(g.buf);
  c.device_read_(dy.buf);
  c.device_write_(dx.buf);
  c.device_write_(dg.buf);
  c.device_write_(db.buf);
  c.device_write_(stats.buf);
  c.device_write_(partials.buf);
  float* px = context::off_(x.buf, x.off);
  float* pg = context::off_(g.buf, g.off);
  float* pdy = context::off_(dy.buf, dy.off);
  float* pdx = context::off_(dx.buf, dx.off);
  float* pdg = context::off_(dg.buf, dg.off);
  float* pdb = context::off_(db.buf, db.off);
  float* ps = context::off_(stats.buf, stats.off);
  float* pp = context::off_(partials.buf, partials.off);
  unsigned ur = (unsigned)rows, uc = (unsigned)cols, uk = (unsigned)chunks;
  unsigned up = (unsigned)per_chunk;
  unsigned block = 256;
  if (!c.launch_(c.layer_norm_bwd_dx_(), {ur}, {block},
                 2 * block * sizeof(float), px, pg, pdy, pdx, ps, ur, uc, eps)) {
    return false;
  }
  if (!c.launch_(c.layer_norm_bwd_gb_(), {(uc + 31) / 32, uk}, {32, 8}, 0,
                 px, pdy, ps, pp, ur, uc, up)) {
    return false;
  }
  return c.launch1d_(c.layer_norm_bwd_gb_fold_(), uc, pp, pdg, pdb, uk, uc);
}

}  // namespace cuda
}  // namespace tl
