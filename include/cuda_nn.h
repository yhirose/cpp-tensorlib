#pragma once

// The CUDA backend's nn layer (layers: gpu_abi.h): its own ops for the fused ops over plain
// tensors (the GEMVs and GEMMs over bf16, f16 and block-quantized weights, the
// f16 activation, attention, RoPE, DeltaNet, layer norm's pullback) and their
// launch helpers.
//
// Included by cuda.h, after the device core.

#include "ggml_quant.h"
#include "gpu_abi_nn.h"

#include <iterator>  // std::size

namespace tl {
namespace cuda {

struct own_nn {
  static bool split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                          int64_t T, int64_t ld, int64_t off, int64_t H,
                          int64_t D);
  // x[rows, D] RMS-normalized by w and then rotated, in place (rows = heads * T); D 64,
  // 128 or 256 and f.rot a multiple of 64, else false.
  static bool rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows, int64_t T, int64_t D,
                           int64_t pos, const gpu::rope_freq& f, float eps);
  // The activation a[K] prepared in form f into pa, and block-quantized rows dotted with
  // it. q8 (K % 32 == 0): int8 blocks of 32 (K int8, then K/32 float2 of scale and sum),
  // read with dp4a. h16 (K % 256 == 0): K f16, K/32 f32 sums and K/256 f32 scales, read
  // by the tensor cores. act_layout_ is where a form keeps its parts, the one place the launchers and
  // act_bytes read it; quant_fmts_ says which format has a kernel for which form
  // (quant_act_reads), and act_slot_ is a form's column there.
  struct act_layout {  // byte offsets from the start; bytes 0: no such form
    int64_t meta = 0;   // q8: the float2 (scale, sum) of each 32; h16: the f32 sum of each 32
    int64_t scale = 0;  // h16: the f32 scale of each 256
    int64_t bytes = 0;
  };
  static act_layout act_layout_(gpu::act_form f, int64_t k);
  static int act_slot_(gpu::act_form f);
  static const char* act_kernel_(int type, gpu::act_form f, int64_t k, int& fi, int& slot);
  static int64_t act_bytes(gpu::act_form f, int64_t k) { return act_layout_(f, k).bytes; }
  static bool prepare_act(gpu::act_form f, gpu::span a, gpu::span pa, int64_t k);
  static bool quant_act_reads(int type, gpu::act_form f, int64_t k);
  static bool gemv_quant_act(int type, gpu::act_form f, gpu::span pa, gpu::span W, gpu::span y,
                             int64_t n, int64_t k);
  // C[m, ldc-strided n columns] = A[m,K] . W[n,K]^T over GGUF block-quantized rows (type: a
  // format of quant_fmts_, K a whole number of its blocks). a16: A is f16 and the sums f16,
  // else A f32, the operands f32 to ~2^-16 (bf16 pairs) and the sums f32 (the exact form:
  // no f16 range to overflow, and the products an f32 GEMV's to ~1e-5). n even. mmq (with
  // a16, a format whose quant_fmt has MMQ kernels, and K % 256 == 0): A is quantized to
  // int8 and the product runs on the int8 tensor cores, nothing dequantized
  // (tl_gemm_mmq_*): faster, and the activation rounded to 8 bits.
  static bool gemm_quant(int type, bool a16, gpu::span A, gpu::span W, gpu::span C, int64_t m,
                          int64_t n, int64_t k, int64_t ldc, bool mmq = false);
  // H16[m, ff] = act(A16 . Wg^T) * (A16 . Wu^T) in f16, W the fused [Wg | Wu] (2 ff rows of
  // one format with MMQ kernels, K % 256 == 0) on the int8 tensor cores, as gemm_quant(mmq) then
  // glu_h16 would make it. False where it does not apply (the caller does those two).
  static bool gemm_quant_glu(int type, gpu::glu_act act, gpu::span A16, gpu::span W,
                             gpu::span H16, int64_t m, int64_t ff, int64_t k);
  // y[n] = W[n,K] . a over GGUF block-quantized rows (type: the ggml_type code, a format
  // of quant_fmts_), the blocks read where they lie. K a whole number of blocks.
  static bool gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y, int64_t n, int64_t k);
  // Whether the kernels read `type`'s blocks in rows of k (a format of quant_fmts_, k a
  // whole number of its blocks).
  static bool quant_reads(int type, int64_t k);
  // The f16 prefill GEMM reading an activation that is already f16, and the
  // producers that round it where it is made (see their definitions).
  static bool gemm_bf16_nt_f16acc_a16(gpu::span a, gpu::span B, gpu::span C,
                                      int64_t M, int64_t N, int64_t K);
  static bool rmsnorm_h16(gpu::span x, gpu::span w, gpu::span o, gpu::span o16,
                          int64_t n, float eps, int64_t rows);
  // (wd: the branch normalized first, as gpu::rmsnorm_res's)
  static bool rmsnorm_res_h16(gpu::span x, gpu::span delta, gpu::span w,
                              gpu::span xout, gpu::span hout, gpu::span hout16,
                              int64_t n, float eps, int64_t rows, gpu::span wd = {});
  static bool glu_h16(gpu::glu_act act, gpu::span gu, gpu::span o16, int64_t ff, int64_t rows);
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
                           bool kv_f16, int64_t pos0, int64_t window);
  static bool attn_prefill_f16(gpu::span q, gpu::span K, gpu::span V, gpu::span out,
                               int64_t n_q_heads, int64_t n_kv_heads, int64_t T,
                               int64_t kv_max, int64_t D, float scale,
                               bool kv_f16, int64_t pos0, int64_t window);
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
  // Every gpu::rope_freq: a part of the row (Qwen3.5), a scaled position (Gemma 3's
  // global layers), per-pair factors (Llama 3).
  static bool rope(gpu::span x, gpu::span out, int64_t rows, int64_t T, int64_t D,
                   int64_t pos, const gpu::rope_freq& f, gpu::span bias);
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
    const auto sh = context::gemm_tc_shape(t, form, w);
    return std::pair<unsigned, unsigned>{(n + sh.cols - 1) / sh.cols, (m + sh.rows - 1) / sh.rows};
  };
  auto blocks = [&](tile t) { return (int64_t)grid(t).first * grid(t).second; };
  // The f16 weight's long-K tile (a big model's projections): one block a SM, so it
  // wants a grid that is nearly whole waves (the 6144 x 5120 at 96 blocks is 59% of
  // two waves and loses to the 128-row tile), and K = 1024 shapes lose to the others.
  constexpr int64_t sms = traits::fill_groups / 2;
  const int64_t bx = blocks(tile::xl), waves = (bx + sms - 1) / sms;
  const bool xl = context::gemm_tc_has(tile::xl, form, w) && k >= 3072 &&
                  bx >= sms - 2 && bx * 4 >= waves * sms * 3;
  const tile t = xl                        ? tile::xl
                 : blocks(tile::l) >= busy && context::gemm_tc_has(tile::l, form, w) ? tile::l
                 : blocks(tile::m) >= busy ? tile::m
                                           : tile::s;
  const auto sh = context::gemm_tc_shape(t, form, w);
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
  if (blocks < 128 && c.d.MemsetD8Async && !c.no_splitk) {
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

// The GGUF block formats the kernels read, a row each: the ggml type; the format's
// index in the GEMM's staging and the device code (context::tc_weight, cuda_nn.cuh's
// WF); its GEMV, f16 dequantization and row-lookup kernels; its MMQ kernels' index
// (context::mmq_), -1 where it has none; its GEMVs reading a prepared activation, by
// form (act_slot_: q8, h16), null where it reads none; and its float GEMV's launch, the
// rows a block (the kernel's own count) and the block's threads (most: a row a block of
// 4 warps), with a float a row and warp of shared memory for the sums. A block's
// geometry is ggml's (gguf::ggml_block). Every reader of a format takes it from here.
struct quant_fmt {
  int type;
  context::tc_weight wf;
  const char *gemv, *dequant, *embed;
  bool mmq;  // has MMQ kernels (tl_gemm_mmq_*, named by wf: context::mmq_)
  const char* gemv_act[context::act_forms_max];
  unsigned gemv_rows = 1, gemv_threads = 128;
};
inline constexpr quant_fmt quant_fmts_[] = {
    {12, context::tc_weight::q4k, "tl_gemv_q4k_row", "tl_dequant_f16_q4k", "tl_embed_row_q4k", true,
     {"tl_gemv_q4k_q8_row", "tl_gemv_q4k_h16_row"}},
    {14, context::tc_weight::q6k, "tl_gemv_q6k_row", "tl_dequant_f16_q6k", "tl_embed_row_q6k", false, {}},
    {8, context::tc_weight::q80, "tl_gemv_q80_row", "tl_dequant_f16_q80", "tl_embed_row_q80", true, {}},
    {6, context::tc_weight::q50, "tl_gemv_q50_row", "tl_dequant_f16_q50", "tl_embed_row_q50", false, {}},
    {13, context::tc_weight::q5k, "tl_gemv_q5k_row", "tl_dequant_f16_q5k", "tl_embed_row_q5k", true,
     {"tl_gemv_q5k_q8_row", "tl_gemv_q5k_h16_row"}},
    {2, context::tc_weight::q40, "tl_gemv_q40_row", "tl_dequant_f16_q40", "tl_embed_row_q40", true,
     {"tl_gemv_q40_q8_row", nullptr}, 2, 64},
    {3, context::tc_weight::q41, "tl_gemv_q41_row", "tl_dequant_f16_q41", "tl_embed_row_q41", true,
     {"tl_gemv_q41_q8_row", nullptr}, 2, 64},
};
static_assert(sizeof(quant_fmts_) / sizeof(quant_fmts_[0]) <= context::quant_fmts_max, "slots");
// The row of `type` where rows of k weights are a whole number of its blocks, else -1.
inline int quant_fmt_(int type, int64_t k) {
  for (int i = 0; i < (int)(sizeof(quant_fmts_) / sizeof(quant_fmts_[0])); i++) {
    if (quant_fmts_[i].type != type) continue;
    uint64_t elems = 0, bytes = 0;
    gguf::ggml_block(static_cast<gguf::ggml_type>(type), elems, bytes);
    return k > 0 && k % (int64_t)elems == 0 ? i : -1;
  }
  return -1;
}
inline bool own_nn::quant_reads(int type, int64_t k) { return quant_fmt_(type, k) >= 0; }

// The MMQ GEMM (tl_gemm_mmq_*): A (f16, M x K) rounded to int8 into a scratch, then the
// product with the block rows at W (format w, one with MMQ kernels) into out: f32 C columns (epi 0), or f16 act(gate) * up of a fused gate|up W, n
// the outputs (epi 1 + glu_act: context::mmq_). False where the GLU would need K split
// (the caller then takes the unfused path).
inline bool mmq_run_(context& c, context::tc_weight w, int epi, void* pa, void* pw, void* out, unsigned uM,
                     unsigned uN, unsigned uK, unsigned ldc) {
  constexpr unsigned tile = context::mmq_tile;
  // A tile a block and one block an SM: a grid short of the SMs (a narrow projection,
  // N ~ 1K) leaves some idle, so K is split into as many parts as still fit in one
  // wave (rounding down: a part past it would start a second, nearly empty wave), each
  // at least two super-blocks. The parts' sums go through a scratch and are added in order.
  const unsigned mpad = (uM + tile - 1) / tile * tile, nk = uK / 256;
  // Weight tiles tn, activation tiles tm (the grid's y and x: tl_gemm_mmq_*'s order).
  const unsigned tn = epi ? (uN + tile / 2 - 1) / (tile / 2) : (uN + tile - 1) / tile, tm = mpad / tile;
  constexpr unsigned sms = traits::fill_groups / 2;
  const unsigned parts = std::max(1u, std::min(sms / (tn * tm), nk / 2));
  if (epi && parts > 1) return false;
  const unsigned kchunk = (nk + parts - 1) / parts;
  const unsigned z = (nk + kchunk - 1) / kchunk;
  const size_t scales = (size_t)(uK / 32) * mpad * 8;  // float2 (d, d * sum) a group and row
  const CUdeviceptr scr = c.mmq_scratch_(scales + (size_t)mpad * uK);
  CUfunction quant = c.cached_(c.mmq_quant_fn, "tl_quant_act_mmq");
  CUfunction gemm = c.mmq_(w, epi);
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
  const int fi = quant_fmt_(type, k);
  if (!c.ready || m <= 0 || n <= 0 || fi < 0 || n % 2 || ldc % 2 || ldc < n) return false;
  const quant_fmt& f = quant_fmts_[fi];
  float* pa = context::off_(A.buf, A.off);
  float* pw = context::off_(W.buf, W.off);
  float* pc = context::off_(C.buf, C.off);
  c.device_read_(A.buf);
  c.device_read_(W.buf);
  c.device_rmw_(C.buf);  // a part writes some columns of a shared output
  if (mmq && a16 && f.mmq && k % 256 == 0)
    return mmq_run_(c, f.wf, 0, pa, pw, pc, (unsigned)m, (unsigned)n, (unsigned)k, (unsigned)ldc);
  if (!a16) {  // the exact form dequantizes in the GEMM's staging
    return gemm_tc_run_(c, context::tc_form::f32, pa, pw, pc, (unsigned)m, (unsigned)n, (unsigned)k,
                        (unsigned)ldc, f.wf);
  }
  // The fast form: the weight to f16 first, a slab of rows at a time (the GEMM then
  // streams plain f16 and runs at 72 TFLOPS rather than the 45 of dequantizing in
  // every row block of the activation). The values are the staging's, rounded the
  // same, so the product is the same bits.
  const size_t row_bytes = gguf::bytes_of(static_cast<gguf::ggml_type>(type), (uint64_t)k);
  // 192 MB: a 27B's largest projection (down, 5120 x 17408) in one slab. Split in two it left
  // each GEMM too few blocks for the long-K tile (pp512 1027 -> 1140 tok/s on the 27B).
  constexpr size_t slab_bytes = 192u << 20;
  const int64_t slab = std::max<int64_t>(128, (int64_t)(slab_bytes / ((size_t)k * 2)) / 128 * 128);
  const CUfunction fn = c.cached_(c.dequant_fn[fi], f.dequant);
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

inline bool own_nn::gemm_quant_glu(int type, gpu::glu_act act, gpu::span A16, gpu::span W,
                                 gpu::span H16, int64_t m, int64_t ff, int64_t k) {
  auto& c = context::get();
  const int fi = quant_fmt_(type, k);
  if (!c.ready || m <= 0 || ff <= 0 || ff % 2 || k % 256 || fi < 0 || !quant_fmts_[fi].mmq)
    return false;
  void* pa = context::off_(A16.buf, A16.off);
  void* pw = context::off_(W.buf, W.off);
  void* ph = context::off_(H16.buf, H16.off);
  c.device_read_(A16.buf);
  c.device_read_(W.buf);
  c.device_write_(H16.buf);
  int epi = 0;
  switch (act) {  // context::mmq_'s epilogues
    case gpu::glu_act::silu: epi = 1; break;
    case gpu::glu_act::gelu_tanh: epi = 2; break;
  }
  return mmq_run_(c, quant_fmts_[fi].wf, epi, pa, pw, ph, (unsigned)m, (unsigned)ff, (unsigned)k,
                  (unsigned)ff);
}

inline own_nn::act_layout own_nn::act_layout_(gpu::act_form f, int64_t k) {
  if (k <= 0) return {};
  switch (f) {
    case gpu::act_form::q8:  // a scale and a sum each 32
      if (k % 32 == 0) return {k, 0, k + k / 32 * 8};
      break;
    case gpu::act_form::h16:  // a sum each 32, a scale each 256
      if (k % 256 == 0) return {2 * k, 2 * k + k / 8, 2 * k + k / 8 + k / 64};
      break;
    case gpu::act_form::f32: break;
  }
  return {};
}

inline bool own_nn::prepare_act(gpu::act_form f, gpu::span a, gpu::span pa, int64_t k) {
  auto& c = context::get();
  const act_layout L = act_layout_(f, k);
  if (!c.ready || !pa.buf || !L.bytes) return false;
  float* pf = context::off_(a.buf, a.off);
  char* pp = reinterpret_cast<char*>(context::off_(pa.buf, pa.off));
  c.device_read_(a.buf);
  c.device_write_(pa.buf);
  switch (f) {
    case gpu::act_form::q8: {
      const unsigned nblk = (unsigned)(k / 32);
      return c.launch_(c.cached_(c.quant_act_fn, "tl_quant_act_q8"), {(nblk + 7) / 8}, {256}, 0,
                       pf, reinterpret_cast<signed char*>(pp),
                       reinterpret_cast<void*>(pp + L.meta), nblk);
    }
    case gpu::act_form::h16:
      return c.launch_(c.cached_(c.act_h16_fn, "tl_act_h16"), {(unsigned)(k / 256)}, {256}, 0,
                       pf, reinterpret_cast<void*>(pp), reinterpret_cast<void*>(pp + L.meta),
                       reinterpret_cast<void*>(pp + L.scale));
    case gpu::act_form::f32: break;
  }
  return false;
}

// A form's column of quant_fmts_' gemv_act: its place in gpu::prepared_act_forms.
static_assert(std::size(gpu::prepared_act_forms) == context::act_forms_max, "a column a form");
inline int own_nn::act_slot_(gpu::act_form f) {
  for (int i = 0; i < context::act_forms_max; i++)
    if (gpu::prepared_act_forms[i] == f) return i;
  return -1;
}

// The format's GEMV for form f over rows of k weights (its row fi and the form's column
// slot of quant_fmts_), null where it has none or the form does not take k.
inline const char* own_nn::act_kernel_(int type, gpu::act_form f, int64_t k, int& fi,
                                       int& slot) {
  fi = quant_fmt_(type, k);
  slot = act_slot_(f);
  if (fi < 0 || slot < 0 || !act_layout_(f, k).bytes) return nullptr;
  return quant_fmts_[fi].gemv_act[slot];
}
inline bool own_nn::quant_act_reads(int type, gpu::act_form f, int64_t k) {
  int fi, slot;
  return act_kernel_(type, f, k, fi, slot) != nullptr;
}

// Every format's kernel for a form launches the same way: q8, a warp a row and four
// rows a block; h16, 16 rows a block of 4 warps.
inline bool own_nn::gemv_quant_act(int type, gpu::act_form f, gpu::span pa, gpu::span W,
                                   gpu::span y, int64_t n, int64_t k) {
  auto& c = context::get();
  int fi, slot;
  const char* name = act_kernel_(type, f, k, fi, slot);
  if (!c.ready || !pa.buf || n <= 0 || !name) return false;
  const act_layout L = act_layout_(f, k);
  char* pp = reinterpret_cast<char*>(context::off_(pa.buf, pa.off));
  float* pw = context::off_(W.buf, W.off);
  c.device_read_(pa.buf);
  c.device_read_(W.buf);
  c.device_write_(y.buf);
  CUfunction fn = c.cached_(c.gemv_act_fn[fi][slot], name);
  switch (f) {
    case gpu::act_form::q8:
      return c.launch_(fn, {((unsigned)n + 3) / 4}, {128}, 0, reinterpret_cast<signed char*>(pp),
                       reinterpret_cast<void*>(pp + L.meta), reinterpret_cast<unsigned char*>(pw),
                       context::off_(y.buf, y.off), (unsigned)n, (unsigned)k);
    case gpu::act_form::h16:
      return c.launch_(fn, {((unsigned)n + 15) / 16}, {128}, 0, reinterpret_cast<void*>(pp),
                       reinterpret_cast<void*>(pp + L.meta), reinterpret_cast<void*>(pp + L.scale),
                       reinterpret_cast<unsigned char*>(pw), context::off_(y.buf, y.off),
                       (unsigned)n, (unsigned)k);
    case gpu::act_form::f32: break;
  }
  return false;
}

inline bool own_nn::gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y, int64_t n,
                             int64_t k) {
  auto& c = context::get();
  const int fi = quant_fmt_(type, k);
  if (!c.ready || n <= 0 || fi < 0) return false;
  float* pa = context::off_(a.buf, a.off);
  float* pw = context::off_(W.buf, W.off);
  c.device_read_(a.buf);
  c.device_read_(W.buf);
  c.device_write_(y.buf);
  const quant_fmt& f = quant_fmts_[fi];
  return c.launch_(c.cached_(c.gemv_quant_fn[fi], f.gemv),
                   {((unsigned)n + f.gemv_rows - 1) / f.gemv_rows}, {f.gemv_threads},
                   f.gemv_rows * f.gemv_threads / 32 * sizeof(float), pa,
                   reinterpret_cast<unsigned char*>(pw),
                   context::off_(y.buf, y.off), (unsigned)n, (unsigned)k);
}

inline bool own_nn::rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows, int64_t T, int64_t D,
                              int64_t pos, const gpu::rope_freq& f, float eps) {
  const int64_t rot = f.rot_of(D);
  auto& c = context::get();
  // tl_rmsnorm_rope_<D>: a warp a row, for the head widths whose rotation pairs a lane
  // holds whole; other shapes are refused (gpu::rmsnorm_rope then runs rmsnorm and rope).
  const int e = D == 64 ? 0 : D == 128 ? 1 : D == 256 ? 2 : -1;
  if (!c.ready || e < 0 || rows <= 0 || T <= 0 || rot <= 0 || rot % 64 || rot > D || rows % T) return false;
  c.device_rmw_(x.buf);
  c.device_read_(w.buf);
  if (f.factors) c.device_read_(f.factors.buf);
  float* pf = f.factors ? context::off_(f.factors.buf, f.factors.off) : nullptr;
  static const char* const names[3] = {"tl_rmsnorm_rope_64", "tl_rmsnorm_rope_128", "tl_rmsnorm_rope_256"};
  return c.launch_(c.cached_(c.rmsnorm_rope_fn[e], names[e]), {(unsigned)((rows + 7) / 8)}, {256u}, 0,
                   context::off_(x.buf, x.off), context::off_(w.buf, w.off), (unsigned)rows, (unsigned)T,
                   (unsigned)pos, f.base, (unsigned)rot, eps, f.pos_scale, pf);
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
                                 int64_t n, float eps, int64_t rows, gpu::span wd) {
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::rmsnorm_params p{static_cast<uint32_t>(n), eps};
  if (wd.buf) {
    const gpu::arg args[] = {gpu::in(x),     gpu::in(delta), gpu::in(wd),    gpu::in(w),
                             gpu::out(xout), gpu::out(hout), gpu::out(hout16)};
    return c.dispatch_(c.cached_(c.norm_add_rmsnorm_h16_fn, "tl_norm_add_rmsnorm_h16"), args, 7,
                       &p, sizeof(p), gpu::policy::one_group_per_row(rows));
  }
  const gpu::arg args[] = {gpu::in(x),    gpu::in(delta), gpu::in(w),
                           gpu::out(xout), gpu::out(hout), gpu::out(hout16)};
  return c.dispatch_(c.cached_(c.add_rmsnorm_h16_fn, "tl_add_rmsnorm_h16"), args, 6, &p,
                     sizeof(p), gpu::policy::one_group_per_row(rows));
}
inline bool own_nn::glu_h16(gpu::glu_act act, gpu::span gu, gpu::span o16, int64_t ff,
                            int64_t rows) {
  auto& c = context::get();
  if (!c.ready) return false;
  int k = 0;
  const char* name = nullptr;
  switch (act) {
    case gpu::glu_act::silu: k = 0, name = "tl_swiglu_h16"; break;
    case gpu::glu_act::gelu_tanh: k = 1, name = "tl_geglu_h16"; break;
  }
  const gpu::glu_params p{static_cast<uint32_t>(ff)};
  const gpu::arg args[] = {gpu::in(gu), gpu::out(o16)};
  return c.dispatch_(c.cached_(c.glu_h16_fn[k], name), args, 2, &p, sizeof(p),
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

// The decode attention's blocks (attn_dec_core): a kv head and attn_dec_heads(D)
// of the q heads that share it, attn_dec_threads(D) threads. The kernel's
// attn_dec_heads<AD> and attn_dec_threads<AD> (where the measurements behind
// them are) are the same table, its ABI: a block derives its heads from it.
// rows is grid.x, the blocks one split has.
inline constexpr int64_t attn_dec_heads(int64_t D) { return D == 256 ? 1 : 2; }
inline constexpr unsigned attn_dec_threads(int64_t D) { return 32 * (D == 512 ? 4 : 8); }
inline unsigned attn_dec_rows(int64_t n_q_heads, int64_t n_kv_heads, int64_t D) {
  return static_cast<unsigned>(
      gpu::policy::kv_head_slices(n_q_heads, n_kv_heads, attn_dec_heads(D)));
}

// Split-KV split-count heuristic: how many ctx-splits make grid = rows×S fill
// the SMs (rows alone are a few blocks « 82 SMs). Shared by attn_decode
// (evaluated at the live ctx) and attn_decode_dpos (evaluated at max_ctx, so
// the CUDA-graph grid is pos-independent). Twice the fill (~4 blocks a SM); no
// split under 64 keys, a split at least 32 keys and a multiple of 4. Against
// 256 and 64 (the rule of the one-q-head-a-block form), the same shapes as
// attn_dec_core's took 128 to 512 cached tokens from 7-14 us a layer to 4-8;
// 16 keys a split or 128 to start were no better.
inline constexpr gpu::policy::split_rule attn_split_rule{2 * traits::fill_groups, 64, 32, 4};
inline unsigned attn_split_count(unsigned rows, int64_t ctx) {
  return static_cast<unsigned>(gpu::policy::split_parts(rows, ctx, attn_split_rule));
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
// up to a multiple of 4. The device-side attn_dpos_chunk in cuda_llm.cuh is the
// twin of attn_split_count + this rounding and MUST stay in lockstep — the
// host/dpos bit-identity rests on it. Guarded by the attn64 ctest's
// host-vs-dpos bit-equality sweep.
inline unsigned attn_split_chunk(unsigned rows, int64_t ctx) {
  return static_cast<unsigned>(gpu::policy::split_chunk(
      ctx, attn_split_count(rows, ctx), attn_split_rule));
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

// The head widths the decode-attention kernels are instantiated for, each with
// either cache dtype.
inline bool attn_decode_head_ok(int64_t D) {
  return D == 64 || D == 128 || D == 256 || D == 512;
}

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
  const unsigned rows = attn_dec_rows(n_q_heads, n_kv_heads, D);

  // Split ctx over gridDim.y so grid = rows×S fills the SMs (see
  // attn_split_count; chunk below is rounded to a multiple of 4).
  unsigned S = attn_split_count(rows, ctx);
  if (S == 1) {
    return c.launch_(c.attn_(context::attn_k::decode, D, kv_f16), {rows}, {attn_dec_threads(D)}, 0,
                     pq, pk, pv, po, uctx, kv_stride, group, scale);
  }

  unsigned chunk = attn_split_chunk(rows, ctx);
  S = (uctx + chunk - 1) / chunk;  // recompute after rounding
  size_t hs = (size_t)uh * S;
  CUdeviceptr scr = c.attn_scratch_(attn_partials::bytes(hs, D));
  if (!scr) return false;
  attn_partials p(reinterpret_cast<float*>(scr), hs);
  if (!c.launch_(c.attn_(context::attn_k::split, D, kv_f16), {rows, S}, {attn_dec_threads(D)}, 0, pq,
                 pk, pv, p.pm, p.pl, p.pacc, uctx, kv_stride, group, chunk, scale)) {
    return false;
  }
  return p.combine(c, po, uh, static_cast<unsigned>(D), S);
}

// M9 causal prefill attention: q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D]
// cache read over [0,pos0+T). Query p is at absolute position pos0+p and attends
// keys 0..pos0+p (with a window, its last `window` of them), so a long prompt can
// be run in chunks (and a later turn appended to a live cache). GQA via group.
// One block per (head, query tile); grid = (n_q_heads, ceil(T/tile)).
inline bool own_nn::attn_prefill(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span out, int64_t n_q_heads,
                              int64_t n_kv_heads, int64_t T, int64_t kv_max,
                              int64_t D, float scale, bool kv_f16,
                              int64_t pos0, int64_t window) {
  auto& c = context::get();
  // D 64 or 128; 256 and 512 for an f16 cache (an f32 one there takes
  // gpu::attn_prefill's GEMM form, which reads only f32).
  if (!c.ready || (D != 128 && D != 64 && !((D == 256 || D == 512) && kv_f16))) return false;
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
                   {attn_tile_threads}, context::attn_smem(context::attn_k::tiled, D, kv_f16), pq, pk,
                   pv, po, uT, kv_stride, group, scale, up0, static_cast<unsigned>(window));
}

// The same causal prefill on the tensor cores (tl_attn_prefill_tc_*), its
// operands rounded to f16 as staged (an f16 cache's K and V already are): head
// width 64, 128, 256 or 512, operands where the kernel declares them. Inference
// only (gpu::attn_prefill_f16).
inline bool own_nn::attn_prefill_f16(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span out, int64_t n_q_heads,
                                  int64_t n_kv_heads, int64_t T, int64_t kv_max,
                                  int64_t D, float scale, bool kv_f16,
                                  int64_t pos0, int64_t window) {
  auto& c = context::get();
  if (!c.ready || (D != 64 && D != 128 && D != 256 && D != 512)) return false;
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
  // A block's queries and threads: 4 warps of 16 queries each, or at 512
  // attn_tc_wide_mq tiles of 16 queries, 4 warps a tile splitting the dims.
  const unsigned uT = static_cast<unsigned>(T);
  const unsigned tiles = D == 512 ? context::attn_tc_wide_mq : 4u, bq = 16 * tiles;
  const unsigned threads = D == 512 ? 128 * tiles : 128u;
  return c.launch_(c.attn_(context::attn_k::tc, D, kv_f16),
                   {static_cast<unsigned>(n_q_heads), (uT + bq - 1) / bq}, {threads},
                   context::attn_smem(context::attn_k::tc, D, kv_f16), pq,
                   pk, pv, po, uT, static_cast<unsigned>(kv_max * D),
                   static_cast<unsigned>(n_q_heads / n_kv_heads), scale,
                   static_cast<unsigned>(pos0), static_cast<unsigned>(window));
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

// The chunked delta rule (tl_gated_delta_fused_<V>): dk 128, dv 128, 64 or 32,
// and at least 8 tokens (half a chunk); fewer take the per-token kernel, whose
// time grows with each token where this one's grows a chunk at a time. Measured
// against it (RTX 3090), the fused kernel is level at 7 tokens on 48 heads (V
// 128: 25.6 against 25.3 us, then ahead) and at 3 on 16 (V 32: 11.3 against
// 11.3); at 16 tokens it is 1.9x and 3.5x ahead.
inline bool own_nn::gated_delta_chunked(gpu::span qkv, gpu::span ab, int64_t ab_ld,
                                     gpu::span adt, gpu::span state, gpu::span o,
                                     int64_t H, int64_t dk, int64_t dv, int64_t T,
                                     float eps, int64_t Hk) {
  auto& c = context::get();
  if (!c.ready || dk != 128 || (dv != 128 && dv != 64 && dv != 32) || T < 8 || Hk <= 0 ||
      H % Hk)
    return false;
  // A head's value columns in as many slices (V >= 32 a block) as keep the
  // grid within a block an SM: a block's chunks run in order, so more blocks
  // than SMs would wait a whole pass for the rest.
  constexpr int64_t sms = traits::fill_groups / 2;
  int64_t slices = 1;
  while (dv / (2 * slices) >= 32 && H * 2 * slices <= sms) slices *= 2;
  const int64_t V = dv / slices;
  CUfunction f = c.gdn_fused_(V);
  float* pqkv = context::off_(qkv.buf, qkv.off);
  float* pab = context::off_(ab.buf, ab.off);
  float* padt = context::off_(adt.buf, adt.off);
  float* pst = context::off_(state.buf, state.off);
  float* po = context::off_(o.buf, o.off);
  // Refused before any residency change if qkv's rows are misaligned for float4.
  if (!f || !c.fits_(f, pqkv, pab, padt, pst, po)) return false;
  c.device_read_(qkv.buf);
  c.device_read_(ab.buf);
  c.device_read_(adt.buf);
  c.device_rmw_(state.buf);
  c.device_write_(o.buf);
  return c.launch_(f, {(unsigned)(H * slices)}, {context::gdn_threads}, context::gdn_smem(V),
                   pqkv, pab, padt, pst, po, (unsigned)H, (unsigned)dv, (unsigned)T,
                   (unsigned)ab_ld, eps, (unsigned)Hk);
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
inline bool own_nn::rope(gpu::span x, gpu::span out, int64_t rows, int64_t T, int64_t D,
                         int64_t pos, const gpu::rope_freq& f, gpu::span bias) {
  auto& c = context::get();
  const int64_t rot = f.rot_of(D);
  if (!c.ready || D <= 0 || (D & 1) || rot <= 0 || (rot & 1) || rot > D) return false;
  c.device_read_(x.buf);
  if (bias.buf) c.device_read_(bias.buf);
  if (f.factors) c.device_read_(f.factors.buf);
  c.device_write_(out.buf);
  float* px = context::off_(x.buf, x.off);
  float* pbias = bias.buf ? context::off_(bias.buf, bias.off) : nullptr;
  float* pf = f.factors ? context::off_(f.factors.buf, f.factors.off) : nullptr;
  float* po = context::off_(out.buf, out.off);
  unsigned uT = static_cast<unsigned>(T), uD = static_cast<unsigned>(D),
           upos = static_cast<unsigned>(pos), urot = static_cast<unsigned>(rot);
  return c.launch_(c.rope_(), {static_cast<unsigned>(rows)},
                   {static_cast<unsigned>(D / 2)}, 0, px, pbias, po, uT, uD,
                   upos, f.base, urot, f.pos_scale, pf);
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
