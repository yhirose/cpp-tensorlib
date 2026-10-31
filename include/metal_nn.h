#pragma once

// The Metal backend's nn layer (layers: gpu_abi.h): its own ops for the fused
// ops over plain tensors (layer norm's pullback, RoPE, attention and its
// pullback, the GEMVs, the bf16 GEMM, split_heads) and their launch helpers.
//
// Included by metal.h, after the device core.

#include "gpu_abi_nn.h"

namespace tl {
namespace metal {

// This layer's kernels that are neither a kop nor one of metal.h's families
// (gtile, attn_kernel_): keyed below attn's range in the one pipeline cache, so
// a new nn kernel joins this list rather than starting another family.
enum class nn_kernel {
  gemv_q4k, gemv_q6k, gemv_q80,
  gemm_q4k, gemm_q4k_32, gemm_q6k, gemm_q6k_32, gemm_q80, gemm_q80_32,
  rmsnorm_rope_64, rmsnorm_rope_128, rmsnorm_rope_256, gated_rmsnorm,
  gemm_bf16_steel, gemm_bf16_steel_32, gemm_bf16_f16acc, gemm_bf16_f16acc_32,
};
constexpr int nn_kernel_key_(nn_kernel t) { return -201 - static_cast<int>(t); }
inline const char* nn_kernel_name_(nn_kernel t) {
  switch (t) {
    case nn_kernel::gemv_q4k: return "gemv_q4k_";
    case nn_kernel::gemv_q6k: return "gemv_q6k_";
    case nn_kernel::gemv_q80: return "gemv_q80_";
    case nn_kernel::gemm_q4k: return "gemm_q4k_nt_";
    case nn_kernel::gemm_q4k_32: return "gemm_q4k_nt32_";
    case nn_kernel::gemm_q6k: return "gemm_q6k_nt_";
    case nn_kernel::gemm_q6k_32: return "gemm_q6k_nt32_";
    case nn_kernel::gemm_q80: return "gemm_q80_nt_";
    case nn_kernel::gemm_q80_32: return "gemm_q80_nt32_";
    case nn_kernel::rmsnorm_rope_64: return "rmsnorm_rope_64_";
    case nn_kernel::rmsnorm_rope_128: return "rmsnorm_rope_128_";
    case nn_kernel::rmsnorm_rope_256: return "rmsnorm_rope_256_";
    case nn_kernel::gated_rmsnorm: return "gated_rmsnorm_";
    case nn_kernel::gemm_bf16_steel: return "gemm_bf16_steel_";
    case nn_kernel::gemm_bf16_steel_32: return "gemm_bf16_steel32_";
    case nn_kernel::gemm_bf16_f16acc: return "gemm_bf16_f16acc_";
    case nn_kernel::gemm_bf16_f16acc_32: return "gemm_bf16_f16acc32_";
  }
  return "";
}
static_assert(attn_key_last_ > nn_kernel_key_(nn_kernel::gemv_q4k),
              "attn and nn_kernel pipeline keys must stay in disjoint ranges");

namespace detail_ {
inline void bind_(context& c, nn_kernel t) {
  c.bind_id_(nn_kernel_key_(t), nn_kernel_name_(t));
}
}  // namespace detail_

struct own_nn {
  static bool split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                          int64_t T, int64_t ld, int64_t off, int64_t H,
                          int64_t D);
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
  static bool attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span dO, gpu::span O, gpu::span dq,
                              gpu::span stats, int64_t H, int64_t T, int64_t D,
                              float scale);
  static bool attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                               gpu::span dO, gpu::span stats, gpu::span dK,
                               gpu::span dV, int64_t H, int64_t T, int64_t D,
                               float scale);
  static bool gemm_bf16_nt(gpu::span A, gpu::span B, gpu::span C, int64_t M,
                           int64_t N, int64_t K);
  static bool gemm_bf16_nt_f16acc(gpu::span A, gpu::span B, gpu::span C,
                                  int64_t M, int64_t N, int64_t K);
  static bool gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y,
                         int64_t n, int64_t k);
  static bool gemm_quant(int type, bool a16, gpu::span A, gpu::span W,
                         gpu::span C, int64_t m, int64_t n, int64_t k,
                         int64_t ldc, bool mmq);
  static bool rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                   int64_t D, int64_t pos, float base, gpu::span bias = {});
  static bool rope_partial(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                           int64_t D, int64_t pos, float base, gpu::span bias,
                           int64_t rot);
  static bool rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows, int64_t T,
                           int64_t D, int64_t pos, float base, int64_t rot,
                           float eps);
  static bool gated_rmsnorm(gpu::span o, gpu::span z, int64_t ldz, gpu::span w,
                            gpu::span out, bool h16, int64_t T, int64_t H,
                            int64_t D, float eps);
  static bool layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                             gpu::span dx, gpu::span dg, gpu::span db,
                             gpu::span stats, gpu::span partials, int64_t rows,
                             int64_t cols, int64_t per_chunk, int64_t chunks,
                             float eps);
};

namespace detail_ {
struct layer_norm_bwd_params {
  uint32_t rows, cols, rows_per_chunk, chunks;
  float eps;
};
}  // namespace detail_

// Layer norm's pullback: cuda.h's contract and its three launches -- a row
// kernel (dx and the row stats), a column-strip kernel, a fold.
inline bool own_nn::layer_norm_bwd(gpu::span x, gpu::span g, gpu::span dy,
                                gpu::span dx, gpu::span dg, gpu::span db,
                                gpu::span stats, gpu::span partials,
                                int64_t rows, int64_t cols, int64_t per_chunk,
                                int64_t chunks, float eps) {
  auto& c = context::get();
  if (!c.device || rows <= 0 || cols <= 0 || chunks <= 0) return false;
  detail_::layer_norm_bwd_params p{
      static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
      static_cast<uint32_t>(per_chunk), static_cast<uint32_t>(chunks), eps};
  const auto ur = static_cast<unsigned long>(rows);
  const auto uc = static_cast<unsigned long>(cols);
  const auto uk = static_cast<unsigned long>(chunks);

  c.bind_(kop::layer_norm_bwd_dx_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, g, 1ul);
  detail_::set_buf_(c.enc, dy, 2ul);
  detail_::set_buf_(c.enc, dx, 3ul);
  detail_::set_buf_(c.enc, stats, 4ul);
  detail_::set_bytes_(c.enc, p, 5ul);
  detail_::dispatch_grid_(c.enc, {ur, 1, 1}, {256, 1, 1});

  c.bind_(kop::layer_norm_bwd_gb_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, dy, 1ul);
  detail_::set_buf_(c.enc, stats, 2ul);
  detail_::set_buf_(c.enc, partials, 3ul);
  detail_::set_bytes_(c.enc, p, 4ul);
  detail_::dispatch_grid_(c.enc, {(uc + 31) / 32, uk, 1}, {32, 8, 1});

  c.bind_(kop::layer_norm_bwd_gb_fold_);
  detail_::set_buf_(c.enc, partials, 0ul);
  detail_::set_buf_(c.enc, dg, 1ul);
  detail_::set_buf_(c.enc, db, 2ul);
  detail_::set_bytes_(c.enc, p, 3ul);
  detail_::dispatch_grid_(c.enc, {(uc + 255) / 256, 1, 1}, {256, 1, 1});
  return true;
}

namespace detail_ {
struct rope_params {
  uint32_t T, D, pos;
  uint32_t half_;
  float base;
  uint32_t n;
  uint32_t has_bias;
  uint32_t rot;
};
struct rmsnorm_rope_params {
  uint32_t rows, T, pos;
  float base;
  uint32_t rot;
  float eps;
};
struct gated_rmsnorm_params {
  uint32_t rows, H, D, ldz;
  float eps;
};
}  // namespace detail_

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors tensorlib_cuda.cu's own tl_rope exactly. x is
// [rows, D] contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D] with
// T=1); row r's head-dim vector sits at position `pos + (r % T)`; pairs
// (j, j+D/2) rotate by angle = position * base^(-2j/D). `bias`, when given,
// is [rows, D] added before the rotation (a decode step's q/k bias folded
// in; array.h's gpu_rope_ never passes one). Dispatched flat over rows*(D/2)
// (CUDA instead grids by row, blocks by D/2 -- this file's own kernels are
// all flat-1D, same as sum_to/compare/unary_ext above).
// rope and rope_partial are one kernel: the first `rot` dims turn (rot == D
// for the whole row) and the rest are copied through, as cuda's tl_rope.
inline bool own_nn::rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                      int64_t D, int64_t pos, float base, gpu::span bias) {
  return rope_partial(x, out, rows, T, D, pos, base, bias, D);
}
inline bool own_nn::rope_partial(gpu::span x, gpu::span out, int64_t rows,
                              int64_t T, int64_t D, int64_t pos, float base,
                              gpu::span bias, int64_t rot) {
  auto& c = context::get();
  if (!c.device || D <= 0 || (D & 1) || rot <= 0 || (rot & 1) || rot > D)
    return false;
  int64_t half = D / 2;
  int64_t n = rows * half;
  c.bind_(kop::rope_);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::set_buf_(c.enc, bias ? bias : x, 3ul);  // a slot must be bound
  detail_::rope_params p{};
  p.T = static_cast<uint32_t>(T);
  p.D = static_cast<uint32_t>(D);
  p.pos = static_cast<uint32_t>(pos);
  p.half_ = static_cast<uint32_t>(half);
  p.base = base;
  p.n = static_cast<uint32_t>(n);
  p.has_bias = bias.buf ? 1u : 0u;
  p.rot = static_cast<uint32_t>(rot);
  detail_::set_bytes_(c.enc, p, 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// x[rows, D] = rope(rmsnorm(x) * w) in place, a simdgroup a row: cuda.h's
// contract and shapes (D 64, 128 or 256; rot a multiple of 64), rmsnorm_'s
// sum and rope_'s turn. Other shapes are refused, and gpu::rmsnorm_rope runs
// the two ops.
inline bool own_nn::rmsnorm_rope(gpu::span x, gpu::span w, int64_t rows,
                              int64_t T, int64_t D, int64_t pos, float base,
                              int64_t rot, float eps) {
  auto& c = context::get();
  if (!c.device || (D != 64 && D != 128 && D != 256) || rows <= 0 || T <= 0 ||
      rot <= 0 || rot % 64 || rot > D || rows % T)
    return false;
  detail_::bind_(c, D == 64    ? nn_kernel::rmsnorm_rope_64
                    : D == 128 ? nn_kernel::rmsnorm_rope_128
                               : nn_kernel::rmsnorm_rope_256);
  detail_::set_buf_(c.enc, x, 0ul);
  detail_::set_buf_(c.enc, w, 1ul);
  detail_::rmsnorm_rope_params p{static_cast<uint32_t>(rows), static_cast<uint32_t>(T),
                                 static_cast<uint32_t>(pos), base,
                                 static_cast<uint32_t>(rot), eps};
  detail_::set_bytes_(c.enc, p, 2ul);
  detail_::dispatch_grid_(c.enc, {(static_cast<unsigned long>(rows) + 7) / 8, 1, 1},
                          {256, 1, 1});
  return true;
}

// Qwen3.5's silu(z) * rmsnorm(o) * w over o [T*H, D], a simdgroup a row:
// cuda.h's contract in f32 (h16, the f16 activations', is refused: this
// backend has no f16_activations, so the caller never asks).
inline bool own_nn::gated_rmsnorm(gpu::span o, gpu::span z, int64_t ldz,
                               gpu::span w, gpu::span out, bool h16, int64_t T,
                               int64_t H, int64_t D, float eps) {
  auto& c = context::get();
  if (!c.device || h16 || T <= 0 || H <= 0 || D <= 0) return false;
  detail_::bind_(c, nn_kernel::gated_rmsnorm);
  detail_::set_buf_(c.enc, o, 0ul);
  detail_::set_buf_(c.enc, z, 1ul);
  detail_::set_buf_(c.enc, w, 2ul);
  detail_::set_buf_(c.enc, out, 3ul);
  const int64_t rows = T * H;
  detail_::gated_rmsnorm_params p{static_cast<uint32_t>(rows), static_cast<uint32_t>(H),
                                  static_cast<uint32_t>(D), static_cast<uint32_t>(ldz),
                                  eps};
  detail_::set_bytes_(c.enc, p, 4ul);
  detail_::dispatch_grid_(c.enc, {(static_cast<unsigned long>(rows) + 7) / 8, 1, 1},
                          {256, 1, 1});
  return true;
}

// ---- causal prefill attention and its pullback ------------------------------
// cuda.h's contracts. 128 threads per threadgroup, owning 32 rows (queries,
// or keys for the key/value half): the MSL kernels' 4 simdgroups × 8.

namespace detail_ {
struct attn_params {
  uint32_t T, kv_stride, group, pos0;
  float scale;
};

struct gemv_params {
  uint32_t n, k, chunk;
};

struct gemv_combine_params {
  uint32_t n, parts;
};

// A bf16 W the vectorized STEEL loader reads (SteelLoaderT's 16-byte runs of
// eight): K whole runs, W word_aligned_.
inline bool bf16_rows_vec_(int64_t K, gpu::span W) { return K % 8 == 0 && word_aligned_(W); }

// The quantized GEMVs' threadgroup: gemv_quant_sg_ simdgroups, each the
// kernels' QROWS rows (metal_kernels_nn.metal).
inline constexpr unsigned long gemv_quant_qrows_ = 2, gemv_quant_sg_ = 2,
                               gemv_quant_rows_ = gemv_quant_sg_ * gemv_quant_qrows_;

// The quantized weights this backend reads natively: ggml type codes 12 Q4_K,
// 14 Q6_K and 8 Q8_0, K a whole number of their blocks, and W aligned for the
// kernels' loads (32-bit into a Q4_K block, so word_aligned_; 16-bit into the
// others).
inline bool quant_ok_(int type, int64_t k, gpu::span W) {
  return (type == 12 || type == 14 || type == 8) && k > 0 &&
         k % (type == 8 ? 32 : 256) == 0 && (type == 12 ? word_aligned_(W) : W.off % 2 == 0);
}

// An NT GEMM's launch on STEEL's body (C[M,N] rows ldc apart, W [N,K] the
// transposed B), tiled by g, after the caller has bound g's band's kernel.
inline void gemm_nt_launch_(context& c, const steel_grid& g, gpu::span A, gpu::span W,
                            gpu::span C, int64_t M, int64_t N, int64_t K, int64_t ldc) {
  const auto u = [](int64_t v) { return static_cast<uint32_t>(v); };
  const gemm_params p{u(M), u(N), u(K), u(K), u(K), 0u, 1u, g.swizzle_log, 0u,
                      1.0f, 0.0f, 0u, 0u, u(ldc)};
  gemm_launch_(c, A, W, C, p, g.gx, g.gy, 1);
}

struct attn_decode_params {
  uint32_t ctx, kv_stride, group, chunk;
  float scale;
};

struct attn_combine_params {
  uint32_t splits;
};

// Keys per split, or 0 for the single-pass kernel: one threadgroup a head
// leaves most of the GPU idle when a model has few heads, so cut the keys
// until there are enough threadgroups. The kernel's rule: no split under 256
// keys, a split at least 128 keys and a multiple of 4.
inline unsigned long attn_split_chunk_(int64_t heads, int64_t ctx) {
  constexpr gpu::policy::split_rule rule{traits::fill_groups, 256, 128, 4};
  const int64_t parts = gpu::policy::split_parts(heads, ctx, rule);
  if (parts <= 1) return 0;
  return static_cast<unsigned long>(gpu::policy::split_chunk(ctx, parts, rule));
}

inline void attn_dispatch_(objc::id enc, const attn_params& p,
                           unsigned long idx, int64_t heads, int64_t T) {
  constexpr unsigned long rows = 32;
  set_bytes_(enc, p, idx);
  dispatch_grid_(enc,
                 {static_cast<unsigned long>(heads),
                  (static_cast<unsigned long>(T) + rows - 1) / rows, 1},
                 {128, 1, 1});
}
}  // namespace detail_

// q,out [n_q_heads,T,D]; K/V a [n_kv_heads,kv_max,D] cache read over
// [0,pos0+T), query p at absolute position pos0+p. GQA via the head ratio.
// The head widths and caches attn_kernel_ names (cuda's: 64 and 128 over
// either cache, 256 over an f16 one). An f16 cache is staged four halves at a
// time (load4_'s half4) without a word_aligned_ check: no composition takes an
// f16 cache, so a refusal would leave the op unrun, and on the M1 Pro these
// loads read a cache one half off right (test_nn checks it).
inline bool own_nn::attn_prefill(gpu::span q, gpu::span K, gpu::span V,
                              gpu::span out, int64_t n_q_heads,
                              int64_t n_kv_heads, int64_t T, int64_t kv_max,
                              int64_t D, float scale, bool kv_f16,
                              int64_t pos0) {
  auto& c = context::get();
  const attn_kernel k = attn_kernel_(attn_op::prefill, D, kv_f16);
  if (!c.device || !k) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 || T <= 0) return false;
  c.bind_(k);
  detail_::set_buf_(c.enc, q, 0ul);
  detail_::set_buf_(c.enc, K, 1ul);
  detail_::set_buf_(c.enc, V, 2ul);
  detail_::set_buf_(c.enc, out, 3ul);
  detail_::attn_params p{static_cast<uint32_t>(T),
                         static_cast<uint32_t>(kv_max * D),
                         static_cast<uint32_t>(n_q_heads / n_kv_heads),
                         static_cast<uint32_t>(pos0), scale};
  detail_::attn_dispatch_(c.enc, p, 4ul, n_q_heads, T);
  return true;
}

// y[1,N] = a[1,K] · B[K,N] with f32 or bf16 weights, all contiguous: the
// decode projection. A narrow layer's N/256 threadgroups leave the GPU idle,
// so K is split across the grid's y and a combine pass sums the slices. The
// kernel's rule: a slice is whole 256-wide `a` tiles.
inline bool gemv_(kop op, gpu::span a, gpu::span B, gpu::span y, int64_t n,
                  int64_t k) {
  auto& c = context::get();
  if (!c.device || n <= 0 || k <= 0) return false;
  constexpr unsigned long NT = 256;
  constexpr gpu::policy::split_rule rule{traits::fill_groups, 0, 0, NT};
  const unsigned long cols = (static_cast<unsigned long>(n) + NT - 1) / NT;
  unsigned long chunk = static_cast<unsigned long>(gpu::policy::split_chunk(
      k, gpu::policy::split_parts(cols, k, rule), rule));
  unsigned long parts = (static_cast<unsigned long>(k) + chunk - 1) / chunk;
  gpu::span out = y;
  if (parts > 1) {
    out = {detail_::scratch_(static_cast<int64_t>(parts) * n * 4), 0};
    if (!out) parts = 1, chunk = static_cast<unsigned long>(k), out = y;
  }
  c.bind_(op);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, B, 1ul);
  detail_::set_buf_(c.enc, out, 2ul);
  detail_::gemv_params p{static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                         static_cast<uint32_t>(chunk)};
  detail_::set_bytes_(c.enc, p, 3ul);
  detail_::dispatch_grid_(c.enc, {cols, parts, 1}, {NT, 1, 1});
  if (parts == 1) return true;
  c.bind_(kop::gemv_combine_);
  detail_::set_buf_(c.enc, out, 0ul);
  detail_::set_buf_(c.enc, y, 1ul);
  detail_::gemv_combine_params cp{static_cast<uint32_t>(n),
                                  static_cast<uint32_t>(parts)};
  detail_::set_bytes_(c.enc, cp, 2ul);
  detail_::dispatch_grid_(c.enc, {cols, 1, 1}, {NT, 1, 1});
  return true;
}

inline bool own_nn::gemv_f32(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                          int64_t k) {
  return gemv_(kop::gemv_f32_, a, B, y, n, k);
}

inline bool own_nn::gemv_bf16(gpu::span a, gpu::span B, gpu::span y, int64_t n,
                           int64_t k) {
  return gemv_(kop::gemv_bf16_, a, B, y, n, k);
}

// C[M,N] = A[M,K] · B[N,K]ᵀ, B bf16: the batched prefill's projection, where
// one weight serves a whole chunk of prompt tokens. STEEL's body (the f32
// gemm's, metal.h), its vectorized loader where bf16_rows_vec_ holds, the bf16
// reader otherwise.
inline bool own_nn::gemm_bf16_nt(gpu::span A, gpu::span B, gpu::span C, int64_t M,
                              int64_t N, int64_t K) {
  auto& c = context::get();
  if (!c.device || M <= 0 || N <= 0 || K <= 0) return false;
  const detail_::steel_grid g = detail_::steel_grid_(M, N);
  if (detail_::bf16_rows_vec_(K, B)) {
    detail_::bind_(c, g.band32 ? nn_kernel::gemm_bf16_steel_32 : nn_kernel::gemm_bf16_steel);
  } else {
    c.bind_(g.band32 ? kop::gemm_bf16_nt32_ : kop::gemm_bf16_nt_);
  }
  detail_::gemm_nt_launch_(c, g, A, B, C, M, N, K, N);
  return true;
}

// The same summed in f16, cuda.h's contract: STEEL's body over half tiles,
// for a W the vectorized loader reads (bf16_rows_vec_); otherwise
// false, and the caller's f32 GEMM.
inline bool own_nn::gemm_bf16_nt_f16acc(gpu::span A, gpu::span B, gpu::span C,
                                        int64_t M, int64_t N, int64_t K) {
  auto& c = context::get();
  if (!c.device || M <= 0 || N <= 0 || K <= 0 || !detail_::bf16_rows_vec_(K, B))
    return false;
  const detail_::steel_grid g = detail_::steel_grid_(M, N);
  detail_::bind_(c, g.band32 ? nn_kernel::gemm_bf16_f16acc_32 : nn_kernel::gemm_bf16_f16acc);
  detail_::gemm_nt_launch_(c, g, A, B, C, M, N, K, N);
  return true;
}

// y[n] = W[n,K] . a over a GGUF block-quantized weight read where it lies:
// cuda.h's contract (the activation on 16 bytes, for the kernels' float4
// loads), in threadgroups of gemv_quant_rows_ rows.
inline bool own_nn::gemv_quant(int type, gpu::span a, gpu::span W, gpu::span y,
                               int64_t n, int64_t k) {
  auto& c = context::get();
  if (!c.device || n <= 0 || a.off % 16 || !detail_::quant_ok_(type, k, W)) return false;
  detail_::bind_(c, type == 12   ? nn_kernel::gemv_q4k
                    : type == 14 ? nn_kernel::gemv_q6k
                                 : nn_kernel::gemv_q80);
  detail_::set_buf_(c.enc, a, 0ul);
  detail_::set_buf_(c.enc, W, 1ul);
  detail_::set_buf_(c.enc, y, 2ul);
  gpu::gemv_row_params p{static_cast<uint32_t>(n), static_cast<uint32_t>(k)};
  detail_::set_bytes_(c.enc, p, 3ul);
  const unsigned long groups =
      (static_cast<unsigned long>(n) + detail_::gemv_quant_rows_ - 1) / detail_::gemv_quant_rows_;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {32 * detail_::gemv_quant_sg_, 1, 1});
  return true;
}

// C[m, 0..n) (rows ldc apart) = A[m,k] . W[n,k]ᵀ over a block-quantized
// weight: cuda.h's contract in its exact form, the weight dequantized to f32 as
// STEEL's body stages it (its reader). Only that form: a16 is refused (the caller
// never asks for it here, f16_activations being false on this backend), and
// mmq, which CUDA applies only with a16, is ignored.
inline bool own_nn::gemm_quant(int type, bool a16, gpu::span A, gpu::span W,
                               gpu::span C, int64_t m, int64_t n, int64_t k,
                               int64_t ldc, bool mmq) {
  (void)mmq;
  auto& c = context::get();
  if (!c.device || a16 || m <= 0 || n <= 0 || ldc < n ||
      !detail_::quant_ok_(type, k, W))
    return false;
  const detail_::steel_grid g = detail_::steel_grid_(m, n);
  const bool s = g.band32;
  detail_::bind_(c, type == 12   ? (s ? nn_kernel::gemm_q4k_32 : nn_kernel::gemm_q4k)
                    : type == 14 ? (s ? nn_kernel::gemm_q6k_32 : nn_kernel::gemm_q6k)
                                 : (s ? nn_kernel::gemm_q80_32 : nn_kernel::gemm_q80));
  detail_::gemm_nt_launch_(c, g, A, W, C, m, n, k, ldc);
  return true;
}

// One decode step: q [n_q_heads,D] against a [n_kv_heads,kv_max,D] cache read
// over [0,ctx). GQA via the head ratio; head width 64, 128 or 256 over either
// cache.
inline bool own_nn::attn_decode(gpu::span q, gpu::span K, gpu::span V,
                             gpu::span out, int64_t n_q_heads,
                             int64_t n_kv_heads, int64_t ctx, int64_t kv_max,
                             int64_t D, float scale, bool kv_f16) {
  auto& c = context::get();
  const attn_kernel whole = attn_kernel_(attn_op::decode, D, kv_f16);
  if (!c.device || !whole) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 || ctx <= 0) return false;
  unsigned long chunk = detail_::attn_split_chunk_(n_q_heads, ctx);
  const unsigned long splits =
      chunk ? (static_cast<unsigned long>(ctx) + chunk - 1) / chunk : 1;
  gpu::span dst = out;
  if (splits > 1) {
    // pm[H*S] | pl[H*S] | pacc[H*S*D], the layout attn_combine_ reads.
    const int64_t hs = n_q_heads * static_cast<int64_t>(splits);
    dst = {detail_::scratch_((hs * 2 + hs * D) * 4), 0};
    if (!dst) chunk = 0, dst = out;
  }
  const bool split = chunk != 0;
  c.bind_(split ? attn_kernel_(attn_op::decode_split, D, kv_f16) : whole);
  detail_::set_buf_(c.enc, q, 0ul);
  detail_::set_buf_(c.enc, K, 1ul);
  detail_::set_buf_(c.enc, V, 2ul);
  detail_::set_buf_(c.enc, dst, 3ul);
  detail_::attn_decode_params p{static_cast<uint32_t>(ctx),
                                static_cast<uint32_t>(kv_max * D),
                                static_cast<uint32_t>(n_q_heads / n_kv_heads),
                                static_cast<uint32_t>(chunk), scale};
  detail_::set_bytes_(c.enc, p, 4ul);
  detail_::dispatch_grid_(c.enc,
                          {static_cast<unsigned long>(n_q_heads),
                           split ? splits : 1ul, 1},
                          {static_cast<unsigned long>(D), 1, 1});
  if (!split) return true;
  c.bind_(attn_kernel_(attn_op::combine, D, false));  // partials are f32
  detail_::set_buf_(c.enc, dst, 0ul);
  detail_::set_buf_(c.enc, out, 1ul);
  detail_::attn_combine_params cp{static_cast<uint32_t>(splits)};
  detail_::set_bytes_(c.enc, cp, 2ul);
  detail_::dispatch_grid_(c.enc, {static_cast<unsigned long>(n_q_heads), 1, 1},
                          {static_cast<unsigned long>(D), 1, 1});
  return true;
}

// ---- the KV cache's writes and the decode step's remaining kernels --------
// cuda.h's contracts: raw device buffers, so a model's step builds no graph.

// The query half of the pullback: q, K, V, dO, O and dq all [H,T,D]
// contiguous, `stats` [2,H,T] the row logsumexp and dO·O.
inline bool own_nn::attn_prefill_dq(gpu::span q, gpu::span K, gpu::span V,
                                 gpu::span dO, gpu::span O, gpu::span dq,
                                 gpu::span stats, int64_t H, int64_t T,
                                 int64_t D, float scale) {
  auto& c = context::get();
  const attn_kernel k = attn_kernel_(attn_op::bwd_dq, D, false);
  if (!c.device || !k || H <= 0 || T <= 0) return false;
  c.bind_(k);
  const gpu::span views[] = {q, K, V, dO, O, dq, stats};
  for (unsigned long i = 0; i < 7; i++) detail_::set_buf_(c.enc, views[i], i);
  detail_::attn_params p{static_cast<uint32_t>(T), 0, 0, 0, scale};
  detail_::attn_dispatch_(c.enc, p, 7ul, H, T);
  return true;
}

// The key/value half, reading the stats the call above wrote.
inline bool own_nn::attn_prefill_dkv(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span dO, gpu::span stats, gpu::span dK,
                                  gpu::span dV, int64_t H, int64_t T, int64_t D,
                                  float scale) {
  auto& c = context::get();
  const attn_kernel k = attn_kernel_(attn_op::bwd_dkv, D, false);
  if (!c.device || !k || H <= 0 || T <= 0) return false;
  c.bind_(k);
  const gpu::span views[] = {q, K, V, dO, stats, dK, dV};
  for (unsigned long i = 0; i < 7; i++) detail_::set_buf_(c.enc, views[i], i);
  detail_::attn_params p{static_cast<uint32_t>(T), 0, 0, 0, scale};
  detail_::attn_dispatch_(c.enc, p, 7ul, H, T);
  return true;
}

// "No bias" is a flag here: a kernel cannot test a buffer for null, so an
// absent bias binds src in its place and is never read.
inline bool own_nn::split_heads(gpu::span src, gpu::span bias, gpu::span dst,
                        int64_t T, int64_t ld, int64_t off, int64_t H,
                        int64_t D) {
  struct {
    uint32_t T, ld, off, has_bias;
  } p{static_cast<uint32_t>(T), static_cast<uint32_t>(ld),
      static_cast<uint32_t>(off), bias ? 1u : 0u};
  const gpu::arg args[] = {gpu::in(src), gpu::in(bias ? bias : src),
                           gpu::out(dst)};
  return dispatch(kop::split_heads_, args, 3, &p, sizeof(p),
                  gpu::policy::per_head(H, T, D));
}

}  // namespace metal
}  // namespace tl
