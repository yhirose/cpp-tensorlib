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
EW_BINARY(pow_, pow(a[i], b[i]))

// Rank-2 broadcast binary: out[r,c] = f(a[r*ars + c*acs], b[r*brs + c*bcs]).
// Per-operand strides express every rank-2 broadcast (row vector, column
// vector, scalar) in one kernel, so bias/gamma/beta ops stay on the GPU
// instead of falling back to the CPU mid-graph (each fallback costs a full
// pipeline flush). Output is contiguous row-major [M, N].
struct ew_bcast_params {
  float scale;
  float offset;
  uint M;
  uint N;
  uint ars, acs, brs, bcs;
};

#define EW_BCAST(name, expr)                                     \
  kernel void name(device const float* a [[buffer(0)]],          \
                   device const float* b [[buffer(1)]],          \
                   device float* out [[buffer(2)]],              \
                   constant ew_bcast_params& p [[buffer(3)]],    \
                   uint2 g [[thread_position_in_grid]]) {        \
    if (g.x >= p.N || g.y >= p.M) return;                        \
    float av = a[g.y * p.ars + g.x * p.acs];                     \
    float bv = b[g.y * p.brs + g.x * p.bcs];                     \
    out[g.y * p.N + g.x] = fma(expr, p.scale, p.offset);         \
  }

EW_BCAST(badd_, av + bv)
EW_BCAST(bsub_, av - bv)
EW_BCAST(bmul_, av * bv)
EW_BCAST(bdiv_, av / bv)
EW_BCAST(bpow_, pow(av, bv))

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
EW_UNARY(tanh_, tanh(a[i]))
EW_UNARY(sin_, sin(a[i]))
EW_UNARY(cos_, cos(a[i]))

// Elementwise comparison: out = (a OP b) ? 1.0 : 0.0 (no scale/offset --
// masks don't compose with the affine epilogue). `bstride` is 1 for a
// same-shape `b` and 0 for an explicit size-1 `b` (`x > s` itself is gt_s_
// below) -- mirrors tensorlib_cuda.cu's TL_EW_CMP exactly.
struct cmp_params {
  uint n;
  uint bstride;
};

#define EW_CMP(name, expr)                                     \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device const float* b [[buffer(1)]],        \
                   device float* out [[buffer(2)]],            \
                   constant cmp_params& p [[buffer(3)]],       \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    float bv = b[i * p.bstride];                               \
    out[i] = (expr) ? 1.0f : 0.0f;                              \
  }

EW_CMP(gt_, a[i] > bv)
EW_CMP(lt_, a[i] < bv)
EW_CMP(ge_, a[i] >= bv)
EW_CMP(le_, a[i] <= bv)
EW_CMP(eq_, a[i] == bv)
EW_CMP(ne_, a[i] != bv)
#undef EW_CMP

// clamp(x, lo, hi): Clip's forward. No epilogue -- lo/hi occupy the role
// scale/offset play elsewhere.
struct clamp_params {
  float lo, hi;
  uint n;
};

kernel void clamp_(device const float* a [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   constant clamp_params& p [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = clamp(a[i], p.lo, p.hi);
}

// Tensor-scalar: out = f(a, s) * scale + offset -- mirrors tensorlib_cuda.cu's
// TL_EW_SCALAR.
struct scalar_params {
  float s, scale, offset;
  uint n;
};

#define EW_SCALAR(name, expr)                                  \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device float* out [[buffer(1)]],            \
                   constant scalar_params& p [[buffer(2)]],    \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    out[i] = fma(expr, p.scale, p.offset);                     \
  }

EW_SCALAR(pow_s_, pow(a[i], p.s))
EW_SCALAR(gt_s_, a[i] > p.s ? 1.0f : 0.0f)
EW_SCALAR(lt_s_, a[i] < p.s ? 1.0f : 0.0f)
EW_SCALAR(ge_s_, a[i] >= p.s ? 1.0f : 0.0f)
EW_SCALAR(le_s_, a[i] <= p.s ? 1.0f : 0.0f)
EW_SCALAR(eq_s_, a[i] == p.s ? 1.0f : 0.0f)
EW_SCALAR(ne_s_, a[i] != p.s ? 1.0f : 0.0f)
#undef EW_SCALAR

// ---------------------------------------------------------------------------
// Tiled SGEMM — simdgroup_matrix 8×8 MMA. One shared template body
// (sgemm_body_) instantiated at 32×32×16 and 64×64×16; the 64 band is the
// STEEL-class configuration (4 simdgroups, 4×4 fragments each). Loaders:
// float4-vectorized fast path when the host verified eligibility (no
// transpose, ld % 4 == 0, 16B-aligned base) and the tile is interior;
// bounds-checked strided path otherwise — trans_a/trans_b read transposed
// views in place (silarray: materializing a transpose gives back the win).
// The affine epilogue (out = A@B * scale + offset) is fused into the store;
// interior no-epilogue fragments take the direct simdgroup_store, everything
// else bounces through threadgroup scratch. Both store paths live in this
// single body — never copy-paste a kernel variant (edge-tile bug class).
// ---------------------------------------------------------------------------

struct gemm_params {
  uint M, N, K, lda, ldb, trans_a, trans_b;
  uint a_fast, b_fast;  // host-verified float4 eligibility
  float scale, offset;
};

template <uint BM, uint BN, uint BK>
void sgemm_body_(device const float* A, device const float* B,
                 device float* C, constant gemm_params& p,
                 threadgroup float* As, threadgroup float* Bs,
                 uint3 tgid, uint tid, uint sid, uint lane) {
  constexpr uint N_SM = 2, N_SN = 2, TM = BM / N_SM, TN = BN / N_SN;
  constexpr uint FM = TM / 8, FN = TN / 8, THREADS = N_SM * N_SN * 32;
  constexpr uint aS = BK + 4, bS = BN + 4;  // padding kills bank conflicts

  uint wm = sid / N_SN, wn = sid % N_SN;
  simdgroup_matrix<float, 8, 8> acc[FM][FN];
  for (uint i = 0; i < FM; i++)
    for (uint j = 0; j < FN; j++)
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

  uint row0 = tgid.y * BM, col0 = tgid.x * BN;
  uint a_rs = p.trans_a ? 1u : p.lda, a_cs = p.trans_a ? p.lda : 1u;
  uint b_rs = p.trans_b ? 1u : p.ldb, b_cs = p.trans_b ? p.ldb : 1u;
  bool a_full = row0 + BM <= p.M, b_full = col0 + BN <= p.N;
  bool afast = p.a_fast && a_full, bfast = p.b_fast && b_full;
  uint k_full = (p.K / BK) * BK;

#define TL_LOAD_A(k0)                                                        \
  if (afast) {                                                               \
    constexpr uint F4 = BK / 4;                                              \
    for (uint i = tid; i < BM * F4; i += THREADS) {                          \
      uint r = i / F4, fc = i % F4;                                          \
      auto v = *reinterpret_cast<device const float4*>(                      \
          &A[(row0 + r) * p.lda + (k0) + fc * 4]);                           \
      *reinterpret_cast<threadgroup float4*>(&As[r * aS + fc * 4]) = v;      \
    }                                                                        \
  } else {                                                                   \
    for (uint i = tid; i < BM * BK; i += THREADS) {                          \
      uint r = i / BK, c = i % BK, gr = row0 + r, gc = (k0) + c;             \
      As[r * aS + c] =                                                       \
          (gr < p.M && gc < p.K) ? A[gr * a_rs + gc * a_cs] : 0.0f;          \
    }                                                                        \
  }

#define TL_LOAD_B(k0)                                                        \
  if (bfast) {                                                               \
    constexpr uint F4 = BN / 4;                                              \
    for (uint i = tid; i < BK * F4; i += THREADS) {                          \
      uint r = i / F4, fc = i % F4;                                          \
      auto v = *reinterpret_cast<device const float4*>(                      \
          &B[((k0) + r) * p.ldb + col0 + fc * 4]);                           \
      *reinterpret_cast<threadgroup float4*>(&Bs[r * bS + fc * 4]) = v;      \
    }                                                                        \
  } else {                                                                   \
    for (uint i = tid; i < BK * BN; i += THREADS) {                          \
      uint r = i / BN, c = i % BN, gr = (k0) + r, gc = col0 + c;             \
      Bs[r * bS + c] =                                                       \
          (gr < p.K && gc < p.N) ? B[gr * b_rs + gc * b_cs] : 0.0f;          \
    }                                                                        \
  }

#define TL_MMA                                                               \
  for (uint kk = 0; kk < BK; kk += 8) {                                      \
    simdgroup_matrix<float, 8, 8> af[FM], bf[FN];                            \
    for (uint i = 0; i < FM; i++)                                            \
      simdgroup_load(af[i], &As[(wm * TM + i * 8) * aS + kk], aS);           \
    for (uint j = 0; j < FN; j++)                                            \
      simdgroup_load(bf[j], &Bs[kk * bS + wn * TN + j * 8], bS);             \
    for (uint i = 0; i < FM; i++)                                            \
      for (uint j = 0; j < FN; j++)                                          \
        simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);   \
  }

  for (uint k0 = 0; k0 < k_full; k0 += BK) {
    TL_LOAD_A(k0)
    TL_LOAD_B(k0)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TL_MMA
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k_full < p.K) {
    // K remainder: force the bounds-checked loaders regardless of fast flags.
    bool afast_saved = afast, bfast_saved = bfast;
    afast = false;
    bfast = false;
    TL_LOAD_A(k_full)
    TL_LOAD_B(k_full)
    afast = afast_saved;
    bfast = bfast_saved;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TL_MMA
    // The store reuses As as scratch — another simdgroup may still be
    // reading it for its MMA without this barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

#undef TL_LOAD_A
#undef TL_LOAD_B
#undef TL_MMA

  bool plain = (p.scale == 1.0f && p.offset == 0.0f);
  for (uint i = 0; i < FM; i++) {
    for (uint j = 0; j < FN; j++) {
      uint r = row0 + wm * TM + i * 8, c = col0 + wn * TN + j * 8;
      if (plain && r + 8 <= p.M && c + 8 <= p.N) {
        simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
      } else if (r < p.M && c < p.N) {
        // Scratch bounce: per-simdgroup region of As (4 × 64 floats).
        threadgroup float* sc = As;
        simdgroup_store(acc[i][j], &sc[sid * 64], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
          uint er = r + e / 8, ec = c + e % 8;
          if (er < p.M && ec < p.N)
            C[er * p.N + ec] = fma(sc[sid * 64 + e], p.scale, p.offset);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
}

#define TL_SGEMM_KERNEL(name, BM, BN, BK)                                    \
  kernel void name(device const float* A [[buffer(0)]],                      \
                   device const float* B [[buffer(1)]],                      \
                   device float* C [[buffer(2)]],                            \
                   constant gemm_params& p [[buffer(3)]],                    \
                   uint3 tgid [[threadgroup_position_in_grid]],              \
                   uint tid [[thread_index_in_threadgroup]],                 \
                   uint sid [[simdgroup_index_in_threadgroup]],              \
                   uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float As[BM * (BK + 4)];                                     \
    threadgroup float Bs[BK * (BN + 4)];                                     \
    sgemm_body_<BM, BN, BK>(A, B, C, p, As, Bs, tgid, tid, sid, lane);       \
  }

TL_SGEMM_KERNEL(sgemm_32_, 32, 32, 16)
TL_SGEMM_KERNEL(sgemm_32x64_, 32, 64, 16)
TL_SGEMM_KERNEL(sgemm_64x32_, 64, 32, 16)
TL_SGEMM_KERNEL(sgemm_64_, 64, 64, 16)

// ---------------------------------------------------------------------------
// STEEL SGEMM (ported from silarray, which ported it from MLX — proven at
// MLX parity on this machine class). Covers NN plus single-transposed
// operands (_ta_/_tb_ via the transposing loader); only TT falls back to
// the sgemm_32/64x32 family. Differences from the simple tile above
// that make 16 accumulators per simdgroup work (the naive 64×64 collapses):
// explicit float2 fragment registers (frag_type) instead of full
// simdgroup_matrix locals, loader pointers precomputed once (no div/mod in
// the K loop), serpentine MMA order, and a threadgroup swizzle for L2 reuse.
// Store epilogue is our affine (fma(acc, scale, offset)); interior/edge
// paths share one template (edge-tile bug class).
// ---------------------------------------------------------------------------

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

template <int N> struct Int { STEEL_CONST int value = N; constexpr operator int() const { return N; } };

typedef float2 frag_type;

// A lane's two elements of an 8×8 simdgroup_matrix: row frag_row_, columns
// frag_col_ and frag_col_ + 1.
static inline uint frag_row_(uint lane) {
  return ((lane / 4) & 4u) + ((lane / 2) % 4);
}
static inline uint frag_col_(uint lane) {
  return ((lane / 4) & 2u) * 2 + (lane % 2) * 2;
}

template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct SteelLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup float* dst;
  const device float* src;

  struct alignas(16) ReadVec { float v[vec_size]; };

  METAL_FUNC SteelLoader(const device float* src_, int src_ld_,
                         threadgroup float* dst_, ushort sid, ushort lane)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(sid * 32 + lane) / TCOLS),
        bj(vec_size * (short(sid * 32 + lane) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVec*)(&dst[i * dst_ld])) =
          *((const device ReadVec*)(&src[i * src_ld]));
    }
  }

  // Branchless: all threads issue the same loads (invalid ones read src[0])
  // then mask — avoids simd divergence on the load path (silarray-measured).
  METAL_FUNC void load_safe(short2 tile_dim) const {
    tile_dim -= short2(bj, bi);
    if (tile_dim.x <= 0 || tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS)
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = 0.0f;
      return;
    }
    bool ok[vec_size];
    float val[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        ok[j] = (i < tile_dim.y) && (j < tile_dim.x);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        val[j] = src[ok[j] ? i * src_ld + j : 0];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) val[j] = ok[j] ? val[j] : 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = val[j];
    }
  }

  // Row-clipped: full-width rows keep the vectorized load.
  METAL_FUNC void load_safe_rows(short valid_rows) const {
    valid_rows -= bi;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      if (i < valid_rows) {
        *((threadgroup ReadVec*)(&dst[i * dst_ld])) =
            *((const device ReadVec*)(&src[i * src_ld]));
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = 0.0f;
      }
    }
  }

  // Column-clipped: vectors fully inside keep the vectorized load.
  METAL_FUNC void load_safe_cols(short valid_cols) const {
    valid_cols -= bj;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      if (valid_cols >= vec_size) {
        *((threadgroup ReadVec*)(&dst[i * dst_ld])) =
            *((const device ReadVec*)(&src[i * src_ld]));
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = j < valid_cols ? src[i * src_ld + j] : 0.0f;
      }
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

// Transposing BlockLoader (silarray port) — reads a BROWS x BCOLS tile of the
// SOURCE (vectorized along its contiguous BCOLS dim) and stores it TRANSPOSED
// into threadgroup memory, so the MMA keeps its NN layout while the device
// operand stays in its original (transposed-view) storage. Thread mapping is
// bi = idx % BROWS (one src row per thread): consecutive lanes scatter to
// consecutive threadgroup banks, keeping the transposed stores at worst
// 2-way bank-conflicted. All instantiations cover the tile in one pass.
template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size>
struct SteelLoaderT {
  STEEL_CONST short vec_size = (BCOLS * BROWS) / tgp_size;
  static_assert(BCOLS == vec_size * (tgp_size / BROWS),
                "SteelLoaderT: tile must be covered in one pass");

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup float* dst;
  const device float* src;

  struct alignas(16) ReadVec { float v[vec_size]; };

  METAL_FUNC SteelLoaderT(const device float* src_, int src_ld_,
                          threadgroup float* dst_, ushort sid, ushort lane)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(sid * 32 + lane) % BROWS),
        bj(vec_size * (short(sid * 32 + lane) / BROWS)),
        dst(dst_ + bj * dst_ld + bi),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    ReadVec v = *((const device ReadVec*)(&src[0]));
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = v.v[j];
  }

  // tile_dim is in SOURCE coordinates (x = cols, y = rows), matching
  // SteelLoader::load_safe.
  METAL_FUNC void load_safe(short2 tile_dim) const {
    tile_dim -= short2(bj, bi);
    if (tile_dim.x <= 0 || tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = 0.0f;
      return;
    }
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) {
      bool valid = (j < tile_dim.x);
      dst[j * dst_ld] = valid ? src[valid ? j : 0] : 0.0f;
    }
  }

  // Row-clipped (trailing SOURCE rows out of bounds, all columns valid).
  METAL_FUNC void load_safe_rows(short valid_rows) const {
    if (bi < valid_rows) {
      load_unsafe();
    } else {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = 0.0f;
    }
  }

  // Column-clipped (trailing SOURCE cols out of bounds, all rows valid).
  METAL_FUNC void load_safe_cols(short valid_cols) const {
    valid_cols -= bj;
    if (valid_cols >= vec_size) {
      load_unsafe();
    } else {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        dst[j * dst_ld] = j < valid_cols ? src[j] : 0.0f;
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
};

// Compile-time type selection (MSL has no <type_traits>)
template <bool C, typename Then, typename Else>
struct select_type { using type = Then; };
template <typename Then, typename Else>
struct select_type<false, Then, Else> { using type = Else; };

template <short BM, short BN, short BK, short WM, short WN,
          short lda_tgp, short ldb_tgp>
struct SteelMMA {
  STEEL_CONST short kFrag = 8;
  STEEL_CONST short TM = BM / (kFrag * WM);
  STEEL_CONST short TN = BN / (kFrag * WN);

  STEEL_CONST short A_str_m = lda_tgp;
  STEEL_CONST short A_str_k = 1;
  STEEL_CONST short B_str_k = ldb_tgp;
  STEEL_CONST short B_str_n = 1;

  STEEL_CONST short tile_stride_a = kFrag * A_str_k;
  STEEL_CONST short tile_stride_b = kFrag * B_str_k;

  frag_type Atile[TM];
  frag_type Btile[TN];
  frag_type Ctile[TM * TN];

  short sm, sn;
  short As_off, Bs_off;

  METAL_FUNC SteelMMA(ushort sid, ushort lane) {
    short tm = kFrag * short(sid / WN);
    short tn = kFrag * short(sid % WN);

    short fm = short(frag_row_(lane));
    short fn = short(frag_col_(lane));

    sm = fm; sn = fn;
    As_off = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_off = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm; sn += tn;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM * TN; i++) Ctile[i] = frag_type(0);
  }

  METAL_FUNC static constexpr void load_frag_contig(
      thread frag_type& dst, const threadgroup float* src) {
    dst = *reinterpret_cast<const threadgroup frag_type*>(src);
  }
  METAL_FUNC static constexpr void load_frag_strided(
      thread frag_type& dst, const threadgroup float* src, short str) {
    dst[0] = src[0];
    dst[1] = src[str];
  }

  METAL_FUNC static constexpr void frag_mma(
      thread frag_type& D, thread frag_type& A,
      thread frag_type& B, thread frag_type& C) {
    simdgroup_matrix<float, 8, 8> A_mat, B_mat, C_mat;
    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;
    simdgroup_multiply_accumulate(C_mat, A_mat, B_mat, C_mat);
    D = reinterpret_cast<thread frag_type&>(C_mat.thread_elements());
  }

  // Per kk step: load one K-slice of fragments, immediately MMA, serpentine
  // over N to maximize Btile register reuse. sched_fences pins the compiler
  // schedule for edge tiles (see silarray notes); only the N-edge loop opts in.
  template <bool sched_fences = false>
  METAL_FUNC void mma(const threadgroup float* As, const threadgroup float* Bs) {
    As += As_off;
    Bs += Bs_off;
    constexpr short A_frag_stride = kFrag * WM * A_str_m;
    constexpr short B_frag_stride = kFrag * WN * B_str_n;

    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFrag) {
      if (sched_fences) simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++)
        load_frag_strided(Atile[i], &As[i * A_frag_stride], A_str_k);
      if (sched_fences) simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++)
        load_frag_contig(Btile[j], &Bs[j * B_frag_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; n++) {
          short n_serp = (m % 2) ? (TN - 1 - n) : n;
          frag_mma(Ctile[m * TN + n_serp], Atile[m], Btile[n_serp],
                   Ctile[m * TN + n_serp]);
        }
      }
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  // Affine store pair — interior float2 fragments / per-element edge guard.
  // One template for both so the epilogue cannot diverge per tile path.
  METAL_FUNC void store_affine(device float* C, int ldd, float scale,
                               float offset) {
    C += sm * ldd + sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        int off = (i * kFrag) * WM * ldd + (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] = fma(val[0], scale, offset);
        val[1] = fma(val[1], scale, offset);
        *reinterpret_cast<device float2*>(&C[off]) = val;
      }
    }
  }

  METAL_FUNC void store_affine_safe(device float* C, int ldd, short2 dims,
                                    float scale, float offset) {
    C += sm * ldd + sn;
    dims -= short2(sn, sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        short r = (i * kFrag) * WM;
        short c = (j * kFrag) * WN;
        frag_type val = Ctile[i * TN + j];
        val[0] = fma(val[0], scale, offset);
        val[1] = fma(val[1], scale, offset);
        if (r < dims.y) {
          if (c + 1 < dims.x)
            *reinterpret_cast<device float2*>(&C[r * ldd + c]) = val;
          else if (c < dims.x)
            C[r * ldd + c] = val[0];
        }
      }
    }
  }
};

// Unified STEEL body — BN = 64, WM = WN = 2; BM ∈ {64, 32}. gemm_params
// reuse: `a_fast` carries swizzle_log for STEEL dispatches (32-family
// ignores swizzle; STEEL ignores the fast flags). TA/TB select the
// transposing loader for an operand whose device storage is a transposed
// view (backward-pass matmuls, attention's Q·Kᵀ) — the MMA stays NN.
template <short BM, bool TA = false, bool TB = false>
void steel_body_(device const float* A, device const float* B,
                 device float* C, constant gemm_params& p,
                 threadgroup float* As, threadgroup float* Bs,
                 uint3 tgid, uint sid, uint lane) {
  constexpr short BN = 64, BK = 16, WM = 2, WN = 2, pad = 4;
  constexpr short tgp_size = WM * WN * 32;

  short swizzle = short(p.a_fast);
  short tiles_n = short((p.N + BN - 1) / BN);
  short tiles_m = short((p.M + BM - 1) / BM);
  short tid_y = short((tgid.y << swizzle) + (tgid.x & ((1 << swizzle) - 1)));
  short tid_x = short(tgid.x >> swizzle);
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  short row0 = tid_y * BM, col0 = tid_x * BN;
  A += TA ? int(row0) : row0 * int(p.lda);
  B += TB ? col0 * int(p.ldb) : int(col0);

  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  typename select_type<TA, SteelLoaderT<BK, BM, lda_nn, false, tgp_size>,
                       SteelLoader<BM, BK, lda_nn, true, tgp_size>>::type
      loader_a(A, int(p.lda), As, sid, ushort(lane));
  typename select_type<TB, SteelLoaderT<BN, BK, ldb_nn, true, tgp_size>,
                       SteelLoader<BK, BN, ldb_nn, false, tgp_size>>::type
      loader_b(B, int(p.ldb), Bs, sid, ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, lda_nn, ldb_nn> mma_op(sid, lane);

  int k_iters = int(p.K / BK);
  short lbk = short(p.K) - short(k_iters * BK);
  short tgp_bm = min(short(BM), short(p.M - row0));
  short tgp_bn = min(short(BN), short(p.N - col0));
  bool is_interior = (tgp_bm == BM && tgp_bn == BN);

  if (lbk > 0) {
    // K remainder first: jump loaders to the tail, one masked block, rewind.
    size_t k_jump_a = size_t(k_iters) * size_t(loader_a.tile_stride);
    size_t k_jump_b = size_t(k_iters) * size_t(loader_b.tile_stride);
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // load_safe takes SOURCE-coordinate extents (x = cols, y = rows)
    loader_a.load_safe(TA ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm));
    loader_b.load_safe(TB ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  // Main loop: every K block is full; only M/N edges clip. Branches are
  // uniform per threadgroup; the unclipped operand keeps load_unsafe
  // (silarray: clipping the streaming operand costs ~1.2x on narrow-N).
  bool m_full = tgp_bm == BM, n_full = tgp_bn == BN;
  if (is_interior) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  } else if (n_full) {
    // M-edge only: A clips on the logical M extent (source rows when NN,
    // source cols when TA), B stays on the vectorized load.
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (TA) loader_a.load_safe_cols(tgp_bm);
      else loader_a.load_safe_rows(tgp_bm);
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  } else if (m_full) {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      if (TB) loader_b.load_safe_rows(tgp_bn);
      else loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.template mma<true>(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (TA) loader_a.load_safe_cols(tgp_bm);
      else loader_a.load_safe_rows(tgp_bm);
      if (TB) loader_b.load_safe_rows(tgp_bn);
      else loader_b.load_safe_cols(tgp_bn);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  }

  int ldd = int(p.N);
  C += row0 * ldd + col0;
  if (is_interior) {
    mma_op.store_affine(C, ldd, p.scale, p.offset);
  } else {
    mma_op.store_affine_safe(
        C, ldd,
        short2(min(short(BN), short(p.N - col0)),
               min(short(BM), short(p.M - row0))),
        p.scale, p.offset);
  }
}

#define TL_STEEL_KERNEL(name, BM, TA, TB)                                    \
  kernel void name(device const float* A [[buffer(0)]],                      \
                   device const float* B [[buffer(1)]],                      \
                   device float* C [[buffer(2)]],                            \
                   constant gemm_params& p [[buffer(3)]],                    \
                   uint3 tgid [[threadgroup_position_in_grid]],              \
                   uint sid [[simdgroup_index_in_threadgroup]],              \
                   uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float As[BM * 20];                                           \
    threadgroup float Bs[16 * 68];                                           \
    steel_body_<BM, TA, TB>(A, B, C, p, As, Bs, tgid, sid, lane);            \
  }

TL_STEEL_KERNEL(sgemm_steel_, 64, false, false)
TL_STEEL_KERNEL(sgemm_steel_32x64_, 32, false, false)
// Transposed-operand variants: _ta_ reads A's transposed view in place
// (dW = xᵀ @ g), _tb_ reads B's (dx = g @ Wᵀ, attention's Q·Kᵀ).
TL_STEEL_KERNEL(sgemm_steel_ta_, 64, true, false)
TL_STEEL_KERNEL(sgemm_steel_tb_, 64, false, true)
TL_STEEL_KERNEL(sgemm_steel_32x64_ta_, 32, true, false)
TL_STEEL_KERNEL(sgemm_steel_32x64_tb_, 32, false, true)


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

// Layer norm over the last axis with the affine epilogue: (x - mu) ·
// 1/sqrt(var + eps) · g + b. Two tree sums (x, then the squared deviations),
// each scaled by 1/cols after the tree as row_sum_'s mean is.
struct layer_norm_params {
  uint rows, cols;
  float eps, scale, offset;
};

kernel void layer_norm_(device const float* x [[buffer(0)]],
                        device const float* g [[buffer(1)]],
                        device const float* b [[buffer(2)]],
                        device float* out     [[buffer(3)]],
                        constant layer_norm_params& p [[buffer(4)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[T];
  device const float* src = x + row * p.cols;
  device float* dst = out + row * p.cols;
  float inv_n = 1.0f / float(p.cols);

  float sum = 0.0f;
  for (uint c = lid; c < p.cols; c += T) sum += src[c];
  scratch[lid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mu = scratch[0] * inv_n;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float ss = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float v = src[c] - mu;
    ss += v * v;
  }
  scratch[lid] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = 1.0f / sqrt(scratch[0] * inv_n + p.eps);
  for (uint c = lid; c < p.cols; c += T)
    dst[c] = ((src[c] - mu) * inv * g[c] + b[c]) * p.scale + p.offset;
}

// ---- layer norm's pullback over the last axis, in three launches ----
// cuda's tl_layer_norm_bwd_*: with x̂ = (x − μ)·s and ĝ = dy ⊙ γ,
// dx = s · (ĝ − mean(ĝ) − x̂ · mean(ĝ ⊙ x̂)), dγ = Σ_rows dy ⊙ x̂, dβ = Σ_rows dy.
// The dx kernel recomputes (μ, s) as layer_norm_ does and leaves them in
// `stats` [2, rows] for the column sums, which fold in a fixed order.
struct layer_norm_bwd_params {
  uint rows, cols, rows_per_chunk, chunks;
  float eps;
};

// Tree-sum one value per thread through `scratch`, handing every thread the
// total; the trailing barrier lets a second sum reuse `scratch`.
static inline float tree_sum_(threadgroup float* scratch, uint lid, float v) {
  constexpr uint T = 256;
  scratch[lid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) scratch[lid] += scratch[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float total = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// tree_sum_ for two values at once, in the same order; `scratch` holds 2·256.
static inline float2 tree_sum2_(threadgroup float* scratch, uint lid, float a,
                                float b) {
  constexpr uint T = 256;
  scratch[lid] = a;
  scratch[T + lid] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = T / 2; s > 0; s >>= 1) {
    if (lid < s) {
      scratch[lid] += scratch[lid + s];
      scratch[T + lid] += scratch[T + lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float2 total = float2(scratch[0], scratch[T]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// A row's mean and 1/sqrt(biased variance + eps), the sums layer_norm_ takes
// (cuda's tl_row_stats_).
static inline float2 row_stats_(threadgroup float* scratch, uint lid,
                                device const float* src, uint cols, float eps) {
  constexpr uint T = 256;
  float inv_n = 1.0f / float(cols);
  float sum = 0.0f;
  for (uint c = lid; c < cols; c += T) sum += src[c];
  float mu = tree_sum_(scratch, lid, sum) * inv_n;
  float ss = 0.0f;
  for (uint c = lid; c < cols; c += T) {
    float v = src[c] - mu;
    ss += v * v;
  }
  return float2(mu, 1.0f / sqrt(tree_sum_(scratch, lid, ss) * inv_n + eps));
}

kernel void layer_norm_bwd_dx_(device const float* x  [[buffer(0)]],
                               device const float* g  [[buffer(1)]],
                               device const float* dy [[buffer(2)]],
                               device float* dx       [[buffer(3)]],
                               device float* stats    [[buffer(4)]],
                               constant layer_norm_bwd_params& p [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[2 * T];
  device const float* src = x + row * p.cols;
  device const float* gy = dy + row * p.cols;
  device float* dst = dx + row * p.cols;
  float inv_n = 1.0f / float(p.cols);

  const float2 st = row_stats_(scratch, lid, src, p.cols, p.eps);
  const float mu = st.x, inv = st.y;
  float sg = 0.0f, sgx = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float gh = gy[c] * g[c];
    sg += gh;
    sgx += gh * (src[c] - mu) * inv;
  }
  const float2 sums = tree_sum2_(scratch, lid, sg, sgx);
  const float mean_g = sums.x * inv_n, mean_gx = sums.y * inv_n;
  for (uint c = lid; c < p.cols; c += T) {
    float xhat = (src[c] - mu) * inv;
    dst[c] = inv * (gy[c] * g[c] - mean_g - xhat * mean_gx);
  }
  if (lid == 0) {
    stats[row] = mu;
    stats[p.rows + row] = inv;
  }
}

// The column sums: a threadgroup per (32-column strip, chunk of rows), 32×8
// threads, into `partials` [2, chunks, cols] (dγ's plane first).
kernel void layer_norm_bwd_gb_(device const float* x      [[buffer(0)]],
                               device const float* dy     [[buffer(1)]],
                               device const float* stats  [[buffer(2)]],
                               device float* partials     [[buffer(3)]],
                               constant layer_norm_bwd_params& p [[buffer(4)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float sg[8][32], sgx[8][32];
  uint tx = tid.x, ty = tid.y;
  uint c = tg.x * 32 + tx;
  uint chunk = tg.y;
  uint r0 = chunk * p.rows_per_chunk;
  uint r1 = min(r0 + p.rows_per_chunk, p.rows);
  float ag = 0.0f, agx = 0.0f;
  if (c < p.cols) {
    for (uint r = r0 + ty; r < r1; r += 8) {
      float gy = dy[r * p.cols + c];
      float xhat = (x[r * p.cols + c] - stats[r]) * stats[p.rows + r];
      ag += gy;
      agx += gy * xhat;
    }
  }
  sg[ty][tx] = ag;
  sgx[ty][tx] = agx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ty == 0 && c < p.cols) {
    float tg_ = 0.0f, tgx = 0.0f;
    for (uint l = 0; l < 8; l++) {
      tg_ += sg[l][tx];
      tgx += sgx[l][tx];
    }
    uint at = chunk * p.cols + c;
    partials[at] = tgx;
    partials[p.chunks * p.cols + at] = tg_;
  }
}

// The second half: a thread per column folds its chunks into dγ and dβ.
kernel void layer_norm_bwd_gb_fold_(device const float* partials [[buffer(0)]],
                                    device float* dg             [[buffer(1)]],
                                    device float* db             [[buffer(2)]],
                                    constant layer_norm_bwd_params& p [[buffer(3)]],
                                    uint c [[thread_position_in_grid]]) {
  if (c >= p.cols) return;
  float tg_ = 0.0f, tb = 0.0f;
  for (uint k = 0; k < p.chunks; k++) {
    tg_ += partials[k * p.cols + c];
    tb += partials[(p.chunks + k) * p.cols + c];
  }
  dg[c] = tg_;
  db[c] = tb;
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

// ---------------------------------------------------------------------------
// im2col's pad/fold — gather-style, one invocation per OUTPUT element,
// mirroring kernels/tensorlib_webgpu.wgsl's pad/fold: Metal's device-memory
// atomics are int/uint only (no float atomic add), the same gap that pushed
// WGSL away from CUDA's scatter+atomicAdd. A gather needs no pre-zeroed
// output (every C cell is written exactly once) and no atomics — the
// tradeoff is fold's small bounded loop over the window indices that could
// cover a given output cell. `a` is required contiguous by array.h's
// gpu_pad_/gpu_fold_, so a_strides are derived from a_shape below rather
// than uploaded separately.
// ---------------------------------------------------------------------------

constant uint kPadFoldMaxRank = 8;

struct pad_fold_params {
  uint out_shape[kPadFoldMaxRank];  // pad: length rank; fold: length rank-1
  uint a_shape[kPadFoldMaxRank];    // length rank
  uint rank;
  uint axis;
  int shift;  // pad: `before`; fold: `step`
  uint n;     // output element count (dispatch bound)
};

// Shared by both kernels below (mirrors tensorlib_webgpu.wgsl's decode_idx):
// row-major decode of a dispatch-global thread id against a shape held in
// `shape`, into a fixed-size local array. `rank8` lets callers pass either a
// full-rank or a (rank-1)-length shape.
inline void decode_idx(uint i, constant uint* shape, uint rank8,
                       thread uint* out_idx) {
  uint rem = i;
  for (int d = int(rank8) - 1; d >= 0; d--) {
    uint dim = shape[d];
    out_idx[d] = rem % dim;
    rem /= dim;
  }
}

// Row-major strides of a contiguous tensor whose shape is `a_shape`, length
// `rank` — used to address `a`, which pad_/fold_'s GPU dispatch (array.h's
// gpu_pad_/gpu_fold_) requires to be contiguous.
inline void a_strides_from_shape(constant uint* a_shape, uint rank,
                                 thread uint* out_strides) {
  uint acc = 1;
  for (int d = int(rank) - 1; d >= 0; d--) {
    out_strides[d] = acc;
    acc *= a_shape[d];
  }
}

kernel void pad_(device const float* a [[buffer(0)]],
                 device float* out [[buffer(1)]],
                 constant pad_fold_params& p [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint rank = p.rank;

  uint out_idx[kPadFoldMaxRank];
  decode_idx(i, p.out_shape, rank, out_idx);
  uint a_strides[kPadFoldMaxRank];
  a_strides_from_shape(p.a_shape, rank, a_strides);

  uint src = 0;
  bool in_bounds = true;
  for (uint d = 0; d < rank; d++) {
    int c = int(out_idx[d]);
    if (d == p.axis) {
      c -= p.shift;
      if (c < 0 || c >= int(p.a_shape[d])) in_bounds = false;
    }
    // Clamped even out of range so the address stays valid; only `in_bounds`
    // decides the result (mirrors the WGSL kernel's select-both-operands
    // note).
    int cc = clamp(c, 0, int(p.a_shape[d]) - 1);
    src += uint(cc) * a_strides[d];
  }
  out[i] = in_bounds ? a[src] : 0.0f;
}

kernel void fold_(device const float* a [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant pad_fold_params& p [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint rank = p.rank;
  uint out_rank = rank - 1;
  int step = p.shift;

  uint out_idx[kPadFoldMaxRank];
  decode_idx(i, p.out_shape, out_rank, out_idx);
  uint a_strides[kPadFoldMaxRank];
  a_strides_from_shape(p.a_shape, rank, a_strides);

  int win = int(p.a_shape[rank - 1]);
  int nwin = int(p.a_shape[p.axis]);
  int j = int(out_idx[p.axis]);

  int w_min = 0;
  if (j - win + 1 > 0) w_min = (j - win + 1 + step - 1) / step;
  int w_max = min(j / step, nwin - 1);

  // Every non-axis dimension's contribution to `src` is the same across the
  // whole w loop below (only the axis and window-offset terms vary per w) —
  // hoisted out so each window iteration is O(1) address math instead of
  // O(out_rank).
  uint base = 0;
  for (uint d = 0; d < out_rank; d++) {
    if (d != p.axis) base += out_idx[d] * a_strides[d];
  }

  float sum = 0.0f;
  for (int w = w_min; w <= w_max; w++) {
    int k = j - w * step;
    if (k < 0 || k >= win) continue;
    uint src = base + uint(w) * a_strides[p.axis] + uint(k) * a_strides[rank - 1];
    sum += a[src];
  }
  out[i] = sum;
}

// ---------------------------------------------------------------------------
// Embedding-table lookup (index_select/index_add) and pooling-style one-hot
// scatter (scatter_to_axis) — the Metal counterparts of tl_index_select/
// tl_index_add/tl_scatter_axis in kernels/tensorlib_cuda.cu.
//
// index_select and scatter_to_axis are gathers already (every output element
// is written by exactly one invocation, reading whatever it needs), so they
// port the CUDA kernel body directly. index_add is CUDA's one true scatter+
// atomicAdd here — repeated indices really do collide. Float atomics would
// need MSL 3 for the whole library, so index_add_ is a gather too: each
// output row sums the source rows whose index matches it, in source order,
// which also keeps it deterministic.
// ---------------------------------------------------------------------------

struct gather_params {
  uint row_size;
  uint n;  // dispatch bound: k * row_size
};

kernel void index_select_(device const float* a [[buffer(0)]],
                          device const float* idx [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant gather_params& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint row = i / p.row_size, col = i % p.row_size;
  uint src_row = uint(idx[row] + 0.5f);
  out[i] = a[src_row * p.row_size + col];
}

struct index_add_params {
  uint row_size;
  uint k;  // number of source rows to scan
};

// A threadgroup per output row. The 256 threads scan the indices a block of
// 2048 at a time, 8 consecutive each, and compact the matching source rows
// into `list` in source order (a scan over the per-thread counts); then each
// thread adds those rows into the columns it owns. Every index is read once
// per output row rather than once per output element, and the sums run in
// ascending source order as a plain loop over k would.
kernel void index_add_(device const float* idx [[buffer(0)]],
                       device const float* values [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant index_add_params& p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256, PER = 8, BLOCK = T * PER;
  threadgroup uint list[BLOCK];
  threadgroup uint offsets[T + 1];
  device float* dst = out + row * p.row_size;
  for (uint c = lid; c < p.row_size; c += T) dst[c] = 0.0f;

  for (uint b = 0; b < p.k; b += BLOCK) {
    const uint s0 = b + lid * PER;
    uint hit = 0;  // bit j: source row s0 + j matches
    for (uint j = 0; j < PER; j++) {
      const uint s = s0 + j;
      if (s < p.k && uint(idx[s] + 0.5f) == row) hit |= 1u << j;
    }
    offsets[lid + 1] = popcount(hit);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      offsets[0] = 0;
      for (uint t = 1; t <= T; t++) offsets[t] += offsets[t - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = offsets[lid];
    for (uint j = 0; j < PER; j++) {
      if (hit & (1u << j)) list[at++] = s0 + j;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint total = offsets[T];
    if (total != 0) {
      for (uint c = lid; c < p.row_size; c += T) {
        float sum = dst[c];
        for (uint m = 0; m < total; m++) {
          sum += values[list[m] * p.row_size + c];
        }
        dst[c] = sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // list/offsets reused
  }
}

struct scatter_axis_params {
  uint size;
  uint n;  // dispatch bound: num_positions * size
};

kernel void scatter_axis_(device const float* idx [[buffer(0)]],
                          device const float* values [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant scatter_axis_params& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint pos = i / p.size, k = i % p.size;
  out[i] = (uint(idx[pos] + 0.5f) == k) ? values[pos] : 0.0f;
}

// ---------------------------------------------------------------------------
// N-D broadcast binary (any rank) and N-D broadcast ternary select
// (Tensor.where's masking) — the Metal counterparts of tl_b*_nd/tl_where_nd
// in kernels/tensorlib_cuda.cu. Same flat-index decode as pad_/fold_ above,
// against strides supplied by the host (broadcast_strides(), 0 on a
// broadcast axis) rather than derived from a shape.
// ---------------------------------------------------------------------------

struct bcast_nd_params {
  uint out_shape[kPadFoldMaxRank];
  uint a_strides[kPadFoldMaxRank];
  uint b_strides[kPadFoldMaxRank];
  uint rank;
  uint n;
  float scale;
  float offset;
};

#define EW_BCAST_ND(name, expr)                                              \
  kernel void name(device const float* a [[buffer(0)]],                     \
                   device const float* b [[buffer(1)]],                     \
                   device float* out [[buffer(2)]],                         \
                   constant bcast_nd_params& p [[buffer(3)]],               \
                   uint i [[thread_position_in_grid]]) {                    \
    if (i >= p.n) return;                                                    \
    uint rem = i;                                                            \
    uint a_off = 0, b_off = 0;                                               \
    for (int d = int(p.rank) - 1; d >= 0; d--) {                            \
      uint dim = p.out_shape[d];                                            \
      uint coord = rem % dim;                                               \
      rem /= dim;                                                           \
      a_off += coord * p.a_strides[d];                                      \
      b_off += coord * p.b_strides[d];                                      \
    }                                                                        \
    float av = a[a_off], bv = b[b_off];                                     \
    out[i] = fma((expr), p.scale, p.offset);                                \
  }

EW_BCAST_ND(badd_nd_, av + bv)
EW_BCAST_ND(bsub_nd_, av - bv)
EW_BCAST_ND(bmul_nd_, av * bv)
EW_BCAST_ND(bdiv_nd_, av / bv)
EW_BCAST_ND(bpow_nd_, pow(av, bv))
#undef EW_BCAST_ND

struct where_nd_params {
  uint out_shape[kPadFoldMaxRank];
  uint c_strides[kPadFoldMaxRank];
  uint a_strides[kPadFoldMaxRank];
  uint b_strides[kPadFoldMaxRank];
  uint rank;
  uint n;
};

kernel void where_nd_(device const float* cond [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      constant where_nd_params& p [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint rem = i;
  uint c_off = 0, a_off = 0, b_off = 0;
  for (int d = int(p.rank) - 1; d >= 0; d--) {
    uint dim = p.out_shape[d];
    uint coord = rem % dim;
    rem /= dim;
    c_off += coord * p.c_strides[d];
    a_off += coord * p.a_strides[d];
    b_off += coord * p.b_strides[d];
  }
  out[i] = cond[c_off] != 0.0f ? a[a_off] : b[b_off];
}

// ---------------------------------------------------------------------------
// copy_nd: clone()'s device arm for a strided view (a transpose, a permute, a
// stride-0 widening) -- the Metal counterpart of tl_copy_nd. The same flat-
// index decode as where_nd_ above with one input: a gather into a contiguous
// output, so a permuted view clones on the device instead of through a flush.
// ---------------------------------------------------------------------------

struct copy_nd_params {
  uint out_shape[kPadFoldMaxRank];
  uint a_strides[kPadFoldMaxRank];
  uint rank;
  uint n;
};

kernel void copy_nd_(device const float* a [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant copy_nd_params& p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint rem = i;
  uint a_off = 0;
  for (int d = int(p.rank) - 1; d >= 0; d--) {
    uint dim = p.out_shape[d];
    uint coord = rem % dim;
    rem /= dim;
    a_off += coord * p.a_strides[d];
  }
  out[i] = a[a_off];
}

// ---------------------------------------------------------------------------
// sum_to: sum `a` down to a smaller broadcast-target shape (the dual of
// broadcast_to that every arithmetic op's backward uses to un-broadcast a
// gradient) -- the Metal counterpart of tl_sum_to in kernels/tensorlib_cuda.cu.
// Gather, not scatter: one thread per OUTPUT element sums every `a` element
// that broadcasts onto it, so unlike index_add there is no write conflict
// and no atomics needed.
// ---------------------------------------------------------------------------

struct sum_to_params {
  uint a_shape[kPadFoldMaxRank];
  uint a_strides[kPadFoldMaxRank];
  uint acc[kPadFoldMaxRank];  // a's shape broadcast-aligned against the
                              // output's own strides; 0 on a reduced axis
  uint rank;
  uint out_n;
  uint reduced_n;  // product of a_shape over exactly the zero-acc axes
};

// Output t's reduction: where it starts in `a` and which axes it walks.
struct sum_to_walk {
  uint base;
  uint red_axis[kPadFoldMaxRank];
  uint red_count;
};

static inline sum_to_walk sum_to_walk_of_(constant sum_to_params& p, uint t) {
  sum_to_walk w;
  w.base = 0;
  w.red_count = 0;
  for (uint d = 0; d < p.rank; d++) {
    if (p.acc[d] != 0) {
      uint idx = (t / p.acc[d]) % p.a_shape[d];
      w.base += idx * p.a_strides[d];
    } else {
      w.red_axis[w.red_count++] = d;
    }
  }
  return w;
}

// The offset in `a` of the reduction's r-th element, the last axis fastest.
static inline uint sum_to_offset_(constant sum_to_params& p,
                                  thread const sum_to_walk& w, uint r) {
  uint rem = r;
  uint off = w.base;
  for (int k = int(w.red_count) - 1; k >= 0; k--) {
    uint d = w.red_axis[k];
    uint dim = p.a_shape[d];
    uint coord = rem % dim;
    rem /= dim;
    off += coord * p.a_strides[d];
  }
  return off;
}

kernel void sum_to_(device const float* a [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant sum_to_params& p [[buffer(2)]],
                    uint t [[thread_position_in_grid]]) {
  if (t >= p.out_n) return;
  const sum_to_walk w = sum_to_walk_of_(p, t);
  float sum = 0.0f;
  for (uint r = 0; r < p.reduced_n; r++) sum += a[sum_to_offset_(p, w, r)];
  out[t] = sum;
}

// sum_to_ with a threadgroup per output element: the flat kernel above walks
// the whole reduced range on one thread, so a [N, C] -> [C] bias gradient does
// N adds in sequence per column. Here 256 threads stride over that range and
// finish in a tree, as cuda's tl_sum_to_blocked does.
kernel void sum_to_blocked_(device const float* a [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant sum_to_params& p [[buffer(2)]],
                            uint t [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float scratch[T];
  const sum_to_walk w = sum_to_walk_of_(p, t);
  float sum = 0.0f;
  for (uint r = lid; r < p.reduced_n; r += T) sum += a[sum_to_offset_(p, w, r)];
  const float total = tree_sum_(scratch, lid, sum);
  if (lid == 0) out[t] = total;
}

// concat_part: writes `a` (contiguous, one part of an N-ary Tensor.concat)
// into `out` at `p.shift` (already before*out_strides[axis], a flat
// element offset) along one axis. One invocation per SOURCE element (this
// part's own count, `p.n`) -- unlike pad_ above, there is no zero border
// to fill and no bounds check: concat's parts exhaustively and disjointly
// cover `out`, so this is a plain scatter, one write per source element,
// never colliding across the separate per-part dispatches that build up
// one `out`. Mirrors tensorlib_cuda.cu's own reuse of tl_pad's body for
// this, just as its own kernel rather than sharing pad_'s PSO (that one
// dispatches over OUTPUT elements, which this doesn't want).
struct concat_part_params {
  uint out_strides[kPadFoldMaxRank];
  uint a_shape[kPadFoldMaxRank];
  uint rank;
  uint shift;
  uint n;
};

kernel void concat_part_(device const float* a [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant concat_part_params& p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint rem = i;
  uint dst = 0;
  for (int d = int(p.rank) - 1; d >= 0; d--) {
    uint dim = p.a_shape[d];
    uint coord = rem % dim;
    rem /= dim;
    dst += coord * p.out_strides[d];
  }
  out[dst + p.shift] = a[i];
}

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors tensorlib_cuda.cu's own tl_rope. `a` is [rows, D]
// contiguous (rows = H*T: a [H,T,D] tensor flattened, or [H,D] with T=1);
// row r's head-dim vector sits at position `pos + (r % T)`; pairs
// (j, j+D/2) rotate by angle = position * base^(-2j/D). Dispatched flat
// over rows*(D/2), one invocation per (r, j) pair (CUDA instead grids by
// row, blocks by D/2 -- every kernel in this file is flat-1D).
struct rope_params {
  uint T, D, pos;
  uint half_;
  float base;
  uint n;
};

kernel void rope_(device const float* x [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant rope_params& p [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint r = i / p.half_;
  uint j = i % p.half_;
  uint t = p.T > 0 ? (r % p.T) : 0u;
  float position = float(p.pos + t);
  float theta = pow(p.base, -2.0f * float(j) / float(p.D));
  float ang = position * theta;
  float c = cos(ang);
  float s = sin(ang);
  uint bi = r * p.D;
  float x0 = x[bi + j];
  float x1 = x[bi + j + p.half_];
  out[bi + j] = x0 * c - x1 * s;
  out[bi + j + p.half_] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// Causal prefill attention and its pullback: cuda's attn_prefill_tiled_core /
// attn_bwd_dq_core / attn_bwd_dkv_core, the same online softmax and causal
// rule, on simdgroup_matrix. A threadgroup is 4 simdgroups of 8 rows each
// (queries, or keys for dK/dV); the products are 8×8 MMAs and the softmax
// runs on each lane's own fragment elements, so scores never leave registers.
// Loops over fragments are fully unrolled: a fragment array indexed by a loop
// variable is spilled to memory, which costs more than the MMAs save.
// Tiles are 16 rows at D=128: two tiles of 32 rows would fill all 32 KB of
// threadgroup memory.
// ---------------------------------------------------------------------------

struct attn_params {
  uint T, kv_stride, group, pos0;
  float scale;
};

// Across the four lanes that hold one fragment row (lane bits 0 and 3).
static inline float frag_row_max_(float v) {
  v = max(v, simd_shuffle_xor(v, 1));
  return max(v, simd_shuffle_xor(v, 8));
}
static inline float frag_row_sum_(float v) {
  v += simd_shuffle_xor(v, 1);
  return v + simd_shuffle_xor(v, 8);
}

// Rows [base, end) of row-major [*, AD] sources into ROWS rows of threadgroup
// tiles; rows past `end` read as zero. Two sources share one loop, so both
// loads of an iteration are in flight together.
template <int ROWS, int AD>
static inline void attn_stage_rows_(threadgroup float* dst,
                                    device const float* src, uint base,
                                    uint end, uint tid) {
  constexpr uint A4 = AD / 4;
  threadgroup float4* d4 = (threadgroup float4*)dst;
  device const float4* s4 = (device const float4*)src;
  for (uint i = tid; i < uint(ROWS) * A4; i += 128) {
    const uint r = i / A4;
    d4[i] = base + r < end ? s4[(base + r) * A4 + i % A4] : float4(0.f);
  }
}
template <int ROWS, int AD>
static inline void attn_stage_rows_(threadgroup float* dst0,
                                    device const float* src0,
                                    threadgroup float* dst1,
                                    device const float* src1, uint base,
                                    uint end, uint tid) {
  constexpr uint A4 = AD / 4;
  threadgroup float4* d0 = (threadgroup float4*)dst0;
  threadgroup float4* d1 = (threadgroup float4*)dst1;
  device const float4* s0 = (device const float4*)src0;
  device const float4* s1 = (device const float4*)src1;
  for (uint i = tid; i < uint(ROWS) * A4; i += 128) {
    const uint r = i / A4;
    const bool live = base + r < end;
    const uint at = (base + r) * A4 + i % A4;
    d0[i] = live ? s0[at] : float4(0.f);
    d1[i] = live ? s1[at] : float4(0.f);
  }
}

// A lane's fragment row (`fn` its first column) written out, scaled by k.
template <int DF>
static inline void attn_store_row_(device float* row,
                                   thread simdgroup_matrix<float, 8, 8> (&f)[DF],
                                   float k) {
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    thread auto& e = f[i].thread_elements();
    *(device float2*)(row + i * 8) = float2(e[0], e[1]) * k;
  }
}

// q, out [H,T,D]; K/V a cache of kv_stride floats per kv head read over
// [0, pos0+T); query p sits at pos0+p. BK keys per tile.
template <int AD, int BK>
kernel void attn_prefill_(device const float* q   [[buffer(0)]],
                          device const float* K   [[buffer(1)]],
                          device const float* V   [[buffer(2)]],
                          device float* out       [[buffer(3)]],
                          constant attn_params& p [[buffer(4)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 ntg [[threadgroups_per_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 32, DF = AD / 8, KF = BK / 8;
  static_assert(2 * BK >= BQ, "Q is staged through the K/V tiles");
  threadgroup float4 KV4[2 * BK * AD / 4];  // K tile rows, then V tile rows
  threadgroup float* Ks = (threadgroup float*)KV4;
  threadgroup float* Vs = Ks + BK * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint qbase = (ntg.y - 1u - tg.y) * BQ;  // heaviest tiles first
  const uint kv_h = p.group ? h / p.group : h;
  device const float* Kh = K + kv_h * p.kv_stride;
  device const float* Vh = V + kv_h * p.kv_stride;

  attn_stage_rows_<BQ, AD>(Ks, q + h * T * AD, qbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> qf[DF], of[DF];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(qf[i], &Ks[sid * 8 * AD + i * 8], AD);
    of[i] = simdgroup_matrix<float, 8, 8>(0);
  }

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  const uint pabs = p.pos0 + qbase + sid * 8 + fm;
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const uint kmax = p.pos0 + last;
  const float sl2 = p.scale * M_LOG2E_F;
  float m = -1e30f, l = 0.0f;

  for (uint kt = 0; kt <= kmax; kt += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<BK, AD>(Ks, Kh, Vs, Vh, kt, kmax + 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[KF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      simdgroup_matrix<float, 8, 8> kf;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], qf[i], kf, s[j]);
      }
    }

    float mt = -1e30f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const uint col = kt + j * 8 + fn + k;
        e[k] = col <= pabs ? e[k] * sl2 : -1e30f;  // causal mask
        mt = max(mt, e[k]);
      }
    }
    const float m_new = max(m, frag_row_max_(mt));
    const float corr = exp2(m - m_new);
    float ls = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        e[k] = exp2(e[k] - m_new);
        ls += e[k];
      }
    }
    l = l * corr + frag_row_sum_(ls);
    m = m_new;

    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < DF; i++) of[i].thread_elements() *= corr;
    simdgroup_matrix<float, 8, 8> vf;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(vf, &Vs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(of[i], s[j], vf, of[i]);
      }
    }
  }

  const uint qi = qbase + sid * 8 + fm;
  if (qi < T) attn_store_row_(out + (h * T + qi) * AD + fn, of, 1.0f / l);
}
template [[host_name("attn_prefill_64_")]] kernel void
attn_prefill_<64, 32>(device const float*, device const float*,
                      device const float*, device float*,
                      constant attn_params&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_prefill_128_")]] kernel void
attn_prefill_<128, 16>(device const float*, device const float*,
                       device const float*, device float*,
                       constant attn_params&, uint2, uint2, uint, uint, uint);

// The query half of the pullback: dq and `stats` [2,H,T] (the row
// logsumexp, then Δ = dO·O) for the key/value half. dq = scale·C·K / l with
// C = exp(S − m)·(dP − Δ), S = scale·QKᵀ and dP = dO·Vᵀ.
template <int AD, int BK>
kernel void attn_bwd_dq_(device const float* q     [[buffer(0)]],
                         device const float* K     [[buffer(1)]],
                         device const float* V     [[buffer(2)]],
                         device const float* dO    [[buffer(3)]],
                         device const float* O     [[buffer(4)]],
                         device float* dq          [[buffer(5)]],
                         device float* stats       [[buffer(6)]],
                         constant attn_params& p   [[buffer(7)]],
                         uint2 tg  [[threadgroup_position_in_grid]],
                         uint2 ntg [[threadgroups_per_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sid  [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 128, BQ = 32, DF = AD / 8, KF = BK / 8, A4 = AD / 4;
  static_assert(2 * BK >= BQ, "Q and dO are each staged through the K/V tiles");
  static_assert(NT % A4 == 0 && A4 <= 32, "a row's float4s share a simdgroup");
  threadgroup float4 KV4[2 * BK * AD / 4];  // K tile rows, then V tile rows
  threadgroup float Ds[BQ];                 // Δ = dO·O per query row
  threadgroup float* Ks = (threadgroup float*)KV4;
  threadgroup float* Vs = Ks + BK * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint qbase = (ntg.y - 1u - tg.y) * BQ;
  device const float* Kh = K + h * T * AD;
  device const float* Vh = V + h * T * AD;

  // Q, then dO, into registers (each staged through the K tile).
  simdgroup_matrix<float, 8, 8> qf[DF], gf[DF], acc[DF];
  attn_stage_rows_<BQ, AD>(Ks, q + h * T * AD, qbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(qf[i], &Ks[sid * 8 * AD + i * 8], AD);
    acc[i] = simdgroup_matrix<float, 8, 8>(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    // dO staged like attn_stage_rows_, with Δ reduced over the A4 lanes that
    // share a row as it passes.
    device const float4* G4 = (device const float4*)(dO + h * T * AD);
    device const float4* O4 = (device const float4*)(O + h * T * AD);
    threadgroup float4* d4 = KV4;
    for (uint i = tid; i < uint(BQ * A4); i += NT) {
      const uint r = i / A4, c = i % A4;
      const bool live = qbase + r < T;
      const float4 g = live ? G4[(qbase + r) * A4 + c] : float4(0.f);
      d4[i] = g;
      float d = live ? dot(g, O4[(qbase + r) * A4 + c]) : 0.f;
      for (ushort o = A4 / 2; o > 0; o /= 2) d += simd_shuffle_xor(d, o);
      if (c == 0) Ds[r] = d;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++)
    simdgroup_load(gf[i], &Ks[sid * 8 * AD + i * 8], AD);

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  const uint prow = qbase + sid * 8 + fm;
  const float delta = Ds[sid * 8 + fm];
  const uint last = qbase + BQ - 1 < T ? qbase + BQ - 1 : T - 1;
  const float sl2 = p.scale * M_LOG2E_F;
  float m = -1e30f, l = 0.0f;

  for (uint kt = 0; kt <= last; kt += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<BK, AD>(Ks, Kh, Vs, Vh, kt, last + 1, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[KF], dp[KF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      simdgroup_matrix<float, 8, 8> kf, vf;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      dp[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], qf[i], kf, s[j]);
        simdgroup_load(vf, &Vs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(dp[j], gf[i], vf, dp[j]);
      }
    }

    float mt = -1e30f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        e[k] = kt + j * 8 + fn + k <= prow ? e[k] * sl2 : -1e30f;
        mt = max(mt, e[k]);
      }
    }
    const float m_new = max(m, frag_row_max_(mt));
    const float corr = exp2(m - m_new);
    float ls = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      thread auto& e = s[j].thread_elements();
      thread auto& d = dp[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const float x = exp2(e[k] - m_new);
        ls += x;
        e[k] = x * (d[k] - delta);  // C, in place of S
      }
    }
    l = l * corr + frag_row_sum_(ls);
    m = m_new;

    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < DF; i++) acc[i].thread_elements() *= corr;
    simdgroup_matrix<float, 8, 8> kf;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < KF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(kf, &Ks[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(acc[i], s[j], kf, acc[i]);
      }
    }
  }

  if (prow < T) {
    attn_store_row_(dq + (h * T + prow) * AD + fn, acc, p.scale / l);
    if (fn == 0) {
      // m is in log2 units scaled like the scores; stats keep natural log.
      stats[h * T + prow] = (m + log2(l)) * M_LN2_F;
      stats[ntg.x * T + h * T + prow] = delta;
    }
  }
}
template [[host_name("attn_bwd_dq_64_")]] kernel void
attn_bwd_dq_<64, 32>(device const float*, device const float*,
                     device const float*, device const float*,
                     device const float*, device float*, device float*,
                     constant attn_params&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_bwd_dq_128_")]] kernel void
attn_bwd_dq_<128, 16>(device const float*, device const float*,
                      device const float*, device const float*,
                      device const float*, device float*, device float*,
                      constant attn_params&, uint2, uint2, uint, uint, uint);

// The key/value half: one threadgroup per (head, 32-key block), walking
// the query tiles that can see those keys, P read off the dq half's
// stats: dV = Pᵀ·dO and dK = scale·Cᵀ·Q, C = P·(dP − Δ).
template <int AD, int TQ>
kernel void attn_bwd_dkv_(device const float* q     [[buffer(0)]],
                          device const float* K     [[buffer(1)]],
                          device const float* V     [[buffer(2)]],
                          device const float* dO    [[buffer(3)]],
                          device const float* stats [[buffer(4)]],
                          device float* dK          [[buffer(5)]],
                          device float* dV          [[buffer(6)]],
                          constant attn_params& p   [[buffer(7)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 ntg [[threadgroups_per_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = 128, BK = 32, DF = AD / 8, QF = TQ / 8;
  static_assert(2 * TQ >= BK, "K and V are each staged through the Q/dO tiles");
  threadgroup float4 QG4[2 * TQ * AD / 4];  // Q tile rows, then dO tile rows
  threadgroup float Ls[TQ], Ds[TQ];         // log2-scaled logsumexp, Δ
  threadgroup float* Qs = (threadgroup float*)QG4;
  threadgroup float* Gs = Qs + TQ * AD;

  const uint T = p.T;
  const uint h = tg.x;
  const uint kbase = tg.y * BK;
  device const float* qh = q + h * T * AD;
  device const float* Gh = dO + h * T * AD;
  device const float* Lh = stats + h * T;
  device const float* Dh = stats + ntg.x * T + h * T;

  // This block's K, then V, rows into registers (each staged through Qs).
  simdgroup_matrix<float, 8, 8> kf[DF], vf[DF], dk[DF], dv[DF];
  attn_stage_rows_<BK, AD>(Qs, K + h * T * AD, kbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++) {
    simdgroup_load(kf[i], &Qs[sid * 8 * AD + i * 8], AD);
    dk[i] = simdgroup_matrix<float, 8, 8>(0);
    dv[i] = simdgroup_matrix<float, 8, 8>(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  attn_stage_rows_<BK, AD>(Qs, V + h * T * AD, kbase, T, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < DF; i++)
    simdgroup_load(vf[i], &Qs[sid * 8 * AD + i * 8], AD);

  const uint fm = frag_row_(lane), fn = frag_col_(lane);
  const uint krow = kbase + sid * 8 + fm;
  const float sl2 = p.scale * M_LOG2E_F;

  for (uint qt = (kbase / TQ) * TQ; qt < T; qt += TQ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    attn_stage_rows_<TQ, AD>(Qs, qh, Gs, Gh, qt, T, tid);
    for (uint r = tid; r < uint(TQ); r += NT) {
      const bool live = qt + r < T;
      Ls[r] = live ? Lh[qt + r] * M_LOG2E_F : 0.f;
      Ds[r] = live ? Dh[qt + r] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> s[QF], dp[QF];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      simdgroup_matrix<float, 8, 8> t;
      s[j] = simdgroup_matrix<float, 8, 8>(0);
      dp[j] = simdgroup_matrix<float, 8, 8>(0);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(t, &Qs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[j], kf[i], t, s[j]);
        simdgroup_load(t, &Gs[j * 8 * AD + i * 8], AD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(dp[j], vf[i], t, dp[j]);
      }
    }

    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      thread auto& e = s[j].thread_elements();
      thread auto& d = dp[j].thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 2; k++) {
        const uint ql = j * 8 + fn + k, qi = qt + ql;
        const float x =
            (qi < T && krow <= qi) ? exp2(e[k] * sl2 - Ls[ql]) : 0.f;
        e[k] = x;                   // Pᵀ
        d[k] = x * (d[k] - Ds[ql]);  // Cᵀ
      }
    }

    simdgroup_matrix<float, 8, 8> t;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < QF; j++) {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < DF; i++) {
        simdgroup_load(t, &Gs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(dv[i], s[j], t, dv[i]);
        simdgroup_load(t, &Qs[j * 8 * AD + i * 8], AD);
        simdgroup_multiply_accumulate(dk[i], dp[j], t, dk[i]);
      }
    }
  }

  if (krow < T) {
    attn_store_row_(dK + (h * T + krow) * AD + fn, dk, p.scale);
    attn_store_row_(dV + (h * T + krow) * AD + fn, dv, 1.0f);
  }
}
template [[host_name("attn_bwd_dkv_64_")]] kernel void
attn_bwd_dkv_<64, 32>(device const float*, device const float*,
                      device const float*, device const float*,
                      device const float*, device float*, device float*,
                      constant attn_params&, uint2, uint2, uint, uint, uint);
template [[host_name("attn_bwd_dkv_128_")]] kernel void
attn_bwd_dkv_<128, 16>(device const float*, device const float*,
                       device const float*, device const float*,
                       device const float*, device float*, device float*,
                       constant attn_params&, uint2, uint2, uint, uint, uint);

// ---------------------------------------------------------------------------
// Decode GEMV: y[n] = sum_k a[k]·B[k,n], cuda's tl_gemv_f32/tl_gemv_bf16.
// Batch~1 decode is memory-bandwidth bound on the K×N weight, so one thread
// owns one output column and consecutive threads read consecutive columns —
// the B reads coalesce. `a` is staged a tile at a time, one read per
// threadgroup rather than one per thread.
//
// Split-K over the grid's y: a narrow layer (N/256 threadgroups) leaves most
// of the GPU idle and runs slower than the M=1 GEMM it replaces. Each slice
// sums its own K range into out[s·n + col] and a combine pass adds the slices
// up; with one slice `out` is y itself and the pass is skipped.
// ---------------------------------------------------------------------------

struct gemv_params {
  uint n, k, chunk;
};

// The weight as f32: bf16 is the top 16 bits of the pattern.
static inline float widen_(float w) { return w; }
static inline float widen_(ushort h) { return as_type<float>(uint(h) << 16); }

// `as` is the caller's threadgroup tile of NT floats (threadgroup memory can
// only be declared in a kernel).
template <typename WT>
static inline void gemv_(device const float* a, device const WT* B,
                         device float* out, constant gemv_params& p,
                         threadgroup float* as, uint2 tgp, uint tid) {
  constexpr uint NT = 256;
  const uint col = tgp.x * NT + tid;
  const uint k0 = tgp.y * p.chunk;
  const uint k1 = min(p.k, k0 + p.chunk);
  float acc = 0.0f;
  for (uint kt = k0; kt < k1; kt += NT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    as[tid] = kt + tid < k1 ? a[kt + tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kn = min(NT, k1 - kt);
    if (col < p.n) {
      device const WT* Bk = B + (ulong)kt * p.n + col;
      for (uint i = 0; i < kn; i++) {
        acc += as[i] * widen_(Bk[(ulong)i * p.n]);
      }
    }
  }
  if (col < p.n) out[(ulong)tgp.y * p.n + col] = acc;
}

kernel void gemv_f32_(device const float* a   [[buffer(0)]],
                      device const float* B   [[buffer(1)]],
                      device float* out       [[buffer(2)]],
                      constant gemv_params& p [[buffer(3)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]]) {
  threadgroup float as[256];
  gemv_<float>(a, B, out, p, as, tgp, tid);
}

kernel void gemv_bf16_(device const float* a   [[buffer(0)]],
                       device const ushort* B  [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       constant gemv_params& p [[buffer(3)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]]) {
  threadgroup float as[256];
  gemv_<ushort>(a, B, out, p, as, tgp, tid);
}

// Sum the split-K slices: parts × n partials down to y[n].
struct gemv_combine_params {
  uint n, parts;
};

kernel void gemv_combine_(device const float* parts        [[buffer(0)]],
                          device float* y                  [[buffer(1)]],
                          constant gemv_combine_params& p  [[buffer(2)]],
                          uint col [[thread_position_in_grid]]) {
  if (col >= p.n) return;
  float acc = 0.0f;
  for (uint s = 0; s < p.parts; s++) acc += parts[(ulong)s * p.n + col];
  y[col] = acc;
}

// int4 weights: cuda's tl_gemv_q4. One threadgroup per output row, whose
// quantization groups are contiguous in the [N,K] packing; a thread takes one
// word (8 packed int4) per step and the threadgroup reduces at the end.
struct gemv_q4_params {
  uint n, k, group;
};

kernel void gemv_q4_(device const float* a       [[buffer(0)]],
                     device const uint* qw       [[buffer(1)]],
                     device const float* scales  [[buffer(2)]],
                     device float* y             [[buffer(3)]],
                     constant gemv_q4_params& p  [[buffer(4)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]],
                     uint nt   [[threads_per_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint nsg  [[simdgroups_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  device const uint* qrow = qw + (ulong)row * (p.k >> 3);
  device const float* srow = scales + (ulong)row * (p.k / p.group);
  float acc = 0.0f;
  for (uint k0 = tid * 8; k0 < p.k; k0 += nt * 8) {
    const uint w = qrow[k0 >> 3];
    const float sc = srow[k0 / p.group];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < 8; j++) {
      const int q = int((w >> (j * 4)) & 0xFu) - 8;
      acc += a[k0 + j] * (sc * float(q));
    }
  }
  const float s = simd_sum(acc);
  if (lane == 0) red[sgid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.0f;
    for (uint w = 0; w < nsg; w++) t += red[w];
    y[row] = t;
  }
}

// ---------------------------------------------------------------------------
// Fused decode attention: one query row against the whole cache. cuda's
// attn_decode_core, the same online softmax and merge. A threadgroup owns one
// query head as AD threads, so a lane holds the output dims lane, lane + 32,
// …; each simdgroup walks every NW-th key, reduces the score with simd_sum,
// and the simdgroups' softmax states are merged through threadgroup memory.
//
// Split-KV: a model with few heads leaves most of the GPU idle at one
// threadgroup a head, so the keys are cut into chunks over the grid's y, each
// chunk writing its own (m, l, acc) partial for attn_combine_ to merge.
// ---------------------------------------------------------------------------

struct attn_decode_params {
  uint ctx, kv_stride, group, chunk;
  float scale;
};

// The softmax state of keys [k0, k1) for this threadgroup's head: `gm`, `gl`
// and this thread's output dim `o`. The threadgroup arrays are the caller's
// (threadgroup memory can only be declared in a kernel): qs [AD], sm/sl [NW]
// and sacc [NW][AD].
template <int AD>
static inline void attn_span_(device const float* qh, device const float* Kh,
                             device const float* Vh, uint k0, uint k1,
                             float scale, threadgroup float* qs,
                             threadgroup float* sm, threadgroup float* sl,
                             threadgroup float* sacc, uint tid, uint sgid,
                             uint lane, thread float& gm, thread float& gl,
                             thread float& o) {
  constexpr int NW = AD / 32;
  qs[tid] = qh[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m = -1e30f, l = 0.0f, acc[NW];
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NW; r++) acc[r] = 0.0f;
  for (uint i = k0 + sgid; i < k1; i += NW) {
    device const float* Ki = Kh + i * AD;
    float dot = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NW; r++) dot += qs[lane + r * 32] * Ki[lane + r * 32];
    const float sc = simd_sum(dot) * scale;
    const float m_new = max(m, sc);
    const float corr = exp(m - m_new), pr = exp(sc - m_new);
    l = l * corr + pr;
    device const float* Vi = Vh + i * AD;
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < NW; r++)
      acc[r] = acc[r] * corr + pr * Vi[lane + r * 32];
    m = m_new;
  }

  // Merge the simdgroups' states (one whose key range was empty carries
  // m = -1e30 and l = 0, so it contributes nothing).
  if (lane == 0) {
    sm[sgid] = m;
    sl[sgid] = l;
  }
  _Pragma("clang loop unroll(full)")
  for (int r = 0; r < NW; r++) sacc[sgid * AD + lane + r * 32] = acc[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  gm = -1e30f;
  _Pragma("clang loop unroll(full)")
  for (int w = 0; w < NW; w++) gm = max(gm, sm[w]);
  gl = 0.0f;
  o = 0.0f;
  _Pragma("clang loop unroll(full)")
  for (int w = 0; w < NW; w++) {
    const float e = exp(sm[w] - gm);
    gl += sl[w] * e;
    o += sacc[w * AD + tid] * e;
  }
}

template <int AD>
kernel void attn_decode_(device const float* q          [[buffer(0)]],
                         device const float* K          [[buffer(1)]],
                         device const float* V          [[buffer(2)]],
                         device float* out              [[buffer(3)]],
                         constant attn_decode_params& p [[buffer(4)]],
                         uint h    [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NW = AD / 32;
  threadgroup float qs[AD], sm[NW], sl[NW], sacc[NW * AD];
  const uint kv_h = p.group ? h / p.group : h;
  float gm, gl, o;
  attn_span_<AD>(q + h * AD, K + kv_h * p.kv_stride, V + kv_h * p.kv_stride, 0,
                 p.ctx, p.scale, qs, sm, sl, sacc, tid, sgid, lane, gm, gl, o);
  out[h * AD + tid] = o / gl;
}
template [[host_name("attn_decode_64_")]] kernel void
attn_decode_<64>(device const float*, device const float*, device const float*,
                 device float*, constant attn_decode_params&, uint, uint, uint,
                 uint);
template [[host_name("attn_decode_128_")]] kernel void
attn_decode_<128>(device const float*, device const float*, device const float*,
                  device float*, constant attn_decode_params&, uint, uint, uint,
                  uint);

// One key chunk of one head: partials pm[H*S] | pl[H*S] | pacc[H*S*AD], the
// layout attn_combine_ reads (cuda's attn_partials).
template <int AD>
kernel void attn_decode_split_(device const float* q          [[buffer(0)]],
                               device const float* K          [[buffer(1)]],
                               device const float* V          [[buffer(2)]],
                               device float* parts            [[buffer(3)]],
                               constant attn_decode_params& p [[buffer(4)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               uint2 ntg [[threadgroups_per_grid]],
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr int NW = AD / 32;
  threadgroup float qs[AD], sm[NW], sl[NW], sacc[NW * AD];
  const uint h = tgp.x, s = tgp.y;
  const uint kv_h = p.group ? h / p.group : h;
  const uint k0 = s * p.chunk, k1 = min(p.ctx, k0 + p.chunk);
  float gm = -1e30f, gl = 0.0f, o = 0.0f;
  if (k0 < k1) {
    attn_span_<AD>(q + h * AD, K + kv_h * p.kv_stride, V + kv_h * p.kv_stride,
                   k0, k1, p.scale, qs, sm, sl, sacc, tid, sgid, lane, gm, gl,
                   o);
  }
  const uint hs = ntg.x * ntg.y, at = h * ntg.y + s;
  if (tid == 0) {
    parts[at] = gm;
    parts[hs + at] = gl;
  }
  parts[2 * hs + at * AD + tid] = o;
}
template [[host_name("attn_decode_split_64_")]] kernel void
attn_decode_split_<64>(device const float*, device const float*,
                       device const float*, device float*,
                       constant attn_decode_params&, uint2, uint2, uint, uint,
                       uint);
template [[host_name("attn_decode_split_128_")]] kernel void
attn_decode_split_<128>(device const float*, device const float*,
                        device const float*, device float*,
                        constant attn_decode_params&, uint2, uint2, uint, uint,
                        uint);

// Merge one head's S partials, rescaling each by exp(m_s − max m).
struct attn_combine_params {
  uint splits;
};

template <int AD>
kernel void attn_combine_(device const float* parts        [[buffer(0)]],
                          device float* out                [[buffer(1)]],
                          constant attn_combine_params& p  [[buffer(2)]],
                          uint h   [[threadgroup_position_in_grid]],
                          uint ntg [[threadgroups_per_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const uint S = p.splits, hs = ntg * S;
  device const float* pm = parts + h * S;
  device const float* pl = parts + hs + h * S;
  device const float* pacc = parts + 2 * hs + (ulong)h * S * AD;
  float gm = -1e30f;
  for (uint s = 0; s < S; s++) gm = max(gm, pm[s]);
  float gl = 0.0f, o = 0.0f;
  for (uint s = 0; s < S; s++) {
    const float e = exp(pm[s] - gm);
    gl += pl[s] * e;
    o += pacc[(ulong)s * AD + tid] * e;
  }
  out[h * AD + tid] = o / gl;
}
template [[host_name("attn_combine_64_")]] kernel void
attn_combine_<64>(device const float*, device float*,
                  constant attn_combine_params&, uint, uint, uint);
template [[host_name("attn_combine_128_")]] kernel void
attn_combine_<128>(device const float*, device float*,
                   constant attn_combine_params&, uint, uint, uint);

// ---------------------------------------------------------------------------
// Cross-entropy and the optimizer: cuda's tl_gather_axis, tl_row_logsumexp,
// tl_xent_bwd and tl_adam_step, the same arithmetic.
// ---------------------------------------------------------------------------

struct gather_axis_params {
  uint size;
  uint n;
};

// out[i] = src[i*size + idx[i]]: the one element each position labels.
kernel void gather_axis_(device const float* src [[buffer(0)]],
                         device const float* idx [[buffer(1)]],
                         device float* out       [[buffer(2)]],
                         constant gather_axis_params& p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  out[i] = src[i * p.size + uint(idx[i] + 0.5f)];
}

// Row logsumexp in one pass: each thread carries a running (max, sum of exp
// below that max) over its columns and the tree merges the pairs.
kernel void row_logsumexp_(device const float* in [[buffer(0)]],
                           device float* out      [[buffer(1)]],
                           constant reduce_params& p [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
  constexpr uint T = 256;
  threadgroup float sm[T], ss[T];
  device const float* src = in + row * p.cols;
  float m = -FLT_MAX, s = 0.0f;
  for (uint c = lid; c < p.cols; c += T) {
    float v = src[c];
    if (v > m) {
      s *= exp(m - v);  // s is 0 on the first step, so -FLT_MAX is safe here
      m = v;
    }
    s += exp(v - m);
  }
  sm[lid] = m;
  ss[lid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = T / 2; h > 0; h >>= 1) {
    if (lid < h) {
      float m1 = sm[lid], s1 = ss[lid], m2 = sm[lid + h], s2 = ss[lid + h];
      float mm = max(m1, m2);
      ss[lid] = s1 * exp(m1 - mm) + s2 * exp(m2 - mm);
      sm[lid] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[row] = (sm[0] + log(ss[0])) * p.scale + p.offset;
}

struct xent_bwd_params {
  uint cols;
  uint n;
};

// dx[i,j] = g[i] · (exp(x[i,j] - lse[i]) - [j == target[i]]).
kernel void xent_bwd_(device const float* x   [[buffer(0)]],
                      device const float* lse [[buffer(1)]],
                      device const float* tgt [[buffer(2)]],
                      device const float* g   [[buffer(3)]],
                      device float* out       [[buffer(4)]],
                      constant xent_bwd_params& p [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  uint row = i / p.cols, col = i % p.cols;
  float e = exp(x[i] - lse[row]);
  if (col == uint(tgt[row] + 0.5f)) e -= 1.0f;
  out[i] = e * g[row];
}

struct adam_params {
  float b1, b2, eps, lr_over_bc1, inv_bc2;
  uint n;
};

// Adam's update in one pass: m and v advance, p moves by the bias-corrected
// ratio the host folded into lr_over_bc1 and inv_bc2.
kernel void adam_step_(device float* p_      [[buffer(0)]],
                       device float* m       [[buffer(1)]],
                       device float* v       [[buffer(2)]],
                       device const float* g [[buffer(3)]],
                       constant adam_params& p [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  float gi = g[i];
  float mi = p.b1 * m[i] + (1.0f - p.b1) * gi;
  float vi = p.b2 * v[i] + (1.0f - p.b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p_[i] -= (mi * p.lr_over_bc1) / (sqrt(vi * p.inv_bc2) + p.eps);
}
