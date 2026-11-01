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

// Params structs follow the shared kernel ABI (gpu_abi.h): 4-byte fields in
// the order gpu_ops.h declares them, which is the order the CUDA kernel of the
// same name takes its scalars.
struct ew_params {
  uint n;
  float scale;
  float offset;
};

// bf16 is the top 16 bits of the f32 pattern, widened by a shift. A bf16 weight is
// stored as ushort; the arithmetic is always f32.
static inline float widen_(float w) { return w; }
static inline float widen_(ushort h) { return as_type<float>(uint(h) << 16); }
// An f16 KV cache's element (caps::kv_f16): read exactly, written rounded to
// nearest even.
static inline float widen_(half h) { return float(h); }
// SiLU, g * sigmoid(g): one function for every kernel that applies it, so a
// fused kernel and the ones it replaces round it the same.
static inline float silu_(float g) { return g / (1.0f + exp(-g)); }
static inline void narrow_(device float* dst, float v) { *dst = v; }
static inline void narrow_(device half* dst, float v) { *dst = half(v); }
// Four consecutive elements as a float4 (i4 counts float4s).
static inline float4 load4_(device const float* p, uint i4) {
  return ((device const float4*)p)[i4];
}
static inline float4 load4_(device const half* p, uint i4) {
  return float4(((device const half4*)p)[i4]);
}

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
  uint M;
  uint N;
  uint ars, acs, brs, bcs;
  float scale;
  float offset;
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
  uint n;
  float lo, hi;
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
  uint n;
  float s, scale, offset;
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
  uint sa, sb;  // slice strides in floats (C's is M * N), grid z the slice
  uint ldc;     // C's row stride (STEEL's store; N but for a GEMM into columns)
};

// Slice tgid.z of a batched launch: the operands' and C's starts moved to it.
template <typename BT>
inline void gemm_slice_(device const float* thread& A,
                        device const BT* thread& B, device float* thread& C,
                        constant gemm_params& p, uint z) {
  A += size_t(z) * p.sa;
  B += size_t(z) * p.sb;
  C += size_t(z) * p.M * p.N;
}

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
    gemm_slice_(A, B, C, p, tgid.z);                                         \
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
// explicit two-element fragment registers (frag_type) instead of full
// simdgroup_matrix locals, loader pointers precomputed once (no div/mod in
// the K loop), serpentine MMA order, and a threadgroup swizzle for L2 reuse.
// Store epilogue is our affine (fma(acc, scale, offset)); interior/edge
// paths share one template (edge-tile bug class).
// ---------------------------------------------------------------------------

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

template <int N> struct Int { STEEL_CONST int value = N; constexpr operator int() const { return N; } };

// A lane's two elements of an 8×8 simdgroup_matrix: row frag_row_, columns
// frag_col_ and frag_col_ + 1.
static inline uint frag_row_(uint lane) {
  return ((lane / 4) & 4u) + ((lane / 2) % 4);
}
static inline uint frag_col_(uint lane) {
  return ((lane / 4) & 2u) * 2 + (lane % 2) * 2;
}

// MLX's padding: a tile's row ends 16 bytes past its data, so rows start on
// 16-byte boundaries and stagger across the threadgroup banks.
template <typename ST> constexpr short steel_pad_() { return 16 / sizeof(ST); }

// The tiles hold ST: float, or half for a GEMM summed in f16 (the MMA
// multiplies and sums in the tiles' type).
template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size,
          typename ST = float>
struct SteelLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup ST* dst;
  const device float* src;

  static_assert(vec_size % 4 == 0, "SteelLoader: whole float4 runs");

  METAL_FUNC SteelLoader(const device float* src_, int src_ld_,
                         threadgroup ST* dst_, ushort sid, ushort lane)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(sid * 32 + lane) / TCOLS),
        bj(vec_size * (short(sid * 32 + lane) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld_ + bj) {}

  // Row i's run, narrowed to the tile's type as it is stored.
  METAL_FUNC void copy_(short i) const {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j += 4)
      *((threadgroup vec<ST, 4>*)(&dst[i * dst_ld + j])) =
          vec<ST, 4>(*((const device float4*)(&src[i * src_ld + j])));
  }

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) copy_(i);
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
        copy_(i);
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
        copy_(i);
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++)
          dst[i * dst_ld + j] = j < valid_cols ? src[i * src_ld + j] : 0.0f;
      }
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
  METAL_FUNC void skip(int tiles) { src += long(tiles) * tile_stride; }
};

// Transposing BlockLoader (silarray port) — reads a BROWS x BCOLS tile of the
// SOURCE (vectorized along its contiguous BCOLS dim) and stores it TRANSPOSED
// into threadgroup memory, so the MMA keeps its NN layout while the device
// operand stays in its original (transposed-view) storage. Thread mapping is
// bi = idx % BROWS (one src row per thread): consecutive lanes scatter to
// consecutive threadgroup banks, keeping the transposed stores at worst
// 2-way bank-conflicted. All instantiations cover the tile in one pass. The
// source may be narrower than f32 (T: a bf16 weight's ushort): widened to
// f32 as it is read, then stored in the tile's type ST.
template <short BROWS, short BCOLS, short dst_ld, short reduction_dim, short tgp_size,
          typename T = float, typename ST = float>
struct SteelLoaderT {
  STEEL_CONST short vec_size = (BCOLS * BROWS) / tgp_size;
  static_assert(BCOLS == vec_size * (tgp_size / BROWS),
                "SteelLoaderT: tile must be covered in one pass");

  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  threadgroup ST* dst;
  const device T* src;

  struct alignas(16) ReadVec { T v[vec_size]; };

  METAL_FUNC SteelLoaderT(const device T* src_, int src_ld_,
                          threadgroup ST* dst_, ushort sid, ushort lane)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(short(sid * 32 + lane) % BROWS),
        bj(vec_size * (short(sid * 32 + lane) / BROWS)),
        dst(dst_ + bj * dst_ld + bi),
        src(src_ + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const {
    ReadVec v = *((const device ReadVec*)(&src[0]));
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = widen_(v.v[j]);
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
      dst[j * dst_ld] = valid ? widen_(src[valid ? j : 0]) : 0.0f;
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
        dst[j * dst_ld] = j < valid_cols ? widen_(src[j]) : 0.0f;
    }
  }

  METAL_FUNC void next() { src += tile_stride; }
  METAL_FUNC void skip(int tiles) { src += long(tiles) * tile_stride; }
};

// SteelLoaderT's tile and thread mapping over a source read through a
// functor f(n, k) rather than a pointer: a row of a GGUF block format,
// dequantized an element at a time, or a bf16 row whose K is not whole
// vectors. BROWS is B's N extent (n0 its first row), BCOLS the K step.
template <short BROWS, short BCOLS, short dst_ld, short tgp_size, typename F,
          typename ST = float>
struct SteelLoaderF {
  STEEL_CONST short vec_size = (BCOLS * BROWS) / tgp_size;
  static_assert(BCOLS == vec_size * (tgp_size / BROWS),
                "SteelLoaderF: tile must be covered in one pass");
  // A thread's run is a reader's f.vec: eight weights from a k that stays a
  // multiple of 8 (bj is 0 or 8, and k steps by BCOLS).
  static_assert(vec_size == 8 && BCOLS % 8 == 0, "SteelLoaderF: runs of eight");

  const F f;
  const short bi;
  const short bj;
  const uint n;
  uint k;
  threadgroup ST* dst;

  METAL_FUNC SteelLoaderF(F f_, uint n0, threadgroup ST* dst_, ushort sid,
                          ushort lane)
      : f(f_),
        bi(short(sid * 32 + lane) % BROWS),
        bj(vec_size * (short(sid * 32 + lane) / BROWS)),
        n(n0 + bi),
        k(bj),
        dst(dst_ + bj * dst_ld + bi) {}

  METAL_FUNC void load_unsafe() const {
    float v[vec_size];
    f.vec(n, k, v);
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = v[j];
  }
  METAL_FUNC void load_safe(short2 tile_dim) const {
    tile_dim -= short2(bj, bi);
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++)
      dst[j * dst_ld] = (tile_dim.y > 0 && j < tile_dim.x) ? f(n, k + j) : 0.0f;
  }
  METAL_FUNC void load_safe_rows(short valid_rows) const {
    if (bi < valid_rows) {
      load_unsafe();
    } else {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[j * dst_ld] = 0.0f;
    }
  }
  METAL_FUNC void load_safe_cols(short valid_cols) const {
    valid_cols -= bj;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < vec_size; j++)
      dst[j * dst_ld] = j < valid_cols ? f(n, k + j) : 0.0f;
  }
  METAL_FUNC void next() { k += BCOLS; }
  METAL_FUNC void skip(int tiles) { k += tiles * BCOLS; }
};

// Compile-time type selection (MSL has no <type_traits>)
template <bool C, typename Then, typename Else>
struct select_type { using type = Then; };
template <typename Then, typename Else>
struct select_type<false, Then, Else> { using type = Else; };

// steel_body_'s B loader for the tile at column col0: B a pointer (f32, or a
// bf16 weight's ushort, read in place, transposed with TB), or a functor
// f(n, k) over W [N, K] (TB only).
template <bool TB, short BN, short BK, short ld, short tgp, typename T, typename ST>
METAL_FUNC typename select_type<TB, SteelLoaderT<BN, BK, ld, true, tgp, T, ST>,
                                SteelLoader<BK, BN, ld, false, tgp, ST>>::type
steel_b_(device const T* B, constant gemm_params& p, uint col0,
         threadgroup ST* Bs, ushort sid, ushort lane) {
  return {B + (TB ? size_t(col0) * p.ldb : size_t(col0)), int(p.ldb), Bs, sid,
          lane};
}
template <bool TB, short BN, short BK, short ld, short tgp, typename F, typename ST>
METAL_FUNC SteelLoaderF<BN, BK, ld, tgp, F, ST> steel_b_(
    F f, constant gemm_params&, uint col0, threadgroup ST* Bs, ushort sid,
    ushort lane) {
  static_assert(TB, "a functor B is W [N, K], read transposed");
  return {f, col0, Bs, sid, lane};
}

template <short BM, short BN, short BK, short WM, short WN,
          short lda_tgp, short ldb_tgp, typename ST = float>
struct SteelMMA {
  typedef vec<ST, 2> frag_type;
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
      thread frag_type& dst, const threadgroup ST* src) {
    dst = *reinterpret_cast<const threadgroup frag_type*>(src);
  }
  METAL_FUNC static constexpr void load_frag_strided(
      thread frag_type& dst, const threadgroup ST* src, short str) {
    dst[0] = src[0];
    dst[1] = src[str];
  }

  METAL_FUNC static constexpr void frag_mma(
      thread frag_type& D, thread frag_type& A,
      thread frag_type& B, thread frag_type& C) {
    simdgroup_matrix<ST, 8, 8> A_mat, B_mat, C_mat;
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
  METAL_FUNC void mma(const threadgroup ST* As, const threadgroup ST* Bs) {
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
        float2 val = float2(Ctile[i * TN + j]);
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
        float2 val = float2(Ctile[i * TN + j]);
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
// view (backward-pass matmuls, attention's Q·Kᵀ); the MMA stays NN. B is
// whatever steel_b_ takes: a pointer, or a functor over W's rows. C's rows
// are p.ldc apart.
template <short BM, bool TA = false, bool TB = false, typename BSrc, typename ST>
void steel_body_(device const float* A, BSrc B,
                 device float* C, constant gemm_params& p,
                 threadgroup ST* As, threadgroup ST* Bs,
                 uint3 tgid, uint sid, uint lane) {
  constexpr short BN = 64, BK = 16, WM = 2, WN = 2, pad = steel_pad_<ST>();
  constexpr short tgp_size = WM * WN * 32;

  // The tile's origin and the operands' starts in 32 and 64 bits, so a
  // product past 32767 rows or columns (an lm_head over a vocabulary) reaches
  // its far tiles; within a tile, short and int suffice.
  uint swizzle = p.a_fast;
  uint tiles_n = (p.N + BN - 1) / BN;
  uint tiles_m = (p.M + BM - 1) / BM;
  uint tid_y = (tgid.y << swizzle) + (tgid.x & ((1u << swizzle) - 1));
  uint tid_x = tgid.x >> swizzle;
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  uint row0 = tid_y * BM, col0 = tid_x * BN;
  A += TA ? size_t(row0) : size_t(row0) * p.lda;

  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  typename select_type<TA, SteelLoaderT<BK, BM, lda_nn, false, tgp_size, float, ST>,
                       SteelLoader<BM, BK, lda_nn, true, tgp_size, ST>>::type
      loader_a(A, int(p.lda), As, sid, ushort(lane));
  auto loader_b = steel_b_<TB, BN, BK, ldb_nn, tgp_size>(
      B, p, col0, Bs, ushort(sid), ushort(lane));
  SteelMMA<BM, BN, BK, WM, WN, lda_nn, ldb_nn, ST> mma_op(sid, lane);

  int k_iters = int(p.K / BK);
  short lbk = short(p.K) - short(k_iters * BK);
  short tgp_bm = short(min(uint(BM), p.M - row0));
  short tgp_bn = short(min(uint(BN), p.N - col0));
  bool is_interior = (tgp_bm == BM && tgp_bn == BN);

  if (lbk > 0) {
    // K remainder first: jump loaders to the tail, one masked block, rewind.
    loader_a.skip(k_iters);
    loader_b.skip(k_iters);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // load_safe takes SOURCE-coordinate extents (x = cols, y = rows)
    loader_a.load_safe(TA ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm));
    loader_b.load_safe(TB ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.skip(-k_iters);
    loader_b.skip(-k_iters);
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

  int ldd = int(p.ldc);
  C += size_t(row0) * p.ldc + col0;
  if (is_interior) {
    mma_op.store_affine(C, ldd, p.scale, p.offset);
  } else {
    mma_op.store_affine_safe(C, ldd, short2(tgp_bn, tgp_bm), p.scale,
                             p.offset);
  }
}

// BT: B's element type (float here; a bf16 weight's ushort in the nn layer);
// BSRC what steel_body_ reads B through: B itself, or a reader f(n, k) over
// it (the nn layer's GGUF formats), built after the slice moved B; ST the
// tiles' type, which the product is summed in.
#define TL_STEEL_KERNEL_ST(name, BM, TA, TB, BT, BSRC, ST)                   \
  kernel void name(device const float* A [[buffer(0)]],                      \
                   device const BT* B [[buffer(1)]],                         \
                   device float* C [[buffer(2)]],                            \
                   constant gemm_params& p [[buffer(3)]],                    \
                   uint3 tgid [[threadgroup_position_in_grid]],              \
                   uint sid [[simdgroup_index_in_threadgroup]],              \
                   uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup ST As[BM * (16 + steel_pad_<ST>())];                         \
    threadgroup ST Bs[16 * (64 + steel_pad_<ST>())];                         \
    gemm_slice_(A, B, C, p, tgid.z);                                         \
    steel_body_<BM, TA, TB>(A, BSRC, C, p, As, Bs, tgid, sid, lane);         \
  }
#define TL_STEEL_KERNEL_SRC(name, BM, TA, TB, BT, BSRC)                      \
  TL_STEEL_KERNEL_ST(name, BM, TA, TB, BT, BSRC, float)
#define TL_STEEL_KERNEL(name, BM, TA, TB, BT)                                \
  TL_STEEL_KERNEL_SRC(name, BM, TA, TB, BT, B)

TL_STEEL_KERNEL(sgemm_steel_, 64, false, false, float)
TL_STEEL_KERNEL(sgemm_steel_32x64_, 32, false, false, float)
// Transposed-operand variants: _ta_ reads A's transposed view in place
// (dW = xᵀ @ g), _tb_ reads B's (dx = g @ Wᵀ, attention's Q·Kᵀ).
TL_STEEL_KERNEL(sgemm_steel_ta_, 64, true, false, float)
TL_STEEL_KERNEL(sgemm_steel_tb_, 64, false, true, float)
TL_STEEL_KERNEL(sgemm_steel_32x64_ta_, 32, true, false, float)
TL_STEEL_KERNEL(sgemm_steel_32x64_tb_, 32, false, true, float)

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

// argmax of one length-n vector, the smallest index on ties (the host scan's
// `v[i] > best`), so greedy decoding reads one index back instead of the
// logits, or leaves it on the device for the next step (gpu_ops_llm.h's
// argmax_to). Indices are f32, exact up to 2^24. Two launches of this kernel:
// groups reduce contiguous chunks of the logits to a (value, index) pair each,
// then one group reduces those pairs, each value's index taken from `idx`
// (`mapped`). Chunks run in index order, so the smaller index still wins a tie. One group alone took ~215 us over a 152k vocabulary
// on the M1 Pro, a decode step's 4%.
struct argmax_params {
  uint n, chunk, mapped;
};

kernel void argmax_(device const float* in      [[buffer(0)]],
                    device const float* idx     [[buffer(1)]],
                    device float* out_v         [[buffer(2)]],
                    device float* out_i         [[buffer(3)]],
                    constant argmax_params& p   [[buffer(4)]],
                    uint g    [[threadgroup_position_in_grid]],
                    uint tid  [[thread_index_in_threadgroup]],
                    uint nt   [[threads_per_threadgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sval[8];
  threadgroup int sidx[8];
  float best = -3.402823466e+38f;
  int besti = 0;
  const uint end = min(p.n, (g + 1) * p.chunk);
  for (uint i = g * p.chunk + tid; i < end; i += nt) {
    const float v = in[i];
    if (v > best) {
      best = v;
      besti = p.mapped ? int(idx[i]) : int(i);
    }
  }
  // Within a simdgroup, then across them: take the other if strictly greater,
  // or equal with a smaller index.
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(best, off);
    const int oi = simd_shuffle_down(besti, off);
    if (ov > best || (ov == best && oi < besti)) {
      best = ov;
      besti = oi;
    }
  }
  if (lane == 0) {
    sval[sgid] = best;
    sidx[sgid] = besti;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    for (uint w = 1; w < nt / 32; w++) {
      if (sval[w] > sval[0] || (sval[w] == sval[0] && sidx[w] < sidx[0])) {
        sval[0] = sval[w];
        sidx[0] = sidx[w];
      }
    }
    out_v[g] = sval[0];
    out_i[g] = float(sidx[0]);
  }
}
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
