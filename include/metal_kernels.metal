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
// MLX parity on this machine class). NN operands only; transposed views fall
// back to the sgemm_32/64x32 family. Differences from the simple tile above
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

    short qid = short(lane) / 4;
    short fm = (qid & 4) + ((short(lane) / 2) % 4);
    short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;

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
// ignores swizzle; STEEL ignores the fast flags).
template <short BM>
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
  A += row0 * int(p.lda);
  B += int(col0);

  constexpr short lda_nn = BK + pad, ldb_nn = BN + pad;
  SteelLoader<BM, BK, lda_nn, true, tgp_size> loader_a(A, int(p.lda), As, sid,
                                                       ushort(lane));
  SteelLoader<BK, BN, ldb_nn, false, tgp_size> loader_b(B, int(p.ldb), Bs, sid,
                                                        ushort(lane));
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
    loader_a.load_safe(short2(lbk, tgp_bm));
    loader_b.load_safe(short2(tgp_bn, lbk));
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
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe_rows(tgp_bm);
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
      loader_b.load_safe(short2(tgp_bn, BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.template mma<true>(As, Bs);
      loader_a.next();
      loader_b.next();
    }
  } else {
    for (int k = 0; k < k_iters; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe_rows(tgp_bm);
      loader_b.load_safe_cols(tgp_bn);
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

#define TL_STEEL_KERNEL(name, BM)                                            \
  kernel void name(device const float* A [[buffer(0)]],                      \
                   device const float* B [[buffer(1)]],                      \
                   device float* C [[buffer(2)]],                            \
                   constant gemm_params& p [[buffer(3)]],                    \
                   uint3 tgid [[threadgroup_position_in_grid]],              \
                   uint sid [[simdgroup_index_in_threadgroup]],              \
                   uint lane [[thread_index_in_simdgroup]]) {                \
    threadgroup float As[BM * 20];                                           \
    threadgroup float Bs[16 * 68];                                           \
    steel_body_<BM>(A, B, C, p, As, Bs, tgid, sid, lane);                    \
  }

TL_STEEL_KERNEL(sgemm_steel_, 64)
TL_STEEL_KERNEL(sgemm_steel_32x64_, 32)


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
