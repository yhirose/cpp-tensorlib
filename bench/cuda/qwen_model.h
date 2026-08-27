#pragma once
// Qwen2.5-0.5B-Instruct model: GGUF (F16) weight loading + decoder forward, shared
// by check_qwen (numeric validation) and chat_qwen (real text generation). The
// weights are loaded F32 (exact F16 widen), linear weights transposed from GGML's
// [out,in] to our dot's [K,N]. Structure: RMSNorm(eps=1e-6) -> QKV proj + bias ->
// RoPE(base=1e6, half-split) -> GQA(14q/2kv) attn via kv_cache -> o proj -> res ->
// RMSNorm -> SwiGLU MLP -> res, x24, then final RMSNorm -> logits (output.weight).
// CUDA-only (tl::cuda::kv_cache).

#include <tensorlib.h>
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qwenmodel {

using tl::array;
namespace gg = tl::gguf;

constexpr int64_t NL = 24, NE = 896, NH = 14, NKV = 2, HD = 64, FF = 4864;
constexpr int64_t VOCAB = 151936, MAXC = 4096;
// Prompt tokens per batched prefill pass. Sets the PrefillScratch size (linear:
// ~47 MiB here, which with the ~1.6 GiB of weights stays under the WSL2 ~2 GB
// sysmem-fallback cliff) and how well the projections fill the GPU — the GEMMs
// are still latency-bound at 256 (3.20 us/token) and reach 2.60 at 512.
constexpr int64_t PREFILL_CHUNK = 512;
// Below this the prompt is not worth a batched pass: the GEMM path only pays off
// once a chunk has enough rows to fill the SMs, and short prompts are dominated
// by the fixed per-chunk staging anyway.
constexpr int64_t PREFILL_MIN = 4;
constexpr float EPS = 1e-6f, SCALE = 1.0f / 8.0f /* 1/sqrt(64) */, ROPE_BASE = 1e6f;

// IEEE half -> float (GGUF F16 tensors). Handles subnormals/inf/nan.
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) { man <<= 1; exp--; }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float o;
  std::memcpy(&o, &f, 4);
  return o;
}

inline const gg::tensor_info* need(const gg::model& m, const std::string& n) {
  const auto* t = m.tensor(n);
  if (!t) { std::fprintf(stderr, "missing tensor %s\n", n.c_str()); std::exit(2); }
  return t;
}

// Fused weight load: widen several GGML [out,in] F16 tensors that share the same
// `in` (K) and stack them into ONE array, so a single GEMV/GEMM produces all
// their outputs contiguously (q|k|v or gate|up). Block b occupies output slots
// [base, base+out_b), and the fused output [Ntot] is q|k|v (or gate|up) laid end
// to end whichever layout is used — the downstream slices never change. Fusing
// is the M9 decode lever: fewer kernels amortize the ~14us per-launch floor.
// Imperative path only; the array path keeps the separate loads as its oracle.
//
// `row` picks the destination layout, which is the ONLY difference between the
// two forms — one place to read, so they cannot drift:
//   false: [in, Ntot] = [K,N] column-major, what the split-K GEMV and .dot()
//          expect. Bit-identical per column to loading each part separately
//          (same widen, same layout).
//   true:  [Ntot, in] = [N,K] row-major = GGML's own layout, so this drops the
//          transpose entirely — the warp-per-row GEMV (lever A) and the prefill
//          GEMM both read it. Its reduction order differs from split-K, so it is
//          greedy-equivalent to the array oracle, not bit-identical.
inline array load_w_cat_(const gg::model& m,
                         const std::vector<std::pair<std::string, int64_t>>& parts,
                         int64_t in, tl::dtype wdt, bool row) {
  int64_t Ntot = 0;
  for (const auto& p : parts) Ntot += p.second;
  std::vector<float> v((size_t)in * Ntot);
  int64_t base = 0;
  for (const auto& p : parts) {
    const auto* raw = reinterpret_cast<const uint16_t*>(need(m, p.first)->data);
    const int64_t out = p.second;
    // Split on the layout OUTSIDE the element loops: this runs over every weight
    // byte in the model, so the destination index must stay branch-free.
    if (row) {
      for (int64_t o = 0; o < out; o++)
        for (int64_t i = 0; i < in; i++)
          v[(size_t)(base + o) * in + i] = f16_to_f32(raw[(size_t)o * in + i]);
    } else {
      for (int64_t o = 0; o < out; o++)
        for (int64_t i = 0; i < in; i++)
          v[(size_t)i * Ntot + (base + o)] = f16_to_f32(raw[(size_t)o * in + i]);
    }
    base += out;
  }
  array a = array::from(std::move(v),
                        row ? tl::shape_t{Ntot, in} : tl::shape_t{in, Ntot});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// The four named entry points: {single, concatenated} x {[K,N], [N,K]}. The
// [K,N] pair takes the storage dtype — f32 (exact, ~2GB total -> WSL2 cliff) or
// bf16 (~1GB, stays under it; the decode GEMV consumes bf16 natively). The [N,K]
// pair does not: row-major weights exist only to feed the warp-per-row GEMV and
// the prefill GEMM, and both are bf16-only.
inline array load_w_T_cat(const gg::model& m,
                          const std::vector<std::pair<std::string, int64_t>>& parts,
                          int64_t in, tl::dtype wdt) {
  return load_w_cat_(m, parts, in, wdt, /*row=*/false);
}
inline array load_w_T_cat_row(
    const gg::model& m,
    const std::vector<std::pair<std::string, int64_t>>& parts, int64_t in) {
  return load_w_cat_(m, parts, in, tl::dtype::bf16, /*row=*/true);
}
inline array load_w_T(const gg::model& m, const std::string& n, int64_t in,
                      int64_t out, tl::dtype wdt) {
  return load_w_T_cat(m, {{n, out}}, in, wdt);
}
inline array load_w_T_row(const gg::model& m, const std::string& n, int64_t in,
                          int64_t out) {
  return load_w_T_cat_row(m, {{n, out}}, in);
}

// Allocate `n` f32 of device scratch (no host mirror — these buffers are only
// ever read by kernels). The scratch structs below are lists of these.
inline void* alloc_f32(int64_t n) { return tl::cuda::alloc(n * 4, nullptr); }

// Slice a device f32 buffer by element offset. The result is a mid-buffer
// pointer: reads through it are stream-ordered after whatever wrote the base and
// need no host sync, since the base is already device-live.
inline void* off_f32(void* p, int64_t nfloats) {
  return static_cast<char*>(p) + nfloats * 4;
}

// Several F32 1-D tensors laid end to end -> array [1, sum(len)]. The q|k|v
// biases in that order match the fused QKV projection's column layout, so the
// batched path adds all of them in one pass over all heads.
// F32 1-D tensors (norms and biases are stored F32 in the GGUF) -> array
// [1, sum(len)]; load_f32 below is the single-tensor case.
inline array load_f32_cat(const gg::model& m,
                          const std::vector<std::pair<std::string, int64_t>>& parts) {
  std::vector<float> v;
  for (const auto& p : parts) {
    const auto* raw = reinterpret_cast<const float*>(need(m, p.first)->data);
    v.insert(v.end(), raw, raw + p.second);
  }
  int64_t len = (int64_t)v.size();
  return array::from(std::move(v), {1, len});
}
inline array load_f32(const gg::model& m, const std::string& n, int64_t len) {
  return load_f32_cat(m, {{n, len}});  // the single-part case
}

struct Layer {
  array wq, bq, wk, bk, wv, bv, wo, wg, wu, wd, an, fn;
  // Fused decode weights (imperative path only): wqkv = [wq|wk|wv], wgu = [wg|wu]
  // concatenated. One GEMV each instead of 3/2, amortizing the per-launch floor.
  // The separate wq/.../wu stay resident as the array-path oracle.
  //
  // Layout depends on the storage dtype (set in build): in bf16 mode these hold
  // ROW-major [N,K] and feed the warp-per-row gemv_bf16_row (lever A, ~1.4-1.9x
  // on the small-N floor-bound shapes); wo_row/wd_row are the row-major copies of
  // the two shared weights (wo/wd stay [K,N] for the array oracle). In f32 mode
  // wqkv/wgu hold column-major [K,N] for the split-K path and wo_row/wd_row are
  // empty. lm_head stays split-K [K,N] in both (neutral + saves the [N,K] copy).
  // bqkv = [bq|bk|bv] — the fused QKV projection's own column order, so the
  // batched prefill splits all NH+2*NKV heads (and adds their bias) in one pass.
  array wqkv, wgu, bqkv, wo_row, wd_row;
  // q4 (group-symmetric int4, group=32) copies of the large bandwidth-bound MLP
  // weights (imperative bf16 path, when Model.q4_mlp). ~0.625 B/wt vs bf16's 2 →
  // wgu 1.74×, wd 1.38× in the isolated bench. These REPLACE wgu/wd_row (the row
  // bf16 fields stay empty), saving memory. Attention proj (QKV.fused, wo) stays
  // bf16-row — q4 loses there (still floor-bound; dequant > byte savings).
  array wgu_q4, wd_q4;
  tl::cuda::kv_cache cache;
};

// Persistent device scratch for the imperative decode step (C1): every per-step
// intermediate is a device buffer allocated ONCE and reused each token, so the
// step builds zero array nodes (no host graph construction) and does zero
// per-step allocation (no WSL2 churn). res[2] ping-pong the residual stream;
// q/k/v and gate/up live only as slices of their fused projection outputs
// (qkvb, gub).
struct Scratch {
  void* res[2] = {nullptr, nullptr};  // residual ping-pong [NE]
  void* hb = nullptr;                 // input-norm out [NE]
  void* h2b = nullptr;                // post-attn norm out [NE]
  void* qb = nullptr;                 // [NH*HD] query fixture (bench_qwen_ctx's
                                      // isolated-attention timing; decode reads
                                      // q as a slice of qkvb)
  void* qkvb = nullptr;               // fused QKV out [(NH+2*NKV)*HD] = [1152]
  void* ab = nullptr;                 // attn out [NH*HD]
  void* mb = nullptr;                 // swiglu out [FF]
  void* gub = nullptr;                // fused gate|up out [2*FF] = [9728]
  void* mdb = nullptr;                // mlp-down out [NE]
  void* logitsb = nullptr;            // [VOCAB]
  void* logits_scratch = nullptr;     // [VOCAB] sink for forwards run only for
                                      // their side effects (capture warm-up)
  float* logits_host = nullptr;       // host mirror of logitsb (divergence checks)
  void* embedb = nullptr;             // staged embedding row [NE] (capture input)
  float* embed_host = nullptr;        // embedb's own host mirror (gather target)
  void init() {
    embedb = tl::cuda::alloc(NE * 4, &embed_host);  // host mirror: gather target
    logitsb = tl::cuda::alloc(VOCAB * 4, &logits_host);
    res[0] = alloc_f32(NE);
    res[1] = alloc_f32(NE);
    hb = alloc_f32(NE);
    h2b = alloc_f32(NE);
    qb = alloc_f32(NH * HD);
    qkvb = alloc_f32((NH + 2 * NKV) * HD);
    ab = alloc_f32(NH * HD);
    mb = alloc_f32(FF);
    gub = alloc_f32(2 * FF);
    mdb = alloc_f32(NE);
    logits_scratch = alloc_f32(VOCAB);
  }
};

// Device scratch for the BATCHED prefill (M9): the same per-token intermediates
// as Scratch, but `cap` rows wide, so a chunk of prompt tokens flows through the
// forward as one [T, n] matrix. Sized once for the chunk and reused across
// chunks and across prompts. ~47 MiB at PREFILL_CHUNK, dominated by the fused
// gate|up buffer (cap x 2*FF) — which is why a long prompt is chunked at all
// rather than run in one [T_prompt, ...] pass under the WSL2 ~2 GB cliff.
struct PrefillScratch {
  int64_t cap = 0;
  void* emb = nullptr;                // embedding rows [cap, NE]
  void* res[2] = {nullptr, nullptr};  // residual ping-pong [cap, NE]
  void* h = nullptr;                  // input-norm out [cap, NE]
  void* h2 = nullptr;                 // post-attn norm out [cap, NE]
  void* qkv = nullptr;                // fused QKV out [cap, (NH+2*NKV)*HD]
  // q|k|v head-major in ONE buffer [NH+2*NKV, cap, HD]: the fused projection
  // already emits them contiguously per token, so one split_heads pass covers
  // every head, and k/v are just mid-buffer pointers into it.
  void* qkvh = nullptr;
  void* ah = nullptr;                 // attn out head-major [NH, cap, HD]
  void* at = nullptr;                 // attn out token-major [cap, NH*HD]
  void* gu = nullptr;                 // fused gate|up [cap, 2*FF]
  void* mb = nullptr;                 // swiglu out [cap, FF]
  void* md = nullptr;                 // mlp-down out [cap, NE]
  float* emb_host = nullptr;          // emb's own host mirror (gather target)

  bool init(int64_t chunk) {
    if (cap >= chunk) return true;  // already big enough
    destroy();
    cap = chunk;
    emb = tl::cuda::alloc(cap * NE * 4, &emb_host);
    res[0] = alloc_f32(cap * NE);
    res[1] = alloc_f32(cap * NE);
    h = alloc_f32(cap * NE);
    h2 = alloc_f32(cap * NE);
    qkv = alloc_f32(cap * (NH + 2 * NKV) * HD);
    qkvh = alloc_f32((NH + 2 * NKV) * cap * HD);
    ah = alloc_f32(NH * cap * HD);
    at = alloc_f32(cap * NH * HD);
    gu = alloc_f32(cap * 2 * FF);
    mb = alloc_f32(cap * FF);
    md = alloc_f32(cap * NE);
    return md != nullptr;
  }
  void destroy() {
    for (void** p : {&emb, &res[0], &res[1], &h, &h2, &qkv, &qkvh, &ah, &at,
                     &gu, &mb, &md}) {
      if (*p) tl::cuda::release(*p, 0, nullptr);
      *p = nullptr;
    }
    emb_host = nullptr;
    cap = 0;
  }
};

struct Model {
  const uint16_t* embed_f16;  // token_embd raw F16 [vocab, NE] (row-gather source)
  array outwT;                // logits weight [NE, VOCAB] (transposed output.weight)
  array onorm;                // final norm [1, NE]
  array outwT_q4;             // q4 lm_head (imperative, when q4_lmhead); outwT
                              // stays bf16 as the array-path oracle
  std::vector<Layer> layers;
  Scratch scratch;
  PrefillScratch pscratch;  // batched-prefill buffers (lazily sized)
  bool row = false;  // imperative gemvs use warp-per-row [N,K] (bf16 only)
  bool q4_mlp = false;     // imperative MLP gemvs (wgu, wd) use q4
  bool q4_lmhead = false;  // imperative lm_head gemv uses q4
};

// Decode GEMV picking the weight-dtype kernel: y(n) = a(1,k) @ W[k,n].
inline bool gemv_w(const array& W, void* a, void* y, int64_t n, int64_t k) {
  return W.dt() == tl::dtype::bf16
             ? tl::cuda::gemv_bf16(a, W.native(), y, n, k)
             : tl::cuda::gemv_f32(a, W.native(), y, n, k);
}

// Greedy token from the logits the most recent forward left in scratch (the one
// terminal sync of a step lives inside cuda::argmax).
inline int64_t argmax_logits(Model& M) {
  int64_t idx = 0;
  tl::cuda::argmax(M.scratch.logitsb, VOCAB, &idx);
  return idx;
}

// Reset all KV caches to position 0 (replay from a fresh prefill). Used by the
// bench's array-vs-imperative greedy-equivalence check.
inline void reset_cache(Model& M) {
  for (auto& L : M.layers) L.cache.pos = 0;
}
// Set all KV caches' host pos. The device-pos (capture) path leaves host pos
// untouched — a captured replay advances only the device counter — so the
// orchestrator sets host pos explicitly for state bookkeeping / the dpos-vs-host
// correctness check (every layer is at the same sequence position).
inline void set_cache_pos(Model& M, int64_t p) {
  for (auto& L : M.layers) L.cache.pos = p;
}

// ---- M9 batched prefill: run T prompt tokens through the layer stack at once.
//
// Decode is a GEMV regime (one token, weights read once per token, bandwidth-
// bound). A prompt is not: T tokens share the same weights, so every projection
// becomes a GEMM whose cost per token collapses (~25x measured across the four
// Qwen shapes in bench_qwen_prefill), and attention becomes one tiled causal
// pass instead of T growing-ctx decode calls (~45x). This is the same forward as
// run_layers_, one row per token: fused QKV -> split to head-major -> rope ->
// causal attn over the cache -> merge back -> o proj -> fused gate|up -> swiglu
// -> down, with the residual adds folded into the following RMSNorms exactly as
// the decode seam does.
//
// The chunk's absolute start position comes from the caches (kv_cache::prefill
// appends there; rope reads the same source), so chunks — and later turns —
// compose. No logits here: prefill_batched runs the lm_head once, on the final
// chunk's last row, since a prompt needs no others.
//
// bf16 row-major weights only (the layout gemm_bf16_nt shares with the decode
// GEMV); callers check Model.row and fall back to token-by-token otherwise.
inline void prefill_chunk_(Model& M, int64_t T) {
  namespace cu = tl::cuda;
  PrefillScratch& P = M.pscratch;
  // Absolute position of this chunk's first token — the caches are all at the
  // same place, and kv_cache::prefill appends there, so rope reads it from the
  // same source rather than taking it as a parameter that could disagree.
  const int64_t pos0 = M.layers[0].cache.pos;
  constexpr int64_t QKVN = (NH + 2 * NKV) * HD;
  void* x = P.emb;
  cu::rmsnorm(x, M.layers[0].an.native(), P.h, NE, EPS, T);
  for (int64_t l = 0; l < NL; l++) {
    Layer& L = M.layers[l];
    void* ro = P.res[l & 1];
    cu::gemm_bf16_nt(P.h, L.wqkv.native(), P.qkv, T, QKVN, NE);
    // One pass turns the fused [T, q|k|v] output into head-major [18, T, D] and
    // adds the fused bias — rope's own fused-bias form only indexes correctly at
    // T == 1, so the bias rides along here instead.
    cu::split_heads(P.qkv, L.bqkv.native(), P.qkvh, T, QKVN, 0, NH + 2 * NKV, HD);
    void* kh = off_f32(P.qkvh, NH * T * HD);
    void* vh = off_f32(P.qkvh, (NH + NKV) * T * HD);
    // [H,T,D] flattened: row r = h*T + t, so rope's `pos + r % T` is pos0 + t.
    cu::rope(P.qkvh, P.qkvh, NH * T, T, HD, pos0, ROPE_BASE);
    cu::rope(kh, kh, NKV * T, T, HD, pos0, ROPE_BASE);
    L.cache.prefill(P.qkvh, kh, vh, P.ah, T, NH, SCALE);
    cu::merge_heads(P.ah, P.at, T, NH, HD);
    cu::gemm_bf16_nt(P.at, L.wo_row.native(), ro, T, NE, NH * HD);
    cu::rmsnorm_res(ro, x, L.fn.native(), ro, P.h2, NE, EPS, T);
    cu::gemm_bf16_nt(P.h2, L.wgu.native(), P.gu, T, 2 * FF, NE);
    cu::swiglu(P.gu, P.mb, FF, T);
    cu::gemm_bf16_nt(P.mb, L.wd_row.native(), P.md, T, NE, FF);
    void* nextw = (l + 1 < NL) ? M.layers[l + 1].an.native() : M.onorm.native();
    cu::rmsnorm_res(ro, P.md, nextw, ro, P.h, NE, EPS, T);
    x = ro;
  }
  // Leaves P.h holding the final RMSNorm for every row of the chunk. Logits are
  // the caller's business: only the LAST chunk's last row is ever read, so
  // running the 272 MB lm_head here would waste a full GEMV per earlier chunk.
}

// Widen `n` token embedding rows (F16 in the GGUF) into `dst`. The one place
// that knows the embedding's storage convention; both stagers go through it.
inline void gather_embed_rows_(const Model& M, const int* ids, int64_t n,
                               float* dst) {
  for (int64_t i = 0; i < n; i++) {
    const uint16_t* r = M.embed_f16 + (size_t)ids[i] * NE;
    for (int64_t j = 0; j < NE; j++) dst[i * NE + j] = f16_to_f32(r[j]);
  }
}

// Stage `n` prompt embeddings starting at ids[from] into the batched scratch.
inline void stage_embed_rows_(Model& M, const std::vector<int>& ids, int64_t from,
                              int64_t n) {
  float* v = M.pscratch.emb_host;
  gather_embed_rows_(M, ids.data() + from, n, v);
  tl::cuda::upload(M.pscratch.emb, v, n * NE);
}

// Whether a prompt of `T` tokens should go through the batched prefill: the
// weights must be the bf16 row-major ones its GEMM consumes, the prompt must be
// long enough to be worth a batched pass, and it must fit the cache. One place
// answers it, so build(), begin() and prefill_batched() cannot disagree.
inline bool can_prefill_batched(const Model& M, int64_t T) {
  return M.row && !M.q4_mlp && !M.layers.empty() && T >= PREFILL_MIN &&
         M.layers[0].cache.pos + T <= MAXC;
}

// Prefill a whole prompt in chunks and return the greedy next token. Leaves the
// caches at pos_before + ids.size(), so a decoder captured afterwards continues
// from there. Returns -1 if the batched path is unavailable or the prompt does
// not fit the cache.
inline int64_t prefill_batched(Model& M, const std::vector<int>& ids,
                               int64_t chunk = PREFILL_CHUNK) {
  const int64_t T = (int64_t)ids.size();
  if (!can_prefill_batched(M, T) || !M.pscratch.init(chunk)) return -1;
  int64_t last = 0;
  for (int64_t s = 0; s < T; s += chunk) {
    last = std::min(chunk, T - s);
    stage_embed_rows_(M, ids, s, last);
    prefill_chunk_(M, last);
  }
  // Only the final row of the final chunk needs logits — one GEMV for the whole
  // prompt rather than one per chunk.
  gemv_w(M.outwT, off_f32(M.pscratch.h, (last - 1) * NE), M.scratch.logitsb,
         VOCAB, NE);
  return argmax_logits(M);
}

// wdt = the linear-weight storage dtype (f32 exact, or bf16 for the <2GB/decode
// path). Biases, norms and the embedding stay F32 (elementwise / row-gather).
// q4_mlp / q4_lmhead quantize the large bandwidth-bound gemvs to group-int4 in
// the imperative decode path (bf16 base only). Lossy — greedy diverges from the
// F16 oracle; validated by coherence + divergence, not greedy-exact.
inline Model build(const gg::model& m, tl::dtype wdt = tl::dtype::f32,
                   bool q4_mlp = false, bool q4_lmhead = false) {
  const bool row = (wdt == tl::dtype::bf16);  // warp-per-row [N,K] imperative gemvs
  const bool q4m = q4_mlp && row;             // q4 requires the bf16/row base
  const bool q4lm = q4_lmhead && row;
  const tl::dtype f32 = tl::dtype::f32;
  Model M{
      .embed_f16 =
          reinterpret_cast<const uint16_t*>(need(m, "token_embd.weight")->data),
      .outwT = load_w_T(m, "output.weight", NE, VOCAB, wdt),  // stays [K,N]
      .onorm = load_f32(m, "output_norm.weight", NE),
      // q4 lm_head (imperative): quantize the f32 [K,VOCAB], keep only the q4.
      .outwT_q4 =
          q4lm ? load_w_T(m, "output.weight", NE, VOCAB, f32).to_q4() : array{},
      .row = row,
      .q4_mlp = q4m,
      .q4_lmhead = q4lm};
  for (int64_t l = 0; l < NL; l++) {
    std::string p = "blk." + std::to_string(l) + ".";
    // Named per field: the ternaries below decide LAYOUT, and a positional list
    // of 17 same-typed arrays is one transposed line away from a silent
    // wrong-weight bug that only shows up as slightly-off logits.
    const std::vector<std::pair<std::string, int64_t>> qkv_parts = {
        {p + "attn_q.weight", NH * HD},
        {p + "attn_k.weight", NKV * HD},
        {p + "attn_v.weight", NKV * HD}};
    const std::vector<std::pair<std::string, int64_t>> gu_parts = {
        {p + "ffn_gate.weight", FF}, {p + "ffn_up.weight", FF}};
    Layer L{
        // The array path's own weights ([K,N], the .dot() layout) — the oracle.
        .wq = load_w_T(m, p + "attn_q.weight", NE, NH * HD, wdt),
        .bq = load_f32(m, p + "attn_q.bias", NH * HD),
        .wk = load_w_T(m, p + "attn_k.weight", NE, NKV * HD, wdt),
        .bk = load_f32(m, p + "attn_k.bias", NKV * HD),
        .wv = load_w_T(m, p + "attn_v.weight", NE, NKV * HD, wdt),
        .bv = load_f32(m, p + "attn_v.bias", NKV * HD),
        .wo = load_w_T(m, p + "attn_output.weight", NH * HD, NE, wdt),
        .wg = load_w_T(m, p + "ffn_gate.weight", NE, FF, wdt),
        .wu = load_w_T(m, p + "ffn_up.weight", NE, FF, wdt),
        .wd = load_w_T(m, p + "ffn_down.weight", FF, NE, wdt),
        .an = load_f32(m, p + "attn_norm.weight", NE),
        .fn = load_f32(m, p + "ffn_norm.weight", NE),
        // Fused QKV and gate|up (imperative decode path). In bf16 mode: ROW
        // [N,K] for the warp-per-row gemv (lever A). In f32: column [K,N] for
        // split-K. +466MB over 24 layers either way. wo_row/wd_row (bf16 only)
        // are the row copies of the two shared weights (+247MB), keeping the
        // total under the WSL2 ~2GB cliff (lm_head deliberately not copied).
        .wqkv = row ? load_w_T_cat_row(m, qkv_parts, NE)
                    : load_w_T_cat(m, qkv_parts, NE, wdt),
        // wgu: bf16-row when row & !q4m; column [K,N] in f32; empty when q4m
        // (replaced by wgu_q4). wd_row likewise.
        .wgu = row ? (q4m ? array{} : load_w_T_cat_row(m, gu_parts, NE))
                   : load_w_T_cat(m, gu_parts, NE, wdt),
        // bqkv = [bq|bk|bv], matching wqkv's column order (batched path).
        .bqkv = row ? load_f32_cat(m, {{p + "attn_q.bias", NH * HD},
                                       {p + "attn_k.bias", NKV * HD},
                                       {p + "attn_v.bias", NKV * HD}})
                    : array{},
        .wo_row =
            row ? load_w_T_row(m, p + "attn_output.weight", NH * HD, NE) : array{},
        .wd_row = row ? (q4m ? array{}
                             : load_w_T_row(m, p + "ffn_down.weight", FF, NE))
                      : array{},
        // q4 MLP (imperative, when q4m): quantize the f32 [K,N] fused/down.
        .wgu_q4 = q4m ? load_w_T_cat(m, gu_parts, NE, f32).to_q4() : array{},
        .wd_q4 =
            q4m ? load_w_T(m, p + "ffn_down.weight", FF, NE, f32).to_q4() : array{}};
    L.cache.init(NKV, MAXC, HD);
    M.layers.push_back(std::move(L));
  }
  M.scratch.init();  // persistent imperative-step buffers (GPU must be active)
  // Pre-size AND warm the batched-prefill path. CUDA loads a kernel lazily on
  // first use, so without this the first prompt pays the module load plus the
  // scratch allocation — ~90 ms, several times its own compute. Both are fixed
  // per-run costs, so they belong here beside the weight upload; the price is
  // that a decode-only consumer also carries the scratch (see PrefillScratch).
  // It goes through prefill_batched — which owns the eligibility check and
  // no-ops with -1 when the weights can't batch — so every kernel the real
  // entry point touches, the lm_head GEMV included, is loaded here. The warm
  // pass dirties cache rows [0, PREFILL_MIN) of every layer, which the reset
  // releases: attention only ever reads [0, pos), so the stale rows are
  // unreachable.
  if (prefill_batched(M, std::vector<int>((size_t)PREFILL_MIN, 0)) >= 0) {
    tl::cuda::flush();
    reset_cache(M);
  }
  return M;
}

inline array embed_row(const Model& M, int64_t id) {
  std::vector<float> v(NE);
  const int one = (int)id;
  gather_embed_rows_(M, &one, 1, v.data());
  return array::from(std::move(v), {1, NE});
}

// Per-region decode-step profile (roadmap M9 overhead census). Accumulates
// host wall-clock ms across steps; `n` counts steps. Zero cost when the step()
// prof arg is null (the default). qkv_eval/x2_eval isolate the per-layer forced
// syncs (each .eval() = graph::run + gpu::flush = a CtxSynchronize) — the C2
// (sync-free realize) target — from the launch-only work; logits_d2h isolates
// the 608KB VOCAB->host copy that GPU argmax removes.
struct StepProf {
  double embed = 0, construct = 0, cache = 0, qkv_eval = 0, x2_eval = 0,
         logits_eval = 0, logits_d2h = 0;
  int64_t n = 0;
  static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
};

// Embed + 24 decoder layers + final RMSNorm, returning the unevaluated logits
// array [1, VOCAB]. Shared by step() (host D2H, the numeric-oracle path) and
// step_greedy() (GPU argmax, the generation path). Optional out-params record
// the layer-0 residual / final-norm for checkpointing; `prof` accumulates the
// per-region overhead census.
inline array forward(Model& M, int64_t id, int64_t pos,
                     std::vector<float>* l0_out, std::vector<float>* fnorm_out,
                     StepProf* prof) {
  double t = prof ? StepProf::now_ms() : 0.0;
  array x = embed_row(M, id);
  if (prof) { prof->embed += StepProf::now_ms() - t; }
  for (int64_t l = 0; l < NL; l++) {
    Layer& L = M.layers[l];
    if (prof) t = StepProf::now_ms();
    array h = array::rmsnorm(x, L.an, EPS);
    array q = array::rope((h.dot(L.wq) + L.bq).reshape({NH, HD}), pos, ROPE_BASE);
    array k = array::rope((h.dot(L.wk) + L.bk).reshape({NKV, HD}), pos, ROPE_BASE);
    array v = h.dot(L.wv) + L.bv;  // [1, NKV*HD]
    if (prof) { prof->construct += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    // realize() (not eval()): launch q/k/v on the null stream WITHOUT syncing;
    // the kv_cache append/attn kernels are stream-ordered after them, so they
    // see the writes. Removes 3 CtxSynchronize/layer.
    q.realize(); k.realize(); v.realize();
    if (prof) { prof->qkv_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    L.cache.append(k.native(), v.native());
    array a_out = array::empty({NH, HD});
    L.cache.attn(q.native(), a_out.native(), NH, SCALE);
    if (prof) { prof->cache += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    array x1 = x + a_out.reshape({1, NE}).dot(L.wo);
    array h2 = array::rmsnorm(x1, L.fn, EPS);
    array mlp = array::swiglu(h2.dot(L.wg), h2.dot(L.wu)).dot(L.wd);
    array x2 = x1 + mlp;
    if (prof) { prof->construct += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    // realize() keeps x2 in flight (no sync) but adopts its storage, so the
    // next layer reads it by stream order and the graph stays bounded. Removes
    // the 4th CtxSynchronize/layer; the whole step drains at the logits tail.
    x2.realize();
    if (prof) { prof->x2_eval += StepProf::now_ms() - t; }
    if (l == 0 && l0_out) { const float* p = x2.raw(); l0_out->assign(p, p + NE); }
    x = x2;
  }
  array xf = array::rmsnorm(x, M.onorm, EPS);
  if (fnorm_out) { xf.eval(); const float* p = xf.raw(); fnorm_out->assign(p, p + NE); }
  return xf.dot(M.outwT);
}

// One decode step. Returns logits [VOCAB] on the host — the numeric-oracle path
// (check_qwen compares the full vector). Pays the 608KB VOCAB D2H.
inline std::vector<float> step(Model& M, int64_t id, int64_t pos,
                               std::vector<float>* l0_out = nullptr,
                               std::vector<float>* fnorm_out = nullptr,
                               StepProf* prof = nullptr) {
  array logits = forward(M, id, pos, l0_out, fnorm_out, prof);
  double t = prof ? StepProf::now_ms() : 0.0;
  logits.eval();
  if (prof) { prof->logits_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
  const float* p = logits.raw();
  std::vector<float> out(p, p + VOCAB);
  if (prof) { prof->logits_d2h += StepProf::now_ms() - t; prof->n++; }
  return out;
}

// One decode step returning the greedy token directly — the generation path.
// GPU argmax collapses the per-token host transfer from 608KB (full logits) to
// 4 bytes (the index). Bit-identical to argmax(step(...)) by construction (the
// kernel's tie-break matches the host loop). Falls back to a host scan if the
// GPU argmax is unavailable (non-CUDA build).
inline int64_t step_greedy(Model& M, int64_t id, int64_t pos,
                           StepProf* prof = nullptr) {
  array logits = forward(M, id, pos, nullptr, nullptr, prof);
  double t = prof ? StepProf::now_ms() : 0.0;
  // realize() (not eval()): the lm_head gemv launches without syncing; the
  // argmax kernel is stream-ordered after it and cuda::argmax does the single
  // terminal CtxSynchronize before its 4-byte D2H. One sync for the whole step.
  logits.realize();
  if (prof) { prof->logits_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
  int64_t idx = 0;
  if (!tl::cuda::argmax(logits.native(), VOCAB, &idx)) {
    const float* p = logits.raw();
    idx = 0;
    for (int64_t i = 1; i < VOCAB; i++)
      if (p[i] > p[idx]) idx = i;
  }
  if (prof) { prof->logits_d2h += StepProf::now_ms() - t; prof->n++; }
  return idx;
}

inline int64_t argmax(const std::vector<float>& v) {
  int64_t bi = 0;
  for (int64_t i = 1; i < (int64_t)v.size(); i++)
    if (v[i] > v[bi]) bi = i;
  return bi;
}

// q4 decode GEMV: y(N) = a(1,K) @ dequant(Wq). Wq is a q4 array (logical [K,N],
// storage [packed [N][K/2] | scales [N][K/32]]); the scales pointer is mid-buffer
// (rides along the base upload — same split as the array-path gpu_gemv_q4).
inline bool gemv_q4_w(const array& Wq, void* a, void* y) {
  const int64_t K = Wq.shape()[0], N = Wq.shape()[1];
  void* scales = static_cast<char*>(Wq.native()) + N * K / 2;
  return tl::cuda::gemv_q4(a, Wq.native(), scales, y, N, K, tl::kQ4Group);
}

// The 24 decoder layers + final RMSNorm + lm_head gemv as direct cuda:: calls
// on the Scratch buffers, reading layer-0 residual from x0 and writing logits
// to S.logitsb. NO array nodes, NO host sync / blocking copy — so this region
// is CUDA-graph-capturable (embed staging + argmax stay outside). All kernels
// target context.stream (default null, or the capture stream during recording).
// logits_out (optional): where the lm_head writes. Defaults to the scratch
// logits every reader expects; a caller running the forward only for its side
// effects (capture warm-up) points it elsewhere so the buffer keeps whatever the
// last real forward left there.
// d_pos (optional): a device u32 scalar holding the decode position. When set,
// the three pos-dependent ops (rope, kv append, decode attn) read it from the
// device instead of the host `pos`, and a tl_incr_u32 at the tail advances it —
// so the whole forward is CUDA-graph-capturable and one instantiated graph
// replays correctly as pos grows (A-min). All caches share the one counter (every
// layer is at the same sequence position). d_pos==nullptr = the normal host path.
inline void run_layers_(Model& M, void* x0, int64_t pos, void* d_pos = nullptr,
                        void* logits_out = nullptr) {
  namespace cu = tl::cuda;
  const bool cap = d_pos != nullptr;
  Scratch& S = M.scratch;
  // Fused seams (kills 4 elementwise launches/layer): q/k bias folds into rope
  // (cu::rope bias arg); the two residual adds fold into the following RMSNorms
  // via rmsnorm_res (writes the residual sum AND its norm). The layer input's
  // norm is therefore produced by the *previous* layer's mlp seam — so precompute
  // layer 0's here, and carry each layer's next-input norm in hb.
  // Decode GEMV: warp-per-row [N,K] (lever A, bf16 imperative) in row mode, else
  // the split-K [K,N] path — so `W` must be the weight in THIS mode's layout
  // (the fused wqkv/wgu already are; wo/wd have a separate row copy). The y
  // layout is identical either way (contiguous [n]), so the fused-output slices
  // below are unchanged.
  const bool row = M.row;
  auto gv = [&](const array& W, void* a, void* y, int64_t n, int64_t k) {
    if (row) tl::cuda::gemv_bf16_row(a, W.native(), y, n, k);
    else gemv_w(W, a, y, n, k);
  };
  void* x = x0;
  cu::rmsnorm(x, M.layers[0].an.native(), S.hb, NE, EPS);
  for (int64_t l = 0; l < NL; l++) {
    Layer& L = M.layers[l];
    void* ro = S.res[l & 1];  // res_out (x1 then x2) for this layer (ping-pong)
    // Fused QKV: one GEMV -> [q(NH*HD) | k(NKV*HD) | v(NKV*HD)] in S.qkvb, then
    // slice: rope q & k in place, bias-add v. Same per-column split-K as the
    // separate wq/wk/wv GEMVs (bx=1, chunk=32), so bit-identical per column.
    gv(L.wqkv, S.hb, S.qkvb, (NH + 2 * NKV) * HD, NE);
    void* qp = S.qkvb;
    void* kp = off_f32(S.qkvb, NH * HD);
    void* vp = off_f32(S.qkvb, (NH + NKV) * HD);
    if (cap) {
      cu::rope_dpos(qp, qp, NH, 1, HD, d_pos, ROPE_BASE, L.bq.native());
      cu::rope_dpos(kp, kp, NKV, 1, HD, d_pos, ROPE_BASE, L.bk.native());
    } else {
      cu::rope(qp, qp, NH, 1, HD, pos, ROPE_BASE, L.bq.native());
      cu::rope(kp, kp, NKV, 1, HD, pos, ROPE_BASE, L.bk.native());
    }
    cu::binary(cu::kop::add, vp, 0, L.bv.native(), 0, vp, 0, NKV * HD, 1, 0);
    if (cap) {
      L.cache.append_dpos(kp, vp, d_pos);
      L.cache.attn_dpos(qp, S.ab, NH, d_pos, SCALE);
    } else {
      L.cache.append(kp, vp);
      L.cache.attn(qp, S.ab, NH, SCALE);
    }
    gv(row ? L.wo_row : L.wo, S.ab, ro, NE, NH * HD);  // ro = attn @ wo
    // x1 = x + (attn@wo); h2 = rmsnorm(x1, fn) — fused.
    cu::rmsnorm_res(ro, x, L.fn.native(), ro, S.h2b, NE, EPS);
    // Fused gate|up: one GEMV -> [gate(FF) | up(FF)] in S.gub; swiglu reads both.
    if (M.q4_mlp) gemv_q4_w(L.wgu_q4, S.h2b, S.gub);  // [K=NE, N=2*FF]
    else gv(L.wgu, S.h2b, S.gub, 2 * FF, NE);
    cu::swiglu(S.gub, S.mb, FF);
    if (M.q4_mlp) gemv_q4_w(L.wd_q4, S.mb, S.mdb);    // [K=FF, N=NE]
    else gv(row ? L.wd_row : L.wd, S.mb, S.mdb, NE, FF);
    // x2 = x1 + mlp; next input norm = rmsnorm(x2, next an | final onorm) — fused.
    void* nextw = (l + 1 < NL) ? M.layers[l + 1].an.native() : M.onorm.native();
    cu::rmsnorm_res(ro, S.mdb, nextw, ro, S.hb, NE, EPS);
    x = ro;
  }
  // hb now holds the final RMSNorm output (folded into the last layer's seam).
  void* logits = logits_out ? logits_out : S.logitsb;
  if (M.q4_lmhead) gemv_q4_w(M.outwT_q4, S.hb, logits);  // [K=NE, N=VOCAB]
  else gemv_w(M.outwT, S.hb, logits, VOCAB, NE);
  // Tail of the captured region: advance the shared device pos so the next
  // graph replay reads pos+1 (lm_head is pos-independent, so order vs it is free).
  if (cap) cu::incr_u32(d_pos);
}

// Fully imperative decode step (C1): embed row -> run_layers_ -> GPU argmax.
// Kills the ~45% host graph-construction cost and per-step allocation; one sync
// (inside cuda::argmax). Numerically identical to step_greedy (same
// gemv/rope/attn kernels; fused rmsnorm/swiglu match the array compositions) —
// guarded greedy-exact by check_qwen and bench_qwen_decode. Returns the token.
inline int64_t step_imperative(Model& M, int64_t id, int64_t pos) {
  array e = embed_row(M, id);
  e.realize();  // embed on device (native valid; uploaded on first read)
  run_layers_(M, e.native(), pos);
  return argmax_logits(M);
}

// Run the imperative forward and return the logits [VOCAB] on the host (device
// mirror of scratch.logitsb). For the q4-vs-bf16 divergence measurement — the
// quantization-quality analogue of step_imperative's greedy token.
inline const float* imperative_logits(Model& M, int64_t id, int64_t pos) {
  array e = embed_row(M, id);
  e.realize();
  run_layers_(M, e.native(), pos);
  tl::cuda::sync_to_host(M.scratch.logitsb, false);
  return M.scratch.logits_host;
}

// Gather the embedding row for token `id` into the staged capture buffer
// S.embedb (host gather + blocking upload; marks it device-current). Kept out
// of the captured region since it involves a host gather + blocking copy.
inline void stage_embed(Model& M, int64_t id) {
  const int one = (int)id;
  gather_embed_rows_(M, &one, 1, M.scratch.embed_host);
  tl::cuda::upload(M.scratch.embedb, M.scratch.embed_host, NE);
}

// CUDA-graph-captured greedy decoder (A-min). Captures the device-pos forward
// once (after prefill, at the first decode position) and replays it per token —
// collapsing the ~360 per-token kernel launches into one graph submit. Correct
// across positions because rope/kv-append/attn read the position from the device
// counter d_pos, which a captured tl_incr_u32 advances each replay. embed staging
// + argmax stay on the host side of each step (cheap; see bench census). Falls
// back gracefully: ok()==false when graph support is missing or capture failed,
// and the caller should use step_imperative instead. f32 KV; attention is
// split-KV on a capacity-static grid (pos-independent, so capturable; flat in
// ctx like the host path — see cuda.h attn_decode_dpos).
// It owns the WHOLE prompt->generate lifecycle: begin() captures and consumes
// the prompt, step() produces each generated token, and feed() degrades to the
// imperative path by itself when capture is unavailable — so a caller never
// branches on ok() (that is only a label) and never tracks a position of its
// own. All position state lives in cur_pos here; the host cache pos is
// reconciled at the phase boundary, the only point where a caller could switch
// to a host-pos API.
struct captured_decoder {
  void* d_pos = nullptr;
  tl::cuda::CUgraphExec exec = nullptr;  // opaque handle; non-null == captured
  int64_t cur_pos = 0, max_ctx = 0;      // capacity bound owned by the mechanism
  double capture_ms = 0;  // one-time init cost (warm + record + instantiate)
  bool batched = false;   // begin() ran the prompt as GEMMs, not token-by-token

  // Capture the forward at `pos` (the first position this decoder will process),
  // priming it with `first_id`'s embedding. Leaves d_pos = pos, so the first
  // feed()/step() processes position `pos`. Leaves exec null — and the object
  // still usable, via feed()'s imperative fallback — when graph capture is
  // unavailable. Records its own cost in capture_ms (warm + record +
  // instantiate), which callers time-slicing a prompt subtract.
  void init(Model& M, int64_t first_id, int64_t pos) {
    double t0 = StepProf::now_ms();
    init_(M, first_id, pos);
    capture_ms = StepProf::now_ms() - t0;
  }
  void init_(Model& M, int64_t first_id, int64_t pos) {
    namespace cu = tl::cuda;
    if (M.layers.empty()) return;
    max_ctx = M.layers[0].cache.max_ctx;  // bound applies to the fallback too
    cur_pos = pos;
    if (!cu::graph_available()) return;
    d_pos = cu::alloc(4, nullptr);
    if (!d_pos) return;
    stage_embed(M, first_id);
    cu::upload_u32(d_pos, (unsigned)pos);
    // Warm run: real launches, but its logits go to the throwaway sink so a
    // prompt's logits (already in scratch when begin() batched) survive.
    run_layers_(M, M.scratch.embedb, pos, d_pos, M.scratch.logits_scratch);
    cu::flush();
    cu::upload_u32(d_pos, (unsigned)pos);          // reset after warm
    if (!cu::capture_begin()) return;
    run_layers_(M, M.scratch.embedb, pos, d_pos);  // recorded, not executed
    exec = cu::capture_end();
  }

  // Consume a whole prompt and return the first generated token (the greedy
  // argmax of the LAST prompt token's logits), leaving scratch logits holding
  // that token's row. Two branches, one post-condition:
  //
  //  - BATCHED (`batched`, whenever can_prefill_batched allows): the prompt runs
  //    as GEMMs, then the decode graph is captured at the position the prompt
  //    left the cache at. The prompt itself never enters the graph.
  //  - token-by-token: capture first, then feed() every prompt token but the
  //    last through it (no argmax; their greedy output is discarded). ONE graph
  //    then serves both phases, since the device pos counter advances across the
  //    boundary. ids[0]'s forward runs twice here — once to warm/grow scratch
  //    before recording, once through the loop; one token out of T, and skipping
  //    it would change init's post-condition for the decode-only callers.
  //
  // Capture is taken at the cache's CURRENT position either way, so this also
  // serves a turn appended to a live cache. Returns -1 for an empty prompt.
  int64_t begin(Model& M, const std::vector<int>& ids) {
    if (ids.empty() || M.layers.empty()) return -1;
    // Batched first: a prompt is a GEMM regime, not a GEMV one, so running it
    // token-by-token wastes the whole batch dimension (~20x measured). The
    // capture that follows is for the DECODE phase, taken at the position the
    // prompt left the cache at.
    {
      int64_t next = prefill_batched(M, ids);
      if (next >= 0) {
        batched = true;
        init(M, next, M.layers[0].cache.pos);
        return next;
      }
    }
    // Token-by-token: one capture serves prompt and decode alike.
    init(M, ids[0], M.layers[0].cache.pos);
    for (size_t i = 0; i + 1 < ids.size(); i++) feed(M, ids[i]);
    int64_t last = step(M, ids.back());  // only the last token's logits are read
    sync_host_pos(M);
    return last;
  }

  // Feed token `id` at the current position, WITHOUT reading a token back.
  // Returns false once the KV cache is full (the bound lives here, not on the
  // caller — kv_append_dpos has no OOB guard, unlike the host append()).
  //
  // Captured mode replays the graph; d_pos auto-advances (captured tl_incr_u32).
  // Skipping argmax saves its 4-byte D2H *sync* per token (~6%, the replay-only
  // vs replay+argmax gap in bench_qwen_decode) — safe because the next
  // stage_embed's H2D is a blocking null-stream copy, so it cannot overwrite
  // S.embedb while the previous replay still reads it. Without a graph this is
  // step_imperative minus the argmax, on the same scratch and host cache pos.
  bool feed(Model& M, int64_t id) {
    namespace cu = tl::cuda;
    if (cur_pos >= max_ctx) return false;
    stage_embed(M, id);  // gather id's row -> S.embedb (host + blocking H2D)
    if (exec) cu::graph_launch(exec);  // replay: append@d_pos, attn, logits, incr
    else run_layers_(M, M.scratch.embedb, cur_pos);  // host-pos imperative
    cur_pos++;
    return true;
  }
  // One decode step: feed + read the greedy token (-1 when the cache is full).
  int64_t step(Model& M, int64_t id) {
    return feed(M, id) ? argmax_logits(M) : -1;
  }
  // Reconcile the host cache pos with the device counter. Captured replays
  // advance only d_pos, so the host pos would otherwise be stale for anyone
  // switching to a host-pos path (the fallback keeps it current by itself).
  void sync_host_pos(Model& M) const {
    if (exec) set_cache_pos(M, cur_pos);
  }

  bool ok() const { return exec != nullptr; }
  void destroy() {
    namespace cu = tl::cuda;
    if (exec) cu::graph_destroy(exec);
    if (d_pos) cu::release(d_pos, 0, nullptr);
    exec = nullptr;
    d_pos = nullptr;
  }
};

inline std::string default_path() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/models/qwen2.5-0.5b-instruct-fp16.gguf";
}

// Deterministic short-prefill fixture shared by the decode benches (arbitrary
// valid token ids — enough to make the cache/pos realistic without a tokenizer).
constexpr int PREFILL_IDS[4] = {40, 2610, 264, 3974};

// Assert the GGUF metadata matches our compile-time constants (this driver is
// hard-coded for Qwen2.5-0.5B). Returns false on mismatch.
inline bool check_config(const gg::model& m) {
  return m.kv("qwen2.block_count").as_u32() == NL &&
         m.kv("qwen2.embedding_length").as_u32() == NE &&
         m.kv("qwen2.attention.head_count").as_u32() == NH &&
         m.kv("qwen2.attention.head_count_kv").as_u32() == NKV;
}

}  // namespace qwenmodel
