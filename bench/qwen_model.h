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
constexpr int64_t GROUP = NH / NKV;  // 7
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

// F16 weight, GGML physical [out,in] (ne dims [in,out]) -> array logical
// [in,out] = [K,N] (our dot's expected layout): out[i*out+o] = f16(raw[o*in+i]).
// wdt selects the storage dtype: f32 (exact, ~2GB total → WSL2 cliff) or bf16
// (~1GB, stays under the cliff; the decode GEMV consumes bf16 weights natively).
inline array load_w_T(const gg::model& m, const std::string& n, int64_t in,
                      int64_t out, tl::dtype wdt = tl::dtype::f32) {
  const auto* t = need(m, n);
  const auto* raw = reinterpret_cast<const uint16_t*>(t->data);
  std::vector<float> v((size_t)in * out);
  for (int64_t o = 0; o < out; o++)
    for (int64_t i = 0; i < in; i++)
      v[(size_t)i * out + o] = f16_to_f32(raw[(size_t)o * in + i]);
  array a = array::from(std::move(v), {in, out});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// Column-concatenated fused weight: load several GGML [out,in] tensors that
// share the same `in` (K) and stack them along the output (N) axis into one
// [in, Ntot] = [K, Ntot] array, so a single decode GEMV produces all their
// outputs contiguously (q|k|v or gate|up). Column block b occupies
// [col_base, col_base+out_b); v[i*Ntot + col_base+o] = f16(raw_b[o*in+i]).
// Bit-identical per column to loading each separately (same widen, same layout),
// so the fused GEMV's per-column result matches the separate GEMVs — the M9
// decode fusion (fewer kernels → amortize the ~14us per-launch floor). Imperative
// path only; the array path keeps the separate loads as the numeric oracle.
inline array load_w_T_cat(const gg::model& m,
                          const std::vector<std::pair<std::string, int64_t>>& parts,
                          int64_t in, tl::dtype wdt = tl::dtype::f32) {
  int64_t Ntot = 0;
  for (const auto& p : parts) Ntot += p.second;
  std::vector<float> v((size_t)in * Ntot);
  int64_t col_base = 0;
  for (const auto& p : parts) {
    const auto* t = need(m, p.first);
    const auto* raw = reinterpret_cast<const uint16_t*>(t->data);
    const int64_t out = p.second;
    for (int64_t o = 0; o < out; o++)
      for (int64_t i = 0; i < in; i++)
        v[(size_t)i * Ntot + (col_base + o)] = f16_to_f32(raw[(size_t)o * in + i]);
    col_base += out;
  }
  array a = array::from(std::move(v), {in, Ntot});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// Row-major weight for the warp-per-row decode GEMV (lever A): GGML physical
// [out,in] widened straight to array [out,in] = [N,K] — NO transpose (K stays
// contiguous per output row, the layout tl_gemv_bf16_row wants). The imperative
// bf16 decode path uses this; the array path keeps the [K,N] load_w_T as oracle.
inline array load_w_T_row(const gg::model& m, const std::string& n, int64_t in,
                          int64_t out, tl::dtype wdt = tl::dtype::bf16) {
  const auto* t = need(m, n);
  const auto* raw = reinterpret_cast<const uint16_t*>(t->data);
  std::vector<float> v((size_t)out * in);
  for (int64_t o = 0; o < out; o++)
    for (int64_t i = 0; i < in; i++)
      v[(size_t)o * in + i] = f16_to_f32(raw[(size_t)o * in + i]);
  array a = array::from(std::move(v), {out, in});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// Row-concatenated fused weight for the warp-per-row GEMV: stack several GGML
// [out,in] tensors that share `in` (K) along the OUTPUT (row) axis into one
// [Ntot, in] = [N,K] array (q|k|v or gate|up). Row block b occupies rows
// [row_base, row_base+out_b); the fused GEMV's output [Ntot] is q|k|v (or
// gate|up) contiguous — same output layout as the separate/column-cat GEMVs, so
// the downstream slices are unchanged. Reduction order differs from split-K, so
// it is greedy-equivalent (not bit-identical) to the array oracle.
inline array load_w_T_cat_row(
    const gg::model& m,
    const std::vector<std::pair<std::string, int64_t>>& parts, int64_t in,
    tl::dtype wdt = tl::dtype::bf16) {
  int64_t Ntot = 0;
  for (const auto& p : parts) Ntot += p.second;
  std::vector<float> v((size_t)Ntot * in);
  int64_t row_base = 0;
  for (const auto& p : parts) {
    const auto* t = need(m, p.first);
    const auto* raw = reinterpret_cast<const uint16_t*>(t->data);
    const int64_t out = p.second;
    for (int64_t o = 0; o < out; o++)
      for (int64_t i = 0; i < in; i++)
        v[(size_t)(row_base + o) * in + i] = f16_to_f32(raw[(size_t)o * in + i]);
    row_base += out;
  }
  array a = array::from(std::move(v), {Ntot, in});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// F32 1-D tensor (norms, biases are stored F32 in the GGUF) -> array [1, len].
inline array load_f32(const gg::model& m, const std::string& n, int64_t len) {
  const auto* t = need(m, n);
  const auto* raw = reinterpret_cast<const float*>(t->data);
  return array::from(std::vector<float>(raw, raw + len), {1, len});
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
  array wqkv, wgu, wo_row, wd_row;
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
// per-step allocation (no WSL2 churn). res[2] ping-pong the residual stream; hb
// doubles as h and h2 (h is fully consumed by the q/k/v gemvs before the MLP
// reuses it).
struct Scratch {
  void* res[2] = {nullptr, nullptr};  // residual ping-pong [NE]
  void* hb = nullptr;                 // input-norm out [NE]
  void* h2b = nullptr;                // post-attn norm out [NE]
  void* qb = nullptr;                 // [NH*HD]
  void *kb = nullptr, *vb = nullptr;  // [NKV*HD]
  void* qkvb = nullptr;               // fused QKV out [(NH+2*NKV)*HD] = [1152]
  void* ab = nullptr;                 // attn out [NH*HD]
  void *gb = nullptr, *ub = nullptr, *mb = nullptr;  // gate/up/swiglu [FF]
  void* gub = nullptr;                // fused gate|up out [2*FF] = [9728]
  void* mdb = nullptr;                // mlp-down out [NE]
  void* logitsb = nullptr;            // [VOCAB]
  float* logits_host = nullptr;       // host mirror of logitsb (divergence checks)
  void* embedb = nullptr;             // staged embedding row [NE] (capture input)
  bool ready = false;
  void init() {
    embedb = tl::cuda::alloc(NE * 4, nullptr);
    res[0] = tl::cuda::alloc(NE * 4, nullptr);
    res[1] = tl::cuda::alloc(NE * 4, nullptr);
    hb = tl::cuda::alloc(NE * 4, nullptr);
    h2b = tl::cuda::alloc(NE * 4, nullptr);
    qb = tl::cuda::alloc(NH * HD * 4, nullptr);
    kb = tl::cuda::alloc(NKV * HD * 4, nullptr);
    vb = tl::cuda::alloc(NKV * HD * 4, nullptr);
    qkvb = tl::cuda::alloc((NH + 2 * NKV) * HD * 4, nullptr);
    ab = tl::cuda::alloc(NH * HD * 4, nullptr);
    gb = tl::cuda::alloc(FF * 4, nullptr);
    ub = tl::cuda::alloc(FF * 4, nullptr);
    mb = tl::cuda::alloc(FF * 4, nullptr);
    gub = tl::cuda::alloc(2 * FF * 4, nullptr);
    mdb = tl::cuda::alloc(NE * 4, nullptr);
    logitsb = tl::cuda::alloc(VOCAB * 4, &logits_host);
    ready = logitsb != nullptr;
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
  bool row = false;  // imperative gemvs use warp-per-row [N,K] (bf16 only)
  bool q4_mlp = false;     // imperative MLP gemvs (wgu, wd) use q4
  bool q4_lmhead = false;  // imperative lm_head gemv uses q4
};

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
  Model M{reinterpret_cast<const uint16_t*>(need(m, "token_embd.weight")->data),
          load_w_T(m, "output.weight", NE, VOCAB, wdt),  // lm_head stays [K,N]
          load_f32(m, "output_norm.weight", NE),
          // q4 lm_head (imperative): quantize the f32 [K,VOCAB] then keep only q4.
          q4lm ? load_w_T(m, "output.weight", NE, VOCAB, f32).to_q4() : array{}};
  M.row = row;
  M.q4_mlp = q4m;
  M.q4_lmhead = q4lm;
  for (int64_t l = 0; l < NL; l++) {
    std::string p = "blk." + std::to_string(l) + ".";
    Layer L{load_w_T(m, p + "attn_q.weight", NE, NH * HD, wdt),
            load_f32(m, p + "attn_q.bias", NH * HD),
            load_w_T(m, p + "attn_k.weight", NE, NKV * HD, wdt),
            load_f32(m, p + "attn_k.bias", NKV * HD),
            load_w_T(m, p + "attn_v.weight", NE, NKV * HD, wdt),
            load_f32(m, p + "attn_v.bias", NKV * HD),
            load_w_T(m, p + "attn_output.weight", NH * HD, NE, wdt),
            load_w_T(m, p + "ffn_gate.weight", NE, FF, wdt),
            load_w_T(m, p + "ffn_up.weight", NE, FF, wdt),
            load_w_T(m, p + "ffn_down.weight", FF, NE, wdt),
            load_f32(m, p + "attn_norm.weight", NE),
            load_f32(m, p + "ffn_norm.weight", NE),
            // Fused QKV and gate|up (imperative decode path). In bf16 mode: ROW
            // [N,K] for the warp-per-row gemv (lever A). In f32: column [K,N] for
            // split-K. +466MB over 24 layers either way. wo_row/wd_row (bf16 only)
            // are the row copies of the two shared weights (+247MB), keeping the
            // total under the WSL2 ~2GB cliff (lm_head deliberately not copied).
            row ? load_w_T_cat_row(m, {{p + "attn_q.weight", NH * HD},
                                       {p + "attn_k.weight", NKV * HD},
                                       {p + "attn_v.weight", NKV * HD}}, NE)
                : load_w_T_cat(m, {{p + "attn_q.weight", NH * HD},
                                   {p + "attn_k.weight", NKV * HD},
                                   {p + "attn_v.weight", NKV * HD}}, NE, wdt),
            // wgu: bf16-row when row & !q4; column [K,N] in f32; empty when q4m
            // (replaced by wgu_q4). wd_row likewise.
            (row && !q4m) ? load_w_T_cat_row(m, {{p + "ffn_gate.weight", FF},
                                                 {p + "ffn_up.weight", FF}}, NE)
            : row ? array{}
                  : load_w_T_cat(m, {{p + "ffn_gate.weight", FF},
                                     {p + "ffn_up.weight", FF}}, NE, wdt),
            row ? load_w_T_row(m, p + "attn_output.weight", NH * HD, NE) : array{},
            (row && !q4m) ? load_w_T_row(m, p + "ffn_down.weight", FF, NE) : array{},
            // q4 MLP (imperative, when q4m): quantize the f32 [K,N] fused/down.
            q4m ? load_w_T_cat(m, {{p + "ffn_gate.weight", FF},
                                   {p + "ffn_up.weight", FF}}, NE, f32).to_q4()
                : array{},
            q4m ? load_w_T(m, p + "ffn_down.weight", FF, NE, f32).to_q4() : array{},
            {}};
    L.cache.init(NKV, MAXC, HD);
    M.layers.push_back(std::move(L));
  }
  M.scratch.init();  // persistent imperative-step buffers (GPU must be active)
  return M;
}

inline array embed_row(const Model& M, int64_t id) {
  std::vector<float> v(NE);
  const uint16_t* r = M.embed_f16 + (size_t)id * NE;
  for (int64_t i = 0; i < NE; i++) v[i] = f16_to_f32(r[i]);
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
    (void)q; (void)k; (void)v; (void)a_out;
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

// Decode GEMV picking the weight-dtype kernel: y(n) = a(1,k) @ W[k,n].
inline bool gemv_w(const array& W, void* a, void* y, int64_t n, int64_t k) {
  return W.dt() == tl::dtype::bf16
             ? tl::cuda::gemv_bf16(a, W.native(), y, n, k)
             : tl::cuda::gemv_f32(a, W.native(), y, n, k);
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
inline void run_layers_(Model& M, void* x0, int64_t pos) {
  namespace cu = tl::cuda;
  Scratch& S = M.scratch;
  // Fused seams (kills 4 elementwise launches/layer): q/k bias folds into rope
  // (cu::rope bias arg); the two residual adds fold into the following RMSNorms
  // via rmsnorm_res (writes the residual sum AND its norm). The layer input's
  // norm is therefore produced by the *previous* layer's mlp seam — so precompute
  // layer 0's here, and carry each layer's next-input norm in hb.
  // Slice a fused f32 device buffer by element offset (mid-buffer pointer; the
  // slice reads are stream-ordered after the fused GEMV and need no host sync).
  auto off = [](void* p, int64_t nfloats) -> void* {
    return static_cast<char*>(p) + nfloats * 4;
  };
  // Decode GEMV: warp-per-row [N,K] (lever A, bf16 imperative) when Wrow is
  // populated, else the split-K [K,N] path on Wcol. y layout is identical either
  // way (contiguous [n]), so the fused-output slices below are unchanged.
  const bool row = M.row;
  auto gv = [&](const array& Wrow, const array& Wcol, void* a, void* y, int64_t n,
                int64_t k) {
    if (row) tl::cuda::gemv_bf16_row(a, Wrow.native(), y, n, k);
    else gemv_w(Wcol, a, y, n, k);
  };
  void* x = x0;
  cu::rmsnorm(x, M.layers[0].an.native(), S.hb, NE, EPS);
  for (int64_t l = 0; l < NL; l++) {
    Layer& L = M.layers[l];
    void* ro = S.res[l & 1];  // res_out (x1 then x2) for this layer (ping-pong)
    // Fused QKV: one GEMV -> [q(NH*HD) | k(NKV*HD) | v(NKV*HD)] in S.qkvb, then
    // slice: rope q & k in place, bias-add v. Same per-column split-K as the
    // separate wq/wk/wv GEMVs (bx=1, chunk=32), so bit-identical per column.
    gv(L.wqkv, L.wqkv, S.hb, S.qkvb, (NH + 2 * NKV) * HD, NE);
    void* qp = S.qkvb;
    void* kp = off(S.qkvb, NH * HD);
    void* vp = off(S.qkvb, (NH + NKV) * HD);
    cu::rope(qp, qp, NH, 1, HD, pos, ROPE_BASE, L.bq.native());
    cu::rope(kp, kp, NKV, 1, HD, pos, ROPE_BASE, L.bk.native());
    cu::binary(cu::kop::add, vp, 0, L.bv.native(), 0, vp, 0, NKV * HD, 1, 0);
    L.cache.append(kp, vp);
    L.cache.attn(qp, S.ab, NH, SCALE);
    gv(L.wo_row, L.wo, S.ab, ro, NE, NH * HD);            // ro = attn @ wo
    // x1 = x + (attn@wo); h2 = rmsnorm(x1, fn) — fused.
    cu::rmsnorm_res(ro, x, L.fn.native(), ro, S.h2b, NE, EPS);
    // Fused gate|up: one GEMV -> [gate(FF) | up(FF)] in S.gub; swiglu reads both.
    if (M.q4_mlp) gemv_q4_w(L.wgu_q4, S.h2b, S.gub);  // [K=NE, N=2*FF]
    else gv(L.wgu, L.wgu, S.h2b, S.gub, 2 * FF, NE);
    cu::swiglu(S.gub, off(S.gub, FF), S.mb, FF);
    if (M.q4_mlp) gemv_q4_w(L.wd_q4, S.mb, S.mdb);    // [K=FF, N=NE]
    else gv(L.wd_row, L.wd, S.mb, S.mdb, NE, FF);
    // x2 = x1 + mlp; next input norm = rmsnorm(x2, next an | final onorm) — fused.
    void* nextw = (l + 1 < NL) ? M.layers[l + 1].an.native() : M.onorm.native();
    cu::rmsnorm_res(ro, S.mdb, nextw, ro, S.hb, NE, EPS);
    x = ro;
  }
  // hb now holds the final RMSNorm output (folded into the last layer's seam).
  if (M.q4_lmhead) gemv_q4_w(M.outwT_q4, S.hb, S.logitsb);  // [K=NE, N=VOCAB]
  else gemv_w(M.outwT, S.hb, S.logitsb, VOCAB, NE);
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
  int64_t idx = 0;
  tl::cuda::argmax(M.scratch.logitsb, VOCAB, &idx);
  return idx;
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
  std::vector<float> v(NE);
  const uint16_t* r = M.embed_f16 + (size_t)id * NE;
  for (int64_t i = 0; i < NE; i++) v[i] = f16_to_f32(r[i]);
  tl::cuda::upload(M.scratch.embedb, v.data(), NE);
}

// Reset all KV caches to position 0 (replay from a fresh prefill). Used by the
// bench's array-vs-imperative greedy-equivalence check.
inline void reset_cache(Model& M) {
  for (auto& L : M.layers) L.cache.pos = 0;
}

inline std::string default_path() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/models/qwen2.5-0.5b-instruct-fp16.gguf";
}

// Assert the GGUF metadata matches our compile-time constants (this driver is
// hard-coded for Qwen2.5-0.5B). Returns false on mismatch.
inline bool check_config(const gg::model& m) {
  return m.kv("qwen2.block_count").as_u32() == NL &&
         m.kv("qwen2.embedding_length").as_u32() == NE &&
         m.kv("qwen2.attention.head_count").as_u32() == NH &&
         m.kv("qwen2.attention.head_count_kv").as_u32() == NKV;
}

}  // namespace qwenmodel
