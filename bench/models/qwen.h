#pragma once
// Qwen2.5 / Qwen3 decoder: GGUF (F16) weight loading + forward, shared by
// check_qwen (numeric validation) and chat_qwen (real text generation). The
// weights are loaded F32 (exact F16 widen), linear weights transposed from GGML's
// [out,in] to our dot's [K,N]. Structure: RMSNorm -> QKV proj (+ bias in Qwen2)
// -> per-head q/k RMSNorm (Qwen3 only) -> RoPE(half-split) -> GQA attn via
// tl::kv_cache -> o proj -> res -> RMSNorm -> SwiGLU MLP -> res, x NL, then final
// RMSNorm -> logits (output.weight, or the embedding when tied). The dims and the
// three Qwen2/Qwen3 switches are a runtime Config read from the GGUF.
//
// Backend-neutral: every device call goes through tl::gpu (the facade gpu.h
// picks), every scratch buffer is a tl::storage (the backend's pool reclaims
// it), and the three things only CUDA has today are asked of gpu::caps rather
// than of a #ifdef — graph_capture (the captured decoder), bf16_gemm (the
// batched prefill) and row_gemv ([N,K] weights). Where a cap is false the
// model keeps the path that every backend has: re-encode per token, prefill
// token by token, column-major [K,N] weights.

#include <kv_cache.h>
#include <storage.h>
#include <tensorlib.h>
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwenmodel {

using tl::array;
using tl::storage;
namespace gg = tl::gguf;
namespace gpu = tl::gpu;

// The shape of one Qwen decoder, read from the GGUF metadata (config_from_gguf)
// rather than compiled in, so Qwen2.5, Qwen3 and Qwen3.5 — and every size of
// each — share one forward. Everything that differs between those families is
// a field here: the dims, the norm eps / rope base, and the structural switches.
struct Config {
  std::string arch;  // "qwen2" | "qwen3" | "qwen35": the family, for drivers
  int64_t NL = 0, NE = 0, NH = 0, NKV = 0, HD = 0, FF = 0, VOCAB = 0;
  int64_t MAXC = 4096;  // KV capacity per layer (f32): NKV*MAXC*HD*8 B per layer
  float EPS = 1e-6f, ROPE_BASE = 1e6f;
  // Qwen2 puts a bias on the q/k/v projections; Qwen3 drops it.
  bool qkv_bias = false;
  // Qwen3 RMS-normalizes each q and k head (weight [HD]) before RoPE.
  bool qk_norm = false;
  // Small Qwen3 ties the lm_head to the embedding: no output.weight tensor.
  bool tied_output = false;
  // Qwen3.5: RoPE turns only the first ROT dims of a head (ROT == HD before),
  // and the q projection carries a per-head output gate, interleaved with q.
  int64_t ROT = 0;
  bool attn_gate = false;
  // Qwen3.5 is hybrid: every INTERVAL-th layer is full attention and the rest
  // are Gated DeltaNet (INTERVAL 1: all attention). A DeltaNet layer has SH
  // heads with a [SDK, SDV] state each and a SCONV-tap causal conv.
  int64_t INTERVAL = 1, SH = 0, SDK = 0, SDV = 0, SCONV = 0;
  std::string ffn_norm = "ffn_norm.weight";  // Qwen3.5: post_attention_norm
  float L2EPS = 1e-6f;  // DeltaNet's q / k L2 norms (the reference's constant)
  float scale() const { return 1.0f / std::sqrt((float)HD); }
  bool hybrid() const { return INTERVAL > 1; }
  bool is_delta(int64_t l) const { return hybrid() && (l + 1) % INTERVAL != 0; }
  // Output widths of the fused projections (and so of the scratch they land in).
  int64_t attn_proj() const { return (NH + 2 * NKV) * HD + (attn_gate ? NH * HD : 0); }
  int64_t ssm_conv_ch() const { return 2 * SH * SDK + SH * SDV; }  // q|k|v
  // q|k|v | alpha | beta | z, then a slot for the rule's output right after z,
  // so the gated norm's silu(z) * o reads them as one fused gate|up buffer.
  int64_t ssm_proj() const { return ssm_conv_ch() + 2 * SH + SH * SDV; }
  std::string output_name() const {
    return tied_output ? "token_embd.weight" : "output.weight";
  }
};

// Read a Config from the GGUF: architecture-prefixed metadata for the dims, and
// the presence of the optional tensors for the structural switches (so a file
// that has them is run with them). Throws on an
// architecture this forward does not implement.
inline Config config_from_gguf(const gg::model& m, int64_t maxc = 4096) {
  const std::string& arch = m.kv("general.architecture").as_str();
  if (arch != "qwen2" && arch != "qwen3" && arch != "qwen35")
    throw std::runtime_error("qwen: unsupported architecture " + arch);
  auto u32 = [&](const char* k) { return (int64_t)m.kv(arch + "." + k).as_u32(); };
  Config c;
  c.arch = arch;
  c.NL = u32("block_count");
  c.NE = u32("embedding_length");
  c.NH = u32("attention.head_count");
  c.NKV = u32("attention.head_count_kv");
  c.FF = u32("feed_forward_length");
  // Qwen3 states head_dim (key_length); Qwen2 derives it from the width.
  c.HD = m.has(arch + ".attention.key_length") ? u32("attention.key_length")
                                               : c.NE / c.NH;
  c.EPS = m.kv(arch + ".attention.layer_norm_rms_epsilon").as_f32();
  c.ROPE_BASE = m.kv(arch + ".rope.freq_base").as_f32();
  c.MAXC = maxc;
  const auto* emb = m.tensor("token_embd.weight");
  if (!emb) throw std::runtime_error("qwen: token_embd.weight missing");
  c.VOCAB = (int64_t)emb->dims[1];
  c.ROT = m.has(arch + ".rope.dimension_count") ? u32("rope.dimension_count") : c.HD;
  if (m.has(arch + ".full_attention_interval")) {
    c.INTERVAL = u32("full_attention_interval");
    c.SH = u32("ssm.time_step_rank");  // value heads
    c.SDK = u32("ssm.state_size");
    c.SDV = u32("ssm.inner_size") / c.SH;
    c.SCONV = u32("ssm.conv_kernel");
    // Fewer key heads than value heads repeats them, in an order the GGUF
    // conversion has its own convention for; nothing here implements that.
    if (u32("ssm.group_count") != c.SH)
      throw std::runtime_error("qwen: DeltaNet key heads != value heads");
  }
  // The switches are read off the first attention layer (layer 0 may be DeltaNet).
  const std::string a0 = "blk." + std::to_string(c.INTERVAL - 1) + ".";
  c.qkv_bias = m.tensor(a0 + "attn_q.bias") != nullptr;
  c.qk_norm = m.tensor(a0 + "attn_q_norm.weight") != nullptr;
  const auto* wq = m.tensor(a0 + "attn_q.weight");
  if (!wq) throw std::runtime_error("qwen: " + a0 + "attn_q.weight missing");
  c.attn_gate = (int64_t)wq->dims[1] == 2 * c.NH * c.HD;
  c.tied_output = m.tensor("output.weight") == nullptr;
  if (m.tensor("blk.0.post_attention_norm.weight")) c.ffn_norm = "post_attention_norm.weight";
  return c;
}
// Prompt tokens per batched prefill pass. Sets the PrefillScratch size (linear in
// it: ~47 MiB for Qwen2.5-0.5B, dominated by the fused gate|up buffer) and how
// well the projections fill the GPU — the GEMMs
// are still latency-bound at 256 (3.20 us/token) and reach 2.60 at 512.
constexpr int64_t PREFILL_CHUNK = 512;
// Below this the prompt is not worth a batched pass: the GEMM path only pays off
// once a chunk has enough rows to fill the SMs, and short prompts are dominated
// by the fixed per-chunk staging anyway.
constexpr int64_t PREFILL_MIN = 4;

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
//
// A part normally takes all of its tensor's rows. With blk set it takes a strided
// subset instead — output row o reads source row (o / blk) * stride + first +
// o % blk — which is how Qwen3.5's q projection, q and gate interleaved per
// head, splits into a contiguous q block and a contiguous gate block at load.
struct wpart {
  std::string name;
  int64_t out;
  int64_t blk = 0, stride = 0, first = 0;
  int64_t src(int64_t o) const { return blk ? (o / blk) * stride + first + o % blk : o; }
};
inline array load_w_cat_(const gg::model& m, const std::vector<wpart>& parts,
                         int64_t in, tl::dtype wdt, bool row) {
  int64_t Ntot = 0;
  for (const auto& p : parts) Ntot += p.out;
  std::vector<float> v((size_t)in * Ntot);
  int64_t base = 0;
  for (const auto& p : parts) {
    const auto* raw = reinterpret_cast<const uint16_t*>(need(m, p.name)->data);
    const int64_t out = p.out;
    // Split on the layout OUTSIDE the element loops: this runs over every weight
    // byte in the model, so the destination index must stay branch-free.
    if (row) {
      for (int64_t o = 0; o < out; o++)
        for (int64_t i = 0, so = p.src(o); i < in; i++)
          v[(size_t)(base + o) * in + i] = f16_to_f32(raw[(size_t)so * in + i]);
    } else {
      for (int64_t o = 0; o < out; o++)
        for (int64_t i = 0, so = p.src(o); i < in; i++)
          v[(size_t)i * Ntot + (base + o)] = f16_to_f32(raw[(size_t)so * in + i]);
    }
    base += out;
  }
  array a = array::from(std::move(v),
                        row ? tl::shape_t{Ntot, in} : tl::shape_t{in, Ntot});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}

// The four named entry points: {single, concatenated} x {[K,N], [N,K]}. The
// [K,N] pair takes the storage dtype — f32 (exact, 4 B/weight) or bf16 (2 B/weight;
// the decode GEMV consumes bf16 natively). The [N,K]
// pair does not: row-major weights exist only to feed the warp-per-row GEMV and
// the prefill GEMM, and both are bf16-only.
inline array load_w_T_cat(const gg::model& m, const std::vector<wpart>& parts,
                          int64_t in, tl::dtype wdt) {
  return load_w_cat_(m, parts, in, wdt, /*row=*/false);
}
inline array load_w_T_cat_row(const gg::model& m, const std::vector<wpart>& parts,
                              int64_t in) {
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

// `n` f32 of device scratch. A tl::storage rather than a raw gpu::alloc: the
// pool is keyed by byte size and a hand-paired alloc/release is one edit away
// from returning a buffer under the wrong key, whereas storage carries its own
// deleter. `.native` is the device handle, `.ptr` the host-readable bytes
// (unified on Metal, the mirror on CUDA).
inline storage scratch_f32(int64_t n) { return storage::make(n); }

// Device views. A storage or an array names its whole buffer; at_f32 slices a
// view by element. The offset travels beside the handle (gpu::span), so a slice
// is a view on every backend — no copy, and reads through it are ordered after
// whatever wrote the base.
inline gpu::span sp(const storage& s) { return s.device_span(); }
inline gpu::span sp(const array& a) { return a.device_span(); }
inline gpu::span at_f32(gpu::span s, int64_t nfloats) { return s.at(nfloats * 4); }

// Make a device buffer's bytes readable on the host: drain the queue (on
// unified memory that is all it takes) and pull the mirror back where there is
// one. The one place that knows both backends' coherence rules.
inline const float* host_read(const storage& s) {
  gpu::cpu_barrier();
  gpu::sync_to_host(s.native, false);
  return s.ptr;
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
  // Qwen3's per-head q / k RMSNorm weights, [1, HD] (empty when !Config::qk_norm).
  // bq/bk/bv above are likewise empty when !Config::qkv_bias.
  array qn, kn;
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
  tl::kv_cache cache;  // attention layers only
  // Gated DeltaNet layers (Qwen3.5) instead of the attention weights above:
  // wssm the fused [qkv | alpha | beta | z] projection and wsout the output
  // projection (both in the decode GEMV's layout, like wqkv / wo_row), wconv
  // the conv taps [SCONV, C] tap-major, adt = A | dt_bias [2*SH], sn the gated
  // norm's weight [SDV]; and the recurrent state the layer carries from token
  // to token in place of a KV cache: the conv window [SCONV-1, C] and the
  // rule's state [SH, SDK, SDV].
  array wssm, wsout, wconv, adt, sn;
  storage conv_state, ssm_state;
};

// Persistent device scratch for the imperative decode step (C1): every per-step
// intermediate is a device buffer allocated ONCE and reused each token, so the
// step builds zero array nodes (no host graph construction) and does zero
// per-step allocation (no WSL2 churn). res[2] ping-pong the residual stream;
// q/k/v and gate/up live only as slices of their fused projection outputs
// (qkvb, gub).
struct Scratch {
  storage res[2];       // residual ping-pong [NE]
  storage hb;           // input-norm out [NE]
  storage h2b;          // post-attn norm out [NE]
  storage qb;           // [NH*HD] query fixture (bench_qwen_ctx's isolated-
                        // attention timing; decode reads q as a slice of qkvb)
  storage qkvb;         // fused projection out: attention's q|k|v(|gate), or
                        // DeltaNet's q|k|v|alpha|beta|z plus the rule's output
  storage ab;           // attn (or DeltaNet gated-norm) out
  storage convb;        // DeltaNet conv out [ssm_conv_ch]
  storage gateb;        // Qwen3.5 attention: sigmoid(gate) [NH*HD]
  storage mb;           // swiglu out [FF]
  storage gub;          // fused gate|up out [2*FF] = [9728]
  storage mdb;          // mlp-down out [NE]
  storage logits;       // [VOCAB]
  storage logits_sink;  // [VOCAB] sink for forwards run only for their side
                        // effects (capture warm-up)
  storage embed;        // staged embedding row [NE] (the step's input)
  std::vector<float> embed_stage;  // host gather target, uploaded into embed
  void init(const Config& c) {
    embed = scratch_f32(c.NE);
    embed_stage.assign(c.NE, 0.0f);
    logits = scratch_f32(c.VOCAB);
    res[0] = scratch_f32(c.NE);
    res[1] = scratch_f32(c.NE);
    hb = scratch_f32(c.NE);
    h2b = scratch_f32(c.NE);
    qb = scratch_f32(c.NH * c.HD);
    qkvb = scratch_f32(std::max(c.attn_proj(), c.ssm_proj() + c.SH * c.SDV));
    ab = scratch_f32(std::max(c.NH * c.HD, c.SH * c.SDV));
    if (c.hybrid()) convb = scratch_f32(c.ssm_conv_ch());
    if (c.attn_gate) gateb = scratch_f32(c.NH * c.HD);
    mb = scratch_f32(c.FF);
    gub = scratch_f32(2 * c.FF);
    mdb = scratch_f32(c.NE);
    logits_sink = scratch_f32(c.VOCAB);
  }
};

// Device scratch for the BATCHED prefill (M9): the same per-token intermediates
// as Scratch, but `cap` rows wide, so a chunk of prompt tokens flows through the
// forward as one [T, n] matrix. Sized once for the chunk and reused across
// chunks and across prompts. Dominated by the fused gate|up buffer (cap x 2*FF),
// which is why a long prompt is chunked at all rather than run in one
// [T_prompt, ...] pass.
struct PrefillScratch {
  int64_t cap = 0;
  storage emb;     // embedding rows [cap, NE]
  storage res[2];  // residual ping-pong [cap, NE]
  storage h;       // input-norm out [cap, NE]
  storage h2;      // post-attn norm out [cap, NE]
  storage qkv;     // fused QKV out [cap, (NH+2*NKV)*HD]
  // q|k|v head-major in ONE buffer [NH+2*NKV, cap, HD]: one split_heads pass
  // over every head, and k / v are views into it.
  storage qkvh;
  storage ah;  // attn out head-major [NH, cap, HD]
  storage at;  // attn out token-major [cap, NH*HD]
  storage gu;  // fused gate|up [cap, 2*FF]
  storage mb;  // swiglu out [cap, FF]
  storage md;  // mlp-down out [cap, NE]
  std::vector<float> emb_stage;  // host gather target, uploaded into emb

  bool init(const Config& c, int64_t chunk) {
    if (cap >= chunk) return true;  // already big enough
    cap = chunk;
    emb = scratch_f32(cap * c.NE);
    emb_stage.assign((size_t)cap * c.NE, 0.0f);
    res[0] = scratch_f32(cap * c.NE);
    res[1] = scratch_f32(cap * c.NE);
    h = scratch_f32(cap * c.NE);
    h2 = scratch_f32(cap * c.NE);
    qkv = scratch_f32(cap * (c.NH + 2 * c.NKV) * c.HD);
    qkvh = scratch_f32((c.NH + 2 * c.NKV) * cap * c.HD);
    ah = scratch_f32(c.NH * cap * c.HD);
    at = scratch_f32(cap * c.NH * c.HD);
    gu = scratch_f32(cap * 2 * c.FF);
    mb = scratch_f32(cap * c.FF);
    md = scratch_f32(cap * c.NE);
    return md.native != nullptr;
  }
};

struct Model {
  Config c;
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
inline bool gemv_w(const array& W, gpu::span a, gpu::span y, int64_t n,
                   int64_t k) {
  return W.dt() == tl::dtype::bf16 ? gpu::gemv_bf16(a, sp(W), y, n, k)
                                   : gpu::gemv_f32(a, sp(W), y, n, k);
}

// Greedy token from the logits the most recent forward left in scratch (the one
// terminal sync of a step lives inside gpu::argmax).
inline int64_t argmax_logits(Model& M) {
  const Config& c = M.c;
  int64_t idx = 0;
  gpu::argmax(sp(M.scratch.logits), c.VOCAB, &idx);
  return idx;
}

// The sequence position lives in the KV caches, every one at the same place.
// Layer 0 is DeltaNet in Qwen3.5 and has none, so this is the first that does.
inline const tl::kv_cache& seq_cache(const Model& M) {
  return M.layers[M.c.INTERVAL - 1].cache;
}

// A DeltaNet layer's recurrent state back to zero: the conv window and the
// rule's state, both read before they are written on the first token.
inline void zero_state_(const Config& c, Layer& L) {
  const int64_t nconv = (c.SCONV - 1) * c.ssm_conv_ch(), nssm = c.SH * c.SDK * c.SDV;
  const std::vector<float> z((size_t)std::max(nconv, nssm), 0.0f);
  gpu::upload(L.conv_state.native, z.data(), nconv);
  gpu::upload(L.ssm_state.native, z.data(), nssm);
}

// Reset the sequence to position 0 (replay from a fresh prefill): every KV
// cache, and every DeltaNet layer's state. Used by the bench's
// array-vs-imperative greedy-equivalence check.
inline void reset_cache(Model& M) {
  for (int64_t l = 0; l < M.c.NL; l++) {
    M.layers[l].cache.pos = 0;
    if (M.c.is_delta(l)) zero_state_(M.c, M.layers[l]);
  }
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
  const Config& c = M.c;
  PrefillScratch& P = M.pscratch;
  // Absolute position of this chunk's first token — the caches are all at the
  // same place, and kv_cache::prefill appends there, so rope reads it from the
  // same source rather than taking it as a parameter that could disagree.
  const int64_t pos0 = seq_cache(M).pos;
  const int64_t QKVN = (c.NH + 2 * c.NKV) * c.HD;
  gpu::span x = sp(P.emb);
  gpu::rmsnorm(x, sp(M.layers[0].an), sp(P.h), c.NE, c.EPS, T);
  for (int64_t l = 0; l < c.NL; l++) {
    Layer& L = M.layers[l];
    const gpu::span ro = sp(P.res[l & 1]);
    gpu::gemm_bf16_nt(sp(P.h), sp(L.wqkv), sp(P.qkv), T, QKVN, c.NE);
    // split_heads turns the [T, q|k|v] output into head-major [H, T, D] and
    // adds the bias — rope's own fused-bias form only indexes correctly at
    // T == 1, so the bias rides along here instead. One pass over every head;
    // k and v are views into its output.
    gpu::split_heads(sp(P.qkv), sp(L.bqkv), sp(P.qkvh), T, QKVN, 0, c.NH + 2 * c.NKV,
                     c.HD);
    const gpu::span qh = sp(P.qkvh);
    const gpu::span kh = at_f32(qh, c.NH * T * c.HD);
    const gpu::span vh = at_f32(qh, (c.NH + c.NKV) * T * c.HD);
    // Qwen3: every q / k head row (HD wide, one per h*T + t) is RMS-normalized
    // in place, before rope. v is untouched.
    if (c.qk_norm) {
      gpu::rmsnorm(qh, sp(L.qn), qh, c.HD, c.EPS, c.NH * T);
      gpu::rmsnorm(kh, sp(L.kn), kh, c.HD, c.EPS, c.NKV * T);
    }
    // [H,T,D] flattened: row r = h*T + t, so rope's `pos + r % T` is pos0 + t.
    gpu::rope(qh, qh, c.NH * T, T, c.HD, pos0, c.ROPE_BASE, {}, c.ROT);
    gpu::rope(kh, kh, c.NKV * T, T, c.HD, pos0, c.ROPE_BASE, {}, c.ROT);
    L.cache.prefill(qh, kh, vh, sp(P.ah), T, c.NH, c.scale());
    gpu::merge_heads(sp(P.ah), sp(P.at), T, c.NH, c.HD);
    gpu::gemm_bf16_nt(sp(P.at), sp(L.wo_row), ro, T, c.NE, c.NH * c.HD);
    gpu::rmsnorm_res(ro, x, sp(L.fn), ro, sp(P.h2), c.NE, c.EPS, T);
    gpu::gemm_bf16_nt(sp(P.h2), sp(L.wgu), sp(P.gu), T, 2 * c.FF, c.NE);
    gpu::swiglu(sp(P.gu), sp(P.mb), c.FF, T);
    gpu::gemm_bf16_nt(sp(P.mb), sp(L.wd_row), sp(P.md), T, c.NE, c.FF);
    const gpu::span nextw = sp((l + 1 < c.NL) ? M.layers[l + 1].an : M.onorm);
    gpu::rmsnorm_res(ro, sp(P.md), nextw, ro, sp(P.h), c.NE, c.EPS, T);
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
  const Config& c = M.c;
  for (int64_t i = 0; i < n; i++) {
    const uint16_t* r = M.embed_f16 + (size_t)ids[i] * c.NE;
    for (int64_t j = 0; j < c.NE; j++) dst[i * c.NE + j] = f16_to_f32(r[j]);
  }
}

// Stage `n` prompt embeddings starting at ids[from] into the batched scratch.
inline void stage_embed_rows_(Model& M, const std::vector<int>& ids, int64_t from,
                              int64_t n) {
  const Config& c = M.c;
  float* v = M.pscratch.emb_stage.data();
  gather_embed_rows_(M, ids.data() + from, n, v);
  gpu::upload(M.pscratch.emb.native, v, n * c.NE);
}

// Whether a prompt of `T` tokens should go through the batched prefill: the
// backend must have the bf16-weight GEMM, the weights must be the row-major
// ones it consumes, the model must have no DeltaNet layers (their batched form
// is not written yet: a hybrid prompt runs token by token), the prompt must be
// long enough to be worth a batched pass, and it must fit the cache. One place
// answers it, so build(), begin() and
// prefill_batched() cannot disagree.
inline bool can_prefill_batched(const Model& M, int64_t T) {
  return gpu::caps::bf16_gemm && M.row && !M.q4_mlp && !M.layers.empty() &&
         !M.c.hybrid() && T >= PREFILL_MIN && seq_cache(M).pos + T <= M.c.MAXC;
}

// Prefill a whole prompt in chunks and return the greedy next token. Leaves the
// caches at pos_before + ids.size(), so a decoder captured afterwards continues
// from there. Returns -1 if the batched path is unavailable or the prompt does
// not fit the cache.
inline int64_t prefill_batched(Model& M, const std::vector<int>& ids,
                               int64_t chunk = PREFILL_CHUNK) {
  const Config& c = M.c;
  const int64_t T = (int64_t)ids.size();
  if (!can_prefill_batched(M, T) || !M.pscratch.init(c, chunk)) return -1;
  int64_t last = 0;
  for (int64_t s = 0; s < T; s += chunk) {
    last = std::min(chunk, T - s);
    stage_embed_rows_(M, ids, s, last);
    prefill_chunk_(M, last);
  }
  // Only the final row of the final chunk needs logits — one GEMV for the whole
  // prompt rather than one per chunk, read in place as a view of that row.
  gemv_w(M.outwT, at_f32(sp(M.pscratch.h), (last - 1) * c.NE), sp(M.scratch.logits),
         c.VOCAB, c.NE);
  return argmax_logits(M);
}

// wdt = the linear-weight storage dtype (f32 exact, or bf16 for the decode
// path). Biases, norms and the embedding stay F32 (elementwise / row-gather).
// q4_mlp / q4_lmhead quantize the large bandwidth-bound gemvs to group-int4 in
// the imperative decode path (bf16 base only). Lossy — greedy diverges from the
// F16 oracle; validated by coherence + divergence, not greedy-exact.
inline Model build(const gg::model& m, tl::dtype wdt = tl::dtype::f32,
                   bool q4_mlp = false, bool q4_lmhead = false,
                   int64_t maxc = 4096, bool oracle = true) {
  const Config c = config_from_gguf(m, maxc);
  // bf16 weights need the backend's bf16 [K,N] GEMV: the lm_head reads it in
  // every mode, and so does every projection off the row path. It has no
  // composition to fall back to, so a backend without it would leave the
  // logits unwritten; it decodes from f32 weights instead.
  if (wdt == tl::dtype::bf16 && !gpu::has_gemv_bf16) wdt = tl::dtype::f32;
  // Row-major [N,K] weights exist only to feed gemv_bf16_row and the prefill
  // GEMM; a backend without them decodes from the column-major [K,N] split-K
  // GEMV every dtype has, so the layout follows the cap, not just the dtype.
  const bool row = (wdt == tl::dtype::bf16) && gpu::caps::row_gemv;
  const bool q4m = q4_mlp && row;             // q4 requires the bf16/row base
  const bool q4lm = q4_lmhead && row;
  // The array path's separate [K,N] weights duplicate what the fused/row copies
  // hold (≈2x the weight memory). Only forward()/step() read them, so a caller
  // that decodes through the imperative path alone can skip them (oracle=false),
  // and a hybrid model, which the array path does not run, never loads them.
  // Without row weights the imperative path still reads wo/wd from there, so
  // those two stay.
  const bool oracle_w = oracle && !c.hybrid();
  const bool col_wo = oracle_w || !row;
  const tl::dtype f32 = tl::dtype::f32;
  Model M{
      .c = c,
      .embed_f16 =
          reinterpret_cast<const uint16_t*>(need(m, "token_embd.weight")->data),
      .outwT = load_w_T(m, c.output_name(), c.NE, c.VOCAB, wdt),  // stays [K,N]
      .onorm = load_f32(m, "output_norm.weight", c.NE),
      // q4 lm_head (imperative): quantize the f32 [K,VOCAB], keep only the q4.
      .outwT_q4 =
          q4lm ? load_w_T(m, c.output_name(), c.NE, c.VOCAB, f32).to_q4() : array{},
      .row = row,
      .q4_mlp = q4m,
      .q4_lmhead = q4lm};
  for (int64_t l = 0; l < c.NL; l++) {
    std::string p = "blk." + std::to_string(l) + ".";
    // Named per field: the ternaries below decide LAYOUT, and a positional list
    // of same-typed arrays is one transposed line away from a silent
    // wrong-weight bug that only shows up as slightly-off logits.
    const std::vector<wpart> gu_parts = {
        {p + "ffn_gate.weight", c.FF}, {p + "ffn_up.weight", c.FF}};
    // [K,N] weights: the array path's own (the oracle), and wo/wd, which the
    // imperative path reads too where it has no row copies.
    auto W = [&](bool want, const char* n, int64_t in, int64_t out) {
      return want ? load_w_T(m, p + n, in, out, wdt) : array{};
    };
    // A fused projection in the decode GEMV's layout: in bf16 mode ROW [N,K]
    // for the warp-per-row gemv (lever A), in f32 column [K,N] for split-K.
    auto fused = [&](const std::vector<wpart>& parts, int64_t in) {
      return row ? load_w_T_cat_row(m, parts, in) : load_w_T_cat(m, parts, in, wdt);
    };
    // Every layer: the norms and the MLP.
    Layer L{
        .wg = W(oracle_w, "ffn_gate.weight", c.NE, c.FF),
        .wu = W(oracle_w, "ffn_up.weight", c.NE, c.FF),
        .wd = W(col_wo, "ffn_down.weight", c.FF, c.NE),
        .an = load_f32(m, p + "attn_norm.weight", c.NE),
        .fn = load_f32(m, p + c.ffn_norm, c.NE),
        // wgu: bf16-row when row & !q4m; column [K,N] in f32; empty when q4m
        // (replaced by wgu_q4). wd_row likewise.
        .wgu = (row && q4m) ? array{} : fused(gu_parts, c.NE),
        .wd_row = row ? (q4m ? array{}
                             : load_w_T_row(m, p + "ffn_down.weight", c.FF, c.NE))
                      : array{},
        // q4 MLP (imperative, when q4m): quantize the f32 [K,N] fused/down.
        .wgu_q4 = q4m ? load_w_T_cat(m, gu_parts, c.NE, f32).to_q4() : array{},
        .wd_q4 =
            q4m ? load_w_T(m, p + "ffn_down.weight", c.FF, c.NE, f32).to_q4() : array{}};
    if (c.is_delta(l)) {
      // Gated DeltaNet. The fused projection's order puts z right before the
      // slot the rule writes its output to (Config::ssm_proj).
      const int64_t C = c.ssm_conv_ch(), V = c.SH * c.SDV;
      L.wssm = fused({{p + "attn_qkv.weight", C},
                      {p + "ssm_alpha.weight", c.SH},
                      {p + "ssm_beta.weight", c.SH},
                      {p + "attn_gate.weight", V}},
                     c.NE);
      L.wsout = row ? load_w_T_row(m, p + "ssm_out.weight", V, c.NE)
                    : load_w_T(m, p + "ssm_out.weight", V, c.NE, wdt);
      // The conv taps are F32 [C][SCONV] in the GGUF; the op reads them
      // tap-major, so a transpose here.
      const auto* cw = reinterpret_cast<const float*>(need(m, p + "ssm_conv1d.weight")->data);
      std::vector<float> taps((size_t)c.SCONV * C);
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t k = 0; k < c.SCONV; k++)
          taps[(size_t)k * C + ch] = cw[(size_t)ch * c.SCONV + k];
      L.wconv = array::from(std::move(taps), {c.SCONV, C});
      L.adt = load_f32_cat(m, {{p + "ssm_a", c.SH}, {p + "ssm_dt.bias", c.SH}});
      L.sn = load_f32(m, p + "ssm_norm.weight", c.SDV);
      L.conv_state = scratch_f32((c.SCONV - 1) * C);
      L.ssm_state = scratch_f32(c.SH * c.SDK * c.SDV);
      zero_state_(c, L);
    } else {
      // Attention. Qwen3.5's q projection interleaves q and its output gate per
      // head; the fused load takes q's rows first and the gate's last, so the
      // output is q | k | v | gate and every slice downstream is contiguous.
      const int64_t Q = c.NH * c.HD;
      std::vector<wpart> qkv_parts = {
          c.attn_gate ? wpart{p + "attn_q.weight", Q, c.HD, 2 * c.HD, 0}
                      : wpart{p + "attn_q.weight", Q},
          {p + "attn_k.weight", c.NKV * c.HD},
          {p + "attn_v.weight", c.NKV * c.HD}};
      if (c.attn_gate) qkv_parts.push_back({p + "attn_q.weight", Q, c.HD, 2 * c.HD, c.HD});
      auto bias = [&](const char* n, int64_t len) {
        return c.qkv_bias ? load_f32(m, p + n, len) : array{};
      };
      auto qk_norm = [&](const char* n) {
        return c.qk_norm ? load_f32(m, p + n, c.HD) : array{};
      };
      L.wq = W(oracle_w, "attn_q.weight", c.NE, Q);
      L.bq = bias("attn_q.bias", Q);
      L.wk = W(oracle_w, "attn_k.weight", c.NE, c.NKV * c.HD);
      L.bk = bias("attn_k.bias", c.NKV * c.HD);
      L.wv = W(oracle_w, "attn_v.weight", c.NE, c.NKV * c.HD);
      L.bv = bias("attn_v.bias", c.NKV * c.HD);
      L.wo = W(col_wo, "attn_output.weight", Q, c.NE);
      L.qn = qk_norm("attn_q_norm.weight");
      L.kn = qk_norm("attn_k_norm.weight");
      L.wqkv = fused(qkv_parts, c.NE);
      // bqkv = [bq|bk|bv], matching wqkv's column order (batched path).
      if (row && c.qkv_bias)
        L.bqkv = load_f32_cat(m, {{p + "attn_q.bias", Q},
                                  {p + "attn_k.bias", c.NKV * c.HD},
                                  {p + "attn_v.bias", c.NKV * c.HD}});
      if (row) L.wo_row = load_w_T_row(m, p + "attn_output.weight", Q, c.NE);
      L.cache.init(c.NKV, c.MAXC, c.HD);
    }
    M.layers.push_back(std::move(L));
  }
  M.scratch.init(M.c);  // persistent imperative-step buffers (GPU must be active)
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
    gpu::flush();
    reset_cache(M);
  }
  return M;
}

inline array embed_row(const Model& M, int64_t id) {
  const Config& c = M.c;
  std::vector<float> v(c.NE);
  const int one = (int)id;
  gather_embed_rows_(M, &one, 1, v.data());
  return array::from(std::move(v), {1, c.NE});
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

// Embed + NL decoder layers + final RMSNorm, returning the unevaluated logits
// array [1, VOCAB]. Shared by step() (host D2H, the numeric-oracle path) and
// step_greedy() (GPU argmax, the generation path). Optional out-params record
// the layer-0 residual / final-norm for checkpointing; `prof` accumulates the
// per-region overhead census.
inline array forward(Model& M, int64_t id, int64_t pos,
                     std::vector<float>* l0_out, std::vector<float>* fnorm_out,
                     StepProf* prof) {
  const Config& c = M.c;
  if (!M.layers[0].wq.defined())
    throw std::logic_error(
        "qwen: the array path needs build(..., oracle=true) and a non-hybrid model");
  double t = prof ? StepProf::now_ms() : 0.0;
  array x = embed_row(M, id);
  if (prof) { prof->embed += StepProf::now_ms() - t; }
  for (int64_t l = 0; l < c.NL; l++) {
    Layer& L = M.layers[l];
    if (prof) t = StepProf::now_ms();
    array h = array::rmsnorm(x, L.an, c.EPS);
    array qp = h.dot(L.wq), kp = h.dot(L.wk), v = h.dot(L.wv);
    if (c.qkv_bias) { qp = qp + L.bq; kp = kp + L.bk; v = v + L.bv; }
    qp = qp.reshape({c.NH, c.HD});
    kp = kp.reshape({c.NKV, c.HD});
    if (c.qk_norm) {  // per-head RMSNorm over HD, before rope
      qp = array::rmsnorm(qp, L.qn, c.EPS);
      kp = array::rmsnorm(kp, L.kn, c.EPS);
    }
    array q = array::rope(qp, pos, c.ROPE_BASE);
    array k = array::rope(kp, pos, c.ROPE_BASE);
    // v: [1, NKV*HD]
    if (prof) { prof->construct += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    // realize() (not eval()): launch q/k/v on the null stream WITHOUT syncing;
    // the kv_cache append/attn kernels are stream-ordered after them, so they
    // see the writes. Removes 3 CtxSynchronize/layer.
    q.realize(); k.realize(); v.realize();
    if (prof) { prof->qkv_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    L.cache.append(sp(k), sp(v));
    array a_out = array::empty({c.NH, c.HD});
    L.cache.attn(sp(q), sp(a_out), c.NH, c.scale());
    if (prof) { prof->cache += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    array x1 = x + a_out.reshape({1, c.NH * c.HD}).dot(L.wo);
    array h2 = array::rmsnorm(x1, L.fn, c.EPS);
    array mlp = array::swiglu(h2.dot(L.wg), h2.dot(L.wu)).dot(L.wd);
    array x2 = x1 + mlp;
    if (prof) { prof->construct += StepProf::now_ms() - t; t = StepProf::now_ms(); }
    // realize() keeps x2 in flight (no sync) but adopts its storage, so the
    // next layer reads it by stream order and the graph stays bounded. Removes
    // the 4th CtxSynchronize/layer; the whole step drains at the logits tail.
    x2.realize();
    if (prof) { prof->x2_eval += StepProf::now_ms() - t; }
    if (l == 0 && l0_out) { const float* p = x2.raw(); l0_out->assign(p, p + c.NE); }
    x = x2;
  }
  array xf = array::rmsnorm(x, M.onorm, c.EPS);
  if (fnorm_out) { xf.eval(); const float* p = xf.raw(); fnorm_out->assign(p, p + c.NE); }
  return xf.dot(M.outwT);
}

// One decode step. Returns logits [VOCAB] on the host — the numeric-oracle path
// (check_qwen compares the full vector). Pays the 608KB VOCAB D2H.
inline std::vector<float> step(Model& M, int64_t id, int64_t pos,
                               std::vector<float>* l0_out = nullptr,
                               std::vector<float>* fnorm_out = nullptr,
                               StepProf* prof = nullptr) {
  const Config& c = M.c;
  array logits = forward(M, id, pos, l0_out, fnorm_out, prof);
  double t = prof ? StepProf::now_ms() : 0.0;
  logits.eval();
  if (prof) { prof->logits_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
  const float* p = logits.raw();
  std::vector<float> out(p, p + c.VOCAB);
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
  const Config& c = M.c;
  array logits = forward(M, id, pos, nullptr, nullptr, prof);
  double t = prof ? StepProf::now_ms() : 0.0;
  // realize() (not eval()): the lm_head gemv launches without syncing; the
  // argmax kernel is ordered after it and gpu::argmax does the single terminal
  // drain before its 4-byte read. One sync for the whole step.
  logits.realize();
  if (prof) { prof->logits_eval += StepProf::now_ms() - t; t = StepProf::now_ms(); }
  int64_t idx = 0;
  if (!gpu::argmax(sp(logits), c.VOCAB, &idx)) {
    const float* p = logits.raw();
    idx = 0;
    for (int64_t i = 1; i < c.VOCAB; i++)
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
// storage [packed [N][K/2] | scales [N][K/32]]); the scales are a view into the
// same buffer (they ride along the base upload — the array path's split too).
inline bool gemv_q4_w(const array& Wq, gpu::span a, gpu::span y) {
  const int64_t K = Wq.shape()[0], N = Wq.shape()[1];
  const gpu::span qw = sp(Wq);
  return gpu::gemv_q4(a, qw, qw.at(N * K / 2), y, N, K, tl::kQ4Group);
}

// The NL decoder layers + final RMSNorm + lm_head gemv as direct gpu:: calls
// on the Scratch buffers, reading layer-0 residual from x0 and writing logits
// to S.logits. NO array nodes, NO host sync / blocking copy — so this region
// is graph-capturable where a backend has capture (embed staging + argmax stay
// outside). All kernels
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
// layer is at the same sequence position). A null d_pos = the normal host path.
inline void run_layers_(Model& M, gpu::span x0, int64_t pos,
                        gpu::span d_pos = {}, gpu::span logits_out = {}) {
  const Config& c = M.c;
  const bool cap = static_cast<bool>(d_pos);
  Scratch& S = M.scratch;
  // Fused seams (kills 4 elementwise launches/layer): q/k bias folds into rope
  // (gpu::rope's bias arg); the two residual adds fold into the following RMSNorms
  // via rmsnorm_res (writes the residual sum AND its norm). The layer input's
  // norm is therefore produced by the *previous* layer's mlp seam — so precompute
  // layer 0's here, and carry each layer's next-input norm in hb.
  // Decode GEMV: warp-per-row [N,K] (lever A, bf16 imperative) in row mode, else
  // the split-K [K,N] path — so `W` must be the weight in THIS mode's layout
  // (the fused wqkv/wgu already are; wo/wd have a separate row copy). The y
  // layout is identical either way (contiguous [n]), so the fused-output slices
  // below are unchanged.
  const bool row = M.row;
  auto gv = [&](const array& W, gpu::span a, gpu::span y, int64_t n, int64_t k) {
    if (row) gpu::gemv_bf16_row(a, sp(W), y, n, k);
    else gemv_w(W, a, y, n, k);
  };
  gpu::span x = x0;
  gpu::rmsnorm(x, sp(M.layers[0].an), sp(S.hb), c.NE, c.EPS);
  for (int64_t l = 0; l < c.NL; l++) {
    Layer& L = M.layers[l];
    const gpu::span ro = sp(S.res[l & 1]);  // res_out (x1 then x2), ping-pong
    if (c.is_delta(l)) {
      // Gated DeltaNet: one GEMV -> q|k|v | alpha | beta | z in S.qkvb, the conv
      // over q|k|v, the rule (its output lands right after z), the gated norm
      // silu(z) * rmsnorm(o) as one swiglu over [z | o], then the out projection.
      const int64_t C = c.ssm_conv_ch(), V = c.SH * c.SDV;
      const gpu::span pj = sp(S.qkvb), o = at_f32(pj, c.ssm_proj());
      gv(L.wssm, sp(S.hb), pj, c.ssm_proj(), c.NE);
      gpu::ssm_conv_step(pj, sp(L.conv_state), sp(L.wconv), sp(S.convb), C, c.SCONV);
      gpu::gated_delta_step(sp(S.convb), at_f32(pj, C), sp(L.adt), sp(L.ssm_state), o,
                            c.SH, c.SDK, c.SDV, c.L2EPS);
      gpu::rmsnorm(o, sp(L.sn), o, c.SDV, c.EPS, c.SH);
      gpu::swiglu(at_f32(pj, C + 2 * c.SH), sp(S.ab), V);
      gv(L.wsout, sp(S.ab), ro, c.NE, V);  // ro = mixer @ ssm_out
    } else {
      // Fused QKV: one GEMV -> [q(NH*HD) | k(NKV*HD) | v(NKV*HD)] in S.qkvb (then
      // Qwen3.5's gate), and q, k and v are views of it: rope q & k in place,
      // bias-add v. Same per-column split-K as the separate wq/wk/wv GEMVs (bx=1,
      // chunk=32), so bit-identical per column.
      gv(L.wqkv, sp(S.hb), sp(S.qkvb), c.attn_proj(), c.NE);
      const gpu::span qp = sp(S.qkvb);
      const gpu::span kp = at_f32(qp, c.NH * c.HD);
      const gpu::span vp = at_f32(qp, (c.NH + c.NKV) * c.HD);
      // Qwen3: RMSNorm every q / k head (one row of HD each) in place, before rope.
      if (c.qk_norm) {
        gpu::rmsnorm(qp, sp(L.qn), qp, c.HD, c.EPS, c.NH);
        gpu::rmsnorm(kp, sp(L.kn), kp, c.HD, c.EPS, c.NKV);
      }
      if (cap) {
        gpu::rope_dpos(qp, qp, c.NH, 1, c.HD, d_pos, c.ROPE_BASE, sp(L.bq), c.ROT);
        gpu::rope_dpos(kp, kp, c.NKV, 1, c.HD, d_pos, c.ROPE_BASE, sp(L.bk), c.ROT);
      } else {
        gpu::rope(qp, qp, c.NH, 1, c.HD, pos, c.ROPE_BASE, sp(L.bq), c.ROT);
        gpu::rope(kp, kp, c.NKV, 1, c.HD, pos, c.ROPE_BASE, sp(L.bk), c.ROT);
      }
      if (c.qkv_bias) gpu::binary(gpu::kop::add, vp, sp(L.bv), vp, c.NKV * c.HD, 1, 0);
      if (cap) {
        L.cache.append_dpos(kp, vp, d_pos);
        L.cache.attn_dpos(qp, sp(S.ab), c.NH, d_pos, c.scale());
      } else {
        L.cache.append(kp, vp);
        L.cache.attn(qp, sp(S.ab), c.NH, c.scale());
      }
      // Qwen3.5: the attention output times sigmoid of its gate (q|k|v's tail).
      gpu::span ao = sp(S.ab);
      if (c.attn_gate) {
        const int64_t Q = c.NH * c.HD;
        const gpu::span gate = at_f32(qp, (c.NH + 2 * c.NKV) * c.HD);
        gpu::unary(gpu::kop::sigmoid, gate, sp(S.gateb), Q, 1, 0);
        gpu::binary(gpu::kop::mul, ao, sp(S.gateb), sp(S.gateb), Q, 1, 0);
        ao = sp(S.gateb);
      }
      gv(row ? L.wo_row : L.wo, ao, ro, c.NE, c.NH * c.HD);  // ro = attn @ wo
    }
    // x1 = x + (attn@wo); h2 = rmsnorm(x1, fn) — fused.
    gpu::rmsnorm_res(ro, x, sp(L.fn), ro, sp(S.h2b), c.NE, c.EPS);
    // Fused gate|up: one GEMV -> [gate(FF) | up(FF)] in S.gub; swiglu reads both.
    if (M.q4_mlp) gemv_q4_w(L.wgu_q4, sp(S.h2b), sp(S.gub));  // [NE, 2*FF]
    else gv(L.wgu, sp(S.h2b), sp(S.gub), 2 * c.FF, c.NE);
    gpu::swiglu(sp(S.gub), sp(S.mb), c.FF);
    if (M.q4_mlp) gemv_q4_w(L.wd_q4, sp(S.mb), sp(S.mdb));  // [FF, NE]
    else gv(row ? L.wd_row : L.wd, sp(S.mb), sp(S.mdb), c.NE, c.FF);
    // x2 = x1 + mlp; next input norm = rmsnorm(x2, next an | final onorm) — fused.
    const gpu::span nextw = sp((l + 1 < c.NL) ? M.layers[l + 1].an : M.onorm);
    gpu::rmsnorm_res(ro, sp(S.mdb), nextw, ro, sp(S.hb), c.NE, c.EPS);
    x = ro;
  }
  // hb now holds the final RMSNorm output (folded into the last layer's seam).
  const gpu::span logits = logits_out ? logits_out : sp(S.logits);
  if (M.q4_lmhead) gemv_q4_w(M.outwT_q4, sp(S.hb), logits);  // [NE, VOCAB]
  else gemv_w(M.outwT, sp(S.hb), logits, c.VOCAB, c.NE);
  // Tail of the captured region: advance the shared device pos so the next
  // graph replay reads pos+1 (lm_head is pos-independent, so order vs it is free).
  if (cap) gpu::incr_u32(d_pos);
}

// Fully imperative decode step (C1): embed row -> run_layers_ -> GPU argmax.
// Kills the ~45% host graph-construction cost and per-step allocation; one sync
// (inside gpu::argmax). Numerically identical to step_greedy (same
// gemv/rope/attn kernels; fused rmsnorm/swiglu match the array compositions) —
// guarded greedy-exact by check_qwen and bench_qwen_decode. Returns the token.
inline int64_t step_imperative(Model& M, int64_t id, int64_t pos) {
  array e = embed_row(M, id);
  e.realize();  // embed on device (native valid; uploaded on first read)
  run_layers_(M, sp(e), pos);
  return argmax_logits(M);
}

// Run the imperative forward and return the logits [VOCAB] on the host (device
// mirror of scratch.logits). For the q4-vs-bf16 divergence measurement — the
// quantization-quality analogue of step_imperative's greedy token.
inline const float* imperative_logits(Model& M, int64_t id, int64_t pos) {
  array e = embed_row(M, id);
  e.realize();
  run_layers_(M, sp(e), pos);
  return host_read(M.scratch.logits);
}

// Gather the embedding row for token `id` into the staged step input S.embed
// (host gather + blocking upload; marks it device-current). Kept out of a
// captured region since it involves a host gather + blocking copy.
inline void stage_embed(Model& M, int64_t id) {
  const Config& c = M.c;
  const int one = (int)id;
  gather_embed_rows_(M, &one, 1, M.scratch.embed_stage.data());
  gpu::upload(M.scratch.embed.native, M.scratch.embed_stage.data(), c.NE);
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
  storage d_pos;                      // device u32 position counter (capture)
  gpu::graph_exec exec = nullptr;     // opaque handle; non-null == captured
  int64_t cur_pos = 0, max_ctx = 0;   // capacity bound owned by the mechanism
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
    if (M.layers.empty()) return;
    max_ctx = seq_cache(M).max_ctx;  // bound applies to the fallback too
    cur_pos = pos;
    // The captured forms of a hybrid model's position-dependent ops — partial
    // rope, head_dim 256 attention — are not written yet, so it decodes
    // uncaptured.
    if (!gpu::graph_available() || M.c.hybrid()) return;
    d_pos = storage::make(1);
    if (!d_pos.native) return;
    const gpu::span dp = sp(d_pos);
    stage_embed(M, first_id);
    gpu::upload_u32(d_pos.native, (unsigned)pos);
    // Warm run: real launches, but its logits go to the throwaway sink so a
    // prompt's logits (already in scratch when begin() batched) survive.
    run_layers_(M, sp(M.scratch.embed), pos, dp, sp(M.scratch.logits_sink));
    gpu::flush();
    gpu::upload_u32(d_pos.native, (unsigned)pos);  // reset after warm
    if (!gpu::capture_begin()) return;
    run_layers_(M, sp(M.scratch.embed), pos, dp);  // recorded, not executed
    exec = gpu::capture_end();
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
        init(M, next, seq_cache(M).pos);
        return next;
      }
    }
    // Token-by-token: one capture serves prompt and decode alike.
    init(M, ids[0], seq_cache(M).pos);
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
  // S.embed while the previous replay still reads it. Without a graph this is
  // step_imperative minus the argmax, on the same scratch and host cache pos.
  bool feed(Model& M, int64_t id) {
    if (cur_pos >= max_ctx) return false;
    stage_embed(M, id);  // gather id's row -> S.embed (host + blocking H2D)
    if (exec) gpu::graph_launch(exec);  // replay: append@d_pos, attn, logits, incr
    else run_layers_(M, sp(M.scratch.embed), cur_pos);  // host-pos imperative
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
    if (exec) gpu::graph_destroy(exec);
    exec = nullptr;
    d_pos = storage{};  // the pool takes the counter back
  }
};

inline std::string default_path() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/models/qwen2.5-0.5b-instruct-fp16.gguf";
}

// Deterministic short-prefill fixture shared by the decode benches (arbitrary
// valid token ids — enough to make the cache/pos realistic without a tokenizer).
constexpr int PREFILL_IDS[4] = {40, 2610, 264, 3974};

}  // namespace qwenmodel
