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
#include "ggml_quant.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

namespace qwenmodel {

using tl::array;
using tl::storage;
namespace gg = tl::gguf;
namespace gpu = tl::gpu;

// The shape of one decoder, read from the GGUF metadata (config_from_gguf)
// rather than compiled in, so Qwen2.5, Qwen3, Qwen3.5, Gemma 3, Gemma 4, Llama 3.2
// and PLaMo 3 — and every size of each — share one forward. Everything that differs between those
// families is a field here: the dims, the norm eps / rope base, and the
// structural switches.
struct Config {
  std::string arch;  // "qwen2" | "qwen3" | "qwen35" | "gemma3" | "gemma4" | "llama" | "plamo3": the family, for drivers
  // HD and FF are the widest layer's (the scratch's sizes), and NKV the most kv heads
  // a layer has; nkv(l), hd(l) and ff(l) a layer's own.
  int64_t NL = 0, NE = 0, NH = 0, NKV = 0, HD = 0, FF = 0, VOCAB = 0;
  int64_t MAXC = 4096;  // KV capacity per layer (f32): nkv(l)*MAXC*hd(l)*8 B per layer
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
  // SK key heads (q and k) serve the SH value heads, value head h reading key
  // head h % SK (the order GGUF files store the value heads in).
  int64_t INTERVAL = 1, SH = 0, SK = 0, SDK = 0, SDV = 0, SCONV = 0;
  std::string ffn_norm = "ffn_norm.weight";  // Qwen3.5: post_attention_norm
  // The norms after the attention and the MLP where post_norms (Gemma 3's names
  // end in .weight, PLaMo 3's do not).
  std::string post_attn_norm, post_ffn_norm;
  // PLaMo 3 stores a layer's q|k|v as one tensor, attn_qkv, and its gate|up as
  // one, ffn_up (the reads take each projection's rows from them: layer_qkv_parts,
  // layer_gu_parts).
  bool fused_qkv = false, fused_gu = false;
  float L2EPS = 1e-6f;  // DeltaNet's q / k L2 norms (the reference's constant)
  // Gemma 3: the embedding times sqrt(NE) (the lookups scale it: gather_embed_rows_,
  // gpu::embed_row) and the MLP's gate through GELU (tanh form) rather than SiLU
  // (act). Gemma 3 and PLaMo 3: an RMSNorm after the attention and after the MLP too
  // (their outputs normalized before the residual adds), and sliding-window
  // attention on the layers `local` marks (a local layer sees the last SWA positions: its KV cache's window), whose RoPE turns at
  // ROPE_BASE_LOCAL, unscaled, while a global layer's turns at ROPE_BASE with the
  // position times ROPE_SCALE (a linear scaling, which any family's GGUF may state).
  float EMB_SCALE = 1.0f;
  bool post_norms = false;
  gpu::glu_act act = gpu::glu_act::silu;  // the MLP's gate activation
  int64_t SWA = 0;
  std::vector<bool> local;  // per layer; empty: every layer global
  float ROPE_BASE_LOCAL = 10000.0f, ROPE_SCALE = 1.0f;
  bool is_local(int64_t l) const { return l < (int64_t)local.size() && local[l]; }
  // Llama: the GGUF converter permuted the q and k rows for llama.cpp's NORM rope,
  // which pairs adjacent dims. The load reads them back in the model's order
  // (wpart::norm_hd), so the kernels turn the halves, as for the other families.
  bool rope_norm = false;
  // ROT/2 divisors of the pairs' angles on the global layers (Model::rope_freqs,
  // rope_of): Llama 3.1 and later's rope_freqs.weight, or YaRN's (PLaMo 3).
  bool rope_factors = false;
  // YaRN (rope.scaling.type "yarn"), on the global layers: their divisors, and
  // their scores scaled by its mscale squared (scale).
  std::optional<gpu::yarn> yarn;
  // The attention's q scale: 1 / sqrt(HD), unless the family says otherwise
  // (QSCALE > 0: Gemma 3 27B's 1 / sqrt(NE / NH), Gemma 4's 1); a global layer's times YaRN's.
  float QSCALE = 0.0f;
  float scale(int64_t l) const {
    const float s = QSCALE > 0.0f ? QSCALE : 1.0f / std::sqrt((float)HD);
    return yarn && !is_local(l) ? s * yarn->mscale() * yarn->mscale() : s;
  }
  // Gemma 4. Its local layers' heads and RoPE are narrower than its global ones'
  // (HD_LOCAL, ROT_LOCAL; 0 where they are not), and its MLP's width is a layer's
  // own (FF_L; empty: FF on every layer), and so are its kv heads (NKV_L; the 12B and
  // 31B have fewer on the global layers). Its last KV_SHARED layers keep no K and V:
  // each reads the cache of the last layer of its kind before them (kv_src, as
  // llama.cpp's kv_source). v is RMS-normalized without a weight (v_norm). The layers
  // k_eq_v marks (the 12B's and 31B's global ones: HF's attention_k_eq_v) have no v
  // projection: v is the k projection's output, before k's norm and RoPE, which the
  // load makes by reading k's rows for v's too (layer_qkv_parts). PLE wide
  // per-layer embeddings a layer (Model::ple_*, ple_inputs_, layer_tail_), and the
  // logits capped by SOFTCAP: tanh(logit / SOFTCAP) * SOFTCAP.
  int64_t HD_LOCAL = 0, ROT_LOCAL = 0, KV_SHARED = 0, PLE = 0;
  std::vector<int64_t> FF_L, NKV_L;
  std::vector<bool> k_eq_v;  // per layer; empty: every layer has its v projection
  bool v_norm = false;
  float SOFTCAP = 0.0f;
  int64_t hd(int64_t l) const { return is_local(l) && HD_LOCAL ? HD_LOCAL : HD; }
  int64_t rot(int64_t l) const { return is_local(l) && ROT_LOCAL ? ROT_LOCAL : ROT; }
  int64_t ff(int64_t l) const { return FF_L.empty() ? FF : FF_L[l]; }
  int64_t nkv(int64_t l) const { return NKV_L.empty() ? NKV : NKV_L[l]; }
  bool v_is_k(int64_t l) const { return l < (int64_t)k_eq_v.size() && k_eq_v[l]; }
  int64_t kv_src(int64_t l) const {
    const int64_t first = NL - KV_SHARED;
    if (l < first) return l;
    for (int64_t j = first - 1; j >= 0; j--)
      if (is_local(j) == is_local(l)) return j;
    return l;  // none of its kind before (refused when read: config_from_gguf)
  }
  bool has_kv(int64_t l) const { return kv_src(l) == l; }
  bool hybrid() const { return INTERVAL > 1; }
  // The array path (forward(), the per-op oracle) models the plain decoder:
  // no DeltaNet layers, no Gemma post norms or GELU, every layer global, its RoPE
  // neither scaled nor divided by per-pair factors (array::rope has neither).
  bool array_path() const {
    return !hybrid() && !post_norms && act == gpu::glu_act::silu && local.empty() &&
           ROPE_SCALE == 1.0f && !rope_factors && !PLE && !KV_SHARED && !v_norm &&
           SOFTCAP == 0.0f && FF_L.empty() && !HD_LOCAL && NKV_L.empty() && k_eq_v.empty();
  }
  bool is_delta(int64_t l) const { return hybrid() && (l + 1) % INTERVAL != 0; }
  // Layer l's heads in its q|k|v projection: q's, and k's and v's where it keeps
  // its own KV (not where it reads another layer's: Config::kv_src).
  int64_t qkv_heads(int64_t l) const { return NH + (has_kv(l) ? 2 * nkv(l) : 0); }
  // Output widths of the fused projections (and so of the scratch they land in):
  // layer l's q|k|v, and with Qwen3.5's gate after it; and the widest layer's of each.
  int64_t qkv_proj(int64_t l) const { return qkv_heads(l) * hd(l); }
  int64_t attn_proj(int64_t l) const { return qkv_proj(l) + (attn_gate ? NH * hd(l) : 0); }
  int64_t qkv_proj() const { return widest_([&](int64_t l) { return qkv_proj(l); }); }
  int64_t attn_proj() const { return widest_([&](int64_t l) { return attn_proj(l); }); }
  template <class F>
  int64_t widest_(F f) const {
    int64_t w = 0;
    for (int64_t l = 0; l < NL; l++) w = std::max(w, f(l));
    return w;
  }
  int64_t ssm_conv_ch() const { return 2 * SK * SDK + SH * SDV; }  // q|k|v
  int64_t ssm_v() const { return SH * SDV; }                        // the rule's out
  // q|k|v | alpha | beta | z. The decode scratch puts the rule's output right
  // after z, so the gated norm's silu(z) * o reads them as one gate|up buffer.
  int64_t ssm_proj() const { return ssm_conv_ch() + 2 * SH + ssm_v(); }
  // A DeltaNet layer's recurrent state: the conv window, and the rule's.
  int64_t conv_state_n() const { return (SCONV - 1) * ssm_conv_ch(); }
  int64_t ssm_state_n() const { return SH * SDK * SDV; }
  // The first full-attention layer (layer 0 may be DeltaNet).
  int64_t first_attn() const { return INTERVAL - 1; }
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
  if (arch != "qwen2" && arch != "qwen3" && arch != "qwen35" && arch != "gemma3" &&
      arch != "gemma4" && arch != "llama" && arch != "plamo3")
    throw std::runtime_error("qwen: unsupported architecture " + arch);
  auto u32 = [&](const char* k) { return (int64_t)m.kv(arch + "." + k).as_u32(); };
  Config c;
  c.arch = arch;
  c.NL = u32("block_count");
  c.NE = u32("embedding_length");
  c.NH = u32("attention.head_count");
  // A dim that is one number, or one a layer (into `per`, left empty where every layer's
  // is the same): the widest layer's.
  auto layer_dim = [&](const char* k, std::vector<int64_t>& per) {
    const auto& f = m.kv(arch + "." + k);
    if (f.type != gg::val_type::array) return u32(k);
    for (const auto& v : f.arr)  // i32 in the converter's files
      per.push_back(v.type == gg::val_type::i32 ? (int64_t)v.as_i32() : (int64_t)v.as_u32());
    if ((int64_t)per.size() != c.NL) throw std::runtime_error("qwen: " + std::string(k) + " is not one a layer");
    const int64_t widest = *std::max_element(per.begin(), per.end());
    if (std::count(per.begin(), per.end(), widest) == c.NL) per.clear();
    return widest;
  };
  // The kv heads (Gemma 4 12B and 31B: fewer on the global layers) and the MLP's
  // width (Gemma 4's E models), each one a layer or one for all. Every q head reads
  // one kv head: a layer's q heads are a multiple of its kv heads.
  c.NKV = layer_dim("attention.head_count_kv", c.NKV_L);
  c.FF = layer_dim("feed_forward_length", c.FF_L);
  for (int64_t l = 0; l < c.NL; l++)
    if (c.nkv(l) <= 0 || c.NH % c.nkv(l))
      throw std::runtime_error("qwen: layer " + std::to_string(l) + "'s q heads are not a multiple of its kv heads");
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
    // Fewer key heads than value heads are shared (Qwen3.8-27B: 16 for 48).
    // GGUF conversion reorders the value heads so that head h reads key head
    // h % SK, as llama.cpp's ggml_repeat does; the rule's kernels take SK.
    c.SK = u32("ssm.group_count");
    if (c.SK <= 0 || c.SH % c.SK)
      throw std::runtime_error("qwen: DeltaNet value heads not a multiple of key heads");
  }
  // The switches are read off the first attention layer (layer 0 may be DeltaNet).
  const std::string a0 = "blk." + std::to_string(c.first_attn()) + ".";
  c.qkv_bias = m.tensor(a0 + "attn_q.bias") != nullptr;
  c.qk_norm = m.tensor(a0 + "attn_q_norm.weight") != nullptr;
  // A fused q|k|v must be exactly those rows (no gate among them).
  auto rows = [&](const std::string& n) {
    const auto* t = m.tensor(n);
    return t && t->dims.size() == 2 ? (int64_t)t->dims[1] : int64_t{-1};
  };
  c.fused_qkv = rows(a0 + "attn_q.weight") < 0 && rows(a0 + "attn_qkv.weight") >= 0;
  if (c.fused_qkv) {
    if (rows(a0 + "attn_qkv.weight") != c.qkv_proj(c.first_attn()))
      throw std::runtime_error("qwen: " + a0 + "attn_qkv.weight is not q|k|v");
  } else {
    const int64_t wq = rows(a0 + "attn_q.weight");
    if (wq < 0) throw std::runtime_error("qwen: " + a0 + "attn_q.weight missing");
    c.attn_gate = wq == 2 * c.NH * c.HD;
  }
  c.fused_gu = !m.tensor("blk.0.ffn_gate.weight");
  if (c.fused_gu && rows("blk.0.ffn_up.weight") != 2 * c.FF)
    throw std::runtime_error("qwen: no ffn_gate, and ffn_up is not gate|up");
  c.tied_output = m.tensor("output.weight") == nullptr;
  // Llama, as llama.cpp's LLM_ARCH_LLAMA: rope type NORM, and the per-pair factors
  // where the file has them. Only the q and k projections' rows are read back in
  // the model's order (wpart::norm_hd): a q / k bias, which the converter permutes
  // with them, or a q / k norm would stay in the file's, so such a file is refused.
  c.rope_norm = arch == "llama";
  if (c.rope_norm && (c.ROT != c.HD || c.attn_gate || c.qkv_bias || c.qk_norm))
    throw std::runtime_error("qwen: a NORM rope over part of a head, or with a q gate, a q/k "
                             "bias or a q/k norm");
  if (const auto* rf = m.tensor("rope_freqs.weight")) {
    if (rf->dims.size() != 1 || (int64_t)rf->dims[0] != c.ROT / 2)
      throw std::runtime_error("qwen: rope_freqs.weight is not ROT/2 factors");
    c.rope_factors = true;
  }
  // A layer norm's name: with .weight, or bare (PLaMo 3's post norms); "" if neither.
  auto norm = [&](const char* base) -> std::string {
    for (std::string n : {std::string(base) + ".weight", std::string(base)})
      if (m.tensor("blk.0." + n)) return n;
    return "";
  };
  // Qwen3.5 names the MLP's input norm post_attention_norm; Gemma 3 and PLaMo 3
  // have an ffn_norm and a post_attention_norm both, and a post_ffw_norm: the
  // norms of the attention's and the MLP's outputs.
  const std::string post_attn = norm("post_attention_norm");
  c.post_ffn_norm = norm("post_ffw_norm");
  c.post_norms = !c.post_ffn_norm.empty();
  if (c.post_norms) {
    if (post_attn.empty()) throw std::runtime_error("qwen: a post_ffw_norm, no post_attention_norm");
    c.post_attn_norm = post_attn;
  } else if (!post_attn.empty()) {
    c.ffn_norm = post_attn;
  }
  // RoPE's scaling: none, linear (Gemma 3), or YaRN (PLaMo 3), each on the global
  // layers. A key this read does not know is refused rather than run without.
  const std::string rs = arch + ".rope.scaling.";
  const std::string scaling = m.has(rs + "type") ? m.kv(rs + "type").as_str() : "none";
  for (const auto& [k, v] : m.metadata()) {
    if (k.rfind(rs, 0) != 0) continue;
    const std::string key = k.substr(rs.size());
    const bool known = key == "type" || (scaling != "none" && key == "factor") ||
                       (scaling == "yarn" && (key == "original_context_length" || key == "yarn_beta_fast" ||
                                              key == "yarn_beta_slow" || key == "yarn_truncate"));
    if (!known) throw std::runtime_error("qwen: rope scaling " + scaling + " with " + k);
  }
  if (scaling == "linear") {
    c.ROPE_SCALE = 1.0f / m.kv(rs + "factor").as_f32();
  } else if (scaling == "yarn") {
    gpu::yarn& y = c.yarn.emplace();
    y.factor = m.kv(rs + "factor").as_f32();
    y.orig_ctx = (int64_t)m.kv(rs + "original_context_length").as_u32();
    if (m.has(rs + "yarn_beta_fast")) y.beta_fast = m.kv(rs + "yarn_beta_fast").as_f32();
    if (m.has(rs + "yarn_beta_slow")) y.beta_slow = m.kv(rs + "yarn_beta_slow").as_f32();
    if (m.has(rs + "yarn_truncate")) y.truncate = m.kv(rs + "yarn_truncate").as_bool();
    if (c.rope_factors) throw std::runtime_error("qwen: YaRN and rope_freqs.weight both");
    c.rope_factors = true;
  } else if (scaling != "none") {
    throw std::runtime_error("qwen: rope scaling " + scaling);
  }
  // Sliding-window attention on the layers `local` marks (Gemma 3, PLaMo 3).
  if (m.has(arch + ".attention.sliding_window") && u32("attention.sliding_window") > 0) {
    c.SWA = u32("attention.sliding_window");
    // As llama.cpp's load_swa_pattern: a per-layer array of flags, or n (layer
    // l local when l % n < n - 1), when the file says neither the family's n
    // (gemma3.cpp's 6, plamo3.cpp's 8).
    c.local.assign(c.NL, false);
    const std::string key = arch + ".attention.sliding_window_pattern";
    if (m.has(key) && m.kv(key).type == gg::val_type::array) {
      const auto& a = m.kv(key).arr;
      for (int64_t l = 0; l < c.NL && l < (int64_t)a.size(); l++) c.local[l] = a[l].u != 0;
    } else {
      const int64_t n = m.has(key) ? u32("attention.sliding_window_pattern") : arch == "plamo3" ? 8 : 6;
      for (int64_t l = 0; l < c.NL; l++) c.local[l] = n == 0 || l % n < n - 1;
    }
    if (m.has(arch + ".rope.freq_base_swa")) c.ROPE_BASE_LOCAL = m.kv(arch + ".rope.freq_base_swa").as_f32();
  }
  if (arch == "gemma3") {
    // As llama.cpp's gemma3 (src/models/gemma3.cpp): its norms' weights carry
    // Gemma's +1 already (the GGUF conversion adds it).
    c.EMB_SCALE = std::sqrt((float)c.NE);
    c.act = gpu::glu_act::gelu_tanh;
    if (m.has(arch + ".final_logit_softcapping") &&
        m.kv(arch + ".final_logit_softcapping").as_f32() != 0.0f)
      throw std::runtime_error("qwen: a gemma3 with a final logit softcap");
    // The 27B (62 layers) scales q by 1 / sqrt(NE / NH), the others by 1 / sqrt(HD).
    if (c.NL == 62) c.QSCALE = 1.0f / std::sqrt((float)(c.NE / c.NH));
  }
  if (arch == "gemma4") {
    // As llama.cpp's gemma4 (src/models/gemma4.cpp) and HF's modeling_gemma4.py:
    // Gemma 3's embedding scale and GELU, an attention scale of 1, v normalized
    // without a weight, the logits' softcap, the local layers' own head and RoPE
    // widths, the KV-sharing layers and the per-layer embeddings.
    c.EMB_SCALE = std::sqrt((float)c.NE);
    c.act = gpu::glu_act::gelu_tanh;
    c.QSCALE = 1.0f;
    c.v_norm = true;
    if (m.has(arch + ".final_logit_softcapping")) c.SOFTCAP = m.kv(arch + ".final_logit_softcapping").as_f32();
    c.HD_LOCAL = u32("attention.key_length_swa");
    c.ROT_LOCAL = m.has(arch + ".rope.dimension_count_swa") ? u32("rope.dimension_count_swa") : c.HD_LOCAL;
    if (c.HD_LOCAL > c.HD || u32("attention.value_length") != c.HD ||
        u32("attention.value_length_swa") != c.HD_LOCAL)
      throw std::runtime_error("qwen: a gemma4 whose v heads are not its k heads, or local heads wider");
    if (m.has(arch + ".attention.shared_kv_layers")) c.KV_SHARED = u32("attention.shared_kv_layers");
    for (int64_t l = c.NL - c.KV_SHARED; l < c.NL; l++)
      if (c.has_kv(l)) throw std::runtime_error("qwen: a KV-sharing layer with no layer of its kind before");
    // The readers of a q gate and of k / v biases take q|k|v whole (no KV-sharing layer).
    if (c.KV_SHARED && (c.attn_gate || c.qkv_bias))
      throw std::runtime_error("qwen: KV-sharing layers with a q gate or a qkv bias");
    if (m.has(arch + ".embedding_length_per_layer_input")) c.PLE = u32("embedding_length_per_layer_input");
    // A layer that keeps its KV with no attn_v takes k's projection for v (12B, 31B:
    // the global layers). GGUF has no key for it (HF's attention_k_eq_v): the tensor's
    // absence is the switch, as llama.cpp's gemma4.cpp reads it. A v bias would have
    // nothing to add to.
    c.k_eq_v.assign(c.NL, false);
    for (int64_t l = 0; l < c.NL; l++)
      c.k_eq_v[l] = c.has_kv(l) && !m.tensor("blk." + std::to_string(l) + ".attn_v.weight");
    if (std::find(c.k_eq_v.begin(), c.k_eq_v.end(), true) == c.k_eq_v.end()) c.k_eq_v.clear();
    else if (c.qkv_bias || c.fused_qkv) throw std::runtime_error("qwen: a v that is k, with a qkv bias or a fused qkv");
  }
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

// Elements [first, first + n) of a tensor, widened to f32 into dst: the one
// place that reads a weight's bytes, whatever 16- or 32-bit float the file
// stored it as (an F16 GGUF, or a BF16 one) or block-quantized (a format gg::is_quant takes,
// dequantized here: a quantized file runs as bf16 weights, which a model that fits in memory
// as bf16 can do). The type is dispatched once per call, outside the element loop.
inline void widen_(const gg::tensor_info* t, size_t first, size_t n, float* dst) {
  switch (t->type) {
    case gg::ggml_type::f32: {
      const float* src = static_cast<const float*>(t->data) + first;
      std::copy(src, src + n, dst);
      return;
    }
    case gg::ggml_type::f16: {
      const uint16_t* src = static_cast<const uint16_t*>(t->data) + first;
      for (size_t i = 0; i < n; i++) dst[i] = f16_to_f32(src[i]);
      return;
    }
    case gg::ggml_type::bf16: {
      const uint16_t* src = static_cast<const uint16_t*>(t->data) + first;
      for (size_t i = 0; i < n; i++) dst[i] = tl::bf16_to_f32(src[i]);
      return;
    }
    default:
      break;
  }
  if (!gg::is_quant(t->type)) {
    std::fprintf(stderr, "tensor %s: ggml type %u is not a format the loader reads\n",
                 t->name.c_str(), static_cast<unsigned>(t->type));
    std::exit(2);
  }
  // A block format (ggml_quant.h's): whole blocks, a row of K a multiple of the block.
  uint64_t be = 0, bb = 0;
  gg::ggml_block(t->type, be, bb);
  if (first % be || n % be) {
    std::fprintf(stderr, "tensor %s: a block-quantized read must cover whole %llu-weight blocks\n",
                 t->name.c_str(), (unsigned long long)be);
    std::exit(2);
  }
  gg::dequant_range(t->type, t->data, first, n, dst);
}

// Fused weight load: widen several GGML [out,in] tensors that share the same
// `in` (K) and stack them into ONE array, so a single GEMV produces all their
// outputs contiguously (q|k|v or gate|up). Block b occupies output slots
// [base, base+out_b), and the fused output [Ntot] is q|k|v (or gate|up) laid end
// to end — the downstream slices never change. Fusing is the M9 decode lever:
// fewer kernels amortize the ~14us per-launch floor.
//
// The array is [in, Ntot] = [K,N] column-major, what the split-K GEMV and .dot()
// expect: the array path's weights (the oracle), and the imperative path's where
// the weights are not row-major. Bit-identical per column to loading each part
// separately (same widen, same layout). The row-major [N,K] weights are wmats
// (load_wmat below).
//
// A part takes `out` of its tensor's rows from row `first` on (all of them, or one
// projection's of a fused tensor: PLaMo 3's attn_qkv and ffn_up). With blk set it
// takes a strided subset instead — output row o reads source row (o / blk) *
// stride + first + o % blk — which is how Qwen3.5's q projection, q and gate
// interleaved per head, splits into a contiguous q block and a contiguous gate
// block at load.
// `wdt` is the storage dtype: f32 (exact, 4 B/weight) or bf16 (2 B/weight; the
// decode GEMV consumes bf16 natively).
//
// With norm_hd set, the part is a q or k projection whose rows the GGUF converter
// permuted for llama.cpp's NORM rope (convert_hf_to_gguf.py's permute: in a head of
// norm_hd rows, row 2i + s holds the model's row s * norm_hd / 2 + i), and src()
// reads them back in the model's order, the halves the RoPE kernels pair.
struct wpart {
  std::string name;
  int64_t out;
  int64_t blk = 0, stride = 0, first = 0;
  int64_t norm_hd = 0;
  int64_t src(int64_t o) const {
    if (norm_hd) {
      const int64_t half = norm_hd / 2, r = o % norm_hd;
      return o - r + 2 * (r % half) + r / half;
    }
    return blk ? (o / blk) * stride + first + o % blk : first + o;
  }
};
// A layer's projections, each a part of its fused weight (q | k | v, then
// Qwen3.5's gate; gate | up), read from the file's tensors: one a projection, or
// the rows of one fused tensor (Config::fused_qkv, fused_gu). q and k in the
// halves order the kernels turn (a llama file's rows are in NORM order:
// wpart::norm_hd); Qwen3.5's q interleaves q and its gate a head at a time. A layer
// that reads another's KV (Config::kv_src) projects only q, and one whose v is its k
// (Config::v_is_k) reads k's rows for v too: the same projection twice, so v leaves
// the GEMV as k does, before k's norm and RoPE, and every path downstream is a plain
// q|k|v layer's.
inline std::vector<wpart> layer_qkv_parts(const Config& c, int64_t l) {
  const std::string p = "blk." + std::to_string(l) + ".";
  const int64_t hd = c.hd(l), Q = c.NH * hd, KV = c.nkv(l) * hd;
  if (c.fused_qkv) {
    const std::string n = p + "attn_qkv.weight";
    return {{n, Q, 0, 0, 0}, {n, KV, 0, 0, Q}, {n, KV, 0, 0, Q + KV}};
  }
  const int64_t nhd = c.rope_norm ? hd : 0;
  std::vector<wpart> parts = {c.attn_gate ? wpart{p + "attn_q.weight", Q, hd, 2 * hd, 0}
                                          : wpart{p + "attn_q.weight", Q, 0, 0, 0, nhd}};
  if (c.has_kv(l)) {
    const wpart k{p + "attn_k.weight", KV, 0, 0, 0, nhd};
    parts.push_back(k);
    parts.push_back(c.v_is_k(l) ? k : wpart{p + "attn_v.weight", KV});
  }
  if (c.attn_gate) parts.push_back({p + "attn_q.weight", Q, hd, 2 * hd, hd});
  return parts;
}
inline std::vector<wpart> layer_gu_parts(const Config& c, int64_t l) {
  const std::string p = "blk." + std::to_string(l) + ".";
  const int64_t ff = c.ff(l);
  if (c.fused_gu) return {{p + "ffn_up.weight", ff}, {p + "ffn_up.weight", ff, 0, 0, ff}};
  return {{p + "ffn_gate.weight", ff}, {p + "ffn_up.weight", ff}};
}
inline array load_w_T_cat(const gg::model& m, const std::vector<wpart>& parts,
                          int64_t in, tl::dtype wdt) {
  int64_t Ntot = 0;
  for (const auto& p : parts) Ntot += p.out;
  std::vector<float> v((size_t)in * Ntot), rowbuf((size_t)in);
  int64_t base = 0;
  for (const auto& p : parts) {
    const auto* t = need(m, p.name);
    // A source row is contiguous in the file: widened through one row of
    // scratch, then transposed into place.
    for (int64_t o = 0; o < p.out; o++) {
      widen_(t, (size_t)p.src(o) * in, in, rowbuf.data());
      for (int64_t i = 0; i < in; i++) v[(size_t)i * Ntot + (base + o)] = rowbuf[i];
    }
    base += p.out;
  }
  array a = array::from(std::move(v), {in, Ntot});
  return wdt == tl::dtype::bf16 ? a.to_bf16() : a;
}
inline array load_w_T(const gg::model& m, const std::string& n, int64_t in,
                      int64_t out, tl::dtype wdt) {
  return load_w_T_cat(m, {{n, out}}, in, wdt);
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

// Several 1-D tensors (norms, biases, gates' constants) laid end to end ->
// array [1, sum(len)], f32 whatever the file stored them as. The q|k|v biases
// in that order match the fused QKV projection's column layout, so the batched
// path adds all of them in one pass over all heads. load_f32 below is the
// single-tensor case.
inline array load_f32_cat(const gg::model& m,
                          const std::vector<std::pair<std::string, int64_t>>& parts) {
  std::vector<float> v;
  for (const auto& p : parts) {
    const size_t at = v.size();
    v.resize(at + p.second);
    widen_(need(m, p.first), 0, p.second, v.data() + at);
  }
  int64_t len = (int64_t)v.size();
  return array::from(std::move(v), {1, len});
}
inline array load_f32(const gg::model& m, const std::string& n, int64_t len) {
  return load_f32_cat(m, {{n, len}});  // the single-part case
}

// A projection on the device in the row layout [N,K] that the decode GEMV and the
// prefill GEMM read, in a device-only buffer (storage::make_device), so the
// weight sits in memory once: no host copy beside it, and no f32 copy on the way
// in. A run of its rows is in one format: bf16 (ggml type 30), or the file's own
// quantization blocks (Model.quant; a Q4_K weight is 0.56 byte, bf16 2). The
// fused projections (q|k|v, gate|up, ...) are their parts' rows end to end;
// Q4_K_M files mix Q4_K and Q6_K inside one layer, so a fused projection is a
// GEMV per run, writing its slice of the same contiguous output.
struct wmat {
  struct part {
    int type;           // the ggml_type code: 30 bf16, or a block format gpu::quant_reads takes
    int64_t rows;       // output rows of this run
    int64_t byte_off;   // where its rows start in `raw`
    int64_t row_off;    // its first output row
  };
  storage raw;
  std::vector<part> parts;
  int64_t K = 0;
  explicit operator bool() const { return !parts.empty(); }
  gpu::span at(const part& p) const { return sp(raw).at(p.byte_off); }
};
constexpr int bf16_type = static_cast<int>(gg::ggml_type::bf16);

// y = W . a, a GEMV a run. In a prepared form f (gpu::act_form, into `pa`, room for
// gpu::act_bytes_max) the runs whose format has a GEMV for that form
// (gpu::quant_act_reads) read the activation as the form makes it, prepared once for
// the call: int8 (q8) or f16 on the tensor cores (h16) is what makes them run at the
// weights' bandwidth rather than at the float kernel's instruction rate. The other
// runs, f32, and a form the backend has not got read it as float.
inline void gemv_m(const wmat& W, gpu::span a, gpu::span y,
                   gpu::act_form f = gpu::act_form::f32, gpu::span pa = {}) {
  bool prepared = false;
  for (const auto& p : W.parts) {
    const gpu::span w = W.at(p), out = y.at(p.row_off * 4);
    if (p.type == bf16_type) {
      gpu::gemv_bf16_row(a, w, out, p.rows, W.K);
      continue;
    }
    if (gpu::quant_act_reads(p.type, f, W.K) &&
        (prepared || (prepared = gpu::prepare_act(f, a, pa, W.K))) &&
        gpu::gemv_quant_act(p.type, f, pa, w, out, p.rows, W.K))
      continue;
    gpu::gemv_quant(p.type, a, w, out, p.rows, W.K);
  }
}

// The parts' rows (the subset a wpart names, if strided) onto the device. With
// `native` each keeps the file's quantization blocks, which go up straight from
// the file's mapping; otherwise each is bf16, which a BF16 file's rows already
// are and any other float or block format is widened and rounded to, a few rows
// at a time through a small staging buffer, so no copy of the whole weight is
// ever made on the host.
inline wmat load_wmat(const gg::model& m, const std::vector<wpart>& parts, int64_t in,
                      bool native) {
  wmat W;
  W.K = in;
  std::vector<int64_t> row_bytes;
  int64_t off = 0, rows_total = 0;
  for (const auto& p : parts) {
    const auto* t = need(m, p.name);
    const gg::ggml_type ty = native ? t->type : gg::ggml_type::bf16;
    if (native && !gpu::quant_reads((int)ty, in)) {
      std::fprintf(stderr, "tensor %s: this backend does not read its blocks (ggml type %d, %lld "
                   "weights a row) as they are\n", p.name.c_str(), (int)ty, (long long)in);
      std::exit(2);
    }
    const int type = static_cast<int>(ty);
    const int64_t rb = (int64_t)gg::bytes_of(ty, in);
    if (W.parts.empty() || W.parts.back().type != type) {  // a new run, 256-byte aligned
      off = (off + 255) / 256 * 256;
      W.parts.push_back({type, 0, off, rows_total});
    }
    W.parts.back().rows += p.out;
    off += p.out * rb;
    rows_total += p.out;
    row_bytes.push_back(rb);
  }
  const int64_t words = (off + 3) / 4;  // a Q6_K or Q8_0 row ends mid-word
  W.raw = storage::make_device(words, words * 4, tl::dtype::f32);
  // A part goes up in batches of rows, about 2 MB each, put together on the host:
  // rows widened and rounded to bf16, or the file's blocks where its rows are
  // scattered (a NORM-permuted q / k comes a row at a time). On CUDA each upload is
  // a synchronous copy with a fixed cost, so one a row would add up.
  std::vector<float> wide;
  std::vector<char> stage;
  int64_t base = 0;
  size_t pi = 0;
  for (size_t i = 0; i < parts.size(); i++) {
    const wpart& p = parts[i];
    const auto* t = m.tensor(p.name);
    const int64_t rb = row_bytes[i];
    while (pi + 1 < W.parts.size() && W.parts[pi + 1].row_off <= base) pi++;
    const wmat::part& run = W.parts[pi];
    const bool as_stored = run.type == static_cast<int>(t->type);
    const char* data = static_cast<const char*>(t->data);
    const int64_t dst0 = run.byte_off + (base - run.row_off) * rb;
    // Source rows come in contiguous runs (the whole tensor, the blk rows of one
    // head of an interleaved projection, or a single row of a NORM-permuted q / k):
    // the length of the one starting at part row o, within the first `lim` rows.
    auto run_len = [&](int64_t o, int64_t lim) {
      int64_t n = 1;
      while (o + n < lim && p.src(o + n) == p.src(o) + n) n++;
      return n;
    };
    if (as_stored && run_len(0, p.out) == p.out) {  // in order in the file already
      gpu::upload_bytes(W.raw.native, dst0, data + p.src(0) * rb, p.out * rb);
    } else {
      const int64_t batch = std::max<int64_t>(1, (2 << 20) / rb);
      for (int64_t b = 0; b < p.out; b += batch) {
        const int64_t lim = std::min(p.out, b + batch);
        stage.resize((size_t)((lim - b) * rb));
        for (int64_t o = b, n; o < lim; o += n) {
          n = run_len(o, lim);
          char* to = stage.data() + (o - b) * rb;
          if (as_stored) {
            std::memcpy(to, data + p.src(o) * rb, (size_t)(n * rb));
          } else {
            const int64_t len = n * in;
            wide.resize((size_t)len);
            widen_(t, (size_t)p.src(o) * in, (size_t)len, wide.data());
            auto* narrow = reinterpret_cast<uint16_t*>(to);
            for (int64_t j = 0; j < len; j++) narrow[j] = tl::f32_to_bf16(wide[j]);
          }
        }
        gpu::upload_bytes(W.raw.native, dst0 + b * rb, stage.data(), (lim - b) * rb);
      }
    }
    base += p.out;
  }
  return W;
}

struct Layer {
  array wq, bq, wk, bk, wv, bv, wo, wg, wu, wd, an, fn;
  // Qwen3's per-head q / k RMSNorm weights, [1, HD] (empty when !Config::qk_norm).
  // bq/bk/bv above are likewise empty when !Config::qkv_bias.
  array qn, kn;
  // Fused decode weights (imperative path only): q|k|v = [wq|wk|wv] and
  // gate|up = [wg|wu] concatenated. One GEMV each instead of 3/2, amortizing the
  // per-launch floor. The separate wq/.../wu stay resident as the array-path
  // oracle.
  //
  // Where the weights are row-major (Model.row: bf16, or the file's quantized
  // blocks) the decode and prefill read the wmats: [N,K], device-only, the
  // warp-per-row GEMV's layout (lever A, ~1.4-1.9x on the small-N floor-bound
  // shapes); wo_row and wd_row are the attention output and down projections. Otherwise
  // (f32 weights, or a backend without the row GEMV) the split-K GEMV reads
  // column-major [K,N] arrays: wqkv and wgu, and wo/wd shared with the oracle.
  // bqkv = [bq|bk|bv], the fused QKV projection's own column order, so the
  // batched prefill splits all qkv_heads(l) heads (and adds their bias) in one pass.
  wmat wqkv_row, wgu_row, wo_row, wd_row;
  array wqkv, wgu, bqkv;
  // q4 (group-symmetric int4, group=32) copies of the large bandwidth-bound MLP
  // weights (imperative bf16 path, when Model.q4_mlp). ~0.625 B/wt vs bf16's 2 →
  // gate|up 1.74×, down 1.38× in the isolated bench. These REPLACE wgu_row/wd_row (left
  // empty), saving memory. Attention proj (q|k|v, o) stays bf16-row — q4 loses
  // there (still floor-bound; dequant > byte savings).
  array wgu_q4, wd_q4;
  tl::kv_cache cache;  // attention layers only
  // Gated DeltaNet layers (Qwen3.5) instead of the attention weights above:
  // wssm_row the fused [qkv | alpha | beta | z] projection and wsout_row the output
  // projection (wssm / wsout where the weights are column-major, like q|k|v), wconv
  // the conv taps [SCONV, C] tap-major, adt = A | dt_bias [2*SH], sn the gated
  // norm's weight [SDV]; and the recurrent state the layer carries from token
  // to token in place of a KV cache: the conv window [SCONV-1, C] and the
  // rule's state [SH, SDK, SDV].
  wmat wssm_row, wsout_row;
  array wssm, wsout, wconv, adt, sn;
  storage conv_state, ssm_state;
  // Config::post_norms (Gemma 3): the norms of the attention's and the MLP's
  // outputs, [NE] each (rmsnorm_res's wd); empty otherwise.
  array pa, pf;
  // Gemma 4: the residual's scale at the layer's end (layer_output_scale), and the
  // per-layer embedding's gate [PLE, NE], projection [NE, PLE] (f32 rows, as the
  // file has them: mm_f32_) and the norm of what it adds [NE] (Config::PLE; layer_tail_).
  float out_scale = 1.0f;
  array ple_gate, ple_proj, ple_post;
};

// Persistent device scratch for the imperative decode step (C1): every per-step
// intermediate is a device buffer allocated ONCE and reused each token, so the
// step builds zero array nodes (no host graph construction) and does zero
// per-step allocation (no WSL2 churn). res[2] ping-pong the residual stream;
// q/k/v and gate/up live only as slices of their fused projection outputs
// (qkvb, gub).
struct Scratch {
  storage res[2];       // residual ping-pong [NE]
  storage act;          // a block-quantized GEMV's activation prepared in Model::gemv_form
  storage hb;           // input-norm out [NE]
  storage h2b;          // post-attn norm out [NE]
  storage qb;           // [NH*HD] query fixture (bench_qwen_ctx's isolated-
                        // attention timing; decode reads q as a slice of qkvb)
  storage qkvb;         // fused projection out: attention's q|k|v(|gate), or
                        // DeltaNet's q|k|v|alpha|beta|z plus the rule's output
  storage ab;           // attn (or DeltaNet gated-norm) out
  storage convb;        // DeltaNet conv out [ssm_conv_ch]
  storage gateb;        // Qwen3.5 attention: sigmoid(gate) [NH*HD]
  storage mb;           // GLU out [FF]
  storage gub;          // fused gate|up out [2*FF] = [9728]
  storage mdb;          // mlp-down out [NE]
  storage logits;       // [VOCAB]
  storage logits_sink;  // [VOCAB] sink for forwards run only for their side
                        // effects (capture warm-up)
  storage logit_sum;    // [1] the logits' sum: non-finite iff any logit is
  storage embed;        // staged embedding row [NE] (the step's input)
  std::vector<float> embed_stage;  // host gather target, uploaded into embed
  // Config::PLE: the token's own per-layer rows [NL*PLE] (staged), the slots
  // ple_inputs_ fills [(NL+1)*PLE] (and prefill_chunk_, a prompt's last row past
  // the KV-sharing line: nothing lives here across steps), and a layer's gated input [PLE];
  // logits_tmp the softcap's [VOCAB].
  storage ple_tok, ple, ple_g, logits_tmp;
  std::vector<float> ple_stage;
  void init(const Config& c) {
    {  // the widest input of a GEMV (the residual, the MLP's, the heads', DeltaNet's
       // values), up to whole Q4_K superblocks, prepared in any form (a check switches them)
      const int64_t widest = std::max({c.NE, c.FF, c.NH * c.HD, c.hybrid() ? c.ssm_v() : 0});
      act = scratch_f32(std::max<int64_t>(1, (gpu::act_bytes_max((widest + 255) / 256 * 256) + 3) / 4));
    }
    embed = scratch_f32(c.NE);
    embed_stage.assign(c.NE, 0.0f);
    logits = scratch_f32(c.VOCAB);
    res[0] = scratch_f32(c.NE);
    res[1] = scratch_f32(c.NE);
    hb = scratch_f32(c.NE);
    h2b = scratch_f32(c.NE);
    qb = scratch_f32(c.NH * c.HD);
    qkvb = scratch_f32(std::max(c.attn_proj(), c.ssm_proj() + c.ssm_v()));
    ab = scratch_f32(std::max(c.NH * c.HD, c.ssm_v()));
    if (c.hybrid()) convb = scratch_f32(c.ssm_conv_ch());
    if (c.attn_gate) gateb = scratch_f32(c.NH * c.HD);
    mb = scratch_f32(c.FF);
    gub = scratch_f32(2 * c.FF);
    mdb = scratch_f32(c.NE);
    logits_sink = scratch_f32(c.VOCAB);
    logit_sum = scratch_f32(1);
    if (c.PLE) {
      ple_tok = scratch_f32(c.NL * c.PLE);
      ple_stage.assign((size_t)(c.NL * c.PLE), 0.0f);
      ple = scratch_f32((c.NL + 1) * c.PLE);
      ple_g = scratch_f32(c.PLE);
    }
    if (c.SOFTCAP > 0.0f) logits_tmp = scratch_f32(c.VOCAB);
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
  storage qkv;     // fused projection out [cap, attn_proj | ssm_proj]
  // q|k|v head-major in ONE buffer [qkv_heads(l), cap, hd(l)], the widest layer's:
  // one split_heads pass over every head, and k / v are views into it.
  storage qkvh;
  storage ah;  // attn out head-major [NH, cap, HD]
  storage at;  // attn (or DeltaNet gated-norm) out token-major [cap, NH*HD | V]
  // Qwen3.5 attention: the gate copied out of the projection rows, and its
  // sigmoid [cap, NH*HD].
  storage gt, gs;
  // DeltaNet: the conv out [cap, C], and [z | o] — the gate and the rule's
  // output, each [cap, V] — so one swiglu over the flattened rows is the
  // gated norm. zero is the one float a strided copy adds.
  storage conv, zo, zero;
  storage gu;  // fused gate|up [cap, 2*FF]
  storage mb;  // GLU out [cap, FF]
  storage md;  // mlp-down out [cap, NE]
  // The f16 copies the f16-sum GEMMs read (a half-size buffer each, in floats):
  // the two norms' outputs, and the activations only a GEMM reads.
  storage h16, h216, at16, mb16;
  std::vector<float> emb_stage;  // host gather target, uploaded into emb
  // Config::PLE, Scratch's over `cap` rows: the tokens' rows [cap, NL*PLE], the
  // slots [(NL+1), cap, PLE], [gate | ple] rows [cap, 2*PLE] and a layer's gated
  // input [cap, PLE].
  storage ple_tok, ple, ple_gu, ple_g;
  std::vector<float> ple_stage;

  bool init(const Config& c, int64_t chunk) {
    if (cap >= chunk) return true;  // already big enough
    cap = chunk;
    emb = scratch_f32(cap * c.NE);
    emb_stage.assign((size_t)cap * c.NE, 0.0f);
    res[0] = scratch_f32(cap * c.NE);
    res[1] = scratch_f32(cap * c.NE);
    h = scratch_f32(cap * c.NE);
    h2 = scratch_f32(cap * c.NE);
    qkv = scratch_f32(cap * std::max(c.attn_proj(), c.ssm_proj()));
    qkvh = scratch_f32(cap * c.qkv_proj());
    ah = scratch_f32(c.NH * cap * c.HD);
    at = scratch_f32(cap * std::max(c.NH * c.HD, c.ssm_v()));
    if (c.attn_gate) {
      gt = scratch_f32(cap * c.NH * c.HD);
      gs = scratch_f32(cap * c.NH * c.HD);
    }
    if (c.hybrid()) {
      conv = scratch_f32(cap * c.ssm_conv_ch());
      zo = scratch_f32(2 * cap * c.ssm_v());
    }
    zero = scratch_f32(1);
    const float z0 = 0.0f;
    gpu::upload(zero.native, &z0, 1);
    gu = scratch_f32(cap * 2 * c.FF);
    mb = scratch_f32(cap * c.FF);
    md = scratch_f32(cap * c.NE);
    if (c.PLE) {
      ple_tok = scratch_f32(cap * c.NL * c.PLE);
      ple_stage.assign((size_t)(cap * c.NL * c.PLE), 0.0f);
      ple = scratch_f32((c.NL + 1) * cap * c.PLE);
      ple_gu = scratch_f32(cap * 2 * c.PLE);
      ple_g = scratch_f32(cap * c.PLE);
    }
    if constexpr (gpu::f16_activations) {
      auto half = [](int64_t n) { return scratch_f32((n + 1) / 2); };
      h16 = half(cap * c.NE);
      h216 = half(cap * c.NE);
      at16 = half(cap * std::max(c.NH * c.HD, c.ssm_v()));
      mb16 = half(cap * c.FF);
    }
    return md.native != nullptr;
  }
};

struct Model {
  Config c;
  const gg::tensor_info* embed;  // token_embd [vocab, NE] in the file (row-gather source)
  // The lm head. Where the weights are row-major (Model.row) the imperative
  // path reads outw_row [VOCAB, NE], bf16 or the file's blocks, as it reads
  // every projection; outwT [NE, VOCAB] (transposed output.weight) is the
  // column-major one, loaded where the weights are column-major or the array
  // path (the oracle) runs.
  wmat outw_row;
  array outwT;
  array onorm;                // final norm [1, NE]
  array rope_freqs;           // the per-pair RoPE divisors [1, ROT/2] (Config::rope_factors)
  // Config::PLE: the per-layer embeddings' table in the file [VOCAB, NL*PLE] (rows
  // gathered on the host, as the embedding's), the projection of the embedding
  // [NE -> NL*PLE] (bf16 rows, or [K,N] off the row path), and its norm's weight
  // over sqrt(2) [PLE] (ple_inputs_).
  const gg::tensor_info* ple_embd = nullptr;
  wmat ple_proj_row;
  array ple_projT, ple_norm;
  array ones;                 // [HD] of 1: Config::v_norm's weight
  array outwT_q4;             // q4 lm_head (imperative decode, when q4_lmhead)
  tl::dtype wdt = tl::dtype::f32;  // the float weights' dtype, as build() chose
                                   // (bf16 under quant, whose projections are blocks)
  // The form the decode's block-quantized GEMVs read their activation in, where their
  // format has a GEMV for it (TL_GEMV_ACT; gpu::quant_act_reads).
  gpu::act_form gemv_form = gpu::act_form::f32;
  bool mmq = false;           // prefill GEMMs on the int8 tensor cores, formats with MMQ (TL_NO_MMQ: f16)
  bool quant = false;         // the projections are block-quantized blocks, read as such
  std::vector<Layer> layers;
  Scratch scratch;
  PrefillScratch pscratch;  // batched-prefill buffers (lazily sized)
  bool row = false;  // imperative gemvs use warp-per-row [N,K] (bf16 only)
  bool q4_mlp = false;     // imperative MLP gemvs (wgu, wd) use q4
  bool q4_lmhead = false;  // imperative lm_head gemv uses q4
  // The batched prefill's GEMMs take f16 where the backend can
  // (prefill_gemm_f16: f16 sums, twice the tensor-core rate here, or f16
  // operands over f32 sums), at a small cost in precision. A prompt whose
  // logits come out non-finite (an f16 overflow, in these GEMMs or in the
  // attention's f16 operands) is redone exactly (prefill_batched), and
  // counted in f16_retries.
  bool f16_acc = false;
  int64_t f16_retries = 0;
  // The attention layers' KV caches hold f16 where the backend can (caps::kv_f16):
  // half the bytes every decode step streams, and the values the tensor-core
  // prefill rounds K and V to anyway, as llama.cpp's default cache. The decode's
  // attention then reads K and V rounded to f16. An f32-weight model (the exact
  // build) keeps an f32 cache, and so does any with TL_KV_F32=1. init_kv_caches
  // sets it with the caches (check_qwen's reference forward on a quantized file's
  // blocks takes an f32 cache, then the chat's again).
  bool kv_f16 = false;
};

// Layer l's RoPE frequencies: a local layer's base, unscaled, or a global one's,
// its position scaled and its pairs divided by the file's factors where it has them.
inline gpu::rope_freq rope_of(const Model& M, int64_t l) {
  const Config& c = M.c;
  if (c.is_local(l)) return {c.ROPE_BASE_LOCAL, c.rot(l)};
  return {c.ROPE_BASE, c.rot(l), c.ROPE_SCALE, sp(M.rope_freqs)};  // empty: no factors
}

// The KV cache's appends and attentions fail only on a caller's mistake (a position
// past the capacity, an op the backend can neither run nor compose); the forward
// stops there rather than carry stale scratch on.
inline void cache_ok_(bool ok) {
  if (!ok) throw std::logic_error("qwen: the KV cache refused an append or an attention");
}

// Decode GEMV picking the weight-dtype kernel: y(n) = a(1,k) @ W[k,n].
inline bool gemv_w(const array& W, gpu::span a, gpu::span y, int64_t n,
                   int64_t k) {
  return W.dt() == tl::dtype::bf16 ? gpu::gemv_bf16(a, sp(W), y, n, k)
                                   : gpu::gemv_f32(a, sp(W), y, n, k);
}

// y(n) = a(1,k) . a projection: the row-major wmat where the build made one
// (Model.row), else the split-K GEMV over the column-major array. `f` and `pa`
// as gemv_m.
inline void gemv_rw(const wmat& R, const array& W, gpu::span a, gpu::span y, int64_t n,
                    int64_t k, gpu::act_form f = gpu::act_form::f32, gpu::span pa = {}) {
  if (R) gemv_m(R, a, y, f, pa);
  else gemv_w(W, a, y, n, k);
}

// The final logits' cap where the family has one (Config::SOFTCAP: Gemma 4's
// tanh(logit / cap) * cap), in place.
inline void softcap_(const Model& M, gpu::span logits) {
  const Config& c = M.c;
  if (c.SOFTCAP <= 0.0f) return;
  gpu::unary(gpu::kop::affine, logits, sp(M.scratch.logits_tmp), c.VOCAB, 1.0f / c.SOFTCAP, 0.0f);
  gpu::unary_ext(gpu::unary_ext_op::tanh_, sp(M.scratch.logits_tmp), logits, c.VOCAB, c.SOFTCAP, 0.0f);
}

// logits = the lm head . a, capped (softcap_).
inline void lm_head(const Model& M, gpu::span a, gpu::span logits) {
  gemv_rw(M.outw_row, M.outwT, a, logits, M.c.VOCAB, M.c.NE);
  softcap_(M, logits);
}

// y [T, N] = a [T, K] . W^T, an f32 weight in rows [N, K] (GGML's own order): a
// row GEMV for one row.
inline void mm_f32_(const array& W, gpu::span a, gpu::span y, int64_t T, int64_t N, int64_t K) {
  if (T == 1) gpu::gemv_f32_row(a, sp(W), y, N, K);
  else gpu::gemm(a, K, false, sp(W), K, true, y, T, N, K, 1.0f, 0.0f);
}

// Gemma 4's per-layer embeddings of T tokens (Config::PLE; HF's
// get_per_layer_inputs and project_per_layer_inputs) into `slots`, [(NL+1), T, PLE]:
// layer l's at slot l + 1, so that slot l, where layer l's gate lands, sits next to
// it (layer_tail_). The embedding x [T, NE] (already times sqrt(NE)) projected
// [T, NL*PLE], each layer's slice RMS-normalized by the norm's weight over sqrt(2)
// (eps times NE: HF divides the projection by sqrt(NE) first, which is the same
// norm), plus `tok`, the tokens' own rows as staged (stage_ple_rows_: times
// sqrt(PLE / 2)). One row is layer-major already, and goes straight into its
// slots; T rows are summed into `tok` and split a layer at a time from there.
inline void ple_inputs_(const Model& M, gpu::span x, int64_t T, gpu::span tok, gpu::span slots) {
  const Config& c = M.c;
  const int64_t NP = c.NL * c.PLE;
  const gpu::span pj = T == 1 ? at_f32(slots, c.PLE) : slots;  // slots hold >= T * NP
  if (T == 1) gemv_rw(M.ple_proj_row, M.ple_projT, x, pj, NP, c.NE);
  else gpu::gemm_bf16_nt(x, M.ple_proj_row.at(M.ple_proj_row.parts[0]), pj, T, NP, c.NE);
  gpu::rmsnorm(pj, sp(M.ple_norm), pj, c.PLE, c.EPS * (float)c.NE, T * c.NL);
  if (T == 1) {
    gpu::binary(gpu::kop::add, pj, tok, pj, NP, 1.0f, 0.0f);
    return;
  }
  gpu::binary(gpu::kop::add, pj, tok, tok, T * NP, 1.0f, 0.0f);
  gpu::split_heads(tok, {}, at_f32(slots, T * c.PLE), T, NP, 0, c.NL, c.PLE);
}

// The end of layer l over T rows where the family has more than the residual add
// (Gemma 4): x + the MLP's output md post-normed, then (Config::PLE) the per-layer
// embedding's input gelu(x . gate) * ple_l projected back, post-normed and added,
// and the whole residual times the layer's scale. The next layer's input norm is
// the caller's. gu [T, 2*PLE] and g [T, PLE] are scratch; md is reused.
inline void layer_tail_(const Model& M, const Layer& L, int64_t l, int64_t T, gpu::span x,
                        gpu::span md, gpu::span slots, gpu::span gu, gpu::span g) {
  const Config& c = M.c;
  const int64_t n = T * c.NE;
  gpu::rmsnorm(md, sp(L.pf), md, c.NE, c.EPS, T);
  if (c.PLE) {
    gpu::binary(gpu::kop::add, x, md, x, n, 1.0f, 0.0f);
    // Slot l held layer l - 1's input, read by now; the gate lands there, beside ple_l.
    const gpu::span gate = at_f32(slots, l * T * c.PLE);
    mm_f32_(L.ple_gate, x, gate, T, c.PLE, c.NE);
    gpu::span gp = gate;  // [gate | ple_l] a row: one row's slots are
    if (T > 1) {
      gpu::merge_heads(gate, gu, T, 2, c.PLE);
      gp = gu;
    }
    gpu::glu(gpu::glu_act::gelu_tanh, gp, g, c.PLE, T);
    mm_f32_(L.ple_proj, g, md, T, c.NE, c.PLE);
    gpu::rmsnorm(md, sp(L.ple_post), md, c.NE, c.EPS, T);
  }
  gpu::binary(gpu::kop::add, x, md, x, n, L.out_scale, 0.0f);
}

// Whether layer l ends past the residual add (layer_tail_).
inline bool has_tail_(const Model& M, int64_t l) {
  return M.c.PLE > 0 || M.layers[l].out_scale != 1.0f;
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
  return M.layers[M.c.first_attn()].cache;
}

// A DeltaNet layer's recurrent state back to zero: the conv window and the
// rule's state, both read before they are written on the first token.
inline void zero_state_(const Config& c, Layer& L) {
  const std::vector<float> z((size_t)std::max(c.conv_state_n(), c.ssm_state_n()), 0.0f);
  gpu::upload(L.conv_state.native, z.data(), c.conv_state_n());
  gpu::upload(L.ssm_state.native, z.data(), c.ssm_state_n());
}

// A copy of every DeltaNet layer's recurrent state, for a forward run only for
// its side effects (the capture's warm-up) to put back. A KV row such a run
// writes is rewritten by the next real step at the same position; the state is
// advanced in place, so nothing would undo it.
struct delta_snapshot {
  struct saved {
    Layer* L;
    storage conv, ssm;
  };
  std::vector<saved> layers;
  static void copy_(gpu::span from, gpu::span to, int64_t n) { gpu::copy_f32(from, to, n); }
  void save(Model& M) {
    const Config& c = M.c;
    for (int64_t l = 0; l < c.NL; l++) {
      if (!c.is_delta(l)) continue;
      saved& s = layers.emplace_back(
          saved{&M.layers[l], scratch_f32(c.conv_state_n()), scratch_f32(c.ssm_state_n())});
      copy_(sp(s.L->conv_state), sp(s.conv), c.conv_state_n());
      copy_(sp(s.L->ssm_state), sp(s.ssm), c.ssm_state_n());
    }
  }
  void restore(const Config& c) const {
    for (const saved& s : layers) {
      copy_(sp(s.conv), sp(s.L->conv_state), c.conv_state_n());
      copy_(sp(s.ssm), sp(s.L->ssm_state), c.ssm_state_n());
    }
  }
};

// The attention layers' KV caches, empty, in the form f16 says (Model::kv_f16)
// (a layer that reads another's has none):
// build()'s, and a caller's that runs the same weights with an f32 cache (check_qwen's
// reference forward, on a quantized file's own blocks).
inline void init_kv_caches(Model& M, bool f16) {
  const Config& c = M.c;
  M.kv_f16 = f16;
  for (int64_t l = 0; l < c.NL; l++)
    if (!c.is_delta(l) && c.has_kv(l))
      M.layers[l].cache.init(c.nkv(l), c.MAXC, c.hd(l), f16, c.is_local(l) ? c.SWA : 0);
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

// The batched prefill's f16 forms, in one place (prefill_chunk_ takes them, the
// checks ask): its activations are f16 (a16) where f16_acc asks for f16 and the
// backend has the producers that write them; its GEMMs take f16's precision and
// range under f16_acc (prefill_gemm_f16): a bf16 weight's sums f16 on either
// activation (gemm_bf16_nt_f16acc reads f32 too), a block-quantized one's sums
// f16 on f16 activations, or its operands f16 where the backend has that form
// over f32 ones (gemm_quant_f16). The exact form takes none.
inline bool prefill_a16(const Model& M, bool exact) {
  return M.f16_acc && !exact && gpu::f16_activations;
}
inline bool prefill_gemm_f16(const Model& M, bool exact) {
  return M.f16_acc && !exact && (!M.quant || gpu::f16_activations || gpu::has_gemm_quant_f16);
}
// MMQ (block-quantized weights' int8 GEMM) quantizes the f16 activation a16
// makes, so it runs only where that is made.
inline bool prefill_mmq(const Model& M, bool exact) { return M.mmq && prefill_a16(M, exact); }

// ---- M9 batched prefill: run T prompt tokens through the layer stack at once.
//
// Decode is a GEMV regime (one token, weights read once per token, bandwidth-
// bound). A prompt is not: T tokens share the same weights, so every projection
// becomes a GEMM whose cost per token collapses (~25x measured across the four
// Qwen shapes in bench_qwen_prefill), and attention becomes one tiled causal
// pass instead of T growing-ctx decode calls (~45x). This is the same forward as
// run_layers_, one row per token: fused QKV -> split to head-major -> rope ->
// causal attn over the cache -> merge back -> o proj -> fused gate|up -> GLU
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
// `exact` runs the chunk at full precision (the GEMMs' operands and sums f32,
// and the attention's operands f32), as the retry after an f16 overflow does.
//
// The layers past the KV-sharing line (Config::KV_SHARED) write no cache, so a
// prompt needs them for its last row alone (llama.cpp's prune_shared): a chunk
// that is not the prompt's `last` stops before them, and the last one goes on
// with its last row only. Returns the row of P.h holding the chunk's last final
// norm, -1 where it stopped.
inline int64_t prefill_chunk_(Model& M, int64_t T, bool exact = false, bool last = true) {
  const Config& c = M.c;
  PrefillScratch& P = M.pscratch;
  // Absolute position of this chunk's first token — the caches are all at the
  // same place, and kv_cache::prefill appends there, so rope reads it from the
  // same source rather than taking it as a parameter that could disagree.
  int64_t pos0 = seq_cache(M).pos;
  // The f16-sum GEMMs read their activation as f16 where prefill_a16 says; `a16s`
  // is that copy of `a`. One row (a one-token chunk, or the last row past the
  // KV-sharing line) is a decode step's product: the decode's GEMVs, which read the
  // weights once at their bandwidth where a tile GEMM wastes all its rows but one
  // (E2B's 20 shared layers took a quarter of a Metal pp512), and read f32.
  bool a16 = false, mmq = false;
  auto set_forms = [&] {
    a16 = T > 1 && prefill_a16(M, exact);
    mmq = T > 1 && prefill_mmq(M, exact);
  };
  set_forms();
  const bool gemm_f16 = prefill_gemm_f16(M, exact);
  // A GEMM a run of one format, each into its columns of y. The bf16 GEMMs
  // write y densely, so a bf16 run is the whole projection, as every bf16 build
  // loads it (load_wmat).
  auto gemm = [&](gpu::span a, gpu::span a16s, const wmat& W, gpu::span y, int64_t n,
                  int64_t k) {
    if (T == 1)
      return gemv_m(W, a, y, exact ? gpu::act_form::f32 : M.gemv_form, sp(M.scratch.act));
    for (const auto& p : W.parts) {
      const gpu::span w = W.at(p), yp = y.at(p.row_off * 4);
      if (p.type != bf16_type) {
        const bool ran =
            a16        ? gpu::gemm_quant(p.type, true, a16s, w, yp, T, p.rows, k, n, mmq)
            : gemm_f16 ? gpu::gemm_quant_f16(p.type, a, w, yp, T, p.rows, k, n)
                       : gpu::gemm_quant(p.type, false, a, w, yp, T, p.rows, k, n);
        if (!ran)
          throw std::logic_error("qwen: the block-quantized prefill GEMM refused its operands");
        continue;
      }
      if (p.rows != n) throw std::logic_error("qwen: a bf16 run narrower than its projection");
      // No f32 copy of an f16 activation exists to fall back on, so a refusal
      // (misaligned buffer, odd n, k % 8) is an error, not a slower path.
      if (a16) {
        if (!gpu::gemm_bf16_nt_f16acc_a16(a16s, w, y, T, n, k))
          throw std::logic_error("qwen: the f16 prefill GEMM refused its operands");
      } else if (gemm_f16) gpu::gemm_bf16_nt_f16acc(a, w, y, T, n, k);
      else gpu::gemm_bf16_nt(a, w, y, T, n, k);
    }
  };
  // The norm that feeds a GEMM, and the residual-adding one (the branch delta
  // normalized by wd first where the family has it, as the decode's): with a16
  // each leaves the f16 copy the GEMM reads, and the f32 one only where
  // something else reads it (`keep`; the layer's own norm output is read by
  // nothing).
  auto norm = [&](gpu::span in, gpu::span w, gpu::span o, gpu::span o16, bool keep) {
    if (a16) gpu::rmsnorm_h16(in, w, keep ? o : gpu::span{}, o16, c.NE, c.EPS, T);
    else gpu::rmsnorm(in, w, o, c.NE, c.EPS, T);
  };
  auto norm_res = [&](gpu::span xin, gpu::span delta, gpu::span wd, gpu::span w, gpu::span xo,
                      gpu::span o, gpu::span o16, bool keep) {
    if (a16)
      gpu::rmsnorm_res_h16(xin, delta, w, xo, keep ? o : gpu::span{}, o16, c.NE, c.EPS, T, wd);
    else gpu::rmsnorm_res(xin, delta, w, xo, o, c.NE, c.EPS, T, wd);
  };
  // A strided [m, n] block of rows `ld` apart (m = T: the chunk's), copied contiguous.
  auto rows_out = [&](gpu::span src, int64_t ld, gpu::span dst, int64_t n, int64_t m) {
    gpu::generic::copy_2d(src, ld, 1, sp(P.zero), dst, m, n);
  };
  gpu::span x = sp(P.emb), slots = sp(P.ple);
  if (c.PLE) ple_inputs_(M, x, T, sp(P.ple_tok), slots);
  norm(x, sp(M.layers[0].an), sp(P.h), sp(P.h16), false);
  for (int64_t l = 0; l < c.NL; l++) {
    Layer& L = M.layers[l];
    if (c.KV_SHARED && l == c.NL - c.KV_SHARED) {
      if (!last) return -1;
      if (T > 1) {
        // The last row on its own: its residual as a view (nothing writes that
        // buffer before the layer's residual add has read it), its per-layer
        // inputs into one-row slots (the decode's, which hold nothing across
        // steps), its input norm again.
        x = at_f32(x, (T - 1) * c.NE);
        if (c.PLE) {
          rows_out(at_f32(slots, ((l + 1) * T + T - 1) * c.PLE), T * c.PLE,
                   at_f32(sp(M.scratch.ple), (l + 1) * c.PLE), c.PLE, c.NL - l);
          slots = sp(M.scratch.ple);
        }
        pos0 += T - 1;
        T = 1;
        set_forms();
        norm(x, sp(L.an), sp(P.h), sp(P.h16), false);
      }
    }
    const gpu::span ro = sp(P.res[l & 1]);
    if (c.is_delta(l)) {
      // Gated DeltaNet over the chunk: the projection as one GEMM, then the
      // conv and the rule walk the tokens in order (each one launch). The
      // rule's output lands in o, and the gated norm silu(z) * rmsnorm(o) reads
      // z where the projection left it (where the backend has no such pass, z is
      // copied out beside o and the norm and a swiglu over [z | o] make it).
      const int64_t C = c.ssm_conv_ch(), V = c.ssm_v(), PJ = c.ssm_proj();
      const gpu::span z = sp(P.zo), o = at_f32(z, T * V);
      gemm(sp(P.h), sp(P.h16), L.wssm_row, sp(P.qkv), PJ, c.NE);
      gpu::ssm_conv(sp(P.qkv), PJ, sp(L.conv_state), sp(L.wconv), sp(P.conv), C,
                    c.SCONV, T);
      gpu::gated_delta(sp(P.conv), at_f32(sp(P.qkv), C), PJ, sp(L.adt),
                       sp(L.ssm_state), o, c.SH, c.SDK, c.SDV, T, c.L2EPS, c.SK);
      const gpu::span zp = at_f32(sp(P.qkv), C + 2 * c.SH);
      if (!gpu::gated_rmsnorm(o, zp, PJ, sp(L.sn), a16 ? sp(P.at16) : sp(P.at), a16, T, c.SH, c.SDV,
                              c.EPS)) {
        gpu::rmsnorm(o, sp(L.sn), o, c.SDV, c.EPS, T * c.SH);
        rows_out(zp, PJ, z, V, T);
        if (a16) gpu::glu_h16(gpu::glu_act::silu, z, sp(P.at16), T * V, 1);
        else gpu::swiglu(z, sp(P.at), T * V);
      }
      gemm(sp(P.at), sp(P.at16), L.wsout_row, ro, c.NE, V);
    } else {
      // A layer that reads another's KV (Config::kv_src) projects q alone and
      // attends over that layer's cache, which has this chunk already.
      const int64_t hd = c.hd(l), Q = c.NH * hd, AP = c.attn_proj(l), nkv = c.nkv(l);
      const bool own = c.has_kv(l);
      tl::kv_cache& kv = M.layers[c.kv_src(l)].cache;
      gemm(sp(P.h), sp(P.h16), L.wqkv_row, sp(P.qkv), AP, c.NE);
      // split_heads turns the [T, q|k|v] output into head-major [H, T, D] and
      // adds the bias — rope's own fused-bias form only indexes correctly at
      // T == 1, so the bias rides along here instead. One pass over every head;
      // k and v are views into its output.
      gpu::split_heads(sp(P.qkv), sp(L.bqkv), sp(P.qkvh), T, AP, 0, c.qkv_heads(l), hd);
      const gpu::span qh = sp(P.qkvh);
      const gpu::span kh = at_f32(qh, c.NH * T * hd);
      const gpu::span vh = at_f32(qh, (c.NH + nkv) * T * hd);
      // Qwen3: every q / k head row (hd wide, one per h*T + t) is RMS-normalized
      // in place, before rope. v is untouched, but for Gemma 4's weightless norm.
      // [H,T,D] flattened: row r = h*T + t, so rope's `pos + r % T` is pos0 + t.
      const gpu::rope_freq rf = rope_of(M, l);
      if (c.qk_norm) {
        gpu::rmsnorm_rope(qh, sp(L.qn), c.NH * T, T, hd, pos0, rf, c.EPS);
        if (own) gpu::rmsnorm_rope(kh, sp(L.kn), nkv * T, T, hd, pos0, rf, c.EPS);
      } else {
        gpu::rope(qh, qh, c.NH * T, T, hd, pos0, rf);
        if (own) gpu::rope(kh, kh, nkv * T, T, hd, pos0, rf);
      }
      if (own && c.v_norm) gpu::rmsnorm(vh, sp(M.ones), vh, hd, c.EPS, nkv * T);
      cache_ok_(own ? kv.prefill(qh, kh, vh, sp(P.ah), T, c.NH, c.scale(l), exact)
                    : kv.attend(qh, sp(P.ah), T, c.NH, c.scale(l), exact));
      // Qwen3.5: times sigmoid of the gate, the tail of each projection row —
      // an f32 product, so only the plain form is rounded where it is made.
      gpu::span ao = sp(P.at);
      if (c.attn_gate) {
        gpu::merge_heads(sp(P.ah), sp(P.at), T, c.NH, hd);
        rows_out(at_f32(sp(P.qkv), c.qkv_heads(l) * hd), AP, sp(P.gt), Q, T);
        gpu::unary(gpu::kop::sigmoid, sp(P.gt), sp(P.gs), T * Q, 1, 0);
        gpu::binary(gpu::kop::mul, ao, sp(P.gs), sp(P.gt), T * Q, 1, 0);
        ao = sp(P.gt);
        if (a16) gpu::cvt_h16(ao, sp(P.at16), T * Q);
      } else if (a16) {
        gpu::merge_heads_h16(sp(P.ah), sp(P.at16), T, c.NH, hd);
      } else {
        gpu::merge_heads(sp(P.ah), sp(P.at), T, c.NH, hd);
      }
      gemm(ao, sp(P.at16), L.wo_row, ro, c.NE, Q);
    }
    norm_res(x, ro, sp(L.pa), sp(L.fn), ro, sp(P.h2), sp(P.h216), false);
    // With MMQ, gate|up of one format is one GEMM whose epilogue is the GLU.
    const int64_t ff = c.ff(l);
    const bool glu = mmq && L.wgu_row.parts.size() == 1 &&
                     gpu::gemm_quant_glu(L.wgu_row.parts[0].type, c.act, sp(P.h216),
                                         L.wgu_row.at(L.wgu_row.parts[0]),
                                         sp(P.mb16), T, ff, c.NE);
    if (!glu) {
      gemm(sp(P.h2), sp(P.h216), L.wgu_row, sp(P.gu), 2 * ff, c.NE);
      if (a16) gpu::glu_h16(c.act, sp(P.gu), sp(P.mb16), ff, T);
      else gpu::glu(c.act, sp(P.gu), sp(P.mb), ff, T);
    }
    gemm(sp(P.mb), sp(P.mb16), L.wd_row, sp(P.md), c.NE, ff);
    const gpu::span nextw = sp((l + 1 < c.NL) ? M.layers[l + 1].an : M.onorm);
    if (has_tail_(M, l)) {  // the logits read the last layer's norm
      layer_tail_(M, L, l, T, ro, sp(P.md), slots, sp(P.ple_gu), sp(P.ple_g));
      norm(ro, nextw, sp(P.h), sp(P.h16), l + 1 == c.NL);
    } else {
      norm_res(ro, sp(P.md), sp(L.pf), nextw, ro, sp(P.h), sp(P.h16), l + 1 == c.NL);
    }
    x = ro;
  }
  // Leaves P.h holding the chunk's final RMSNorm from the row returned on (row 0
  // alone where the layers past the KV-sharing line ran on the last row). Logits are
  // the caller's business: only the LAST chunk's last row is ever read, so
  // running the 272 MB lm_head here would waste a full GEMV per earlier chunk.
  return T - 1;
}

// Widen `n` token embedding rows (whatever float the GGUF stored) into `dst`.
// Both stagers go through it.
inline void gather_embed_rows_(const Model& M, const int* ids, int64_t n,
                               float* dst) {
  const Config& c = M.c;
  for (int64_t i = 0; i < n; i++) {
    float* row = dst + i * c.NE;
    widen_(M.embed, (size_t)ids[i] * c.NE, c.NE, row);
    // Scaled where the family says (Gemma: sqrt(NE)), as gpu::embed_row scales the
    // row it gathers on the device: the same f32 product.
    if (c.EMB_SCALE != 1.0f)
      for (int64_t j = 0; j < c.NE; j++) row[j] *= c.EMB_SCALE;
  }
}

// Config::PLE: `n` tokens' own per-layer rows, widened into `stage` and scaled
// by sqrt(PLE / 2) (ple_inputs_), then uploaded into `dst`. Nothing where the
// family has none.
inline void stage_ple_rows_(const Model& M, const int* ids, int64_t n, std::vector<float>& stage,
                            const storage& dst) {
  const Config& c = M.c;
  if (!c.PLE) return;
  const int64_t NP = c.NL * c.PLE;
  const float k = std::sqrt((float)c.PLE) / std::sqrt(2.0f);
  for (int64_t i = 0; i < n; i++) {
    float* row = stage.data() + i * NP;
    widen_(M.ple_embd, (size_t)ids[i] * NP, NP, row);
    for (int64_t j = 0; j < NP; j++) row[j] *= k;
  }
  gpu::upload(dst.native, stage.data(), n * NP);
}

// Stage `n` prompt embeddings starting at ids[from] into the batched scratch
// (and their per-layer rows).
inline void stage_embed_rows_(Model& M, const std::vector<int>& ids, int64_t from,
                              int64_t n) {
  const Config& c = M.c;
  float* v = M.pscratch.emb_stage.data();
  gather_embed_rows_(M, ids.data() + from, n, v);
  gpu::upload(M.pscratch.emb.native, v, n * c.NE);
  stage_ple_rows_(M, ids.data() + from, n, M.pscratch.ple_stage, M.pscratch.ple_tok);
}

// Whether a prompt of `T` tokens should go through the batched prefill: the
// backend must have the bf16-weight GEMM, the weights must be the row-major
// ones it consumes, the prompt must be long enough to be worth a batched pass,
// and it must fit the cache. One place answers it, so build(), begin() and
// prefill_batched() cannot disagree.
inline bool can_prefill_batched(const Model& M, int64_t T) {
  return gpu::caps::bf16_gemm && M.row && !M.q4_mlp && (!M.quant || gpu::has_gemm_quant) &&
         !M.layers.empty() &&
         T >= PREFILL_MIN && seq_cache(M).pos + T <= M.c.MAXC;
}

// Prefill a whole prompt in chunks and return the greedy next token. Leaves the
// caches at pos_before + ids.size(), so a decoder captured afterwards continues
// from there. Returns -1 if the batched path is unavailable or the prompt does
// not fit the cache.
inline int64_t prefill_batched(Model& M, const std::vector<int>& ids,
                               int64_t chunk = PREFILL_CHUNK, bool exact = false) {
  const Config& c = M.c;
  const int64_t T = (int64_t)ids.size();
  if (!can_prefill_batched(M, T) || !M.pscratch.init(c, chunk)) return -1;
  auto run = [&](bool full) {
    int64_t row = 0;
    for (int64_t s = 0; s < T; s += chunk) {
      const int64_t n = std::min(chunk, T - s);
      stage_embed_rows_(M, ids, s, n);
      row = prefill_chunk_(M, n, full, s + n == T);
    }
    // Only the final row of the final chunk needs logits — one GEMV for the
    // whole prompt rather than one per chunk, read in place as a view of it.
    lm_head(M, at_f32(sp(M.pscratch.h), row * c.NE), sp(M.scratch.logits));
  };
  if (exact) {
    run(true);
    return argmax_logits(M);
  }
  // f16 can overflow (the attention's operands always, the GEMMs' sums or
  // operands under f16_acc), and an inf anywhere reaches every logit as NaN through the next
  // RMSNorm, so a non-finite logit sum is the test. The retry is exact, so it
  // cannot overflow the same way; first the sequence goes back to where it
  // started: the caches' position, and a DeltaNet layer's state, which the
  // prompt advanced.
  const int64_t pos_before = seq_cache(M).pos;
  delta_snapshot before;
  if (c.hybrid()) before.save(M);
  run(false);
  gpu::row_op(gpu::kop::row_sum, sp(M.scratch.logits), sp(M.scratch.logit_sum), 1, c.VOCAB, 1,
              0);
  if (!std::isfinite(host_read(M.scratch.logit_sum)[0])) {
    set_cache_pos(M, pos_before);
    if (c.hybrid()) before.restore(M.c);
    run(true);
    M.f16_retries++;
  }
  return argmax_logits(M);
}

// Whether the file's projections are block-quantized blocks the native GEMVs read (the
// quantized file's own bytes stay on the device); an F16 / BF16 file is not.
inline bool quant_file(const gg::model& m) {
  const auto* t = m.tensor("blk.0.ffn_down.weight");
  return t && gg::is_quant(t->type);
}

// The forms' names, as TL_GEMV_ACT takes them and the tools print them.
inline const char* act_form_name(gpu::act_form f) {
  switch (f) {
    case gpu::act_form::f32: return "f32";
    case gpu::act_form::q8: return "q8";
    case gpu::act_form::h16: return "h16";
  }
  return "?";
}
// The decode GEMVs' activation form: TL_GEMV_ACT names it (f32, q8 or h16; h16 when
// unset: as fast as q8 on large models, a few percent behind on small ones, and far
// closer to f32: Gemma 3 4B's decode KL from the f32 forward 5e-7 against q8's 3e-3,
// docs/llama-cpp-comparison.md), and a form the backend has not got falls back to f32.
// Whether it has it is asked at one Q4_K superblock; a GEMV whose K the form does not
// take (not whole superblocks) reads its activation as f32, call by call (gemv_q).
inline bool gemv_form_available(gpu::act_form f) {
  return f == gpu::act_form::f32 || gpu::act_bytes(f, 256) > 0;
}
inline gpu::act_form gemv_form_env() {
  gpu::act_form f = gpu::act_form::h16;
  if (const char* e = std::getenv("TL_GEMV_ACT")) {
    bool known = std::strcmp(e, act_form_name(gpu::act_form::f32)) == 0;
    f = gpu::act_form::f32;
    for (gpu::act_form x : gpu::prepared_act_forms)
      if (std::strcmp(e, act_form_name(x)) == 0) f = x, known = true;
    if (!known) {
      std::fprintf(stderr, "TL_GEMV_ACT=%s: not a form (%s", e, act_form_name(gpu::act_form::f32));
      for (gpu::act_form x : gpu::prepared_act_forms) std::fprintf(stderr, ", %s", act_form_name(x));
      std::fprintf(stderr, ")\n");
      std::exit(2);
    }
  }
  return gemv_form_available(f) ? f : gpu::act_form::f32;
}

// wdt = the linear-weight storage dtype (f32 exact, or bf16 for the decode
// path). Biases, norms and the embedding stay F32 (elementwise / row-gather).
// q4_mlp / q4_lmhead quantize the large bandwidth-bound gemvs to group-int4 in
// the imperative decode path (bf16 base only). Lossy — greedy diverges from the
// F16 oracle; validated by coherence + divergence, not greedy-exact.
inline Model build(const gg::model& m, tl::dtype wdt = tl::dtype::f32,
                   bool q4_mlp = false, bool q4_lmhead = false,
                   int64_t maxc = 4096, bool oracle = true, bool f16_acc = true,
                   bool quant = false) {
  Config c = config_from_gguf(m, maxc);
  // A backend whose attention reads no window holds the context to it: within the
  // window a local layer attends as a global one does.
  if (c.SWA > 0 && !gpu::caps::attn_window) c.MAXC = std::min(c.MAXC, c.SWA);
  // bf16 weights need the backend's bf16 [K,N] GEMV: every projection off the
  // row path reads it, the lm head included, and so does the array path. It has
  // no composition to fall back to, so a backend without it would leave the
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
  // quant: the projections stay the file's block-quantized blocks and the GEMVs read
  // them as such (the decode is bound by the bytes it reads; this is what lets a
  // 27B model's 16 GB sit on the device). It replaces the bf16 weights, so there
  // is no array-path oracle and no q4 copy.
  const bool nat = quant && row && gpu::has_gemv_quant;
  if (quant && !nat) throw std::runtime_error("qwen: this backend cannot read block-quantized weights natively");
  // The ropes refuse per-pair factors where the backend has none (caps), and the
  // forward does not stop at each rope's return: refuse the model here instead of
  // running its global layers unturned.
  if (c.rope_factors && !gpu::caps::rope_factors)
    throw std::runtime_error("qwen: this backend's RoPE takes no per-pair factors (rope_freqs, YaRN)");
  const bool oracle_w = oracle && c.array_path() && !nat;
  const bool col_wo = oracle_w || (!row && !nat);
  const tl::dtype f32 = tl::dtype::f32;
  // A (fused) projection in the decode GEMV's layout: row-major on the device
  // where the weights are (row; the file's quantized blocks with nat, else
  // bf16), else column-major [K,N] for split-K. The other one stays empty.
  auto dev = [&](const std::vector<wpart>& parts, int64_t in) {
    return row ? load_wmat(m, parts, in, nat) : wmat{};
  };
  auto col = [&](const std::vector<wpart>& parts, int64_t in) {
    return row ? array{} : load_w_T_cat(m, parts, in, wdt);
  };
  Model M{
      .c = c,
      .embed = need(m, "token_embd.weight"),
      .outw_row = dev({{c.output_name(), c.VOCAB}}, c.NE),
      .outwT = (!row || oracle_w) ? load_w_T(m, c.output_name(), c.NE, c.VOCAB, wdt) : array{},
      .onorm = load_f32(m, "output_norm.weight", c.NE),
      .rope_freqs = !c.rope_factors ? array{}
                    : c.yarn        ? array::from(c.yarn->divisors(c.ROPE_BASE, c.ROT), {1, c.ROT / 2})
                                    : load_f32(m, "rope_freqs.weight", c.ROT / 2),
      // q4 lm_head (imperative): quantize the f32 [K,VOCAB], keep only the q4.
      .outwT_q4 =
          q4lm ? load_w_T(m, c.output_name(), c.NE, c.VOCAB, f32).to_q4() : array{},
      .wdt = wdt,
      .gemv_form = nat ? gemv_form_env() : gpu::act_form::f32,
      .mmq = nat && gpu::has_gemm_quant && !std::getenv("TL_NO_MMQ"),
      .quant = nat,
      .row = row,
      .q4_mlp = q4m,
      .q4_lmhead = q4lm,
      .f16_acc = f16_acc,
      .kv_f16 = wdt == tl::dtype::bf16 && gpu::caps::kv_f16 && !std::getenv("TL_KV_F32")};
  if (c.v_norm) M.ones = array::from(std::vector<float>((size_t)c.HD, 1.0f), {1, c.HD});
  if (c.PLE) {
    const int64_t NP = c.NL * c.PLE;
    M.ple_embd = need(m, "per_layer_token_embd.weight");
    // The projection's rows stay bf16 (the file's own; never block-quantized).
    const std::vector<wpart> pj = {{"per_layer_model_proj.weight", NP}};
    if (row) M.ple_proj_row = load_wmat(m, pj, c.NE, false);
    else M.ple_projT = load_w_T_cat(m, pj, c.NE, wdt);
    std::vector<float> w((size_t)c.PLE);
    widen_(need(m, "per_layer_proj_norm.weight"), 0, w.size(), w.data());
    for (float& v : w) v *= 1.0f / std::sqrt(2.0f);
    M.ple_norm = array::from(std::move(w), {1, c.PLE});
  }
  for (int64_t l = 0; l < c.NL; l++) {
    std::string p = "blk." + std::to_string(l) + ".";
    // Named per field: the ternaries below decide LAYOUT, and a positional list
    // of same-typed arrays is one transposed line away from a silent
    // wrong-weight bug that only shows up as slightly-off logits.
    const std::vector<wpart> gu_parts = layer_gu_parts(c, l);
    // [K,N] weights: the array path's own (the oracle), and wo/wd, which the
    // imperative path reads too where it has no row copies.
    auto W = [&](bool want, const wpart& part, int64_t in) {
      return want ? load_w_T_cat(m, {part}, in, wdt) : array{};
    };
    const std::vector<wpart> down = {{p + "ffn_down.weight", c.NE}};
    // Every layer: the norms and the MLP.
    Layer L{
        .wg = W(oracle_w, gu_parts[0], c.NE),
        .wu = W(oracle_w, gu_parts[1], c.NE),
        .wd = W(col_wo, down[0], c.ff(l)),
        .an = load_f32(m, p + "attn_norm.weight", c.NE),
        .fn = load_f32(m, p + c.ffn_norm, c.NE),
        // gate|up and down: empty when q4m (replaced by wgu_q4 / wd_q4).
        .wgu_row = q4m ? wmat{} : dev(gu_parts, c.NE),
        .wd_row = q4m ? wmat{} : dev(down, c.ff(l)),
        .wgu = col(gu_parts, c.NE),
        // q4 MLP (imperative, when q4m): quantize the f32 [K,N] fused/down.
        .wgu_q4 = q4m ? load_w_T_cat(m, gu_parts, c.NE, f32).to_q4() : array{},
        .wd_q4 = q4m ? load_w_T_cat(m, down, c.ff(l), f32).to_q4() : array{}};
    if (c.post_norms) {
      L.pa = load_f32(m, p + c.post_attn_norm, c.NE);
      L.pf = load_f32(m, p + c.post_ffn_norm, c.NE);
    }
    if (const auto* sc = m.tensor(p + "layer_output_scale.weight")) widen_(sc, 0, 1, &L.out_scale);
    if (c.PLE) {
      L.ple_gate = load_f32(m, p + "inp_gate.weight", c.PLE * c.NE);
      L.ple_proj = load_f32(m, p + "proj.weight", c.NE * c.PLE);
      L.ple_post = load_f32(m, p + "post_norm.weight", c.NE);
    }
    if (c.is_delta(l)) {
      // Gated DeltaNet. The fused projection's order puts z right before the
      // slot the rule writes its output to (Config::ssm_proj).
      const int64_t C = c.ssm_conv_ch(), V = c.ssm_v();
      const std::vector<wpart> ssm_parts = {{p + "attn_qkv.weight", C},
                                            {p + "ssm_alpha.weight", c.SH},
                                            {p + "ssm_beta.weight", c.SH},
                                            {p + "attn_gate.weight", V}};
      const std::vector<wpart> out = {{p + "ssm_out.weight", c.NE}};
      L.wssm_row = dev(ssm_parts, c.NE);
      L.wsout_row = dev(out, V);
      L.wssm = col(ssm_parts, c.NE);
      L.wsout = col(out, V);
      // The conv taps are [C][SCONV] in the GGUF; the op reads them
      // tap-major, so a transpose here.
      std::vector<float> cw((size_t)c.SCONV * C), taps((size_t)c.SCONV * C);
      widen_(need(m, p + "ssm_conv1d.weight"), 0, cw.size(), cw.data());
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t k = 0; k < c.SCONV; k++)
          taps[(size_t)k * C + ch] = cw[(size_t)ch * c.SCONV + k];
      L.wconv = array::from(std::move(taps), {c.SCONV, C});
      L.adt = load_f32_cat(m, {{p + "ssm_a", c.SH}, {p + "ssm_dt.bias", c.SH}});
      L.sn = load_f32(m, p + "ssm_norm.weight", c.SDV);
      L.conv_state = scratch_f32(c.conv_state_n());
      L.ssm_state = scratch_f32(c.ssm_state_n());
      zero_state_(c, L);
    } else {
      // Attention. Qwen3.5's q projection interleaves q and its output gate per
      // head; the fused load takes q's rows first and the gate's last, so the
      // output is q | k | v | gate and every slice downstream is contiguous.
      // A layer that reads another's KV (Config::kv_src) has q's weights only.
      const int64_t hd = c.hd(l), Q = c.NH * hd, KV = c.nkv(l) * hd;
      const bool own = c.has_kv(l);
      const std::vector<wpart> qkv_parts = layer_qkv_parts(c, l);
      auto bias = [&](const char* n, int64_t len) {
        return c.qkv_bias ? load_f32(m, p + n, len) : array{};
      };
      auto qk_norm = [&](const char* n) {
        return c.qk_norm ? load_f32(m, p + n, hd) : array{};
      };
      L.wq = W(oracle_w, qkv_parts[0], c.NE);
      L.bq = bias("attn_q.bias", Q);
      L.qn = qk_norm("attn_q_norm.weight");
      if (own) {
        L.wk = W(oracle_w, qkv_parts[1], c.NE);
        L.bk = bias("attn_k.bias", KV);
        L.wv = W(oracle_w, qkv_parts[2], c.NE);
        L.bv = bias("attn_v.bias", KV);
        L.kn = qk_norm("attn_k_norm.weight");
      }
      const std::vector<wpart> wo_parts = {{p + "attn_output.weight", c.NE}};
      L.wo = W(col_wo, wo_parts[0], Q);
      L.wqkv_row = dev(qkv_parts, c.NE);
      L.wqkv = col(qkv_parts, c.NE);
      // bqkv = [bq|bk|bv], matching q|k|v's column order (batched path).
      if (row && c.qkv_bias)
        L.bqkv = load_f32_cat(m, {{p + "attn_q.bias", Q},
                                  {p + "attn_k.bias", KV},
                                  {p + "attn_v.bias", KV}});
      L.wo_row = dev(wo_parts, Q);
    }
    M.layers.push_back(std::move(L));
  }
  init_kv_caches(M, M.kv_f16);
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
        "qwen: the array path needs build(..., oracle=true) and a plain decoder (array_path())");
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
    cache_ok_(L.cache.append(sp(k), sp(v)));
    array a_out = array::empty({c.NH, c.HD});
    cache_ok_(L.cache.attn(sp(q), sp(a_out), c.NH, c.scale(l)));
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

// The largest relative gap between two logit vectors over the reference's top
// 5: the logits a greedy or a top-k sampler reads. (Over the whole vocabulary
// the gap is larger in the tail, where nothing samples: between the batched
// prefill and the token-by-token steps in their f32 forms, up to ~1.4e-2.)
inline double top5_maxrel(const float* got, const float* ref, int64_t n) {
  std::vector<int64_t> idx((size_t)n);
  for (int64_t i = 0; i < n; i++) idx[(size_t)i] = i;
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int64_t a, int64_t b) { return ref[a] > ref[b]; });
  double mr = 0;
  for (int i = 0; i < 5; i++) {
    const double r = ref[idx[i]];
    mr = tl::max_err(mr, std::fabs(got[idx[i]] - r) / (1.0 + std::fabs(r)));
  }
  return mr;
}

// KL(ref || got) of the two logit vectors' distributions, in nats: how far a
// path's predictions move from a reference's, as llama.cpp's perplexity
// --kl-divergence measures a quantization.
inline double kl_div(const std::vector<float>& ref, const std::vector<float>& got) {
  auto log_softmax = [](const std::vector<float>& x) {
    const double m = *std::max_element(x.begin(), x.end());
    double z = 0;
    for (float v : x) z += std::exp(v - m);
    const double lz = std::log(z);
    std::vector<double> r(x.size());
    for (size_t i = 0; i < x.size(); i++) r[i] = x[i] - m - lz;
    return r;
  };
  const std::vector<double> lp = log_softmax(ref), lq = log_softmax(got);
  double kl = 0;
  for (size_t i = 0; i < lp.size(); i++) kl += std::exp(lp[i]) * (lp[i] - lq[i]);
  return kl;
}

inline int64_t argmax(const std::vector<float>& v) {
  int64_t bi = 0;
  for (int64_t i = 1; i < (int64_t)v.size(); i++)
    if (v[i] > v[bi]) bi = i;
  return bi;
}

// How far a form that rounds the activation to 8 bits (the decode's q8 GEMVs, the
// prefill's MMQ) may move a logit, as |d| / (1 + max |logit|): the bound check_quant
// holds those forms to, and the closeness of two logits a near tie is (compare_logits).
constexpr double BOUND_Q8 = 3e-1;

// One set of logits against a reference's: the worst |d| / (1 + max |logit|), and
// whether the top token agrees, or differs only where the reference's top two were
// closer than `bound` lets a logit move (a near tie, which the forms' rounding may
// turn either way).
struct logits_vs {
  double err;
  bool same_top, near_tie;
};
inline logits_vs compare_logits(const std::vector<float>& want, const std::vector<float>& got,
                                double bound) {
  double mx = 0, ref = 0;
  for (size_t i = 0; i < want.size(); i++) {
    mx = tl::max_err(mx, (double)std::fabs(want[i] - got[i]));
    ref = std::max(ref, (double)std::fabs(want[i]));
  }
  const int64_t ta = argmax(want), tb = argmax(got);
  return {mx / (1.0 + ref), ta == tb, ta != tb && want[ta] - want[tb] < bound * (1.0 + ref)};
}

// How far a form that rounds the activations may move a model's distribution from
// the f32 forward's, as KL in nats (int8 activations move Gemma 3 4B by 1.4e-2; its
// GLU epilogue run as SiLU, 3.7), and the rule the checks hold such a form to with
// it: the same top token, or a near tie turned over (compare_logits), and KL within.
constexpr double KL_BOUND = 0.3;
inline bool close_by_kl(const logits_vs& c, double kl) {
  return (c.same_top || c.near_tie) && kl < KL_BOUND;
}

// q4 decode GEMV: y(N) = a(1,K) @ dequant(Wq). Wq is a q4 array (logical [K,N],
// storage [packed [N][K/2] | scales [N][K/32]]); the scales are a view into the
// same buffer (they ride along the base upload — the array path's split too).
inline bool gemv_q4_w(const array& Wq, gpu::span a, gpu::span y) {
  const int64_t K = Wq.shape()[0], N = Wq.shape()[1];
  const gpu::span qw = sp(Wq);
  return gpu::gemv_q4(a, qw, qw.at(N * K / 2), y, N, K, tl::kQ4Group);
}

// Gather the embedding row for token `id` into the staged step input S.embed
// (host gather + blocking upload; marks it device-current), and its per-layer
// rows where the family has them (run_layers_ reads both): every step's input
// is staged here. Kept out of a captured region since it involves a host
// gather + blocking copy.
inline void stage_embed(Model& M, int64_t id) {
  const Config& c = M.c;
  const int one = (int)id;
  gather_embed_rows_(M, &one, 1, M.scratch.embed_stage.data());
  gpu::upload(M.scratch.embed.native, M.scratch.embed_stage.data(), c.NE);
  stage_ple_rows_(M, &one, 1, M.scratch.ple_stage, M.scratch.ple_tok);
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
  // Decode GEMV (gemv_rw). The y layout is identical in either layout
  // (contiguous [n]), so the fused-output slices below are unchanged.
  auto gv = [&](const wmat& R, const array& W, gpu::span a, gpu::span y, int64_t n, int64_t k) {
    gemv_rw(R, W, a, y, n, k, M.gemv_form, sp(S.act));
  };
  gpu::span x = x0;
  if (c.PLE) ple_inputs_(M, x, 1, sp(S.ple_tok), sp(S.ple));
  gpu::rmsnorm(x, sp(M.layers[0].an), sp(S.hb), c.NE, c.EPS);
  for (int64_t l = 0; l < c.NL; l++) {
    Layer& L = M.layers[l];
    const gpu::span ro = sp(S.res[l & 1]);  // res_out (x1 then x2), ping-pong
    if (c.is_delta(l)) {
      // Gated DeltaNet: one GEMV -> q|k|v | alpha | beta | z in S.qkvb, the conv
      // over q|k|v, the rule (its output lands right after z), the gated norm
      // silu(z) * rmsnorm(o) (one pass, or the norm and a swiglu over [z | o]), then
      // the out projection.
      const int64_t C = c.ssm_conv_ch(), V = c.ssm_v();
      const gpu::span pj = sp(S.qkvb), o = at_f32(pj, c.ssm_proj());
      gv(L.wssm_row, L.wssm, sp(S.hb), pj, c.ssm_proj(), c.NE);
      gpu::ssm_conv(pj, C, sp(L.conv_state), sp(L.wconv), sp(S.convb), C, c.SCONV);
      gpu::gated_delta(sp(S.convb), at_f32(pj, C), 0, sp(L.adt), sp(L.ssm_state), o,
                       c.SH, c.SDK, c.SDV, 1, c.L2EPS, c.SK);
      if (!gpu::gated_rmsnorm(o, at_f32(pj, C + 2 * c.SH), V, sp(L.sn), sp(S.ab), false, 1, c.SH,
                              c.SDV, c.EPS)) {
        gpu::rmsnorm(o, sp(L.sn), o, c.SDV, c.EPS, c.SH);
        gpu::swiglu(at_f32(pj, C + 2 * c.SH), sp(S.ab), V);
      }
      gv(L.wsout_row, L.wsout, sp(S.ab), ro, c.NE, V);  // ro = mixer @ ssm_out
    } else {
      // Fused QKV: one GEMV -> [q(NH*hd) | k(nkv*hd) | v(nkv*hd)] in S.qkvb (then
      // Qwen3.5's gate), and q, k and v are views of it: rope q & k in place,
      // bias-add v. Same per-column split-K as the separate wq/wk/wv GEMVs (bx=1,
      // chunk=32), so bit-identical per column. A layer that reads another's KV
      // (Config::kv_src) projects q alone and attends over that layer's cache,
      // which has this position already.
      const int64_t hd = c.hd(l), Q = c.NH * hd, nkv = c.nkv(l);
      const bool own = c.has_kv(l);
      tl::kv_cache& kv = M.layers[c.kv_src(l)].cache;
      gv(L.wqkv_row, L.wqkv, sp(S.hb), sp(S.qkvb), c.attn_proj(l), c.NE);
      const gpu::span qp = sp(S.qkvb);
      const gpu::span kp = at_f32(qp, Q);
      const gpu::span vp = at_f32(qp, (c.NH + nkv) * hd);
      // Qwen3: RMSNorm every q / k head (one row of hd each) in place, before rope;
      // Gemma 4's v too, without a weight.
      if (c.qk_norm) {
        gpu::rmsnorm(qp, sp(L.qn), qp, hd, c.EPS, c.NH);
        if (own) gpu::rmsnorm(kp, sp(L.kn), kp, hd, c.EPS, nkv);
      }
      if (own && c.v_norm) gpu::rmsnorm(vp, sp(M.ones), vp, hd, c.EPS, nkv);
      const gpu::rope_freq rf = rope_of(M, l);
      if (cap) {
        gpu::rope_dpos(qp, qp, c.NH, 1, hd, d_pos, rf, sp(L.bq));
        if (own) gpu::rope_dpos(kp, kp, nkv, 1, hd, d_pos, rf, sp(L.bk));
      } else {
        gpu::rope(qp, qp, c.NH, 1, hd, pos, rf, sp(L.bq));
        if (own) gpu::rope(kp, kp, nkv, 1, hd, pos, rf, sp(L.bk));
      }
      if (c.qkv_bias) gpu::binary(gpu::kop::add, vp, sp(L.bv), vp, nkv * hd, 1, 0);
      cache_ok_(cap ? (!own || kv.append_dpos(kp, vp, d_pos)) &&
                          kv.attn_dpos(qp, sp(S.ab), c.NH, d_pos, c.scale(l))
                    : (!own || kv.append(kp, vp)) && kv.attn(qp, sp(S.ab), c.NH, c.scale(l)));
      // Qwen3.5: the attention output times sigmoid of its gate (q|k|v's tail).
      gpu::span ao = sp(S.ab);
      if (c.attn_gate) {
        const gpu::span gate = at_f32(qp, c.qkv_heads(l) * hd);
        gpu::unary(gpu::kop::sigmoid, gate, sp(S.gateb), Q, 1, 0);
        gpu::binary(gpu::kop::mul, ao, sp(S.gateb), sp(S.gateb), Q, 1, 0);
        ao = sp(S.gateb);
      }
      gv(L.wo_row, L.wo, ao, ro, c.NE, Q);  // ro = attn @ wo
    }
    // x1 = x + (attn@wo) (Gemma 3: normalized by pa first); h2 = rmsnorm(x1, fn) — fused.
    gpu::rmsnorm_res(x, ro, sp(L.fn), ro, sp(S.h2b), c.NE, c.EPS, 1, sp(L.pa));
    // Fused gate|up: one GEMV -> [gate(ff) | up(ff)] in S.gub; the GLU reads both.
    const int64_t ff = c.ff(l);
    if (M.q4_mlp) gemv_q4_w(L.wgu_q4, sp(S.h2b), sp(S.gub));  // [NE, 2*FF]
    else gv(L.wgu_row, L.wgu, sp(S.h2b), sp(S.gub), 2 * ff, c.NE);
    gpu::glu(c.act, sp(S.gub), sp(S.mb), ff);
    if (M.q4_mlp) gemv_q4_w(L.wd_q4, sp(S.mb), sp(S.mdb));  // [FF, NE]
    else gv(L.wd_row, L.wd, sp(S.mb), sp(S.mdb), c.NE, ff);
    // x2 = x1 + mlp (Gemma 3: normalized by pf first); next input norm = rmsnorm(x2,
    // next an | final onorm) — fused, unless the layer ends past the add (layer_tail_).
    const gpu::span nextw = sp((l + 1 < c.NL) ? M.layers[l + 1].an : M.onorm);
    if (has_tail_(M, l)) {
      layer_tail_(M, L, l, 1, ro, sp(S.mdb), sp(S.ple), {}, sp(S.ple_g));
      gpu::rmsnorm(ro, nextw, sp(S.hb), c.NE, c.EPS);
    } else {
      gpu::rmsnorm_res(ro, sp(S.mdb), nextw, ro, sp(S.hb), c.NE, c.EPS, 1, sp(L.pf));
    }
    x = ro;
  }
  // hb now holds the final RMSNorm output (folded into the last layer's seam).
  const gpu::span logits = logits_out ? logits_out : sp(S.logits);
  if (M.q4_lmhead) {
    gemv_q4_w(M.outwT_q4, sp(S.hb), logits);  // [NE, VOCAB]
    softcap_(M, logits);
  } else {
    lm_head(M, sp(S.hb), logits);
  }
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
  stage_embed(M, id);
  run_layers_(M, sp(M.scratch.embed), pos);
  return argmax_logits(M);
}

// Run the imperative forward and return the logits [VOCAB] on the host (device
// mirror of scratch.logits). For the q4-vs-bf16 divergence measurement — the
// quantization-quality analogue of step_imperative's greedy token.
inline const float* imperative_logits(Model& M, int64_t id, int64_t pos) {
  stage_embed(M, id);
  run_layers_(M, sp(M.scratch.embed), pos);
  return host_read(M.scratch.logits);
}

// Where a step can gather its input on the device (gpu::embed_row), from the
// token the step before left there: the embedding table, when the model keeps
// it on the device as its tied lm_head (outw_row's one run, bf16 or blocks, or
// the [NE, VOCAB] outwT). type -1 where it does not, as for a model with its own
// output.weight. The table is in the build's weight dtype, so a bf16 build of
// an F16 file feeds the embedding rounded to bf16, as its lm_head reads it,
// where stage_embed widens the file's own values (the same for the Qwen F16
// files, converted from bf16).
struct embed_table {
  int type = -1;
  gpu::span W;
  int64_t row_stride = 0, col_stride = 0;
};
inline embed_table device_embed(const Model& M) {
  const Config& c = M.c;
  if (!c.tied_output) return {};
  if (M.outw_row) {
    if (M.outw_row.parts.size() != 1) return {};
    const auto& p = M.outw_row.parts[0];
    return {p.type, M.outw_row.at(p), c.NE, 1};  // a block format ignores the strides
  }
  if (M.outwT.size() == 0) return {};
  return {M.outwT.dt() == tl::dtype::bf16 ? 30 : 0, sp(M.outwT), 1, c.VOCAB};
}

// CUDA-graph-captured decoder (A-min). Captures the device-pos forward
// once (after prefill, at the first decode position) and replays it per token —
// collapsing the ~360 per-token kernel launches into one graph submit. Correct
// across positions because rope/kv-append/attn read the position from the device
// counter d_pos, which a captured tl_incr_u32 advances each replay. The input
// row is gathered on the device when the model allows (tok_in, below), else
// staged by the host; the token is read back outside the graph. Falls
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
//
// Where the model keeps its embedding table on the device and the backend
// gathers from it (tok_in), a step takes its input from the token on the
// device, d_tok, and leaves the token it draws there: feed() and step() put
// the token in, and next() runs a step ahead: it queues the step after the
// token it returns before reading that token back (MLX-LM's async_eval loop;
// SGLang's and TensorRT-LLM's overlap schedulers). Every step is then the same
// launches, so one graph serves all three. Elsewhere the host gathers the
// input row, and next() is step() on the last token.
struct captured_decoder {
  storage d_pos;                      // device u32 position counter (capture)
  // The token a step reads at its start and writes over at its end (an f32):
  // the queue runs in order, and read_later's copy of it sits between one
  // step and the next. Then the read of the step in flight, and the token the
  // next step feeds, which begin(), step() and next() all leave in `last`.
  storage d_tok;
  bool tok_in = false;
  // How each token is drawn (greedy by default; set before begin()), and the
  // tokens returned so far, one stream from the first begin() to destroy():
  // draw n is the n-th token returned, on every path. The draw is on the
  // device where the backend has one (into d_tok, whether or not steps take
  // their input from it), from the counter d_draws, else on the host. Steps
  // whose token is never returned (the prompt fed a token at a time, a step
  // queued ahead and dropped) draw on the device too, so begin() puts `draws`
  // back into d_draws before the draw it returns.
  gpu::sampling sampler;
  storage d_draws;
  uint32_t draws = 0;
  gpu::pending_read in_flight;
  bool flying = false;
  int64_t last = -1;
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
    land_();
    if (M.layers.empty()) return;
    max_ctx = seq_cache(M).max_ctx;  // bound applies to the fallback too
    cur_pos = pos;
    last = first_id;
    if (!sampler.valid()) throw std::invalid_argument("qwen: a sampler setting out of range");
    if (!d_tok.native) d_tok = storage::make(1);
    if (!d_draws.native) d_draws = storage::make(1);
    tok_in = token_input_(M, first_id);
    if (!gpu::graph_available()) return;
    d_pos = storage::make(1);
    if (!d_pos.native) return;
    const gpu::span dp = sp(d_pos);
    put_(M, first_id);
    gpu::upload_u32(d_pos.native, (unsigned)pos);
    // Warm run: real launches, but its logits go to the throwaway sink so a
    // prompt's logits (already in scratch when begin() batched) survive, and
    // a hybrid's DeltaNet state is put back after it (delta_snapshot).
    delta_snapshot before;
    before.save(M);
    run_step_(M, pos, dp, sp(M.scratch.logits_sink));
    before.restore(M.c);
    gpu::flush();
    gpu::upload_u32(d_pos.native, (unsigned)pos);  // reset after warm
    if (!gpu::capture_begin()) return;
    run_step_(M, pos, dp);  // recorded, not executed
    exec = gpu::capture_end();
  }
  // Whether steps take their input from d_tok, which then holds `id`: the
  // table on the device, the backend's argmax_to (and sample_to when
  // sampling), and its reader for the table's format, tried once here (embed_row refuses a format it has none
  // for, and its row is put in again before any step reads it).
  bool token_input_(Model& M, int64_t id) {
    // A family with per-layer embeddings stages their rows on the host (stage_embed).
    if (M.c.PLE || device_embed(M).type < 0 || !gpu::has_argmax_to || !gpu::has_embed_row ||
        (!sampler.greedy() && !gpu::has_sample_to) || M.c.VOCAB > gpu::policy::max_f32_index)
      return false;
    if (!d_tok.native || !d_draws.native) return false;
    upload_token_(id);
    return gather_(M);
  }
  void upload_token_(int64_t id) {
    const float v = static_cast<float>(id);
    gpu::upload(d_tok.native, &v, 1);
  }
  // The input row of the token in d_tok, into S.embed.
  bool gather_(Model& M) {
    const embed_table t = device_embed(M);
    return gpu::embed_row(t.type, t.W, sp(d_tok), sp(M.scratch.embed), M.c.NE, M.c.VOCAB,
                          t.row_stride, t.col_stride, M.c.EMB_SCALE);
  }
  // Token `id` as the next step's input: into d_tok, or its row gathered on
  // the host into S.embed. Both copies wait for the queued work (CUDA's is a
  // blocking null-stream H2D, Metal's upload drains first), so neither
  // overwrites an input a step queued before still reads.
  void put_(Model& M, int64_t id) {
    if (tok_in) upload_token_(id);
    else stage_embed(M, id);
  }
  // One step's launches, the graph's too: the input gathered from d_tok, the
  // layers, and the token drawn over d_tok (from logits_out when given).
  void run_step_(Model& M, int64_t pos, gpu::span dp = {}, gpu::span logits_out = {}) {
    if (tok_in && !gather_(M)) throw std::runtime_error("qwen: a step lost its input row");
    run_layers_(M, sp(M.scratch.embed), pos, dp, logits_out);
    if (tok_in && !draw_to_(logits_out ? logits_out : sp(M.scratch.logits), M.c.VOCAB))
      throw std::runtime_error("qwen: a step lost its token");
  }
  bool draw_to_(gpu::span logits, int64_t n) {
    return gpu::sample_to(logits, n, sampler, sp(d_draws), sp(d_tok));
  }
  // The token drawn from the logits in scratch: over d_tok and read back, or,
  // where the backend cannot draw, on the host.
  int64_t draw_(Model& M) {
    if (sampler.greedy() && !tok_in) return argmax_logits(M);
    if (d_tok.native && d_draws.native && draw_to_(sp(M.scratch.logits), M.c.VOCAB))
      return read_token_();
    return gpu::sample_ref(host_read(M.scratch.logits), M.c.VOCAB, sampler, draws);
  }
  // `draws` into d_draws (a u32, uploaded by its bits).
  void sync_draws_() {
    float bits;
    std::memcpy(&bits, &draws, 4);
    if (d_draws.native) gpu::upload(d_draws.native, &bits, 1);
  }
  // A token returned to the caller: the next draw's number moves on.
  int64_t took_(int64_t t) {
    draws++;
    return last = t;
  }
  int64_t read_token_() {
    gpu::pending_read r;
    if (!gpu::read_later(sp(d_tok), r)) throw std::runtime_error("qwen: a token's read-back failed");
    return finish_token_(r);
  }
  static int64_t finish_token_(gpu::pending_read& r) {
    float v = -1.0f;
    if (!gpu::finish_read(r, &v)) throw std::runtime_error("qwen: a token's read-back failed");
    return static_cast<int64_t>(v);
  }

  // Consume a whole prompt and return the first generated token (drawn as
  // `sampler` says from the LAST prompt token's logits), leaving scratch
  // logits holding that token's row. Two branches, one post-condition:
  //
  //  - BATCHED (`batched`, whenever can_prefill_batched allows): the prompt runs
  //    as GEMMs, then the decode graph is captured at the position the prompt
  //    left the cache at. The prompt itself never enters the graph.
  //  - token-by-token: capture first, then feed() every prompt token but the
  //    last through it (no read-back; what they draw is discarded). ONE graph
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
        // prefill took the greedy token; a sampler draws its own from the
        // prompt's logits, still in scratch (init's warm run used the sink).
        if (!sampler.greedy()) {
          sync_draws_();
          put_(M, next = draw_(M));
        }
        return took_(next);
      }
    }
    // Token-by-token: one capture serves prompt and decode alike.
    init(M, ids[0], seq_cache(M).pos);
    for (size_t i = 0; i + 1 < ids.size(); i++) feed(M, ids[i]);
    if (!sampler.greedy()) sync_draws_();
    int64_t last = step(M, ids.back());  // only the last token's logits are read
    sync_host_pos(M);
    return last;
  }

  // Feed token `id` at the current position, WITHOUT reading a token back.
  // Returns false once the KV cache is full (the bound lives here, not on the
  // caller — kv_append_dpos has no OOB guard, unlike the host append()).
  //
  // Captured mode replays the graph; d_pos auto-advances (captured tl_incr_u32).
  // Skipping the read-back saves its 4-byte D2H *sync* per token (~6%, the
  // replay-only vs replay+argmax gap in bench_qwen_decode) — safe because the
  // next put_ waits for the queued work (see put_). Without a graph this is
  // step_imperative minus the read-back, on the same scratch and host cache pos.
  bool feed(Model& M, int64_t id) {
    if (flying) throw std::logic_error("qwen: feed()/step() after next() has queued a step ahead");
    if (cur_pos >= max_ctx) return false;
    put_(M, id);
    if (exec) gpu::graph_launch(exec);  // replay: append@d_pos, attn, logits, incr
    else run_step_(M, cur_pos);         // host-pos imperative
    cur_pos++;
    return true;
  }
  // One decode step: feed + read the token drawn (-1 when the cache is full):
  // the one the step left in d_tok, or one drawn from the logits on the host.
  int64_t step(Model& M, int64_t id) {
    if (!feed(M, id)) return last = -1;
    return took_(tok_in ? read_token_() : draw_(M));
  }

  // The token after `last`, with the step after it already queued: the host
  // reads one step behind the device. -1 once the cache is full. A step queued
  // ahead has fed the token it read, so when the caller stops, the cache holds
  // the last token returned too, as the next turn would have fed it. After the
  // first next(), keep to next() (feed() and step() would race the step in
  // flight); begin() starts a new turn. The scratch logits are then the step
  // queued ahead's, not the returned token's.
  int64_t next(Model& M) {
    if (!tok_in || !gpu::has_read_later) return step(M, last);
    if (!flying) {
      if (last < 0 || cur_pos >= max_ctx) return -1;
      put_(M, last);
      queue_ahead_(M);
    }
    gpu::pending_read read = std::move(in_flight);
    flying = cur_pos < max_ctx;
    if (flying) queue_ahead_(M);
    return took_(finish_token_(read));
  }
  // The step queued ahead, waited for: its token was fed (the cache holds it)
  // but is never returned.
  void land_() {
    if (!flying) return;
    float drop;
    gpu::finish_read(in_flight, &drop);
    flying = false;
  }
  // One step queued on the token in d_tok, and its own token read later.
  void queue_ahead_(Model& M) {
    if (exec) gpu::graph_launch(exec);
    else run_step_(M, cur_pos);
    if (!gpu::read_later(sp(d_tok), in_flight))
      throw std::runtime_error("qwen: a step queued ahead lost its token");
    cur_pos++;
  }
  // Reconcile the host cache pos with the device counter. Captured replays
  // advance only d_pos, so the host pos would otherwise be stale for anyone
  // switching to a host-pos path (the fallback keeps it current by itself).
  void sync_host_pos(Model& M) const {
    if (exec) set_cache_pos(M, cur_pos);
  }

  bool ok() const { return exec != nullptr; }
  void destroy() {
    land_();
    gpu::flush();
    d_tok = storage{};
    d_draws = storage{};
    draws = 0;
    tok_in = false;
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
// A prompt of n of them, cycled: for the benches that only need its length.
inline std::vector<int> fixture_ids(int64_t n) {
  std::vector<int> ids((size_t)n);
  for (int64_t i = 0; i < n; i++) ids[(size_t)i] = PREFILL_IDS[i % 4];
  return ids;
}
// n token ids spread over the first 30000 of the vocabulary, the same on every
// run: a context no prediction is sure of, where a precision measurement (its
// KL from the f32 forward) sees the whole distribution move.
inline std::vector<int> spread_ids(int64_t n, int64_t vocab) {
  std::vector<int> ids((size_t)n);
  for (int64_t i = 0; i < n; i++)
    ids[(size_t)i] = (int)(100 + (i * 7919 + 13) % (std::min<int64_t>(vocab, 30000) - 100));
  return ids;
}

}  // namespace qwenmodel
