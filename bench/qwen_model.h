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

// F32 1-D tensor (norms, biases are stored F32 in the GGUF) -> array [1, len].
inline array load_f32(const gg::model& m, const std::string& n, int64_t len) {
  const auto* t = need(m, n);
  const auto* raw = reinterpret_cast<const float*>(t->data);
  return array::from(std::vector<float>(raw, raw + len), {1, len});
}

struct Layer {
  array wq, bq, wk, bk, wv, bv, wo, wg, wu, wd, an, fn;
  tl::cuda::kv_cache cache;
};
struct Model {
  const uint16_t* embed_f16;  // token_embd raw F16 [vocab, NE] (row-gather source)
  array outwT;                // logits weight [NE, VOCAB] (transposed output.weight)
  array onorm;                // final norm [1, NE]
  std::vector<Layer> layers;
};

// wdt = the linear-weight storage dtype (f32 exact, or bf16 for the <2GB/decode
// path). Biases, norms and the embedding stay F32 (elementwise / row-gather).
inline Model build(const gg::model& m, tl::dtype wdt = tl::dtype::f32) {
  Model M{reinterpret_cast<const uint16_t*>(need(m, "token_embd.weight")->data),
          load_w_T(m, "output.weight", NE, VOCAB, wdt),
          load_f32(m, "output_norm.weight", NE),
          {}};
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
            {}};
    L.cache.init(NKV, MAXC, HD);
    M.layers.push_back(std::move(L));
  }
  return M;
}

inline array embed_row(const Model& M, int64_t id) {
  std::vector<float> v(NE);
  const uint16_t* r = M.embed_f16 + (size_t)id * NE;
  for (int64_t i = 0; i < NE; i++) v[i] = f16_to_f32(r[i]);
  return array::from(std::move(v), {1, NE});
}

// One decode step at sequence position `pos`. Returns logits [VOCAB]. Optional
// out-params record the layer-0 residual and final-norm for checkpointing.
inline std::vector<float> step(Model& M, int64_t id, int64_t pos,
                               std::vector<float>* l0_out = nullptr,
                               std::vector<float>* fnorm_out = nullptr) {
  array x = embed_row(M, id);
  for (int64_t l = 0; l < NL; l++) {
    Layer& L = M.layers[l];
    array h = array::rmsnorm(x, L.an, EPS);
    array q = array::rope((h.dot(L.wq) + L.bq).reshape({NH, HD}), pos, ROPE_BASE);
    array k = array::rope((h.dot(L.wk) + L.bk).reshape({NKV, HD}), pos, ROPE_BASE);
    array v = h.dot(L.wv) + L.bv;  // [1, NKV*HD]
    q.eval(); k.eval(); v.eval();
    L.cache.append(k.native(), v.native());
    array a_out = array::empty({NH, HD});
    L.cache.attn(q.native(), a_out.native(), NH, SCALE);
    array x1 = x + a_out.reshape({1, NE}).dot(L.wo);
    array h2 = array::rmsnorm(x1, L.fn, EPS);
    array mlp = array::swiglu(h2.dot(L.wg), h2.dot(L.wu)).dot(L.wd);
    array x2 = x1 + mlp;
    x2.eval();
    if (l == 0 && l0_out) { const float* p = x2.raw(); l0_out->assign(p, p + NE); }
    x = x2;
    (void)q; (void)k; (void)v; (void)a_out;
  }
  array xf = array::rmsnorm(x, M.onorm, EPS);
  if (fnorm_out) { xf.eval(); const float* p = xf.raw(); fnorm_out->assign(p, p + NE); }
  array logits = xf.dot(M.outwT);
  logits.eval();
  const float* p = logits.raw();
  return std::vector<float>(p, p + VOCAB);
}

inline int64_t argmax(const std::vector<float>& v) {
  int64_t bi = 0;
  for (int64_t i = 1; i < (int64_t)v.size(); i++)
    if (v[i] > v[bi]) bi = i;
  return bi;
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
