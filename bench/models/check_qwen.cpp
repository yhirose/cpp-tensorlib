// M9 "actually chat" — numeric proof. Load a real Qwen2.5-0.5B-Instruct, Qwen3
// or Qwen3.5 GGUF (F16 or BF16 weights), or Gemma 3 4B (Q4_K_M), and run the
// decoder end to end, verified against a tight numpy reference that reads the
// SAME weights (tools/qwen_oracle.py wrote qwen_oracle_data.h / qwen3_oracle_data.h
// / qwen35_oracle_data.h / gemma3_oracle_data.h; the model's arch picks one). Identical
// weights => the C++ forward must match the reference to ~1e-4 at every checkpoint
// (embedding, layer-0 residual, final norm, top-5 logits) AND reproduce the greedy
// token sequence exactly. Proves weight loading + GGML->our layout transpose +
// Qwen2 structure + the head_dim=64 attention kernels, on whatever backend
// gpu:: resolves to. Tokenizer/sampling are
// separate (see chat_qwen). Model path: argv[1] or ~/models/...fp16.gguf.

#include "gemma3_oracle_data.h"
#include "qwen.h"
#include "qwen35_oracle_data.h"
#include "qwen3_oracle_data.h"
#include "qwen_oracle_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace qm = qwenmodel;

// One model's checkpoints, whichever generated header they came from.
struct Oracle {
  const float *emb_last, *l0_last, *fnorm_last, *top5_vals;
  const int *top5_ids, *greedy30, *prompt_ids;
  size_t n_greedy, n_prompt;
};
#define TL_ORACLE(ns)                                                       \
  Oracle {                                                                  \
    ns::emb_last, ns::l0_last, ns::fnorm_last, ns::top5_vals, ns::top5_ids, \
        ns::greedy30, ns::prompt_ids, sizeof(ns::greedy30) / sizeof(int),   \
        sizeof(ns::prompt_ids) / sizeof(int)                                \
  }

static double maxrel(const std::vector<float>& a, const float* b, int64_t n) {
  double mr = 0;
  for (int64_t i = 0; i < n; i++)
    mr = tl::max_err(mr, std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)b[i])));
  return mr;
}

// The batched prefill's f16 sums (the default where the backend has them),
// against `ref`, an f32 forward's logits for the same prompt: the top-5
// logits to 2e-2 and the same greedy token. Then the safety net: with the
// first layer's FFN norm scaled by 1e5 the f16 sums overflow (the MLP's input
// passes f16's range), and the prompt must be redone once, exactly, from the
// state it started at — an exact prefill of the same model, to the lm_head's
// split-K rounding (its atomic sums land in any order; a missed restore would
// be off by far more).
static bool check_f16_sums(qm::Model& B, const std::vector<int>& ids,
                           const std::vector<float>& ref) {
  auto logits = [&] {
    const float* l = qm::host_read(B.scratch.logits);
    return std::vector<float>(l, l + B.c.VOCAB);
  };
  const bool was = B.f16_acc;
  B.f16_acc = true;
  qm::reset_cache(B);
  const int64_t tok = qm::prefill_batched(B, ids);
  const double t5 = qm::top5_maxrel(logits().data(), ref.data(), B.c.VOCAB);
  const bool close = t5 < 2e-2 && tok == qm::argmax(ref);
  std::printf("\nf16 sums: top-5 logits maxrel %.2e vs f32, token %lld %s\n", t5, (long long)tok,
              close ? "OK" : "FAIL");
  const tl::array fn = B.layers[0].fn;
  B.layers[0].fn = (fn * 1e5f).clone();
  B.layers[0].fn.eval();
  const int64_t retries = B.f16_retries;
  qm::reset_cache(B);
  qm::prefill_batched(B, ids);
  const std::vector<float> redone = logits();
  qm::reset_cache(B);
  qm::prefill_batched(B, ids, qm::PREFILL_CHUNK, /*exact=*/true);
  const std::vector<float> f32 = logits();
  B.layers[0].fn = fn;
  B.f16_acc = was;
  // Both finite first: a NaN compares unequal to nothing, so it would pass
  // the maxrel below.
  bool finite = true;
  for (size_t i = 0; i < f32.size(); i++) finite &= std::isfinite(f32[i]) && std::isfinite(redone[i]);
  const double rmr = maxrel(redone, f32.data(), B.c.VOCAB);
  const bool net = B.f16_retries == retries + 1 && finite && rmr < 1e-5;
  std::printf("f16 overflow: %lld retry, logits maxrel %.2e vs the exact prefill %s\n",
              (long long)(B.f16_retries - retries), rmr, net ? "OK" : "FAIL");
  return close && net;
}

// A hot sampler (temperature 2 over the top 64, so each draw turns on its
// random number even where the model is sure), over two turns on one
// decoder: each token step() returns is the host definition's for the logits
// it left, draw n the n-th token the decoder returned (the first, begin()'s,
// is draw 0); and where steps run ahead, next() returns the same tokens
// (the step() run feeds its last token before the second turn, as next()'s
// step queued ahead leaves it in the cache).
static bool check_sampled_from(qm::Model& B, const std::vector<int>& ids) {
  const tl::gpu::sampling hot{2.0f, 64, 1.0f, 1};
  const std::vector<int> turn2(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 3));
  auto run = [&](bool ahead, bool* ok) {
    qm::reset_cache(B);
    qm::captured_decoder d;
    d.sampler = hot;
    std::vector<int64_t> t;
    for (const auto* turn : {&ids, &turn2}) {
      if (!ahead && !t.empty()) d.feed(B, t.back());
      t.push_back(d.begin(B, *turn));
      for (int i = 0; i < 8; i++) {
        t.push_back(ahead ? d.next(B) : d.step(B, t.back()));
        if (ok)
          *ok &= t.back() == tl::gpu::sample_ref(qm::host_read(B.scratch.logits), B.c.VOCAB,
                                                 hot, uint32_t(t.size() - 1));
      }
    }
    const bool in = d.tok_in;
    d.destroy();
    return std::make_pair(t, in);
  };
  bool ok = true;
  const auto [drawn, ahead] = run(false, &ok);
  std::printf("    step() vs the host definition:");
  for (int64_t t : drawn) std::printf(" %lld", (long long)t);
  std::printf(" %s\n", ok ? "MATCH" : "DIVERGE");
  if (!ahead) {
    std::printf("    next(): no step runs ahead here, skipped\n");
    return ok;
  }
  const bool same = run(true, nullptr).first == drawn;
  std::printf("    next() vs step(): %s\n", same ? "MATCH" : "DIVERGE");
  return ok && same;
}
// The prompt, and its first three tokens, which begin() feeds a token at a
// time (each step it feeds draws too, and is not returned).
static bool check_sampled(qm::Model& B, const std::vector<int>& prompt) {
  bool ok = true;
  for (size_t len : {prompt.size(), size_t(3)}) {
    const std::vector<int> ids(prompt.begin(), prompt.begin() + std::min(len, prompt.size()));
    std::printf("  sampled (temp 2, top_k 64), a %zu-token prompt:\n", ids.size());
    ok &= check_sampled_from(B, ids);
  }
  return ok;
}

// Every attention layer's cached k and v over the first n positions, as f32,
// one vector a layer (DeltaNet layers keep no cache).
static std::vector<std::vector<float>> cache_rows(qm::Model& M, int64_t n) {
  std::vector<std::vector<float>> out;
  for (auto& L : M.layers) {
    const tl::kv_cache& c = L.cache;
    if (!c.K.native) continue;
    std::vector<float>& r = out.emplace_back();
    for (const tl::storage* s : {&c.K, &c.V}) {
      const float* f = qm::host_read(*s);
      const auto* h = reinterpret_cast<const uint16_t*>(f);
      for (int64_t k = 0; k < c.n_kv_heads; k++)
        for (int64_t t = 0; t < n; t++)
          for (int64_t d = 0; d < c.D; d++) {
            const int64_t i = (k * c.max_ctx + t) * c.D + d;
            r.push_back(c.f16 ? qm::f16_to_f32(h[i]) : f[i]);
          }
    }
  }
  return out;
}

// A model the array path does not model (Qwen3.5's DeltaNet, Gemma 3's post
// norms: Config::array_path()) has its checkpoints read off the imperative
// forward: the final norm the last layer's seam leaves in scratch, the logits,
// and the greedy continuation, at F32 against the oracle. Then the bf16 weights
// (the chat's) must reproduce the same greedy opening.
static bool check_imperative(const qm::gg::model& m, const Oracle& O) {
  const int64_t NP = (int64_t)O.n_prompt, NG = (int64_t)O.n_greedy;
  std::vector<int64_t> want;
  // The f32 forward's logits after a spread-out prompt (a quantized file's: the
  // reference its default forms are held to below).
  const std::vector<int> spread = qm::spread_ids(200, qm::config_from_gguf(m).VOCAB);
  std::vector<float> ref_spread;
  bool ok;
  {
    qm::Model M = qm::build(m, tl::dtype::f32);
    const qm::Config& C = M.c;
    std::printf("model built — %lld layers (%lld attention), head_dim=%lld", (long long)C.NL,
                (long long)(C.NL / C.INTERVAL), (long long)C.HD);
    if (C.hybrid())
      std::printf(", DeltaNet %lldx[%lld,%lld]", (long long)C.SH, (long long)C.SDK,
                  (long long)C.SDV);
    std::printf("\n");
    for (int64_t i = 0; i + 1 < NP; i++) qm::step_imperative(M, O.prompt_ids[i], i);
    const float* lp = qm::imperative_logits(M, O.prompt_ids[NP - 1], NP - 1);
    std::vector<float> logits(lp, lp + C.VOCAB);
    const float* fp = qm::host_read(M.scratch.hb);
    double fn_mr = maxrel(std::vector<float>(fp, fp + C.NE), O.fnorm_last, C.NE);
    double logit_mr = 0;
    for (int t = 0; t < 5; t++) {
      const double ref = O.top5_vals[t];
      logit_mr =
          tl::max_err(logit_mr, std::fabs(logits[O.top5_ids[t]] - ref) / (1.0 + std::fabs(ref)));
    }
    const int64_t top1 = qm::argmax(logits);
    std::printf("\ncheckpoints vs numpy-on-same-F16-weights (imperative path):\n");
    std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top1, O.top5_ids[0],
                top1 == O.top5_ids[0] ? "OK" : "FAIL");
    int64_t tok = top1, p = NP;
    bool greedy_ok = true;
    std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      want.push_back(tok);
      if (tok != O.greedy30[i]) greedy_ok = false;
      tok = qm::step_imperative(M, tok, p++);
    }
    std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");
    ok = fn_mr < 5e-3 && logit_mr < 5e-3 && top1 == O.top5_ids[0] && greedy_ok;
    if (qm::quant_file(m)) {
      qm::reset_cache(M);
      const float* sl = nullptr;
      for (size_t i = 0; i < spread.size(); i++) sl = qm::imperative_logits(M, spread[i], (int64_t)i);
      ref_spread.assign(sl, sl + C.VOCAB);
    }
  }
  // The chat's weights: a quantized file's blocks as they lie, else bf16 (a
  // backend without the bf16 GEMV builds f32 instead, and says so). Its default
  // activation forms (the decode's Q4_K GEMVs read f16, the prefill's MMQ int8)
  // round each GEMV input, the MMQ to 8 bits, which moves the logits by an amount
  // that is the model's (Gemma 3 ~70x Qwen's: docs/llama-cpp-comparison.md,
  // "Accuracy"), so the structural checks below read the blocks with float
  // activations, where their bounds hold for every model, and the default forms
  // are held to the f32 reference after them. The checks run the f32 sums, so
  // their bounds stay tight; check_f16_sums holds the default f16 ones to the
  // same prompts after.
  const int64_t NB = 8;
  qm::Model B = qm::build(m, tl::dtype::bf16, false, false, 4096, /*oracle=*/false,
                          /*f16_acc=*/false, /*quant=*/qm::quant_file(m));
  const tl::gpu::act_form chat_form = B.gemv_form;
  const bool chat_mmq = B.mmq;
  B.gemv_form = tl::gpu::act_form::f32;
  B.mmq = false;
  int64_t tok = 0, p = 0;
  auto logits_of = [&](qm::Model& X) {
    const float* l = qm::host_read(X.scratch.logits);
    return std::vector<float>(l, l + X.c.VOCAB);
  };
  for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(B, O.prompt_ids[i], p++);
  const std::vector<float> imp_prompt = logits_of(B);  // the last prompt token's
  const auto imp_cache = cache_rows(B, NP);           // and the k / v the prompt left
  std::vector<float> imp_step;                         // and the first decode step's
  bool bf16_ok = true;
  std::printf("\nthe chat's weights (%s), greedy %lld:\n  cpp:",
              B.quant ? "blocks" : B.outwT.dt() == tl::dtype::bf16 ? "bf16" : "f32",
              (long long)NB);
  for (int64_t i = 0; i < NB; i++) {
    std::printf(" %lld", (long long)tok);
    if (tok != want[i]) bf16_ok = false;
    tok = qm::step_imperative(B, tok, p++);
    if (i == 0) imp_step = logits_of(B);
  }
  std::printf("\n  greedy %s\n", bf16_ok ? "MATCH" : "DIVERGE");

  // The chat's own decoder on the same weights: the prompt through begin() —
  // batched as GEMMs where the backend can, on the tensor cores with the
  // activations rounded to bf16 — then step(), then next() (a step ahead where
  // the backend runs one) for the rest of the greedy run. Its top-5 logits are held to
  // the imperative ones, after the prompt and after the first step: where the
  // backend captures, its warm-up forward must leave the DeltaNet state as it
  // found it, and a drift too small to move a short greedy run is a large one
  // here (~0.3, where the bf16 rounding is ~3e-3; over the whole vocabulary
  // the rounding reaches ~0.1 in the tail, which is printed but not held).
  qm::reset_cache(B);
  qm::captured_decoder dec;
  std::vector<int> ids(O.prompt_ids, O.prompt_ids + NP);
  tok = dec.begin(B, ids);
  const std::vector<float> got_prompt = logits_of(B);
  // The k and v the batched prompt left, held to the token-by-token steps' layer
  // by layer, as ||got - want|| / ||want||: a wrong position or RoPE in one layer
  // moves the whole layer (Gemma's global layers turned unscaled: 0.13 at the
  // first, 0.17 at the last), where the logits see it only through the attention
  // it feeds. The two paths' roundings leave every layer within ~5e-3 (Gemma 3 4B's
  // blocks, Qwen3.5 0.8B), about as much deep as shallow; a single element can
  // differ far more (a large v), which a norm does not let decide.
  const auto got_cache = cache_rows(B, NP);
  double cache_err = 0;
  int64_t cache_worst = -1;
  for (size_t l = 0; l < got_cache.size(); l++) {
    double dd = 0, ww = 0;
    for (size_t i = 0; i < imp_cache[l].size(); i++) {
      const double e = (double)got_cache[l][i] - imp_cache[l][i];
      dd += e * e;
      ww += (double)imp_cache[l][i] * imp_cache[l][i];
    }
    const double err = std::sqrt(dd / ww);
    if (err > cache_err) cache_err = err, cache_worst = (int64_t)l;
  }
  const bool cache_ok = cache_err < 3e-2;
  std::printf("\ndecoder (%s prefill) k / v cache vs imperative: relative error %.2e (attention "
              "layer %lld) %s",
              dec.batched ? "batched" : "token-by-token", cache_err, (long long)cache_worst,
              cache_ok ? "OK" : "FAIL");
  const double pmr = maxrel(got_prompt, imp_prompt.data(), B.c.VOCAB);
  const int64_t first = tok;
  const int64_t second = dec.step(B, first);
  const std::vector<float> got_step = logits_of(B);
  const double smr = maxrel(got_step, imp_step.data(), B.c.VOCAB);
  const double p5 = qm::top5_maxrel(got_prompt.data(), imp_prompt.data(), B.c.VOCAB),
               s5 = qm::top5_maxrel(got_step.data(), imp_step.data(), B.c.VOCAB);
  const bool logits_ok = p5 < 1e-2 && s5 < 1e-2;
  std::printf("\ndecoder (%s prefill) top-5 logits vs imperative: prompt maxrel %.2e, "
              "first step %.2e %s (whole vocabulary %.2e, %.2e)",
              dec.batched ? "batched" : "token-by-token", p5, s5, logits_ok ? "OK" : "FAIL",
              pmr, smr);
  bool dec_ok = true;
  std::printf("\ndecoder (%s), greedy %lld:\n  cpp:", dec.ok() ? "graph capture" : "imperative",
              (long long)NB);
  tok = first;
  for (int64_t i = 0; i < NB; i++) {
    std::printf(" %lld", (long long)tok);
    if (tok != want[i]) dec_ok = false;
    tok = i == 0 ? second : dec.next(B);
  }
  std::printf("\n  greedy %s\n", dec_ok ? "MATCH" : "DIVERGE");

  // A second turn begun on the same decoder, its step queued ahead still in
  // flight: the tokens a fresh decoder gives once that turn has run to the
  // same place (destroy() lands the step ahead, so the cache holds it too).
  const std::vector<int> turn2(O.prompt_ids, O.prompt_ids + std::min<int64_t>(NP, 6));
  auto run_turn2 = [&](qm::captured_decoder& d) {
    std::vector<int64_t> t{d.begin(B, turn2)};
    for (int i = 0; i < 4; i++) t.push_back(d.next(B));
    return t;
  };
  const std::vector<int64_t> same = run_turn2(dec);
  dec.destroy();
  qm::reset_cache(B);
  qm::captured_decoder d1, d2;
  d1.step(B, d1.begin(B, ids));
  for (int64_t i = 1; i < NB; i++) d1.next(B);
  d1.destroy();
  const std::vector<int64_t> fresh = run_turn2(d2);
  d2.destroy();
  const bool turn2_ok = same == fresh;
  std::printf("  second turn on the same decoder:");
  for (int64_t t : same) std::printf(" %lld", (long long)t);
  std::printf(" %s\n", turn2_ok ? "MATCH" : "DIVERGE (a fresh decoder gives another)");

  const bool sample_ok = check_sampled(B, ids);

  // A prompt past one chunk of the delta rule (32 tokens), so the batched
  // prefill takes the backend's chunked form where it has one, and past a
  // sliding window (Gemma 3's local layers), so both paths read one: its last
  // token's top-5 logits against the imperative path's.
  std::vector<int> longp;
  const int64_t n_long = std::max<int64_t>(96, B.c.SWA > 0 ? B.c.SWA + 64 : 0);
  for (int64_t i = 0; i < n_long; i++) longp.push_back(O.prompt_ids[i % NP]);
  qm::reset_cache(B);
  for (int64_t i = 0; i < (int64_t)longp.size(); i++) qm::step_imperative(B, longp[i], i);
  const std::vector<float> imp_long = logits_of(B);
  qm::reset_cache(B);
  bool long_ok = true;
  if (qm::prefill_batched(B, longp) >= 0) {
    const double l5 = qm::top5_maxrel(logits_of(B).data(), imp_long.data(), B.c.VOCAB);
    long_ok = l5 < 1e-2;
    std::printf("\n%lld-token prompt, batched vs imperative: top-5 logits maxrel %.2e %s\n",
                (long long)longp.size(), l5, long_ok ? "OK" : "FAIL");
  }
  const bool f16_ok = !qm::can_prefill_batched(B, (int64_t)longp.size()) ||
                      check_f16_sums(B, longp, imp_long);

  // The chat's own defaults, its activation forms and f16 sums: the decoder's logits
  // after a spread-out prompt, held to the f32 forward's by the KL of their
  // distributions (a chat prompt's next token is too certain for one position to
  // show anything: Gemma 3 4B's KL there is ~1e-13 though its logits move by up to
  // 1). How far int8 moves a model is the model's (Gemma 3 4B 1.4e-2, Qwen3-0.6B
  // 5e-4: docs/llama-cpp-comparison.md); the bound holds an int8 path that broke
  // (Gemma's GLU epilogue run as SiLU: 3.7).
  bool defaults_ok = true;
  if (chat_form != tl::gpu::act_form::f32 || chat_mmq) {
    B.gemv_form = chat_form;
    B.mmq = chat_mmq;
    B.f16_acc = true;
    qm::reset_cache(B);
    qm::captured_decoder d;
    d.begin(B, spread);
    const std::vector<float> got = logits_of(B);
    d.destroy();
    const double kl = qm::kl_div(ref_spread, got);
    const bool top = qm::argmax(got) == qm::argmax(ref_spread);
    defaults_ok = top && kl < 0.3;
    std::printf("\nthe chat's defaults (decode GEMV activations %s, prefill MMQ %s, f16 sums), a "
                "spread-out prompt: KL %.2e from the f32 forward, top token %s %s\n",
                qm::act_form_name(chat_form), chat_mmq ? "on" : "off", kl,
                top ? "same" : "different", defaults_ok ? "OK" : "FAIL");
  }
  return ok && bf16_ok && logits_ok && cache_ok && dec_ok && turn2_ok && sample_ok && long_ok && f16_ok &&
         defaults_ok;
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no GPU device — skipping Qwen forward check\n");
    return 0;
  }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  std::printf("loading %s\n", path.c_str());
  qm::gg::model m(path);
  const qm::Config C = qm::config_from_gguf(m);
  // The models with checkpoints: arch, layers, width.
  const struct {
    const char* arch;
    int64_t nl, ne;
    Oracle o;
  } oracles[] = {{"qwen2", 24, 896, TL_ORACLE(qwenoracle)},      // Qwen2.5-0.5B
                 {"qwen3", 28, 1024, TL_ORACLE(qwen3oracle)},    // Qwen3-0.6B
                 {"qwen35", 24, 1024, TL_ORACLE(qwen35oracle)},  // Qwen3.5-0.8B
                 {"gemma3", 34, 2560, TL_ORACLE(gemma3oracle)}}; // Gemma 3 4B
  const Oracle* found = nullptr;
  for (const auto& r : oracles)
    if (C.arch == r.arch && C.NL == r.nl && C.NE == r.ne) found = &r.o;
  if (!found) {
    std::printf("config mismatch — no oracle for this model\n");
    return 2;
  }
  const Oracle O = *found;
  tl::use_gpu();
  if (!C.array_path()) {
    const bool ok = check_imperative(m, O);
    std::printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
  }
  std::printf("building model (F32, transposing linear weights)...\n");
  const int64_t NP = (int64_t)O.n_prompt;
  bool imp_ok = true, bf16_ok = true, ok_f32 = false;
  {  // scoped so the F32 weights are back in the pool before the bf16 build
  qm::Model M = qm::build(m);
  std::printf("model built — %lld layers, GQA %lldq/%lldkv, head_dim=%lld\n",
              (long long)C.NL, (long long)C.NH, (long long)C.NKV, (long long)C.HD);

  std::vector<float> l0, fnorm, last_logits;
  int64_t pos = 0;
  for (int64_t i = 0; i < NP; i++) {
    bool last = (i == NP - 1);
    last_logits = qm::step(M, O.prompt_ids[i], pos,
                           last ? &l0 : nullptr, last ? &fnorm : nullptr);
    pos++;
  }
  tl::array el = qm::embed_row(M, O.prompt_ids[NP - 1]);
  el.eval();
  double emb_mr = maxrel(std::vector<float>(el.raw(), el.raw() + C.NE),
                         O.emb_last, C.NE);
  double l0_mr = maxrel(l0, O.l0_last, C.NE);
  double fn_mr = maxrel(fnorm, O.fnorm_last, C.NE);

  std::vector<int64_t> top5;
  { std::vector<float> lg = last_logits;
    for (int t = 0; t < 5; t++) { int64_t bi = qm::argmax(lg); top5.push_back(bi); lg[bi] = -1e30f; } }
  bool top1_ok = top5[0] == O.top5_ids[0];
  double logit_mr = 0;
  for (int t = 0; t < 5; t++) {
    int64_t id = O.top5_ids[t];
    logit_mr = tl::max_err(logit_mr, std::fabs((double)last_logits[id] - O.top5_vals[t]) /
                                         (1.0 + std::fabs((double)O.top5_vals[t])));
  }

  std::printf("\ncheckpoints vs numpy-on-same-F16-weights (expect ~1e-4):\n");
  std::printf("  embedding  maxrel %.2e %s\n", emb_mr, emb_mr < 1e-3 ? "OK" : "FAIL");
  std::printf("  layer0 res maxrel %.2e %s\n", l0_mr, l0_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
  std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top5[0],
              O.top5_ids[0], top1_ok ? "OK" : "FAIL");

  const int64_t NG = (int64_t)O.n_greedy;
  std::vector<int64_t> gen;
  int64_t next = qm::argmax(last_logits);
  for (int64_t i = 0; i < NG; i++) {
    gen.push_back(next);
    std::vector<float> lg = qm::step(M, next, pos);
    pos++;
    next = qm::argmax(lg);
  }
  bool greedy_ok = true;
  for (int64_t i = 0; i < NG; i++) if (gen[i] != O.greedy30[i]) greedy_ok = false;
  std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
  for (auto t : gen) std::printf(" %lld", (long long)t);
  std::printf("\n  ref:");
  for (int64_t i = 0; i < NG; i++) std::printf(" %d", O.greedy30[i]);
  std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");

  // The same sequence again, on the imperative path: the fused model-path
  // kernels (rmsnorm/rmsnorm_res/swiglu, the decode GEMVs, kv_append,
  // attn_decode, argmax), or their generic compositions on a backend without
  // them, instead of the array compositions the checkpoints above validated.
  // They are meant to compute the same thing, so at F32 the greedy sequence
  // must be identical — which makes every one of them gated against the numpy
  // reference too, without a second oracle.
  {
    qm::reset_cache(M);
    int64_t p = 0, tok = 0;
    for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(M, O.prompt_ids[i], p++);
    std::printf("\nimperative path (same weights, fused kernels):\n  cpp:");
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      if (tok != O.greedy30[i]) imp_ok = false;
      tok = qm::step_imperative(M, tok, p++);
    }
    std::printf("\n  greedy %s\n", imp_ok ? "MATCH" : "DIVERGE");
  }

  ok_f32 = emb_mr < 1e-3 && l0_mr < 5e-3 && fn_mr < 5e-3 && logit_mr < 5e-3 &&
           top1_ok && greedy_ok && imp_ok;
  }

  // The bf16 weight path is the one a chat actually runs: [N,K] weights for
  // the row GEMV, and the batched prefill's GEMM over a whole chunk of prompt
  // at once. Rounding makes it a different program from the F32 reference
  // above, so the gate is internal consistency — the batched prefill plus the
  // imperative decode must produce the same greedy tokens as THIS model's own
  // array path, which reaches the same rounded weights through .dot() and the
  // separate projections. It is the wiring that check catches: a decode GEMV
  // pointed at the [K,N] oracle weight instead of the [N,K] copy reads the
  // right number of bytes in the wrong order, and nothing but a comparison
  // like this one notices.
  const int64_t NB = 8;
  {
    std::printf("\nbf16 weights (row GEMV + batched prefill) vs this model's array path:\n");
    qm::Model B = qm::build(m, tl::dtype::bf16, false, false, 4096, true, /*f16_acc=*/false);
    std::vector<int> ids(O.prompt_ids, O.prompt_ids + NP);
    if (!qm::can_prefill_batched(B, NP)) {
      std::printf("  backend has no bf16 weight GEMM — skipped\n");
    } else {
      std::vector<int64_t> want, got;
      std::vector<float> array_prompt;  // the f32 array path's last-prompt logits
      {
        qm::reset_cache(B);
        int64_t p = 0;
        std::vector<float> lg;
        for (int64_t i = 0; i < NP; i++) lg = qm::step(B, ids[i], p++);
        array_prompt = lg;
        int64_t next = qm::argmax(lg);
        for (int64_t i = 0; i < NB; i++) {
          want.push_back(next);
          lg = qm::step(B, next, p++);
          next = qm::argmax(lg);
        }
      }
      qm::reset_cache(B);
      int64_t next = qm::prefill_batched(B, ids);
      int64_t p = qm::seq_cache(B).pos;
      for (int64_t i = 0; i < NB; i++) {
        got.push_back(next);
        next = qm::step_imperative(B, next, p++);
      }
      std::printf("  array:");
      for (auto t : want) std::printf(" %lld", (long long)t);
      std::printf("\n  fused:");
      for (auto t : got) std::printf(" %lld", (long long)t);
      bf16_ok = got == want;
      std::printf("\n  greedy %s\n", bf16_ok ? "MATCH" : "DIVERGE");
      bf16_ok &= check_f16_sums(B, ids, array_prompt);
      bf16_ok &= check_sampled(B, ids);
    }
  }

  bool ok = ok_f32 && bf16_ok;
  std::printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
