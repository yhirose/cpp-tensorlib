// M9 "actually chat" — numeric proof. Load a real Qwen2.5-0.5B-Instruct, Qwen3
// or Qwen3.5 GGUF (F16 or BF16 weights, or Qwen3.5 0.8B's Q4_K_M, a fifth of it
// Q5_K), Gemma 3 4B or 1B (Q4_K_M), or Llama 3.2 1B (F16, Q4_K_M, or Q4_0 with
// Q4_1) or 3B (Q4_K_M), and run the decoder end to end, verified
// against a tight numpy reference that reads the SAME weights (tools/qwen_oracle.py
// wrote the *_oracle_data.h headers: Gemma 3 1B's over a prompt past its
// 512-position window, Llama 3.2 1B's over one long enough that its RoPE factors
// turn the low-frequency pairs well apart; the model's arch and shape pick one,
// and of several, the one that read the file's weights). Identical
// weights => the C++ forward must match the reference to ~1e-4 at every checkpoint
// (embedding, layer-0 residual, final norm, top-5 logits) AND reproduce the greedy
// token sequence exactly. A file of other weights than its oracle's (another
// quantization, or another quantizer's: held_to_oracle) skips those two, its paths
// held to its own f32 greedy instead. A block-quantized file runs on its own
// blocks, as a chat does, its f32 reference reading them with f32 activations and
// an f32 KV cache (check_imperative), not widened to four bytes a weight. Proves
// weight loading + GGML->our layout transpose + each family's structure + the
// attention kernels, on whatever backend gpu:: resolves to. Tokenizer/sampling
// are separate (see chat_qwen). Model path: argv[1] or ~/models/...fp16.gguf.

#include "gemma3_1b_oracle_data.h"
#include "gemma3_oracle_data.h"
#include "llama32_1b_oracle_data.h"
#include "llama32_1b_q40_oracle_data.h"
#include "llama32_1b_q4_oracle_data.h"
#include "llama32_3b_oracle_data.h"
#include "qwen.h"
#include "qwen35_oracle_data.h"
#include "qwen35_q4_oracle_data.h"
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
  // The weights its numpy forward read: 0 a float file's (F16; the checks' bounds
  // take F32's and BF16's too, the same model's values), else the fingerprint of
  // the block-quantized file it read (weights_fingerprint), which is that file's
  // and no other's: a Q4_K_M from another quantizer (an imatrix, another choice
  // of Q6_K tensors) carries the same label but other blocks.
  uint64_t weights = 0;
};

// A file's weights by their bytes: FNV-1a over each tensor's name, ggml type and
// first 64 bytes of data, in file order. What a block oracle names (Oracle.weights),
// and what check_qwen prints for one regenerated from another file.
static uint64_t weights_fingerprint(const qm::gg::model& m) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](const void* p, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
  };
  for (const auto& t : m.tensors()) {
    const uint32_t type = static_cast<uint32_t>(t.type);
    mix(t.name.data(), t.name.size());
    mix(&type, sizeof type);
    mix(t.data, std::min<uint64_t>(64, t.nbytes));
  }
  return h;
}

// Whether the oracle read this file's weights, so its checkpoints and greedy are
// this model's: a float oracle any float file's (no blocks), a block-quantized one
// only the file whose fingerprint it names.
static bool read_by(const qm::gg::model& m, const Oracle& O) {
  return O.weights != 0 ? weights_fingerprint(m) == O.weights : !qm::quant_file(m);
}

// read_by, and where it is not, which weights each read and what the paths are
// held to instead.
static bool held_to_oracle(const qm::gg::model& m, const Oracle& O) {
  const bool same = read_by(m, O);
  if (O.weights != 0)
    std::printf("\nweights fingerprint %016llx, the oracle's %016llx\n",
                (unsigned long long)weights_fingerprint(m), (unsigned long long)O.weights);
  if (!same)
    std::printf("the oracle read other weights: its checkpoints and greedy are skipped, and the "
                "paths below are held to this model's own f32 greedy\n");
  return same;
}
#define TL_ORACLE(ns, weights)                                              \
  Oracle {                                                                  \
    ns::emb_last, ns::l0_last, ns::fnorm_last, ns::top5_vals, ns::top5_ids, \
        ns::greedy30, ns::prompt_ids, sizeof(ns::greedy30) / sizeof(int),   \
        sizeof(ns::prompt_ids) / sizeof(int), weights                       \
  }

static double maxrel(const std::vector<float>& a, const float* b, int64_t n) {
  double mr = 0;
  for (int64_t i = 0; i < n; i++)
    mr = tl::max_err(mr, std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)b[i])));
  return mr;
}

// The logits the last forward left (scratch.logits), on the host.
static std::vector<float> logits_of(qm::Model& M) {
  const float* l = qm::host_read(M.scratch.logits);
  return std::vector<float>(l, l + M.c.VOCAB);
}

// The batched prefill held to the token-by-token steps, both in the f32 forms, by
// their last-token top-5 logits (top5_maxrel): after the oracle's prompt, after the
// first step from it, and after a prompt past a delta-rule chunk and a sliding
// window. One bound for all three. The two paths add and round in other orders only
// (every backend's exact GEMM keeps its operands f32, as the GEMVs do), which leaves
// <= 1e-3 on every model here (Gemma 3 1B's 769-token prompt the most, 9.9e-4 on
// CUDA), and a prefill that ignores the sliding window moves Gemma 3 1B's 576-token
// prompt by 1.5e-2 (Metal). Operands rounded to bf16 left up to 1.1e-2 (CUDA's
// GEMM before it kept them f32, on the repeated prompt of Qwen3.5's Q4_K_M).
constexpr double batched_bound = 1e-2;

// The imperative forward's logits after `ids`, from position 0.
static std::vector<float> imperative_prompt(qm::Model& M, const std::vector<int>& ids) {
  qm::reset_cache(M);
  for (size_t i = 0; i < ids.size(); i++) qm::step_imperative(M, ids[i], (int64_t)i);
  return logits_of(M);
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
  const bool was = B.f16_acc;
  B.f16_acc = true;
  if (!qm::prefill_sums_f16(B, false)) {  // nothing to hold: its prefill sums in f32 still
    std::printf("\nf16 sums: none on this build (block-quantized weights, f32 activations) — skipped\n");
    B.f16_acc = was;
    return true;
  }
  qm::reset_cache(B);
  const int64_t tok = qm::prefill_batched(B, ids);
  const double t5 = qm::top5_maxrel(logits_of(B).data(), ref.data(), B.c.VOCAB);
  const bool close = t5 < 2e-2 && tok == qm::argmax(ref);
  std::printf("\nf16 sums: top-5 logits maxrel %.2e vs f32, token %lld %s\n", t5, (long long)tok,
              close ? "OK" : "FAIL");
  const tl::array fn = B.layers[0].fn;
  B.layers[0].fn = (fn * 1e5f).clone();
  B.layers[0].fn.eval();
  const int64_t retries = B.f16_retries;
  qm::reset_cache(B);
  qm::prefill_batched(B, ids);
  const std::vector<float> redone = logits_of(B);
  qm::reset_cache(B);
  qm::prefill_batched(B, ids, qm::PREFILL_CHUNK, /*exact=*/true);
  const std::vector<float> f32 = logits_of(B);
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

// The reference forward's checks, on R: its checkpoints read off the imperative
// forward (the final norm the last layer's seam leaves in scratch, the logits)
// and the greedy continuation, against the oracle where it read these weights.
// `want` gets the greedy run, which the chat's paths are held to after.
static bool check_reference(qm::Model& R, const Oracle& O, bool held, std::vector<int64_t>& want) {
  const int64_t NP = (int64_t)O.n_prompt, NG = (int64_t)O.n_greedy;
  const qm::Config& C = R.c;
  std::printf("model built — %lld layers (%lld attention), head_dim=%lld", (long long)C.NL,
              (long long)(C.NL / C.INTERVAL), (long long)C.HD);
  if (C.hybrid())
    std::printf(", DeltaNet %lldx[%lld,%lld]", (long long)C.SH, (long long)C.SDK,
                (long long)C.SDV);
  std::printf("\n");
  const std::vector<float> logits =
      imperative_prompt(R, std::vector<int>(O.prompt_ids, O.prompt_ids + NP));
  const float* fp = qm::host_read(R.scratch.hb);
  double fn_mr = maxrel(std::vector<float>(fp, fp + C.NE), O.fnorm_last, C.NE);
  double logit_mr = 0;
  for (int t = 0; t < 5; t++) {
    const double ref = O.top5_vals[t];
    logit_mr =
        tl::max_err(logit_mr, std::fabs(logits[O.top5_ids[t]] - ref) / (1.0 + std::fabs(ref)));
  }
  const int64_t top1 = qm::argmax(logits);
  if (held) {
    std::printf("\ncheckpoints vs numpy-on-the-same-weights (imperative path):\n");
    std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top1, O.top5_ids[0],
                top1 == O.top5_ids[0] ? "OK" : "FAIL");
  }
  int64_t tok = top1, p = NP;
  bool greedy_ok = true;
  std::printf("\ngreedy %lld tokens:\n  cpp:", (long long)NG);
  for (int64_t i = 0; i < NG; i++) {
    std::printf(" %lld", (long long)tok);
    want.push_back(tok);
    if (tok != O.greedy30[i]) greedy_ok = false;
    tok = qm::step_imperative(R, tok, p++);
  }
  if (held) std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");
  else std::printf("\n");
  return !held || (fn_mr < 5e-3 && logit_mr < 5e-3 && top1 == O.top5_ids[0] && greedy_ok);
}

// A model the array path does not model (Qwen3.5's DeltaNet, Gemma 3's post
// norms: Config::array_path()), and any block-quantized file, whose blocks only
// the imperative path reads, is checked here, on the imperative path. Its
// reference forward is f32: a float file's weights widened exactly, a quantized
// file's own blocks read with f32 activations, sums and KV cache. A GEMV
// dequantizes each block exactly, so the latter are the widened weights' values,
// summed in another order, at about the file's size rather than four bytes a
// weight (Gemma 3 4B's check peaks at 4.5 GB on Metal, where its widened weights
// alone are 15.5 GB). Then the chat's weights must reproduce the reference's
// greedy opening, and its decoder the imperative path.
static bool check_imperative(const qm::gg::model& m, const Oracle& O, bool held) {
  const int64_t NP = (int64_t)O.n_prompt;
  const bool quant = qm::quant_file(m);
  std::vector<int64_t> want;
  bool ok = true;
  if (!quant) {  // scoped so the f32 weights are back in the pool before the bf16 build
    qm::Model M = qm::build(m, tl::dtype::f32);
    ok = check_reference(M, O, held, want);
  }
  // The chat's weights: a quantized file's blocks as they lie, else bf16 (a
  // backend without the bf16 GEMV builds f32 instead, and says so). Its default
  // activation forms (the decode's Q4_K and Q5_K GEMVs read f16, the prefill's MMQ int8)
  // round each GEMV input, the MMQ to 8 bits, which moves the logits by an amount
  // that is the model's (Gemma 3 ~70x Qwen's: docs/llama-cpp-comparison.md,
  // "Accuracy"), so the structural checks below read the blocks with float
  // activations, where their bounds hold for every model, and the default forms
  // are held to the f32 reference after them. The checks run the f32 sums, so
  // their bounds stay tight; check_f16_sums holds the default f16 ones to the
  // same prompts after.
  const int64_t NB = 8;
  qm::Model B = qm::build(m, tl::dtype::bf16, false, false, 4096, /*oracle=*/false,
                          /*f16_acc=*/false, /*quant=*/quant);
  const tl::gpu::act_form chat_form = B.gemv_form;
  const bool chat_mmq = B.mmq;
  B.gemv_form = tl::gpu::act_form::f32;
  B.mmq = false;
  // A quantized file's reference is these weights in those forms with an f32 KV
  // cache (the chat's rounds K and V to f16: Gemma 3's final norm 1e-2 off the
  // oracle's for 4e-5), and so are its logits after a spread-out prompt, which its
  // default forms are held to below.
  const std::vector<int> spread = qm::spread_ids(200, B.c.VOCAB);
  std::vector<float> ref_spread;
  if (quant) {
    const bool chat_kv = B.kv_f16;
    qm::init_kv_caches(B, false);
    ok = check_reference(B, O, held, want);
    ref_spread = imperative_prompt(B, spread);
    qm::init_kv_caches(B, chat_kv);
    qm::reset_cache(B);
  }
  // The prompt again, token by token: the logits, cache and first step the
  // decoder is held to below, and the greedy opening, the reference's (a
  // quantized file's own, run again from a reset cache).
  int64_t tok = 0, p = 0;
  for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(B, O.prompt_ids[i], p++);
  const std::vector<float> imp_prompt = logits_of(B);  // the last prompt token's
  const auto imp_cache = cache_rows(B, NP);           // and the k / v the prompt left
  std::vector<float> imp_step;                         // and the first decode step's
  bool bf16_ok = true;
  std::printf("\nthe chat's weights (%s), greedy %lld:\n  cpp:",
              B.quant ? "blocks" : tl::dtype_name(B.wdt), (long long)NB);
  for (int64_t i = 0; i < NB; i++) {
    std::printf(" %lld", (long long)tok);
    if (tok != want[i]) bf16_ok = false;
    tok = qm::step_imperative(B, tok, p++);
    if (i == 0) imp_step = logits_of(B);
  }
  std::printf("\n  greedy %s\n", bf16_ok ? "MATCH" : "DIVERGE");

  // The chat's own decoder on the same weights: the prompt through begin() —
  // batched as GEMMs where the backend can — then step(), then next() (a step
  // ahead where the backend runs one) for the rest of the greedy run. Its top-5
  // logits are held to the imperative ones (batched_bound), after the prompt and
  // after the first step: where the backend captures, its warm-up forward must
  // leave the DeltaNet state as it found it, and a drift too small to move a short
  // greedy run is a large one here (~0.3). Over the whole vocabulary the tail moves
  // more (up to 1.4e-2, Gemma 3 1B), which is printed but not held.
  qm::reset_cache(B);
  qm::captured_decoder dec;
  std::vector<int> ids(O.prompt_ids, O.prompt_ids + NP);
  tok = dec.begin(B, ids);
  const std::vector<float> got_prompt = logits_of(B);
  // The k and v the batched prompt left, held to the token-by-token steps' layer
  // by layer, as ||got - want|| / ||want||: a wrong position or RoPE in one layer
  // moves the whole layer (Gemma's global layers turned unscaled: 0.13 at the
  // first, 0.17 at the last), where the logits see it only through the attention
  // it feeds. The two paths' roundings leave every layer within ~5e-4 (every model
  // here), about as much deep as shallow; a single element can
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
  const bool logits_ok = p5 < batched_bound && s5 < batched_bound;
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
  const std::vector<float> imp_long = imperative_prompt(B, longp);
  qm::reset_cache(B);
  bool long_ok = true;
  if (qm::prefill_batched(B, longp) >= 0) {
    const double l5 = qm::top5_maxrel(logits_of(B).data(), imp_long.data(), B.c.VOCAB);
    long_ok = l5 < batched_bound;
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
  B.mmq = chat_mmq;
  B.f16_acc = true;
  // A quantized file's (ref_spread is its f32 forward's), where its forms are not
  // the f32 ones the checks above ran.
  const bool mmq = qm::prefill_mmq(B, false), f16 = qm::prefill_sums_f16(B, false);
  if (B.quant && (chat_form != tl::gpu::act_form::f32 || mmq || f16)) {
    B.gemv_form = chat_form;
    qm::reset_cache(B);
    qm::captured_decoder d;
    d.begin(B, spread);
    const std::vector<float> got = logits_of(B);
    d.destroy();
    const double kl = qm::kl_div(ref_spread, got);
    const bool top = qm::argmax(got) == qm::argmax(ref_spread);
    defaults_ok = top && kl < 0.3;
    std::printf("\nthe chat's defaults (decode GEMV activations %s, prefill MMQ %s, sums in %s), a "
                "spread-out prompt: KL %.2e from the f32 forward, top token %s %s\n",
                qm::act_form_name(chat_form), mmq ? "on" : "off", f16 ? "f16" : "f32", kl,
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
  } oracles[] = {
      {"qwen2", 24, 896, TL_ORACLE(qwenoracle, 0)},      // Qwen2.5-0.5B, F16
      {"qwen3", 28, 1024, TL_ORACLE(qwen3oracle, 0)},    // Qwen3-0.6B, F16
      {"qwen35", 24, 1024, TL_ORACLE(qwen35oracle, 0)},  // Qwen3.5-0.8B, F16
      // unsloth/Qwen3.5-0.8B-GGUF's Qwen3.5-0.8B-Q4_K_M.gguf (its attn_qkv and more Q5_K)
      {"qwen35", 24, 1024, TL_ORACLE(qwen35_q4oracle, 0xea8ff0e6652acd3dull)},
      // unsloth/gemma-3-{4b,1b}-it-GGUF's gemma-3-*-it-Q4_K_M.gguf
      {"gemma3", 34, 2560, TL_ORACLE(gemma3oracle, 0x0b1451ea793e00e9ull)},     // Gemma 3 4B
      {"gemma3", 26, 1152, TL_ORACLE(gemma3_1boracle, 0x1f0d93832cf2bb46ull)},  // Gemma 3 1B
      // unsloth/Llama-3.2-{1B,3B}-Instruct-GGUF's F16, Q4_K_M and Q4_0
      {"llama", 16, 2048, TL_ORACLE(llama32_1boracle, 0)},                     // Llama 3.2 1B, F16
      {"llama", 16, 2048, TL_ORACLE(llama32_1b_q4oracle, 0x00f056cb14a01068ull)},  // 1B Q4_K_M
      {"llama", 16, 2048, TL_ORACLE(llama32_1b_q40oracle, 0xc82864007acd8011ull)},  // 1B Q4_0
      {"llama", 28, 3072, TL_ORACLE(llama32_3boracle, 0x398103f1bd23c54bull)}};    // 3B Q4_K_M
  // The oracle for the model's shape; of several (a float file's and quantized ones'),
  // the one that read this file's weights, else the first.
  const Oracle* found = nullptr;
  for (const auto& r : oracles)
    if (C.arch == r.arch && C.NL == r.nl && C.NE == r.ne && (!found || read_by(m, r.o)))
      found = &r.o;
  if (!found) {
    std::printf("config mismatch — no oracle for this model\n");
    return 2;
  }
  const Oracle O = *found;
  const bool same = held_to_oracle(m, O);
  tl::use_gpu();
  // The array path (forward(), the per-op oracle) reads float weights; a
  // quantized file runs on its own blocks, as a chat does (a backend that reads
  // no blocks refuses one: build() says so).
  if (!C.array_path() || qm::quant_file(m)) {
    const bool ok = check_imperative(m, O, same);
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

  if (same) {
    std::printf("\ncheckpoints vs numpy-on-the-same-weights (expect ~1e-4):\n");
    std::printf("  embedding  maxrel %.2e %s\n", emb_mr, emb_mr < 1e-3 ? "OK" : "FAIL");
    std::printf("  layer0 res maxrel %.2e %s\n", l0_mr, l0_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  final norm maxrel %.2e %s\n", fn_mr, fn_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  top5 logit maxrel %.2e %s\n", logit_mr, logit_mr < 5e-3 ? "OK" : "FAIL");
    std::printf("  argmax id  cpp=%lld ref=%d %s\n", (long long)top5[0],
                O.top5_ids[0], top1_ok ? "OK" : "FAIL");
  }

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
  if (same) {
    std::printf("\n  ref:");
    for (int64_t i = 0; i < NG; i++) std::printf(" %d", O.greedy30[i]);
    std::printf("\n  greedy %s\n", greedy_ok ? "MATCH" : "DIVERGE");
  } else {
    std::printf("\n");
  }

  // The same sequence again, on the imperative path: the fused model-path
  // kernels (rmsnorm/rmsnorm_res/swiglu, the decode GEMVs, kv_append,
  // attn_decode, argmax), or their generic compositions on a backend without
  // them, instead of the array compositions the checkpoints above validated.
  // They are meant to compute the same thing, so at F32 the greedy sequence
  // must be the array path's — which, where that matched the numpy reference,
  // gates every one of them against it too, without a second oracle.
  {
    qm::reset_cache(M);
    int64_t p = 0, tok = 0;
    for (int64_t i = 0; i < NP; i++) tok = qm::step_imperative(M, O.prompt_ids[i], p++);
    std::printf("\nimperative path (same weights, fused kernels):\n  cpp:");
    for (int64_t i = 0; i < NG; i++) {
      std::printf(" %lld", (long long)tok);
      if (tok != gen[i]) imp_ok = false;
      tok = qm::step_imperative(M, tok, p++);
    }
    std::printf("\n  greedy %s\n", imp_ok ? "MATCH" : "DIVERGE");
  }

  ok_f32 = (!same || (emb_mr < 1e-3 && l0_mr < 5e-3 && fn_mr < 5e-3 && logit_mr < 5e-3 &&
                       top1_ok && greedy_ok)) &&
           imp_ok;
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
