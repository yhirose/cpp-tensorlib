// M9 prefill census — the prompt-side twin of bench_qwen_ctx (which measured
// decode). chat_qwen currently prefills by feeding the prompt through the array
// `step` path one token at a time, so a long prompt costs T full forwards of
// the slowest path we have. Two questions, measured separately:
//
//   1. Which of the three EXISTING token-by-token paths (array / imperative /
//      CUDA-graph replay) prefills fastest, and does the answer hold as the
//      prompt grows? Nothing new has to be built to take this win — it is the
//      same forward chat_qwen already runs for decode.
//   2. What is the ceiling above token-by-token, i.e. how much would BATCHING
//      the prompt (M>1) buy? Sized in two isolated parts: the weight
//      projections (M=1 GEMV per token vs one M=T GEMM) and the attention
//      (T growing-ctx decode calls vs one tiled attn_prefill). This is the
//      unbuilt lever — measure it before building it.
//
// Usage: bench_qwen_prefill [model.gguf] [max_T]

#include "qwen_model.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace qm = qwenmodel;
namespace cu = tl::cuda;
static double now_ms() { return qm::StepProf::now_ms(); }

enum class Path { Array, Imperative, Captured };

// Zero-filled device buffer of `n` floats, marked host-live so the first kernel
// that reads it uploads (dummy operands — only their shape drives the timing).
static void* zeros(int64_t n) {
  float* h = nullptr;
  void* p = cu::alloc(n * 4, &h);
  if (p) {
    std::memset(h, 0, (size_t)n * 4);
    cu::sync_to_host(p, true);
  }
  return p;
}

// Deterministic prompt of T arbitrary-but-valid ids.
static std::vector<int> make_ids(int64_t T) {
  std::vector<int> ids((size_t)T);
  for (int64_t i = 0; i < T; i++) ids[(size_t)i] = qm::PREFILL_IDS[i % 4];
  return ids;
}

// Prefill T tokens through one token-by-token path; returns wall-clock ms
// (-1 when the path is unavailable). The cache is reset first, so each run
// starts from a real pos=0 prompt.
static double prefill_ms(qm::Model& M, int64_t T, Path p) {
  qm::reset_cache(M);
  std::vector<int> ids = make_ids(T);

  if (p == Path::Captured) {
    // The SHIPPED path (what chat_qwen runs), not a hand-rolled twin: one
    // capture, feed() for every prompt token but the last. Its cost is excluded
    // from the throughput — it is a fixed cost, not per-token work. Still an
    // upper bound in one respect: every replay also runs lm_head, which prefill
    // needs only for the last token.
    qm::captured_decoder dec;
    double t = now_ms();
    dec.begin(M, ids);
    double ms = now_ms() - t - dec.capture_ms;
    bool ok = dec.ok();
    dec.destroy();
    return ok ? ms : -1;
  }
  double t = now_ms();
  for (int64_t i = 0; i < T; i++) {
    if (p == Path::Array) qm::step(M, ids[(size_t)i], i);
    else qm::step_imperative(M, ids[(size_t)i], i);
  }
  return now_ms() - t;
}

// min-of-R timing around a GPU-work lambda (WSL2 boost noise), ms per rep.
template <typename F>
static double min_ms(int rounds, int reps, F&& work) {
  double best = 1e30;
  for (int r = 0; r < rounds; r++) {
    double t = now_ms();
    for (int i = 0; i < reps; i++) work();
    cu::flush();
    best = std::min(best, (now_ms() - t) / reps);
  }
  return best;
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) { std::printf("no CUDA device — skipping\n"); return 0; }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  int64_t max_T = argc > 2 ? std::atoll(argv[2]) : 2048;

  qm::gg::model m(path);
  if (!qm::check_config(m)) { std::printf("config mismatch\n"); return 2; }
  tl::use_gpu();
  qm::Model M = qm::build(m, tl::dtype::bf16);
  std::printf("model %s | bf16 row | prompt lengths up to %lld\n\n", path.c_str(),
              (long long)max_T);

  // ---- 0. Correctness of the captured prefill: running the prompt through the
  // graph captured at pos 0 must leave the same state as feeding it through the
  // array `step` path — same first generated token, which is what lets
  // chat_qwen prefill through the graph. The greedy token is the gate; the
  // logits maxrel is informational and is larger here (~1e-5) than the
  // single-step dpos-vs-hostpos check (~1e-7) because the two paths' GEMV
  // reduction orders (split-K [K,N] vs warp-per-row [N,K]) differ slightly in
  // the k/v they write, and that difference accumulates over T cached rows.
  {
    const int64_t T = 64;
    std::vector<int> ids = make_ids(T);
    qm::reset_cache(M);
    std::vector<float> ref;
    for (int64_t i = 0; i < T; i++) ref = qm::step(M, ids[(size_t)i], i);
    int64_t ref_tok = qm::argmax(ref);

    // Gates: the greedy token must match, and maxrel must stay in the
    // accumulated-reduction-order band (~1e-5). The bound is loose on purpose —
    // it catches a real numeric regression without pinning WSL2/driver noise.
    const double MAXREL_BOUND = 1e-3;
    std::printf("=== 0. captured prefill vs array prefill (T=%lld) ===\n",
                (long long)T);
    auto compare = [&](const char* name, int64_t got_tok) {
      cu::sync_to_host(M.scratch.logitsb, false);
      const float* got = M.scratch.logits_host;
      double maxrel = 0;
      for (int64_t i = 0; i < qm::VOCAB; i++)
        maxrel = std::max(maxrel, (double)std::fabs(got[i] - ref[(size_t)i]) /
                                      (1.0 + std::fabs(ref[(size_t)i])));
      bool tok_ok = got_tok == ref_tok, rel_ok = maxrel < MAXREL_BOUND;
      std::printf("  %-9s first token %lld vs array %lld %s | logits maxrel "
                  "%.2e %s\n",
                  name, (long long)got_tok, (long long)ref_tok,
                  tok_ok ? "(MATCH)" : "(!! MISMATCH)", maxrel,
                  rel_ok ? "OK" : "(!! over bound)");
      return tok_ok && rel_ok;
    };

    qm::reset_cache(M);
    qm::captured_decoder cap;
    bool ok = compare("graph", cap.begin(M, ids));
    cap.destroy();

    // Same check for feed()'s no-graph branch, which a box with CUDA-graph
    // support would otherwise never execute. Driving feed/step directly (rather
    // than begin, which would capture) with only the capacity bound set is what
    // forces it: exec stays null, so every token takes the imperative path.
    qm::reset_cache(M);
    qm::captured_decoder fb;
    fb.max_ctx = qm::MAXC;
    for (size_t i = 0; i + 1 < ids.size(); i++) fb.feed(M, ids[i]);
    ok &= compare("fallback", fb.step(M, ids.back()));
    fb.destroy();
    std::printf("\n");
    if (!ok) return 1;
  }

  // ---- 1. token-by-token prefill: which existing path, and does it hold up?
  // The array path costs ~10ms/token, so sweeping it to max_T would dominate
  // the bench's runtime for a number that is already flat by T=512; it is
  // capped here and the cap is reported rather than silently applied.
  const int64_t ARRAY_MAX = 512;
  std::printf("=== prefill: existing token-by-token paths ===\n");
  std::printf("  %6s  %11s  %11s  %11s   (tok/s)\n", "T", "array", "imperative",
              "graph");
  for (int64_t T : {(int64_t)32, (int64_t)128, (int64_t)512, (int64_t)1024,
                    (int64_t)2048}) {
    if (T > max_T) break;
    double a = T <= ARRAY_MAX ? prefill_ms(M, T, Path::Array) : -1;
    double i = prefill_ms(M, T, Path::Imperative);
    double g = prefill_ms(M, T, Path::Captured);
    char ab[32];
    if (a > 0) std::snprintf(ab, sizeof ab, "%.1f", T * 1000.0 / a);
    else std::snprintf(ab, sizeof ab, "(skipped)");
    std::printf("  %6lld  %11s  %11.1f  %11.1f\n", (long long)T, ab,
                T * 1000.0 / i, g > 0 ? T * 1000.0 / g : 0.0);
  }
  std::printf("  (array capped at T=%lld — ~10ms/token, and already flat there)\n\n",
              (long long)ARRAY_MAX);

  // ---- 2a. Projection batching headroom: one M=1 GEMV per token (what every
  // token-by-token path runs) vs one M=T GEMM for the whole prompt. The
  // production GEMV is bf16 warp-per-row [N,K]; the batched proxy is the
  // existing f32 sgemm_rb, which reads 2x the weight bytes — so it is a
  // CONSERVATIVE stand-in for the bf16 GEMM a batched prefill would want.
  std::printf("=== 2a. projection batching headroom (per prompt token) ===\n");
  std::printf("  gemv = bf16 [N,K] M=1 (production); gemm = f32 M=T (proxy, 2x bytes)\n");
  const int64_t MB = 512;  // batch used for the GEMM column
  struct Shape { const char* name; int64_t K, N; const tl::array* w; };
  qm::Layer& L0 = M.layers[0];
  const Shape shapes[] = {
      {"QKV.fused", qm::NE, (qm::NH + 2 * qm::NKV) * qm::HD, &L0.wqkv},
      {"wo", qm::NH * qm::HD, qm::NE, &L0.wo_row},
      {"gateup.fused", qm::NE, 2 * qm::FF, &L0.wgu},
      {"wd", qm::FF, qm::NE, &L0.wd_row},
  };
  std::printf("  %14s  %6s %7s  %10s  %10s  %8s\n", "shape", "K", "N",
              "gemv(us)", "gemm(us)", "speedup");
  // One max-sized trio reused by every shape (a [MB,Kmax], wf [K*N]max, y
  // [MB,Nmax]) — cu::release recycles into a size-keyed pool rather than
  // returning memory to the driver, so per-shape buffers would stay committed
  // for the whole run (~101 MB) on top of the model. ~65 MB peak this way, and
  // one zero-fill/upload instead of four.
  int64_t maxK = 0, maxN = 0, maxW = 0;
  for (const Shape& s : shapes) {
    maxK = std::max(maxK, s.K);
    maxN = std::max(maxN, s.N);
    maxW = std::max(maxW, s.K * s.N);
  }
  void* a = zeros(MB * maxK);   // [M,K] activations
  void* wf = zeros(maxW);       // [K,N] f32 dummy weights
  void* y = zeros(MB * maxN);
  if (!a || !wf || !y) { std::printf("  alloc failed\n"); return 1; }

  double sum_gemv = 0, sum_gemm = 0;
  for (const Shape& s : shapes) {
    // Warm both paths (module load, first-touch upload) outside the timing.
    cu::gemv_bf16_row(a, s.w->native(), y, s.N, s.K);
    cu::gemm(a, 0, s.K, false, wf, 0, s.N, false, y, 0, MB, s.N, s.K, 1.0f, 0.0f);
    cu::flush();

    double gemv_us = 1000.0 * min_ms(3, 20, [&] {
      cu::gemv_bf16_row(a, s.w->native(), y, s.N, s.K);
    });
    double gemm_us = 1000.0 * min_ms(3, 5, [&] {
      cu::gemm(a, 0, s.K, false, wf, 0, s.N, false, y, 0, MB, s.N, s.K, 1.0f, 0.0f);
    }) / MB;  // per prompt token
    sum_gemv += gemv_us;
    sum_gemm += gemm_us;
    std::printf("  %14s  %6lld %7lld  %10.2f  %10.2f  %7.1fx\n", s.name,
                (long long)s.K, (long long)s.N, gemv_us, gemm_us,
                gemv_us / gemm_us);
  }
  for (void* p : {a, wf, y}) cu::release(p, 0, nullptr);
  std::printf("  %14s  %6s %7s  %10.2f  %10.2f  %7.1fx  (x24 layers)\n", "SUM/layer",
              "", "", sum_gemv, sum_gemm, sum_gemv / sum_gemm);
  std::printf("  (the gemv column repeats one shape back-to-back, so QKV/wo — "
              "whose weights fit L2 — read warm; their speedup is a floor)\n\n");

  // ---- 2b. Attention batching headroom: T growing-ctx decode calls (what
  // token-by-token prefill runs) vs one tiled attn_prefill over the whole
  // prompt. Both kernels already exist; kv_cache::prefill is the batched entry.
  std::printf("=== 2b. attention batching headroom (whole prompt, 1 layer) ===\n");
  std::printf("  %6s  %14s  %14s  %8s\n", "T", "per-token(ms)", "batched(ms)",
              "speedup");
  for (int64_t T : {(int64_t)128, (int64_t)512, (int64_t)2048}) {
    if (T > max_T) break;
    cu::kv_cache c;
    if (!c.init(qm::NKV, qm::MAXC, qm::HD)) { std::printf("  cache init failed\n"); return 1; }
    void* ks = zeros(qm::NKV * T * qm::HD);
    void* vs = zeros(qm::NKV * T * qm::HD);
    void* qs = zeros(qm::NH * T * qm::HD);
    void* os = zeros(qm::NH * T * qm::HD);
    if (!ks || !vs || !qs || !os) { std::printf("  alloc failed\n"); return 1; }

    // Per-token: append + attend at every position 0..T-1 (ctx grows).
    double per_tok = min_ms(3, 1, [&] {
      c.pos = 0;
      for (int64_t i = 0; i < T; i++) {
        c.append(ks, vs);  // same row values each step; only the cost matters
        c.attn(qs, os, qm::NH, qm::SCALE);
      }
    });
    // Batched: bulk kv_fill + one causal tiled attn_prefill.
    double batched = min_ms(3, 1, [&] {
      c.pos = 0;
      c.prefill(qs, ks, vs, os, T, qm::NH, qm::SCALE);
    });
    std::printf("  %6lld  %14.3f  %14.3f  %7.1fx\n", (long long)T, per_tok,
                batched, per_tok / batched);
    c.destroy();
    for (void* p : {ks, vs, qs, os}) cu::release(p, 0, nullptr);
  }
  return 0;
}
