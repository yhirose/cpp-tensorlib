// M9 prefill census — the prompt-side twin of bench_qwen_ctx (which measured
// decode). A prompt is not a decode: its T tokens share one set of weights, so
// feeding them one at a time throws away the batch dimension entirely. This
// bench is what sized that (~25x on the projections, ~45x on attention) before
// the batched path was built, and is now what guards and measures it:
//
//   0. Correctness. Every prefill path must produce the same first generated
//      token as the array `step` reference, at several chunk sizes so the chunk
//      seam (rope pos0, cache append offset, causal bound ACROSS chunks) is
//      exercised rather than assumed.
//   1. Throughput of each path at a range of prompt lengths: the array path
//      chat_qwen used to run, token-by-token imperative, token-by-token
//      CUDA-graph replay, and the shipped BATCHED path.
//   2. Where the batched win comes from, isolated: (a) the projections, M=1
//      GEMV vs one M=T GEMM on the SAME bf16 [N,K] weight — which doubles as
//      the GEMM's correctness check against the validated GEMV; (b) attention,
//      T growing-ctx decode calls vs one tiled causal attn_prefill.
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

enum class Path { Array, Imperative, Captured, Batched };

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
    // The COUNTERFACTUAL: what the prompt costs replayed one token at a time
    // through the captured graph. captured_decoder::begin no longer does this
    // (it batches), so drive feed/step directly; capture is excluded, being a
    // fixed cost rather than per-token work. Also an upper bound in one respect:
    // every replay runs lm_head, which a prompt needs only for its last token.
    qm::captured_decoder dec;
    dec.init(M, ids[0], 0);
    if (!dec.ok()) { dec.destroy(); return -1; }
    double t = now_ms();
    for (int64_t i = 0; i + 1 < T; i++) dec.feed(M, ids[(size_t)i]);
    dec.step(M, ids.back());
    double ms = now_ms() - t;
    dec.destroy();
    return ms;
  }
  if (p == Path::Batched) {
    // The SHIPPED path: what captured_decoder::begin runs for the prompt.
    double t = now_ms();
    bool ok = qm::prefill_batched(M, ids) >= 0;
    return ok ? now_ms() - t : -1;
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

  // ---- 0. Correctness: every prefill path must leave the same state as feeding
  // the prompt through the array `step` reference — same first generated token,
  // which is the contract chat_qwen relies on. The greedy token is the gate; the
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
    std::printf("=== 0. every prefill path vs the array reference (T=%lld) ===\n",
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
    bool ok = compare("begin", cap.begin(M, ids));
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

    // And the BATCHED prefill, at a chunk small enough that the prompt spans
    // several chunks — so the chunk seam (rope pos0, cache append offset, causal
    // bound across chunks) is exercised, not just a single whole-prompt pass.
    for (int64_t ch : {(int64_t)16, (int64_t)64, (int64_t)256}) {
      qm::reset_cache(M);
      char nm[24];
      std::snprintf(nm, sizeof nm, "batch/%lld", (long long)ch);
      ok &= compare(nm, qm::prefill_batched(M, ids, ch));
    }
    std::printf("\n");
    if (!ok) return 1;
  }

  // ---- 1. Throughput per path vs prompt length. The array path costs
  // ~10ms/token, so sweeping it to max_T would dominate the bench's runtime for
  // a number that is already flat by T=512; it is capped here and the cap is
  // reported rather than silently applied.
  const int64_t ARRAY_MAX = 512;
  std::printf("=== prefill throughput by path ===\n");
  std::printf("  %6s  %11s  %11s  %11s  %11s   (tok/s)\n", "T", "array",
              "imperative", "graph 1tok", "BATCHED");
  for (int64_t T : {(int64_t)32, (int64_t)128, (int64_t)512, (int64_t)1024,
                    (int64_t)2048}) {
    if (T > max_T) break;
    double a = T <= ARRAY_MAX ? prefill_ms(M, T, Path::Array) : -1;
    double i = prefill_ms(M, T, Path::Imperative);
    double g = prefill_ms(M, T, Path::Captured);
    double b = prefill_ms(M, T, Path::Batched);
    char ab[32];
    if (a > 0) std::snprintf(ab, sizeof ab, "%.1f", T * 1000.0 / a);
    else std::snprintf(ab, sizeof ab, "(skipped)");
    std::printf("  %6lld  %11s  %11.1f  %11.1f  %11.1f\n", (long long)T, ab,
                T * 1000.0 / i, g > 0 ? T * 1000.0 / g : 0.0,
                b > 0 ? T * 1000.0 / b : 0.0);
  }
  std::printf("  (array capped at T=%lld — ~10ms/token, and already flat there)\n\n",
              (long long)ARRAY_MAX);

  // ---- 2a. The projection lever, now real: one M=1 GEMV per prompt token
  // (what every token-by-token path runs) vs one M=T bf16 GEMM for the chunk.
  // Both consume the SAME [N,K] row-major bf16 weight, so this is a like-for-
  // like comparison, not a proxy. gemv_bf16_row is the validated reference for
  // the correctness column (reduction order differs, so agreement is to ~1e-6,
  // not bit-exact).
  std::printf("=== 2a. projection GEMV(M=1) vs batched GEMM (per prompt token) ===\n");
  const int64_t MB = qm::PREFILL_CHUNK;  // the chunk production runs
  struct Shape { const char* name; int64_t K, N; const tl::array* w; };
  qm::Layer& L0 = M.layers[0];
  const Shape shapes[] = {
      {"QKV.fused", qm::NE, (qm::NH + 2 * qm::NKV) * qm::HD, &L0.wqkv},
      {"wo", qm::NH * qm::HD, qm::NE, &L0.wo_row},
      {"gateup.fused", qm::NE, 2 * qm::FF, &L0.wgu},
      {"wd", qm::FF, qm::NE, &L0.wd_row},
  };
  int64_t maxK = 0, maxN = 0;
  for (const Shape& s : shapes) {
    maxK = std::max(maxK, s.K);
    maxN = std::max(maxN, s.N);
  }
  // One max-sized trio reused by every shape — cu::release recycles into a
  // size-keyed pool rather than returning memory to the driver, so per-shape
  // buffers would stay committed for the whole run.
  float *ha = nullptr, *hy = nullptr, *hy1 = nullptr;
  void* a = cu::alloc(MB * maxK * 4, &ha);   // [M,K] activations
  void* y = cu::alloc(MB * maxN * 4, &hy);
  void* y1 = cu::alloc(maxN * 4, &hy1);      // GEMV reference row
  if (!a || !y || !y1) { std::printf("  alloc failed\n"); return 1; }
  uint32_t st = 20260816u;
  for (int64_t i = 0; i < MB * maxK; i++) {  // deterministic non-zero input
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    ha[i] = (int32_t)st * (1.0f / 2147483648.0f);
  }
  cu::sync_to_host(a, true);

  std::printf("  %14s  %6s %7s  %10s  %10s  %8s  %9s\n", "shape", "K", "N",
              "gemv(us)", "gemm(us)", "speedup", "vs gemv");
  double sum_gemv = 0, sum_gemm = 0;
  bool gemm_ok = true;
  for (const Shape& s : shapes) {
    // Warm both paths (module load, first-touch upload) outside the timing.
    cu::gemv_bf16_row(a, s.w->native(), y1, s.N, s.K);
    cu::gemm_bf16_nt(a, s.w->native(), y, MB, s.N, s.K);
    cu::flush();
    // Correctness: GEMM row 0 must reproduce the GEMV of A's row 0.
    cu::sync_to_host(y1, false);
    cu::sync_to_host(y, false);
    double maxrel = 0;
    for (int64_t i = 0; i < s.N; i++)
      maxrel = std::max(maxrel, (double)std::fabs(hy[i] - hy1[i]) /
                                    (1.0 + std::fabs(hy1[i])));
    bool ok = maxrel < 1e-5;
    gemm_ok &= ok;

    double gemv_us = 1000.0 * min_ms(3, 20, [&] {
      cu::gemv_bf16_row(a, s.w->native(), y1, s.N, s.K);
    });
    double gemm_us = 1000.0 * min_ms(3, 5, [&] {
      cu::gemm_bf16_nt(a, s.w->native(), y, MB, s.N, s.K);
    }) / MB;  // per prompt token
    sum_gemv += gemv_us;
    sum_gemm += gemm_us;
    std::printf("  %14s  %6lld %7lld  %10.2f  %10.2f  %7.1fx  %.1e %s\n", s.name,
                (long long)s.K, (long long)s.N, gemv_us, gemm_us,
                gemv_us / gemm_us, maxrel, ok ? "OK" : "FAIL");
  }
  for (void* p : {a, y, y1}) cu::release(p, 0, nullptr);
  std::printf("  %14s  %6s %7s  %10.2f  %10.2f  %7.1fx  (x24 layers)\n", "SUM/layer",
              "", "", sum_gemv, sum_gemm, sum_gemv / sum_gemm);
  std::printf("  (the gemv column repeats one shape back-to-back, so QKV/wo — "
              "whose weights fit L2 — read warm; their speedup is a floor)\n\n");
  if (!gemm_ok) return 1;

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
