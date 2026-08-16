// M9 decode ctx-scaling census. bench_qwen_decode measures decode at a fixed
// (short) position; real generation shows ~486 tok/s at ctx~40 falling to ~335
// tok/s by ctx~280, so the per-token cost is NOT ctx-independent. The weight
// GEMVs are (they read the same bytes every token), so the ctx-dependent part
// must be attention over the growing KV cache — plus whatever per-token host
// work scales with nothing at all (constant, hence a shrinking share).
//
// Two measurements:
//   1. per-position decode curve — one long greedy run, per-token wall clock
//      bucketed by position, for both the imperative and the CUDA-graph paths.
//   2. attention attribution — the isolated cost of the 24 per-layer attn calls
//      at each ctx, host path (split-KV, S>1 when ctx>=256) vs captured path
//      (attn_dpos: split-KV on a capacity-static grid, work bounded by *d_pos).
//      The dpos column must now TRACK the host column at every ctx — a growing
//      ratio would mean the device-side chunk heuristic drifted from the host's.
//      (Below ctx 256 the host path is a single kernel while dpos always
//      submits split+combine, so those rows carry a constant ~2x-launch offset
//      — irrelevant in production, where the graph replays the pair as one
//      submit. The drift signal is the trend, not the small-ctx constant.)
//
// Usage: bench_qwen_ctx [model.gguf] [max_ctx] [bucket]
//   e.g. bench_qwen_ctx ~/models/qwen2.5-0.5b-instruct-fp16.gguf 2048 256

#include "qwen_model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qm = qwenmodel;
namespace cu = tl::cuda;
static double now_ms() { return qm::StepProf::now_ms(); }

// Greedy-decode `n` tokens from a fresh cache, recording each token's wall
// clock. `captured` selects the CUDA-graph path (capture once after the short
// prefill, replay per token) over the imperative one.
static std::vector<double> decode_curve(qm::Model& M, int64_t n, bool captured) {
  qm::reset_cache(M);
  int64_t pos = 0, next = 0;
  for (int id : qm::PREFILL_IDS) next = qm::step_imperative(M, id, pos++);
  std::vector<double> per_tok;
  per_tok.reserve(n);
  if (!captured) {
    for (int64_t i = 0; i < n; i++) {
      double t = now_ms();
      next = qm::step_imperative(M, next, pos);
      pos++;
      per_tok.push_back(now_ms() - t);
    }
    return per_tok;
  }
  qm::captured_decoder dec;
  dec.init(M, next, pos);
  if (!dec.ok()) { dec.destroy(); return {}; }
  for (int64_t i = 0; i < n; i++) {
    double t = now_ms();
    int64_t id = dec.step(M, next);
    per_tok.push_back(now_ms() - t);
    if (id < 0) break;
    next = id;
  }
  qm::set_cache_pos(M, dec.cur_pos);  // host bookkeeping (device pos advanced)
  dec.destroy();
  return per_tok;
}

static void print_curve(const char* name, const std::vector<double>& c,
                        int64_t bucket) {
  if (c.empty()) { std::printf("  %-12s unavailable\n", name); return; }
  std::printf("  %s:\n", name);
  for (size_t b = 0; b * bucket < c.size(); b++) {
    size_t lo = b * bucket, hi = std::min(c.size(), lo + (size_t)bucket);
    double s = 0;
    for (size_t i = lo; i < hi; i++) s += c[i];
    double ms = s / (hi - lo);
    std::printf("    ctx %5zu-%-5zu  %6.3f ms/tok  %6.1f tok/s   %s\n", lo, hi - 1,
                ms, 1000.0 / ms, b == 0 ? "(reference)" : "");
  }
  double first = 0, last = 0;
  size_t nb = std::min((size_t)bucket, c.size());
  for (size_t i = 0; i < nb; i++) first += c[i];
  for (size_t i = c.size() - nb; i < c.size(); i++) last += c[i];
  std::printf("    -> %.1f%% slower at the end than at the start\n\n",
              100.0 * (last / first - 1.0));
}

int main(int argc, char** argv) {
  if (!tl::gpu_available()) { std::printf("no CUDA device — skipping\n"); return 0; }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  int64_t max_ctx = argc > 2 ? std::atoll(argv[2]) : 2048;
  int64_t bucket = argc > 3 ? std::atoll(argv[3]) : 256;

  qm::gg::model m(path);
  if (!qm::check_config(m)) { std::printf("config mismatch\n"); return 2; }
  tl::use_gpu();
  qm::Model M = qm::build(m, tl::dtype::bf16);
  std::printf("model %s | bf16 row | decode to ctx %lld, bucket %lld\n\n",
              path.c_str(), (long long)max_ctx, (long long)bucket);

  // Pre-size the host split-KV scratch at the largest ctx: attn_scratch_ grows
  // (a MemFree sync + realloc) as S steps up across ctx 256..max, and those
  // reallocs would otherwise land inside timed tokens of the imperative curve.
  qm::set_cache_pos(M, max_ctx);
  M.layers[0].cache.attn(M.scratch.qb, M.scratch.ab, qm::NH, qm::SCALE);
  cu::flush();

  // ---- 1. per-position decode curve.
  std::printf("=== decode cost vs context length (per-token wall clock) ===\n");
  print_curve("imperative", decode_curve(M, max_ctx, false), bucket);
  print_curve("CUDA graph", decode_curve(M, max_ctx, true), bucket);

  // ---- 2. attention attribution. The cache now holds `max_ctx` real tokens
  // from the run above (no uninitialized KV), so we can replay the 24 per-layer
  // attn calls at any pos <= that and time them in isolation.
  std::printf("=== isolated attention: 24 layers x 1 decode step ===\n");
  std::printf("  (host = split-KV when ctx>=256; dpos = capacity-static split grid)\n");
  void* d_pos = cu::alloc(4, nullptr);
  // Value guard alongside the timing: at each ctx, layer 0's host-path output
  // vs the dpos-path output must be BIT-IDENTICAL (the dpos kernel replicates
  // the host chunk heuristic; empty splits combine as exact zeros). This is the
  // long-ctx complement to bench_qwen_decode's dpos-vs-hostpos check (pos 32,
  // which never leaves the S=1 regime).
  float *ref_host = nullptr, *got_host = nullptr;
  void* refb = cu::alloc(qm::NH * qm::HD * 4, &ref_host);
  void* gotb = cu::alloc(qm::NH * qm::HD * 4, &got_host);
  const int REPS = 50;
  bool fail = false;
  std::printf("  %6s  %10s  %10s  %8s  %s\n", "ctx", "host(ms)", "dpos(ms)",
              "dpos/host", "bit-equal");
  for (int64_t ctx : {(int64_t)64, (int64_t)128, (int64_t)256, (int64_t)512,
                      (int64_t)1024, (int64_t)2048}) {
    if (ctx > max_ctx) break;
    qm::set_cache_pos(M, ctx);
    // attn() reads ctx = pos; attn_dpos() reads ctx = *d_pos + 1. Same ctx.
    cu::upload_u32(d_pos, (unsigned)(ctx - 1));
    auto& L0 = M.layers[0];
    L0.cache.attn(M.scratch.qb, refb, qm::NH, qm::SCALE);
    cu::sync_to_host(refb, false);
    L0.cache.attn_dpos(M.scratch.qb, gotb, qm::NH, d_pos, qm::SCALE);
    cu::sync_to_host(gotb, false);
    bool bit_eq = std::memcmp(ref_host, got_host, qm::NH * qm::HD * 4) == 0;
    auto time_24x = [&](auto&& attn1) {  // REPS x 24-layer attn -> ms per rep
      double t = now_ms();
      for (int i = 0; i < REPS; i++)
        for (auto& L : M.layers) attn1(L);
      cu::flush();
      return (now_ms() - t) / REPS;
    };
    auto host1 = [&](qm::Layer& L) {
      L.cache.attn(M.scratch.qb, M.scratch.ab, qm::NH, qm::SCALE);
    };
    auto dpos1 = [&](qm::Layer& L) {
      L.cache.attn_dpos(M.scratch.qb, M.scratch.ab, qm::NH, d_pos, qm::SCALE);
    };
    double host_ms = 1e30, dpos_ms = 1e30;
    for (int r = 0; r < 3; r++) {  // min of 3, host/dpos interleaved (WSL2 noise)
      host_ms = std::min(host_ms, time_24x(host1));
      dpos_ms = std::min(dpos_ms, time_24x(dpos1));
    }
    std::printf("  %6lld  %10.3f  %10.3f  %8.2fx  %s\n", (long long)ctx, host_ms,
                dpos_ms, dpos_ms / host_ms, bit_eq ? "YES" : "NO  <-- FAIL");
    if (!bit_eq) fail = true;
  }
  cu::release(d_pos, 0, nullptr);
  cu::release(refb, 0, nullptr);
  cu::release(gotb, 0, nullptr);
  return fail ? 1 : 0;
}
