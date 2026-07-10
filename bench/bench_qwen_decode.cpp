// M9 decode-overhead census (roadmap M9 "real-model chat" speed work). Loads the
// Qwen2.5-0.5B GGUF, runs a short greedy decode, and reports the per-region
// wall-clock breakdown of one decode step (StepProf) to locate the 6.7x gap to
// llama.cpp. Isolated + short (WSL2 sysmem-cliff discipline): weights loaded
// once, a fixed small number of decode steps, no diverse-shape churn.
//
// Usage: bench_qwen_decode [model.gguf] [f32|bf16] [n_decode]
//   Reports ms/token and the embed / qkv-eval(sync) / x2-eval(sync) /
//   logits-eval / logits-D2H / host-argmax split, plus tok/s.

#include "qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

namespace qm = qwenmodel;

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no CUDA device — skipping\n");
    return 0;
  }
  std::string path = argc > 1 ? argv[1] : qm::default_path();
  tl::dtype wdt = (argc > 2 && std::string(argv[2]) == "f32") ? tl::dtype::f32
                                                              : tl::dtype::bf16;
  int64_t n_dec = argc > 3 ? std::atoll(argv[3]) : 64;

  qm::gg::model m(path);
  if (!qm::check_config(m)) { std::printf("config mismatch\n"); return 2; }
  tl::use_gpu();
  std::printf("model %s | weights %s | %lld decode steps\n", path.c_str(),
              wdt == tl::dtype::bf16 ? "bf16" : "f32", (long long)n_dec);
  qm::Model M = qm::build(m, wdt);

  // Short deterministic prefill so the cache/pos are realistic, then time a
  // fixed run of decode steps. A few warmup steps first (module load, fn cache,
  // first-touch allocs) so the census reflects steady state.
  const int prompt[] = {40, 2610, 264, 3974};  // arbitrary valid ids
  int64_t pos = 0;
  std::vector<float> logits;
  for (int i = 0; i < 4; i++) logits = qm::step(M, prompt[i], pos++);

  int64_t next = qm::argmax(logits);
  for (int i = 0; i < 16; i++) next = qm::step_greedy(M, next, pos++);  // warmup

  // Correctness guard: the imperative step must produce the SAME greedy token as
  // the array path from identical cache state. Snapshot each cache's pos, run
  // the array step_greedy, restore pos (so the imperative step overwrites the
  // same cache slot with identical k,v), run step_imperative, compare.
  {
    int mism = 0;
    int64_t p = pos, nx = next;
    std::vector<int64_t> saved(M.layers.size());
    for (int i = 0; i < 12; i++) {
      for (size_t j = 0; j < M.layers.size(); j++) saved[j] = M.layers[j].cache.pos;
      int64_t ta = qm::step_greedy(M, nx, p);
      for (size_t j = 0; j < M.layers.size(); j++) M.layers[j].cache.pos = saved[j];
      int64_t tb = qm::step_imperative(M, nx, p);
      if (ta != tb) mism++;
      nx = ta; p++;
    }
    std::printf("array-vs-imperative greedy: %d/12 mismatch %s\n", mism,
                mism == 0 ? "(EXACT)" : "(!!)");
    pos = p; next = nx;
  }

  // Robust timing: WSL2 GPU boost-clock state adds ±20 tok/s of run-to-run
  // noise, so a single long run can't resolve a ~10% overhead change. Take the
  // MIN ms/tok over several short rounds — min = the least-perturbed / fully-
  // boosted state, the stable ceiling to measure optimizations against.
  // mode: 0=array host-argmax, 1=array gpu-argmax, 2=imperative.
  auto round = [&](int mode, int64_t steps) {
    int64_t p = pos, nx = next;
    auto tb = clk::now();
    for (int64_t i = 0; i < steps; i++) {
      if (mode == 2) nx = qm::step_imperative(M, nx, p++);
      else if (mode == 1) nx = qm::step_greedy(M, nx, p++);
      else { auto lg = qm::step(M, nx, p++); nx = qm::argmax(lg); }
    }
    return ms_since(tb) / steps;
  };
  const int ROUNDS = 6;
  double host_min = 1e9, gpu_min = 1e9, imp_min = 1e9;
  for (int r = 0; r < ROUNDS; r++) {
    host_min = std::min(host_min, round(0, n_dec));
    gpu_min = std::min(gpu_min, round(1, n_dec));
    imp_min = std::min(imp_min, round(2, n_dec));
  }
  std::printf("\n=== decode timing (min of %d rounds x %lld steps) ===\n", ROUNDS, (long long)n_dec);
  std::printf("  array host-argmax  %6.3f ms/tok   %5.1f tok/s\n", host_min, 1000.0 / host_min);
  std::printf("  array gpu-argmax   %6.3f ms/tok   %5.1f tok/s\n", gpu_min, 1000.0 / gpu_min);
  std::printf("  imperative         %6.3f ms/tok   %5.1f tok/s\n\n", imp_min, 1000.0 / imp_min);

  // CUDA-graph capture POC (C1-2): capture the imperative forward once at a
  // fixed pos and replay it. Numerically meaningless (pos is baked), but it
  // measures the LAUNCH-OVERHEAD CEILING — replay+argmax vs ~19 launches+argmax.
  namespace cu = tl::cuda;
  if (!cu::graph_available()) {
    std::printf("graph POC: CUDA-graph symbols unavailable — skipped\n");
  } else {
    qm::stage_embed(M, next);
    qm::run_layers_(M, M.scratch.embedb, pos);  // warm (grow attn scratch, resident weights)
    cu::flush();
    if (!cu::capture_begin()) {
      std::printf("graph POC: capture_begin failed\n");
    } else {
      qm::run_layers_(M, M.scratch.embedb, pos);
      auto exec = cu::capture_end();
      if (!exec) {
        std::printf("graph POC: capture/instantiate failed\n");
      } else {
        cu::graph_launch(exec); cu::flush();  // warm the exec
        double graph_min = 1e9;
        int64_t idx = 0;
        for (int r = 0; r < ROUNDS; r++) {
          auto tb = clk::now();
          for (int64_t i = 0; i < n_dec; i++) {
            cu::graph_launch(exec);
            cu::argmax(M.scratch.logitsb, qm::VOCAB, &idx);
          }
          graph_min = std::min(graph_min, ms_since(tb) / n_dec);
        }
        std::printf("=== graph-capture CEILING (fixed-pos replay + argmax) ===\n");
        std::printf("  graph replay       %6.3f ms/tok   %5.1f tok/s   (vs imperative %.1f)\n\n",
                    graph_min, 1000.0 / graph_min, 1000.0 / imp_min);
        cu::graph_destroy(exec);
      }
    }
  }

  // Census run (per-region breakdown; prof timers add a little host overhead,
  // so read the %s, not the absolute total, against the timing above).
  qm::StepProf prof;
  double argmax_ms = 0.0;
  auto t0 = clk::now();
  for (int64_t i = 0; i < n_dec; i++) {
    logits = qm::step(M, next, pos++, nullptr, nullptr, &prof);
    auto ta = clk::now();
    next = qm::argmax(logits);
    argmax_ms += ms_since(ta);
  }
  double total_ms = ms_since(t0);

  double n = (double)prof.n;
  double per_tok = total_ms / n;
  std::printf("\n=== decode census (%lld steps, per-token averages) ===\n", (long long)prof.n);
  std::printf("  total          %8.3f ms/tok   (%.1f tok/s)\n", per_tok, 1000.0 / per_tok);
  std::printf("  embed (H2D)    %8.3f ms   %5.1f%%\n", prof.embed / n, 100 * prof.embed / total_ms);
  std::printf("  construct(host)%8.3f ms   %5.1f%%  (array-graph build, ~30 nodes/layer)\n", prof.construct / n, 100 * prof.construct / total_ms);
  std::printf("  cache(append+attn)%5.3f ms   %5.1f%%  (2 async launch/layer)\n", prof.cache / n, 100 * prof.cache / total_ms);
  std::printf("  qkv-eval(sync) %8.3f ms   %5.1f%%  (3 sync/layer x24 + qkv launches)\n", prof.qkv_eval / n, 100 * prof.qkv_eval / total_ms);
  std::printf("  x2-eval(sync)  %8.3f ms   %5.1f%%  (1 sync/layer x24 + o/mlp launches)\n", prof.x2_eval / n, 100 * prof.x2_eval / total_ms);
  std::printf("  logits-eval    %8.3f ms   %5.1f%%\n", prof.logits_eval / n, 100 * prof.logits_eval / total_ms);
  std::printf("  logits-D2H     %8.3f ms   %5.1f%%  (608KB VOCAB copy)\n", prof.logits_d2h / n, 100 * prof.logits_d2h / total_ms);
  std::printf("  host-argmax    %8.3f ms   %5.1f%%  (151936-elem loop)\n", argmax_ms / n, 100 * argmax_ms / total_ms);
  double accounted = prof.embed + prof.construct + prof.cache + prof.qkv_eval + prof.x2_eval + prof.logits_eval + prof.logits_d2h + argmax_ms;
  std::printf("  (unaccounted)  %8.3f ms   %5.1f%%\n", (total_ms - accounted) / n, 100 * (total_ms - accounted) / total_ms);
  std::printf("\nsync/token ~= 24*4 + 2 = 98 CtxSynchronize; launches/token ~= 800\n");
  return 0;
}
