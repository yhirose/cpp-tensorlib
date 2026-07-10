// M9 decode gemv census: WHY does the in-context decode gemv run at ~220 GB/s
// effective when bench_bf16_gemv hits ~936 GB/s? That bench measures llama-7B
// shapes (K=4096, N=12288); Qwen2.5-0.5B decode shapes are far smaller (K=896,
// N∈{128,896,4864,...}) and heavily split-K'd. This isolates the small-shape
// question two ways:
//
//   1. per-shape GB/s at the REAL Qwen decode shapes, split-K auto vs forced OFF
//      (set_no_splitk) — attributes the loss to occupancy vs the atomic combine;
//   2. a back-to-back "layer" sequence (the 7 gemvs of one decoder layer run
//      consecutively, one flush at the end) vs the sum of isolated medians —
//      attributes any extra loss to inter-kernel cadence (launch gap / no overlap).
//
// It also measures the two FUSION candidates (QKV → N=1152, gate+up → N=9728):
// larger N should let the grid fill the SMs and (for K=896) drop split-K entirely.
// Isolated + short (WSL2 sysmem-cliff discipline): buffers released per shape;
// max resident is lm_head (~272MB), far under the ~2GB cliff.
//
// Usage: bench_qwen_gemv   (no args; uses the compiled Qwen0.5B decode shapes)

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tl::cuda;
using clk = std::chrono::steady_clock;

static uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// One bf16 gemv operand set: a(1,K) @ B[K,N] -> y(N), B stored bf16.
struct Op {
  const char* name;
  int64_t K, N;
  void* a = nullptr;
  void* B = nullptr;
  void* y = nullptr;
  void alloc_() {
    float* ha = nullptr;
    float* hB = nullptr;  // host mirror (bytes); gemv's device_read_ uploads it
    a = alloc(K * 4, &ha);
    B = alloc(K * N * 2, &hB);  // bf16 weights (half the bytes)
    y = alloc(N * 4, nullptr);
    uint32_t st = 12345u;
    auto rnd = [&] {
      st ^= st << 13; st ^= st >> 17; st ^= st << 5;
      return static_cast<int32_t>(st) * (1.0f / 2147483648.0f);
    };
    for (int64_t i = 0; i < K; i++) ha[i] = rnd();
    uint16_t* Bb = reinterpret_cast<uint16_t*>(hB);
    for (int64_t i = 0; i < K * N; i++) Bb[i] = f32_to_bf16(rnd());
  }
  void free_() {
    release(a, 0, nullptr); release(B, 0, nullptr); release(y, 0, nullptr);
  }
  void run() const { gemv_bf16(a, B, y, N, K); }
};

static const int R = 50, ROUNDS = 7;

static double time_op(const Op& op) {
  op.run(); flush();  // warmup
  std::vector<double> ms;
  for (int r = 0; r < ROUNDS; r++) {
    auto t0 = clk::now();
    for (int i = 0; i < R; i++) op.run();
    flush();
    ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
  }
  return median(ms);
}

int main() {
  if (!available()) { std::printf("no CUDA device — skipping\n"); return 0; }

  // Qwen2.5-0.5B decode gemv shapes (K=in, N=out). NE=896, NH*HD=896,
  // NKV*HD=128, FF=4864, VOCAB=151936; fused QKV N=(14+2+2)*64=1152, gate+up=2*FF.
  struct S { const char* name; int64_t K, N; } shapes[] = {
      {"wq  [896x896]", 896, 896},   {"wk  [896x128]", 896, 128},
      {"wv  [896x128]", 896, 128},   {"wo  [896x896]", 896, 896},
      {"wg  [896x4864]", 896, 4864}, {"wu  [896x4864]", 896, 4864},
      {"wd  [4864x896]", 4864, 896}, {"lm_head[896x151936]", 896, 151936},
      {"QKV.fused[896x1152]", 896, 1152},
      {"gateup.fused[896x9728]", 896, 9728},
  };

  std::printf("=== per-shape bf16 gemv (median of %d rounds x %d launches) ===\n", ROUNDS, R);
  std::printf("%-22s %10s %10s %8s   %10s %10s\n", "shape", "splitK ms",
              "noK ms", "splitK/noK", "splitK GB/s", "noK GB/s");
  for (const auto& s : shapes) {
    Op op{s.name, s.K, s.N};
    op.alloc_();
    set_no_splitk(false);
    double ms_k = time_op(op);
    set_no_splitk(true);
    double ms_n = time_op(op);
    set_no_splitk(false);
    double bytes = (double)s.K * s.N * 2;
    std::printf("%-22s %10.4f %10.4f %8.2f   %10.1f %10.1f\n", s.name, ms_k, ms_n,
                ms_k / ms_n, bytes / (ms_k * 1e6), bytes / (ms_n * 1e6));
    op.free_();
  }

  // Back-to-back layer sequence: the 7 real gemvs of one decoder layer run
  // consecutively (one flush per R-batch), vs the sum of their isolated medians.
  // A gap between the two = inter-kernel cadence loss (launch/ramp/tail, no
  // overlap) beyond per-shape occupancy. Measured with split-K auto (the decode
  // path). Weight bytes/token for one layer are summed for an effective GB/s.
  struct S layer[] = {
      {"wq", 896, 896}, {"wk", 896, 128}, {"wv", 896, 128}, {"wo", 896, 896},
      {"wg", 896, 4864}, {"wu", 896, 4864}, {"wd", 4864, 896},
  };
  std::vector<Op> ops;
  double sum_iso = 0, layer_bytes = 0;
  for (auto& s : layer) {
    ops.push_back(Op{s.name, s.K, s.N});
    ops.back().alloc_();
    layer_bytes += (double)s.K * s.N * 2;
  }
  set_no_splitk(false);
  for (auto& op : ops) sum_iso += time_op(op);
  // sequence timing
  for (auto& op : ops) op.run();
  flush();
  std::vector<double> seq;
  for (int r = 0; r < ROUNDS; r++) {
    auto t0 = clk::now();
    for (int i = 0; i < R; i++)
      for (auto& op : ops) op.run();
    flush();
    seq.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R);
  }
  double seq_ms = median(seq);
  std::printf("\n=== one-layer 7-gemv sequence (split-K auto) ===\n");
  std::printf("  sum of isolated medians  %8.4f ms   %8.1f GB/s\n", sum_iso,
              layer_bytes / (sum_iso * 1e6));
  std::printf("  back-to-back sequence    %8.4f ms   %8.1f GB/s\n", seq_ms,
              layer_bytes / (seq_ms * 1e6));
  std::printf("  cadence overhead         %8.4f ms   (%.1f%% vs isolated sum)\n",
              seq_ms - sum_iso, 100.0 * (seq_ms - sum_iso) / sum_iso);
  std::printf("  x24 layers -> %.4f ms/token of gemv (%.1f tok/s if gemv-only)\n",
              seq_ms * 24, 1000.0 / (seq_ms * 24));
  for (auto& op : ops) op.free_();
  return 0;
}
