// Local-LLM shape bench — the dev harness M7-M9 kernels are measured in.
// Fast-iterating (fixed launch count + median, seconds total): drives the
// GEMM/attention shapes a real decoder layer hits, at the two regimes that
// matter for the consumer local-LLM scope (docs/roadmap.md "Target scope"):
//
//   decode  (T=1)   batch~1 GEMV, MEMORY-BANDWIDTH-bound — the dominant
//                   interactive cost; where cuBLAS has no moat and M8 int4
//                   dequant-matmul / M9 fused attention are the real levers.
//   prefill (T=512) compute-bound GEMM — the one-time prompt cost; the M6
//                   SGEMM foundation.
//
// Backend-agnostic (array API through the gpu:: facade): runs on CUDA off-Apple,
// Metal on Apple, CPU otherwise — same measurement discipline as census.cpp
// (warmup + interleaved medians). Reports per-shape ms / GFLOP/s / GB/s and an
// aggregate decode tokens/sec estimate for the whole model — the metric that
// maps to the llama.cpp/exllamav2 scoreboard once M7-M9 land. It measures the
// F32 baseline today; BF16 (M7) / int4 (M8) / flash-attn (M9) kernels drop into
// the same shapes and the same numbers show whether they helped.

#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

// llama-2-7B geometry (the reference decoder layer). Change here to profile a
// different model; shapes derive from these.
constexpr int64_t kD = 4096;      // hidden size
constexpr int64_t kFfn = 11008;   // FFN intermediate (SwiGLU: gate+up = 2*)
constexpr int64_t kHeads = 32;    // attention heads (MHA; GQA would shrink KV)
constexpr int64_t kHdim = 128;    // head dim (kD / kHeads)
constexpr int64_t kVocab = 32000;
constexpr int64_t kLayers = 32;

// Cheap xorshift fill — values are irrelevant to timing (this bench measures
// throughput, not correctness), and mt19937 over 100M-element weight matrices
// (lm_head is 4096x32000) would dominate wall time. ~5x faster setup.
tl::array rnd(tl::shape_t shape, unsigned seed) {
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1;
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = static_cast<int32_t>(s) * (1.0f / 2147483648.0f);  // ~[-1,1)
  }
  return tl::array::from(std::move(v), std::move(shape));
}

double median_ms(const std::function<void()>& fn, int trials = 11,
                 int warmup = 3) {
  for (int i = 0; i < warmup; i++) fn();
  std::vector<double> ts;
  for (int i = 0; i < trials; i++) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

struct gemm_shape {
  const char* name;
  int64_t M, N, K;
  bool per_layer;  // summed into the per-layer latency (lm_head runs once)
};

// Time one C(M,N) = A(M,K)·B(K,N) on the current device; return median ms.
double time_gemm(int64_t M, int64_t N, int64_t K, unsigned seed) {
  auto a = rnd({M, K}, seed);
  auto b = rnd({K, N}, seed + 1);
  return median_ms([&] { a.dot(b).eval(); });
}

// Same, with bf16 weights (M7): decode (M=1) hits the native bf16 GEMV on
// CUDA; other shapes/backends widen-fallback (correct but not the fast path).
double time_gemm_bf16(int64_t M, int64_t N, int64_t K, unsigned seed) {
  auto a = rnd({M, K}, seed);
  auto b = rnd({K, N}, seed + 1).to_bf16();
  return median_ms([&] { a.dot(b).eval(); });
}

// A decoder layer's GEMMs at row count T (T=1 decode, T=seq prefill). lm_head
// always predicts one token (last position) even during prefill.
double bench_gemms(const char* regime, int64_t T) {
  const gemm_shape shapes[] = {
      {"qkv_proj", T, 3 * kD, kD, true},        // fused q,k,v projection
      {"attn_out", T, kD, kD, true},            // output projection
      {"ffn_gate_up", T, 2 * kFfn, kD, true},   // SwiGLU gate+up fused
      {"ffn_down", T, kD, kFfn, true},          // down projection
      {"lm_head", 1, kVocab, kD, false},        // logits for next token only
  };
  std::printf("-- %s (T=%lld) --\n", regime, (long long)T);
  std::printf("  %-14s %10s %10s %10s\n", "shape", "ms", "GFLOP/s", "GB/s");
  double layer_ms = 0;
  unsigned seed = 100;
  for (const auto& s : shapes) {
    double ms = time_gemm(s.M, s.N, s.K, seed += 2);
    double gf = 2.0 * s.M * s.N * s.K / (ms * 1e6);
    // Weight-matrix bytes moved / time — the number that matters when M is tiny
    // and the op is bandwidth-bound (decode); f32 today, /2 with BF16, /8 int4.
    double gbs = static_cast<double>(s.K) * s.N * 4 / (ms * 1e6);
    std::printf("  %-14s %10.3f %10.1f %10.1f%s\n", s.name, ms, gf, gbs,
                s.per_layer ? "" : "  (x1)");
    if (s.per_layer) layer_ms += ms;
  }
  return layer_ms;  // GEMM ms per layer (attention added by caller for decode)
}

// Decode attention for one layer (all kHeads). Two ways:
//  unfused — the 3-op composite (q·Kᵀ, softmax, ·V) per head × kHeads;
//  fused   — the single M9 attn_decode op over [kHeads, ctx, kHdim].
// Returns {unfused_ms, fused_ms} per layer.
std::pair<double, double> bench_decode_attention(int64_t ctx) {
  // unfused: representative single head, × kHeads (mirrors the old estimate).
  auto q1 = rnd({1, kHdim}, 300);
  auto k1 = rnd({ctx, kHdim}, 301);
  auto v1 = rnd({ctx, kHdim}, 302);
  auto kt = k1.transpose();
  double unfused = median_ms([&] { q1.dot(kt).softmax().dot(v1).eval(); }) * kHeads;

  // fused: one op over all heads.
  auto q = rnd({kHeads, kHdim}, 310);
  auto k = rnd({kHeads, ctx, kHdim}, 311);
  auto v = rnd({kHeads, ctx, kHdim}, 312);
  float scale = 1.0f / std::sqrt((float)kHdim);
  double fused =
      median_ms([&] { tl::array::attn_decode(q, k, v, scale).eval(); });

  std::printf("-- decode attention (ctx=%lld) --\n", (long long)ctx);
  std::printf("  %-16s %10.3f ms/layer (unfused, 3-op × %lld heads)\n",
              "qk·softmax·v", unfused, (long long)kHeads);
  std::printf("  %-16s %10.3f ms/layer (fused attn_decode)   %.1fx\n",
              "attn_decode", fused, unfused / fused);
  return {unfused, fused};
}

}  // namespace

int main(int argc, char** argv) {
  int64_t prefill_T = 512;
  int64_t ctx = 2048;
  if (argc > 1) prefill_T = std::atoll(argv[1]);
  if (argc > 2) ctx = std::atoll(argv[2]);

  const bool gpu = tl::gpu_available();
  if (gpu)
    tl::use_gpu();
  else
    tl::use_cpu();
  std::printf("tensorlib LLM shape bench — %s backend, llama-7B geometry\n",
              gpu ? "GPU" : "CPU");
  std::printf("d=%lld ffn=%lld heads=%lld hdim=%lld vocab=%lld layers=%lld\n\n",
              (long long)kD, (long long)kFfn, (long long)kHeads,
              (long long)kHdim, (long long)kVocab, (long long)kLayers);

  // Decode: the money regime. Per-layer GEMM + attention → whole-model t/s.
  double decode_layer = bench_gemms("decode", 1);
  auto [attn_unfused, attn_fused] = bench_decode_attention(ctx);
  double lm_head_ms = time_gemm(1, kVocab, kD, 500);
  double per_token_unfused =
      (decode_layer + attn_unfused) * kLayers + lm_head_ms;
  double per_token_ms = (decode_layer + attn_fused) * kLayers + lm_head_ms;
  std::printf(
      "\n  => decode ~%.2f ms/token  ~%.1f tok/s  (F32, fused attn)\n",
      per_token_ms, 1000.0 / per_token_ms);
  std::printf(
      "     unfused attn would be ~%.2f ms/token (~%.1f tok/s) — fusion %.2fx\n",
      per_token_unfused, 1000.0 / per_token_unfused,
      per_token_unfused / per_token_ms);

  // Same decode GEMMs with bf16 weights (M7 storage dtype; attention stays
  // F32 until M9). Halved weight traffic on the bandwidth-bound GEMV.
  {
    std::printf("\n-- decode, bf16 weights (T=1) --\n");
    std::printf("  %-14s %10s %10s\n", "shape", "ms", "GB/s(bf16)");
    struct { const char* name; int64_t N, K; bool per_layer; } ws[] = {
        {"qkv_proj", 3 * kD, kD, true},
        {"attn_out", kD, kD, true},
        {"ffn_gate_up", 2 * kFfn, kD, true},
        {"ffn_down", kD, kFfn, true},
        {"lm_head", kVocab, kD, false},
    };
    double layer_bf16 = 0, lm_bf16 = 0;
    unsigned seed = 700;
    for (const auto& s : ws) {
      double ms = time_gemm_bf16(1, s.N, s.K, seed += 2);
      double gbs = static_cast<double>(s.K) * s.N * 2 / (ms * 1e6);
      std::printf("  %-14s %10.3f %10.1f%s\n", s.name, ms, gbs,
                  s.per_layer ? "" : "  (x1)");
      if (s.per_layer) layer_bf16 += ms;
      else lm_bf16 = ms;
    }
    double per_tok_bf16 = (layer_bf16 + attn_fused) * kLayers + lm_bf16;
    std::printf(
        "\n  => decode ~%.2f ms/token  ~%.1f tok/s  (bf16 weights + fused attn)\n",
        per_tok_bf16, 1000.0 / per_tok_bf16);
    std::printf("     vs F32: %.2fx tokens/sec\n", per_token_ms / per_tok_bf16);
  }
  std::printf("     (int4 dequant-matmul (M8) is the next lever)\n\n");

  // Prefill: compute-bound GEMM foundation.
  bench_gemms("prefill", prefill_T);

  if (gpu) tl::use_cpu();
  return 0;
}
