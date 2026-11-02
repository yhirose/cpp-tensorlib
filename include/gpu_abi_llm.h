#pragma once

// The llm layer's kernel params (the kop ids of TL_KOPS_LLM): the same ABI as
// gpu_abi.h's, a run of 4-byte fields in the order the kernel takes its scalars.
// Then the sampler's definition, which the kernels follow and the host runs.

#include <cmath>
#include <cstdint>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

struct kv_append_params {
  uint32_t pos, kv_stride;  // kv_stride = kv_max * D, elements a head
};
struct kv_fill_params {
  uint32_t T, kv_stride, pos0;
};

// ---- sampling a token from logits (gpu_ops_llm.h's sample_to) ----
//
// As Hugging Face's and vLLM's samplers: the logits divided by the
// temperature, the top_k largest kept, then the fewest of those whose
// probabilities (renormalized over the k) reach top_p, and one drawn from
// what is left. Temperature 0 is greedy (argmax). top_k 0 keeps the whole
// vocabulary, drawn exactly by Gumbel-max, and so needs top_p 1. Draw number
// `counter` of a stream `seed` is the same on every backend: the random bits
// are Philox4x32-10 (Random123's, as cuRAND, PyTorch and JAX use) of
// (counter, element) under (seed, 0).
struct sampling {
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  uint32_t seed = 0;
  bool greedy() const { return temperature == 0.0f; }
  bool valid() const;
};

namespace policy {
// The largest top_k: each group of a device sampler keeps that many
// candidates, picked one round at a time.
inline constexpr int32_t max_top_k = 64;
}  // namespace policy

inline bool sampling::valid() const {
  return std::isfinite(temperature) && temperature >= 0.0f &&
         (temperature == 0.0f || std::isfinite(1.0f / temperature)) && top_k >= 0 &&
         top_k <= policy::max_top_k && top_p > 0.0f && top_p <= 1.0f &&
         (top_k > 0 || top_p == 1.0f);
}

// Philox4x32-10 of counter c under key (k0, k1): the first of its four words.
inline uint32_t philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                       uint32_t k1, uint32_t out[4] = nullptr) {
  for (int r = 0; r < 10; r++) {
    const uint64_t p0 = uint64_t(0xD2511F53u) * c0, p1 = uint64_t(0xCD9E8D57u) * c2;
    const uint32_t n0 = uint32_t(p1 >> 32) ^ c1 ^ k0, n2 = uint32_t(p0 >> 32) ^ c3 ^ k1;
    c1 = uint32_t(p1);
    c3 = uint32_t(p0);
    c0 = n0;
    c2 = n2;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  if (out) out[0] = c0, out[1] = c1, out[2] = c2, out[3] = c3;
  return c0;
}
// The element a draw's own uniform is keyed by (elements are below 2^24).
inline constexpr uint32_t sample_pick_element = 0xFFFFFFFFu;
// 24 random bits as a float in [0, 1); or 23 of them, centred, in (0, 1) when
// `open` (x + 0.5 is exact below 2^23, so the largest is 1 - 2^-24, never 1).
inline float sample_uniform(uint32_t bits, bool open = false) {
  return open ? (float(bits >> 9) + 0.5f) * 0x1p-23f : float(bits >> 8) * 0x1p-24f;
}
// Element i's Gumbel noise for draw `counter`.
inline float sample_gumbel(uint32_t seed, uint32_t counter, uint32_t i) {
  return -std::log(-std::log(sample_uniform(philox(counter, i, 0, 0, seed, 0), true)));
}

// The argmax of a length-n vector, the smallest index on ties: the device
// kernels' scan as one loop. NaN is never taken; with nothing above -FLT_MAX, 0.
inline int64_t argmax_ref(const float* a, int64_t n) {
  float m = -3.402823466e+38f;
  int64_t best = 0;
  for (int64_t i = 0; i < n; i++)
    if (a[i] > m) m = a[i], best = i;
  return best;
}

// The token draw number `counter` takes, on the host: the definition the
// device samplers are held to. A NaN or -inf logit is never taken; with
// nothing else, 0.
inline int64_t sample_ref(const float* a, int64_t n, const sampling& s, uint32_t counter) {
  if (s.greedy()) return argmax_ref(a, n);
  const float inv_t = 1.0f / s.temperature;
  if (s.top_k == 0) {
    float m = -INFINITY;
    int64_t best = 0;
    for (int64_t i = 0; i < n; i++) {  // a NaN or -inf key is never above m
      const float key = a[i] * inv_t + sample_gumbel(s.seed, counter, uint32_t(i));
      if (key > m) m = key, best = i;
    }
    return best;
  }
  // The k largest, larger first and the smaller index first among equals,
  // each element put in its place among those kept so far.
  const auto before = [&](int64_t x, int64_t y) {
    return a[x] > a[y] || (a[x] == a[y] && x < y);
  };
  const auto K = size_t(s.top_k);
  int64_t c[policy::max_top_k];
  size_t k = 0;
  for (int64_t i = 0; i < n; i++) {
    if (!(a[i] > -INFINITY)) continue;  // NaN and -inf are never kept
    if (k == K && !before(i, c[k - 1])) continue;
    size_t j = k < K ? k++ : k - 1;
    for (; j > 0 && before(i, c[j - 1]); j--) c[j] = c[j - 1];
    c[j] = i;
  }
  if (k == 0) return 0;
  // Their weights, in order, as a device sampler's one thread sums them.
  float p[policy::max_top_k], z = 0.0f;
  for (size_t j = 0; j < k; j++) z += p[j] = std::exp((a[c[j]] - a[c[0]]) * inv_t);
  size_t kept = 1;
  float acc = p[0];
  while (kept < k && acc < s.top_p * z) acc += p[kept++];
  const float target =
      sample_uniform(philox(counter, sample_pick_element, 0, 0, s.seed, 0)) * acc;
  float run = 0.0f;
  for (size_t j = 0; j < kept; j++)
    if (target < (run += p[j])) return c[j];
  return c[kept - 1];
}

}  // namespace gpu
}  // namespace tl
