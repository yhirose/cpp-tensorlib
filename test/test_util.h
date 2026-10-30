#pragma once

// What every test file shares: the includes, and the random arrays and
// oracle comparisons the cases build on.

#include "doctest.h"

#include <ggml_quant.h>
#include <gguf.h>
#include <kv_cache.h>
#include <tensorlib.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>

using tl::array;

namespace {

array random_array(tl::shape_t shape, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return array::from(std::move(v), std::move(shape));
}

// Evaluate `build()` twice — accelerated and forced through the ref oracle —
// and require the results to agree.
template <typename F>
bool matches_oracle(F build, float rtol = 1e-4f, float atol = 1e-5f) {
  tl::use_accelerate_ = true;
  auto fast = build().eval();
  // Oracle = ref::: disable both accelerated CPU backends (accel and the
  // own-CPU GEMM), else dot would be validated against cpu:: not ref::.
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  return tl::allclose(fast, oracle, rtol, atol);
}

// Evaluate `build()` through the own-CPU backend (accel off, cpu on) and
// through ref (both off); require agreement. Validates cpu::sgemm — on any
// platform, since it never touches Metal/Accelerate.
template <typename F>
bool cpu_matches_ref(F build, float rtol = 2e-3f, float atol = 2e-4f) {
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = true;
  auto fast = build().eval();
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  return tl::allclose(fast, oracle, rtol, atol);
}

// Evaluate `build()` in gpu mode and forced through the ref oracle; require
// agreement. No-op (trivially true) where no Metal device exists.
template <typename F>
bool matches_gpu_oracle(F build, float rtol = 1e-4f, float atol = 1e-5f) {
  if (!tl::gpu_available()) return true;
  auto prev = tl::device_;
  tl::use_gpu();
  auto fast = build().eval();
  tl::use_cpu();
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  tl::device_ = prev;
  return tl::allclose(fast, oracle, rtol, atol);
}

}  // namespace
