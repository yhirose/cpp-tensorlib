#pragma once

// What every test file shares: the includes, and the random arrays and
// oracle comparisons the cases build on.

#include "doctest.h"

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

// A device copy of `a`, evaluated so its device_span() is live.
array dev(const array& a) {
  array c = a.clone();
  c.eval();
  return c;
}
// Whether `got` is within tol (absolute and relative) of `want`.
bool same(const array& got, const array& want, float tol) {
  return tl::allclose(got, want, tol, tol);
}

// `a`'s values in row-major order, on the host.
std::vector<float> host_values(const array& a) {
  array c = a.contiguous() ? a : a.clone();
  c.eval();
  return std::vector<float>(c.raw(), c.raw() + c.size());
}

// An operand off the alignment the GPU kernels' vector loads want: `a`'s
// values `lead` elements into a fresh 1-D buffer, zeros before them. Viewed
// at .at(4 * lead) (or, made bf16 first, .at(2 * lead)), it holds the same
// values as `a` from a base `lead` elements past the buffer's.
array with_lead(const array& a, int64_t lead) {
  const std::vector<float> h = host_values(a);
  std::vector<float> v((size_t)lead, 0.0f);
  v.insert(v.end(), h.begin(), h.end());
  return array::from(std::move(v), {(int64_t)v.size()});
}

// The `n` values a kernel wrote `lead` elements into `buf` (a with_lead
// output), read back with no GPU work.
std::vector<float> after_lead(const array& buf, int64_t lead, int64_t n) {
  const std::vector<float> h = host_values(buf);
  return std::vector<float>(h.begin() + lead, h.begin() + lead + n);
}

// Whether `got` is within tol (absolute and relative) of `want`, compared on
// the host: a launch that faulted poisons the device, after which a GPU-side
// comparison proves nothing, so the expected values are read back first.
bool host_close(const std::vector<float>& got, const std::vector<float>& want, float tol) {
  if (got.size() != want.size()) return false;
  for (size_t i = 0; i < got.size(); i++)
    if (!(std::fabs(got[i] - want[i]) <= tol + tol * std::fabs(want[i]))) return false;
  return true;
}

// Whether the device still runs work: a kernel that faulted on a misaligned
// view leaves the context unusable, which a refusal or a correct read never
// does. A shared op's own launch, so a failure shows as false.
bool device_healthy() {
  array a = array::from(std::vector<float>{1.0f, 2.0f}, {2}).clone();
  a.eval();
  array o = array::empty({2});
  if (!tl::gpu::binary(tl::gpu::kop::add, a.device_span(), a.device_span(), o.device_span(), 2,
                       1.0f, 0.0f))
    return false;
  tl::gpu::flush();
  const std::vector<float> h = host_values(o);
  return h[0] == 2.0f && h[1] == 4.0f;
}

// An op handed a view off its kernel's alignment: `run` launches it and says
// whether it ran; if it did, `right` must hold of its result. Either way the
// device must still work afterwards, which a kernel that faulted would not
// leave it doing. must_run: the op has a fallback of its own, so never declines.
template <class Run, class Right>
void check_misaligned(const char* what, Run run, Right right, bool must_run = false) {
  INFO(what);
  const bool ran = run();
  tl::gpu::flush();
  if (must_run) CHECK(ran);
  if (ran) CHECK(right());
  CHECK(device_healthy());
}

}  // namespace
