#pragma once
// The two helpers every census program needs: a seeded uniform(-1, 1) array
// and a median-of-trials timer with a short warm-up (misc/census.cpp's
// measurement discipline). Header-only so a census stays one file.
#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

namespace census {

inline tl::array rnd(tl::shape_t s, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1;
  for (auto x : s) n *= (size_t)x;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return tl::array::from(std::move(v), std::move(s));
}

// Median wall time of `trials` calls in microseconds, after 3 untimed calls.
inline double med_us(int trials, const std::function<void()>& f) {
  using clk = std::chrono::steady_clock;
  for (int i = 0; i < 3; i++) f();
  std::vector<double> ts;
  for (int i = 0; i < trials; i++) {
    auto t0 = clk::now();
    f();
    ts.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

}  // namespace census
