// Batched-GEMM launch census: the same [batch,m,k]·[batch,k,n] product as one
// launch (the batch folded into the kernel grid) against `batch` per-slice
// launches (TL_BDOT_PER_SLICE forces that loop in the child), on the GPU, in
// µs. Attention's probs·v ([H,T,T]·[H,T,D]) and scores ([H,T,D]·[H,D,T], NT)
// at H=8 D=64, T=512 and 1024, plus a rank-4 [B,H,…] pair. Medians, warm-up
// first (misc/census.cpp discipline). Run without the env for the fused
// column; the program re-execs itself with it for the loop column.
#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <unistd.h>
#include <vector>

using tl::array;
using clk = std::chrono::steady_clock;

static array rnd(tl::shape_t s, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1;
  for (auto x : s) n *= (size_t)x;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return array::from(std::move(v), std::move(s));
}

static double med_us(int trials, const std::function<void()>& f) {
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

int main(int argc, char** argv) {
  if (!tl::gpu_available()) {
    std::printf("no GPU\n");
    return 0;
  }
  const bool loop = std::getenv("TL_BDOT_PER_SLICE") != nullptr;
  tl::use_gpu();
  struct S { const char* name; tl::shape_t a, b; bool nt; };
  const S shapes[] = {
      {"probs.v  H8 T512 D64", {8, 512, 512}, {8, 512, 64}, false},
      {"probs.v  H8 T1024 D64", {8, 1024, 1024}, {8, 1024, 64}, false},
      {"q.kT     H8 T512 D64", {8, 512, 64}, {8, 512, 64}, true},
      {"q.kT     H8 T1024 D64", {8, 1024, 64}, {8, 1024, 64}, true},
      {"rank4    B2 H8 T256 D64", {2, 8, 256, 256}, {2, 8, 256, 64}, false},
  };
  std::printf("%-26s %10s\n", "shape", loop ? "per-slice" : "one-launch");
  for (const auto& s : shapes) {
    auto A = rnd(s.a, 1), B = rnd(s.b, 2);
    A.eval();
    B.eval();
    double t = med_us(21, [&] {
      (s.nt ? A.dot(B.transpose({0, 2, 1})) : A.dot(B)).eval();
    });
    std::printf("%-26s %10.1f\n", s.name, t);
  }
  if (!loop && argc > 0) {
    std::fflush(stdout);
    setenv("TL_BDOT_PER_SLICE", "1", 1);
    execv(argv[0], argv);
  }
  return 0;
}
