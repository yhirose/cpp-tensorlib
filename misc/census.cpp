// auto-mode crossover census: CPU (Accelerate) vs GPU-total (Metal single
// op + flush) per kernel class. The threshold is where GPU-total first
// beats CPU. Interleaved A/B, warmup, medians (silarray discipline).
#include <tensorlib.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>
using tl::array;
using clk = std::chrono::steady_clock;

static array rnd(tl::shape_t s, unsigned seed) {
  std::mt19937 g(seed); std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1; for (auto x : s) n *= (size_t)x;
  std::vector<float> v(n); for (auto& x : v) x = d(g);
  return array::from(std::move(v), std::move(s));
}

// median of `trials` single-op timings, warmup first
static double med_ms(int trials, const std::function<void()>& f) {
  for (int i = 0; i < 4; i++) f();  // warmup (GPU clock ramp)
  std::vector<double> ts;
  for (int i = 0; i < trials; i++) {
    auto t0 = clk::now(); f();
    ts.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count());
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size()/2];
}

static void row(const char* label, double n, double cpu, double gpu) {
  const char* who = gpu < cpu ? "GPU" : "cpu";
  std::printf("  %-22s n=%-12.0f cpu %8.3f ms   gpu %8.3f ms   %s%s\n",
              label, n, cpu, gpu, gpu < cpu ? "**":"  ", who);
}

int main() {
  std::printf("== MATMUL (square, n=M*N*K) ==\n");
  for (int64_t m : {64, 128, 256, 384, 512, 768, 1024}) {
    auto a = rnd({m, m}, 1), b = rnd({m, m}, 2);
    double cpu = (tl::use_cpu(), med_ms(15, [&]{ a.dot(b).eval(); }));
    double gpu = (tl::use_gpu(), med_ms(15, [&]{ a.dot(b).eval(); })); tl::use_cpu();
    char buf[32]; std::snprintf(buf,sizeof buf,"%lldx%lldx%lld",(long long)m,(long long)m,(long long)m);
    row(buf, (double)m*m*m, cpu, gpu);
  }
  std::printf("== ELEMENTWISE (add, n=elements) ==\n");
  for (int64_t n : {(int64_t)1<<12,(int64_t)1<<14,(int64_t)1<<16,(int64_t)1<<18,(int64_t)1<<20,(int64_t)1<<22}) {
    auto a = rnd({n}, 3), b = rnd({n}, 4);
    double cpu = (tl::use_cpu(), med_ms(15, [&]{ (a+b).eval(); }));
    double gpu = (tl::use_gpu(), med_ms(15, [&]{ (a+b).eval(); })); tl::use_cpu();
    char buf[32]; std::snprintf(buf,sizeof buf,"%lld", (long long)n);
    row(buf, (double)n, cpu, gpu);
  }
  std::printf("== REDUCTION (softmax rows, n=elements) ==\n");
  for (int64_t cols : {256, 1024, 4096}) {
    for (int64_t rows : {64, 256, 1024, 4096}) {
      auto a = rnd({rows, cols}, 5);
      double cpu = (tl::use_cpu(), med_ms(15, [&]{ a.softmax().eval(); }));
      double gpu = (tl::use_gpu(), med_ms(15, [&]{ a.softmax().eval(); })); tl::use_cpu();
      char buf[32]; std::snprintf(buf,sizeof buf,"%lldx%lld", (long long)rows,(long long)cols);
      row(buf, (double)rows*cols, cpu, gpu);
    }
  }
  return 0;
}
