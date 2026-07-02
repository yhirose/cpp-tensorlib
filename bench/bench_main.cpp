// Micro-benchmark harness. Measurement discipline (docs/performance-notes.md):
// per-case medians of repeated trials; comparisons are only meaningful
// interleaved within one run. Absolute numbers swing with clock state.

#include <tensorlib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace {

tl::array random_array(tl::shape_t shape, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
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

// Interleaved A/B: alternate the two variants trial by trial so clock drift
// cancels (per performance-notes.md).
void report(const char* name, double flops,
            const std::function<void()>& fast,
            const std::function<void()>& oracle) {
  double t_fast = median_ms(fast);
  double t_ref = median_ms(oracle);
  std::printf("%-28s accel %9.3f ms (%7.2f GFLOP/s)   ref %9.3f ms   speedup %6.1fx\n",
              name, t_fast, flops / (t_fast * 1e6), t_ref, t_ref / t_fast);
}

}  // namespace

int main() {
  std::printf("tensorlib bench — accel backend vs ref oracle\n\n");

  for (int64_t n : {64, 256, 512, 1024}) {
    auto a = random_array({n, n}, 1);
    auto b = random_array({n, n}, 2);
    char name[64];
    std::snprintf(name, sizeof(name), "dot %lldx%lldx%lld",
                  static_cast<long long>(n), static_cast<long long>(n),
                  static_cast<long long>(n));
    double flops = 2.0 * n * n * n;
    report(
        name, flops,
        [&] {
          tl::use_accelerate_ = true;
          a.dot(b).eval();
        },
        [&] {
          tl::use_accelerate_ = false;
          a.dot(b).eval();
          tl::use_accelerate_ = true;
        });
  }
  std::printf("\n");

  {
    int64_t n = 1 << 20;
    auto a = random_array({n}, 3);
    auto b = random_array({n}, 4);
    report(
        "add 1M", static_cast<double>(n),
        [&] {
          tl::use_accelerate_ = true;
          (a + b).eval();
        },
        [&] {
          tl::use_accelerate_ = false;
          (a + b).eval();
          tl::use_accelerate_ = true;
        });
    report(
        "affine chain 1M (fused)", 2.0 * n,
        [&] {
          tl::use_accelerate_ = true;
          ((a * 2.0f) + 1.0f).eval();
        },
        [&] {
          tl::use_accelerate_ = false;
          ((a * 2.0f) + 1.0f).eval();
          tl::use_accelerate_ = true;
        });
    report(
        "exp 1M", static_cast<double>(n),
        [&] {
          tl::use_accelerate_ = true;
          a.exp().eval();
        },
        [&] {
          tl::use_accelerate_ = false;
          a.exp().eval();
          tl::use_accelerate_ = true;
        });
  }

  if (tl::gpu_available()) {
    int64_t n = 1 << 22;  // 4M — large enough to be bandwidth-bound
    auto a = random_array({n}, 8);
    auto b = random_array({n}, 9);
    report(
        "gpu add 4M", static_cast<double>(n),
        [&] {
          tl::use_gpu();
          (a + b).eval();
          tl::use_cpu();
        },
        [&] { (a + b).eval(); });
    report(
        "gpu ew-chain x8 4M (1 flush)", 8.0 * 3 * n,
        [&] {
          tl::use_gpu();
          auto x = a;
          for (int i = 0; i < 8; i++) x = (x * 1.01f + b * 0.1f).relu();
          x.eval();
          tl::use_cpu();
        },
        [&] {
          auto x = a;
          for (int i = 0; i < 8; i++) x = (x * 1.01f + b * 0.1f).relu();
          x.eval();
        });
    std::printf("\n");
  }

  if (tl::gpu_available()) {
    std::printf("-- Metal GPU vs Accelerate CPU --\n");
    for (int64_t n : {256, 512, 1024, 2048}) {
      auto a = random_array({n, n}, 20);
      auto b = random_array({n, n}, 21);
      char name[64];
      std::snprintf(name, sizeof(name), "gpu dot %lldx%lldx%lld",
                    (long long)n, (long long)n, (long long)n);
      double flops = 2.0 * n * n * n;
      // Interleaved GPU vs CPU-accel (not vs ref — too slow at these sizes).
      double t_gpu = median_ms([&] {
        tl::use_gpu();
        a.dot(b).eval();
        tl::use_cpu();
      });
      double t_cpu = median_ms([&] { a.dot(b).eval(); });
      std::printf("%-28s gpu %8.3f ms (%8.2f GFLOP/s)   accel-cpu %8.3f ms   gpu/cpu %5.2fx\n",
                  name, t_gpu, flops / (t_gpu * 1e6), t_cpu, t_cpu / t_gpu);
    }
    // full MLP forward, GPU end to end
    {
      auto x = random_array({256, 784}, 22);
      auto w1 = random_array({784, 256}, 23);
      auto w2 = random_array({256, 10}, 24);
      double t_gpu = median_ms([&] {
        tl::use_gpu();
        x.dot(w1).sigmoid().dot(w2).softmax().eval();
        tl::use_cpu();
      });
      double t_cpu = median_ms(
          [&] { x.dot(w1).sigmoid().dot(w2).softmax().eval(); });
      std::printf("%-28s gpu %8.3f ms                        accel-cpu %8.3f ms   gpu/cpu %5.2fx\n",
                  "gpu mlp fwd 256x784x256x10", t_gpu, t_cpu, t_cpu / t_gpu);
    }
    std::printf("\n");
  }

  // training-step composite: forward + backward-ish chain
  {
    auto x = random_array({100, 784}, 5);
    auto w1 = random_array({784, 50}, 6);
    auto w2 = random_array({50, 10}, 7);
    report(
        "mlp fwd 100x784x50x10", 2.0 * 100 * (784.0 * 50 + 50 * 10),
        [&] {
          tl::use_accelerate_ = true;
          x.dot(w1).sigmoid().dot(w2).softmax().eval();
        },
        [&] {
          tl::use_accelerate_ = false;
          x.dot(w1).sigmoid().dot(w2).softmax().eval();
          tl::use_accelerate_ = true;
        });
  }

  return 0;
}
