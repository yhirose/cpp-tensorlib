// Browser port of misc/census.cpp for the WebGPU backend (M10, Phase 4).
//
// Same purpose as the native census — find, per kernel class, the size where
// GPU-total (single op + flush) first beats CPU — but it has to run in a page
// because that is the only place this backend exists. It is a separate wasm
// from the test harness so the two never interfere: the census needs long runs
// and the suite needs to stay a fast pass/fail.
//
// Differences from misc/census.cpp, all forced by the environment:
//
//   - The wasm CPU path is scalar single-threaded (~8 GF/s), roughly 30x
//     slower than the M1 Pro's Accelerate/AMX path, so the largest native
//     sizes would take minutes. Trial counts fall off with size.
//   - CPU and GPU alternate *which one runs first* every trial. Blocked A/B
//     (all CPU, then all GPU) puts every one-time cost — first-touch
//     allocation, GPU clock ramp — on whichever side ran first; culebra's
//     Metal calibration was misled by exactly that (see its
//     project_gpu_bench_shape_findings).
//   - A whole-block section, absent from the native census, anchors
//     batch_matmul_bias_threshold_(): that threshold is about a batch of ops
//     riding one device together, which no single-op measurement can show.
//
// Output goes to stdout as plain text; the page just displays it.

#include <emscripten.h>
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
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1, 1);
  size_t n = 1;
  for (auto x : s) n *= (size_t)x;
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return array::from(std::move(v), std::move(s));
}

static double time1(const std::function<void()>& f) {
  auto t0 = clk::now();
  f();
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct ab {
  double cpu, gpu;
};

// Interleaved A/B with the order rotated per trial, medians, warmup on both
// devices first (the GPU one also compiles/caches the pipeline).
static ab bench(int trials, const std::function<void()>& f) {
  for (int i = 0; i < 3; i++) {
    tl::use_cpu();
    f();
    tl::use_gpu();
    f();
  }
  std::vector<double> c, g;
  for (int i = 0; i < trials; i++) {
    if (i % 2 == 0) {
      tl::use_cpu();
      c.push_back(time1(f));
      tl::use_gpu();
      g.push_back(time1(f));
    } else {
      tl::use_gpu();
      g.push_back(time1(f));
      tl::use_cpu();
      c.push_back(time1(f));
    }
  }
  tl::use_cpu();
  return {median(std::move(c)), median(std::move(g))};
}

static void row(const char* label, double n, ab t) {
  std::printf("  %-22s n=%-12.0f cpu %9.3f ms   gpu %9.3f ms   %s %s\n", label, n,
              t.cpu, t.gpu, t.gpu < t.cpu ? "**" : "  ", t.gpu < t.cpu ? "GPU" : "cpu");
}

extern "C" {

// {async: true} from JS: the GPU waits beneath this suspend under JSPI.
EMSCRIPTEN_KEEPALIVE int run_census() {
  std::printf("[census] gpu_available=%d\n", tl::gpu_available() ? 1 : 0);
  if (!tl::gpu_available()) {
    std::printf("[census] FAIL: no GPU device\n");
    return 0;
  }

  std::printf("== MATMUL (square, n=M*N*K) ==\n");
  for (int64_t m : {32, 48, 64, 96, 128, 160, 192, 256, 384, 512}) {
    auto a = rnd({m, m}, 1), b = rnd({m, m}, 2);
    int trials = m <= 128 ? 15 : (m <= 256 ? 9 : 5);
    auto t = bench(trials, [&] { a.dot(b).eval(); });
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lldx%lldx%lld", (long long)m, (long long)m,
                  (long long)m);
    row(buf, (double)m * m * m, t);
  }

  std::printf("== ELEMENTWISE (add, n=elements) ==\n");
  for (int64_t n : {(int64_t)1 << 12, (int64_t)1 << 14, (int64_t)1 << 16,
                    (int64_t)1 << 18, (int64_t)1 << 20, (int64_t)1 << 22}) {
    auto a = rnd({n}, 3), b = rnd({n}, 4);
    auto t = bench(n <= (1 << 20) ? 15 : 7, [&] { (a + b).eval(); });
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", (long long)n);
    row(buf, (double)n, t);
  }

  std::printf("== REDUCTION (softmax rows, n=elements) ==\n");
  for (int64_t cols : {256, 1024, 4096}) {
    for (int64_t rows : {16, 64, 256, 1024}) {
      auto a = rnd({rows, cols}, 5);
      auto t = bench(rows * cols <= (1 << 18) ? 15 : 7,
                     [&] { a.softmax().eval(); });
      char buf[32];
      std::snprintf(buf, sizeof buf, "%lldx%lld", (long long)rows, (long long)cols);
      row(buf, (double)rows * cols, t);
    }
  }

  // Whole-block anchor for batch_matmul_bias_threshold_(). One eval batch
  // shaped like a transformer block: three seq*d*d projections, two
  // seq*seq*d attention gemms, two seq*d*4d FFN gemms, plus the elementwise
  // and softmax that sit between them. The question this answers is not "does
  // the biggest gemm earn the GPU" but "does the whole batch earn it", which
  // is what the batch bias decides.
  std::printf("== BLOCK (transformer-ish, one eval batch; n=sum M*N*K) ==\n");
  const int64_t seq = 128;
  for (int64_t d : {32, 48, 64, 96, 128, 192, 256, 384, 512}) {
    auto x = rnd({seq, d}, 6);
    auto wq = rnd({d, d}, 7), wk = rnd({d, d}, 8), wv = rnd({d, d}, 9);
    auto w1 = rnd({d, 4 * d}, 10), w2 = rnd({4 * d, d}, 11);
    auto t = bench(d <= 256 ? 9 : 5, [&] {
      auto q = x.dot(wq), k = x.dot(wk), v = x.dot(wv);
      auto att = q.dot(k.transpose()).softmax();
      auto h = att.dot(v) + x;
      (h.dot(w1).relu().dot(w2) + h).eval();
    });
    double work = 3.0 * seq * d * d       // q, k, v projections
                  + (double)seq * seq * d  // q @ k^T
                  + (double)seq * d * seq  // att @ v
                  + 2.0 * seq * d * 4 * d; // ffn
    char buf[32];
    std::snprintf(buf, sizeof buf, "seq%lld d%lld", (long long)seq, (long long)d);
    row(buf, work, t);
  }

  for (auto& kv : tl::webgpu::context::get().dispatch_counts)
    std::printf("[census] dispatch %s=%ld\n", kv.first.c_str(), kv.second);
  std::printf("[census] DONE\n");
  return 1;
}

}  // extern "C"
