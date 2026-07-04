// Standalone validation of include/cuda.h against CPU references — exercises
// the full pipeline (dlopen driver -> module load from #include'd PTX ->
// managed alloc -> kernel launch -> flush -> host read) before any array.h
// integration. Mirrors what the array oracle test will check, at the cuda::
// API level.
#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA  // standalone build; the CMake build passes it as a flag
#endif
#include "cuda.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tl::cuda;
using kop = tl::metal::kop;

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("  %-22s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) failures++;
}

int main() {
  if (!available()) {
    // No driver/device — the valid CPU-fallback state (e.g. CI hosted runners
    // build with TENSORLIB_CUDA but have no GPU). Skip, don't fail: the driver-
    // absent fallback is itself a permanent test target (see .github/ci.yml).
    std::printf("no CUDA device — skipping (CPU-fallback build is valid)\n");
    return 0;
  }
  std::printf("cuda backend available\n");
  std::mt19937 g(7);
  std::uniform_real_distribution<float> d(-1, 1);

  // ---- elementwise add with affine epilogue: out = (a+b)*2 + 1 ----
  {
    const int64_t n = 1000;
    float* ca = nullptr;
    float* cb = nullptr;
    float* co = nullptr;
    void* a = alloc(n * 4, &ca);
    void* b = alloc(n * 4, &cb);
    void* o = alloc(n * 4, &co);
    std::vector<float> ref(n);
    for (int64_t i = 0; i < n; i++) {
      ca[i] = d(g);
      cb[i] = d(g);
      ref[i] = (ca[i] + cb[i]) * 2.0f + 1.0f;
    }
    bool launched = tl::cuda::binary(kop::add, a, 0, b, 0, o, 0, n, 2.0f, 1.0f);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < n; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-4f) match = false;
    check(match, "binary add+affine");
    release(a, 0, nullptr);
    release(b, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- unary sigmoid ----
  {
    const int64_t n = 777;
    float* ca = nullptr;
    float* co = nullptr;
    void* a = alloc(n * 4, &ca);
    void* o = alloc(n * 4, &co);
    std::vector<float> ref(n);
    for (int64_t i = 0; i < n; i++) {
      ca[i] = d(g) * 4.0f;
      ref[i] = 1.0f / (1.0f + std::exp(-ca[i]));
    }
    bool launched = tl::cuda::unary(kop::sigmoid, a, 0, o, 0, n, 1.0f, 0.0f);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < n; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-4f) match = false;
    check(match, "unary sigmoid");
    release(a, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- gemm: C = (A@B)*scale, non-transposed, m×k @ k×n ----
  {
    const int64_t m = 64, k = 48, n = 40;
    float* ca = nullptr;
    float* cb = nullptr;
    float* co = nullptr;
    void* a = alloc(m * k * 4, &ca);
    void* b = alloc(k * n * 4, &cb);
    void* o = alloc(m * n * 4, &co);
    for (int64_t i = 0; i < m * k; i++) ca[i] = d(g);
    for (int64_t i = 0; i < k * n; i++) cb[i] = d(g);
    std::vector<float> ref(m * n, 0);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        float s = 0;
        for (int64_t p = 0; p < k; p++) s += ca[i * k + p] * cb[p * n + j];
        ref[i * n + j] = s * 0.5f;
      }
    bool launched = gemm(a, 0, k, false, b, 0, n, false, o, 0, m, n, k, 0.5f, 0);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < m * n; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-3f * (1 + std::fabs(ref[i])))
        match = false;
    check(match, "gemm NN + scale");
    release(a, 0, nullptr);
    release(b, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- gemm with transposed B (Bᵀ read in place): C = A @ (Bt)ᵀ ----
  {
    const int64_t m = 32, k = 32, n = 32;
    float* ca = nullptr;
    float* cb = nullptr;
    float* co = nullptr;
    void* a = alloc(m * k * 4, &ca);
    void* b = alloc(n * k * 4, &cb);  // stored n×k, logically k×n via trans
    void* o = alloc(m * n * 4, &co);
    for (int64_t i = 0; i < m * k; i++) ca[i] = d(g);
    for (int64_t i = 0; i < n * k; i++) cb[i] = d(g);
    std::vector<float> ref(m * n, 0);
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        float s = 0;
        for (int64_t p = 0; p < k; p++) s += ca[i * k + p] * cb[j * k + p];
        ref[i * n + j] = s;
      }
    // trans_b=true, ldb=k (the col stride of the logical k×n = row stride of bt)
    bool launched = gemm(a, 0, k, false, b, 0, k, true, o, 0, m, n, k, 1.0f, 0);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < m * n; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-3f * (1 + std::fabs(ref[i])))
        match = false;
    check(match, "gemm NT (transposed B)");
    release(a, 0, nullptr);
    release(b, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- row_sum with affine epilogue ----
  {
    const int64_t rows = 20, cols = 300;
    float* ci = nullptr;
    float* co = nullptr;
    void* in = alloc(rows * cols * 4, &ci);
    void* o = alloc(rows * 4, &co);
    std::vector<float> ref(rows);
    for (int64_t r = 0; r < rows; r++) {
      float s = 0;
      for (int64_t c = 0; c < cols; c++) {
        ci[r * cols + c] = d(g);
        s += ci[r * cols + c];
      }
      ref[r] = s * 0.25f + 3.0f;
    }
    bool launched = tl::cuda::row_op(kop::row_sum, in, 0, o, 0, rows, cols, 0.25f, 3.0f);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t r = 0; r < rows; r++)
      if (std::fabs(co[r] - ref[r]) > 1e-2f) match = false;
    check(match, "row_sum + affine");
    release(in, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- softmax (stable; scale/offset ignored) ----
  {
    const int64_t rows = 16, cols = 257;
    float* ci = nullptr;
    float* co = nullptr;
    void* in = alloc(rows * cols * 4, &ci);
    void* o = alloc(rows * cols * 4, &co);
    std::vector<float> ref(rows * cols);
    for (int64_t r = 0; r < rows; r++) {
      float mx = -1e30f, sum = 0;
      for (int64_t c = 0; c < cols; c++) {
        ci[r * cols + c] = d(g) * 5.0f;
        mx = std::max(mx, ci[r * cols + c]);
      }
      for (int64_t c = 0; c < cols; c++) sum += std::exp(ci[r * cols + c] - mx);
      for (int64_t c = 0; c < cols; c++)
        ref[r * cols + c] = std::exp(ci[r * cols + c] - mx) / sum;
    }
    bool launched = tl::cuda::row_op(kop::softmax, in, 0, o, 0, rows, cols, 1.0f, 0.0f);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < rows * cols; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-4f) match = false;
    check(match, "softmax");
    release(in, 0, nullptr);
    release(o, 0, nullptr);
  }

  std::printf(failures ? "CUDA cuda.h: FAILED\n" : "CUDA cuda.h: ALL OK\n");
  return failures ? 1 : 0;
}
