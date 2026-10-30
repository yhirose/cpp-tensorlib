// Standalone validation of include/cuda.h against CPU references — exercises
// the full pipeline (dlopen driver -> module load from #include'd PTX ->
// managed alloc -> kernel launch -> flush -> host read) before any array.h
// integration. Mirrors what the array oracle test will check, at the cuda::
// API level.
#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA  // standalone build; the CMake build passes it as a flag
#endif
#include "gpu.h"  // the shared ops (tl::gpu) resolve to cuda here

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace tl::cuda;
using kop = tl::gpu::kop;


// f32 -> f16 bits (round to nearest even; the finite range the checks use) and back.
static uint16_t to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFFu;
  if (e <= 0) {  // subnormal or zero
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    const int sh = 14 - e;
    uint32_t h = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
  return (uint16_t)(sign | h);
}
static float from_half(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float v = e == 0 ? std::ldexp((float)m, -24) : std::ldexp((float)(m | 1024), e - 25);
  return (h & 0x8000) ? -v : v;
}
// One Q4_K weight of a super-block (144 bytes: d, dmin, 12 bytes of 6-bit scales and
// mins, 128 bytes of nibbles), as ggml dequantizes it.
static float q4k_value(const uint8_t* b, int k) {
  const float d = from_half((uint16_t)(b[0] | b[1] << 8)), dmin = from_half((uint16_t)(b[2] | b[3] << 8));
  const uint8_t* q = b + 4;
  const int g = k / 32;
  const int sc = g < 4 ? (q[g] & 63) : ((q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4));
  const int mn = g < 4 ? (q[g + 4] & 63) : ((q[g + 4] >> 4) | ((q[g] >> 6) << 4));
  const uint8_t byte = b[16 + 32 * (g / 2) + (k % 32)];
  return d * sc * (float)((g & 1) ? byte >> 4 : byte & 15) - dmin * mn;
}
// One Q8_0 weight: a block is d (f16) then 32 int8.
static float q80_value(const uint8_t* b, int k) {
  return from_half((uint16_t)(b[0] | b[1] << 8)) * (float)(int8_t)b[2 + k];
}

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
    bool launched =
        tl::gpu::binary(kop::add, {a, 0}, {b, 0}, {o, 0}, n, 2.0f, 1.0f);
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
    bool launched = tl::gpu::unary(kop::sigmoid, {a, 0}, {o, 0}, n, 1.0f, 0.0f);
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
    bool launched = tl::gpu::gemm({a, 0}, k, false, {b, 0}, n, false, {o, 0}, m, n, k, 0.5f, 0);
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
    bool launched = tl::gpu::gemm({a, 0}, k, false, {b, 0}, k, true, {o, 0}, m, n, k, 1.0f, 0);
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
    bool launched = tl::gpu::row_op(kop::row_sum, {in, 0}, {o, 0}, rows, cols, 0.25f, 3.0f);
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
    bool launched = tl::gpu::row_op(kop::softmax, {in, 0}, {o, 0}, rows, cols, 1.0f, 0.0f);
    sync_to_host(o, false);  // D2H the device-written output before host read
    bool match = launched;
    for (int64_t i = 0; i < rows * cols; i++)
      if (std::fabs(co[i] - ref[i]) > 1e-4f) match = false;
    check(match, "softmax");
    release(in, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- the prefill GEMM over Q4_K and Q8_0 weights on the int8 tensor cores (gemm_quant, mmq) ----
  // Rows past the 128-row tile (M = 200), columns past the 128-column one (N = 300) and a
  // wider output row (ldc); K four super-blocks, which a grid this small splits in two
  // (the partial products' reduction is checked with the rest). Two references: the exact product of the
  // dequantized weights with the activation rounded to int8 as the kernel does (the
  // kernel must reproduce it to f32 rounding), and the same with the activation as given
  // (what the rounding costs).
  for (const int type : {12, 8}) {
    const char* name = type == 12 ? "Q4_K" : "Q8_0";
    const int64_t M = 200, N = 300, K = 1024, ldc = N + 8;
    const size_t blk = type == 12 ? 144 : 34, rowb = (size_t)K / (type == 12 ? 256 : 32) * blk;
    std::vector<uint8_t> W((size_t)N * rowb);
    for (auto& v : W) v = (uint8_t)(g() >> 8);
    for (size_t b = 0; b < W.size() / blk; b++) {  // sane d (and dmin)
      const uint16_t dh = to_half(0.002f + 0.001f * (float)(g() % 8)), mh = to_half(0.001f * (float)(g() % 8));
      W[b * blk] = (uint8_t)dh;
      W[b * blk + 1] = (uint8_t)(dh >> 8);
      if (type == 12) {
        W[b * blk + 2] = (uint8_t)mh;
        W[b * blk + 3] = (uint8_t)(mh >> 8);
      }
    }
    auto weight = [&](int64_t n, int64_t k) {
      return type == 12 ? q4k_value(&W[(size_t)n * rowb + (size_t)(k / 256) * 144], (int)(k % 256))
                        : q80_value(&W[(size_t)n * rowb + (size_t)(k / 32) * 34], (int)(k % 32));
    };
    float* cw = nullptr;
    float* ca = nullptr;
    float* cc = nullptr;
    void* w = alloc(W.size(), &cw);
    void* a = alloc(M * K * 2, &ca);
    void* o = alloc(M * ldc * 4, &cc);
    std::memcpy(cw, W.data(), W.size());
    uint16_t* ah = reinterpret_cast<uint16_t*>(ca);
    std::vector<float> x((size_t)M * K);
    for (int64_t m = 0; m < M; m++)
      for (int64_t k = 0; k < K; k++) {
        const uint16_t h = to_half(d(g) * (1.0f + (float)((m + k / 32) % 5)));
        ah[m * K + k] = h;
        x[m * K + k] = from_half(h);
      }
    std::fill(cc, cc + M * ldc, -777.0f);
    const bool launched = tl::gpu::gemm_quant(type, true, {a, 0}, {w, 0}, {o, 0}, M, N, K, ldc, true);
    sync_to_host(o, false);
    double e_kernel = 0, e_round = 0, norm = 0;
    bool untouched = true;
    for (int64_t m = 0; m < M; m++) {
      std::vector<float> xq(K);  // the activation rounded to int8 a group of 32 at a time
      for (int64_t gr = 0; gr < K / 32; gr++) {
        float amax = 0;
        for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(x[m * K + gr * 32 + k]));
        const float dd = amax / 127.0f;
        for (int k = 0; k < 32; k++)
          xq[gr * 32 + k] = dd == 0.0f ? 0.0f : std::nearbyint(x[m * K + gr * 32 + k] / dd) * dd;
      }
      for (int64_t n = 0; n < N; n++) {
        double r_q = 0, r_x = 0;
        for (int64_t k = 0; k < K; k++) {
          const double wv = weight(n, k);
          r_q += wv * xq[k];
          r_x += wv * x[m * K + k];
        }
        e_kernel += (cc[m * ldc + n] - r_q) * (cc[m * ldc + n] - r_q);
        e_round += (r_q - r_x) * (r_q - r_x);
        norm += r_x * r_x;
      }
      for (int64_t n = N; n < ldc; n++) untouched = untouched && cc[m * ldc + n] == -777.0f;
    }
    std::printf("  mmq %s: kernel vs int8-activation reference rel L2 %.2e, the rounding's cost %.2e\n",
                name, std::sqrt(e_kernel / norm), std::sqrt(e_round / norm));
    char what[64];
    std::snprintf(what, sizeof what, "mmq %s launch, ldc gap", name);
    check(launched && untouched, what);
    std::snprintf(what, sizeof what, "mmq %s vs int8 ref", name);
    check(std::sqrt(e_kernel / norm) < 1e-5, what);
    std::snprintf(what, sizeof what, "mmq %s act rounding", name);
    check(std::sqrt(e_round / norm) < 1e-2, what);
    release(w, 0, nullptr);
    release(a, 0, nullptr);
    release(o, 0, nullptr);
  }

  // ---- the same GEMM over a fused gate|up weight with the swiglu in its epilogue ----
  // (gemm_quant_glu): it must be bit for bit the plain MMQ GEMM followed by swiglu_h16's
  // f16 rounding of silu(gate) * up. FF = 150 leaves a partial 64-output tile; K = 512
  // keeps the grid unsplit (glu does not split K).
  for (const int type : {12, 8}) {
    const int64_t M = 200, FF = 150, N = 2 * FF, K = 512;
    const size_t blk = type == 12 ? 144 : 34, rowb = (size_t)K / (type == 12 ? 256 : 32) * blk;
    std::vector<uint8_t> W((size_t)N * rowb);
    for (auto& v : W) v = (uint8_t)(g() >> 8);
    for (size_t b = 0; b < W.size() / blk; b++) {
      const uint16_t dh = to_half(0.002f + 0.001f * (float)(g() % 8)), mh = to_half(0.001f * (float)(g() % 8));
      W[b * blk] = (uint8_t)dh;
      W[b * blk + 1] = (uint8_t)(dh >> 8);
      if (type == 12) {
        W[b * blk + 2] = (uint8_t)mh;
        W[b * blk + 3] = (uint8_t)(mh >> 8);
      }
    }
    float* cw = nullptr;
    float* ca = nullptr;
    float* cc = nullptr;
    float* ch = nullptr;
    float* cr = nullptr;
    void* w = alloc(W.size(), &cw);
    void* a = alloc(M * K * 2, &ca);
    void* o = alloc(M * N * 4, &cc);
    void* h = alloc(M * FF * 2, &ch);
    void* r = alloc(M * FF * 2, &cr);
    std::memcpy(cw, W.data(), W.size());
    uint16_t* ah = reinterpret_cast<uint16_t*>(ca);
    for (int64_t i = 0; i < M * K; i++) ah[i] = to_half(d(g) * 2.0f);
    const bool launched = tl::gpu::gemm_quant(type, true, {a, 0}, {w, 0}, {o, 0}, M, N, K, N, true) &&
                          tl::gpu::swiglu_h16({o, 0}, {r, 0}, FF, M) &&
                          tl::gpu::gemm_quant_glu(type, {a, 0}, {w, 0}, {h, 0}, M, FF, K);
    sync_to_host(r, false);
    sync_to_host(h, false);
    const uint16_t* hh = reinterpret_cast<const uint16_t*>(ch);
    const uint16_t* rr = reinterpret_cast<const uint16_t*>(cr);
    int64_t mismatches = 0;
    for (int64_t i = 0; i < M * FF; i++) mismatches += hh[i] != rr[i];
    std::printf("  mmq %s glu: %lld of %lld outputs differ from gemm + swiglu\n", type == 12 ? "Q4_K" : "Q8_0",
                (long long)mismatches, (long long)(M * FF));
    check(launched && mismatches == 0, type == 12 ? "mmq Q4_K glu" : "mmq Q8_0 glu");
    release(w, 0, nullptr);
    release(a, 0, nullptr);
    release(o, 0, nullptr);
    release(h, 0, nullptr);
    release(r, 0, nullptr);
  }

  std::printf(failures ? "CUDA cuda.h: FAILED\n" : "CUDA cuda.h: ALL OK\n");
  return failures ? 1 : 0;
}
