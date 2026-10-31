// Standalone validation of include/cuda.h against CPU references — exercises
// the full pipeline (dlopen driver -> module load from #include'd PTX ->
// managed alloc -> kernel launch -> flush -> host read) before any array.h
// integration. Mirrors what the array oracle test will check, at the cuda::
// API level.
#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA  // standalone build; the CMake build passes it as a flag
#endif
#include "gpu.h"  // the shared ops (tl::gpu) resolve to cuda here

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace tl::cuda;
using kop = tl::gpu::kop;


// One Q4_K weight of a super-block (144 bytes: d, dmin, 12 bytes of 6-bit scales and
// mins, 128 bytes of nibbles), as ggml dequantizes it.
static float q4k_value(const uint8_t* b, int k) {
  const float d = tl::f16_to_f32((uint16_t)(b[0] | b[1] << 8)), dmin = tl::f16_to_f32((uint16_t)(b[2] | b[3] << 8));
  const uint8_t* q = b + 4;
  const int g = k / 32;
  const int sc = g < 4 ? (q[g] & 63) : ((q[g + 4] & 0xF) | ((q[g - 4] >> 6) << 4));
  const int mn = g < 4 ? (q[g + 4] & 63) : ((q[g + 4] >> 4) | ((q[g] >> 6) << 4));
  const uint8_t byte = b[16 + 32 * (g / 2) + (k % 32)];
  return d * sc * (float)((g & 1) ? byte >> 4 : byte & 15) - dmin * mn;
}
// One Q8_0 weight: a block is d (f16) then 32 int8.
static float q80_value(const uint8_t* b, int k) {
  return tl::f16_to_f32((uint16_t)(b[0] | b[1] << 8)) * (float)(int8_t)b[2 + k];
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
      const uint16_t dh = tl::f32_to_f16(0.002f + 0.001f * (float)(g() % 8)), mh = tl::f32_to_f16(0.001f * (float)(g() % 8));
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
        const uint16_t h = tl::f32_to_f16(d(g) * (1.0f + (float)((m + k / 32) % 5)));
        ah[m * K + k] = h;
        x[m * K + k] = tl::f16_to_f32(h);
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
      const uint16_t dh = tl::f32_to_f16(0.002f + 0.001f * (float)(g() % 8)), mh = tl::f32_to_f16(0.001f * (float)(g() % 8));
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
    for (int64_t i = 0; i < M * K; i++) ah[i] = tl::f32_to_f16(d(g) * 2.0f);
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

  // ---- Qwen3.5's gated norm in one pass (gated_rmsnorm) against the norm and a swiglu ----
  // T x H rows of D, z strided as in the projection's rows; f32 and f16 outputs, each bit
  // for bit the composition it replaces (rmsnorm, then swiglu / swiglu_h16 over [z | norm]).
  // D = 128 is Qwen3.5's; 320 runs past rmsnorm's 256 threads.
  for (const int64_t D : {128, 320}) {
    const int64_t T = 37, H = 6, V = H * D, ldz = 3 * V;
    float *co = nullptr, *cz = nullptr, *cw = nullptr, *cf = nullptr, *ch = nullptr, *czc = nullptr,
          *czn = nullptr, *cr = nullptr, *crh = nullptr;
    void* o = alloc(T * V * 4, &co);
    void* z = alloc(T * ldz * 4, &cz);
    void* w = alloc(D * 4, &cw);
    void* f = alloc(T * V * 4, &cf);
    void* h = alloc(T * V * 2, &ch);
    void* zc = alloc(T * V * 4, &czc);      // z's rows, contiguous
    void* zn = alloc(2 * T * V * 4, &czn);  // [z | rmsnorm(o)], what the swiglu reads, made on the device
    void* r = alloc(T * V * 4, &cr);
    void* rh = alloc(T * V * 2, &crh);
    for (int64_t i = 0; i < T * V; i++) co[i] = d(g) * 3.0f;
    for (int64_t i = 0; i < T * ldz; i++) cz[i] = d(g) * 4.0f;
    for (int64_t i = 0; i < D; i++) cw[i] = 0.5f + d(g);
    for (int64_t t = 0; t < T; t++)
      for (int64_t i = 0; i < V; i++) czc[t * V + i] = cz[t * ldz + V + i];
    const bool ok =
        tl::gpu::gated_rmsnorm({o, 0}, {z, 4 * V}, ldz, {w, 0}, {f, 0}, false, T, H, D, 1e-6f) &&
        tl::gpu::gated_rmsnorm({o, 0}, {z, 4 * V}, ldz, {w, 0}, {h, 0}, true, T, H, D, 1e-6f) &&
        tl::gpu::copy_f32({zc, 0}, {zn, 0}, T * V) &&
        tl::gpu::rmsnorm({o, 0}, {w, 0}, {zn, 4 * T * V}, D, 1e-6f, T * H) &&
        tl::gpu::swiglu({zn, 0}, {r, 0}, T * V) && tl::gpu::swiglu_h16({zn, 0}, {rh, 0}, T * V, 1);
    for (void* b : {f, h, r, rh}) sync_to_host(b, false);
    int64_t d32 = 0, d16 = 0;
    for (int64_t i = 0; i < T * V; i++) {
      d32 += std::memcmp(&cf[i], &cr[i], 4) != 0;
      d16 += reinterpret_cast<const uint16_t*>(ch)[i] != reinterpret_cast<const uint16_t*>(crh)[i];
    }
    std::printf("  gated_rmsnorm D=%lld: %lld f32 and %lld f16 of %lld outputs differ from rmsnorm + swiglu\n",
                (long long)D, (long long)d32, (long long)d16, (long long)(T * V));
    check(ok && d32 == 0 && d16 == 0, D == 128 ? "gated_rmsnorm D=128" : "gated_rmsnorm D=320");
    for (void* b : {o, z, w, f, h, zc, zn, r, rh}) release(b, 0, nullptr);
  }

  // ---- Qwen3's q / k norm and rotation in one pass (rmsnorm_rope) against rmsnorm then
  // rope, bit for bit: the head widths the kernel takes (Qwen3's 128, Qwen3.5's 256
  // turning 64, and 64), and two it refuses (rot 32, D 320), which gpu::rmsnorm_rope
  // then runs as the two. T = 37 leaves rows past a whole number of eight-row blocks.
  {
    struct shape { int64_t D, rot, H; };
    for (const shape s : {shape{128, 128, 5}, shape{256, 64, 3}, shape{64, 64, 4}, shape{128, 32, 3}, shape{320, 320, 2}}) {
      const int64_t T = 37, rows = s.H * T;
      float *cx = nullptr, *cw = nullptr, *cf = nullptr, *cr = nullptr;
      void* x = alloc(rows * s.D * 4, &cx);
      void* w = alloc(s.D * 4, &cw);
      void* f = alloc(rows * s.D * 4, &cf);
      void* r = alloc(rows * s.D * 4, &cr);
      for (int64_t i = 0; i < rows * s.D; i++) cx[i] = cf[i] = d(g) * 3.0f;
      for (int64_t i = 0; i < s.D; i++) cw[i] = 0.5f + d(g);
      const bool ok = tl::gpu::rmsnorm_rope({f, 0}, {w, 0}, rows, T, s.D, 2000, 1e6f, s.rot, 1e-6f) &&
                      tl::gpu::rmsnorm({x, 0}, {w, 0}, {r, 0}, s.D, 1e-6f, rows) &&
                      tl::gpu::rope({r, 0}, {r, 0}, rows, T, s.D, 2000, 1e6f, {}, s.rot);
      sync_to_host(f, false);
      sync_to_host(r, false);
      int64_t diff = 0;
      for (int64_t i = 0; i < rows * s.D; i++) diff += std::memcmp(&cf[i], &cr[i], 4) != 0;
      std::printf("  rmsnorm_rope D=%lld rot=%lld: %lld of %lld values differ from rmsnorm + rope\n", (long long)s.D,
                  (long long)s.rot, (long long)diff, (long long)(rows * s.D));
      char what[48];
      std::snprintf(what, sizeof what, "rmsnorm_rope D=%lld rot=%lld", (long long)s.D, (long long)s.rot);
      check(ok && diff == 0, what);
      for (void* b : {x, w, f, r}) release(b, 0, nullptr);
    }
  }

  // ---- argmax: the host loop's index (the largest value, the first one on ties; NaN
  // never wins; nothing above -FLT_MAX gives 0), over lengths a block short of, at and
  // past one block's 256 and a vocabulary's, with ties, NaN and -inf. Two vectors a
  // case, one call after the other: the grid's finished-block count must be back at 0.
  {
    auto host_argmax = [](const float* v, int64_t n) {
      float best = -3.402823466e+38f;
      int64_t bi = 0;
      for (int64_t i = 0; i < n; i++)
        if (v[i] > best) best = v[i], bi = i;
      return bi;
    };
    int64_t bad = 0, cases = 0;
    bool launched = true;
    std::vector<float> base[2];  // the two vectors' random values, prefixes of these
    for (auto& b : base) {
      b.resize(248320);
      for (float& x : b) x = d(g) * 10.0f;
    }
    for (const int64_t n : {1, 255, 257, 70000, 151936, 248320}) {
      for (int pattern = 0; pattern < 6; pattern++) {
        float* cv[2] = {nullptr, nullptr};
        void* v[2] = {alloc(n * 4, &cv[0]), alloc(n * 4, &cv[1])};
        for (int s = 0; s < 2; s++) {
          float* x = cv[s];
          std::copy(base[s].begin(), base[s].begin() + n, x);
          const int64_t r = (int64_t)(g() % (uint32_t)n);
          if (pattern == 1) {  // the maximum several times, the first copy anywhere
            for (int k = 0; k < 4; k++) x[(r + k * (n / 4 + 1)) % n] = 50.0f;
          } else if (pattern == 2) {  // NaN everywhere around a finite maximum
            for (int64_t i = 0; i < n; i += 3) x[i] = std::nanf("");
            if (r - r % 3 + 1 < n) x[r - r % 3 + 1] = 60.0f;  // a slot the NaNs left
          } else if (pattern == 3) {  // nothing above the floor
            for (int64_t i = 0; i < n; i++) x[i] = -INFINITY;
          } else if (pattern == 4) {
            x[0] = 70.0f;
          } else if (pattern == 5) {
            x[n - 1] = 70.0f;
          }
        }
        for (int s = 0; s < 2; s++) {
          int64_t idx = -1;
          launched = launched && tl::gpu::argmax({v[s], 0}, n, &idx);
          bad += idx != host_argmax(cv[s], n);
          cases++;
        }
        release(v[0], 0, nullptr);
        release(v[1], 0, nullptr);
      }
    }
    std::printf("  argmax: %lld of %lld calls disagree with the host loop\n", (long long)bad, (long long)cases);
    check(launched && bad == 0, "argmax");
  }

  // ---- no kernel keeps an array in local memory, as the module was JIT-compiled here:
  // indexing a byte copy of the Q4_K scales put it there and cost the decode GEMV a fifth
  // of its speed. The kernels below still have a frame, each for the reason given; any
  // other one fails this.
  {
    static const char* const allowed[] = {
        "tl_ssm_conv", "tl_ssm_conv_state",       // the conv window, indexed by the runtime width
        "tl_fold", "tl_sum_to", "tl_sum_to_blocked",  // per-axis shape and stride arrays
    };
    auto& cx = context::get();
    unsigned count = 0;
    if (!cx.d.FuncGetAttribute || !cx.d.FuncGetName || !cx.d.ModuleGetFunctionCount ||
        !cx.d.ModuleEnumerateFunctions || cx.d.ModuleGetFunctionCount(&count, cx.mod) != 0) {
      std::printf("  local memory: this driver cannot list a module's kernels, not checked\n");
    } else {
      std::vector<CUfunction> fns(count);
      std::string framed;
      bool listed = cx.d.ModuleEnumerateFunctions(fns.data(), count, cx.mod) == 0;
      for (CUfunction f : fns) {
        int bytes = 0;
        const char* name = nullptr;
        listed = listed && cx.d.FuncGetAttribute(&bytes, 3 /* LOCAL_SIZE_BYTES */, f) == 0 &&
                 cx.d.FuncGetName(&name, f) == 0;
        if (!listed || bytes == 0) continue;
        if (std::none_of(std::begin(allowed), std::end(allowed), [&](const char* a) { return !std::strcmp(a, name); }))
          framed += std::string(" ") + name + " (" + std::to_string(bytes) + " B)";
      }
      std::printf("  local memory: %u kernels, framed outside the list:%s\n", count, framed.empty() ? " none" : framed.c_str());
      check(listed && framed.empty(), "no local memory");
    }
  }

  // The alignment declarations (TL_ALIGN), against the PTX's own entries in one pass:
  // each names a kernel, covers no more than its leading pointer (.u64) parameters and
  // asks a power of two up to a uint4's 16 (a misspelt name or a miscounted list would
  // otherwise go unchecked, or read a scalar argument as a pointer); and every kernel
  // that moves global memory by vectors (ld/st.global .v2/.v4, cp.async) has one, or
  // guards its alignment itself and is listed here.
  {
    static const char* const self_guarded[] = {
        "tl_gemm_bf16_nt", "tl_gemm_bf16_nt_s", "tl_gemm_bf16_nt_sk",  // vecA / vecB in-kernel
    };
    struct entry {
      size_t ptrs = 0;
      bool vectors = false;
    };
    const std::string ptx = tl::cuda::ptx_source_();
    std::unordered_map<std::string, entry> entries;
    for (size_t e = ptx.find(".entry "); e != std::string::npos;) {
      const size_t open = ptx.find('(', e), close = ptx.find(')', open);
      const size_t next = ptx.find(".entry ", close);
      entry& info = entries[ptx.substr(e + 7, open - e - 7)];
      for (size_t p = ptx.find(".param .", open); p < close && !ptx.compare(p + 8, 3, "u64");
           p = ptx.find(".param .", p + 8))
        info.ptrs++;
      const std::string_view body(ptx.data() + close, (next == std::string::npos ? ptx.size() : next) - close);
      for (const char* op : {"ld.global", "st.global"})
        for (size_t at = body.find(op); at != std::string_view::npos && !info.vectors; at = body.find(op, at + 1)) {
          const std::string_view ins = body.substr(at, body.find_first_of(" \t", at) - at);
          info.vectors = ins.find(".v2") != std::string_view::npos || ins.find(".v4") != std::string_view::npos;
        }
      info.vectors = info.vectors || body.find("cp.async.") != std::string_view::npos;
      e = next;
    }
    std::string bad, undeclared;
    for (const auto& [name, bytes] : tl::cuda::align_decls_()) {
      const auto it = entries.find(name);
      bool ok = it != entries.end() && !bytes.empty() && bytes.size() <= it->second.ptrs;
      for (unsigned char b : bytes) ok = ok && b >= 1 && b <= 16 && (b & (b - 1)) == 0;
      if (!ok) bad += " " + name;
    }
    for (const auto& [name, info] : entries)
      if (info.vectors && !tl::cuda::align_decls_().count(name) &&
          std::none_of(std::begin(self_guarded), std::end(self_guarded), [&](const char* g) { return name == g; }))
        undeclared += " " + name;
    std::printf("  alignment declarations: %zu of %zu kernels, malformed:%s, vector access undeclared:%s\n",
                tl::cuda::align_decls_().size(), entries.size(), bad.empty() ? " none" : bad.c_str(),
                undeclared.empty() ? " none" : undeclared.c_str());
    check(!tl::cuda::align_decls_().empty() && bad.empty() && undeclared.empty(),
          "every vector-accessing kernel declares its alignment");
  }

  std::printf(failures ? "CUDA cuda.h: FAILED\n" : "CUDA cuda.h: ALL OK\n");
  return failures ? 1 : 0;
}
