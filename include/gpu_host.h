#pragma once

// The host reference backend: a "device" that is the CPU, whose kernels are
// plain loops run at once. Build with TENSORLIB_HOST_GPU and gpu.h selects it
// ahead of any real one.
//
// It is here for two reasons. It is the fourth backend, written the way
// docs/backends.md says one is — a device core and kernels that follow the
// kernel ABI, no existing backend's file touched — so the claim that a backend
// is that small is something the build checks rather than something the docs
// say. And it lets a machine with no GPU run the shared layer (gpu_ops.h, the
// launch policy, the census) and the conformance tests, which a CPU fallback
// never enters.
//
// A kernel here is also the plainest statement of what its id computes: the
// views in order, the params struct's fields, one loop. Speed is not a goal;
// the threaded CPU path (cpu.h) is what a build without a GPU should use.

#if defined(TENSORLIB_HOST_GPU)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gpu_abi.h"
#include "profile.h"
#include "types.h"  // bf16 <-> f32

namespace tl {
namespace host_gpu {

using kop = gpu::kop;

// ---- lifecycle. A kernel has run by the time dispatch returns, but the
// backend keeps a device's form: a launch leaves work pending until a flush
// "waits" for it, so the evaluator's flush and barrier paths, and what
// tl::profile records of them, are exercised here as on a real device.
namespace detail_ {
inline bool pending_ = false;
}
inline bool available() { return true; }
inline bool pending() { return detail_::pending_; }
inline void flush() {
  if (!detail_::pending_) return;
  profile::detail::blocked waiting;
  detail_::pending_ = false;
}
inline void cpu_barrier() {
  if (pending()) flush();
}

// ---- memory: unified, so `native` is the address itself and a view's bytes
// are at buf + off.
inline void* alloc(int64_t bytes, float** contents, bool /*host_fill*/ = false) {
  const size_t n = (static_cast<size_t>(bytes > 0 ? bytes : 4) + 63) & ~size_t(63);
#ifdef _WIN32
  void* p = _aligned_malloc(n, 64);
#else
  void* p = std::aligned_alloc(64, n);
#endif
  if (contents) *contents = static_cast<float*>(p);
  return p;
}
inline void release(void* buf, int64_t, float*) {
#ifdef _WIN32
  _aligned_free(buf);
#else
  std::free(buf);
#endif
}
inline void sync_to_host(void*, bool) {}
inline void upload(void* native, const float* src, int64_t n) {
  if (native && src != native && n > 0) {
    std::memcpy(native, src, static_cast<size_t>(n) * sizeof(float));
  }
}

namespace detail_ {

template <class T>
inline T* at(const gpu::arg& a) {
  return reinterpret_cast<T*>(static_cast<char*>(a.s.buf) + a.s.off);
}
template <class P>
inline const P& as(const void* params) {
  return *static_cast<const P*>(params);
}

// out[i] = f(a[i], b[i]) * scale + offset
template <class F>
inline bool binary(const gpu::arg* v, const void* params, F f) {
  const auto& p = as<gpu::ew_params>(params);
  const float *a = at<const float>(v[0]), *b = at<const float>(v[1]);
  float* o = at<float>(v[2]);
  for (uint32_t i = 0; i < p.n; i++) o[i] = f(a[i], b[i]) * p.scale + p.offset;
  return true;
}
template <class F>
inline bool unary(const gpu::arg* v, const void* params, F f) {
  const auto& p = as<gpu::ew_params>(params);
  const float* a = at<const float>(v[0]);
  float* o = at<float>(v[1]);
  for (uint32_t i = 0; i < p.n; i++) o[i] = f(a[i]) * p.scale + p.offset;
  return true;
}
template <class F>
inline bool bcast(const gpu::arg* v, const void* params, F f) {
  const auto& p = as<gpu::bcast_params>(params);
  const float *a = at<const float>(v[0]), *b = at<const float>(v[1]);
  float* o = at<float>(v[2]);
  for (uint32_t r = 0; r < p.m; r++) {
    for (uint32_t c = 0; c < p.n; c++) {
      o[size_t(r) * p.n + c] =
          f(a[size_t(r) * p.ars + size_t(c) * p.acs],
            b[size_t(r) * p.brs + size_t(c) * p.bcs]) * p.scale + p.offset;
    }
  }
  return true;
}
template <class F>
inline bool compare(const gpu::arg* v, const void* params, F f) {
  const auto& p = as<gpu::cmp_params>(params);
  const float *a = at<const float>(v[0]), *b = at<const float>(v[1]);
  float* o = at<float>(v[2]);
  for (uint32_t i = 0; i < p.n; i++) {
    o[i] = f(a[i], b[size_t(i) * p.bstride]) ? 1.0f : 0.0f;
  }
  return true;
}
template <class F>
inline bool scalar(const gpu::arg* v, const void* params, F f) {
  const auto& p = as<gpu::scalar_params>(params);
  const float* a = at<const float>(v[0]);
  float* o = at<float>(v[1]);
  for (uint32_t i = 0; i < p.n; i++) o[i] = f(a[i], p.s) * p.scale + p.offset;
  return true;
}
// One value a row of [rows, cols]: reduce(row) * scale + offset.
template <class F>
inline bool row_reduce(const gpu::arg* v, const void* params, F reduce) {
  const auto& p = as<gpu::reduce_params>(params);
  const float* a = at<const float>(v[0]);
  float* o = at<float>(v[1]);
  for (uint32_t r = 0; r < p.rows; r++) {
    o[r] = reduce(a + size_t(r) * p.cols, p.cols) * p.scale + p.offset;
  }
  return true;
}
// sum(exp(x - max)) and the max: what softmax and logsumexp both reduce to.
// Subtracting the max first keeps huge logits from overflowing.
inline double sum_exp(const float* x, uint32_t n, float* max_out) {
  float mx = x[0];
  for (uint32_t c = 1; c < n; c++) mx = std::max(mx, x[c]);
  *max_out = mx;
  if (std::isinf(mx)) return mx > 0 ? 1.0 : 0.0;  // a row with no finite max
  double s = 0;
  for (uint32_t c = 0; c < n; c++) s += std::exp(double(x[c]) - mx);
  return s;
}
inline float row_logsumexp(const float* x, uint32_t n) {
  float mx;
  const double s = sum_exp(x, n, &mx);
  return std::isinf(mx) ? mx : mx + static_cast<float>(std::log(s));
}
inline bool softmax(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::reduce_params>(params);
  const float* a = at<const float>(v[0]);
  float* o = at<float>(v[1]);
  for (uint32_t r = 0; r < p.rows; r++) {
    const float* x = a + size_t(r) * p.cols;
    float mx;
    const double s = sum_exp(x, p.cols, &mx);
    for (uint32_t c = 0; c < p.cols; c++) {
      const float y = static_cast<float>(std::exp(double(x[c]) - mx) / s);
      o[size_t(r) * p.cols + c] = y * p.scale + p.offset;
    }
  }
  return true;
}
inline bool layer_norm(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::layer_norm_params>(params);
  const float *x = at<const float>(v[0]), *g = at<const float>(v[1]),
              *b = at<const float>(v[2]);
  float* o = at<float>(v[3]);
  for (uint32_t r = 0; r < p.rows; r++) {
    const float* row = x + size_t(r) * p.cols;
    double mean = 0, var = 0;
    for (uint32_t c = 0; c < p.cols; c++) mean += row[c];
    mean /= p.cols;
    for (uint32_t c = 0; c < p.cols; c++) var += (row[c] - mean) * (row[c] - mean);
    const double inv = 1.0 / std::sqrt(var / p.cols + p.eps);
    for (uint32_t c = 0; c < p.cols; c++) {
      const float y = static_cast<float>((row[c] - mean) * inv) * g[c] + b[c];
      o[size_t(r) * p.cols + c] = y * p.scale + p.offset;
    }
  }
  return true;
}
// hout = v * rsqrt(mean(v^2) + eps) * w per row, v = x (+ delta, also stored).
inline bool rmsnorm(const gpu::arg* v, const void* params, const gpu::grid& g,
                    bool add) {
  const auto& p = as<gpu::rmsnorm_params>(params);
  const float* x = at<const float>(v[0]);
  const float* delta = add ? at<const float>(v[1]) : nullptr;
  const float* w = at<const float>(v[add ? 2 : 1]);
  float* xout = add ? at<float>(v[3]) : nullptr;
  float* hout = at<float>(v[add ? 4 : 2]);
  for (uint32_t r = 0; r < g.gx; r++) {  // one group a row
    const size_t base = size_t(r) * p.n;
    double ss = 0;
    for (uint32_t i = 0; i < p.n; i++) {
      const float val = add ? x[base + i] + delta[base + i] : x[base + i];
      if (add) xout[base + i] = val;
      ss += double(val) * val;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / p.n + p.eps));
    const float* src = add ? xout : x;
    for (uint32_t i = 0; i < p.n; i++) hout[base + i] = src[base + i] * inv * w[i];
  }
  return true;
}
inline bool swiglu(const gpu::arg* v, const void* params, const gpu::grid& g) {
  const uint32_t ff = as<gpu::swiglu_params>(params).ff;
  const float* gu = at<const float>(v[0]);
  float* o = at<float>(v[1]);
  for (uint32_t r = 0; r < g.gy; r++) {  // rows ride on the grid's y
    for (uint32_t f = 0; f < ff; f++) {
      const float gate = gu[size_t(r) * 2 * ff + f], up = gu[size_t(r) * 2 * ff + ff + f];
      o[size_t(r) * ff + f] = gate / (1.0f + std::exp(-gate)) * up;
    }
  }
  return true;
}
inline bool index_select(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::gather_params>(params);
  const float *a = at<const float>(v[0]), *idx = at<const float>(v[1]);
  float* o = at<float>(v[2]);
  for (uint32_t i = 0; i < p.n; i++) {
    const size_t row = i / p.row_size, col = i % p.row_size;
    o[i] = a[static_cast<size_t>(idx[row] + 0.5f) * p.row_size + col];
  }
  return true;
}
inline bool gather_axis(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::gather_axis_params>(params);
  const float *src = at<const float>(v[0]), *idx = at<const float>(v[1]);
  float* o = at<float>(v[2]);
  for (uint32_t i = 0; i < p.n; i++) {
    o[i] = src[size_t(i) * p.size + static_cast<size_t>(idx[i] + 0.5f)];
  }
  return true;
}
inline bool xent_bwd(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::xent_bwd_params>(params);
  const float *x = at<const float>(v[0]), *lse = at<const float>(v[1]),
              *tgt = at<const float>(v[2]), *g = at<const float>(v[3]);
  float* o = at<float>(v[4]);
  for (uint32_t i = 0; i < p.n; i++) {
    const size_t row = i / p.cols, col = i % p.cols;
    float prob = std::exp(x[i] - lse[row]);
    if (col == static_cast<size_t>(tgt[row] + 0.5f)) prob -= 1.0f;
    o[i] = prob * g[row];
  }
  return true;
}
inline bool adam_step(const gpu::arg* v, const void* params) {
  const auto& q = as<gpu::adam_params>(params);
  float *p = at<float>(v[0]), *m = at<float>(v[1]), *vv = at<float>(v[2]);
  const float* g = at<const float>(v[3]);
  for (uint32_t i = 0; i < q.n; i++) {
    m[i] = q.b1 * m[i] + (1.0f - q.b1) * g[i];
    vv[i] = q.b2 * vv[i] + (1.0f - q.b2) * g[i] * g[i];
    p[i] -= (m[i] * q.lr_over_bc1) / (std::sqrt(vv[i] * q.inv_bc2) + q.eps);
  }
  return true;
}
// y[n] = a[k] . W[n, k], the weight row-major: bf16, or int4 with a scale a
// group of `group` weights (8 nibbles a word, low nibble first, offset 8).
inline bool gemv_bf16_row(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::gemv_row_params>(params);
  const float* a = at<const float>(v[0]);
  const uint16_t* W = at<const uint16_t>(v[1]);
  float* y = at<float>(v[2]);
  for (uint32_t r = 0; r < p.n; r++) {
    double acc = 0;
    for (uint32_t c = 0; c < p.k; c++) acc += double(a[c]) * bf16_to_f32(W[size_t(r) * p.k + c]);
    y[r] = static_cast<float>(acc);
  }
  return true;
}
inline bool gemv_q4(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::gemv_q4_params>(params);
  const float* a = at<const float>(v[0]);
  const uint32_t* qw = at<const uint32_t>(v[1]);
  const float* scales = at<const float>(v[2]);
  float* y = at<float>(v[3]);
  for (uint32_t r = 0; r < p.n; r++) {
    double acc = 0;
    for (uint32_t c = 0; c < p.k; c++) {
      const uint32_t word = qw[size_t(r) * (p.k / 8) + c / 8];
      const int q = static_cast<int>((word >> ((c % 8) * 4)) & 0xFu) - 8;
      acc += double(a[c]) * (scales[size_t(r) * (p.k / p.group) + c / p.group] * q);
    }
    y[r] = static_cast<float>(acc);
  }
  return true;
}
// The KV cache's writes, f32 or bf16: heads ride on the grid's x, a fill's T
// rows on its y, and the head dim is the group's thread count.
template <class KT, class Narrow>
inline bool kv_write(const gpu::arg* v, const gpu::grid& g, uint32_t kv_stride,
                     uint32_t pos0, uint32_t T, Narrow narrow) {
  KT *Kc = at<KT>(v[0]), *Vc = at<KT>(v[1]);
  const float *k = at<const float>(v[2]), *val = at<const float>(v[3]);
  const uint32_t D = g.tx;
  for (uint32_t h = 0; h < g.gx; h++) {
    for (uint32_t t = 0; t < T; t++) {
      for (uint32_t d = 0; d < D; d++) {
        const size_t dst = size_t(h) * kv_stride + size_t(pos0 + t) * D + d;
        const size_t src = (size_t(h) * T + t) * D + d;
        Kc[dst] = narrow(k[src]);
        Vc[dst] = narrow(val[src]);
      }
    }
  }
  return true;
}
inline bool kv_append(const gpu::arg* v, const void* params, const gpu::grid& g,
                      bool bf16) {
  const auto& p = as<gpu::kv_append_params>(params);
  return bf16 ? kv_write<uint16_t>(v, g, p.kv_stride, p.pos, 1, f32_to_bf16)
              : kv_write<float>(v, g, p.kv_stride, p.pos, 1, [](float f) { return f; });
}
inline bool kv_fill(const gpu::arg* v, const void* params, const gpu::grid& g,
                    bool bf16) {
  const auto& p = as<gpu::kv_fill_params>(params);
  return bf16 ? kv_write<uint16_t>(v, g, p.kv_stride, p.pos0, p.T, f32_to_bf16)
              : kv_write<float>(v, g, p.kv_stride, p.pos0, p.T, [](float f) { return f; });
}
inline bool merge_heads(const gpu::arg* v, const void* params) {
  const auto& p = as<gpu::merge_heads_params>(params);
  const float* src = at<const float>(v[0]);
  float* dst = at<float>(v[1]);
  for (uint32_t h = 0; h < p.H; h++)
    for (uint32_t t = 0; t < p.T; t++)
      for (uint32_t d = 0; d < p.D; d++)
        dst[(size_t(t) * p.H + h) * p.D + d] = src[(size_t(h) * p.T + t) * p.D + d];
  return true;
}

}  // namespace detail_

// ---- launch: the kernel table and the kernels in one switch. An id with no
// case declines, and the op above falls back to the CPU.
inline bool kernel_(kop k, const gpu::arg* v, const void* params,
                    const gpu::grid& g) {
  namespace d = detail_;
  switch (k) {
    case kop::add: return d::binary(v, params, [](float a, float b) { return a + b; });
    case kop::sub: return d::binary(v, params, [](float a, float b) { return a - b; });
    case kop::mul: return d::binary(v, params, [](float a, float b) { return a * b; });
    case kop::div: return d::binary(v, params, [](float a, float b) { return a / b; });
    case kop::pow_: return d::binary(v, params, [](float a, float b) { return std::pow(a, b); });
    case kop::exp_: return d::unary(v, params, [](float a) { return std::exp(a); });
    case kop::log_: return d::unary(v, params, [](float a) { return std::log(a); });
    case kop::sqrt_: return d::unary(v, params, [](float a) { return std::sqrt(a); });
    case kop::sigmoid: return d::unary(v, params, [](float a) { return 1.0f / (1.0f + std::exp(-a)); });
    case kop::relu: return d::unary(v, params, [](float a) { return a > 0 ? a : 0.0f; });
    case kop::affine: return d::unary(v, params, [](float a) { return a; });
    case kop::tanh_: return d::unary(v, params, [](float a) { return std::tanh(a); });
    case kop::sin_: return d::unary(v, params, [](float a) { return std::sin(a); });
    case kop::cos_: return d::unary(v, params, [](float a) { return std::cos(a); });
    case kop::badd: return d::bcast(v, params, [](float a, float b) { return a + b; });
    case kop::bsub: return d::bcast(v, params, [](float a, float b) { return a - b; });
    case kop::bmul: return d::bcast(v, params, [](float a, float b) { return a * b; });
    case kop::bdiv: return d::bcast(v, params, [](float a, float b) { return a / b; });
    case kop::bpow: return d::bcast(v, params, [](float a, float b) { return std::pow(a, b); });
    case kop::gt_: return d::compare(v, params, [](float a, float b) { return a > b; });
    case kop::lt_: return d::compare(v, params, [](float a, float b) { return a < b; });
    case kop::ge_: return d::compare(v, params, [](float a, float b) { return a >= b; });
    case kop::le_: return d::compare(v, params, [](float a, float b) { return a <= b; });
    case kop::eq_: return d::compare(v, params, [](float a, float b) { return a == b; });
    case kop::ne_: return d::compare(v, params, [](float a, float b) { return a != b; });
    case kop::clamp_: {
      const auto& p = d::as<gpu::clamp_params>(params);
      const float* a = d::at<const float>(v[0]);
      float* o = d::at<float>(v[1]);
      for (uint32_t i = 0; i < p.n; i++) o[i] = std::min(std::max(a[i], p.lo), p.hi);
      return true;
    }
    case kop::pow_s_: return d::scalar(v, params, [](float a, float s) { return std::pow(a, s); });
    case kop::gt_s_: return d::scalar(v, params, [](float a, float s) { return a > s ? 1.0f : 0.0f; });
    case kop::lt_s_: return d::scalar(v, params, [](float a, float s) { return a < s ? 1.0f : 0.0f; });
    case kop::ge_s_: return d::scalar(v, params, [](float a, float s) { return a >= s ? 1.0f : 0.0f; });
    case kop::le_s_: return d::scalar(v, params, [](float a, float s) { return a <= s ? 1.0f : 0.0f; });
    case kop::eq_s_: return d::scalar(v, params, [](float a, float s) { return a == s ? 1.0f : 0.0f; });
    case kop::ne_s_: return d::scalar(v, params, [](float a, float s) { return a != s ? 1.0f : 0.0f; });
    case kop::softmax: return d::softmax(v, params);
    case kop::row_sum:
      return d::row_reduce(v, params, [](const float* x, uint32_t n) {
        double s = 0;
        for (uint32_t c = 0; c < n; c++) s += x[c];
        return static_cast<float>(s);
      });
    case kop::row_max:
      return d::row_reduce(v, params, [](const float* x, uint32_t n) {
        return *std::max_element(x, x + n);
      });
    case kop::row_logsumexp_: return d::row_reduce(v, params, d::row_logsumexp);
    case kop::layer_norm_: return d::layer_norm(v, params);
    case kop::index_select: return d::index_select(v, params);
    case kop::gather_axis_: return d::gather_axis(v, params);
    case kop::xent_bwd_: return d::xent_bwd(v, params);
    case kop::adam_step_: return d::adam_step(v, params);
    case kop::rmsnorm_: return d::rmsnorm(v, params, g, false);
    case kop::add_rmsnorm_: return d::rmsnorm(v, params, g, true);
    case kop::swiglu_: return d::swiglu(v, params, g);
    case kop::gemv_bf16_row_: return d::gemv_bf16_row(v, params);
    case kop::gemv_q4_: return d::gemv_q4(v, params);
    case kop::kv_append_: return d::kv_append(v, params, g, false);
    case kop::kv_append_bf16_: return d::kv_append(v, params, g, true);
    case kop::kv_fill_: return d::kv_fill(v, params, g, false);
    case kop::kv_fill_bf16_: return d::kv_fill(v, params, g, true);
    case kop::merge_heads_: return d::merge_heads(v, params);
    default: return false;
  }
}

// A kernel that ran is a row under tl::profile, as on any backend.
inline bool dispatch(kop k, const gpu::arg* v, size_t /*n*/, const void* params,
                     size_t /*params_bytes*/, const gpu::grid& g) {
  detail_::pending_ = true;  // declining leaves it set too: a flush then waits on nothing
  if (!kernel_(k, v, params, g)) return false;
  gpu::launched(kop_name(k));
  return true;
}

// ---- the ops this backend runs its own way: the ones with no single-kernel
// form in gpu_ops.h that the conformance test expects of every backend. Each
// is the definition of its op, as a loop.
struct own {
  // C(m,n) = (A @ B) * scale + offset; a transposed operand is read in place.
  static bool gemm(gpu::span a, int64_t lda, bool ta, gpu::span b, int64_t ldb,
                   bool tb, gpu::span out, int64_t m, int64_t n, int64_t k,
                   float scale, float offset) {
    const float *A = view<const float>(a), *B = view<const float>(b);
    float* C = view<float>(out);
    detail_::pending_ = true;
    for (int64_t i = 0; i < m; i++) {
      for (int64_t j = 0; j < n; j++) {
        double acc = 0;
        for (int64_t q = 0; q < k; q++) {
          acc += double(ta ? A[q * lda + i] : A[i * lda + q]) *
                 (tb ? B[j * ldb + q] : B[q * ldb + j]);
        }
        C[i * n + j] = static_cast<float>(acc) * scale + offset;
      }
    }
    gpu::launched("gemm");
    return true;
  }

  // `a` into a zero buffer of out_shape, shifted by `before` along `axis`.
  static bool pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                  const int64_t* out_shape, int rank, int axis, int64_t before,
                  int64_t n, int64_t out_n) {
    const float* A = view<const float>(a);
    float* O = view<float>(out);
    std::fill(O, O + out_n, 0.0f);
    int64_t inner = 1;
    for (int d = axis + 1; d < rank; d++) inner *= a_shape[d];
    const int64_t a_axis = a_shape[axis], o_axis = out_shape[axis];
    for (int64_t i = 0; i < n; i++) {
      const int64_t outer = i / (a_axis * inner), rest = i % (a_axis * inner);
      O[outer * o_axis * inner + before * inner + rest] = A[i];
    }
    gpu::launched("pad");
    return true;
  }

  // out row idx[i] += values row i, for k rows, into a zeroed out.
  static bool index_add(gpu::span idx, gpu::span values, gpu::span out,
                        int64_t row_size, int64_t k, int64_t out_n) {
    const float *I = view<const float>(idx), *V = view<const float>(values);
    float* O = view<float>(out);
    std::fill(O, O + out_n, 0.0f);
    for (int64_t i = 0; i < k; i++) {
      const int64_t row = static_cast<int64_t>(I[i] + 0.5f);
      for (int64_t c = 0; c < row_size; c++) O[row * row_size + c] += V[i * row_size + c];
    }
    gpu::launched("index_add");
    return true;
  }

  // out[i, idx[i]] = values[i], zero elsewhere, over a trailing axis of `size`.
  static bool scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                              int64_t n, int64_t size) {
    const float *I = view<const float>(idx), *V = view<const float>(values);
    float* O = view<float>(out);
    std::fill(O, O + n * size, 0.0f);
    for (int64_t i = 0; i < n; i++) O[i * size + static_cast<int64_t>(I[i] + 0.5f)] = V[i];
    gpu::launched("scatter_to_axis");
    return true;
  }

 private:
  template <class T>
  static T* view(gpu::span s) {
    return reinterpret_cast<T*>(static_cast<char*>(s.buf) + s.off);
  }
};

// ---- what the shared layer may assume.
struct traits {
  static constexpr bool cells_2d = false;  // a cell is read from a flat index
  static constexpr bool times_launches = false;  // a row is counted, not timed
};
struct caps {
  // The decoder's single-kernel ops are here, but not attention, rope or the
  // GEMVs a model also needs, so a model keeps to the array ops.
  static constexpr bool model_path = false;
  static constexpr bool graph_capture = false;
  static constexpr bool row_gemv = true;
  static constexpr bool bf16_gemm = false;
};
using graph_exec = void*;
inline bool graph_available() { return false; }
inline bool capture_begin() { return false; }
inline graph_exec capture_end() { return nullptr; }
inline bool graph_launch(graph_exec) { return false; }
inline void graph_destroy(graph_exec) {}
inline void upload_u32(void*, unsigned) {}
inline int64_t attn_dpos_partials_bytes(int64_t, int64_t, int64_t) { return 0; }

}  // namespace host_gpu
}  // namespace tl

#endif  // TENSORLIB_HOST_GPU
