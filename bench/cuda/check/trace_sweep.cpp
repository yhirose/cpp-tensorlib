// What the launch trace (tools/cuda_trace) cannot reach through the test suite
// and the checkers: the kernels no test selects, and the calls no evaluator
// makes — above all a non-zero OUTPUT offset, which array.h never passes but
// the contract allows. It checks nothing. It exists to be traced, so that a
// change to cuda.h's host side shows up in a diff on a machine with no GPU.
#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "gpu.h"  // the shared ops (tl::gpu) resolve to cuda here

#include <cstdio>
#include <vector>

namespace cu = tl::cuda;
using kop = tl::gpu::kop;

namespace {

struct buf {
  void* native = nullptr;
  float* host = nullptr;
  int64_t bytes = 0;
  explicit buf(int64_t b) : bytes(b) { native = cu::alloc(b, &host); }
  ~buf() { cu::release(native, bytes, host); }
  buf(const buf&) = delete;
  operator void*() const { return native; }
};

constexpr int64_t kOff = 64;  // bytes; 16B-aligned so the tiled GEMM still takes it

void elementwise() {
  const int64_t n = 1000;
  buf a(n * 4 + kOff), b(n * 4 + kOff), o(n * 4 + kOff);
  for (kop op : {kop::add, kop::sub, kop::mul, kop::div, kop::pow_}) {
    tl::gpu::binary(op, {a, kOff}, {b, 0}, {o, kOff}, n, 2.0f, 1.0f);
  }
  for (kop op : {kop::badd, kop::bsub, kop::bmul, kop::bdiv, kop::bpow}) {
    tl::gpu::binary_bcast(op, {a, kOff}, 40, 1, {b, 0}, 0, 1, {o, kOff}, 25, 40,
                          1.0f, 0.0f);
  }
  const int64_t shape[3] = {5, 8, 25}, as[3] = {200, 25, 1}, bs[3] = {0, 25, 1};
  for (kop op : {kop::badd, kop::bsub, kop::bmul, kop::bdiv, kop::bpow}) {
    cu::binary_bcast_nd(op, a, kOff, as, b, 0, bs, o, kOff, shape, 3, n, 1.0f, 0.0f);
  }
  using cu::cmp_op;
  for (cmp_op op : {cmp_op::gt, cmp_op::lt, cmp_op::ge, cmp_op::le, cmp_op::eq,
                    cmp_op::ne}) {
    tl::gpu::compare(op, {a, kOff}, {b, 0}, {o, kOff}, n, 1);
  }
}

void gemm_bias() {
  struct shape { int64_t m, n, k; };
  for (shape s : {shape{8, 8, 8}, {64, 64, 64}, {256, 512, 512},
                  {256, 2048, 512}, {512, 896, 896}}) {
    buf a(s.m * s.k * 4 + kOff), b(s.k * s.n * 4 + kOff), bias(s.n * 4),
        o(s.m * s.n * 4 + kOff);
    for (int layout = 0; layout < 4; layout++) {
      const bool ta = layout & 1, tb = layout & 2;
      cu::gemm_bias(a, kOff, ta ? s.m : s.k, ta, b, kOff, tb ? s.k : s.n, tb,
                    bias, 0, o, kOff, s.m, s.n, s.k, 1.0f, 0.0f);
    }
  }
}

// The four ops that zero their output before scattering into it.
void zero_then_scatter() {
  const int64_t a_shape[2] = {4, 6}, out_shape[2] = {4, 10};
  buf a(24 * 4 + kOff), o(40 * 4 + kOff);
  cu::pad(a, kOff, o, kOff, a_shape, out_shape, 2, 1, 2, 24, 40);

  const int64_t w_shape[3] = {4, 4, 3}, f_shape[2] = {4, 6};
  buf w(48 * 4 + kOff), f(24 * 4 + kOff);
  cu::fold(w, kOff, f, kOff, w_shape, f_shape, 3, 1, 1, 48, 24);

  buf idx(8 * 4 + kOff), vals(8 * 16 * 4 + kOff), table(32 * 16 * 4 + kOff);
  cu::index_add(idx, kOff, vals, kOff, table, kOff, 16, 8, 32 * 16);
  cu::scatter_to_axis(idx, kOff, vals, kOff, table, kOff, 8, 16);
}

void llm() {
  for (int64_t m : {8, 96, 512}) {
    for (int64_t n : {896, 4864}) {
      const int64_t k = 896;
      buf a(m * k * 4), B(n * k * 2), o(m * n * 4);
      cu::gemm_bf16_nt(a, B, o, m, n, k);
    }
  }
  for (int64_t D : {64, 128}) {
    const int64_t hq = 14, hkv = 2, kv_max = 4096;
    buf q(hq * D * 4), K(hkv * kv_max * D * 2), V(hkv * kv_max * D * 2),
        o(hq * D * 4);
    for (int64_t ctx : {17, 700, 4000}) {
      cu::attn_decode(q, K, V, o, hq, hkv, ctx, kv_max, D, 0.125f, true);
    }
  }
}

// The graph-capture group: device-position variants, recorded then replayed.
void capture(int64_t D) {
  const int64_t hq = 14, hkv = 2, kv_max = 2048;
  buf pos(4), x(hq * D * 4), xo(hq * D * 4), knew(hkv * D * 4), vnew(hkv * D * 4),
      K(hkv * kv_max * D * 4), V(hkv * kv_max * D * 4), o(hq * D * 4),
      partials(cu::attn_dpos_partials_bytes(hq, kv_max, D));
  cu::upload_u32(pos, 5);
  if (!cu::capture_begin()) return;
  cu::rope_dpos(x, xo, hq, 1, D, pos, 10000.0f);
  cu::kv_append_dpos(K, V, knew, vnew, pos, kv_max, hkv, D);
  cu::attn_decode_dpos(xo, K, V, o, hq, hkv, pos, kv_max, D, 0.125f, partials);
  cu::incr_u32(pos);
  auto graph = cu::capture_end();
  cu::graph_launch(graph);
  cu::flush();
  cu::graph_destroy(graph);
}

}  // namespace

int main() {
  if (!cu::available()) return std::printf("no CUDA driver: nothing to trace\n"), 0;
  elementwise();
  gemm_bias();
  zero_then_scatter();
  llm();
  capture(64);
  capture(128);
  cu::flush();
  return 0;
}
