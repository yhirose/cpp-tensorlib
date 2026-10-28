// Dequantization of the GGML block formats a GGUF stores its weights in, on the
// host: the reference the GPU kernels are checked against, and what a row
// gather (the embedding table) reads. Bit for bit what ggml's
// dequantize_row_q4_K / _q6_K / _q8_0 produce. A format is a run of blocks along
// a row of K weights (K a multiple of the block's 256, or 32 for Q8_0), each
// block self-contained.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "gguf.h"

namespace tl::gguf {

// half -> float, the exact conversion (subnormals included).
inline float half_to_float(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    v = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
  } else {
    v = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
  }
  return sign ? -v : v;
}

// Q4_K: 256 weights in 8 groups of 32. d, dmin: the super-block's scale and min
// (half); scales: 8 six-bit scales and 8 six-bit mins packed in 12 bytes; qs:
// 4-bit weights, group pairs sharing 32 bytes (low nibbles the even group).
// Weight = d * scale_g * q - dmin * min_g.
struct block_q4_k {
  uint16_t d, dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
static_assert(sizeof(block_q4_k) == 144, "block_q4_k");

// Scale j and min j (0..7) of a block's packed 12 bytes.
inline void q4k_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

inline void dequant_q4_k(const block_q4_k* x, float* y, size_t nblocks) {
  for (size_t i = 0; i < nblocks; i++) {
    const float d = half_to_float(x[i].d), mn = half_to_float(x[i].dmin);
    const uint8_t* q = x[i].qs;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, m;
      q4k_scale_min(is + 0, x[i].scales, sc, m);
      const float d1 = d * sc, m1 = mn * m;
      q4k_scale_min(is + 1, x[i].scales, sc, m);
      const float d2 = d * sc, m2 = mn * m;
      for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// Q6_K: 256 weights in 16 groups of 16. ql: the low 4 bits, qh: the high 2,
// scales: a signed byte a group; weight = d * scale_g * (q - 32).
struct block_q6_k {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};
static_assert(sizeof(block_q6_k) == 210, "block_q6_k");

inline void dequant_q6_k(const block_q6_k* x, float* y, size_t nblocks) {
  for (size_t i = 0; i < nblocks; i++) {
    const float d = half_to_float(x[i].d);
    const uint8_t* ql = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* sc = x[i].scales;
    for (int n = 0; n < 256; n += 128) {
      for (int l = 0; l < 32; l++) {
        const int is = l / 16;
        const int q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// Q8_0: 32 weights, a half scale and 32 signed bytes; weight = d * q.
struct block_q8_0 {
  uint16_t d;
  int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");

inline void dequant_q8_0(const block_q8_0* x, float* y, size_t nblocks) {
  for (size_t i = 0; i < nblocks; i++) {
    const float d = half_to_float(x[i].d);
    for (int j = 0; j < 32; j++) *y++ = x[i].qs[j] * d;
  }
}

// Whether the type is one of the block formats above.
inline bool is_quant(ggml_type t) {
  return t == ggml_type::q4_k || t == ggml_type::q6_k || t == ggml_type::q8_0;
}

// Elements [first, first + n) of a quantized tensor widened to f32, `first` and
// `n` multiples of the format's block (256, or 32 for Q8_0).
inline void dequant_range(ggml_type t, const void* data, size_t first, size_t n, float* dst) {
  uint64_t elems = 0, bytes = 0;
  ggml_block(t, elems, bytes);
  const char* base = static_cast<const char*>(data) + (first / elems) * bytes;
  if (t == ggml_type::q4_k) {
    dequant_q4_k(reinterpret_cast<const block_q4_k*>(base), dst, n / elems);
  } else if (t == ggml_type::q6_k) {
    dequant_q6_k(reinterpret_cast<const block_q6_k*>(base), dst, n / elems);
  } else {
    dequant_q8_0(reinterpret_cast<const block_q8_0*>(base), dst, n / elems);
  }
}

}  // namespace tl::gguf
