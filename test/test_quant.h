#pragma once

// GGUF block-quantized rows for the tests of the ops that read them (the nn
// layer's GEMV and GEMM, the llm layer's embed_row).

#include "test_util.h"

#include <ggml_quant.h>

namespace {

// N rows of K weights in a GGUF block format (ggml type 12 Q4_K, 14 Q6_K, 8
// Q8_0) as one buffer (for the caller to move to the device), and the same
// rows dequantized by the CPU reference. Random bytes are valid blocks; only
// the half scales are set (a random one could be inf or NaN): Q4_K d, dmin at
// the start, Q6_K d at byte 208, Q8_0 d at the start, each a small positive
// half ~ 2^-5 .. 2^-4.
struct quant_rows {
  array W;               // the blocks, packed into floats
  std::vector<float> w;  // [N,K], dequantized
};
quant_rows random_quant_rows(int type, int64_t N, int64_t K, unsigned seed) {
  const auto t = static_cast<tl::gguf::ggml_type>(type);
  const bool q4 = t == tl::gguf::ggml_type::q4_k, q8 = t == tl::gguf::ggml_type::q8_0;
  uint64_t elems, block_bytes;
  tl::gguf::ggml_block(t, elems, block_bytes);
  const int64_t nb = K / (int64_t)elems, bb = (int64_t)block_bytes;
  std::mt19937 rng(seed);
  std::vector<unsigned char> bytes((size_t)N * nb * bb);
  for (auto& b : bytes) b = (unsigned char)rng();
  auto set_half = [&](size_t at) {
    const uint16_t h = (uint16_t)(0x2800 + (rng() & 0x3ff));
    std::memcpy(&bytes[at], &h, 2);
  };
  for (int64_t i = 0; i < N * nb; i++) {
    if (q4) {
      set_half((size_t)i * bb);
      set_half((size_t)i * bb + 2);
    } else {
      set_half((size_t)i * bb + (q8 ? 0 : 208));
    }
  }
  quant_rows r;
  r.w.resize((size_t)N * K);
  for (int64_t n = 0; n < N; n++)
    tl::gguf::dequant_range(t, bytes.data() + (size_t)n * nb * bb, 0, (size_t)K,
                            &r.w[(size_t)n * K]);
  std::vector<float> packed((bytes.size() + 3) / 4);
  std::memcpy(packed.data(), bytes.data(), bytes.size());
  r.W = array::from(std::move(packed), {(int64_t)((bytes.size() + 3) / 4)});
  return r;
}

}  // namespace
