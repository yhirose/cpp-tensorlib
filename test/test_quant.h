#pragma once

// GGUF block-quantized rows for the tests of the ops that read them (the nn
// layer's GEMV and GEMM, the llm layer's embed_row).

#include "test_util.h"

#include <ggml_quant.h>

#include <cstddef>

namespace {

// The GGUF block formats the kernels read, a row each: the ggml type, its name, the
// byte alignment its GEMVs need of a weight (4 for the word loads into a Q4_K or Q5_K
// block, 2 for the others' halves), and the byte offsets of a block's half scales (its
// d, and its dmin or m where it has one). Every test over the formats takes them from
// here.
struct block_format {
  int type;
  const char* name;
  int align;
  size_t halves[2];
  int nhalves;
};
namespace gq = tl::gguf;
inline constexpr block_format block_formats[] = {
    {12, "Q4_K", 4, {offsetof(gq::block_q4_k, d), offsetof(gq::block_q4_k, dmin)}, 2},
    {14, "Q6_K", 2, {offsetof(gq::block_q6_k, d)}, 1},
    {8, "Q8_0", 2, {offsetof(gq::block_q8_0, d)}, 1},
    {6, "Q5_0", 2, {offsetof(gq::block_q5_0, d)}, 1},
    {13, "Q5_K", 4, {offsetof(gq::block_q5_k, d), offsetof(gq::block_q5_k, dmin)}, 2},
    {2, "Q4_0", 2, {offsetof(gq::block_q4_0, d)}, 1},
    {3, "Q4_1", 2, {offsetof(gq::block_q4_1, d), offsetof(gq::block_q4_1, m)}, 2},
};
// A row of K weights for a test over `type`: K itself for a 256-weight format, 35
// blocks for a 32-weight one (an odd count: a row of 18-, 22- or 34-byte blocks puts
// every other row 2 bytes off 4).
inline int64_t test_row_k(int type, int64_t K) {
  uint64_t elems = 0, bytes = 0;
  tl::gguf::ggml_block(static_cast<tl::gguf::ggml_type>(type), elems, bytes);
  return elems == 32 ? 35 * 32 : K;
}

// N rows of K weights in a GGUF block format (a row of block_formats) as one buffer
// (for the caller to move to the device), and the same rows dequantized by the CPU
// reference. Random bytes are valid blocks; only the half scales are set (a random
// one could be inf or NaN), each a small positive half ~ 2^-5 .. 2^-4.
struct quant_rows {
  array W;               // the blocks, packed into floats
  std::vector<float> w;  // [N,K], dequantized
};
quant_rows random_quant_rows(int type, int64_t N, int64_t K, unsigned seed) {
  const auto t = static_cast<tl::gguf::ggml_type>(type);
  const block_format* f = nullptr;
  for (const auto& b : block_formats)
    if (b.type == type) f = &b;
  REQUIRE(f);
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
  for (int64_t i = 0; i < N * nb; i++)
    for (int h = 0; h < f->nhalves; h++) set_half((size_t)i * bb + f->halves[h]);
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

// Whether the backend reads `type`'s blocks in rows of k (gpu::quant_reads), held to
// what it must: a backend with the block kernels reads every format of
// block_formats, so a test skipping a format it does not read skips none it must.
inline bool reads_quant(int type, int64_t k) {
  const bool reads = tl::gpu::quant_reads(type, k);
  CHECK(reads);
  return reads;
}

}  // namespace
