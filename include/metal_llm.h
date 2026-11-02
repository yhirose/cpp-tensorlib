#pragma once

// The Metal backend's llm layer (layers: gpu_abi.h): its own ops for decoding
// a token, the sampler.
//
// Included by metal.h, after the nn layer.

#include "gpu_abi_llm.h"

namespace tl {
namespace metal {

struct own_llm {
  static bool sample_to(gpu::span a, int64_t n, const gpu::sampling& s, gpu::span counter,
                        gpu::span idx);
};

// Launches of sample_ (the kernel says how they split the work): groups of
// at most `per_group` keys, each keeping k, over the logits and then over the
// groups' keys before (values, then their indices), until one group draws the
// token into idx. Those keys go back and forth between two halves of the
// kernel scratch.
inline bool own_llm::sample_to(gpu::span a, int64_t n, const gpu::sampling& s,
                               gpu::span counter, gpu::span idx) {
  constexpr int64_t per_group = 256 * 16;
  const bool gumbel = s.top_k == 0;
  const int64_t k = gumbel ? 1 : s.top_k;
  const int64_t first_chunk = std::min(detail_::index_chunk_(n), per_group);
  const int64_t half = 8 * k * ((n + first_chunk - 1) / first_chunk);
  const gpu::span scratch = {detail_::scratch_(2 * half), 0};
  if (!scratch) return false;
  const float inv_t = 1.0f / s.temperature;
  gpu::span keys = a, ids = a;
  int64_t m = n, chunk = first_chunk;
  for (uint32_t level = 0;; level++, chunk = per_group) {
    const bool draw = m <= per_group;
    const int64_t groups = draw ? 1 : (m + chunk - 1) / chunk;
    const gpu::span out = draw ? idx : scratch.at(level % 2 * half),
                    out_ids = draw ? idx : out.at(4 * groups * k);
    const gpu::arg args[] = {gpu::in(keys), gpu::in(ids), gpu::inout(counter), gpu::out(out),
                             gpu::out(out_ids)};
    const gpu::sample_params p{static_cast<uint32_t>(m), static_cast<uint32_t>(draw ? m : chunk),
                               static_cast<uint32_t>(k), level > 0, gumbel, draw, s.seed,
                               inv_t, s.top_p};
    if (!dispatch(kop::sample_, args, 5, &p, sizeof(p),
                  {static_cast<uint32_t>(groups), 1, 1, 256, 1, 1, 0}))
      return false;
    if (draw) return true;
    keys = out;
    ids = out_ids;
    m = groups * k;
  }
}

}  // namespace metal
}  // namespace tl
