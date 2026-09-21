#pragma once

// tl::kv_cache — a persistent, device-resident KV cache: K and V as
// [n_kv_heads, max_ctx, D] storage plus the running position. Decode is
// stateful, which the immutable node model does not fit, so it lives outside
// the lazy graph and reaches its backend through the gpu:: facade — one
// definition serves CUDA, Metal and WebGPU, and each buffer is a tl::storage,
// which returns to the backend's pool by itself.
//
// append() writes one token's k/v and advances; attn() runs the fused decode
// attention over the cached prefix [0, pos); prefill() appends a block of T
// tokens and runs the causal attention over it, so a long prompt runs in
// chunks and a later turn extends a live cache. The cache is f32, or bf16
// (half the bytes every step streams) with init(..., dtype::bf16); q, out and
// the arithmetic stay f32.
//
// The *_dpos variants read the position from a device scalar instead of the
// host `pos`, so a captured forward replays at the advancing position (CUDA
// graph capture, gpu::caps::graph_capture). They leave the host `pos` alone;
// the caller keeps it in step out-of-band.

#include <gpu.h>
#include <storage.h>
#include <types.h>

#include <cstdint>

namespace tl {

struct kv_cache {
  storage K, V;
  int64_t n_kv_heads = 0, max_ctx = 0, D = 0, pos = 0;
  bool kv_bf16 = false;

  bool init(int64_t kv_heads, int64_t maxctx, int64_t d,
            dtype kv_dt = dtype::f32) {
    n_kv_heads = kv_heads;
    max_ctx = maxctx;
    D = d;
    pos = 0;
    kv_bf16 = kv_dt == dtype::bf16;
    K = storage::make(n_kv_heads * max_ctx * D, kv_dt);
    V = storage::make(n_kv_heads * max_ctx * D, kv_dt);
    return K.native && V.native;  // a heap fallback has no device buffer
  }

  // k_new/v_new: [n_kv_heads, D] device views (this step's projected k, v).
  bool append(gpu::span k_new, gpu::span v_new) {
    if (pos >= max_ctx) return false;
    if (!gpu::kv_append(K.device_span(), V.device_span(), k_new, v_new, pos,
                        max_ctx, n_kv_heads, D, kv_bf16)) {
      return false;
    }
    pos++;
    return true;
  }

  // q/out: [n_q_heads, D] device views. Attends over the cached prefix.
  bool attn(gpu::span q, gpu::span out, int64_t n_q_heads, float scale) {
    return gpu::attn_decode(q, K.device_span(), V.device_span(), out, n_q_heads,
                            n_kv_heads, pos, max_ctx, D, scale, kv_bf16);
  }

  // T tokens at once: k_src/v_src [n_kv_heads, T, D] appended at `pos`, and
  // the causal attention of q/out [n_q_heads, T, D] over everything cached
  // before them. Leaves pos advanced by T.
  bool prefill(gpu::span q, gpu::span k_src, gpu::span v_src, gpu::span out,
               int64_t T, int64_t n_q_heads, float scale) {
    if (T <= 0 || pos + T > max_ctx) return false;
    if (!gpu::kv_fill(K.device_span(), V.device_span(), k_src, v_src, T, max_ctx,
                      n_kv_heads, D, kv_bf16, pos)) {
      return false;
    }
    const int64_t p0 = pos;
    pos += T;
    return gpu::attn_prefill(q, K.device_span(), V.device_span(), out,
                             n_q_heads, n_kv_heads, T, max_ctx, D, scale, kv_bf16,
                             p0);
  }

  // Graph-capture forms (f32 cache only): the position is *d_pos.
  bool append_dpos(gpu::span k_new, gpu::span v_new, gpu::span d_pos) {
    return gpu::kv_append_dpos(K.device_span(), V.device_span(), k_new, v_new,
                               d_pos, max_ctx, n_kv_heads, D);
  }
  // The split-KV partials a captured graph bakes in are this cache's own (a
  // shared scratch could be freed under a live graph), sized once from the
  // capacity on the first call.
  storage dpos_partials;
  bool attn_dpos(gpu::span q, gpu::span out, int64_t n_q_heads,
                 gpu::span d_pos, float scale) {
    if (!dpos_partials.native) {
      const int64_t bytes =
          gpu::attn_dpos_partials_bytes(n_q_heads, max_ctx, D);
      if (bytes <= 0) return false;
      dpos_partials = storage::make(bytes / 4);
      if (!dpos_partials.native) return false;
    }
    return gpu::attn_decode_dpos(q, K.device_span(), V.device_span(), out,
                                 n_q_heads, n_kv_heads, d_pos, max_ctx, D, scale,
                                 dpos_partials.device_span());
  }
};

}  // namespace tl
