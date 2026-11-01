#pragma once

// The llm layer of the GPU ops (layers: gpu_abi.h): what keeps a decoder's
// state -- the KV cache's writes, and the graph-capture forms that read the
// decode position from the device. Optional like the nn layer: the cache
// writes have a composition of the core ops, the capture forms only their
// backend's kernel.
//
// Included by gpu.h, after gpu_ops_nn.h.

#include <cstdint>

#include "gpu_abi_llm.h"
#include "gpu_ops_nn.h"

namespace tl {
namespace gpu {

namespace generic {

// A row into the cache is a contiguous copy per head.
inline bool kv_append(span Kc, span Vc, span k_new, span v_new, int64_t pos,
                      int64_t kv_max, int64_t n_kv_heads, int64_t D) {
  for (int64_t h = 0; h < n_kv_heads; h++) {
    const int64_t at = (h * kv_max + pos) * D * 4, from = h * D * 4;
    if (!unary(kop::affine, k_new.at(from), Kc.at(at), D, 1.0f, 0.0f) ||
        !unary(kop::affine, v_new.at(from), Vc.at(at), D, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

inline bool kv_fill(span Kc, span Vc, span K, span V, int64_t T, int64_t kv_max,
                    int64_t n_kv_heads, int64_t D, int64_t pos0) {
  for (int64_t h = 0; h < n_kv_heads; h++) {
    const int64_t at = (h * kv_max + pos0) * D * 4, from = h * T * D * 4;
    if (!unary(kop::affine, K.at(from), Kc.at(at), T * D, 1.0f, 0.0f) ||
        !unary(kop::affine, V.at(from), Vc.at(at), T * D, 1.0f, 0.0f)) {
      return false;
    }
  }
  return true;
}

// The f32 at src, after a drain: copied into a buffer of this layer's own and
// brought to the host, as generic::argmax reads its vector.
inline bool read_now(span src, float* out) {
  scratch c(1);
  if (!c || !unary(kop::affine, src, c, 1, 1.0f, 0.0f)) return false;
  cpu_barrier();
  sync_to_host(c.buf, false);
  *out = c.contents[0];
  return true;
}

}  // namespace generic

// One decode step's k, v (each [n_kv_heads, D]) into row `pos` of a
// [n_kv_heads, kv_max, D] cache, f32, or f16 with kv_f16 (where caps::kv_f16).
inline bool kv_append(span Kc, span Vc, span k_new, span v_new, int64_t pos,
                      int64_t kv_max, int64_t n_kv_heads, int64_t D,
                      bool kv_f16 = false) {
  if (n_kv_heads <= 0 || D <= 0) return false;
  // The kernels run a head as one group of D threads; a head wider than a group
  // can be takes the composition.
  return (D <= policy::max_group_threads &&
          launch(kv_f16 ? kop::kv_append_f16_ : kop::kv_append_,
                 {out(Kc), out(Vc), in(k_new), in(v_new)},
                 kv_append_params{static_cast<uint32_t>(pos),
                                  static_cast<uint32_t>(kv_max * D)},
                 policy::per_head(n_kv_heads, 1, D))) ||
         (!kv_f16 &&
          generic::kv_append(Kc, Vc, k_new, v_new, pos, kv_max, n_kv_heads, D));
}

// A prefill's k, v (each [n_kv_heads, T, D]) into cache rows [pos0, pos0 + T).
inline bool kv_fill(span Kc, span Vc, span K, span V, int64_t T, int64_t kv_max,
                    int64_t n_kv_heads, int64_t D, bool kv_f16 = false,
                    int64_t pos0 = 0) {
  if (n_kv_heads <= 0 || D <= 0 || T <= 0) return false;
  return (D <= policy::max_group_threads &&  // a group a head, as in kv_append
          launch(kv_f16 ? kop::kv_fill_f16_ : kop::kv_fill_,
                 {out(Kc), out(Vc), in(K), in(V)},
                 kv_fill_params{static_cast<uint32_t>(T),
                                static_cast<uint32_t>(kv_max * D),
                                static_cast<uint32_t>(pos0)},
                 policy::per_head(n_kv_heads, T, D))) ||
         (!kv_f16 &&
          generic::kv_fill(Kc, Vc, K, V, T, kv_max, n_kv_heads, D, pos0));
}

// ---- the graph-capture forms (caps::graph_capture): a decode step whose
// position is a device scalar, so one captured step replays as the cache
// advances. `partials` is attn_dpos_partials_bytes() of scratch.
TL_GPU_DETECT_OWN(rope_dpos)
template <class Own = own>
inline bool rope_dpos(span x, span o, int64_t rows, int64_t T, int64_t D,
                      span d_pos, float base, span bias = {}, int64_t rot = 0) {
  if constexpr (detail::owns_rope_dpos<Own>::value) {
    return detail::ran(Own::rope_dpos(x, o, rows, T, D, d_pos, base, bias, rot));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(kv_append_dpos)
template <class Own = own>
inline bool kv_append_dpos(span Kc, span Vc, span k_new, span v_new, span d_pos,
                           int64_t kv_max, int64_t n_kv_heads, int64_t D, bool kv_f16 = false) {
  if constexpr (detail::owns_kv_append_dpos<Own>::value) {
    return detail::ran(Own::kv_append_dpos(Kc, Vc, k_new, v_new, d_pos, kv_max, n_kv_heads, D, kv_f16));
  } else {
    return false;
  }
}

TL_GPU_DETECT_OWN(attn_decode_dpos)
template <class Own = own>
inline bool attn_decode_dpos(span q, span K, span V, span o, int64_t n_q_heads,
                             int64_t n_kv_heads, span d_pos, int64_t kv_max,
                             int64_t D, float scale, span partials, bool kv_f16 = false) {
  if constexpr (detail::owns_attn_decode_dpos<Own>::value) {
    return detail::ran(Own::attn_decode_dpos(q, K, V, o, n_q_heads, n_kv_heads, d_pos, kv_max, D, scale, partials, kv_f16));
  } else {
    return false;
  }
}

// *d_pos += 1 on the device.
TL_GPU_DETECT_OWN(incr_u32)
template <class Own = own>
inline bool incr_u32(span d_pos) {
  if constexpr (detail::owns_incr_u32<Own>::value) {
    return detail::ran(Own::incr_u32(d_pos));
  } else {
    return false;
  }
}

// ---- a decode step run ahead: the next step is queued before the host reads
// this one's token, which stays on the device (an f32 index, exact up to 2^24)
// for the next step's input. Each answers false where its backend has no form
// of its own, and a decoder then reads the token back and takes the host path.

// The argmax of a length-n vector (the smallest index on ties, as argmax) as an
// f32 at idx[0], with no read-back and no drain.
TL_GPU_DETECT_OWN(argmax_to)
template <class Own = own>
inline bool argmax_to(span a, int64_t n, span idx) {
  if (!a || !idx || n <= 0 || n > policy::max_f32_index) return false;
  if constexpr (detail::owns_argmax_to<Own>::value) {
    return detail::ran(Own::argmax_to(a, n, idx));
  } else {
    return false;
  }
}

// The f32 src holds once the work queued so far has run: read_later takes it
// behind that work and submits the work without waiting, and finish_read
// waits for it, so the host can queue the next step in between (which may
// write src, or let it go). A backend without the form drains and reads at
// once.
TL_GPU_DETECT_OWN(read_later)
template <class Own = own>
inline bool read_later(span src, pending_read& r) {
  r = {};
  if (!src) return false;
  if constexpr (detail::owns_read_later<Own>::value) {
    if (Own::read_later(src, r)) return true;
  }
  return generic::read_now(src, &r.value);
}
template <class Own = own>
inline bool finish_read(pending_read& r, float* out) {
  if constexpr (detail::owns_read_later<Own>::value) {
    if (r.state) return Own::finish_read(r, out);
  }
  *out = r.value;
  return true;
}

// Row idx[0] (an f32 index, as argmax_to leaves it) of a `rows`-row table into
// out[0, k) as f32: the next step's input. W is of ggml `type`: f32 (0) or
// bf16 (30) read by strides in elements (out[j] = W[idx * row_stride + j *
// col_stride], so a [rows, k] table and its [k, rows] transpose both serve),
// or a block format whose rows are whole blocks, read as gemv_quant reads them
// (Q4_K 12, Q6_K 14, Q8_0 8; the strides unused). An index past the table
// gives zeros. False for a format the backend has no reader for.
TL_GPU_DETECT_OWN(embed_row)
template <class Own = own>
inline bool embed_row(int type, span W, span idx, span out, int64_t k, int64_t rows,
                      int64_t row_stride, int64_t col_stride) {
  if (!W || !idx || !out || k <= 0 || rows <= 0 || rows > policy::max_f32_index ||
      row_stride < 0 || col_stride < 0 || k > UINT32_MAX || row_stride > UINT32_MAX ||
      col_stride > UINT32_MAX)
    return false;
  if constexpr (detail::owns_embed_row<Own>::value) {
    return detail::ran(Own::embed_row(type, W, idx, out, k, rows, row_stride, col_stride));
  } else {
    return false;
  }
}

}  // namespace gpu
}  // namespace tl
