#pragma once

// The CUDA backend's llm layer (layers: gpu_abi.h): the graph-capture plumbing and the forms
// that read the decode position from the device.
//
// Included by cuda.h, after the device core.

#include "gpu_abi_llm.h"
#include "gpu_abi_nn.h"  // rope_freq

namespace tl {
namespace cuda {

struct own_llm {
  // The graph-capture forms: the position is a device scalar, so a captured
  // step replays against an advancing cache row.
  static bool rope_dpos(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                        int64_t D, gpu::span d_pos, const gpu::rope_freq& f,
                        gpu::span bias);
  static bool kv_append_dpos(gpu::span Kc, gpu::span Vc, gpu::span k_new,
                             gpu::span v_new, gpu::span d_pos, int64_t kv_max,
                             int64_t n_kv_heads, int64_t D, bool kv_f16 = false);
  static bool attn_decode_dpos(gpu::span q, gpu::span K, gpu::span V,
                               gpu::span out, int64_t n_q_heads,
                               int64_t n_kv_heads, gpu::span d_pos,
                               int64_t kv_max, int64_t D, float scale,
                               gpu::span partials, bool kv_f16, int64_t window);
  static bool incr_u32(gpu::span d_pos);
  // A decode step run ahead (gpu_ops_llm.h): its input row gathered from the
  // token on the device.
  static bool embed_row(int type, gpu::span W, gpu::span idx, gpu::span out, int64_t k,
                        int64_t rows, int64_t row_stride, int64_t col_stride, float scale);
  static bool read_later(gpu::span src, gpu::pending_read& r);
  static bool finish_read(gpu::pending_read& r, float* out);
  // A token drawn from the logits (gpu_ops_llm.h), left on the device.
  static bool sample_to(gpu::span a, int64_t n, const gpu::sampling& s, gpu::span counter,
                        gpu::span idx);
};

// ---- CUDA-graph capture (M9 C1-2): record a fixed launch sequence once and
// replay it as a single submit, erasing per-launch host overhead. Only the
// imperative decode step (no host sync / blocking copy mid-stream) is
// capturable; embed staging + argmax happen outside the captured region.

inline bool graph_available() { return context::get().d.graph_ok(); }

// Begin capturing: route every subsequent launch/async-copy onto a private
// capture stream. Drains the default stream first. Returns false if graph
// support is missing. Pair with capture_end().
inline bool capture_begin() {
  auto& c = context::get();
  if (!c.ready || !c.d.graph_ok()) return false;
  if (c.pending) flush();
  if (!c.cap_stream && c.d.StreamCreate(&c.cap_stream, 0) != 0) return false;
  if (c.d.StreamBeginCapture(c.cap_stream, 0 /*GLOBAL*/) != 0) return false;
  c.stream = c.cap_stream;
  c.capturing = true;
  c.capture_touches.clear();
  return true;
}
// End capture and instantiate an executable graph (nullptr on failure).
// Restores the default stream.
inline CUgraphExec capture_end() {
  auto& c = context::get();
  CUgraph g = nullptr;
  CUresult r = c.d.StreamEndCapture(c.cap_stream, &g);
  c.stream = nullptr;
  c.capturing = false;
  if (r != 0 || !g) return nullptr;
  CUgraphExec e = nullptr;
  if (c.d.GraphInstantiate(&e, g, 0) != 0) e = nullptr;
  c.d.GraphDestroy(g);
  if (e) {
    // A touch a buffer has already had leaves its residency where it was, so
    // the first of each (buffer, access) pair, in order, replays the same:
    // one walk of a few dozen entries a replay rather than one per argument.
    auto& kept = c.graph_touches[e];
    for (const auto& t : c.capture_touches)
      if (std::find(kept.begin(), kept.end(), t) == kept.end()) kept.push_back(t);
  }
  c.capture_touches.clear();
  return e;
}
// Replay a captured graph on the default stream (marks work pending; the caller
// flushes or reads results as usual).
inline bool graph_launch(CUgraphExec e) {
  auto& c = context::get();
  if (!c.ready || !e) return false;
  if (auto it = c.graph_touches.find(e); it != c.graph_touches.end()) {
    for (const auto& [native, a] : it->second) c.before_kernel_(native, a);
  }
  if (c.d.GraphLaunch(e, nullptr) != 0) return false;
  c.pending = true;
  return true;
}
inline void graph_destroy(CUgraphExec e) {
  auto& c = context::get();
  c.graph_touches.erase(e);
  if (e && c.d.GraphExecDestroy) c.d.GraphExecDestroy(e);
}

// Set a device u32 scalar (e.g. the capture pos counter) via its mirror. Raw
// 4-byte H2D — the mirror's host bytes are set to `val` then copied to device.
inline void upload_u32(void* native, unsigned val) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.host_mirror_(native);
  if (!m || m->bytes < 4) return;
  std::memcpy(m->host, &val, 4);
  c.d.MemcpyHtoD(m->dev, m->host, 4);
  m->live.uploaded();
}

// ---- CUDA-graph-capture device-pos launchers (A-min). Each mirrors its
// by-value sibling but sources pos/ctx from a device scalar `d_pos` (a 4-byte
// device buffer the caller owns) and launches on c.stream so the op is captured.
// Grids are pos-independent throughout: attn_decode_dpos gets its split grid
// from the cache CAPACITY (max_ctx) and lets the kernel bound the work by pos.

// RoPE reading pos from *d_pos (else identical to own_nn::rope).
inline bool own_llm::rope_dpos(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                           int64_t D, gpu::span d_pos, const gpu::rope_freq& f,
                           gpu::span bias) {
  auto& c = context::get();
  const int64_t rot = f.rot_of(D);
  if (!c.ready || D <= 0 || (D & 1) || rot <= 0 || (rot & 1) || rot > D) return false;
  c.device_read_(x.buf);
  if (bias.buf) c.device_read_(bias.buf);
  if (f.factors) c.device_read_(f.factors.buf);
  c.device_write_(out.buf);
  c.device_read_(d_pos.buf);
  float* px = context::off_(x.buf, x.off);
  float* pbias = bias.buf ? context::off_(bias.buf, bias.off) : nullptr;
  float* pf = f.factors ? context::off_(f.factors.buf, f.factors.off) : nullptr;
  float* po = context::off_(out.buf, out.off);
  float* pp = context::off_(d_pos.buf, d_pos.off);
  unsigned uT = (unsigned)T, uD = (unsigned)D, urot = (unsigned)rot;
  return c.launch_(c.rope_dpos_(), {(unsigned)rows}, {(unsigned)(D / 2)}, 0, px,
                   pbias, po, uT, uD, pp, f.base, urot, f.pos_scale, pf);
}

// One-thread *d_pos += 1 (tail of a captured forward; advances the counter).
inline bool own_llm::incr_u32(gpu::span d_pos) {
  auto& c = context::get();
  if (!c.ready) return false;
  c.device_write_(d_pos.buf);
  float* pp = context::off_(d_pos.buf, d_pos.off);
  return c.launch_(c.incr_u32_(), {1}, {1}, 0, pp);
}

// KV append with write-row = *d_pos (else identical to kv_append(); f32 KV).
inline bool own_llm::kv_append_dpos(gpu::span Kc, gpu::span Vc, gpu::span k_new,
                                gpu::span v_new, gpu::span d_pos,
                                int64_t kv_max, int64_t n_kv_heads, int64_t D, bool kv_f16) {
  auto& c = context::get();
  if (!c.ready || D <= 0 || D > gpu::policy::max_group_threads) return false;
  c.device_read_(k_new.buf);
  c.device_read_(v_new.buf);
  c.device_read_(d_pos.buf);
  c.device_write_(Kc.buf);
  c.device_write_(Vc.buf);
  float* pKc = context::off_(Kc.buf, Kc.off);
  float* pVc = context::off_(Vc.buf, Vc.off);
  float* pk = context::off_(k_new.buf, k_new.off);
  float* pv = context::off_(v_new.buf, v_new.off);
  float* pp = context::off_(d_pos.buf, d_pos.off);
  unsigned kv_stride = (unsigned)(kv_max * D);
  return c.launch_(c.kv_append_dpos_(kv_f16), {(unsigned)n_kv_heads}, {(unsigned)D},
                   0, pKc, pVc, pk, pv, pp, kv_stride);
}

// The splits the graph-capture decode attention launches over a cache of
// `keys` keys (its capacity, or a window's length): sizes the partials
// (attn_dpos_partials_bytes) and the grid (attn_decode_dpos) alike, so the
// partials always hold what is launched. 0 for a head count GQA cannot split.
inline unsigned attn_dpos_splits(int64_t n_q_heads, int64_t n_kv_heads, int64_t D, int64_t keys) {
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads) return 0;
  return attn_split_count(attn_dec_rows(n_q_heads, n_kv_heads, D), keys);
}

// Decode attention with ctx = *d_pos + 1, split-KV on a capacity-static grid:
// gridDim.y = S_max = attn_dpos_splits at max_ctx, a constant for the cache's
// lifetime — so the launch is capturable — while each block derives its live
// split bounds from *d_pos (attn_dpos_chunk, under attn_split_rule passed as
// arguments). Splits past the live count are empty (attn_partials), which the
// combine's sums stop before, so the output is bit-identical to the host
// attn_decode at every pos, and the captured path stays split-KV-flat at long
// ctx.
// `partials` is a caller-OWNED device buffer of attn_dpos_partials_bytes —
// like d_pos, it is graph-lifetime state (a captured graph bakes its address
// in), so it must not be a shared growable scratch; kv_cache owns one per
// cache. The cache f32, or f16 with kv_f16.
inline bool own_llm::attn_decode_dpos(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span out, int64_t n_q_heads,
                                  int64_t n_kv_heads, gpu::span d_pos,
                                  int64_t kv_max, int64_t D, float scale,
                                  gpu::span partials, bool kv_f16, int64_t window) {
  auto& c = context::get();
  if (!c.ready || !attn_decode_head_ok(D) || !partials.buf) return false;
  if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
  c.device_read_(q.buf);
  c.device_read_(K.buf);
  c.device_read_(V.buf);
  c.device_read_(d_pos.buf);
  c.device_write_(out.buf);
  c.device_write_(partials.buf);
  float* pq = context::off_(q.buf, q.off);
  float* pk = context::off_(K.buf, K.off);
  float* pv = context::off_(V.buf, V.off);
  float* po = context::off_(out.buf, out.off);
  float* pp = context::off_(d_pos.buf, d_pos.off);
  unsigned uh = (unsigned)n_q_heads;
  unsigned kv_stride = (unsigned)(kv_max * D);
  unsigned group = (unsigned)(n_q_heads / n_kv_heads);
  // A window (< kv_max) bounds the live keys, and so the splits, at its length; the
  // partials, sized from kv_max, hold at least as many (split_parts is monotone).
  const unsigned S = attn_dpos_splits(n_q_heads, n_kv_heads, D, window > 0 ? window : kv_max);
  const gpu::policy::split_rule& r = attn_split_rule;
  attn_partials p(context::off_(partials.buf, partials.off), (size_t)uh * S);
  if (!c.launch_(c.attn_(context::attn_k::split_dpos, D, kv_f16),
                 {attn_dec_rows(n_q_heads, n_kv_heads, D), S}, {attn_dec_threads(D)}, 0, pq, pk, pv,
                 p.pm, p.pl, p.pacc, pp, kv_stride, group, scale, (unsigned)window,
                 (unsigned)r.target, (unsigned)r.min_k, (unsigned)r.min_part, (unsigned)r.granule)) {
    return false;
  }
  return p.combine(c, po, uh, static_cast<unsigned>(D), S);
}

// Row idx[0] of a table into f32 (gpu_ops_llm.h's contract): a thread an
// element by strides for f32 / bf16, eight weights a thread for the block
// formats (cuda_nn.h's quant_fmts_), whose rows K keeps whole.
inline bool own_llm::embed_row(int type, gpu::span W, gpu::span idx, gpu::span out, int64_t k,
                               int64_t rows, int64_t row_stride, int64_t col_stride, float scale) {
  auto& c = context::get();
  // An f32 (type 0) or bf16 (30) table by strides, else a block format's rows.
  const bool plain = type == 0 || type == 30;
  const int fi = plain ? -1 : quant_fmt_(type, k);
  if (!c.ready || (!plain && fi < 0)) return false;
  CUfunction f = plain ? c.cached_(c.embed_row_fn[type ? 1 : 0],
                                   type ? "tl_embed_row_bf16" : "tl_embed_row_f32")
                       : c.cached_(c.embed_quant_fn[fi], quant_fmts_[fi].embed);
  void* pw = context::off_(W.buf, W.off);
  float* pi = context::off_(idx.buf, idx.off);
  float* po = context::off_(out.buf, out.off);
  const unsigned uk = (unsigned)k, urows = (unsigned)rows, threads = 256;
  // Refused before any residency change if the table is misaligned for its reads.
  if (!f || !c.fits_(f, pw, pi, po)) return false;
  c.device_read_(W.buf);
  c.device_read_(idx.buf);
  c.device_write_(out.buf);
  if (plain) {
    const unsigned rs = (unsigned)row_stride, cs = (unsigned)col_stride;
    return c.launch_(f, {(uk + threads - 1) / threads}, {threads}, 0, pw, pi, po, uk, urows, rs, cs,
                     scale);
  }
  const unsigned per = uk / 8;
  return c.launch_(f, {(per + threads - 1) / threads}, {threads}, 0, pw, pi, po, uk, urows, scale);
}

// src's f32 copied into a pinned slot on the stream, behind every launch
// queued so far, and an event after it; finish_read waits on the event. The
// launches are already submitted, so nothing waits here. Not for use under a
// capture, like every read-back (the decoder reads behind the graph's replay).
inline bool own_llm::read_later(gpu::span src, gpu::pending_read& r) {
  auto& c = context::get();
  if (!c.ready || !c.d.read_ok()) return false;
  context::read_slot* slot = c.read_slot_();
  if (!slot) return false;
  c.device_read_(src.buf);  // a host-live src goes up first, on the same stream
  const auto at = reinterpret_cast<CUdeviceptr>(context::off_(src.buf, src.off));
  if (c.d.MemcpyDtoHAsync(slot->host, at, sizeof(float), c.stream) != 0 ||
      c.d.EventRecord(slot->done, c.stream) != 0) {
    c.spare_reads.push_back(slot);
    return false;
  }
  profile::detail::transfer("d2h", sizeof(float), 0.0);
  r.state = std::shared_ptr<context::read_slot>(
      slot, [](context::read_slot* s) { context::get().spare_reads.push_back(s); });
  return true;
}
inline bool own_llm::finish_read(gpu::pending_read& r, float* out) {
  auto& c = context::get();
  auto* slot = static_cast<context::read_slot*>(r.state.get());
  {
    profile::detail::blocked waiting;
    if (c.d.EventSynchronize(slot->done) != 0) return false;
  }
  *out = *slot->host;
  r.state.reset();
  return true;
}

// metal_llm.h's launches of the same kernel, but the first level's blocks
// take per_block keys too, and the two halves the keys go back and forth
// between are sized for the longest vector and the largest k, once: every
// grid and address is then n's and k's alone, so a graph captures the draw.
inline bool own_llm::sample_to(gpu::span a, int64_t n, const gpu::sampling& s,
                               gpu::span counter, gpu::span idx) {
  constexpr int64_t per_block = 256 * 16;
  constexpr int64_t half = 8 * gpu::policy::max_top_k *
                           ((gpu::policy::max_f32_index + per_block - 1) / per_block);
  auto& c = context::get();
  if (!c.ready) return false;
  const gpu::span scratch = {reinterpret_cast<void*>(c.sample_keys_(2 * half)), 0};
  if (!scratch) return false;
  const bool gumbel = s.top_k == 0;
  const int64_t k = gumbel ? 1 : s.top_k;
  const float inv_t = 1.0f / s.temperature;
  gpu::span keys = a, ids = a;
  int64_t m = n;
  for (uint32_t level = 0;; level++) {
    const bool draw = m <= per_block;
    const int64_t blocks = draw ? 1 : (m + per_block - 1) / per_block;
    const gpu::span out = draw ? idx : scratch.at(level % 2 * half),
                    out_ids = draw ? idx : out.at(4 * blocks * k);
    const gpu::arg args[] = {gpu::in(keys), gpu::in(ids), gpu::inout(counter), gpu::out(out),
                             gpu::out(out_ids)};
    const gpu::sample_params p{static_cast<uint32_t>(m), static_cast<uint32_t>(per_block),
                               static_cast<uint32_t>(k), level > 0, gumbel, draw, s.seed,
                               inv_t, s.top_p};
    if (!dispatch(kop::sample_, args, 5, &p, sizeof(p),
                  {static_cast<uint32_t>(blocks), 1, 1, 256, 1, 1, 0}))
      return false;
    if (draw) return true;
    keys = out;
    ids = out_ids;
    m = blocks * k;
  }
}

// The KV cache itself is tl::kv_cache (kv_cache.h), written once over the
// gpu:: facade; its graph-capture forms call kv_append_dpos / attn_decode_dpos
// above and size their partials here.
inline int64_t attn_dpos_partials_bytes(int64_t n_q_heads, int64_t n_kv_heads, int64_t max_ctx,
                                        int64_t D) {
  const unsigned S = attn_dpos_splits(n_q_heads, n_kv_heads, D, max_ctx);
  if (S == 0) return 0;
  return static_cast<int64_t>(
      attn_partials::bytes(static_cast<size_t>(n_q_heads) * S, D));
}

}  // namespace cuda
}  // namespace tl
