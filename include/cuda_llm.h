#pragma once

// The CUDA backend's llm layer (layers: gpu_abi.h): the graph-capture plumbing and the forms
// that read the decode position from the device.
//
// Included by cuda.h, after the device core.

#include "gpu_abi_llm.h"

namespace tl {
namespace cuda {

struct own_llm {
  // The graph-capture forms: the position is a device scalar, so a captured
  // step replays against an advancing cache row.
  static bool rope_dpos(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                        int64_t D, gpu::span d_pos, float base,
                        gpu::span bias = {}, int64_t rot = 0);
  static bool kv_append_dpos(gpu::span Kc, gpu::span Vc, gpu::span k_new,
                             gpu::span v_new, gpu::span d_pos, int64_t kv_max,
                             int64_t n_kv_heads, int64_t D, bool kv_f16 = false);
  static bool attn_decode_dpos(gpu::span q, gpu::span K, gpu::span V,
                               gpu::span out, int64_t n_q_heads,
                               int64_t n_kv_heads, gpu::span d_pos,
                               int64_t kv_max, int64_t D, float scale,
                               gpu::span partials, bool kv_f16 = false);
  static bool incr_u32(gpu::span d_pos);
  // A decode step run ahead (gpu_ops_llm.h): its input row gathered from the
  // token on the device.
  static bool embed_row(int type, gpu::span W, gpu::span idx, gpu::span out, int64_t k,
                        int64_t rows, int64_t row_stride, int64_t col_stride);
  static bool read_later(gpu::span src, gpu::pending_read& r);
  static bool finish_read(gpu::pending_read& r, float* out);
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
  context::mirror* m = c.mirror_(native);
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

// RoPE reading pos from *d_pos (else identical to rope_partial()).
inline bool own_llm::rope_dpos(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                           int64_t D, gpu::span d_pos, float base,
                           gpu::span bias, int64_t rot) {
  auto& c = context::get();
  if (rot == 0) rot = D;
  if (!c.ready || D <= 0 || (D & 1) || rot <= 0 || (rot & 1) || rot > D) return false;
  c.device_read_(x.buf);
  if (bias.buf) c.device_read_(bias.buf);
  c.device_write_(out.buf);
  c.device_read_(d_pos.buf);
  float* px = context::off_(x.buf, x.off);
  float* pbias = bias.buf ? context::off_(bias.buf, bias.off) : nullptr;
  float* po = context::off_(out.buf, out.off);
  float* pp = context::off_(d_pos.buf, d_pos.off);
  unsigned uT = (unsigned)T, uD = (unsigned)D, urot = (unsigned)rot;
  return c.launch_(c.rope_dpos_(), {(unsigned)rows}, {(unsigned)(D / 2)}, 0, px,
                   pbias, po, uT, uD, pp, base, urot);
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

// Decode attention with ctx = *d_pos + 1, split-KV on a capacity-static grid:
// gridDim.y = S_max = attn_split_count(H, max_ctx), a constant for the cache's
// lifetime — so the launch is capturable — while each block derives its live
// split bounds from *d_pos (attn_dpos_chunk, the device twin of
// attn_split_count + attn_split_chunk). Splits past the live count write
// neutral partials that combine as exact zeros, so the output is bit-identical
// to the host attn_decode at every pos, and the captured path stays
// split-KV-flat at long ctx (the S=1 pin it replaces was linear in ctx).
// `partials` is a caller-OWNED device buffer of attn_partials::bytes(H*S_max, D)
// — like d_pos, it is graph-lifetime state (a captured graph bakes its address
// in), so it must not be a shared growable scratch; kv_cache owns one per
// cache. The cache f32, or f16 with kv_f16.
inline bool own_llm::attn_decode_dpos(gpu::span q, gpu::span K, gpu::span V,
                                  gpu::span out, int64_t n_q_heads,
                                  int64_t n_kv_heads, gpu::span d_pos,
                                  int64_t kv_max, int64_t D, float scale,
                                  gpu::span partials, bool kv_f16) {
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
  unsigned uh = (unsigned)n_q_heads, uD = (unsigned)D;
  unsigned kv_stride = (unsigned)(kv_max * D);
  unsigned group = (unsigned)(n_q_heads / n_kv_heads);
  unsigned S = attn_split_count(uh, kv_max);
  attn_partials p(context::off_(partials.buf, partials.off), (size_t)uh * S);
  if (!c.launch_(c.attn_(context::attn_k::split_dpos, D, kv_f16), {uh, S}, {uD}, 0, pq, pk, pv, p.pm,
                 p.pl, p.pacc, pp, kv_stride, group, scale)) {
    return false;
  }
  return p.combine(c, po, uh, uD, S);
}

// Row idx[0] of a table into f32 (gpu_ops_llm.h's contract): a thread an
// element by strides for f32 / bf16, eight weights a thread for the block
// formats, whose rows gemv_quant's K bounds (K % 256, Q8_0's K % 32) keep whole.
inline bool own_llm::embed_row(int type, gpu::span W, gpu::span idx, gpu::span out, int64_t k,
                               int64_t rows, int64_t row_stride, int64_t col_stride) {
  auto& c = context::get();
  // 0 f32, 1 bf16, then the block formats on gemv_quant's terms (quant_wf_).
  const int wf = quant_wf_(type, k), e = type == 0 ? 0 : type == 30 ? 1 : wf > 0 ? wf + 1 : -1;
  if (!c.ready || e < 0) return false;
  static const char* const names[5] = {"tl_embed_row_f32", "tl_embed_row_bf16", "tl_embed_row_q4k",
                                       "tl_embed_row_q6k", "tl_embed_row_q80"};
  CUfunction f = c.cached_(c.embed_row_fn[e], names[e]);
  void* pw = context::off_(W.buf, W.off);
  float* pi = context::off_(idx.buf, idx.off);
  float* po = context::off_(out.buf, out.off);
  const unsigned uk = (unsigned)k, urows = (unsigned)rows, threads = 256;
  // Refused before any residency change if the table is misaligned for its reads.
  if (!f || !c.fits_(f, pw, pi, po)) return false;
  c.device_read_(W.buf);
  c.device_read_(idx.buf);
  c.device_write_(out.buf);
  if (e < 2) {
    const unsigned rs = (unsigned)row_stride, cs = (unsigned)col_stride;
    return c.launch_(f, {(uk + threads - 1) / threads}, {threads}, 0, pw, pi, po, uk, urows, rs, cs);
  }
  const unsigned per = uk / 8;
  return c.launch_(f, {(per + threads - 1) / threads}, {threads}, 0, pw, pi, po, uk, urows);
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

// The KV cache itself is tl::kv_cache (kv_cache.h), written once over the
// gpu:: facade; its graph-capture forms call kv_append_dpos / attn_decode_dpos
// above and size their partials here.
inline int64_t attn_dpos_partials_bytes(int64_t n_q_heads, int64_t max_ctx,
                                        int64_t D) {
  const unsigned S = attn_split_count(static_cast<unsigned>(n_q_heads), max_ctx);
  return static_cast<int64_t>(
      attn_partials::bytes(static_cast<size_t>(n_q_heads) * S, D));
}

}  // namespace cuda
}  // namespace tl
