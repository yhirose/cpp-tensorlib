// The llm layer's MSL kernels (layers: gpu_abi.h): the KV cache's writes.
// #embed'd by metal.h after metal_kernels_nn.metal.

// One step's k/v ([n_kv_heads, D]) into cache row `pos`; grid (n_kv_heads),
// D threads. KT is the cache element type.
struct kv_append_params {
  uint pos, kv_stride;
};
struct kv_fill_params {
  uint T, kv_stride, pos0;
};

template <typename KT>
kernel void kv_append_(device KT* Kc             [[buffer(0)]],
                       device KT* Vc             [[buffer(1)]],
                       device const float* k_new [[buffer(2)]],
                       device const float* v_new [[buffer(3)]],
                       constant kv_append_params& p [[buffer(4)]],
                       uint h [[threadgroup_position_in_grid]],
                       uint d [[thread_index_in_threadgroup]],
                       uint D [[threads_per_threadgroup]]) {
  const uint dst = h * p.kv_stride + p.pos * D + d;
  const uint src = h * D + d;
  narrow_(Kc + dst, k_new[src]);
  narrow_(Vc + dst, v_new[src]);
}
template [[host_name("kv_append_")]] kernel void
kv_append_<float>(device float*, device float*, device const float*,
                  device const float*, constant kv_append_params&, uint, uint, uint);
template [[host_name("kv_append_f16_")]] kernel void
kv_append_<half>(device half*, device half*, device const float*,
                 device const float*, constant kv_append_params&, uint, uint, uint);

// A prefill's k/v ([n_kv_heads, T, D]) into cache rows [pos, pos + T); grid
// (n_kv_heads, T), D threads.
template <typename KT>
kernel void kv_fill_(device KT* Kc             [[buffer(0)]],
                     device KT* Vc             [[buffer(1)]],
                     device const float* K     [[buffer(2)]],
                     device const float* V     [[buffer(3)]],
                     constant kv_fill_params& p [[buffer(4)]],
                     uint2 g  [[threadgroup_position_in_grid]],
                     uint d   [[thread_index_in_threadgroup]],
                     uint2 nt [[threads_per_threadgroup]]) {
  const uint h = g.x, t = g.y, D = nt.x;
  const uint dst = h * p.kv_stride + (p.pos0 + t) * D + d;
  const uint src = (h * p.T + t) * D + d;
  narrow_(Kc + dst, K[src]);
  narrow_(Vc + dst, V[src]);
}
template [[host_name("kv_fill_")]] kernel void
kv_fill_<float>(device float*, device float*, device const float*,
                device const float*, constant kv_fill_params&, uint2, uint, uint2);
template [[host_name("kv_fill_f16_")]] kernel void
kv_fill_<half>(device half*, device half*, device const float*,
               device const float*, constant kv_fill_params&, uint2, uint, uint2);

// gpu_abi_llm.h's sample_ref on the device, in launches of this kernel. Keys
// rank larger value first, then smaller index, so a NaN or a -inf is never
// taken and an empty slot is (-inf, -1). A key is the logit, or under
// Gumbel-max the logit over the temperature plus element i's noise. A group
// reads at most 256 x 16 keys, 16 a lane held in registers, and keeps the k
// first: each simdgroup its lanes' k first, a round a key with no barrier
// (the lane that gave a key finds its next, below that one), then the first
// simdgroup the k first of those eight lists. The first launch's groups read
// chunks of the logits; each later one reads the groups' keys before it
// (`mapped`: each key's index from idx), until one group does, whose first
// thread then draws as sample_ref does (`draw`), summing the weights in the
// same order, writes the token at out_i[0] and advances the counter.
struct sample_params {
  uint n, chunk, k, mapped, gumbel, draw, seed;
  float inv_t, top_p;
};

// Philox4x32-10 of (c0, c1, 0, 0) under (k0, 0): the first word.
inline uint philox_(uint c0, uint c1, uint k0) {
  uint c2 = 0, c3 = 0, k1 = 0;
  for (int r = 0; r < 10; r++) {
    const uint n0 = mulhi(0xCD9E8D57u, c2) ^ c1 ^ k0, n2 = mulhi(0xD2511F53u, c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * c2;
    c3 = 0xD2511F53u * c0;
    c0 = n0;
    c2 = n2;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

// gpu_abi_llm.h's sample_uniform and sample_pick_element.
inline float sample_uniform_(uint bits, bool open) {
  return open ? (float(bits >> 9) + 0.5f) * 0x1p-23f : float(bits >> 8) * 0x1p-24f;
}
constant constexpr uint kSamplePickElement = 0xFFFFFFFFu;

// Whether key (v, i) ranks before (w, j).
inline bool ranks_before_(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

// The first of the simdgroup's keys, in every lane: the largest value, then
// the smallest index holding it. No lane holds a NaN, and an empty one (-inf,
// -1) is first only when all are.
inline void simd_first_(thread float& v, thread int& i) {
  const float m = simd_max(v);
  i = simd_min(v == m ? i : INT_MAX);
  v = m;
}

constant constexpr uint kSampleKeys = 16;  // a lane's keys

// The first of a lane's keys after (last, lasti).
inline void lane_next_(thread const float* x, thread const int* xi, float last, int lasti,
                       thread float& v, thread int& vi) {
  v = -INFINITY;
  vi = -1;
  _Pragma("clang loop unroll(full)")
  for (uint e = 0; e < kSampleKeys; e++) {
    if (ranks_before_(last, lasti, x[e], xi[e]) && ranks_before_(x[e], xi[e], v, vi)) {
      v = x[e];
      vi = xi[e];
    }
  }
}

kernel void sample_(device const float* in      [[buffer(0)]],
                    device const float* idx     [[buffer(1)]],
                    device uint* counter        [[buffer(2)]],
                    device float* out_v         [[buffer(3)]],
                    device float* out_i         [[buffer(4)]],
                    constant sample_params& p   [[buffer(5)]],
                    uint g    [[threadgroup_position_in_grid]],
                    uint tid  [[thread_index_in_threadgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lv[8][64];  // each simdgroup's k first (policy::max_top_k)
  threadgroup int li[8][64];
  threadgroup float tv[64];     // the drawing group's k first
  threadgroup int ti[64];
  const uint ctr = counter[0];
  const uint begin = g * p.chunk, end = min(p.n, begin + p.chunk);
  float x[kSampleKeys];
  int xi[kSampleKeys];
  _Pragma("clang loop unroll(full)")
  for (uint e = 0; e < kSampleKeys; e++) {
    const uint i = begin + tid + e * 256;
    x[e] = -INFINITY;
    xi[e] = -1;
    if (i >= end) continue;
    xi[e] = p.mapped ? int(idx[i]) : int(i);
    x[e] = in[i];
    if (p.gumbel && !p.mapped) {
      const uint bits = philox_(ctr, i, p.seed);
      x[e] = x[e] * p.inv_t - precise::log(-precise::log(sample_uniform_(bits, true)));
    }
  }
  float hv;
  int hi;
  lane_next_(x, xi, INFINITY, -1, hv, hi);
  for (uint r = 0; r < p.k; r++) {
    float v = hv;
    int vi = hi;
    simd_first_(v, vi);
    if (lane == 0) lv[sgid][r] = v, li[sgid][r] = vi;
    if (hi == vi && vi >= 0) lane_next_(x, xi, hv, hi, hv, hi);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid != 0) return;
  // The eight lists, a lane each.
  uint at = 0;
  hv = lane < 8 ? lv[lane][0] : -INFINITY;
  hi = lane < 8 ? li[lane][0] : -1;
  for (uint r = 0; r < p.k; r++) {
    float v = hv;
    int vi = hi;
    simd_first_(v, vi);
    if (lane == 0) {
      if (p.draw) {
        tv[r] = v;
        ti[r] = vi;
      } else {
        out_v[g * p.k + r] = v;
        out_i[g * p.k + r] = float(vi);
      }
    }
    if (hi == vi && vi >= 0) {
      at++;
      hv = at < p.k ? lv[lane][at] : -INFINITY;
      hi = at < p.k ? li[lane][at] : -1;
    }
  }
  if (!p.draw || tid != 0) return;
  uint k = 0;
  while (k < p.k && ti[k] >= 0) k++;
  int tok = 0;
  if (k > 0 && p.gumbel) {
    tok = ti[0];
  } else if (k > 0) {
    // Each key's weight over its own, in order.
    const float top = tv[0];
    float z = 0.0f;
    for (uint j = 0; j < k; j++) z += tv[j] = precise::exp((tv[j] - top) * p.inv_t);
    uint kept = 1;
    float acc = tv[0];
    while (kept < k && acc < p.top_p * z) acc += tv[kept++];
    const float target = sample_uniform_(philox_(ctr, kSamplePickElement, p.seed), false) * acc;
    float run = 0.0f;
    tok = ti[kept - 1];
    for (uint j = 0; j < kept; j++) {
      if (target < (run += tv[j])) {
        tok = ti[j];
        break;
      }
    }
  }
  out_i[0] = float(tok);
  counter[0] = ctr + 1;
}
