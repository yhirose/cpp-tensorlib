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
