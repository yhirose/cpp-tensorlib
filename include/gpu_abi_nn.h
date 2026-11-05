#pragma once

// The nn layer's kernel params (the kop ids of TL_KOPS_NN): the same ABI as
// gpu_abi.h's, a run of 4-byte fields in the order the kernel takes its scalars.

#include <cstdint>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

// The form a block-quantized GEMV reads its activation in (gpu_ops_nn.h's prepare_act
// and gemv_quant_act): f32, the activation as it is; q8, int8 blocks of 32 with an f32
// scale and sum each (llama.cpp's q8_1), read with dp4a; h16, f16 (each 256 values
// scaled by a power of two where they would pass f16's range), read by the tensor
// cores with f32 sums: the f32 result on the activation rounded to f16. Every form
// but f32 is prepared from the f32 activation once per GEMV call; prepared_act_forms
// lists them.
enum class act_form { f32, q8, h16 };
inline constexpr act_form prepared_act_forms[] = {act_form::q8, act_form::h16};

// RoPE's frequencies (gpu_ops_nn.h's rope and the forms that fuse it): the pair
// (j, j + rot/2) of a row at position p turns by
//   p * pos_scale * base^(-2j/rot) / factors[j],
// rot the dims that turn (0: all of the row; Qwen3.5 turns a part), pos_scale a
// linear scaling of the position (Gemma 3's global layers), factors rot/2 divisors
// on the device (Llama 3's rope_freqs, llama.cpp's freq_factors; none: all 1).
struct rope_freq {
  float base = 10000.0f;
  int64_t rot = 0;
  float pos_scale = 1.0f;
  span factors{};
  int64_t rot_of(int64_t D) const { return rot ? rot : D; }  // the dims that turn in a row of D
};

struct layer_norm_params {
  uint32_t rows, cols;
  float eps, scale, offset;
};
struct xent_bwd_params {
  uint32_t cols, n;  // n = rows * cols
};
struct adam_params {
  float b1, b2, eps, lr_over_bc1, inv_bc2;
  uint32_t n;
};
struct rmsnorm_params {  // per row of n: x * rsqrt(mean(x^2) + eps) * w
  uint32_t n;
  float eps;
};
struct glu_params {  // swiglu_ / geglu_: out[rows, ff] = act(gate) * up
  uint32_t ff;
};
struct gemv_row_params {  // y[n] = a[k] . W[n, k], one group an output row
  uint32_t n, k;
};
struct gemv_q4_params {
  uint32_t n, k, group;
};
struct merge_heads_params {
  uint32_t T, H, D;
};
struct ssm_conv_params {  // C channels, a K-tap causal window, T tokens
  uint32_t C, K, T, ld;   // ld: elements between the input's token rows
};
struct gated_delta_params {  // H heads, a [dk, dv] state each, T tokens
  uint32_t H, dk, dv, T, ab_ld;  // ab_ld: elements between alpha|beta rows
  float eps;                     // of the q / k L2 norms
  uint32_t Hk;                   // q / k heads: head h reads key head h % Hk
};

}  // namespace gpu
}  // namespace tl
