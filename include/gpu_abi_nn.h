#pragma once

// The nn layer's kernel params (the kop ids of TL_KOPS_NN): the same ABI as
// gpu_abi.h's, a run of 4-byte fields in the order the kernel takes its scalars.

#include <cstdint>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

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
struct swiglu_params {
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
