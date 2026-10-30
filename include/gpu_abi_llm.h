#pragma once

// The llm layer's kernel params (the kop ids of TL_KOPS_LLM): the same ABI as
// gpu_abi.h's, a run of 4-byte fields in the order the kernel takes its scalars.

#include <cstdint>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

struct kv_append_params {
  uint32_t pos, kv_stride;  // kv_stride = kv_max * D, elements a head
};
struct kv_fill_params {
  uint32_t T, kv_stride, pos0;
};

}  // namespace gpu
}  // namespace tl
