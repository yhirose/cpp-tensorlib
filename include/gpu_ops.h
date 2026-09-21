#pragma once

// The GPU ops, written once. An op is a kernel id, the views it touches and
// how, a params struct, and a launch shape (gpu_abi.h); `launch` hands those to
// the selected backend's `dispatch`. Nothing here names a backend, so an op
// added here exists on all of them, and a backend without the kernel declines
// (false) and the caller falls back, exactly as before.
//
// Included by gpu.h, after the backend is selected.

#include <array>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include "gpu_abi.h"

namespace tl {
namespace gpu {

namespace detail {
// Launches per kernel id since the last census_reset(): how a test tells a
// kernel that ran on the device from an op that quietly fell back to the CPU.
inline std::array<uint64_t, kKopCount> census_counts{};
}  // namespace detail

inline uint64_t census(kop k) {
  return detail::census_counts[static_cast<size_t>(k)];
}
inline void census_reset() { detail::census_counts.fill(0); }

// Every shared op launches through here.
template <class P>
inline bool launch(kop k, std::initializer_list<arg> args, const P& params,
                   const grid& g) {
  static_assert(std::is_trivially_copyable<P>::value && sizeof(P) % 4 == 0 &&
                    alignof(P) == 4,
                "kernel params are a run of 4-byte fields (gpu_abi.h)");
  if (!dispatch(k, args.begin(), args.size(), &params, sizeof(P), g)) {
    return false;
  }
  detail::census_counts[static_cast<size_t>(k)]++;
  return true;
}

// out[i] = (a[i] OP b[i]) * scale + offset over n contiguous elements.
inline bool binary(kop op, span a, span b, span o, int64_t n, float scale,
                   float offset) {
  return launch(op, {in(a), in(b), out(o)},
                ew_params{static_cast<uint32_t>(n), scale, offset},
                policy::flat(n));
}

// out[i] = OP(a[i]) * scale + offset over n contiguous elements.
inline bool unary(kop op, span a, span o, int64_t n, float scale, float offset) {
  return launch(op, {in(a), out(o)},
                ew_params{static_cast<uint32_t>(n), scale, offset},
                policy::flat(n));
}

// tanh / sin / cos: unary's shape under their own vocabulary (gpu_abi.h).
inline bool unary_ext(unary_ext_op op, span a, span o, int64_t n, float scale,
                      float offset) {
  kop k = kop::tanh_;
  switch (op) {
    case unary_ext_op::tanh_: k = kop::tanh_; break;
    case unary_ext_op::sin_: k = kop::sin_; break;
    case unary_ext_op::cos_: k = kop::cos_; break;
  }
  return unary(k, a, o, n, scale, offset);
}

}  // namespace gpu
}  // namespace tl
