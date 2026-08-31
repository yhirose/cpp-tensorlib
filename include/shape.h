#pragma once

// Shape/stride math shared across headers that can't see array.h's own
// detail namespace: cuda.h and webgpu.h are pulled in — via storage.h's
// gpu.h — before array.h defines it, so a helper they need has to live
// somewhere below both instead. This file has no dependency on array.h
// (or on shape_t, which array.h itself defines), so it can sit at the
// bottom of the include graph without constraining anyone above it.

#include <cstdint>

namespace tl {

// Row-major (contiguous) strides for a shape given as a raw pointer + rank —
// the allocation-free core that both array.h's detail::contiguous_strides
// (which wraps this for a shape_t, returning a std::vector) and cuda.h's
// pad/fold GPU dispatch (which writes into a fixed-size stack array, once per
// call on a hot path) share.
inline void contiguous_strides_into(const int64_t* shape, int rank,
                                    int64_t* strides) {
  int64_t acc = 1;
  for (int d = rank - 1; d >= 0; --d) {
    strides[d] = acc;
    acc *= shape[d];
  }
}

}  // namespace tl
