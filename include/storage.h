#pragma once

#include <cstdint>
#include <memory>

namespace tl {

// Flat F32 buffer shared between arrays (views share one storage). This is
// the seam where device residency (Metal buffer, CUDA allocation) and pooled
// recycling attach in later milestones; the CPU pointer stays the canonical
// handle for the reference backend.
struct storage {
  std::shared_ptr<float[]> buf;
  int64_t size = 0;  // elements

  static storage make(int64_t n) {
    storage s;
    s.buf = std::shared_ptr<float[]>(new float[static_cast<size_t>(n)]);
    s.size = n;
    return s;
  }

  float* data() const { return buf.get(); }
};

}  // namespace tl
