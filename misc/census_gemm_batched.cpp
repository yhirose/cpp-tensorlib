// Batched-GEMM launch census: the same [batch,m,k]·[batch,k,n] product as one
// launch (the batch folded into the kernel grid) against `batch` per-slice
// launches (tl::bdot_one_launch_ off), on the GPU, in µs. Attention's probs·v
// ([H,T,T]·[H,T,D]) and scores ([H,T,D]·[H,D,T], NT) at H=8 D=64, T=512 and
// 1024, plus a rank-4 [B,H,…] pair. Medians, warm-up first.
#include "census_common.h"

#include <cstdio>

using tl::array;

int main() {
  if (!tl::gpu_available()) {
    std::printf("no GPU\n");
    return 0;
  }
  tl::use_gpu();
  struct S { const char* name; tl::shape_t a, b; bool nt; };
  const S shapes[] = {
      {"probs.v  H8 T512 D64", {8, 512, 512}, {8, 512, 64}, false},
      {"probs.v  H8 T1024 D64", {8, 1024, 1024}, {8, 1024, 64}, false},
      {"q.kT     H8 T512 D64", {8, 512, 64}, {8, 512, 64}, true},
      {"q.kT     H8 T1024 D64", {8, 1024, 64}, {8, 1024, 64}, true},
      {"rank4    B2 H8 T256 D64", {2, 8, 256, 256}, {2, 8, 256, 64}, false},
  };
  std::printf("%-26s %10s %10s\n", "shape", "one-launch", "per-slice");
  for (const auto& s : shapes) {
    auto A = census::rnd(s.a, 1), B = census::rnd(s.b, 2);
    A.eval();
    B.eval();
    auto product = [&] {
      (s.nt ? A.dot(B.transpose({0, 2, 1})) : A.dot(B)).eval();
    };
    tl::bdot_one_launch_ = true;
    double fused = census::med_us(21, product);
    tl::bdot_one_launch_ = false;
    double loop = census::med_us(21, product);
    tl::bdot_one_launch_ = true;
    std::printf("%-26s %10.1f %10.1f\n", s.name, fused, loop);
  }
  return 0;
}
