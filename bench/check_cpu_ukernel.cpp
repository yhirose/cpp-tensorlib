// Microkernel correctness oracle for the M5 CPU backend. Validates each
// available ukernel (scalar always; NEON on ARM; AVX2 on x86) against a
// naive triple loop on random packed panels, across full and edge tiles
// (mr<MR, nr<NR). This is the kernel-level check that the array-API oracle
// test (test_array.cpp) can't reach directly, and — crucially — it is how
// the AVX2 path gets its FIRST real numerical validation on the x86 box,
// where select_ukernel actually picks it (on Apple, Rosetta stops at SSE4.2
// so the AVX2 kernel compiles but cannot run — see docs/roadmap.md M5).
//
// Build + run (native):
//   c++ -std=c++2b -O2 -I include bench/check_cpu_ukernel.cpp -o chk && ./chk
// On the x86 box this exercises scalar + AVX2; on ARM, scalar + NEON.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu.h"

using namespace tl::cpu;
using namespace tl::cpu::detail;

static bool test_uk(ukernel_fn uk, const char* name) {
  std::mt19937 g(123);
  std::uniform_real_distribution<float> d(-1, 1);
  for (int kc : {1, 3, 7, 16, 100}) {
    for (int mr = 1; mr <= MR; mr++) {
      for (int nr = 1; nr <= NR; nr++) {
        // Packed panels are zero-padded past mr/nr, as the packers guarantee.
        std::vector<float> ap(kc * MR, 0), bp(kc * NR, 0);
        for (int p = 0; p < kc; p++) {
          for (int i = 0; i < mr; i++) ap[p * MR + i] = d(g);
          for (int j = 0; j < nr; j++) bp[p * NR + j] = d(g);
        }
        std::vector<float> c(mr * nr, 0), ref(mr * nr, 0);
        for (int i = 0; i < mr; i++)
          for (int j = 0; j < nr; j++) {
            float s = 0;
            for (int p = 0; p < kc; p++) s += ap[p * MR + i] * bp[p * NR + j];
            ref[i * nr + j] = s;
          }
        // ldc = nr: the kernel writes only the valid mr×nr corner.
        uk(kc, ap.data(), bp.data(), c.data(), nr, mr, nr);
        for (int i = 0; i < mr * nr; i++)
          if (std::fabs(c[i] - ref[i]) > 1e-3f * (1 + std::fabs(ref[i]))) {
            std::printf("%s MISMATCH kc=%d mr=%d nr=%d\n", name, kc, mr, nr);
            return false;
          }
      }
    }
  }
  std::printf("%s: ok\n", name);
  return true;
}

int main() {
  bool ok = test_uk(&ukernel_scalar, "scalar");
#ifdef TL_CPU_NEON
  ok &= test_uk(&ukernel_neon, "neon");
#endif
#ifdef TL_CPU_X86
  ok &= test_uk(&ukernel_avx2, "avx2");
#endif
  std::printf(ok ? "ALL OK\n" : "FAILED\n");
  return ok ? 0 : 1;
}
