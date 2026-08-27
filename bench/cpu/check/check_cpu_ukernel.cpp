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
//   c++ -std=c++2b -O2 -I include bench/cpu/check/check_cpu_ukernel.cpp -o chk && ./chk
// On the x86 box this exercises scalar + AVX2; on ARM, scalar + NEON.
// (Also wired into CMake as the `cpu_ukernel` ctest — see CMakeLists.txt.)

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu.h"

using namespace tl::cpu;
using namespace tl::cpu::detail;

// Each kernel packs to its own tile (mrt×nrt) — 8×8 for scalar/NEON, 6×16 for
// AVX2 — so the harness takes the tile and lays panels out with that stride,
// exactly as cpu::sgemm's driver does per descriptor.
static bool test_uk(ukernel_fn uk, int mrt, int nrt, const char* name) {
  std::mt19937 g(123);
  std::uniform_real_distribution<float> d(-1, 1);
  for (int kc : {1, 3, 7, 16, 100}) {
    for (int mr = 1; mr <= mrt; mr++) {
      for (int nr = 1; nr <= nrt; nr++) {
        // Packed panels are zero-padded past mr/nr, as the packers guarantee.
        std::vector<float> ap(kc * mrt, 0), bp(kc * nrt, 0);
        for (int p = 0; p < kc; p++) {
          for (int i = 0; i < mr; i++) ap[p * mrt + i] = d(g);
          for (int j = 0; j < nr; j++) bp[p * nrt + j] = d(g);
        }
        std::vector<float> c(mr * nr, 0), ref(mr * nr, 0);
        for (int i = 0; i < mr; i++)
          for (int j = 0; j < nr; j++) {
            float s = 0;
            for (int p = 0; p < kc; p++) s += ap[p * mrt + i] * bp[p * nrt + j];
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
  // The scalar kernel is the oracle: always compiled, always callable, and the
  // one every vector kernel is checked against.
  bool ok = test_uk(&ukernel_scalar, MR, NR, "scalar");
  // Then whatever dispatch ACTUALLY picks here, on its own descriptor. Asking
  // select_ukernel rather than re-deriving the choice from ISA macros keeps the
  // harness honest in two ways: it exercises the CPUID gate itself (calling an
  // AVX2 kernel unguarded would fault on an older CPU rather than test
  // anything), and a new kernel or a new register tile needs no edit here.
  const ukernel_desc sel = select_ukernel();
  ok &= test_uk(sel.fn, sel.mr, sel.nr, "selected");
  std::printf(ok ? "ALL OK\n" : "FAILED\n");
  return ok ? 0 : 1;
}
