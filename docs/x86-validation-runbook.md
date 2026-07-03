# x86 validation runbook (M5 AVX2 — WSL2 / Linux)

Step-by-step for taking the M5 CPU backend from "compile-checked on the Mac"
to "executed, numerically verified, and tuned to the OpenBLAS-90% gate" on a
native x86 box (the RTX 3090 Ubuntu-on-WSL2 machine, or any x86 Linux). The
AVX2 microkernel and the CPUID dispatch were written and disassembly-checked
on Apple but **never executed** there — Rosetta stops at SSE4.2 — so this is
where they run for real. Companion to [roadmap.md](roadmap.md) (M5 scope +
the MSVC/header-only decision) and [performance-notes.md](performance-notes.md)
(measurement discipline, the NEON census this mirrors).

All commands assume the repo root as the working directory. The repo is
portable and travels via git, so on the box: clone, then run these.

## 0. One-time environment

```sh
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopenblas-dev
# Compiler: any g++ >= 11 or clang++ >= 13 (whatever accepts -std=c++2b) —
# the code is C++17-level compiled under the C++23 flag, so it isn't picky;
# Ubuntu 22.04's default g++-11 already works. CMake >= 3.20 is needed for
# CMAKE_CXX_STANDARD 23. libopenblas-dev is the gate baseline; pkg-config
# resolves its include/lib paths portably. Substitute clang++ for g++ in the
# bench commands below if you prefer — both are tested (the Mac builds clang).
```

Sanity-check the CPU actually has the ISA the kernel dispatches to:

```sh
lscpu | grep -iE 'model name|^flags' | grep -oiE 'avx2|fma|avx512[a-z]*|model name.*' 
nproc                      # thread count the pool will use
```

If `avx2` and `fma` are absent, `select_ukernel` will fall to scalar — expected
on very old CPUs, but then there is nothing to validate here. `avx512f` present
is a bonus for the deferred AVX-512 work (not built yet).

## 1. Build + run the test suite (native Linux)

CMake needs no Linux-specific flags — the Accelerate/Metal frameworks are added
only `if(APPLE)`, and metal.h compiles to non-Apple stubs (its `#embed` is
inside the `__APPLE__` block, so no C23 `#embed` is needed on Linux).

```sh
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j"$(nproc)"
cd build-linux && ctest --output-on-failure && cd ..
```

Expect cpu/gpu/auto all pass (gpu/auto degrade to the CPU/ref path with no
device). This confirms the refactored function-pointer dispatch is correct on
x86 with AVX2 actually selected (not just the ARM/NEON path the Mac tested).

## 2. AVX2 numerical validation — the first real execution

`bench/check_cpu_ukernel.cpp` compares each available microkernel against a
naive triple loop over full and edge tiles. On x86 this is the **first time the
AVX2 kernel runs and is checked numerically** (on the Mac it only ran scalar +
NEON).

```sh
g++ -std=c++2b -O2 -I include bench/check_cpu_ukernel.cpp -o check_uk -pthread
./check_uk        # expect: scalar: ok / avx2: ok / ALL OK
```

`-pthread` is required on Linux (the thread pool uses `std::thread`). If
`avx2: ok` does not print, confirm the CPU has AVX2+FMA (step 0) — the harness
only tests kernels the build compiled in (`#ifdef TL_CPU_X86`).

## 3. GEMM benchmark vs ref vs OpenBLAS — the gate

Linux build of the dispatch-path bench: drop the Mac `-framework` flags, add
`-pthread`, and link OpenBLAS via pkg-config for the gate.

```sh
# own vs ref only:
g++ -std=c++2b -O2 -I include bench/bench_cpu_gemm.cpp -o bcg -pthread
./bcg

# with the OpenBLAS gate (threads matched inside the harness):
g++ -std=c++2b -O2 -I include -DBENCH_HAS_OPENBLAS $(pkg-config --cflags openblas) \
    bench/bench_cpu_gemm.cpp -o bcg_gate -pthread $(pkg-config --libs openblas)
./bcg_gate
```

If `pkg-config openblas` is missing on the distro, substitute the explicit
paths (`-I/usr/include/x86_64-linux-gnu -lopenblas`, adjust per distro). Unlike
the Mac — where Homebrew's OpenBLAS was an untuned ARMv8 build and only a loose
lower bound — a properly built x86 OpenBLAS here is the **real** gate baseline.

**Gate: own ≥ 90% of OpenBLAS** (the `own/blas %` column), single- and
multi-threaded, and ≥ PyTorch CPU. This is the verdict the Mac could not give.

## 4. Re-tune block sizes for this microarchitecture

The MC/KC/NC and the 8×8 tile were tuned on M1 Pro NEON; the x86 cache
hierarchy and the AVX2 register file differ, so re-sweep. `bench_cpu_sweep.cpp`
calls `cpu::sgemm` directly (≈1s compile) with `-D` overrides:

```sh
for kc in 256 384 512 768; do
  g++ -std=c++2b -O2 -I include -DTL_CPU_KC=$kc bench/bench_cpu_sweep.cpp \
      -o sw_$kc -pthread
done
# interleave, take per-size medians (see below), pick the KC; repeat for MC.
for i in 1 2 3 4; do for kc in 256 384 512 768; do ./sw_$kc; done; done
```

Also worth an A/B on this µarch: the current **8×8** AVX2 tile vs a **6×16**
(6 rows × 2 ymm cols = 12 accumulators of 16 — often the better AVX2 register
blocking; needs a NR=16 pack variant). The 8×8 was chosen to share the NEON
packing, not because it is x86-optimal.

## 5. Measurement discipline (from performance-notes.md)

- Check `uptime` first — the load-1 average should be low (< ~nproc/2) or the
  numbers are noise. On WSL2, also close Windows-side CPU hogs.
- **Interleave** A/B builds and take **medians**, never a single run.
- Record which side of the WSL2 boundary a number came from (WSL2 adds
  submission latency vs native Linux).
- Track **% of AVX2 fp32 peak** as the architecture-independent yardstick:
  `peak_GFLOPs = cores × freq_GHz × 2 (FMA units) × 8 (fp32/ymm) × 2 (flops/FMA)`.
  Fill in the box's `cores`/`freq` and compare the 2048³ number against it, the
  way the NEON census tracked ~86–89% of the M1 Pro's ~410 GFLOP/s.

## 6. Recording results

Add an "Own CPU GEMM x86 census" section to performance-notes.md mirroring the
NEON one: the own/ref/OpenBLAS table, % of AVX2 peak, the chosen MC/KC/NC, the
8×8-vs-6×16 verdict, and any refuted approaches. Update roadmap M5: flip the
"AVX2 execution + gate" remaining item and note the OpenBLAS-90% verdict.

## Deferred (still not this pass)

- **AVX-512 kernel** — wants NR=16 (a second packing layout). The dispatch seam
  is ready: add an `avx512f` branch to `select_ukernel` and the wider ukernel.
  Only worth it if the box's CPU has AVX-512 and the AVX2 gate is met first.
- **Native Windows / MSVC AVX2** — see the roadmap Open-decisions entry
  (three-tier: header-only default, `/arch:AVX2` opt-in, `cpu_avx2.cpp` source
  TU for runtime dispatch). Independent of this Linux pass.
