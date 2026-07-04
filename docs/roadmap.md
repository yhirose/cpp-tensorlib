# cpp-tensorlib roadmap

Forward-looking plan: milestone scope, environment/hardware constraints, the
approach for each remaining milestone, and open decisions. Complements
[architecture.md](architecture.md) (how the code works today) and
[performance-notes.md](performance-notes.md) (measurement methodology, gate
results, refuted approaches). This is the portable source of truth for the
plan — it travels with the repo, so any checkout on any machine has it.

## Where things stand

M1–M4 are done: reference backend, lazy graph + fusion, the Accelerate and
Metal (STEEL SGEMM) backends behind a single dispatch seam, culebra
integration, and a tiny-tensor per-op sprint. The macOS backend is complete
and competitive (see the silarray comparison in performance-notes.md). What
remains is the non-Apple backends (M5 CPU, M6 CUDA), BF16 (M7), and a tail of
deferred perf/CI/API work.

Design invariants that constrain everything below:

- **Zero third-party runtime dependencies.** macOS links only OS frameworks;
  CPU/CUDA use own kernels (no OpenBLAS/cuBLAS/CUTLASS); the CUDA driver is
  `dlopen`'d. doctest is vendored (tests only).
- **One dispatch seam.** Backends plug in at `graph::eval_one`; each returns
  nullopt/false when ineligible, so the evaluator has no platform `#ifdef`s.
  A new backend is a new set of `*_mode_()` guards + kernels, nothing else.
- **The reference `ref::` backend is permanent** — correctness oracle and the
  fallback for any unsupported shape/target.
- **F32 only** (BF16 as a storage type in M7); no F64.

## Milestones

### M5 — Own CPU backend (Linux / Windows)  🔨 in progress

BLIS-style GEMM: cache-blocking + packing (architecture-independent C) around
a register-blocked SIMD microkernel per ISA (AVX2, AVX-512, NEON), selected
at runtime by CPUID. Plus a small threadpool for the M/N parallel split, and
own SIMD elementwise/reduction kernels. Gate: ≥90% of OpenBLAS and ≥ PyTorch
CPU on the target ISA. Escape hatch: a `-DUSE_OPENBLAS` CMake option for any
shape band that can't meet the gate.

**Done (first cut):** `cpu.h` + `cpu_threadpool.h` — the full BLIS structure
(blocking, stride-aware packing, persistent thread pool, M-parallel), an 8×8
NEON microkernel + a portable scalar fallback, wired into the dot dispatch
(`metal → accel → cpu → ref`) and gated by `cpu::enabled_`. Correct on every
shape/transpose/epilogue (own == ref oracle test); ~235 GFLOP/s at 1024³ on
M1 Pro NEON. Elementwise/reductions stay on array.h's autovectorized flat
loops (memory-bound), by design.

**Done (NEON tuning pass, 2026-07-03):** lane-indexed-FMA microkernel
(K-unroll ×4 + prefetch), thread-local reusable pack buffers, MR-panel-
granular M-parallelism, KC 256→512. Quiet-census verified: **~86–89% of
NEON fp32 peak at 2048³, ~84% at 1024³** (was 54–64%); the sub-256³ loss
to the naive loop is fixed (128³ now 2.4× ref). KC=512 confirmed and an
8×12 kernel measured-and-rejected (within noise). The on-Mac NEON tuning
is complete — see performance-notes.md "tuning pass" for the census and
the rejected 8×12.

**Done (AVX2 + runtime dispatch scaffold, 2026-07-03):** the compile-time
`#ifdef` microkernel pick is now a runtime function pointer (`select_ukernel`)
— ARM fixes to NEON at compile time; x86 probes `__builtin_cpu_supports`
and picks AVX2 (`target("avx2,fma")`) or the scalar fallback. Added an AVX2
8×8 microkernel (`_mm256_fmadd_ps` + `_mm256_broadcast_ss`, the x86 analogue
of the NEON lane-FMA kernel, K-unroll ×4 + prefetch), sharing the MR=8/NR=8
packed layout. Scalar + NEON validated against a naive oracle on ARM; the
AVX2 path cross-compiled (`-target x86_64 … -c`) and disassembly-confirmed
to emit `vfmadd*`/`vbroadcastss` — but **not executed** (Rosetta stops at
SSE4.2). Full suite green on native ARM.

**Done (AVX2 execution + 6×16 tuning + gate, 2026-07-03):** first real run on
the i7-12700KF WSL2 box (see performance-notes.md "x86 census" + the runbook).
AVX2 numerically verified vs the naive oracle and the full suite. Tuned the
tile **8×8 → 6×16** (12 ymm accumulators): single-core **~91% of AVX2 peak**
(was ~77%), matching NEON; **own ≈ 106% of OpenBLAS at 2048³ — gate MET**. Each
kernel now packs to its own tile via `ukernel_desc` (the anticipated second
packing layout; AVX-512 reuses NR=16). KC=512 re-confirmed for x86; 8×8 kept
behind `-DTL_CPU_AVX2_8X8` for A/B.

**Remaining:**
- *AVX-512 microkernel* — deferred: a real AVX-512 kernel wants NR=16 — now a
  ready-made layout (the 6×16 AVX2 kernel packs NR=16), so `select_ukernel`
  needs only an `avx512f` branch + the wider ukernel. This box has no AVX-512;
  needs a Sapphire/Ice Lake / Zen4 box.
- *Problem-size-aware thread count* — see the deferred table: the pool statically
  splits M across all 20 threads, over-parallelizing mid-size GEMM (512³/1024³
  lose to OpenBLAS, which caps threads for small problems). Single-core is 91%
  of peak, so this is a thread-policy gap, not a kernel one. Best measured on
  native Linux (WSL2 hides the P/E topology the heuristic needs).

**Environment:** Apple Silicon is ARM64 = NEON, so the NEON microkernel is
fully developable, natively runnable, and perf-tunable on a Mac (it is also
the production path on ARM Linux — Graviton/Ampere). The **AVX2/AVX-512
microkernels can be written and compile-checked on a Mac but not executed**
(Rosetta 2 stops at SSE4.2; there is no native x86). Validating the x86 path
needs x86 Linux — the WSL2 box (see M6) or x86 CI (correctness only; CI perf
is too noisy for the gate). Note the macOS *production* CPU path stays
Accelerate (AMX beats hand-written kernels here); M5 on a Mac is validated
against the `ref::` baseline, and its real payoff is off-Apple.

**Approach:** scaffolding (blocking loops, packing, threadpool, CPUID
dispatch) and a correct NEON microkernel first — a naive-but-correct kernel
should already beat `ref::`. Then a dedicated tuning pass to reach the
OpenBLAS-90% gate (register blocking, FMA scheduling, packing layout) — this
is the performance-sensitive part; treat it like the STEEL sprint. Defer
AVX tuning to the x86 box.

### M6 — Own CUDA backend (Linux / Windows)

FP32 SIMT SGEMM (own kernels, no cuBLAS/CUTLASS), driver API `dlopen`'d so a
binary runs — and falls back to CPU — without a driver present. Kernels AOT
to PTX/fatbin and `#embed`'d, loaded via the driver API. Gate: ≥90% of cuBLAS
FP32 and ≥ PyTorch CUDA. Escape hatch: per-shape-band CUTLASS instances if a
band can't meet the gate.

**Environment:** needs an NVIDIA GPU. Available: **RTX 3090 (sm_86) on Ubuntu
under WSL2.** The WSL2 CUDA driver lives at `/usr/lib/wsl/lib/libcuda.so` —
the dlopen loader's search path must include it. WSL2 adds submission latency
vs native Linux; record which side of that line a measurement came from.
Windows-native is not yet set up. Real numerical/perf validation is manual on
this box (no self-hosted or paid GPU CI — see the CI note). CI covers the
CUDA *build* + the driver-absent CPU-fallback path only.

**Approach:** two stages. (1) Scaffolding — dlopen loader (with the WSL2
path), PTX `#embed` + driver-API kernel launch, build integration
(`TENSORLIB_CUDA` option, only when nvcc is found), elementwise/reduction
kernels. This mirrors the Metal M3b-1 foundation. (2) The SGEMM ladder tuned
to the cuBLAS-90% gate — the performance-sensitive sprint. The x86 WSL2 box
also validates the M5 AVX path.

**Done (stage-1 kernels + cuda.h backend, validated on GPU, 2026-07-03):**
`kernels/tensorlib_cuda.cu` (elementwise add/sub/mul/div, unary exp/log/sqrt/
sigmoid/relu/affine, row_sum/row_max/softmax, and a correctness-first SGEMM
with trans/ld support) compiles to PTX via nvcc 13.0 for sm_86. `include/cuda.h`
is the metal.h analogue: a hand-declared driver API loaded by dlopen, primary-
context + module-from-PTX loader, managed alloc/release, and binary/unary/gemm/
row_op launchers — same signatures as metal.h so the seam is backend-agnostic,
reusing `metal::kop`. The PTX is embedded via **bin2c** (a build-generated
`tensorlib_cuda_ptx.inc` byte array), *not* C23 `#embed`, because the off-Apple
compilers (g++ 11 / clang 14) predate `#embed`. `bench/check_cuda.cpp` validates
all ops on the RTX 3090 vs CPU references (binary+affine, sigmoid, GEMM NN + NT,
row_sum+affine, softmax — all ok). Stubs compile cleanly with no `TENSORLIB_CUDA`
and on Apple, so non-CUDA builds are unaffected.

**Remaining (stage-1 wiring, then stage-2 gate):**
- *Build + seam integration:* the `TENSORLIB_CUDA` CMake path (nvcc → PTX →
  bin2c → include path + `-ldl`), `storage::make_device_` trying `cuda::alloc`,
  and array.h's eval_one routing GPU ops to CUDA off-Apple (a `gpu::` facade
  aliasing metal on Apple / cuda elsewhere, so eval_one stays #ifdef-free), then
  `ctest` cpu/gpu/auto exercising the real CUDA path.
- *Stage-2 SGEMM ladder:* the correctness-first one-output-per-thread kernel →
  shared-memory tiling / register blocking to the cuBLAS-90% gate.

**Done (loader probe + memory model, 2026-07-03):** the dlopen'd-driver design
is validated on the RTX 3090 box — a standalone probe declares the driver API
itself (no CUDA headers), dlopens `/usr/lib/wsl/lib/libcuda.so.1`, and reaches
the device (`cuInit` → driver API 13.1, RTX 3090 sm_86, 82 SMs) with **no CUDA
Toolkit installed**. So the host side needs only libdl; nvcc is a build-time-
only prereq for the kernel PTX (`cuda-toolkit-13-0`, matching the 13.1 driver).
**Memory model decided: CUDA managed memory** (`cuMemAllocManaged`) — one
allocation visible to host and device, so `storage.native == storage.contents`
and the existing unified-memory seam (built for Apple) is reused unchanged; the
driver migrates pages on demand. Chosen for minimal churn + correctness-first,
mirroring Metal. The explicit device-buffer + H2D/D2H alternative is deferred to
the stage-2 gate *iff* managed-memory page migration proves a bottleneck
(prefetch hints `cuMemPrefetchAsync` are the first lever before that).

### M7 — BF16 storage type

BF16 as a *storage* type (memory is BF16, load widens to F32, store narrows),
so the conversion is a bit-shift confined to load/store and the compute path
and kernels stay F32 — no dtype × ISA × kernel explosion. Halves bandwidth on
memory-bound ops. Pairs with a language-level dtype surface in culebra. A BF16
*compute* path (Tensor Core / bf16 SIMD) is a later, demand-driven step and is
where reconsidering CUTLASS/library kernels would make sense.

## Deferred / conditional work

Tracked so it isn't re-discovered. Each is gated on a trigger, not scheduled.

| Item | Trigger to act |
|---|---|
| **SteelLoaderT** — transposed-operand STEEL for backward matmuls (`xᵀ@g`, `g@Wᵀ`) | a real GPU workload whose backward is transpose-bound (census-driven) |
| **GPU fused kernels** — `linear_sigmoid`, online-softmax (silarray has these; tensorlib's GPU softmax/MLP trail silarray) | a GPU training/inference workload that is fusion-bound |
| **Tiny-tensor corner** (n_embd≈16) — node pooling / small-vector fields | a real workload that lives in the ~16-element regime (documented as accepted, not a bug) |
| **View-crossing fusion** — eager view eval breaks fusion across reshape/transpose | a transformer workload where this shows up in a census |
| **Problem-size-aware CPU thread count** — pool statically uses all `hardware_concurrency()` threads; over-parallelizes mid-size GEMM (512³/1024³ < OpenBLAS on hybrid P/E+HT), where OpenBLAS caps threads | a native-Linux x86 measurement session (WSL2 hides P/E topology); or a workload bound by mid-size CPU GEMM |
| **culebra default → `auto`** — currently `cpu` | large-tensor workloads appear, or after CUDA lands and thresholds are re-measured |
| **`reshape` on non-contiguous**, **other-axis slice/concat** | a workload / language feature that needs it (tensorlib can already materialize) |

## CI / infrastructure

Current CI (GitHub Actions, free tier): 3 OSes × cpu/gpu/auto modes;
`macos-15` runs real Metal on its paravirtual GPU; ubuntu/windows run the CPU
path + a CUDA build check + the driver-absent fallback. **No GPU numerical CI**
— self-hosted and paid GPU runners were declined (cost/ceremony); CUDA numeric
regression is manual on the RTX 3090 box. To add: a binary-size job, and (once
M5/M6 land) a `just` recipe for the manual CUDA/x86 regression run, and a
Tensor-unused-binary size check on the culebra side.

## Open decisions

- **culebra default device** — stays `cpu` for now; revisit per the deferred
  table. Rationale in performance-notes.md.
- **auto thresholds are M1-Pro-specific** — must be re-measured per Mac and a
  full set produced once CUDA lands.
- **Windows-native environment** — not set up; Windows is currently
  build-checked via CI only.
- **MSVC support vs header-only (AVX2 on Windows)** — decided (2026-07-03).
  header-only is a hard invariant and consumers are MSVC-majority, so the
  tension is: MSVC has no per-function target attribute and `/arch:AVX2` is
  TU-global, so header-only runtime-dispatched AVX2 (what GCC/Clang get via
  `__attribute__((target)))`) cannot be replicated on MSVC. Resolution — a
  three-tier model, dropping no one:
  1. *Default, header-only:* GCC/Clang do runtime dispatch (already built).
     MSVC compiles clean and runs the **scalar** kernel today — the x86 SIMD
     path is gated on `__x86_64__`, which MSVC (`_M_X64`) doesn't define, so
     nothing MSVC-hostile is even reached. Correct, just not accelerated.
  2. *MSVC AVX2, opt-in, header-only preserved:* build with `/arch:AVX2` →
     `select_ukernel` picks AVX2 at compile time (`#if defined(__AVX2__)`).
     Caveat: `/arch:AVX2` is TU-wide, so such a binary `#UD`s on pre-AVX2
     CPUs with no runtime fallback — fine for source-built-for-a-known-target
     or an AVX2 baseline, not for an adaptive redistributable.
  3. *MSVC runtime dispatch (adaptive redistributable), opt-in:* link a small
     AVX2-only **source** file (`cpu_avx2.cpp`, not yet written) compiled
     per-file with `/arch:AVX2`; baseline TUs stay AVX2-free; CPUID picks via
     the existing `ukernel_fn` pointer (the TU boundary is the isolation the
     target attribute gives on GCC/Clang). Ship **source, never a prebuilt
     `.obj`** — an obj would need a {toolset}×{CRT: /MT,/MTd,/MD,/MDd}×{arch}
     matrix (LNK2038 on CRT mismatch), can't cover future toolsets, and a
     binary blob violates the zero-dep / build-from-source / auditable ethos.
     A source file auto-matches the consumer's toolchain and stays auditable.
  Small mechanical seams alongside this: `__builtin_prefetch` → `_mm_prefetch`
  and `__builtin_cpu_supports` → `__cpuid`/`__cpuidex` under `_MSC_VER`. None
  of this blocks the gates — WSL2 unblocks all x86 M5 + M6 CUDA; native-Windows
  AVX2 is a scoped follow-up.

## Notes on effort

The remaining milestones split the same way M3b and M6 do: **scaffolding**
(dispatch seams, loaders, packing loops, build/CMake) is mechanical and
follows documented designs; the **kernels themselves** — reaching the
OpenBLAS/cuBLAS-90% and beat-PyTorch gates — are performance-sensitive sprints
where the STEEL-sprint discipline applies (per-kernel census, interleaved A/B,
record refuted approaches). Plan each milestone as *scaffold to correct, then
a dedicated tuning pass to the gate.*
