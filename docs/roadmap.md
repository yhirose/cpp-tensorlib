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

**Remaining:**
- *AVX2 / AVX-512 microkernels* — write behind the same interface; validate
  and tune on the x86 WSL2 box (can't execute on ARM).
- *Runtime CPUID dispatch* — currently the microkernel is compile-time
  selected (NEON vs scalar); x86 needs a runtime AVX2/AVX-512/scalar pick.

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

## Notes on effort

The remaining milestones split the same way M3b and M6 do: **scaffolding**
(dispatch seams, loaders, packing loops, build/CMake) is mechanical and
follows documented designs; the **kernels themselves** — reaching the
OpenBLAS/cuBLAS-90% and beat-PyTorch gates — are performance-sensitive sprints
where the STEEL-sprint discipline applies (per-kernel census, interleaved A/B,
record refuted approaches). Plan each milestone as *scaffold to correct, then
a dedicated tuning pass to the gate.*
