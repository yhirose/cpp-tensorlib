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
and competitive (see the silarray comparison in performance-notes.md). The
non-Apple backends have landed their F32 foundation (M5 CPU met the OpenBLAS
gate; M6 CUDA has a correct, split-K SGEMM at ~0.82 of cuBLAS). What remains is
reoriented by the target scope below: BF16/FP16 **compute** (M7), quantized
dequant-matmul (M8), and fused attention + KV cache (M9) — the kernels that
actually make the target workloads fast — plus a tail of deferred perf/CI/API
work.

Design invariants that constrain everything below:

- **Zero third-party runtime dependencies.** macOS links only OS frameworks;
  CPU/CUDA use own kernels (no OpenBLAS/cuBLAS/CUTLASS); the CUDA driver is
  `dlopen`'d. doctest is vendored (tests only).
- **One dispatch seam.** Backends plug in at `graph::eval_one`; each returns
  nullopt/false when ineligible, so the evaluator has no platform `#ifdef`s.
  A new backend is a new set of `*_mode_()` guards + kernels, nothing else.
- **The reference `ref::` backend is permanent** — correctness oracle and the
  fallback for any unsupported shape/target.
- **F32 is the compute baseline**; BF16/FP16 storage *and* compute arrive in M7,
  and int4/int8 as a *storage+dequant* type in M8 (compute stays F32/BF16-accum,
  no F64). Widening the dtype surface is now in scope — see below.

## Target scope & the bar

The project targets **consumer/prosumer local AI**: running and fine-tuning
neural nets — Local LLM inference (chat/decode) and LoRA/QLoRA fine-tuning — on
**one to a few consumer GPUs** (RTX 5090 = Blackwell sm_120; A6000 = Ampere
sm_86). Large-scale multi-node LLM *training* is explicitly **out of scope**.

This scope resets the performance bar, and it is not "beat cuBLAS at 4096³":

- **The relevant competitor is llama.cpp / exllamav2 / ggml / MLC-LLM, not
  PyTorch+cuBLAS.** Those are the SOTA local runtimes, and they are hand-written
  with **no cuBLAS on the hot paths** — proof that own-kernels is the *right*
  approach here, not a compromise. The operative metric is **tokens/sec on a
  real quantized model**, not GFLOP/s vs cuBLAS.
- **Interactive decode (the dominant cost) is memory-bandwidth-bound**
  GEMV / dequant-matmul (batch≈1, each weight streamed once), where cuBLAS has
  **no moat** — a hand kernel that saturates VRAM bandwidth matches or beats it.
  Attention is flash-attention-style fused work cuBLAS doesn't even do. Only
  **prefill / larger batch** is compute-bound dense GEMM (the regime where the
  hand SGEMM tops ~0.90 of cuBLAS — see M6), and it is a one-time, minority cost
  for interactive use.
- Therefore the M6 **cuBLAS-90%-on-square gate is consciously de-prioritized**
  (measured, reached ~0.82 with split-K, kept as the prefill foundation — not
  chased with a CUTLASS escape hatch the scope doesn't need). The high-leverage
  work is the dtype/quant/attention kernels (M7–M9), all hand-written-friendly
  and cuBLAS-irrelevant. Full analysis (the decode=bandwidth-bound regime split,
  the silarray/llama.cpp precedent) is in performance-notes.md.

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

**Done (stage-1 build + seam integration, 2026-07-03):** the `TENSORLIB_CUDA`
CMake path compiles the kernels to PTX (nvcc), bin2c's them to a byte-array
`.inc` on the include path, defines `TENSORLIB_CUDA`, and links `libdl` — no
CUDA runtime link (`cmake/bin2c.cmake` is the portable, xxd-free converter).
The eval seam is now a **`gpu::` facade** (in cuda.h: `namespace gpu = cuda`
off-Apple, `= metal` on Apple / non-CUDA builds), so array.h renamed its
`metal_*` helpers to `gpu_*` and its `metal::` calls to `gpu::` — one alias is
the only platform `#ifdef`. `storage::make_device_` uses `gpu::alloc`, so CUDA
managed memory flows in with no extra branch. **`ctest` cpu/gpu/auto + a
`cuda_ukernel` oracle all pass with the real CUDA backend on the RTX 3090**
(gpu/auto route to CUDA; `cuda_ukernel` skips-as-pass with no device, matching
the driver-absent CI fallback). Non-CUDA build unchanged (3 tests green). The
old `kernels/smoke.cu` build-check is removed — the real kernel → PTX compile
now serves that role.

**Done (stage-2 memory-model pivot, 2026-07-03):** standing up the SGEMM census
exposed that the managed-memory model (below) collapses compute-bound GEMM ~88×
on WSL2 (`ConcurrentManagedAccess=0` → no page migration; prefetch/advise both
"invalid device ordinal"). Since no kernel tuning matters on managed memory and
PyTorch uses device memory, the backend moved to the pre-authorized **explicit
device-buffer model**: a persistent host/device **mirror** per allocation (host
malloc + `cuMemAlloc` device buffer; dirty state keyed by the device pointer in
cuda.h so views sharing a storage share one entry), lazy H2D before a kernel
reads a host-dirty buffer, D2H before the CPU reads a device-dirty one
(`array::raw()/data()` → `gpu::sync_to_host`). Metal stays a strict no-op
(genuinely unified). storage.h is unchanged; the seam addition is two lines in
array.h. ctest cpu/gpu/auto + the cuda_ukernel oracle validate the coherence.
See performance-notes.md "Own CUDA GEMM stage-2" and [[cuda-wsl2-managed-memory-cliff]].

**Done (stage-2 SGEMM ladder, first pass, 2026-07-03):** `tl_sgemm_rb` — a
128×128×8 blocktile / 64×32 warp-tile / 8×8-microtile kernel (float4 global
loads, register-staged double buffer, conflict-free warp-broadcast smem reads),
dispatched as the fast path for NN-contiguous K%8/N%4/16B-aligned shapes;
everything else keeps the correctness-first `tl_sgemm`. Ladder (interleaved
own/cuBLAS): naive ~0.09 → register-block 0.68/0.75 → +warp-tile 0.71/0.80 →
+double-buffer **0.72–0.84 (2048³) / 0.79–0.83 (4096³)**. Correct throughout
(max rel ≤ 6e-5). **Gate not yet met:** own ≈ 80% of cuBLAS, ≈ 75% of PyTorch-
FP32 (which bundles a faster cuBLAS 12.8). BK=16 measured-and-rejected.

**Done (stage-2 ladder ① rejected + ② split-K landed, 2026-07-04):** the
128×256 large-tile rung (①) was built (`tl_sgemm_rb2`, 8×16 microtile) and
**rejected** — 128 accumulators → 203 regs → 1 block/SM (16.7% occ vs rb's 33%),
losing at every size *including* 4096³ (0.81→0.75, where wave-quant is a
non-issue), so the arithmetic-intensity gain is dwarfed by the occupancy loss;
big spatial tiles are ceiling-capped at rb on sm_86 (see the refuted table). The
census reframed the deficit as small-size *fill* (own/cuB rising with size), so
**split-K (②) landed**: a single `ksplit`/`blockIdx.z` addition to `tl_sgemm_rb`
partitions K into S z-slices (S× more blocks), atomicAdd'ing partials into a
zeroed C (identity scale/offset only). S=2 is the measured optimum — 1024³
0.63→0.75, 2048³ 0.74→0.76; 4096³ stays S=1 (6 waves already, split is pure
overhead). Auto: S=2 for base<512 blocks & K≥512. Post: 1024³ ~0.72–0.82, 2048³
0.76, 4096³ 0.82–0.83. Still 127 regs / 2 blocks/SM; ctest cpu/gpu/auto green.

**Done (stage-2 ladder ③ cp.async rejected, 2026-07-04):** the Ampere `cp.async`
staged pipeline (`tl_sgemm_cp`, 3/4-stage) was built and **rejected** — it loses
at 4096³ (0.83→0.73) and 2048³ because the transposed-A smem layout the fragment
read needs can't be filled by an efficient 16-byte cp.async (cp.async can't
transpose), forcing 4×4-byte scatter (the slow granularity) + an extra barrier.
Efficient cp.async here needs `ldmatrix` (tensor-core-only). See the refuted
table. The register-staged `tl_sgemm_rb` load path stands.

**Remaining — the efficiency ceiling (~0.82 at 4096³) is the open gate.** All
three documented rungs are now spent: ① 128×256 large tile (rejected, occupancy
cliff), ② split-K (**landed**, fixed the small-size fill deficit), ③ cp.async
(rejected, transpose mismatch). The remaining ~8–18% to the cuBLAS-90% / beat-
PyTorch gate is per-SM kernel efficiency at large sizes, where the hand-written
SIMT kernel structurally trails cuBLAS (which uses tensor-core-adjacent
`ldmatrix`/mma paths even for its FP32 SGEMM on Ampere). Options: (a) the
pre-authorized **per-shape CUTLASS escape hatch** for the large band; (b) accept
~0.82 and document the band as a known gap; (c) further micro-scheduling with
uncertain payoff.

**Resolved (2026-07-04): option (b) — accept ~0.82, no CUTLASS.** Per the target
scope, large-square compute-bound GEMM is *prefill only* (a minority, one-time
cost); the dominant local-LLM cost is bandwidth-bound decode where cuBLAS has no
moat. So the ~0.82 split-K SGEMM is kept as the **prefill/compute-bound
foundation** and the cuBLAS-90% chase is dropped. The escape hatch stays
*pre-authorized but unused* — if a real workload ever proves prefill-GEMM-bound,
reopen it (with the binary-size feature-flag design: CUTLASS confined to the
build-time kernel `.cu`, `-DTENSORLIB_CUDA_CUTLASS` off by default, so
non-matmul builds carry zero CUTLASS bytes). Reaching ~0.90 by hand is also
possible (Boehm hits ~0.93 on sm_86 via a full tile autotune + fragment-register
pipeline + swizzled cp.async) but is deferred as lower-leverage than M7–M9.
Also unchanged: SteelLoaderT-equivalent for transposed operands (backward
matmuls use the naive `tl_sgemm` today). All measured on the RTX 3090, WSL2 noted.

**Done (loader probe + memory model, 2026-07-03):** the dlopen'd-driver design
is validated on the RTX 3090 box — a standalone probe declares the driver API
itself (no CUDA headers), dlopens `/usr/lib/wsl/lib/libcuda.so.1`, and reaches
the device (`cuInit` → driver API 13.1, RTX 3090 sm_86, 82 SMs) with **no CUDA
Toolkit installed**. So the host side needs only libdl; nvcc is a build-time-
only prereq for the kernel PTX (`cuda-toolkit-13-0`, matching the 13.1 driver).
**Memory model (SUPERSEDED at the stage-2 gate — see the memory-model pivot
above): originally CUDA managed memory** (`cuMemAllocManaged`) — one allocation
visible to host and device, reusing the Apple unified-memory seam, the driver
migrating pages on demand. Chosen for minimal churn + correctness-first. The
explicit device-buffer + H2D/D2H alternative was deferred *iff* managed-memory
page migration proved a bottleneck (prefetch the first lever). It did — on WSL2
`ConcurrentManagedAccess=0` means pages never migrate and prefetch is
unavailable, so the device-buffer mirror replaced managed at stage 2.

### M7 — BF16 / FP16 storage **and compute**

BF16/FP16 as a *storage* type first (memory is 16-bit, load widens to F32, store
narrows — a bit-shift confined to load/store, compute stays F32, no dtype × ISA ×
kernel explosion). Halves bandwidth on memory-bound ops — directly the decode
lever. Pairs with a language-level dtype surface in culebra.

**The compute path is now in scope (was "later, demand-driven" — the scope
demands it).** LoRA/QLoRA fine-tuning trains in BF16/FP16 with F32 accumulation,
and a 16-bit compute GEMM halves weight bandwidth for decode too. So M7 adds a
BF16/FP16 GEMM with F32-accumulate: on CUDA the `tl_sgemm_rb` register-blocked
kernel generalizes (16-bit smem tiles, F32 accumulators — the microtile/warp
structure is unchanged); on Ampere+ a Tensor-Core (`mma.sync` bf16→f32) path is
the fast option and, unlike the FP32 SGEMM, here Tensor Cores are the *native*
datapath so the hand kernel is on equal footing with cuBLAS. FP8 (sm_120 /
Blackwell) is a forward-looking extension of the same seam.

### M8 — Quantized (int4/int8) dequant-fused matmul

The heart of consumer local-LLM inference: 4-bit/8-bit weights (VRAM fit +
bandwidth) with a **fused dequantize-and-matmul** kernel — read packed int4/int8
+ per-group scales, widen to BF16/F32 in registers, accumulate. This is a
*storage+dequant* dtype (activations stay BF16/F32; only weights are quantized),
so it rides the M7 dtype surface and the existing GEMM/GEMV structure with a
widen-on-load epilogue. Decode is GEMV-shaped and bandwidth-bound, so the win is
reading 4× fewer weight bytes — a regime cuBLAS doesn't serve and where
llama.cpp/exllamav2 hand-write. GGUF-style group quantization (Q4_K etc.) is the
reference format to stay interop-friendly. Gate: **tokens/sec within reach of
llama.cpp/exllamav2** on the same quantized model + GPU (not GFLOP/s vs cuBLAS).

### M9 — Fused attention + KV cache

Flash-attention-style fused attention (tiled QKᵀ → online softmax → ·V, never
materializing the S×S scores), causal masking, and a **KV cache** (append per
decode step, GQA/MQA head sharing). cuBLAS does not do attention at all, so this
is squarely own-kernel territory (FlashAttention is itself hand-written CUDA).
Promotes the deferred "GPU fused kernels" item. Decode attention is KV-cache-
bandwidth-bound; prefill attention is compute-bound and long-context-sensitive.
This is what turns the M6/M7/M8 GEMM kernels into an actual runnable LLM
(alongside the model layer — RoPE, norm, MLP — built in culebra or an app layer).

## Deferred / conditional work

Tracked so it isn't re-discovered. Each is gated on a trigger, not scheduled.

| Item | Trigger to act |
|---|---|
| **SteelLoaderT** — transposed-operand STEEL for backward matmuls (`xᵀ@g`, `g@Wᵀ`) | a real GPU workload whose backward is transpose-bound (census-driven) |
| **GPU fused kernels** — `linear_sigmoid`, online-softmax (silarray has these; tensorlib's GPU softmax/MLP trail silarray) | **promoted → M9** (fused attention is the flagship case); general fused-epilogue kernels ride the same seam |
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

- **CUDA performance bar — RESOLVED (2026-07-04).** The bar is **tokens/sec vs
  llama.cpp/exllamav2 on a real quantized model**, not cuBLAS-90% on square GEMM.
  Rationale: the target scope (consumer local-LLM inference/fine-tuning) is
  decode-dominated = bandwidth-bound, where cuBLAS has no moat and hand-written
  is SOTA. The M6 cuBLAS gate is kept only as the prefill/compute-bound
  foundation (~0.82, split-K). Full regime analysis in performance-notes.md.
- **CUTLASS escape hatch — RESOLVED (2026-07-04): pre-authorized but unused.**
  Not needed for the scope (see M6 Resolved). If ever reopened, it is confined
  to the build-time kernel `.cu` behind `-DTENSORLIB_CUDA_CUTLASS` (off by
  default) so non-matmul / non-opted-in binaries carry zero CUTLASS bytes and
  the header-only consumer surface is preserved.
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
