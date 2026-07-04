# Performance notes — how to measure, and what not to retry

Started on day one (a silarray lesson: record refuted approaches so dead
ends are never re-walked). Every dispatch gate and design choice on a hot
path must carry the measurement that set it.

## Performance gates

The reference target is **PyTorch/libtorch** on the same hardware for the raw
GEMM foundation (silarray benched against MLX; this project is cross-platform, so
PyTorch is the portable BLAS-level bar).

| Backend | Foundation gate (raw GEMM) |
|---|---|
| macOS (Accelerate + Metal) | ≥ silarray (which is ≈ MLX) and ≥ PyTorch MPS |
| Own CPU GEMM | ≥ 90% of OpenBLAS single/multi-thread, ≥ PyTorch CPU |
| Own CUDA GEMM | ≥ 90% of cuBLAS FP32 — **scoped: prefill/compute-bound only** |

Where a gate cannot be met on a shape band after honest effort, record the
band and the data here, then consider the escape hatch (OpenBLAS / CUTLASS
for that band only) — do not ship a silent loss.

**The CUDA GEMM gate is scoped, not absolute (2026-07-04).** The target scope is
consumer local-LLM inference/fine-tuning (roadmap.md "Target scope"), which is
decode-dominated and therefore *bandwidth-bound*, not compute-bound-GEMM-bound.
So the operative CUDA bar is **tokens/sec vs llama.cpp/exllamav2 on a real
quantized model**, and the cuBLAS-90%-on-square gate applies only to the
minority prefill/large-batch band — where it is **consciously accepted at ~0.82**
(split-K), not chased. See "CUDA GEMM regime analysis" below.

## CUDA GEMM regime analysis — why the cuBLAS gate is de-prioritized (2026-07-04)

The decision chain that reset the CUDA bar, recorded so it isn't relitigated:

- **cuBLAS's moat is SASS, and it only matters in one regime.** Hand-written FP32
  SIMT SGEMM tops ~0.90–0.93 of cuBLAS on sm_86 (Boehm's public kernel is the
  existence proof; ours is ~0.82 with split-K and untuned tiles). The last
  ~7–18% is register-bank/instruction scheduling that ptxas won't reach from
  CUDA C++ — cuBLAS's ~15-year SASS tuning. This gap is real but lives **only in
  large compute-bound dense GEMM**.
- **The target workload rarely hits that regime.** Local-LLM interactive
  **decode** (the dominant cost) is batch≈1 GEMV / dequant-GEMV — each weight
  streamed once, FLOP-intensity ~2 → **100% VRAM-bandwidth-bound**. cuBLAS has
  **no advantage** on bandwidth-bound work; a hand kernel that saturates memory
  ties or beats it. **Attention** is flash-attention-style fused work cuBLAS
  doesn't do. Only **prefill / larger batch** is compute-bound GEMM — a one-time,
  minority cost for interactive use.
- **The SOTA precedent confirms it.** llama.cpp / exllamav2 / ggml / MLC-LLM are
  the dominant local runtimes and hand-write their CUDA/Metal kernels with **no
  cuBLAS on the hot paths**. Own-kernels is the *correct* approach for this scope,
  not a compromise. And silarray's own data (bench/README) is the same shape:
  it *tied* PyTorch-MPS on large square (0.96–1.0×) but *won 1.8×* on the
  DL/small-batch shape (32×4096×768) — the win was never in the big-square regime.
- **Conclusion.** Keep the ~0.82 split-K SGEMM as the prefill foundation; do not
  spend the CUTLASS escape hatch or a SASS-level sprint on the square gate. The
  high-leverage CUDA work is the dtype/quant/attention kernels (roadmap M7–M9),
  all bandwidth- or fusion-bound, all hand-written-friendly, all cuBLAS-irrelevant.
  Benchmark them as **tokens/sec vs llama.cpp/exllamav2**, not GFLOP/s vs cuBLAS.

## Measurement discipline (inherited from silarray, verified there)

- GPU clock state swings absolute times ±30–60%: only same-run interleaved
  A/B comparisons are meaningful. Check machine load before believing ratios.
- Per-kernel census → fix the single most expensive kernel → repeat. Don't
  guess at bottlenecks.
- Warm up (~300 ms) before timing GPU work after CPU-only sections.
- Scale numeric tolerances with value magnitude (fp32 summation-order noise).
- Every dispatch gate is a measured value: comment the number and the bench
  that produced it next to the gate.

## Reference hardware

| Machine | Notes |
|---|---|
| Apple M1 Pro | macOS backend; paravirtual GPU on CI is correctness-only — never bench on CI |
| RTX 3090 (sm_86), Ubuntu on WSL2 | CUDA backend; driver at `/usr/lib/wsl/lib/libcuda.so`. WSL2 adds submission latency vs native Linux — record which side of that line a number came from |

## Current baselines (M1 Pro, interleaved medians)

Sprint history for Metal SGEMM at 2048³ (all measured under load avg 5–9,
so treat as lower bounds):

| Step | 2048³ | 1024³ |
|---|---|---|
| M3b-2 basic 32×32×16 tile, scalar loads | ~1300 GFLOP/s | ~1080 |
| + float4 fast loaders (host-gated) | ~2170 | ~1060 |
| + STEEL port (frag registers, swizzle, split edge loops) | **~3200** | **~1700–1960** |

Reference points: Accelerate (AMX) sgemm ≈ 980 GFLOP/s at 1024³ and stays
~constant; M1 Pro GPU fp32 ceiling ≈ 4–5 TFLOP/s; MLX lands ≈ 3.5–4. The
STEEL numbers were taken on a loaded machine — re-measure quiet before
comparing against PyTorch-MPS for the gate.

**Tile-config census (2026-07-02, loaded machine, simple-tile family):**
32×32 ≈ 64×32 ≈ 32×64 all plateau at ~2000–2200 GFLOP/s at 2048³ — the
bottleneck was the loader/pipeline structure, not tile size. 64×32 had the
best small-size behavior and is the non-STEEL fallback.

**Remaining SGEMM work:**
- Transposed operands still use the simple-tile family (STEEL is NN-only
  until SteelLoaderT is ported) — backward-pass matmuls (`xᵀ@g`, `g@Wᵀ`)
  are the motivating shapes.
- Function-constant specialization (fc_mn_aligned / fc_k_aligned) not yet
  ported; runtime branches instead. silarray measured conv2d FCs equivalent,
  but gemm may differ — measure before porting.
- Narrow-N band (n < 48) and small shapes stay CPU-bound → auto-mode
  thresholds are the fix, not kernel work.

## MNIST training gate (M4 integration, 2026-07-02)

Workload: culebra `benchmarks/mnist/train_bench.cul` — 784→30→10 MLP,
batch=10, hand-coded backprop, 6000 steps/epoch. The small-shape stress
test where per-op overhead dominates. Measured interleaved on a LOADED
machine (load avg ~9; medians were stable across 6 rounds regardless):

| Stack | warm s/epoch |
|---|---|
| **culebra + tensorlib (cpu, JIT)** | **0.044** |
| PyTorch CPU (same recipe) | 0.066–0.075 |
| culebra + pre-M4 direct cblas | 0.076 |
| PyTorch MPS | 0.52 |
| culebra + tensorlib (gpu) | 3.8 |

**Gate: PASSED for this workload** — 1.5–1.7x faster than PyTorch CPU,
1.7x faster than the pre-M4 implementation. All rows converge to
accuracy=0.9079 (numerics verified across devices).

**The fix that flipped it** (M4 initially measured 1.6x SLOWER than
pre-M4): culebra's wrapper lifted `t * 2.0` and every VJP scalar to a
rank-0 tensor and called the tensor⊙tensor path — bypassing tl's scalar
overloads, so nothing fused and `W - d.dot(x)*lr` ran a ref:: broadcast
loop over the weight matrix (23520 elems × 6000 steps; `sample` showed
map_binary<multiplies> at 15x the samples of anything else). Routing
materialized rank-0 operands through the scalar overloads restored
affine/GEMM-epilogue fusion. Lesson: **fusion is only as good as the
embedder's entry points — audit the wrapper's lowering, not just the
kernels.** (Also: tl-level step microbench = 23 ms/epoch vs 44 through
culebra; the remaining ~2x is interp/JIT + tape-node overhead, known and
acceptable at batch=10.)

GPU mode loses 87x on this shape (blocking flush per step × 6000; torch
MPS hides latency with async queues and still loses 12x) — that is what
the auto thresholds are for, not kernel work.

## Tiny-tensor sprint (2026-07-02, microgpt gate)

Workload: culebra microgpt (transformer, n_embd=16, 16–256 element
tensors) measured 2.6x slower than the pre-M4 direct executor — the
regime where per-op allocation dominates and the lazy-graph machinery is
pure overhead. Fixes, each measured on a 256-element op microbench
(loaded machine; ratios stable):

| Change | Effect |
|---|---|
| Contiguous flat-loop fast paths in map_unary/map_binary/clone/add_ | add_ 1082→63 ns, clone 869→136 ns |
| graph::run: thread-local scratch + visit-stamp marking (no hash set) | part of ~2x lazy-op win |
| eval_one fast paths on raw node storage (elementwise + tiny dot ≤16K MNK, strided ok, epilogue folded) | dot 814→290 ns |
| as_node memoization + evaluated-node-as-const-cache on adoption | repeat-operand allocs gone |
| Eager-tiny in graph builders (materialized operands, ≤4096 elems, CPU-side): run flat loop, skip node/shell/eval entirely | lazy add 643→204 ns |
| culebra: broadcast check without shape construction; strides mirror → method | wrapper allocs −2/op |

End state vs the pre-M4 executor (interleaved, per-step):

| Workload | old/new |
|---|---|
| microgpt n_embd=16 (torture corner) | 0.55 — old wins 1.8x |
| microgpt n_embd=64 | 0.73–0.86 |
| **microgpt n_embd=128** | **1.4–1.55 — new wins** |
| **MNIST train (batch=10)** | **2.0 — new wins** (was 1.7 before sprint) |

The remaining tiny-corner deficit is the tape+graph double bookkeeping
(~4–5 allocations/op that a direct executor doesn't pay). Grinding it
out needs node pooling / small-vector fields — invasive, and n_embd=16
is a toy regime; the crossover sits between n_embd 64 and 128. Recorded
as accepted, not refuted: revisit only if a real workload lands in it.
Large-size and GPU numbers verified unchanged after the sprint
(checksums identical; 2048³ within load noise of its baseline).

## auto-mode thresholds (M1 Pro, quiet-machine census 2026-07-03)

`use_gpu_`-style gate in graph::metal_mode_: never break a pending GPU
pipeline; otherwise start GPU only above a per-kernel-class size
threshold (types.h auto_threshold_). Finalized from the crossover census
(`misc/census.cpp`, load ~4, interleaved medians — CPU=Accelerate vs
GPU-total=Metal single op + flush):

| Class | Crossover (measured) | Threshold set | Old (provisional) |
|---|---|---|---|
| matmul | 1280³=2.1e9 tie, 1536³=3.6e9 GPU | **2e9** (~1260³) | 5e8 |
| elementwise | 1M cpu / 4M GPU | **2e6** | 4e6 |
| reduction (softmax) | 65536 tie / 262144 GPU | **2e5** | 8e6 |

M1 Pro's AMX makes standalone GPU matmul only pay off past ~1300³;
reductions flip to GPU early (CPU's per-row loop is slow). Verified auto
tracks the faster backend at all four boundaries (mm 512³→cpu,
1536³→gpu; ew 256K→cpu, 8M→gpu) and stays CPU-fast on MNIST training
(auto 0.036 = cpu 0.039 s/epoch — tiny matmuls below threshold).

**These are M1-Pro-specific.** They must be re-measured per Mac GPU/AMX
balance, and a full set is needed once the CUDA backend lands (M6). The
matmul value is the *standalone* crossover; a chained large-matmul
pipeline amortizes the flush and would justify a lower threshold —
unmeasured, so the conservative standalone value stands.

**culebra default device:** left at `cpu` (not flipped to `auto`).
Rationale: the thresholds are machine-specific, current culebra
workloads (MNIST/microgpt-scale) sit below every threshold so auto would
behave like cpu anyway, and cpu is the predictable default; users opt
into GPU with `Tensor.use_gpu()`/`use_auto()`. Revisit if a large-tensor
workload appears or once CUDA lands.

## Own CPU GEMM first cut (M5, M1 Pro NEON, 2026-07-03)

Correctness-first scaffolding (BLIS blocking + packing + 8×8 NEON
microkernel + thread pool), untuned. Numbers on a loaded machine (load
5–11), so directional only:

| | own (NEON, 8 thr) | vs Accelerate (AMX) | vs ref (naive) |
|---|---|---|---|
| 128³ | 24 GFLOP/s | 36× slower | 0.8× (loses) |
| 256³ | 69 GFLOP/s | 17× slower | 1.1× |
| 512³ | 156 GFLOP/s | 8× slower | 3.6× |
| 1024³ | 235 GFLOP/s | 8× slower | ~15× |

Reading: scales correctly (compute-bound win grows with size), ~235
GFLOP/s at 1024³ ≈ 55% of M1 Pro's NEON fp32 peak — decent for an untuned
kernel. **Loses to Accelerate by ~8× because AMX ≫ NEON — expected; the
own kernel is the off-Apple path, not a Mac competitor.** Loses to the
naive loop below ~256³ because per-call overhead (a `std::vector` pack
buffer per call/task, thread-pool sync even for one M-block) dominates
small sizes — the tuning pass's first targets. The real gate (OpenBLAS
90%) is measured off-Apple; deferred with the AVX kernels to the x86 box.

Tuning targets, in likely priority: thread-local reusable pack buffers,
skip the pool for single-block work, MR/NR + MC/KC/NC sweep, software
prefetch, K-unroll in the microkernel. → Done; see the tuning pass below.

**Gate caveat (2026-07-03):** `bench/bench_cpu_gemm.cpp` links OpenBLAS
when present (`tensorlib_bench_cpu`). But Homebrew's OpenBLAS on M1
appears to use generic ARMV8 microkernels, not M1-tuned ones — the
*untuned* own kernel already beats it ~1.8× at 1024³ (threads matched
via `openblas_set_num_threads`), which is implausible against a
properly-tuned OpenBLAS. So on a Mac, treat the OpenBLAS number as a
loose lower-bound only; the meaningful Mac target is **% of NEON fp32
peak** (~410 GFLOP/s on M1 Pro → own is at ~54–64%, the real headroom).
The definitive OpenBLAS-90% gate is measured on the x86 box (M6) with a
properly-tuned OpenBLAS. The tuning sprint should track % of NEON peak
on the Mac and defer the OpenBLAS-90% verdict to x86.

## Own CPU GEMM tuning pass (M5, M1 Pro NEON, 2026-07-03)

The tuning sprint over the first cut. All changes oracle-verified (own
== ref, full suite green). Numbers below are the **quieter census**
(load ~4, one core taken by the `ccusage` statusline that couldn't be
stopped — so figures sit a few % under a truly idle box, but were
stable across runs); the sprint itself was done under heavy load
(4–28), where every change was accepted only on interleaved same-load
A/B.

What landed, in order of impact:

1. **Lane-indexed FMA microkernel** (biggest win). The first cut did 8
   `vdupq_n_f32` broadcasts per k-step; now A is loaded as two vectors
   and consumed with `vfmaq_laneq_f32` (16 FMA + 4 loads per k-step,
   zero dups; 20/32 NEON regs). Plus K-unroll ×4 and
   `__builtin_prefetch` on both panels. Peak (2048³) reached ~354
   GFLOP/s through the dispatch path, ~365 kernel-isolated ≈ **86–89%
   of NEON fp32 peak** (~410 GFLOP/s on M1 Pro), up from 54–64%.
2. **Thread-local reusable pack buffers** — the per-call/per-task
   `std::vector` allocs are gone. Trap for later: a thread_local named
   inside the worker lambda resolves to the *worker's* instance — the
   first attempt read each worker's empty `bpack` and crashed; the fix
   passes the caller's pointer in.
3. **MR-panel-granular M-parallelism** (was MC-block-granular): 256³
   has only 2 MC blocks → 2 of 8 threads used. Threads now take
   contiguous MR-panel ranges, grouped into MC-row chunks to keep the
   L2 blocking.
4. **KC 256→512** (MC=128, NC=2048 unchanged). **Confirmed in the quiet
   census** (4 interleaved sweep rounds): KC=512 wins the compute-bound
   regime — 512³ ~317 vs 256's ~304, 1024³ ~350 vs ~332, 2048³ ~363 vs
   ~349 GFLOP/s — by ~1–5%. KC=256 leads only at 256³ (~231 vs ~216).
   Since the gate is large-matmul throughput, KC=512 stands; the 256³
   give-up is accepted.

Net effect through the dispatch path (`bench_cpu_gemm`, quiet census
vs the pre-tuning binary; medians of interleaved runs):

| | before | after | vs ref |
|---|---|---|---|
| 128³ | 21 GF/s (0.8× ref — lost) | 65 GF/s | 2.4× |
| 256³ | 75 GF/s | 217 GF/s | 9.0× |
| 512³ | 131 GF/s | 310 GF/s | 12.3× |
| 1024³ | 208 GF/s | 340 GF/s | 14× |
| 2048³ | 246 GF/s | 348 GF/s | — |

The small-size loss to the naive loop is fixed (128³ now 2.4× faster
than ref). **8×12 kernel: measured and rejected.** With the machine
quiet the 8×12 variant (24 FMA : 5 loads, 29/32 regs) was mixed and
within noise — a ~4% edge only at 2048³, a ~2% *loss* at 512³, a tie at
1024³ (KC=512, 4 interleaved rounds). Not worth its messier NR=12 edge
handling and near-full register file; 8×8 (20/32 regs) stays and keeps
headroom for future unroll/prefetch work. Block-size sweep harness:
`bench/bench_cpu_sweep.cpp` — calls `cpu::sgemm` directly (compiles in
~1s) with `-DTL_CPU_MC/KC/NC` overrides (the constants in cpu.h are
`#ifndef`-guarded for this).

Remaining for the M5 gate (needs the x86 box): AVX2/AVX-512
microkernels, runtime CPUID dispatch, and the real OpenBLAS-90%
measurement. The on-Mac tuning is done.

## AVX2 + runtime dispatch scaffold (M5, 2026-07-03)

Written and compile-checked on the Mac; **not executed** (Rosetta stops
at SSE4.2 — the x86 box runs and tunes it). The compile-time `#ifdef`
microkernel pick became a runtime function pointer: `select_ukernel()`
returns NEON on ARM (fixed at compile time), and on x86 probes
`__builtin_cpu_supports("avx2"/"fma")` to pick the AVX2 kernel or the
scalar fallback. `sgemm` caches the pointer in a magic-static and calls
through it — the indirect call is amortized over the kc-loop.

The AVX2 8×8 kernel is the x86 analogue of the NEON one: NR=8 = one
`__m256` per row → 8 accumulators (of 16 ymm), each k-step does one
B load broadcast-multiplied by the 8 A values (`_mm256_broadcast_ss` +
`_mm256_fmadd_ps`), K-unrolled ×4 with prefetch. It shares the MR=8/NR=8
packing, so no second pack path. `__attribute__((target("avx2,fma")))`
lets it live in a baseline-x86 TU and be reached only after the CPUID
check — verified by cross-compiling (`-target x86_64-apple-macos -c`)
and disassembling: the body emits `vfmadd213/231ps` + `vbroadcastss`,
confirming the attribute took. Scalar + NEON kernels validated against a
naive triple-loop oracle across full and edge (mr<8, nr<8) tiles.

**Deferred to the x86 box:** executing/numerically-verifying AVX2,
register-tuning the tile (8×8 vs a 6×16 that better fits 16 ymm), and an
AVX-512 kernel — a real one wants NR=16 (a second packing layout), so it
is a genuine x86-side task, not a Mac compile-check. The dispatch seam is
ready: `select_ukernel` just needs an `avx512f` branch.

## Own CPU GEMM x86 census (M5, i7-12700KF, 2026-07-03)

First real execution of the AVX2 path (per docs/x86-validation-runbook.md).
Box: 12th-gen i7-12700KF (Alder Lake, 8 P-core + 4 E-core = 20 logical),
Ubuntu 22.04 under **WSL2**, g++ 11.4, OpenBLAS 0.3.20. AVX2+FMA present,
no AVX-512.

**AVX2 first-run correctness:** `check_cpu_ukernel` — scalar, avx2-8×8, and
the new avx2-6×16 all match the naive oracle across full + edge tiles; the
full array suite (cpu/gpu/auto) is green with the descriptor-driven driver.
This is the first time the AVX2 kernel has executed anywhere (Rosetta stopped
at SSE4.2 on the Mac).

**8×8 → 6×16 register blocking (the tuning win).** The original 8×8 tile uses
only 8 of 16 ymm registers as accumulators; a 6×16 tile (6 rows × 2 ymm = 12
accumulators, the canonical Haswell+ blocking, BLIS/OpenBLAS haswell) raises
in-register reuse. Each kernel now packs to its own tile via `ukernel_desc`
(the "second packing layout" the roadmap anticipated; AVX-512 will reuse the
NR=16 layout). Single-core (`taskset -c 0`, interleaved ×3, quiet), GFLOP/s:

| size | 6×16 | 8×8 | 6×16 gain |
|---|---|---|---|
| 512³ | 131–142 | 114–119 | +17% |
| 1024³ | **140–148** | 116–122 | +18–20% |
| 2048³ | 128–138 | 110–118 | +15% |

Single-P-core AVX2 peak ≈ 4.9 GHz × 2 FMA × 8 fp32 × 2 ≈ **157 GFLOP/s**, so
6×16 at ~143–148 (1024³) is **~91% of single-core peak** — matches/exceeds
the NEON kernel's 86–89%. The 8×8 sat at ~77%. 6×16 is now the x86 default;
8×8 is kept behind `-DTL_CPU_AVX2_8X8` for A/B.

**KC re-sweep on 6×16 (single-core, interleaved ×3):** KC=512 is best/tied at
every size (1024³: 147/141/148; 2048³: 137/136/138); KC=256/384 slightly
lower, 768 ~= 512 at large N but down at 1024³. **KC=512 confirmed for x86**,
same as NEON — kept. MC=128/NC=2048 kept (not exhaustively re-swept; KC is the
dominant cache-blocking param and it held).

**OpenBLAS-90% gate.** Full machine, threads matched (`openblas_set_num_threads`
= pool size), the stable compute-bound point is **2048³: own (6×16) ≈ 106% of
OpenBLAS** (~810–890 vs ~20–23 ms), interleaved — **gate MET**. 8×8 was
~95–103% there. Confined to 8 logical CPUs, own(6×16) 2048³ throughput
(~712 GF/s median) beat own(8×8) (~609) by ~17%, mirroring single-core.

**Caveats (WSL2, recorded per methodology).** (1) WSL2 flattens the hybrid
topology — `lscpu` shows 10×2 with no MAXMHZ, so P/E cores can't be told apart
and `taskset` can't pin to P-cores; the Windows host scheduler places threads.
(2) `std::thread::hardware_concurrency()` ignores the taskset affinity mask
(always reports 20), so the pool always spawns 20 threads. (3) Mid-size
multi-thread (512³/1024³) is noisy and often < OpenBLAS — but single-core
1024³ is 91% of peak, so this is **not** a kernel deficit. It is the thread
pool statically splitting M across all 20 threads regardless of problem size,
over-parallelizing small GEMMs on the P/E+HT topology where OpenBLAS caps
threads. See the deferred "problem-size-aware thread count" item in
roadmap.md — best measured on native Linux, not this WSL2 box.

## Own CUDA GEMM stage-2 (M6, RTX 3090 / WSL2, 2026-07-03)

The SGEMM tuning sprint over the correctness-first one-output-per-thread
kernel. Bench: `bench/bench_cuda_gemm.cpp` links cuBLAS + the CUDA runtime as a
**measurement reference only** (like OpenBLAS on the CPU side — never a library
dependency; cuda.h still dlopen's the driver). Timing is CUDA events on the null
stream, R launches per batch, median of interleaved own-vs-cuBLAS rounds. All
numbers are **device-side event timing on the RTX 3090 under WSL2** — WSL2 adds
submission latency, but events measure device-side, so the own/cuBLAS ratio is
WSL2-insensitive (wall-clock would not be).

### The finding that reframed the sprint: managed memory was the first blocker

The stage-1 backend used `cuMemAllocManaged` (chosen to reuse the Apple unified-
memory seam). Standing up the census immediately exposed an **~88× cliff** —
same cuBLAS SGEMM at 2048³, only the allocator differs:

| allocation | cuBLAS 2048³ |
|---|---|
| `cudaMalloc` (device) | **~22,900 GFLOP/s** |
| `cudaMallocManaged` (managed) | **~260 GFLOP/s** |

Root cause: **`cudaDevAttrConcurrentManagedAccess = 0` on WSL2** — the GPU
cannot fault-migrate managed pages, so it reads them over PCIe every access (the
~260 GF/s ≈ PCIe-bound). No managed escape exists on this box: `cuMemPrefetchAsync`
and every `cudaMemAdvise` variant return *"invalid device ordinal"*, and a
device-only-initialized managed buffer (host never touches it) still runs 256
GF/s — migration is fundamentally off, not a first-touch issue. So the roadmap's
"first lever" (prefetch hints) is a dead end here. Because PyTorch uses
`cudaMalloc` device memory, *beating PyTorch requires device memory too* — no
kernel tuning matters on managed memory. This triggered the roadmap's pre-
authorized **device-buffer pivot**: the CUDA backend now keeps a persistent
host/device **mirror** per allocation (host malloc + `cuMemAlloc` device buffer,
dirty state keyed by the device pointer in cuda.h, so views sharing a storage
share one entry), with lazy H2D before a kernel reads a host-dirty buffer and
D2H before the CPU reads a device-dirty one (`array::raw()/data()` →
`gpu::sync_to_host`). Metal stays a strict no-op (genuinely unified). This
replaced managed everywhere; the array oracle + ctest cpu/gpu/auto validate the
coherence end-to-end. Recorded in [[cuda-wsl2-managed-memory-cliff]].

### SGEMM kernel ladder (interleaved own/cuBLAS, device memory)

With operands on device memory, the naive kernel is the bottleneck. The tuned
`tl_sgemm_rb` fast path (NN, contiguous, K%8==0, N%4==0, 16B-aligned; everything
else — transpose/strides/odd shapes — falls to the correctness-first `tl_sgemm`):

| step | 2048³ own/cuB | 4096³ own/cuB |
|---|---|---|
| naive one-output-per-thread (device mem) | ~0.09 | ~0.09 |
| 128×128×8 register block, 8×8 microtile, float4 loads | 0.68 | 0.75 |
| + warp-tiling (64×32 warp tile, WNITER=2) — conflict-free smem reads | 0.71 | 0.80 |
| **+ register-staged double buffer (current)** | **0.72–0.84** | **0.79–0.83** |

The warp-tiled thread→output map makes every As/Bs fragment read a broadcast
within a warp, removing the 4-way bank conflict the plain `tid/16,tid%16` map
hits (its stride-8 B reads serialize). `ptxas`: 127 regs, no spills, 16KB smem
(double-buffered) → 2 blocks/SM (33% occupancy, register-limited by the 64
accumulators — normal for GEMM). Absolute: ~18,500–19,400 GF/s at 4096³ (cuBLAS
~23,400). Correct throughout — max rel err vs cuBLAS ≤ 6e-5 at 2048³, exact at
4096³; full ctest green including the fast path (check_cuda's 64×48×40 NN case).

**Gate status: not yet met.** Own ≈ **80% of cuBLAS**, and PyTorch-FP32 (TF32
disabled; it bundles cuBLAS 12.8 and measured ~25,900 GF/s at 4096³, *faster*
than the linked cuBLAS 13.1 ~23,400) is the tougher bar at ~0.75. The remaining
levers to close 80%→90%+ (in likely order): larger block tiles (128×256 for more
C-reuse per load — needs register/spill management), split-K for the mid-size
wave-quantization tail (2048³ is only ~2 waves at 128² tiles — the ratio dips vs
4096³'s ~6 waves), and finer instruction scheduling. Per the roadmap escape
hatch, a per-shape-band CUTLASS instance is the fallback if a band can't reach
the gate after that.

### Split-K census (ladder ②, 2026-07-04)

The census signature — own/cuB rising monotonically with size (1024³ 0.63,
2048³ 0.74, 4096³ 0.82) — is the classic occupancy/wave-quantization tell: 128²
tiles make ceil(m/128)·ceil(n/128) blocks, and the RTX 3090's 82 SMs hold 2
blocks each = 164 concurrent. So 1024³ = 64 blocks fills only ~39% of the SMs
(0.39 wave), 2048³ = 256 blocks = 1.56 waves (half-empty tail), 4096³ = 1024
blocks = 6.24 waves (tail negligible → the 0.82 there is pure kernel efficiency,
not a fill problem). Split-K partitions the K axis into S z-slices (gridDim.z=S)
so S× more blocks run at once; crucially it *partitions* K rather than
replicating it, so A/B global traffic is unchanged — the only added cost is C
written S× via `atomicAdd` into a pre-zeroed buffer (hence gated to identity
scale/offset; fused-affine GEMM keeps the single-slice store). Forced-S census
(interleaved own/cuB, maxrel ≤1.2e-4 — fp32 atomic-order noise, acceptable):

| size | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|
| 1024³ | 0.63 | **0.75** | 0.75 | 0.73 |
| 2048³ | 0.74 | **0.76** | 0.76 | 0.68 |
| 4096³ | **0.82** | 0.82 | 0.79 | 0.75 |

**S=2 is the robust optimum** for the underfilled mid sizes (1024³ +19%, 2048³
+3%). S≥4 regresses: the S× C-atomic traffic + the shorter per-block K (fixed
smem-pipeline fill/drain amortized over fewer slabs) overtake the occupancy gain;
by S=8 both mid sizes drop below their S=1. 4096³ is already 6 waves so any split
is pure overhead → S=1 (and S=1 keeps C exact, maxrel 0.0). Kernel change is a
single `ksplit` param + `blockIdx.z` K-range + a `gridDim.z>1 ? atomicAdd : store`
epilogue on the *same* `tl_sgemm_rb` body (no copy-paste divergence; still 127
regs / 2 blocks/SM). Auto policy: S=2 when base<512 blocks and K≥512 (each half
≥256 K, enough to amortize the pipeline), else S=1; `TL_SPLITK` forces S for the
census. Post-split state: 1024³ ~0.72–0.82, 2048³ 0.76, 4096³ 0.82–0.83.

**Gate still open** at 2048³/4096³ (~0.76/0.82). Split-K fixed the small-size
*fill* deficit but the per-SM *efficiency* ceiling (~0.82, visible at 4096³ where
fill is a non-issue) is untouched — that is the remaining ~10-18% and needs the
kernel-efficiency lever (ladder ③: Ampere `cp.async` staged pipeline — removes
the register-staged global loads, frees registers, enables >2-stage overlap),
then the per-band CUTLASS escape hatch if it still misses.

**Refuted (do not retry without new evidence):**

| Approach | Verdict | Data |
|---|---|---|
| BK 8→16 (BK=16 slab, 2×float4 loads) | **Rejected** | Helps 1024³ (0.62→0.70) but *regresses* the gate sizes: 2048³ 0.72→0.62, 4096³ 0.83→0.73. The 32KB smem + 130 regs hurt where large-matmul throughput matters. |
| Register double-buffer *before* warp-tiling | **No effect** | On the conflict-bound (pre-warptile) kernel, hiding global latency did nothing (2048³ 0.68→0.67) — the kernel was smem-bound, not latency-bound. Double buffering only paid off *after* warp-tiling made the smem reads conflict-free. |
| **128×256 large tile (ladder ①)** — 256-thread block, 8×16 microtile, WNITER=4 (`tl_sgemm_rb2`), +33% arithmetic intensity (64→85) | **Rejected** | Loses at *every* size (interleaved own/cuB): 1024³ 0.69→0.41, 2048³ 0.75→0.68, 4096³ 0.81→0.75. 128 accumulators/thread → 203 regs (no spills) but **1 block/SM = 16.7% occupancy** (rb keeps 2 blocks/SM = 33%). It loses even at 4096³ (6 waves, machine full, wave-quant irrelevant), so the arithmetic-intensity gain (−25% DRAM traffic) is dwarfed by the occupancy/latency-hiding loss — same class as the BK=16 rejection. 1024³ collapses because 128×256 halves the block count (64→32) so only ~39% of the 82 SMs get work. Big spatial tiles are ceiling-capped at rb on sm_86 for this kernel: any 128×256 variant is ≥1 block/SM. Pivoted to split-K (below) for the small-size deficit that census confirms is the real problem. |
| **cp.async staged pipeline (ladder ③)** — Ampere `cp.async` global→shared, 3/4-stage smem pipeline (`tl_sgemm_cp`), replacing the register-staged loads | **Rejected** | Loses where it was meant to win: 4096³ 0.83→0.72–0.74, 2048³ 0.75–0.80→0.62–0.69 (interleaved vs rb S=1); ties at 1024³. Correct throughout (maxrel = rb, 4096³ exact). Root cause is a **structural mismatch**: A must land *transposed* in smem (As[k][m]) for the regM fragment read to stay a contiguous float4, but cp.async copies a contiguous global chunk to a contiguous smem chunk — it can't transpose. So A is forced to 4×**4-byte** cp.async scatter (one per k of the thread's float4), and 4-byte cp.async is the inefficient granularity (16-byte `.cg` is the fast path); that penalty + the extra `__syncthreads` (2/slab vs rb's 1) sink it. 4 stages don't rescue it (4096³ still 0.73). Efficient cp.async for transposed-A SIMT needs `ldmatrix` (tensor-core-only) — out of scope for the FP32 SIMT kernel. rb's float4-ldg-then-scatter stays the load path; regs dropped 127→107 with cp.async but occupancy stayed 2 blocks/SM (64 accumulators, not loads, are the register floor), so even the freed registers bought nothing. |

## M7–M9 decode kernels (RTX 3090 / WSL2, 2026-07-04)

The consumer-local-LLM decode path (batch≈1, bandwidth-bound). Metric is
tokens/sec on llama-7B decoder shapes (d=4096, ffn=11008, 32 heads × head_dim
128, 32 layers, vocab 32000), measured via `bench_llm`; kernel-level numbers via
the direct `bench_bf16_gemv` / `bench_attn_decode` / `bench_q4_gemv` benches
(cuda:: API, cuda::flush + steady_clock, no cudart). RTX 3090 memory-bandwidth
peak ≈ 936 GB/s — the ceiling that matters here, not GFLOP/s.

**Whole-decode tok/s arc (F32 baseline → M7+M9):**

| stage | tok/s | lever |
|---|---|---|
| tile-GEMM (session start) | 3.5 | M=1 dot ran the 128² tile → 127/128 rows wasted |
| + GEMV dispatch (F32) | 12.2 | route M=1 dots (incl. attention's) to `tl_gemv_f32` |
| + bf16 weights | 14.8 | `tl_gemv_bf16v8`, halved weight bytes |
| + fused attention (F32) | 26.1 | `attn_decode` — attention was 72% of token time |
| + bf16 weights + fused attn | **41.2** | both |

**bf16 decode GEMV** (`bench_bf16_gemv`, layer weight shapes): f32 kernels run
at 850–935 GB/s (≈ peak — bandwidth-bound). Scalar 2-byte bf16 loads under-use
that (layer sum 1.62× vs f32); `tl_gemv_bf16v8` (8 cols/thread, one 16-byte uint4
load, `n%8==0`) closes it to **1.84×** (ffn_gate_up/lm_head at the 2× ceiling,
~890 GB/s). Small-N shapes (N=4096) lag — split-K atomic + launch overhead is a
bigger fraction of their tiny work.

**Fused decode attention** (`bench_attn_decode`, H=32, ctx=2048): the unfused
3-op array path (q·Kᵀ, softmax, ·V, per head × layer ≈ 1024×) is
launch/materialize-bound, ~48 ms/model. `tl_attn_decode_f32` (one block/head,
online softmax) → 11.7 ms (4×) but only ~180 GB/s: grid = H = 32 blocks fills 3%
of the 82 SMs, occupancy-bound. **Split-KV** (partition ctx over `gridDim.y`,
partial states → combine pass) → **2.54 ms, 844 GB/s (~19×)**, bandwidth-bound.
Numerically exact throughout (maxrel ~1e-7). Array-integrated at 0.110 ms/layer
(13.5× the unfused path).

**int4 dequant GEMV** (`bench_q4_gemv`, group=32 symmetric int4, [N,K] layout):
correct vs a host dequant reference (maxrel ~1e-5), 0.625 bytes/weight (0.5
packed + 4/32 scale) vs bf16's 2.0. **1.79× vs bf16** (layer sum 0.436 vs 0.78
ms). *Not yet bandwidth-bound* (best ~565/936 GB/s): the global-a kernel re-reads
the activation from L2 per output warp; shared-a staging (`tl_gemv_q4s`, gated to
K·4 ≤ 24 KB — larger K collapses occupancy) helps but the strided a_sh reads
bank-conflict. Not yet an array op.

*Refuted (2026-07-04): interleaved (exllama-style) repack for conflict-free
reads.* Assigning lane l the stride-32 keys k = base+l+m·32 (packed one word/lane)
makes both the a and qweight reads coalesced/conflict-free — but measured
**slower**, not faster: shared-a 488 vs the conflicted 565 GB/s, global-a 438 vs
shared 565. So the bank conflict was *not* the binding constraint (a's L2 re-read
per output warp, which shared-a addresses, dominates; the interleaved layout also
reads 8 scales/word vs 1). Reverted. Reaching bandwidth-bound needs a different
lever (cut a-traffic further / dp4a int8 path), found by profiling, not the
obvious coalescing fix — a reminder to measure before assuming.

**Progression pattern (all three kernels).** Each landed correct-but-un-tuned,
then a second pass to bandwidth-bound: bf16 1.62→1.84×, attention 4→19×, int4
1.24→1.79× (mid-tune). Occupancy (blocks vs 82 SMs) was the recurring first
bottleneck — split-K/split-KV the recurring fix.

## vs silarray (M1 Pro, 2026-07-03)

Head-to-head with the predecessor across cpu/gpu/auto. Two separate
binaries (header coexistence blocked by objc/Metal bridge collisions),
run alternating with per-process warmup, steady-state min of 3 rounds,
load ~3. Both use blocking eval (silarray's free `sil::eval()`, not the
non-flushing member `.eval()`). `tl/sil` < 1 means tensorlib faster.

| Workload | cpu | gpu | auto |
|---|---|---|---|
| SGEMM 1024³ | 1.01 | **0.76** | 0.92 |
| SGEMM 256³ | 1.00 | 0.98 | **0.10** |
| elementwise 4M | 0.90 | 1.04 | 0.95 |
| softmax 1024² | **0.03** | 1.42 | **0.003** |
| MLP fwd 256×784×256×10 | **0.38** | 1.67 | 0.55 |

Reading:
- **CPU**: tensorlib parity-to-faster. matmul parity (shared AMX). The
  huge softmax/MLP wins are because **silarray's CPU softmax is a naive
  correctness-only fallback** (its docs say so — silarray is GPU-focused)
  — discount these as "sil didn't implement it," not an engineering edge.
- **GPU**: split. tensorlib **wins matmul** (STEEL port, 1024³ 24%
  faster); silarray **wins softmax (1.4×) and the fused MLP (1.7×)** — its
  online-softmax and `linear_sigmoid` GPU kernels are more mature. This is
  the motivation for the deferred "GPU fused kernels" item.
- **auto**: tensorlib clearly better — the finalized thresholds route
  correctly (256³→cpu 10×, softmax→gpu 300×) where silarray's auto
  mis-picks (256³→slow GPU, softmax→slow CPU). auto quality is the payoff
  of the measured thresholds.

Caveats: single-op micro-bench, cross-process (looser than same-run).

## Refuted approaches — do not retry without new evidence

| Approach | Verdict | Data |
|---|---|---|
| Naive 64×64 tile via the simple `sgemm_body_` template (full `simdgroup_matrix` locals, 16 accumulators) | **Catastrophic** | 133 GFLOP/s at 2048³ vs 2175 for 32×32 — the "16-accumulator occupancy crisis" silarray hit; STEEL's explicit float2 fragment registers are what make 16 accumulators viable |
| Host-side alignment gating of float4 loaders (require ld%4==0, 16B base) | **Unnecessary** | silarray/MLX issue vector loads at any alignment on Apple GPUs in practice (bench shapes with ldb=50); gate removed |

Inherited from silarray (Metal; re-verify before assuming they transfer to
CUDA): spin-wait on command-buffer status (slower), split commits to overlap
CPU/GPU (slower), trace/replay compile (no payoff — graph build was 0.2% of
step time), AOT vs JIT kernel compilation (no effect on SGEMM).

## Bug classes worth remembering

- silarray's only two real correctness bugs were the same class: copy-pasted
  kernel bodies diverging between aligned and edge tile paths. Rule: kernel
  bodies are shared templates; every fused-epilogue kernel gets edge-shape
  numerical tests (M-edge, N-edge, band boundaries).
