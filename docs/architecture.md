# cpp-tensorlib architecture

Cross-platform F32 tensor library. Informed by silarray (the macOS-only
experiment this project succeeds) but designed fresh for three platforms; the
goal is a simple, elegant implementation that outperforms PyTorch on both CPU
and GPU. Serves as the matrix foundation for culebra's Tensor type (culebra
keeps autograd/VJP; this library owns the graph, fusion, and execution).

## Target layer map

```
 user code / culebra   tl::array, tl::eval(), free functions
                            │
 array.h               lazy graph + peephole fusion (M2)     ← today: eager
                            │            │
                       device dispatch (per-kernel-class thresholds, M3+)
                            │
        ┌───────────────────┼──────────────────────┐
 macOS: Accelerate+Metal │ own CPU microkernels │ own CUDA kernels
 (M3)                    │ BLIS-style, AVX2/512/ │ PTX #embed, driver API
                         │ NEON + threadpool(M5) │ via dlopen (M6)
                            │
 reference backend      detail:: strided loops in array.h — the correctness
 (M1, permanent)        oracle and universal fallback
```

Forward-looking plan (milestones, environment constraints, open decisions)
lives in [roadmap.md](roadmap.md). This file documents how the code works
today; performance methodology and gate results are in
[performance-notes.md](performance-notes.md).

## Current state (M7–M9: dtype surface + decode kernels)

The consumer-local-LLM decode path (batch≈1) is the focus (see roadmap "Target
scope"). Decode is memory-bandwidth-bound, so the levers are cutting weight
bytes (bf16/int4 storage) and fusing attention — not F32 GFLOP/s.

- **Storage dtype (M7).** `types.h` has `dtype{f32,bf16}` + RNE converters;
  `storage`/`array` carry a dtype and byte-sized allocation. bf16 is a
  *weight-container* type: `array::to_bf16()`/`to_f32()`; the CUDA decode GEMV
  consumes bf16 natively, and every other op transparently widens bf16 inputs to
  an F32 copy in `eval_one`'s input funnel — so ref/cpu/accel/Metal kernels stay
  F32-only, zero changes. Compute and results are always F32.
- **Decode GEMV (M7).** `tl_gemv_f32` / `tl_gemv_bf16v8` (8 cols/thread, 16-byte
  loads, split-K for small N). `eval_one`'s dot case routes M=1 to the GEMV
  (the 128×128 tile wastes 127 rows at M=1) — this alone tripled the F32 decode
  baseline; bf16 weights add ~1.84× on the bandwidth-bound GEMV.
- **Fused attention (M9).** `array::attn_decode(q,K,V,scale)` (op_t::attn_dec)
  → `tl_attn_decode_*`: flash-attention online-softmax in one pass (scores never
  materialized), split-KV across `gridDim.y` to fill the SMs (KV-bandwidth-bound,
  ~844 GB/s). CPU reference fallback; Metal returns false (widen/CPU path).
- **int4 storage dtype (M8).** `dtype::q4`: group-symmetric int4 weights, one
  buffer holding packed [N,K] int4 + appended f32 scales, logical shape [K,N] so
  `a.dot(Wq)` type-checks like f32 and rides the bf16 widen-fallback seam (decode
  → `tl_gemv_q4`, else dequant to F32). `array::to_q4()`/`to_f32()`. Correct;
  1.79× vs bf16 on the GEMV (not yet bandwidth-bound).
- **CUDA buffer pool.** `gpu::alloc`/`release` recycle device buffers via a
  size-keyed free list (like Metal's), so alloc/free-heavy workloads (training)
  don't fragment the driver allocator (real-MNIST gate 7.8 → 5.8 s).
- **Result (RTX 3090, llama-7B decode shapes):** 3.5 → 61.7 tok/s over M7+M9+M8
  (F32 12.2, +fused-attn 26.1, +bf16 42.5, +int4 61.7). Full census in
  performance-notes.md. Dev harness: `bench_llm` (f32/bf16 + attn; kept short so
  WSL2 timing stays clean), plus direct kernel benches `bench_bf16_gemv` /
  `bench_attn_decode` / `bench_q4_gemv` (each flushes — the authoritative
  per-kernel numbers).

## Current state (M5 first cut: own CPU GEMM)

- `cpu.h` + `cpu_threadpool.h` — BLIS-style SGEMM for platforms without
  Accelerate: cache-blocking (MC/KC/NC) + stride-aware packing (transposed
  views feed in place, no materialization) around a register-blocked
  microkernel, parallelized over the M dimension by a persistent thread
  pool. Microkernel: NEON (8×8 tile, native on Apple Silicon / ARM Linux)
  and a portable scalar fallback; AVX2/AVX-512 are the next drop-in behind
  the same interface. Raw-pointer API (no array.h dependency, like metal.h).
- Dispatch (`eval_one` dot): `metal → accel::gemm → cpu::sgemm → ref::dot`.
  `cpu::enabled_` gates it (default on; below Accelerate on Apple, primary
  off-Apple; oracle tests force ref with it + `use_accelerate_` off).
- **Status: correct and complete, untuned.** Matches ref across edge/
  transposed/epilogue shapes on all modes. Scales to ~235 GFLOP/s at 1024³
  on M1 Pro NEON (vs Accelerate's AMX ~8× faster — expected; the own kernel
  is the *off-Apple* path). Per-call overhead (pack-buffer allocation,
  thread-pool sync even for one block) loses to the naive loop below ~256³;
  removing it + the OpenBLAS-90% gate + AVX are the tuning pass (roadmap).

## Earlier state (M4: culebra integration + tiny-tensor sprint)

- **culebra integration** (M4): `culebra::TensorImpl` is an autograd tape
  node wrapping a `tl::array`; culebra keeps VJPs, this library owns the
  graph/fusion/execution. Vendored into culebra as a git submodule.
  Feature-gated via `TL_RUNTIME_HOOKS`: an embedder's tensor-free binaries
  reference no backend symbol (allocation/eval/barrier route through hooks
  installed by `install_runtime_hooks()`).
- **Tiny-tensor sprint**: contiguous flat-loop fast paths (map/clone/add_),
  eager-tiny evaluation in the graph builders (materialized ≤4096-elem
  operands skip the node/eval machinery), thread-local topo-sort scratch
  with visit-stamp marking, memoized const-wrap, tiny-dot/tiny-ew direct
  paths in `eval_one`, pooled MTLBuffer contents caching. Per-op overhead
  cut 2–3×; MNIST training 2× faster than the pre-M4 executor.
- **auto thresholds** finalized from a quiet-machine census (matmul 2e9,
  elementwise 2e6, reduction 2e5 on M1 Pro) — see performance-notes.md.

## Earlier state (M3b-3: STEEL SGEMM sprint)

- STEEL kernels (`sgemm_steel_` 64×64, `sgemm_steel_32x64_` for M < 97)
  ported from silarray/MLX: float2 fragment registers, precomputed loader
  pointers, serpentine MMA, threadgroup swizzle, split edge loops
  (interior / M-edge / N-edge / corner), affine-epilogue store shared
  between interior and edge paths. NN operands only for now.
- Dispatch ladder in `metal::gemm`: STEEL band (NN, m ≥ 16, n ≥ 48,
  k ≥ 16, BM by M band) → simple-tile family (64×32 / 32×32; handles
  transposed views in place via strided loaders + float4 fast path).
- 2048³ sgemm: ~1300 → **~3200 GFLOP/s** over the sprint (loaded machine;
  see performance-notes.md for the step-by-step history and census).

## Earlier state (M3b-2: Metal SGEMM, softmax, reductions)

- `metal_kernels.metal` adds: `sgemm_` (32×32×16 simdgroup-matrix tile,
  trans_a/trans_b in-place loaders, affine epilogue fused in the store,
  one edge-safe store path), `softmax_` (row-per-threadgroup, stable),
  `row_sum_`/`row_max_` (last-axis reductions, epilogue applied).
- `eval_one` dispatch: dot → `metal_gemm` (maps transposed views to
  trans flags, same classify as accel) → accel → ref; softmax and last-axis
  sum/max → `metal_row` → ref. Other reduction axes stay on CPU.
- Full MLP forward (gemm→sigmoid→gemm→softmax) runs GPU end to end,
  verified against the oracle across tile-boundary and transposed shapes.
- **Perf status**: correct and complete, not yet tuned — see
  performance-notes.md "Known gap". The SGEMM optimization sprint (STEEL-class
  kernel vs the PyTorch gate) is a later dedicated pass.

## Earlier state (M3b stage 1: Metal elementwise + command-buffer batching)

- `objc.h` — minimal objc_msgSend bridge (header-only, no .mm files).
- `metal.h` — device/queue/PSO context (lazy singleton), size-keyed
  MTLBuffer pool, `#embed`'d MSL JIT-compiled on first dispatch, one
  long-lived command buffer: dispatches accumulate, `flush()` = end +
  commit + waitUntilCompleted at the end of every `graph::run`.
  Non-Apple builds get inline stubs — callers have no `#ifdef`s.
- `storage.h` — when a Metal device exists every buffer is a pooled
  shared-mode MTLBuffer (unified memory); heap otherwise.
- **CPU/GPU handoff**: every CPU-side read funnels through
  `array::raw()`/`data()`, which call `metal::cpu_barrier()` (flush if
  pending). One choke point makes arbitrarily mixed graphs safe — tested
  gpu→cpu and cpu→gpu both ways.
- Kernels: elementwise binary/unary/affine, all applying the graph's affine
  epilogue in the store (`fma(op, scale, offset)`), kernel bodies shared via
  macro (silarray edge-tile lesson). Metal gemm/softmax/reductions are
  stage 2.
- Dispatch order in `eval_one`: `metal (gpu mode) → accel → ref`.
  `auto` joins a pending GPU pipeline but never starts one — measured
  thresholds land with bench data.

## Earlier state (M3a: Accelerate backend + dispatch seam)

- `accel::` (in array.h; graduates to its own header with Metal) — vDSP /
  vForce / CBLAS fast paths tried from `graph::eval_one`, `ref::` as
  fallback. Every accel function returns nullopt/false when ineligible or on
  non-Apple builds, so the evaluator carries no platform conditionals — this
  is the dispatch seam Metal / CPU microkernels / CUDA plug into.
- GEMM maps transposed views onto CblasTrans (no materialization) and takes
  the fused epilogue scale as alpha. `tl::use_accelerate_ = false` forces
  the oracle; tests compare both paths on every dispatchable op class
  including edge shapes.
- `bench/bench_main.cpp` — interleaved A/B medians (accel vs ref).

## Earlier state (M2)

- `types.h` — device mode switch (`use_cpu`/`use_gpu`/`use_auto`),
  `gpu_available()`.
- `storage.h` — shared flat F32 buffer; the seam where device residency and
  pooled recycling attach later.
- `array.h` — `tl::array`: shapes/strides, zero-copy views (transpose,
  reshape, slice), numpy broadcast rules, lazy graph + fusion + evaluation:
  - **Graph**: ops build `detail::node`s; output shapes are computed (and
    shape errors thrown) at build time. `tl::eval(...)` / any data access
    runs one iterative topo-sort pass over all roots.
  - **Fusion**: every node carries an affine epilogue
    (`result = op(...) * scale + offset`). Scalar chains compose into the
    producing node — including binaries, `dot` and reductions — by copying
    it with a composed epilogue. Fusion never mutates an existing node, so
    shared intermediates stay valid (tested).
  - **`ref::` backend**: naive strided kernels over materialized arrays.
    Everything reduces to strides via `detail::for_each_index`, so one loop
    shape serves every op/view/broadcast. The oracle and fallback; real
    backends replace it per-op in `detail::graph::eval_one`.

## Milestones

Status summary; scope, environment needs, and approach are in
[roadmap.md](roadmap.md).

| | Scope | Status |
|---|---|---|
| M1 | Core skeleton + reference CPU implementation + tests + CI | ✅ |
| M2 | Lazy graph, topo-sort eval, build-time peephole fusion | ✅ |
| M3a | Accelerate CPU backend + dispatch seam + bench harness | ✅ |
| M3b-1 | Metal foundation: context, buffer pool, elementwise, batching | ✅ |
| M3b-2 | Metal SGEMM (tiled), softmax, last-axis reductions; GPU MLP fwd | ✅ |
| M3b-3 | STEEL SGEMM sprint: 2048³ ~1300 → ~3200 GFLOP/s | ✅ |
| M4 | culebra integration; tiny-tensor per-op sprint; auto thresholds | ✅ |
| M5 | Own CPU backend: threadpool + BLIS-style microkernels (AVX2/AVX-512/NEON) | 🔨 NEON + AVX2 done (OpenBLAS gate met); AVX-512 deferred (needs hw) |
| M6 | Own CUDA backend: dlopen'd driver API, PTX `#embed`, SGEMM + split-K | 🔨 correct; ~0.82 of cuBLAS (prefill foundation — see roadmap "Target scope") |
| M7 | BF16/FP16 **storage** + decode GEMV dispatch | 🔨 storage dtype + bf16 GEMV done & integrated; bf16 **compute** GEMM (fine-tuning) deferred |
| M8 | Quantized (int4/int8) dequant-fused matmul (local-LLM inference) | 🔨 int4 GEMV done & integrated as a storage dtype (decode 61.7 tok/s); bandwidth-bound tuning + int8/GGUF-format remain |
| M9 | Fused attention (flash-attn style) + KV cache | 🔨 fused decode attention done & integrated (split-KV, ~19×); KV cache / GQA / causal mask remain |

Milestones are now driven by the **target scope** (consumer local-LLM
inference/fine-tuning on 1–few consumer GPUs — RTX 5090 / A6000); see
`roadmap.md`. The CUDA bar is tokens/sec vs llama.cpp/exllamav2, not cuBLAS
GFLOP/s — full rationale in `performance-notes.md`.

## Conventions

- snake_case types/functions; trailing `_` for private members and internal
  helpers; `k`-prefixed compile-time constants.
- Headers follow the cpp-httplib layout: declarations first, bodies below the
  `// Implementation` divider.
- Comments state constraints and measured rationale, not narration.
- Kernel bodies must be shared templates across aligned/edge tile paths
  (silarray bug-class lesson); every fused-epilogue kernel gets edge-shape
  numerical tests.

## Dependency policy

culebra links everything statically (OS-guaranteed dynamic libraries are the
only exception), so this library carries **zero third-party dependencies**
(doctest is vendored, tests only):

- macOS: Accelerate/Metal/MPS frameworks (OS-provided).
- CPU: own microkernels; no OpenBLAS (escape hatch: `-DUSE_OPENBLAS` may be
  added later if a shape class demands it).
- CUDA: own kernels; `libcuda`/`nvcuda.dll` is dlopen'd at runtime (on WSL2:
  `/usr/lib/wsl/lib/libcuda.so`), so binaries run without a driver and fall
  back to CPU. No cudart, no cuBLAS, no CUTLASS (escape hatch: per-shape-band
  CUTLASS instances if measurement demands).

Size budget: < 1MB added per platform (< 2MB with CUDA kernels). AOT culebra
binaries that don't use Tensor must contain none of this — no unconditional
global initializers; lazy singletons only.
