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

## Current state (M2)

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

| | Scope |
|---|---|
| M1 ✅ | Core skeleton + reference CPU implementation + tests + CI |
| M2 ✅ | Lazy graph, topo-sort eval, build-time peephole fusion |
| M3 | macOS backend (Accelerate + Metal, `#embed` MSL) + bench harness |
| M4 | culebra integration (TensorImpl wraps tl::array; F32 unification) |
| M5 | Own CPU backend: threadpool + BLIS-style microkernels (AVX2/AVX-512/NEON, runtime dispatch) |
| M6 | Own CUDA backend: dlopen'd driver API, PTX `#embed`, SGEMM ladder |
| M7 | BF16 storage type; ongoing measured optimization |

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
