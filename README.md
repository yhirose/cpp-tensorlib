cpp-tensorlib
=============

Cross-platform F32 tensor library — the matrix foundation for
[culebra](https://github.com/yhirose/culebra)'s Tensor type.

* Header-only C++23 — `#include <tensorlib.h>`
* MLX-style **lazy evaluation**: ops build a graph; `tl::eval()` evaluates
  multiple arrays in one topological pass, with build-time peephole fusion
  (every node carries an affine epilogue, so scalar chains and
  dot-then-scale collapse to a single dispatch).
* Switchable backend via `tl::use_cpu()` / `tl::use_gpu()` / `tl::use_auto()`
  (auto picks CPU or GPU per op from measured per-kernel-class thresholds).
* Zero-copy views (transpose / reshape / slice), numpy broadcast rules.
* Data type: `float` (BF16 storage type planned; see the milestones).

> **Status: work in progress.** The macOS backend (Accelerate + Metal) is
> complete and fast; the reference implementation runs everywhere as an
> oracle and fallback. The own-CPU (Linux/Windows SIMD) and CUDA backends
> are not yet implemented — see the milestones below.

Design informed by [silarray](https://github.com/yhirose/silarray) (the
macOS-only experiment this succeeds), rebuilt for three platforms with a
single dispatch seam the backends plug into.

Example
-------

```cpp
#include <tensorlib.h>

auto a = tl::array::ones({1000, 1000});
auto b = tl::array::ones({1000, 1000});

auto c = a.dot(b) * 0.5f + 1.0f;   // lazy: one fused GEMM+epilogue dispatch
auto d = (a + b).relu();           // also lazy

tl::eval(c, d);                    // evaluate both in one pass

tl::use_gpu();                     // route to Metal (macOS)
auto e = a.dot(b);
tl::use_auto();                    // CPU or GPU per op, by measured size
```

Backends
--------

| Platform | CPU | GPU |
|----------|-----|-----|
| macOS | Accelerate (vDSP / vForce / CBLAS) ✅ | Metal / MSL (`#embed` JIT), STEEL SGEMM ✅ |
| Linux / Windows | own BLIS-style microkernels (planned, M5) | own CUDA kernels, dlopen'd driver API (planned, M6) |
| any | reference strided implementation (oracle + fallback) ✅ | — |

Everything reduces to strides through one index walker, so views, broadcast
and transposed operands share a single code path. Accelerated backends
replace it per-op at the `graph::eval_one` dispatch seam; the seam carries
no platform `#ifdef`s (non-Apple builds get inline stubs).

**Dependency policy:** zero third-party dependencies (doctest is vendored,
tests only). macOS links only OS frameworks. The planned CPU/CUDA backends
use own kernels — no OpenBLAS, cuBLAS or CUTLASS; the CUDA driver is
`dlopen`'d so binaries run (and fall back to CPU) without it.

Build and test
--------------

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # cpu / gpu / auto modes
./build/tensorlib_bench                       # micro-benchmarks
```

Requires a C++23 compiler (`#embed` support: Clang 19+ / GCC 15+). The GPU
backend needs macOS with Metal; elsewhere the tests exercise the CPU path
and the GPU-mode fallback.

Documentation
-------------

* [docs/architecture.md](docs/architecture.md) — layer map, dispatch seam,
  current state, conventions.
* [docs/roadmap.md](docs/roadmap.md) — milestone scope + status, environment
  constraints, per-milestone approach, open decisions.
* [docs/performance-notes.md](docs/performance-notes.md) — measurement
  methodology, gate results, the silarray comparison, and refuted approaches
  (read before any performance work).

License
-------

MIT license (c) 2026 yhirose
