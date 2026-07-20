cpp-tensorlib
=============

[![](https://github.com/yhirose/cpp-tensorlib/workflows/CI/badge.svg)](https://github.com/yhirose/cpp-tensorlib/actions)

A C++17 **header-only, cross-platform** tensor library with **zero
third-party dependencies** — CPU (own SIMD kernels) and GPU (own Metal /
CUDA / WGSL kernels) on macOS, Linux, Windows and the browser, from a single
`#include <tensorlib.h>`. No BLAS, no cuBLAS/cuDNN, no CUTLASS: every accelerated
kernel — GEMM, GEMV, attention, quantized (int4) matmul — is hand-written
in this repo, so the only thing a consumer links against is the OS itself
(and the CUDA *driver*, which is `dlopen`'d at runtime, not linked).

As a full end-to-end proof of that promise, the library also drives a real
local-LLM chat pipeline — GGUF loader, BPE tokenizer, and a Qwen2.5 decoder,
all on the same own-kernel, zero-dependency foundation (see
[LLM inference](#llm-inference) below).

* **Header-only, C++17 minimum** (builds cleanly through C++23) —
  `#include <tensorlib.h>`, nothing to link or build separately.
* **Cross-platform**: macOS (Accelerate + Metal), Linux/Windows (own
  BLIS-style CPU microkernels — scalar/AVX2/NEON — and own CUDA kernels),
  and WebAssembly (own WGSL kernels over WebGPU). One dispatch seam, no
  platform `#ifdef`s in consumer code.
* **Zero third-party dependencies**: no OpenBLAS, cuBLAS, cuDNN, or
  CUTLASS anywhere in the accelerated paths (see
  [Dependency policy](#dependency-policy)).
* MLX-style **lazy evaluation**: ops build a graph; `tl::eval()` evaluates
  multiple arrays in one topological pass, with build-time peephole fusion
  (every node carries an affine epilogue, so scalar chains and
  dot-then-scale collapse to a single dispatch).
* Switchable backend via `tl::use_cpu()` / `tl::use_gpu()` / `tl::use_auto()`
  (auto picks CPU or GPU per op from measured per-kernel-class thresholds).
* Zero-copy views (transpose / reshape / slice), numpy broadcast rules.
* Data types: `float` (primary), plus `bf16`/`int4` storage dtypes for the
  bandwidth-bound LLM decode path (see the milestones).

Design informed by [silarray](https://github.com/yhirose/silarray) (the
macOS-only experiment this succeeds), rebuilt for three platforms with a
single dispatch seam the backends plug into. Serves as the matrix
foundation for [culebra](https://github.com/yhirose/culebra)'s Tensor type
(culebra keeps autograd/VJP; this library owns the graph, fusion, and
execution).

Example
-------

```cpp
#include <tensorlib.h>

auto a = tl::array::ones({1000, 1000});
auto b = tl::array::ones({1000, 1000});

auto c = a.dot(b) * 0.5f + 1.0f;   // lazy: one fused GEMM+epilogue dispatch
auto d = (a + b).relu();           // also lazy

tl::eval(c, d);                    // evaluate both in one pass

tl::use_gpu();                     // Metal (macOS), CUDA (Linux/Windows), WebGPU (wasm)
auto e = a.dot(b);
tl::use_auto();                    // CPU or GPU per op, by measured size
```

Backends
--------

| Platform | CPU | GPU |
|----------|-----|-----|
| macOS | Accelerate (vDSP / vForce / CBLAS) ✅ | Metal / MSL (`#embed` JIT), STEEL SGEMM ✅ |
| Linux / Windows | own BLIS-style microkernels (scalar / AVX2 / NEON, runtime-dispatched) ✅ | own CUDA kernels, PTX JIT'd by a `dlopen`'d driver ✅ |
| WebAssembly | own scalar microkernels ✅ | own WGSL kernels over WebGPU ✅ |
| any | reference strided implementation (oracle + fallback) ✅ | — |

Everything reduces to strides through one index walker, so views, broadcast
and transposed operands share a single code path. Accelerated backends
replace it per-op at the `graph::eval_one` dispatch seam; the seam carries
no platform `#ifdef`s (non-Apple builds get inline stubs).

All three GPU backends cover the array surface — GEMM, elementwise
unary/binary, rank-2 broadcast, row reductions (softmax / row sum / row max)
— each with the same fused affine epilogue, and each validated against the
reference implementation as an oracle. The LLM decode kernels (GEMV,
attention, RoPE, int4 dequant-matmul) are CUDA-only; on Metal and WebGPU
those ops decline at the seam and run on the CPU.

**The WebGPU backend** is built with `emcc --use-port=emdawnwebgpu
-DTENSORLIB_WEBGPU`, and differs from the native two in two ways worth
knowing before you target it:

* **Chrome in practice.** It needs JSPI to keep `flush()` and `sync_to_host()`
  synchronous. Firefox has it behind a flag and Safari has not shipped it;
  there, device acquisition fails, `available()` stays false, and every op
  routes to CPU — correct, just not accelerated. That is also the fallback if
  the page hands in no device.
* **No STEEL-equivalent tiling.** The auto-mode thresholds are calibrated
  against measurements, but the WGSL SGEMM is a straightforward tiled kernel;
  macOS gets a considerably more tuned one.

CI runs the wasm suite on every push — `test/wasm/` builds the same oracle
suite as the native backends and runs it headless via Deno (a browser is not
required, locally either), asserting a per-kernel dispatch census so a silent
all-CPU fallback fails the build rather than passing green.

Dependency policy
------------------

**Zero third-party dependencies, on every platform.** No OpenBLAS, no
cuBLAS/cuDNN, no CUTLASS — every accelerated kernel (CPU SIMD, CUDA GEMM/
GEMV/attention, int4 dequant-matmul) is hand-written in this repo. What each
platform links against:

* **macOS** — only OS frameworks (`Accelerate`, `Metal`, `Foundation`).
* **Linux / Windows (CPU)** — nothing; own microkernels compiled straight
  into the consumer binary.
* **Linux / Windows (CUDA)** — nothing linked. Kernels are compiled to PTX
  at *build* time (`nvcc`, build-time only — not a runtime dependency) and
  the CUDA driver is `dlopen`'d (`LoadLibrary` on Windows) at *run* time, so
  a binary built with `-DTENSORLIB_CUDA=ON` still runs — falling back to
  CPU — on a machine with no NVIDIA driver at all.
* **WebAssembly** — nothing linked. The WGSL kernels are committed as a C
  string and the WebGPU entry points come from Emscripten's bundled
  `emdawnwebgpu` port, so the toolchain is the only requirement and it is a
  build-time one.
* **Tests only** — `doctest` is vendored (not a runtime dependency of the
  library itself).

Build and test
--------------

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure   # cpu / gpu / auto modes
./build/tensorlib_bench                       # micro-benchmarks
```

On Linux/Windows with an NVIDIA GPU, add `-DTENSORLIB_CUDA=ON` (needs `nvcc`
on `PATH` at *build* time only — see [Dependency policy](#dependency-policy)):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTENSORLIB_CUDA=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires **C++17 or newer** — the headers use inline variables,
`std::optional`, and structured bindings, but nothing from C++20/23, so any
`g++ >= 11` / `clang++ >= 13` works (Apple Clang too). The bundled CMake build
compiles the tests at C++23, but consuming the headers only needs C++17.
Exception: the **macOS/Metal** backend `#embed`s its shader source, so
*building on macOS* additionally needs a `#embed`-capable compiler (Clang 19+);
non-Apple builds never reach that `#embed`. The WebGPU backend has the same
problem and solves it the other way — its WGSL is committed as a generated C
string (`kernels/tensorlib_webgpu_wgsl.inc`, regenerated by
`kernels/gen_wgsl_inc.sh` when the `.wgsl` changes), so it needs no `#embed`
and no build step.

For a WebAssembly build, see `test/wasm/build.sh` — a flat `emcc` line with
`--use-port=emdawnwebgpu -DTENSORLIB_WEBGPU`, plus `-sJSPI=1`, which is what
lets `flush()` keep a synchronous signature.

LLM inference
-------------

`bench/cuda/chat_qwen.cpp` is the end-to-end proof of the header-only,
zero-dependency, cross-platform claims above: it loads a real
Qwen2.5-Instruct GGUF model and chats, using nothing but this library's own
kernels — own GGUF v3 reader (`include/gguf.h`), own GPT-2-byte-BPE
tokenizer (`include/tokenizer.h`), and own CUDA decode kernels. No GGML, no
llama.cpp, no third-party inference runtime anywhere in the path.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTENSORLIB_CUDA=ON
cmake --build build --target tensorlib_chat_qwen
./build/tensorlib_chat_qwen path/to/qwen2.5-0.5b-instruct.gguf "What is the capital of France?"
```

Decode throughput on an RTX 3090 has gone from 3.5 to ~330 tok/s over the
course of the CUDA decode-path work — about 1.35x off llama.cpp on the same
shape. See `bench/cuda/speed/bench_qwen_decode.cpp` and
[docs/performance-notes.md](docs/performance-notes.md) for the measurement
methodology and the full tuning history.

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
