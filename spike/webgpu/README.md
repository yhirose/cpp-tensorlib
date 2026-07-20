# WebGPU backend spike

A standalone probe, not a backend. Nothing here is included by `include/`; the
point was to measure the things that would otherwise be guessed at before
committing to a design.

Build and run:

```sh
./build.sh jspi        # or: asyncify
python3 -m http.server 8731
# open http://localhost:8731/site/index.html
```

Measured on an M1 Pro (`apple / metal-3`), Chrome, emsdk 6.0.3,
emdawnwebgpu `v20260423.175430`.

## What it proves

**1. `gpu::flush()` can stay synchronous.** This was the stated hard part and it
turns out to be a solved problem upstream rather than something we have to
engineer around:

- Device acquisition needs no async on the C++ side at all. JS does
  `requestAdapter`/`requestDevice` and passes the result as
  `Module.preinitializedWebGPUDevice`; C++ picks it up with a plain
  `emscripten_webgpu_get_device()`.
- Blocking waits are a supported, documented path:
  `wgpuInstanceWaitAny(timeoutNS > 0)` after creating the instance with the
  `TimedWaitAny` feature. Internally it suspends via Asyncify/JSPI
  (`webgpu.cpp:560`, "To handle timeouts, use Asyncify and proxy back into
  JS"). If the link lacks Asyncify/JSPI, `CreateInstance` returns null — it
  fails loudly at startup rather than deadlocking later.

So `flush()` and `sync_to_host()` keep their existing signatures. The suspend
surface is exactly two call sites (`OnSubmittedWorkDone`, `MapAsync`), not the
whole interpreter. Uploads (`WriteBuffer`) are queued, not awaited, and cost
nothing here.

**2. A hand-written WGSL sgemm is clearly worth it.** 64x64 threadgroup tile,
16x16 invocations, 4x4 register accumulator — the shape of `sgemm_64_` in
`metal_kernels.metal`, minus the MMA intrinsics, which WGSL has no equivalent
for. Medians of 7 (GPU) / 3 (CPU), interleaved, after warmup:

| n    | GPU        | CPU (tensorlib, scalar wasm) | speedup | maxerr  |
|------|------------|------------------------------|---------|---------|
| 128  | 0.6–1.0 ms (4–7 GF/s)   | 0.5 ms (8.4 GF/s)  | ~1x   | 4.8e-07 |
| 256  | 1.0–1.1 ms (30–34 GF/s) | 4.2 ms (8.0 GF/s)  | ~4x   | 7.8e-07 |
| 512  | 1.7 ms (158 GF/s)       | 33–37 ms (7–8 GF/s)| ~20x  | 1.4e-06 |
| 1024 | 3.4–3.7 ms (580–630 GF/s)| 263–271 ms (~8 GF/s)| ~75x | 4.8e-06 |

## What it changes about the plan

**There is a ~0.6–1.0 ms fixed floor per dispatch+wait.** Metal's is orders of
magnitude lower. Two consequences:

- `auto_threshold_()` cannot reuse the Metal arm. Crossover here is around
  n≈128–160, i.e. work ≈ 4–8e6 in the `num_elements * k` units `gpu_gemm` uses,
  against Metal's 5e8. It is *lower* than Metal's, not higher, because the wasm
  CPU baseline is scalar and weak (~8 GF/s, no threads, no SIMD) — not because
  WebGPU is fast. Port `misc/census.cpp` to the browser before picking numbers.
- Batching matters more than on Metal. One `flush()` per graph eval is fine;
  one per op would be dominated by the floor.

**Asyncify vs JSPI is a real fork, and it is not a performance question.** Both
work and both measure the same (an early run suggesting otherwise was a
cold-start outlier — worth repeating, given how easily this one misleads):

|                     | Asyncify | JSPI |
|---------------------|----------|------|
| spike wasm size     | 397 KB   | 258 KB |
| `-fwasm-exceptions` | emcc warns they are incompatible; the spike links anyway, but it does not throw across a suspend — culebra's interpreter might | clean |
| browser support     | universal | Chrome shipped; Firefox behind a flag; Safari not yet |
| extra link flags    | — | `-sJSPI_EXPORTS=<entry points>`, and JS must `await` those calls |

The catch: `-sJSPI` produces a wasm that *requires* JSPI, so it will not run at
all on Safari. Shipping GPU via JSPI therefore means two builds plus a JS
feature check, not one build that degrades. Asyncify is one universal build but
pays size everywhere and has an unresolved interaction with the exception model
culebra already uses.

## Not covered

Only `gemm`, only F32, only the untransposed case, no buffer pooling, no
`storage::native` integration. The host-visible-pointer assumption in
`gpu::alloc(bytes, float** contents)` is the next thing to design against:
WebGPU storage buffers have no CPU pointer, so the backend needs the
host-shadow + `sync_to_host()` mirror that `cuda.h` already implements.
