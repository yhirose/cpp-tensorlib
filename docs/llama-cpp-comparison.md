# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; llama.cpp `19cce6919` (2026-10-27), built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. tok/s; this library's column
is the median of 5 runs, llama.cpp's the mean of 5 (what `llama-bench` prints).
Where llama.cpp was measured more than once the range is given: its decode
moved by up to 5% between invocations here (one Qwen3 run, ± 139, is left
out).

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 461 | 429-439 | 1.05-1.07 |
| | tg128 @ d2048 | 423 | 407-430 | 0.98-1.04 |
| | pp512 | 56,417 | 49,132 | 1.15 |
| | pp512 @ d2048 | 42,721 | 41,106 | 1.04 |
| Qwen3-0.6B F16 | tg128 | 400 | 351-373 | 1.07-1.14 |
| | tg128 @ d2048 | 342 | 330-334 | 1.02-1.04 |
| | pp512 | 36,966 | 38,881 | 0.95 |
| | pp512 @ d2048 | 24,510 | 30,004 | 0.82 |
| Qwen3.5-0.8B F16 | tg128 | 386 | 290-291 | 1.33 |
| | tg128 @ d2048 | 376 | 289-292 | 1.29-1.30 |
| | pp512 | 24,996 | 27,916 | 0.90 |
| | pp512 @ d2048 | 19,745 | 27,401 | 0.72 |
| Qwen3.5-0.8B BF16 | tg128 | 379 | 288 | 1.32 |
| | tg128 @ d2048 | 372 | 288 | 1.29 |
| | pp512 | 25,485 | 23,963 | 1.06 |
| | pp512 @ d2048 | 21,058 | 23,208 | 0.91 |

Decode is at parity or ahead on every model, with or without 2048 tokens
cached. Until the decode attention overlapped its key loads and split its
keys finer (2026-10-27), Qwen2.5 trailed by 18% at depth: a block's warps
walked ~64 keys each in series, one load at a time.

Prefill is between 1.15x ahead and 1.4x behind. Like llama.cpp, it sums the
GEMMs in f16 by default (below). Its GEMM and its attention run on the tensor
cores: the GEMM over `mma.sync` and `ldmatrix` with two shared-memory stages
(two blocks share an SM, which beat deeper pipelines), the weights converted
to f16 as a block stages them, and the activation rounded to f16 once, by the
norm, SwiGLU or head merge that writes it, rather than by every block that
reads it; the attention with q, k and v rounded to f16. Qwen3.5's DeltaNet
layers run the gated delta rule in 32-token chunks (as small matrix
products, f32). What is left of the gap is those layers: ~8 ms of a 512-token
prompt against ~3 ms for llama.cpp's fused tensor-core kernel, the scan being
bound by a serial chain of ~8 barriers per chunk. The llama.cpp column was
measured before the GEMM change and not repeated.

## Accumulation precision

**This was most of the prefill gap.** llama.cpp multiplies F16 weights
through cuBLAS with `CUBLAS_COMPUTE_16F`, summing in f16
(`ggml_cuda_mul_mat_cublas`); BF16 weights get `CUBLAS_COMPUTE_32F`. On this
GPU the tensor cores run f16-accumulate at twice the f32-accumulate rate,
measured with `mma.sync.m16n8k16`: 142 against 71 TFLOPS (bf16 with f32
accumulation is 71 as well). This library sums in f32, so its ceiling is
the lower one. llama.cpp's own F16 files prefill 1.16-1.21x faster than the
same weights as BF16.

**What f16 sums cost, measured in llama.cpp** (its own F16 weights, the
cuBLAS compute type switched with `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, KL
divergence over this repository's README and docs/backends.md): mean KLD
0.0002-0.0003 against f32 sums and the same top token 98.5-98.9% of the
time, where Q8_0 weights (with f32 sums) are 0.0028 and 96.9% for Qwen3-0.6B.
Perplexity moves by under 0.04%.

**So this library does the same by default**: the batched prefill's GEMMs
round their operands to f16 and sum in f16 (`Model::f16_acc`;
`chat_qwen ... f32acc` and `bench_qwen_llama ... f32acc` turn it off). f16's
range is the risk — an activation or a partial sum past 65504 becomes inf —
and an inf reaches every logit as NaN, so `prefill_batched` checks the
prompt's logits and, if any is not finite, puts the caches and the DeltaNet
state back and redoes the prompt exactly (f32 sums and f32 attention).
`check_qwen` forces that overflow and holds the retry to an exact prefill.
Decode is unaffected; it sums in f32.

**With f32 sums on both sides** — bf16 weights, f32 sums on the tensor cores,
which is what llama.cpp does for a BF16 file (the Qwen2.5 and Qwen3 BF16
files below were made from the F16 ones with `llama-quantize ... BF16`) —
pp512 measured:

| model | tensorlib (f32acc) | llama.cpp | ratio |
|---|---:|---:|---:|
| Qwen2.5-0.5B BF16 | 37,678 | 41,255 | 0.91 |
| Qwen3-0.6B BF16 | 27,225 | 33,111 | 0.82 |
| Qwen3.5-0.8B BF16 | 18,697 | 24,801 | 0.75 |

What is left there is mostly the GEMM: this library's runs at ~46% of the
71 TFLOPS ceiling, and the narrow projections (QKV, O, down) fall to its
64x64 tile to fill the GPU.

## What differs between the two runs

- **Prefill precision.** This library's prefill GEMM sums in f16 by default
  (above; with `f32acc`, bf16 operands and f32 sums), and its attention rounds
  q, k and v to f16 (normalized q and k sit far inside f16's range — Qwen2.5's
  largest is ~220). Decode and the array API's attention stay f32. Against
  the f32 path, the top-5 logits of a prompt's last token move by up to ~8e-3;
  a long thinking-mode answer can take another word where two were near-tied.
- **Weights.** llama.cpp computes from the F16 (or BF16) weights as the file
  holds them; this library stores them as bf16. Both read two bytes a weight,
  so decode, which is bound by reading them, compares like for like. For the
  Qwen3 and Qwen3.5 F16 files every value is exact in bf16, so the weights
  are identical too; Qwen's own Qwen2.5 F16 file has values that bf16 rounds
  (by at most 0.39%).
- **KV cache.** llama.cpp's default is F16 and `llama-bench` offers no f32;
  this library's is f32, so at depth it reads twice the cache bytes per step.
  At 2048 tokens that is tens of microseconds of a ~2.5 ms step.
- **Variance.** Within one invocation, llama.cpp's prefill rows vary by
  5-10% here and this library's by 1-2%, decode by 1-2% on both. Between
  invocations llama.cpp's decode moved by up to 5%, hence the ranges.

## Reproducing

The files: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (`qwen2.5-0.5b-instruct-fp16.gguf`),
`ggml-org/Qwen3-0.6B-GGUF` (`Qwen3-0.6B-f16.gguf`),
`AaryanK/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B.fp16.gguf`) and
`unsloth/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B-BF16.gguf`) on Hugging Face.

```bash
# this library (a CUDA build)
build/tensorlib_bench_qwen_llama model.gguf 512 128 5 0
build/tensorlib_bench_qwen_llama model.gguf 512 128 5 2048

# llama.cpp
llama-bench -m model.gguf -p 512 -n 128 -r 5 -d 0 -ngl 99
llama-bench -m model.gguf -p 512 -n 128 -r 5 -d 2048 -ngl 99
```

`bench/cuda/speed/bench_qwen_llama.cpp` runs each test once untimed, then
the given number of times. Its tg goes through the chat's own decoder, the
captured CUDA graph, one greedy token a step; capturing it is set-up and is
not timed. Run the two engines back to back with nothing else on the GPU.
