# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; tensorlib `982a1e3`, llama.cpp `19cce6919` (2026-10-27), built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. tok/s; this library's column
is the median of 5 runs, llama.cpp's the mean of 5 (what `llama-bench` prints).
llama.cpp was run twice per row and the range is given: its decode moved by
up to 5% between invocations here, its prefill by up to 10% (the deep-context
prefill ratios are soft for that reason).

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 482 | 442-445 | 1.08 |
|  | tg128 @ d2048 | 436 | 437-442 | 0.99-1.00 |
|  | pp512 | 52,867 | 50,825-51,209 | 1.03-1.04 |
|  | pp512 @ d2048 | 43,750 | 43,865-44,494 | 0.98-1.00 |
| Qwen3-0.6B F16 | tg128 | 418 | 383-385 | 1.09 |
|  | tg128 @ d2048 | 329 | 329-351 | 0.94-1.00 |
|  | pp512 | 37,672 | 38,825-39,766 | 0.95-0.97 |
|  | pp512 @ d2048 | 23,199 | 28,892-31,436 | 0.74-0.80 |
| Qwen3.5-0.8B F16 | tg128 | 392 | 306-308 | 1.27-1.28 |
|  | tg128 @ d2048 | 381 | 295-301 | 1.27-1.29 |
|  | pp512 | 26,989 | 29,356-29,815 | 0.91-0.92 |
|  | pp512 @ d2048 | 21,583 | 25,157-28,194 | 0.77-0.86 |
| Qwen3.5-0.8B BF16 | tg128 | 384 | 283-295 | 1.30-1.36 |
|  | tg128 @ d2048 | 382 | 296-298 | 1.28-1.29 |
|  | pp512 | 25,284 | 21,805-25,139 | 1.01-1.16 |
|  | pp512 @ d2048 | 21,765 | 24,250-24,410 | 0.89-0.90 |

Decode is at parity or ahead on every model, with or without 2048 tokens
cached. Until the decode attention overlapped its key loads and split its
keys finer (2026-10-27), Qwen2.5 trailed by 18% at depth: a block's warps
walked ~64 keys each in series, one load at a time.

Prefill is at parity at 512 tokens on Qwen2.5 and within 5-9% on Qwen3 and
Qwen3.5 F16, and 1.1-1.35x behind with 2048 tokens cached. Like llama.cpp, it
sums the GEMMs in f16 by default (below). Its GEMM and its attention run on the tensor
cores: the GEMM over `mma.sync` and `ldmatrix` with two shared-memory stages
(two blocks share an SM, which beat deeper pipelines), the weights converted
to f16 as a block stages them, and the activation rounded to f16 once, by the
norm, SwiGLU or head merge that writes it, rather than by every block that
reads it; the attention with q, k and v rounded to f16. Qwen3.5's DeltaNet
layers run the gated delta rule in 32-token chunks (as small matrix
products, f32). What is left at short context is those layers: ~7 ms of a 512-token
prompt against ~3 ms for llama.cpp's fused tensor-core kernel (the scan runs
on 16 warps a block, since it is bound by a chain of ~8 barriers per chunk,
and the prep is not on the tensor cores). At depth the attention over the
cached keys is the next suspect; it has not been profiled yet.

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
