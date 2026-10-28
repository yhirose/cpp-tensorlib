# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; tensorlib `a0a0d43`, llama.cpp `19cce6919` (2026-10-27), built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. tok/s; this library's column
is the median of 5 runs, llama.cpp's the mean of 5 (what `llama-bench` prints).
Each row is two runs of each engine, alternated, and the range is given: llama.cpp's
decode moved by up to 5% between invocations here and its prefill by up to 10%
(Qwen2.5's pp512 has an sd of 9-11% within a run), so the ratios there are soft.

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 486 | 462-464 | 1.05 |
|  | tg128 @ d2048 | 444 | 455-456 | 0.97-0.98 |
|  | pp512 | 58,924-59,535 | 51,289-51,560 | 1.14-1.16 |
|  | pp512 @ d2048 | 47,164-47,713 | 43,366-43,982 | 1.08-1.10 |
| Qwen3-0.6B F16 | tg128 | 418 | 378-385 | 1.09-1.11 |
|  | tg128 @ d2048 | 352 | 344 | 1.03 |
|  | pp512 | 40,442-40,573 | 39,014-40,046 | 1.01-1.04 |
|  | pp512 @ d2048 | 28,324-28,447 | 31,012-31,324 | 0.90-0.92 |
| Qwen3.5-0.8B F16 | tg128 | 393 | 310 | 1.27 |
|  | tg128 @ d2048 | 383 | 294-310 | 1.24-1.30 |
|  | pp512 | 27,836-29,641 | 28,780-29,943 | 0.97-0.99 |
|  | pp512 @ d2048 | 26,802-27,111 | 28,074-28,133 | 0.95-0.97 |
| Qwen3.5-0.8B BF16 | tg128 | 388 | 296 | 1.31 |
|  | tg128 @ d2048 | 379-383 | 304-308 | 1.24-1.26 |
|  | pp512 | 29,025-29,903 | 25,181-25,248 | 1.15-1.19 |
|  | pp512 @ d2048 | 27,219-27,467 | 24,341-24,412 | 1.12-1.13 |

Decode is at parity or ahead on every model, with or without 2048 tokens
cached. Until the decode attention overlapped its key loads and split its
keys finer (2026-10-27), Qwen2.5 trailed by 18% at depth: a block's warps
walked ~64 keys each in series, one load at a time.

Prefill is ahead on Qwen2.5 and the BF16 Qwen3.5 file, level at 512 tokens on
Qwen3 and Qwen3.5 F16, and 3-10% behind with 2048 tokens cached on Qwen3 and
Qwen3.5 F16. Like llama.cpp, it sums the GEMMs in f16 by default (below). Its
GEMM and its attention run on the tensor cores. The GEMM is `mma.sync` over
`ldmatrix` with two shared-memory stages (two blocks share an SM, which beat
deeper pipelines); the weights are converted to f16 as a block stages them, and
the activation is rounded to f16 once, by the norm, SwiGLU or head merge that
writes it, rather than by every block that reads it. The attention keeps a
warp's q in registers (shared memory at head width 256), reads k and v with
`ldmatrix` from a double buffer, and rounds q, k and v to f16. Qwen3.5's
DeltaNet layers run the gated delta rule in 32-token chunks (as small matrix
products, f32); what is left there is the chunk prep, ~2.5 ms of a 512-token
prompt, which is not on the tensor cores, and the scan, bound by a chain of
~8 barriers per chunk.

## Quantized files

A GGUF whose projections are Q4_K, Q6_K or Q8_0 blocks keeps them as blocks on
the device (uploaded from the file's mapping, never through a host copy): the
decode GEMVs and the prefill GEMM's B staging dequantize as they read, as ggml
does (bit for bit, checked over every quantized value of Qwen3.8-27B's file). The
decode is bound by the bytes it reads, so this is what lets a 27B model's 17.7 GiB
sit on a 24 GB card. Each row is one invocation of each engine (3 rounds), not
alternated, so a few percent of noise is in the ratios.

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | tg128 | 519 | 528 | 0.98 |
| | pp512 | 37,300 | 28,900 | 1.29 |
| Qwen3.5-0.8B Q4_K_M | tg128 | 500 | 439 | 1.14 |
| | pp512 | 26,100 | 25,900 | 1.01 |
| Qwen3.5-0.8B Q8_0 | tg128 | 469 | 389 | 1.20 |
| | pp512 | 26,300 | 28,300 | 0.93 |
| Qwen3.8-27B Q4_K_M | tg128 | 35.6 | 38.1 | 0.93 |
| | pp512 | 1,140 | 1,564 | 0.73 |

Qwen3.8-27B is a Qwen3.5 architecture: 64 layers, 48 of them Gated DeltaNet with
16 q / k heads for 48 value heads (value head h reads key head h % 16, the order
the GGUF conversion stores them in), attention with 24 q / 4 kv heads of 256.
Its Q4_K_M holds the MLP in Q4_K, `attn_output` and `output` in Q6_K and every
other projection in Q8_0. The Q4_K GEMV (the MLP, half the bytes) reads the
activation as int8, as llama.cpp's does: quantized once per call to 8 bits per
32 values, then `dp4a`. With the activation as float that GEMV ran at ~450 GB/s,
held back by the activation loads (8 `float4` a lane for every 32 weights), where
the weights alone stream at ~850; on int8 it runs at ~750 and the 27B's decode
goes from 24.9 to 35.6 tok/s. The price is the 8-bit rounding of the activation
(`TL_NO_Q8ACT=1` turns it off; `check_quant` bounds both against the dequantized
oracle). The Q8_0 and Q6_K GEMVs already ran at 700-850 GB/s on float activations
and are unchanged. The prefill dequantizes each projection to f16 once per chunk
(a slab of rows at a time, ahead of the GEMM) rather than in the staging of every
row block of the activation, which lets the GEMM stream plain f16: 23.7k to
37.3k tok/s on Qwen3-0.6B Q4_K_M, with bit-identical results. On the 27B the
f16 GEMM uses a 256x128 tile (BK 32, three `cp.async` stages, 16 warps) for these
long-K shapes, 86 TFLOPS against 63 for the 128x128 tile, cuBLAS's ~95; the
dequantization slab is 192 MB so that no projection is split into GEMMs too small
to fill the GPU. The prefill is then 61% GEMM, 21% dequantization. llama.cpp's
1,564 is more than an f16 GEMM can reach (the 27B's 26.6 TFLOP take 310 ms at 86
TFLOPS): it multiplies the quantized weights on the int8 tensor cores (twice the
f16 rate, and no dequantization pass), which is what is left of the 0.73.

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
