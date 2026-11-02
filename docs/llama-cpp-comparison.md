# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; tensorlib `2782b61` (Qwen3.5-0.8B BF16: `cdc5fdf`,
whose DeltaNet conv is a little faster), llama.cpp `19cce6919`, built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`; measured 2026-11-02. tok/s; this
library's column is the median of 5 runs, llama.cpp's the mean of 5 (what
`llama-bench` prints). Each row is two runs of each engine, alternated, and the
range is given: llama.cpp's decode moved by up to 5% between invocations here and
its prefill by up to 10% (Qwen2.5's pp512 has an sd of 9-16% within a run), so
the ratios there are soft.

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 510-512 | 469-470 | 1.09 |
|  | tg128 @ d2048 | 471-472 | 448 | 1.05 |
|  | pp512 | 60,681-60,821 | 50,265-51,104 | 1.19-1.21 |
|  | pp512 @ d2048 | 48,679-48,810 | 43,513-45,155 | 1.08-1.12 |
| Qwen3-0.6B F16 | tg128 | 426-428 | 382-386 | 1.10-1.12 |
|  | tg128 @ d2048 | 379-381 | 350-359 | 1.06-1.09 |
|  | pp512 | 45,943-46,566 | 39,840-40,284 | 1.14-1.17 |
|  | pp512 @ d2048 | 31,970-32,071 | 32,217-32,243 | 0.99-1.00 |
| Qwen3.5-0.8B F16 | tg128 | 424-425 | 300-314 | 1.35-1.42 |
|  | tg128 @ d2048 | 413-416 | 313-314 | 1.31-1.33 |
|  | pp512 | 34,085-35,374 | 30,254-30,258 | 1.13-1.17 |
|  | pp512 @ d2048 | 30,934-31,137 | 28,769-28,929 | 1.07-1.08 |
| Qwen3.5-0.8B BF16 | tg128 | 407-408 | 295-298 | 1.37-1.38 |
|  | tg128 @ d2048 | 397-399 | 283-293 | 1.36-1.41 |
|  | pp512 | 31,663-31,933 | 24,742-25,008 | 1.27-1.29 |
|  | pp512 @ d2048 | 30,331-30,629 | 23,603-23,782 | 1.28-1.30 |

Decode is ahead on every model, with or without 2048 tokens cached. Until the
decode attention overlapped its key loads and split its keys finer
(2026-10-27), Qwen2.5 trailed by 18% at depth: a block's warps walked ~64 keys
each in series, one load at a time.

Prefill is ahead on every file at 512 tokens, and with 2048 tokens cached on
all but Qwen3, where it is level. Like llama.cpp, it sums the GEMMs in f16 by default (below). Its
GEMM and its attention run on the tensor cores. The GEMM is `mma.sync` over
`ldmatrix` with two shared-memory stages (two blocks share an SM, which beat
deeper pipelines); the weights are converted to f16 as a block stages them, and
the activation is rounded to f16 once, by the norm, SwiGLU or head merge that
writes it, rather than by every block that reads it. The attention keeps a
warp's q in registers (shared memory at head width 256), reads k and v with
`ldmatrix` from a double buffer, and rounds q, k and v to f16. Qwen3.5's
DeltaNet layers run the gated delta rule as llama.cpp's CUDA backend does: one
kernel, a block a head (or a slice of its value columns, where heads alone
leave SMs idle), walking 16-token chunks with the state in tensor-core
accumulators, every product a warp's own output tile over its whole reduction,
as bf16 high / residual pairs in three mmas (about 7e-6 of an fp64 rule over
512 tokens). On the 27B it is 0.34 ms a layer at 512 tokens, the q / k norms
and gates included; the two kernels it replaced (a chunk prep, then a scan that
summed the state's rows across warps every chunk) took 0.63.

## Quantized files

A GGUF whose projections are Q4_K, Q6_K or Q8_0 blocks keeps them as blocks on
the device (uploaded from the file's mapping, never through a host copy): the
decode GEMVs read them as they lie, dequantizing as ggml does (bit for bit,
checked over every quantized value of Qwen3.8-27B's file), and the prefill
multiplies Q4_K and Q8_0 on the int8 tensor cores (MMQ, below). The decode is
bound by the bytes it reads, so this is what lets a 27B model's 17.7 GiB sit on a
24 GB card. Measured as the table above (same commits and day, two runs of
each engine, alternated):

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | tg128 | 606-610 | 573-583 | 1.04-1.06 |
| | tg128 @ d2048 | 521 | 498-506 | 1.03-1.05 |
| | pp512 | 46,394-46,904 | 32,637-33,396 | 1.39-1.44 |
| | pp512 @ d2048 | 31,696-32,082 | 26,509-27,119 | 1.17-1.21 |
| Qwen3.5-0.8B Q4_K_M | tg128 | 648-650 | 477-482 | 1.34-1.36 |
| | tg128 @ d2048 | 615-623 | 442-452 | 1.36-1.41 |
| | pp512 | 34,196-34,263 | 27,482-28,321 | 1.21-1.25 |
| | pp512 @ d2048 | 28,922-29,963 | 26,826-26,897 | 1.08-1.12 |
| Qwen3.5-0.8B Q8_0 | tg128 | 582-586 | 415-417 | 1.40-1.41 |
| | tg128 @ d2048 | 567-573 | 348-430 | 1.32-1.65 |
| | pp512 | 35,894-36,027 | 30,188-30,427 | 1.18-1.19 |
| | pp512 @ d2048 | 31,936-32,089 | 28,885-29,022 | 1.10-1.11 |
| Qwen3.8-27B Q4_K_M | tg128 | 41.3-41.5 | 36.6-40.6 | 1.02-1.13 |
| | tg128 @ d2048 | 40.3-40.4 | 36.3-36.5 | 1.10-1.11 |
| | pp512 | 1,614-1,663 | 1,709-1,713 | 0.94-0.97 |
| | pp512 @ d2048 | 1,585 | 1,661-1,682 | 0.94-0.95 |

The 27B's prefill is the one row behind. llama.cpp's 27B decode and Qwen3.5
Q8_0's decode at depth moved widely within a run here (an sd of 9 tok/s on the
27B's ~37, and of 176 on Q8_0's), so those ratios are the softest.

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
and are unchanged. Without MMQ (`TL_NO_MMQ=1`, and for Q6_K always) the prefill
dequantizes each projection to f16 once per chunk
(a slab of rows at a time, ahead of the GEMM) rather than in the staging of every
row block of the activation, which lets the GEMM stream plain f16: 23.7k to
37.3k tok/s on Qwen3-0.6B Q4_K_M, with bit-identical results. On the 27B the
f16 GEMM uses a 256x128 tile (BK 32, three `cp.async` stages, 16 warps) for these
long-K shapes, 86 TFLOPS against 63 for the 128x128 tile, cuBLAS's ~95; the
dequantization slab is 192 MB so that no projection is split into GEMMs too small
to fill the GPU. That prefill is then 61% GEMM, 21% dequantization, and reaches
0.69 of llama.cpp: its pp512 is more than an f16 GEMM can reach (the 27B's 26.6
TFLOP take 310 ms at 86 TFLOPS). llama.cpp multiplies the quantized weights on the
int8 tensor cores instead (twice the f16 rate, and no dequantization pass).

So does the default prefill here, for Q4_K and Q8_0, as llama.cpp's MMQ does: the
activation is rounded to int8 (a scale per 32 values) and multiplied with the
weight's own integers on `mma.m16n8k32` s8, so nothing is dequantized and no f16
copy of the weight is written and read back. As in llama.cpp's Ampere kernel, a
128 x 128 tile of 8 warps unpacks the weight to int8 rows in shared memory once
as it stages (the multiply loop is then the same for every format), K is split
where the grid would leave SMs idle, and the fused gate|up projection writes
silu(gate) * up in its epilogue. The price is the activation's 8 bits, as in
llama.cpp: they change the logits (a greedy reply departs from the f16 path's
after a few hundred tokens; `check_cuda` holds the kernels to the int8-activation
reference at 1e-7 and bounds the rounding's cost at 4e-3), and `TL_NO_MMQ=1`
takes the f16 path instead. At the shape
llama.cpp's own perf test has (4096 x 14336, 512 tokens) its Q4_K and Q8_0 MMQ
run at 83 TFLOPS, the activation's rounding included; these run at 83 and 85
without it (it adds 4%). Alternated, pp512:

| model (GGUF) | `TL_NO_MMQ=1` (f16) | MMQ (default) | llama.cpp |
|---|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | 36.2-36.6k | 41.4-41.5k | 32.0-32.5k |
| Qwen3.5-0.8B Q4_K_M | 28.3-29.4k | 32.6-33.2k | 26.7-28.6k |
| Qwen3.5-0.8B Q8_0 | 30.4-30.5k | 31.9-34.7k | 30.1-30.5k |
| Qwen3.8-27B Q4_K_M | 1,105-1,199 | 1,534-1,605 | 1,334-1,684 (median ~1,585) |

On the 27B a 64-token chunk goes from 314 to 605 tok/s, where the dequantization
is most of the work. llama.cpp's 27B pp512 moved by a quarter between runs when
this table was taken, so the ratio there was level; in the later runs above it
held at ~1,710, 3-6% ahead.

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
  largest is ~220). The decode's attention sums in f32 over the f16 cache
  (below); the array API's attention stays f32. Against
  the f32 path, the top-5 logits of a prompt's last token move by up to ~8e-3;
  a long thinking-mode answer can take another word where two were near-tied.
- **Weights.** llama.cpp computes from the F16 (or BF16) weights as the file
  holds them; this library stores them as bf16. Both read two bytes a weight,
  so decode, which is bound by reading them, compares like for like. For the
  Qwen3 and Qwen3.5 F16 files every value is exact in bf16, so the weights
  are identical too; Qwen's own Qwen2.5 F16 file has values that bf16 rounds
  (by at most 0.39%).
- **KV cache.** f16 on both by default: llama.cpp's default, and this
  library's where the backend has it (CUDA; `TL_KV_F32=1` keeps it f32, as an
  f32-weight model does). The prefill's attention rounds K and V to f16 either
  way, so a prompt's logits are the same with either cache; the decode then
  reads K and V rounded, and its logits move (check_qwen's greedy runs match).
  Half the bytes per step at depth: Qwen3-0.6B's cache is 470 MB at 2048 tokens
  in f32.
- **Variance.** Within one invocation, llama.cpp's prefill rows vary by
  5-16% here and this library's by 1-2%, decode by 1-2% on both (llama.cpp's
  on the 27B and on Qwen3.5 Q8_0 at depth by far more). Between invocations
  llama.cpp's decode moved by up to 5%, hence the ranges.

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
