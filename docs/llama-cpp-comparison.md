# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; tensorlib `95eef4d`, llama.cpp `19cce6919`, built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`; measured 2026-11-02. tok/s; this
library's column is the median of 5 runs, llama.cpp's the mean of 5 (what
`llama-bench` prints). Each row is two runs of each engine, alternated, and the
range is given: llama.cpp's decode moved by up to 5% between invocations here and
its prefill by up to 10% (Qwen2.5's pp512 has an sd of 9-16% within a run), so
the ratios there are soft.

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 499-508 | 437-464 | 1.08-1.16 |
|  | tg128 @ d2048 | 470-472 | 453-457 | 1.03-1.04 |
|  | pp512 | 60,155-60,479 | 49,718-52,180 | 1.15-1.22 |
|  | pp512 @ d2048 | 48,459-48,585 | 44,445-44,694 | 1.08-1.09 |
| Qwen3-0.6B F16 | tg128 | 428 | 384-395 | 1.08-1.12 |
|  | tg128 @ d2048 | 380-383 | 349-351 | 1.08-1.10 |
|  | pp512 | 45,678-45,724 | 40,171-40,419 | 1.13-1.14 |
|  | pp512 @ d2048 | 31,590-31,795 | 31,238-32,021 | 0.99-1.02 |
| Qwen3.5-0.8B F16 | tg128 | 425-426 | 305-310 | 1.37-1.39 |
|  | tg128 @ d2048 | 416-418 | 310-312 | 1.34-1.35 |
|  | pp512 | 34,873-36,322 | 29,866-29,931 | 1.17-1.22 |
|  | pp512 @ d2048 | 33,066-33,234 | 28,731-28,807 | 1.15-1.16 |
| Qwen3.5-0.8B BF16 | tg128 | 423-425 | 298 | 1.42-1.43 |
|  | tg128 @ d2048 | 417-418 | 299-311 | 1.34-1.40 |
|  | pp512 | 35,455-35,503 | 25,252-25,415 | 1.40-1.41 |
|  | pp512 @ d2048 | 33,284-33,493 | 24,689-24,923 | 1.34-1.36 |

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
| Qwen3-0.6B Q4_K_M | tg128 | 575-580 | 525-574 | 1.00-1.11 |
|  | tg128 @ d2048 | 426-455 | 413-437 | 0.97-1.10 |
|  | pp512 | 45,866-46,426 | 32,195-33,019 | 1.39-1.44 |
|  | pp512 @ d2048 | 31,539-31,746 | 26,564-26,844 | 1.17-1.20 |
| Qwen3.5-0.8B Q4_K_M | tg128 | 550-600 | 420-436 | 1.31-1.37 |
|  | tg128 @ d2048 | 573-618 | 407-451 | 1.27-1.52 |
|  | pp512 | 35,921-36,324 | 28,287-28,494 | 1.26-1.28 |
|  | pp512 @ d2048 | 31,975-32,380 | 26,919-27,079 | 1.18-1.20 |
| Qwen3.5-0.8B Q8_0 | tg128 | 596-597 | 350-427 | 1.40-1.71 |
|  | tg128 @ d2048 | 580-585 | 427-428 | 1.36-1.37 |
|  | pp512 | 38,301-38,464 | 30,302-30,528 | 1.25-1.27 |
|  | pp512 @ d2048 | 33,927-34,214 | 28,814-29,003 | 1.17-1.19 |
| Qwen3.8-27B Q4_K_M | tg128 | 36.7-40.9 | 35.5-37.4 | 1.04-1.09 |
|  | tg128 @ d2048 | 37.2-38.0 | 37.0-37.9 | 0.98-1.03 |
|  | pp512 | 1,722-1,734 | 1,698-1,702 | 1.01-1.02 |
|  | pp512 @ d2048 | 1,650-1,664 | 1,674-1,682 | 0.98-0.99 |
| Gemma 3 4B Q4_K_M | tg128 | 198-199 | 184-186 | 1.07-1.08 |
|  | pp512 | 10,580-10,902 | 10,753-10,769 | 0.98-1.01 |

Gemma 3's rows were taken later (2026-11-03), the two engines alternated in
the same way; its context stops at its 1024-token sliding window here, so it
has no row at depth 2048. The Q4_K_M files' decode rows were taken again then,
after the decode's Q4_K GEMV moved to f16 activations (below), on a noisier day
(llama.cpp's own runs moved by up to 9%); the int8 form (`TL_GEMV_ACT=q8`), run
in the same rounds, was up to 11% faster on the 0.6-4B files and level on the
27B.

The 27B's prefill is level: ahead at 512 tokens, 1-2% behind with 2048 cached.
It trailed by 3-6% until the DeltaNet rule became one kernel on llama.cpp's
schedule (above). llama.cpp's 27B decode and Qwen3.5 Q8_0's decode moved widely
within a run here (an sd of 9 tok/s on the 27B's ~37), so those ratios are the
softest.

Qwen3.8-27B is a Qwen3.5 architecture: 64 layers, 48 of them Gated DeltaNet with
16 q / k heads for 48 value heads (value head h reads key head h % 16, the order
the GGUF conversion stores them in), attention with 24 q / 4 kv heads of 256.
Its Q4_K_M holds the MLP in Q4_K, `attn_output` and `output` in Q6_K and every
other projection in Q8_0. The Q4_K GEMV (the MLP, half the bytes) ran at ~450
GB/s with the activation as float, held back by its instructions, not its bytes
(each weight a nibble shifted out, converted and multiplied on its own, ~12
instructions a weight: it runs as slowly with its weights in the cache, and as
slowly reading the activation as f16), where the weights alone stream at ~850.
Reading the activation as int8, as llama.cpp's does (quantized once per call to
8 bits per 32 values, then `dp4a`, ~3 instructions a weight), it runs at ~750,
and the 27B's decode went from 24.9 to 35.6 tok/s. It now reads it as f16 on the
tensor cores, as fast on the 27B and without the 8-bit rounding (below);
`TL_GEMV_ACT=q8` gives the int8 form and `=f32` the float one, and `check_quant`
bounds each against the dequantized oracle. The Q8_0 and Q6_K GEMVs already ran
at 700-850 GB/s on float activations and are unchanged. Without MMQ (`TL_NO_MMQ=1`, and for Q6_K always) the prefill
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

## Accuracy of the quantized paths

How far each path's logits move from an f32 reference, on the same tokens:
`tensorlib_dump_logits` runs a 200-token prompt and six more tokens through
every path, beside an f32 build of the same file (its blocks widened to f32
exactly, f32 activations, sums and KV cache) whose token-by-token logits are
the reference; `tools/precision_logits.py` reports the KL divergence from it
(nats, mean over the prompt's last token and the six steps, as llama.cpp's
`perplexity --kl-divergence` measures quantization). llama.cpp's logits for
the same tokens (the prompt in one `llama_decode`, then a token at a time, its
defaults: flash attention, f16 KV) are written by `tools/llama_logits.cpp`.

| model (GGUF) | chat path (batched prompt, then steps) | with `TL_NO_MMQ=1` | with an int8 decode (`TL_GEMV_ACT=q8`) | llama.cpp |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | 2.8e-4 | 3.8e-5 | 2.2e-3 | 2.3e-3 |
| Qwen3.5-0.8B Q4_K_M | 1.7e-4 | 3.4e-5 | 3.4e-3 | 2.0e-3 |
| Gemma 3 4B Q4_K_M | 3.3e-2 | 4.3e-3 | 6.0e-2 | 2.1e-1 |

The error is the int8 activations', not the f16 sums: the prefill's MMQ GEMMs
round their activation to int8 per 32 values, as llama.cpp's do (q8_1), and
until 2026-11-03 the decode's Q4_K GEMVs did too (`Model::gemv_form` q8, dp4a:
the fourth column); they now read it as f16 (below). With `TL_NO_MMQ=1` as well
nothing is rounded to int8, and the logits are as close as with float
activations throughout (3.9e-5, 3.4e-5, 4.4e-3). Gemma 3 is far more
sensitive to that rounding than Qwen: rounding every GEMV input to int8 per
32 values in a numpy forward (no kernel of either engine involved) moves its
logits by KL 0.13 against Qwen3-0.6B's 0.002, though the inputs themselves are
rounded alike (relative error ~1e-2 in both; no outliers past Qwen's). Every
one of the four inputs contributes (the MLP's input most), and the residual's
error grows with depth, from ~8e-3 after the first layer to ~4e-2 after the
last. Token by token through the int8 decode, a 200-token Gemma prompt ends at
KL 0.30 (top token 5 of 7), through the f16 one at 1.1e-5; llama.cpp's path, at
0.21, is no better than the int8 one.

`TL_NO_MMQ=1` (the prefill's block GEMMs through f16 instead of int8) takes the
rest of the int8 rounding out, at a cost on the prompt: Gemma 3 4B's pp512
10,640 -> 7,020 tok/s, Qwen3-0.6B's -11%.

The decode reads the f16 activation at close to int8's speed: its Q4_K GEMV
multiplies on the tensor cores (exact products, f32 sums: the float GEMV of the
activation rounded to f16, each 256 values first scaled by a power of two where
they would pass f16's range). The float GEMV's cost is its instructions, ~12 a
weight, which f16 activations on the CUDA cores do not change; on the tensor
cores a weight costs ~3.3, against int8's ~2.8. tg128, int8 / f16 / float,
alternated in one session:

| model (GGUF) | int8 | f16 | float |
|---|---:|---:|---:|
| Qwen3-0.6B Q4_K_M | 560-564 | 518-541 | 490-509 |
| Qwen3.5-0.8B Q4_K_M | 593-623 | 572-596 | 541-546 |
| Gemma 3 4B Q4_K_M | 202-211 | 191-208 | 155-157 |
| Qwen3.8-27B Q4_K_M | 38.9-39.7 | 38.5-39.2 | 28.5-28.8 |

The small models' gap is that arithmetic: a large layer's memory wait hides
it, a small one's does not (the kernel reading its blocks without the
arithmetic runs as fast as int8's, and more warps a block, or half a superblock
a warp, do not help). The prompts above are spread-out ids without a BOS, which
Gemma 3's chat prompts start with; after one (its first-token activations reach
~9e3) Gemma's distributions move far less, and the forms rank the same. Gemma 3
4B, BOS then 199 spread-out ids, KL of the six steps after the batched prompt:
int8 1.8e-3, f16 2.9e-4 (the float form's too: what is left comes from the
prompt, through the MMQ's int8), f16 with `TL_NO_MMQ=1` 8.5e-5; llama.cpp 1.3e-3.
Token by token through the decode alone, 2.7e-3 int8 against 5.3e-7 f16.

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
