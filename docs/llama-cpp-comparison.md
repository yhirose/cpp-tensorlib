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
| | pp512 | 35,685 | 49,132 | 0.73 |
| | pp512 @ d2048 | 30,773 | 41,106 | 0.75 |
| Qwen3-0.6B F16 | tg128 | 400 | 351-373 | 1.07-1.14 |
| | tg128 @ d2048 | 342 | 330-334 | 1.02-1.04 |
| | pp512 | 25,682 | 38,881 | 0.66 |
| | pp512 @ d2048 | 20,003 | 30,004 | 0.67 |
| Qwen3.5-0.8B F16 | tg128 | 386 | 290-291 | 1.33 |
| | tg128 @ d2048 | 376 | 289-292 | 1.29-1.30 |
| | pp512 | 11,999 | 27,916 | 0.43 |
| | pp512 @ d2048 | 10,544 | 27,401 | 0.38 |
| Qwen3.5-0.8B BF16 | tg128 | 379 | 288 | 1.32 |
| | tg128 @ d2048 | 372 | 288 | 1.29 |
| | pp512 | 11,747 | 23,963 | 0.49 |
| | pp512 @ d2048 | 10,439 | 23,208 | 0.45 |

Decode is at parity or ahead on every model, with or without 2048 tokens
cached. Until the decode attention overlapped its key loads and split its
keys finer (2026-10-27), Qwen2.5 trailed by 18% at depth: a block's warps
walked ~64 keys each in series, one load at a time.

Prefill is 1.3-2.6x behind. Its GEMM and its attention now run on the tensor
cores too: the GEMM with the activations rounded to bf16, the attention with
q, k and v rounded to f16 (both computed in f32 on the CUDA cores until
2026-10-27, when prefill was 2.6-5.6x behind and Qwen3's dropped 3x at
depth). The widest gap left is Qwen3.5's DeltaNet layers, which run their
recurrence token by token within each launch.

## What differs between the two runs

- **Prefill precision.** This library's prefill GEMM rounds its activations
  to bf16 for the tensor cores (bf16 for the range), and its attention rounds
  q, k and v to f16 (f16 for the mantissa: a score is an exponent, and
  normalized q and k sit far inside f16's range — Qwen2.5's largest is ~220).
  Decode and the array API's attention stay f32. The greedy tokens and the
  chat output are unchanged; the top-5 logits of a prompt's last token move
  by ~5e-3 against the f32 path, the low-probability tail by up to ~0.2.
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
