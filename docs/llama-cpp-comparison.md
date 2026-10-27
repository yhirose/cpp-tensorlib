# Against llama.cpp

The same GGUF files through both engines on one machine, measured the way
`llama-bench` measures: `pp512` is a 512-token prompt into the cache, `tg128`
is 128 tokens decoded one at a time, and `@ d2048` means 2048 tokens were
already cached when the test started.

## Results

RTX 3090, driver 610.88, WSL2; llama.cpp `19cce6919` (2026-10-27), built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. tok/s; this library's column
is the median of 5 runs, llama.cpp's the mean of 5 (what `llama-bench` prints).

| model (GGUF) | test | tensorlib | llama.cpp | ratio |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct F16 | tg128 | 436 | 439 | 0.99 |
| | tg128 @ d2048 | 347 | 421 | 0.82 |
| | pp512 | 18,440 | 49,132 | 0.38 |
| | pp512 @ d2048 | 12,127 | 41,106 | 0.30 |
| Qwen3-0.6B F16 | tg128 | 392 | 373 | 1.05 |
| | tg128 @ d2048 | 326 | 330 | 0.99 |
| | pp512 | 11,743 | 38,881 | 0.30 |
| | pp512 @ d2048 | 5,303 | 30,004 | 0.18 |
| Qwen3.5-0.8B F16 | tg128 | 326 | 291 | 1.12 |
| | tg128 @ d2048 | 320 | 289 | 1.11 |
| | pp512 | 4,537 | 27,916 | 0.16 |
| | pp512 @ d2048 | 4,280 | 27,401 | 0.16 |
| Qwen3.5-0.8B BF16 | tg128 | 324 | 288 | 1.12 |
| | tg128 @ d2048 | 320 | 288 | 1.11 |
| | pp512 | 4,507 | 23,963 | 0.19 |
| | pp512 @ d2048 | 4,224 | 23,208 | 0.18 |

Decode is at parity or ahead, except Qwen2.5 at depth, where it is 18% behind.
Prefill is 2.6-6x behind: this library's prefill GEMM computes in f32 on the
CUDA cores, where llama.cpp's uses the tensor cores, and Qwen3.5's DeltaNet
layers run their recurrence token by token within each launch.

## What differs between the two runs

- **Weights.** llama.cpp computes from the F16 (or BF16) weights as the file
  holds them; this library stores them as bf16. Both read two bytes a weight,
  so decode, which is bound by reading them, compares like for like. For the
  Qwen3 and Qwen3.5 F16 files every value is exact in bf16, so the weights
  are identical too; Qwen's own Qwen2.5 F16 file has values that bf16 rounds
  (by at most 0.39%).
- **KV cache.** llama.cpp's default is F16 and `llama-bench` offers no f32;
  this library's is f32, so at depth it reads twice the cache bytes per step.
  At 2048 tokens that is tens of microseconds of a ~3 ms step, too little to
  explain the Qwen2.5 gap at depth, whose cause is not yet known.
- **Variance.** llama.cpp's prefill rows vary by 5-10% between runs here,
  this library's by 1-2%; decode varies by about 1% on both.

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
