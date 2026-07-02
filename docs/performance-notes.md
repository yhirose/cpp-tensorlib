# Performance notes — how to measure, and what not to retry

Started on day one (a silarray lesson: record refuted approaches so dead
ends are never re-walked). Every dispatch gate and design choice on a hot
path must carry the measurement that set it.

## Performance gates

The reference target is **PyTorch/libtorch** on the same hardware (silarray
benched against MLX; this project is cross-platform, so PyTorch is the bar).

| Backend | Gate |
|---|---|
| macOS (Accelerate + Metal) | ≥ silarray (which is ≈ MLX) and ≥ PyTorch MPS |
| Own CPU GEMM | ≥ 90% of OpenBLAS single/multi-thread, ≥ PyTorch CPU |
| Own CUDA GEMM | ≥ 90% of cuBLAS FP32, ≥ PyTorch CUDA |

Where a gate cannot be met on a shape band after honest effort, record the
band and the data here, then consider the escape hatch (OpenBLAS / CUTLASS
for that band only) — do not ship a silent loss.

## Measurement discipline (inherited from silarray, verified there)

- GPU clock state swings absolute times ±30–60%: only same-run interleaved
  A/B comparisons are meaningful. Check machine load before believing ratios.
- Per-kernel census → fix the single most expensive kernel → repeat. Don't
  guess at bottlenecks.
- Warm up (~300 ms) before timing GPU work after CPU-only sections.
- Scale numeric tolerances with value magnitude (fp32 summation-order noise).
- Every dispatch gate is a measured value: comment the number and the bench
  that produced it next to the gate.

## Reference hardware

| Machine | Notes |
|---|---|
| Apple M1 Pro | macOS backend; paravirtual GPU on CI is correctness-only — never bench on CI |
| RTX 3090 (sm_86), Ubuntu on WSL2 | CUDA backend; driver at `/usr/lib/wsl/lib/libcuda.so`. WSL2 adds submission latency vs native Linux — record which side of that line a number came from |

## Current baselines (M1 Pro, interleaved medians)

| Op | Backend | Number | Note |
|---|---|---|---|
| sgemm 1024³ | Accelerate | ~980 GFLOP/s | AMX; strong at all sizes |
| sgemm 2048³ | Metal (32×32×16 tile) | ~1300 GFLOP/s | **functional, not yet tuned** |
| sgemm ≤1024³ | Metal | slower than accel-cpu | GPU wins only ≥2048 so far |

**Known gap (M3b-2):** the Metal SGEMM is a basic 32×32×16 tile (4
simdgroups, FM=FN=2). M1 Pro GPU fp32 ceiling is ~4–5 TFLOP/s, so there is
~3× headroom. The optimization pass — larger 64×64 tiles with register
blocking, float4 vectorized loads on the non-transposed path, threadgroup
swizzle for L2 reuse (silarray's STEEL design) — is deferred to a dedicated
kernel-census sprint against the PyTorch-MPS gate. Do NOT hand-tune before
running the census; the current kernel is correct and that is what M3b-2
locked in. Elementwise/softmax/reductions are bandwidth-bound and already
near their ceiling.

## Refuted approaches — do not retry without new evidence

(none yet — add entries as they are refuted, with data)

Inherited from silarray (Metal; re-verify before assuming they transfer to
CUDA): spin-wait on command-buffer status (slower), split commits to overlap
CPU/GPU (slower), trace/replay compile (no payoff — graph build was 0.2% of
step time), AOT vs JIT kernel compilation (no effect on SGEMM).

## Bug classes worth remembering

- silarray's only two real correctness bugs were the same class: copy-pasted
  kernel bodies diverging between aligned and edge tile paths. Rule: kernel
  bodies are shared templates; every fused-epilogue kernel gets edge-shape
  numerical tests (M-edge, N-edge, band boundaries).
