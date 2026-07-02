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

Sprint history for Metal SGEMM at 2048³ (all measured under load avg 5–9,
so treat as lower bounds):

| Step | 2048³ | 1024³ |
|---|---|---|
| M3b-2 basic 32×32×16 tile, scalar loads | ~1300 GFLOP/s | ~1080 |
| + float4 fast loaders (host-gated) | ~2170 | ~1060 |
| + STEEL port (frag registers, swizzle, split edge loops) | **~3200** | **~1700–1960** |

Reference points: Accelerate (AMX) sgemm ≈ 980 GFLOP/s at 1024³ and stays
~constant; M1 Pro GPU fp32 ceiling ≈ 4–5 TFLOP/s; MLX lands ≈ 3.5–4. The
STEEL numbers were taken on a loaded machine — re-measure quiet before
comparing against PyTorch-MPS for the gate.

**Tile-config census (2026-07-02, loaded machine, simple-tile family):**
32×32 ≈ 64×32 ≈ 32×64 all plateau at ~2000–2200 GFLOP/s at 2048³ — the
bottleneck was the loader/pipeline structure, not tile size. 64×32 had the
best small-size behavior and is the non-STEEL fallback.

**Remaining SGEMM work:**
- Transposed operands still use the simple-tile family (STEEL is NN-only
  until SteelLoaderT is ported) — backward-pass matmuls (`xᵀ@g`, `g@Wᵀ`)
  are the motivating shapes.
- Function-constant specialization (fc_mn_aligned / fc_k_aligned) not yet
  ported; runtime branches instead. silarray measured conv2d FCs equivalent,
  but gemm may differ — measure before porting.
- Narrow-N band (n < 48) and small shapes stay CPU-bound → auto-mode
  thresholds are the fix, not kernel work.

## Refuted approaches — do not retry without new evidence

| Approach | Verdict | Data |
|---|---|---|
| Naive 64×64 tile via the simple `sgemm_body_` template (full `simdgroup_matrix` locals, 16 accumulators) | **Catastrophic** | 133 GFLOP/s at 2048³ vs 2175 for 32×32 — the "16-accumulator occupancy crisis" silarray hit; STEEL's explicit float2 fragment registers are what make 16 accumulators viable |
| Host-side alignment gating of float4 loaders (require ld%4==0, 16B base) | **Unnecessary** | silarray/MLX issue vector loads at any alignment on Apple GPUs in practice (bench shapes with ldb=50); gate removed |

Inherited from silarray (Metal; re-verify before assuming they transfer to
CUDA): spin-wait on command-buffer status (slower), split commits to overlap
CPU/GPU (slower), trace/replay compile (no payoff — graph build was 0.2% of
step time), AOT vs JIT kernel compilation (no effect on SGEMM).

## Bug classes worth remembering

- silarray's only two real correctness bugs were the same class: copy-pasted
  kernel bodies diverging between aligned and edge tile paths. Rule: kernel
  bodies are shared templates; every fused-epilogue kernel gets edge-shape
  numerical tests (M-edge, N-edge, band boundaries).
