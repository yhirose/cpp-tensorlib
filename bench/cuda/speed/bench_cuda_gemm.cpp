// M6 stage-2 SGEMM census harness — the CUDA analogue of bench_cpu_gemm.cpp.
// Times own tl::cuda::gemm against cuBLAS (the gate reference, like OpenBLAS on
// the CPU side) and a naive-oracle correctness check, interleaved A/B in one
// run so GPU clock swings hit both sides equally (see performance-notes.md
// measurement discipline). GFLOP/s across a square-shape census.
//
// cuBLAS + the CUDA runtime are linked *only into this bench* — they are a
// measurement reference, never a library dependency (cuda.h still dlopen's the
// driver and ships zero third-party runtime deps). Managed pointers from
// cuMemAllocManaged double as device pointers, so cuBLAS reads them directly.
//
// Timing: CUDA events on the null stream (own kernels launch there via the
// driver API; cudart's default stream is the same null stream, so events
// serialize correctly across both). R launches per timed batch amortize fixed
// submission latency and give steady-state throughput. Measured on the RTX 3090
// under WSL2 — WSL2 adds submission latency, but events measure device-side, so
// the throughput ratio is WSL2-insensitive; wall-clock would not be.

#ifndef TENSORLIB_CUDA
#define TENSORLIB_CUDA
#endif
#include "cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace tl::cuda;

// GFLOP/s for a full m×n×k multiply-add.
static double gflops(int64_t m, int64_t n, int64_t k, double secs) {
  return 2.0 * m * n * k / secs / 1e9;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (!available()) {
    std::printf("no CUDA device — skipping bench\n");
    return 0;
  }

  // Time each shape with R fused launches between one event pair, ROUNDS times,
  // report the median GFLOP/s. Reduce R for the biggest shapes (they are slow).
  const int ROUNDS = 7;
  std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
  if (argc > 1) {  // optional: bench_cuda_gemm 1024 2048 ...
    sizes.clear();
    for (int i = 1; i < argc; i++) sizes.push_back(std::atoi(argv[i]));
  }

  cublasHandle_t cub = nullptr;
  cublasCreate(&cub);

  std::mt19937 g(11);
  std::uniform_real_distribution<float> dist(-1, 1);

  std::printf("%-7s %10s %10s %8s   %8s\n", "size", "own GF/s", "cuBLAS", "own/cuB",
              "maxrel");
  for (int S : sizes) {
    int64_t m = S, n = S, k = S;
    float* ca = nullptr;
    float* cb = nullptr;
    float* cc = nullptr;
    float* cref = nullptr;
    void* A = alloc(m * k * 4, &ca);
    void* B = alloc(k * n * 4, &cb);
    void* C = alloc(m * n * 4, &cc);
    void* Cref = alloc(m * n * 4, &cref);
    for (int64_t i = 0; i < m * k; i++) ca[i] = dist(g);
    for (int64_t i = 0; i < k * n; i++) cb[i] = dist(g);

    // native handles are the DEVICE pointers (device-mirror model). Both own
    // and cuBLAS operate on device memory. run_own's first call H2D-uploads the
    // host-filled A,B into their device buffers (device_read_), after which the
    // device copies stay resident — so cuBLAS reads the same operands.
    float* dA = (float*)A;
    float* dB = (float*)B;
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; for row-major C(m,n)=A(m,k)·B(k,n) compute the
    // transpose C^T(n,m)=B^T·A^T by feeding B,A swapped (the standard idiom):
    //   cublasSgemm(N,N, n,m,k, B(ldb=n), A(lda=k), C(ldc=n)).
    auto run_cublas = [&](void* out) {
      cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha,
                  dB, (int)n, dA, (int)k, &beta, (float*)out, (int)n);
    };
    auto run_own = [&](void* out) {
      gemm(A, 0, k, false, B, 0, n, false, out, 0, m, n, k, 1.0f, 0.0f);
    };

    // ---- correctness: own vs cuBLAS (cuBLAS is the trusted oracle here) ----
    run_own(C);        // also uploads A,B to device
    run_cublas(Cref);  // reads the same device A,B
    cudaDeviceSynchronize();
    std::vector<float> ownv(m * n), refv(m * n);
    cudaMemcpy(ownv.data(), C, m * n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(refv.data(), Cref, m * n * 4, cudaMemcpyDeviceToHost);
    double maxrel = 0;
    for (int64_t i = 0; i < m * n; i++) {
      double err = std::fabs(ownv[i] - refv[i]) / (1.0 + std::fabs(refv[i]));
      maxrel = std::max(maxrel, err);
    }

    // ---- timing: R launches per event pair, interleaved own/cuBLAS ----
    int R = S >= 4096 ? 4 : (S >= 2048 ? 8 : 30);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time_batch = [&](auto&& fn) {
      // warmup
      fn(C);
      cudaDeviceSynchronize();
      std::vector<double> gs;
      for (int r = 0; r < ROUNDS; r++) {
        cudaEventRecord(e0, 0);
        for (int i = 0; i < R; i++) fn(C);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        gs.push_back(gflops(m, n, k, (ms / 1e3) / R));
      }
      return median(gs);
    };
    double own_gf = time_batch(run_own);
    double cub_gf = time_batch(run_cublas);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    std::printf("%-7d %10.0f %10.0f %8.2f   %8.1e\n", S, own_gf, cub_gf,
                own_gf / cub_gf, maxrel);
    release(A, 0, nullptr);
    release(B, 0, nullptr);
    release(C, 0, nullptr);
    release(Cref, 0, nullptr);
  }
  cublasDestroy(cub);
  return 0;
}
