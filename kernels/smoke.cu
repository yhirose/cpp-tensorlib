// Build-check kernel: proves the CI CUDA toolchain (nvcc + host compiler on
// Linux/Windows) compiles and links our kernel sources. Never launched on
// GitHub-hosted runners — they have no NVIDIA driver; execution is covered
// by the cuda-gpu job on a GPU runner.
__global__ void tensorlib_smoke_(float* p) {
  if (p) p[0] = 1.0f;
}
