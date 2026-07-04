cmake_minimum_required(VERSION 3.20)
project(tensorlib LANGUAGES CXX)

# This project's own build (tests/benches) compiles at C++23. The public
# headers, however, only require C++17 (inline variables, std::optional,
# structured bindings — nothing from C++20/23); consumers may include them at
# C++17+. The INTERFACE target does NOT force a standard on consumers. The one
# C++23-ish dependency, metal.h's #embed, is macOS-only and unreached elsewhere.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TENSORLIB_CUDA "Build the CUDA backend (needs CUDA Toolkit at build time only)" OFF)

add_library(tensorlib INTERFACE)
target_include_directories(tensorlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(APPLE)
  target_link_libraries(tensorlib INTERFACE
    "-framework Accelerate" "-framework Metal" "-framework Foundation")
endif()

if(TENSORLIB_CUDA)
  # The CUDA backend does NOT link CUDA — the driver is dlopen'd (cuda.h) and
  # kernels run as PTX JIT'd by the driver. So nvcc is used only to compile
  # kernels/tensorlib_cuda.cu to PTX, which bin2c turns into a byte array the
  # header #includes (off-Apple compilers predate C23 #embed). enable_language
  # locates nvcc (CMAKE_CUDA_COMPILER); nothing links against libcudart.
  enable_language(CUDA)
  # PTX virtual arch: compute_80 PTX JITs onto any sm_80+ device (the RTX 3090
  # is sm_86), so one PTX blob is portable across recent GPUs.
  if(NOT TENSORLIB_CUDA_PTX_ARCH)
    set(TENSORLIB_CUDA_PTX_ARCH sm_80)
  endif()
  set(TL_CUDA_PTX ${CMAKE_BINARY_DIR}/tensorlib_cuda.ptx)
  set(TL_CUDA_INC ${CMAKE_BINARY_DIR}/tensorlib_cuda_ptx.inc)
  add_custom_command(
    OUTPUT ${TL_CUDA_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER} -arch=${TENSORLIB_CUDA_PTX_ARCH} -ptx
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/tensorlib_cuda.cu -o ${TL_CUDA_PTX}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/tensorlib_cuda.cu
    COMMENT "nvcc: tensorlib_cuda.cu -> PTX (${TENSORLIB_CUDA_PTX_ARCH})"
    VERBATIM)
  add_custom_command(
    OUTPUT ${TL_CUDA_INC}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${TL_CUDA_PTX} -DOUTPUT=${TL_CUDA_INC}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c.cmake
    DEPENDS ${TL_CUDA_PTX}
    COMMENT "bin2c: PTX -> tensorlib_cuda_ptx.inc"
    VERBATIM)
  add_custom_target(tensorlib_cuda_ptx DEPENDS ${TL_CUDA_INC})
  # Everything a CUDA-enabled consumer target needs: the generated .inc on the
  # include path, the TENSORLIB_CUDA switch, and libdl for dlopen.
  function(tensorlib_enable_cuda tgt)
    add_dependencies(${tgt} tensorlib_cuda_ptx)
    target_include_directories(${tgt} PRIVATE ${CMAKE_BINARY_DIR})
    target_compile_definitions(${tgt} PRIVATE TENSORLIB_CUDA)
    target_link_libraries(${tgt} PRIVATE ${CMAKE_DL_LIBS})
  endfunction()
endif()

enable_testing()

add_executable(tensorlib_test test/test_basic.cpp test/test_array.cpp)
target_link_libraries(tensorlib_test PRIVATE tensorlib)
if(TENSORLIB_CUDA)
  # The test suite's gpu/auto modes now exercise the real CUDA backend (the
  # gpu:: facade routes to cuda off-Apple), falling back to CPU with no device.
  tensorlib_enable_cuda(tensorlib_test)

  # Kernel-level CUDA oracle (like check_cpu_ukernel for the CPU backend):
  # every op vs a CPU reference, direct on the cuda:: API. Only built/run when
  # CUDA is enabled.
  add_executable(tensorlib_check_cuda bench/check_cuda.cpp)
  target_link_libraries(tensorlib_check_cuda PRIVATE tensorlib)
  tensorlib_enable_cuda(tensorlib_check_cuda)
  add_test(NAME cuda_ukernel COMMAND tensorlib_check_cuda)

  # Real-MNIST convergence gate — asserts the GPU op stack (hand-rolled backprop)
  # drives a 784-256-10 ReLU MLP to a real test-accuracy target. Skips-as-pass
  # with no GPU (CI) or no data (offline); fetches MNIST once via curl into the
  # build dir. A few seconds on-device.
  add_executable(tensorlib_check_mnist bench/check_mnist.cpp)
  target_link_libraries(tensorlib_check_mnist PRIVATE tensorlib)
  tensorlib_enable_cuda(tensorlib_check_mnist)
  add_test(NAME mnist_converge COMMAND tensorlib_check_mnist
           ${CMAKE_BINARY_DIR}/mnist-data)

  # CUDA SGEMM census (M6 stage-2 tuning). Links cuBLAS + the CUDA runtime as a
  # measurement REFERENCE only (the gate is >=90% cuBLAS) — never a library dep;
  # cuda.h still dlopen's the driver and ships zero third-party runtime deps.
  # Not a ctest (it's a perf bench, and cuBLAS may be absent on CI). Only built
  # when the CUDA Toolkit's cuBLAS is found.
  find_library(CUBLAS_LIB cublas
    PATHS ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64
          ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/targets/x86_64-linux/lib
          /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
  find_library(CUDART_LIB cudart
    PATHS ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64
          ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/targets/x86_64-linux/lib
          /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
  find_path(CUDA_TK_INC cublas_v2.h
    PATHS ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/include
          ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/targets/x86_64-linux/include
          /usr/local/cuda/include)
  if(CUBLAS_LIB AND CUDART_LIB AND CUDA_TK_INC)
    add_executable(tensorlib_bench_cuda bench/bench_cuda_gemm.cpp)
    target_link_libraries(tensorlib_bench_cuda PRIVATE tensorlib
      ${CUBLAS_LIB} ${CUDART_LIB})
    # The project include MUST come before the CUDA Toolkit include: the bench's
    # #include "cuda.h" means the project header, but the CUDA TK ships its own
    # cuda.h (the driver API) — BEFORE prepends so ours wins the quote-include.
    target_include_directories(tensorlib_bench_cuda BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(tensorlib_bench_cuda PRIVATE ${CUDA_TK_INC})
    tensorlib_enable_cuda(tensorlib_bench_cuda)
    message(STATUS "cuda bench: cuBLAS gate enabled (${CUBLAS_LIB})")
  else()
    message(STATUS "cuda bench: cuBLAS not found — SGEMM gate bench skipped")
  endif()
endif()

add_executable(tensorlib_bench bench/bench_main.cpp)
target_link_libraries(tensorlib_bench PRIVATE tensorlib)

# auto-mode CPU/GPU crossover census (misc/census.cpp). Device-agnostic array
# API, so the gpu:: facade routes it to Metal on Apple and CUDA off-Apple —
# tensorlib_enable_cuda below embeds the PTX so gpu_available() is true.
add_executable(tensorlib_census misc/census.cpp)
target_link_libraries(tensorlib_census PRIVATE tensorlib)

# Local-LLM shape bench — the dev harness for M7-M9 (decode/prefill GEMM +
# attention at real decoder-layer shapes). Same backend-agnostic array API.
add_executable(tensorlib_bench_llm bench/bench_llm.cpp)
target_link_libraries(tensorlib_bench_llm PRIVATE tensorlib)

# MNIST-shaped MLP training bench (synthetic data, hand-rolled backprop) — a
# real end-to-end workload (loss down / acc up) exercising the full training op
# mix on the GPU. Fixed-step throughput, sub-second; no data download.
add_executable(tensorlib_bench_mlp bench/bench_mlp_train.cpp)
target_link_libraries(tensorlib_bench_mlp PRIVATE tensorlib)

# These benches drive the GPU through the backend-agnostic eval seam; on a CUDA
# build they must carry the embedded PTX + TENSORLIB_CUDA define, same as the
# tests, or gpu_available() stays false and every GPU section silently skips.
if(TENSORLIB_CUDA)
  tensorlib_enable_cuda(tensorlib_bench)
  tensorlib_enable_cuda(tensorlib_census)
  tensorlib_enable_cuda(tensorlib_bench_llm)
  tensorlib_enable_cuda(tensorlib_bench_mlp)
endif()

# Own CPU GEMM bench (M5 tuning). Links OpenBLAS for the off-Apple gate when
# present (ARM OpenBLAS = same NEON ISA); own-vs-ref only otherwise.
add_executable(tensorlib_bench_cpu bench/bench_cpu_gemm.cpp)
target_link_libraries(tensorlib_bench_cpu PRIVATE tensorlib)
find_library(OPENBLAS_LIB openblas
  PATHS /opt/homebrew/opt/openblas/lib /usr/local/opt/openblas/lib)
if(OPENBLAS_LIB)
  target_compile_definitions(tensorlib_bench_cpu PRIVATE BENCH_HAS_OPENBLAS)
  target_link_libraries(tensorlib_bench_cpu PRIVATE ${OPENBLAS_LIB})
  find_path(OPENBLAS_INC cblas.h
    PATHS /opt/homebrew/opt/openblas/include /usr/local/opt/openblas/include)
  if(OPENBLAS_INC)
    target_include_directories(tensorlib_bench_cpu PRIVATE ${OPENBLAS_INC})
  endif()
  message(STATUS "cpu bench: OpenBLAS gate enabled (${OPENBLAS_LIB})")
endif()

# The same binary runs in three device modes (silarray convention).
add_test(NAME cpu  COMMAND tensorlib_test --cpu)
add_test(NAME gpu  COMMAND tensorlib_test --gpu)
add_test(NAME auto COMMAND tensorlib_test --auto)
