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
  enable_language(CUDA)
  # Build-size policy: SASS for the majors we ship + PTX fallback for newer
  # (see docs/performance-notes.md once it lands). 80 alone is fine for the
  # CI build check.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  add_library(tensorlib_cuda_kernels OBJECT kernels/smoke.cu)
endif()

enable_testing()

add_executable(tensorlib_test test/test_basic.cpp test/test_array.cpp)
target_link_libraries(tensorlib_test PRIVATE tensorlib)
if(TENSORLIB_CUDA)
  target_sources(tensorlib_test PRIVATE $<TARGET_OBJECTS:tensorlib_cuda_kernels>)
  target_compile_definitions(tensorlib_test PRIVATE TENSORLIB_CUDA_ENABLED)
endif()

add_executable(tensorlib_bench bench/bench_main.cpp)
target_link_libraries(tensorlib_bench PRIVATE tensorlib)

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
