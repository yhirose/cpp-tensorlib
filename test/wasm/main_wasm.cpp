// Browser test entry point for the WebGPU backend (M10, Phase 2).
//
// Native ctest cannot exercise this backend at all, so the same doctest suite
// is relinked for wasm and driven from a page. test_array.cpp compiles in
// unchanged: its matches_gpu_oracle helper (test/test_array.cpp:53-67) already
// runs a graph in gpu mode and again through the ref oracle, is
// backend-agnostic, and self-skips on !gpu_available() — so it exercises
// WebGPU exactly as it exercises Metal, with no WebGPU-specific assertions to
// keep in sync.
//
// Ops the backend has not ported yet return false and fall to CPU, so the
// whole suite is expected green from Phase 1 onward; what changes per phase is
// how much of it actually ran on the GPU.
//
// The verdict goes to the console as a single OK/FAIL line so a
// browser-automation check can read it without scraping the page.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <emscripten.h>
#include <tensorlib.h>

#include <cstdio>

extern "C" {

// Called from JS with {async: true}: under JSPI this returns a promise,
// because any GPU wait beneath it suspends. Nothing in C++ is async.
EMSCRIPTEN_KEEPALIVE int run_tests(int gpu_mode) {
  if (gpu_mode) {
    tl::use_gpu();
  } else {
    tl::use_cpu();
  }
  std::printf("[tensorlib_wasm] mode=%s gpu_available=%d\n",
              gpu_mode ? "gpu" : "cpu", tl::gpu_available() ? 1 : 0);

  // gpu_available() false in gpu mode means the device or JSPI is missing and
  // every GPU assertion would trivially pass — report that rather than a green
  // run that proved nothing.
  if (gpu_mode && !tl::gpu_available()) {
    std::printf("[tensorlib_wasm] FAIL: gpu mode requested but no device\n");
    return 0;
  }

  doctest::Context ctx;
  int failed = ctx.run();
  // A green suite does not prove the backend ran — unported ops fall back to
  // CPU and pass either way. Report which kernels actually dispatched.
  for (auto& kv : tl::webgpu::context::get().dispatch_counts)
    std::printf("[tensorlib_wasm] dispatch %s=%ld\n", kv.first.c_str(), kv.second);
  std::printf("[tensorlib_wasm] %s\n", failed ? "FAIL" : "OK");
  return failed ? 0 : 1;
}

}  // extern "C"
