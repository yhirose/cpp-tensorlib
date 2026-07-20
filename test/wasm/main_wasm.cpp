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
// mode: 0 = cpu, 1 = gpu, 2 = auto. auto is here because the TENSORLIB_WEBGPU
// arm of auto_threshold_() (types.h) is unreachable from native ctest, so this
// is the only place its size-based routing gets exercised against the oracle.
EMSCRIPTEN_KEEPALIVE int run_tests(int mode) {
  const bool gpu_mode = mode != 0;
  const char* name = mode == 0 ? "cpu" : (mode == 1 ? "gpu" : "auto");
  if (mode == 0) {
    tl::use_cpu();
  } else if (mode == 1) {
    tl::use_gpu();
  } else {
    tl::use_auto();
  }
  std::printf("[tensorlib_wasm] mode=%s gpu_available=%d\n", name,
              tl::gpu_available() ? 1 : 0);

  // gpu_available() false in gpu mode means the device or JSPI is missing and
  // every GPU assertion would trivially pass — report that rather than a green
  // run that proved nothing.
  if (gpu_mode && !tl::gpu_available()) {
    std::printf("[tensorlib_wasm] FAIL: gpu mode requested but no device\n");
    return 0;
  }

  // dispatch_counts is cumulative for the life of the module, so snapshot it:
  // with more than one mode run per page load, the raw totals would credit
  // this run with the previous one's dispatches.
  auto before = tl::webgpu::context::get().dispatch_counts;

  doctest::Context ctx;
  int failed = ctx.run();

  // A green suite does not prove the backend ran: an op the backend declines
  // falls to CPU and its assertions pass either way. So the dispatch census is
  // not just reported, it is asserted — every family that has a kernel must
  // have dispatched at least once. Without this the whole suite stays green
  // when the WGSL fails to compile, which is precisely the Phase 3 failure
  // mode (a bad shader module makes every kernel decline, silently).
  //
  // gpu mode only. auto routes by size, so a family legitimately reaching zero
  // there is a threshold decision, not a regression.
  static const char* kFamilies[] = {"sgemm",    "ew_unary", "ew_binary",
                                    "ew_bcast", "softmax",  "row_reduce"};
  auto& counts = tl::webgpu::context::get().dispatch_counts;
  for (auto& kv : counts) {
    auto it = before.find(kv.first);
    long n = kv.second - (it == before.end() ? 0 : it->second);
    if (n) std::printf("[tensorlib_wasm] dispatch %s=%ld\n", kv.first.c_str(), n);
  }
  if (mode == 1) {
    for (const char* f : kFamilies) {
      auto now = counts.find(f);
      auto was = before.find(f);
      long n = (now == counts.end() ? 0 : now->second) -
               (was == before.end() ? 0 : was->second);
      if (n <= 0) {
        std::printf("[tensorlib_wasm] FAIL: no %s dispatch — the backend "
                    "declined every op in this family\n", f);
        failed = 1;
      }
    }
  }

  std::printf("[tensorlib_wasm] %s %s\n", name, failed ? "FAIL" : "OK");
  return failed ? 0 : 1;
}

}  // extern "C"
