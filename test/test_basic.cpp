#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <tensorlib.h>

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  const char* mode = "cpu";
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--cpu")) {
      tl::use_cpu();
      mode = "cpu";
    } else if (!std::strcmp(argv[i], "--gpu")) {
      tl::use_gpu();
      mode = "gpu";
    } else if (!std::strcmp(argv[i], "--auto")) {
      tl::use_auto();
      mode = "auto";
    }
  }
  std::printf("[tensorlib_test] mode=%s gpu_available=%d\n", mode,
              tl::gpu_available() ? 1 : 0);

  doctest::Context ctx(argc, argv);
  return ctx.run();
}
