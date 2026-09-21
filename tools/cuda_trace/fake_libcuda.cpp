// A stand-in libcuda.so.1 that records what the CUDA backend asks of the
// driver instead of doing it. cuda.h dlopens the driver and looks kernels up
// by name, so putting this library first on LD_LIBRARY_PATH is the whole
// installation: nothing in the backend knows it is being watched.
//
// "Device" memory is host memory, copies are memcpy, and a launch runs no
// kernel — it writes one trace line. What a refactor of the backend's host
// side must preserve is that the same kernels get the same arguments in the
// same order, and that is exactly what the trace holds; whether the kernels
// compute the right thing is the GPU suite's question, not this one's.
//
// Addresses never appear in the trace: a pointer prints as #N+off, the N-th
// allocation and a byte offset into it, so two runs compare with diff.
//
// TL_CUDA_TRACE names the output file (default: stderr).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace {

struct kernel {
  const char* name;
  const char* sig;  // P pointer, u 4-byte integer, f float
};
const kernel kKernels[] = {
#include "kernel_sigs.inc"
};

struct block {
  size_t bytes;
  int id;
};
std::map<uintptr_t, block> g_live;  // by base address
int g_next_id = 0;
long g_unresolved = 0;

FILE* out() {
  static FILE* f = [] {
    const char* path = std::getenv("TL_CUDA_TRACE");
    FILE* file = path ? std::fopen(path, "w") : nullptr;
    return file ? file : stderr;
  }();
  return f;
}

// "#N+off" for an address inside (or one past) a live allocation.
std::string where(uintptr_t p) {
  if (!p) return "null";
  auto it = g_live.upper_bound(p);
  if (it != g_live.begin()) {
    --it;
    if (p <= it->first + it->second.bytes) {
      return "#" + std::to_string(it->second.id) + "+" +
             std::to_string(p - it->first);
    }
  }
  g_unresolved++;
  return "UNRESOLVED";
}

struct at_exit {
  ~at_exit() {
    std::fprintf(out(), "end allocations=%d unresolved=%ld\n", g_next_id,
                 g_unresolved);
    std::fflush(out());
  }
} g_at_exit;

}  // namespace

extern "C" {

using CUresult = int;
using CUdeviceptr = unsigned long long;

CUresult cuInit(unsigned) { return 0; }
CUresult cuDeviceGetCount(int* n) { return *n = 1, 0; }
CUresult cuDeviceGet(int* dev, int) { return *dev = 0, 0; }
CUresult cuDevicePrimaryCtxRetain(void** ctx, int) {
  static int the_context;
  return *ctx = &the_context, 0;
}
CUresult cuCtxSetCurrent(void*) { return 0; }
CUresult cuCtxSynchronize() { return std::fprintf(out(), "sync\n"), 0; }

CUresult cuModuleLoadData(void** mod, const void*) {
  static int the_module;
  return *mod = &the_module, 0;
}
CUresult cuModuleGetFunction(void** fn, void*, const char* name) {
  for (const kernel& k : kKernels) {
    if (!std::strcmp(k.name, name)) {
      return *fn = const_cast<void*>(static_cast<const void*>(&k)), 0;
    }
  }
  std::fprintf(out(), "nofunc %s\n", name);
  return *fn = nullptr, 500;  // CUDA_ERROR_NOT_FOUND
}

CUresult cuLaunchKernel(void* fn, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz, unsigned smem,
                        void* /*stream*/, void** argv, void**) {
  const auto* k = static_cast<const kernel*>(fn);
  std::fprintf(out(), "launch %s grid=%u,%u,%u block=%u,%u,%u smem=%u", k->name,
               gx, gy, gz, bx, by, bz, smem);
  for (size_t i = 0; k->sig[i]; i++) {
    if (k->sig[i] == 'P') {
      uintptr_t p;
      std::memcpy(&p, argv[i], sizeof p);
      std::fprintf(out(), " %s", where(p).c_str());
    } else if (k->sig[i] == 'u') {
      uint32_t v;
      std::memcpy(&v, argv[i], 4);
      std::fprintf(out(), " u:%u", v);
    } else {
      float v;
      std::memcpy(&v, argv[i], 4);
      std::fprintf(out(), " f:%.9g", static_cast<double>(v));
    }
  }
  std::fputc('\n', out());
  return 0;
}

CUresult cuMemAlloc_v2(CUdeviceptr* p, size_t bytes) {
  void* mem = std::calloc(bytes ? bytes : 1, 1);
  if (!mem) return 2;  // CUDA_ERROR_OUT_OF_MEMORY
  g_live[reinterpret_cast<uintptr_t>(mem)] = {bytes, g_next_id};
  std::fprintf(out(), "alloc #%d %zu\n", g_next_id++, bytes);
  return *p = reinterpret_cast<uintptr_t>(mem), 0;
}
CUresult cuMemFree_v2(CUdeviceptr p) {
  auto it = g_live.find(static_cast<uintptr_t>(p));
  if (it == g_live.end()) return 1;
  std::fprintf(out(), "free #%d\n", it->second.id);
  g_live.erase(it);
  std::free(reinterpret_cast<void*>(static_cast<uintptr_t>(p)));
  return 0;
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t bytes) {
  std::fprintf(out(), "h2d %s %zu\n", where(dst).c_str(), bytes);
  std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), src, bytes);
  return 0;
}
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t bytes,
                              void*) {
  std::fprintf(out(), "h2d_async %s %zu\n", where(dst).c_str(), bytes);
  std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), src, bytes);
  return 0;
}
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t bytes) {
  std::fprintf(out(), "d2h %s %zu\n", where(src).c_str(), bytes);
  std::memcpy(dst, reinterpret_cast<void*>(static_cast<uintptr_t>(src)), bytes);
  return 0;
}
CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char v, size_t bytes) {
  std::fprintf(out(), "memset %s %zu %u\n", where(dst).c_str(), bytes, v);
  std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), v, bytes);
  return 0;
}
CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char v, size_t bytes, void*) {
  std::fprintf(out(), "memset_async %s %zu %u\n", where(dst).c_str(), bytes, v);
  std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), v, bytes);
  return 0;
}

// Streams and graph capture: enough for the capture group to take its real
// path. A captured region's launches are traced as they are recorded; a replay
// is one line.
CUresult cuStreamCreate(void** s, unsigned) {
  return *s = std::malloc(1), std::fprintf(out(), "stream_create\n"), 0;
}
CUresult cuStreamDestroy_v2(void* s) { return std::free(s), 0; }
CUresult cuStreamSynchronize(void*) {
  return std::fprintf(out(), "stream_sync\n"), 0;
}
CUresult cuStreamBeginCapture_v2(void*, int mode) {
  return std::fprintf(out(), "capture_begin mode=%d\n", mode), 0;
}
CUresult cuStreamEndCapture(void*, void** graph) {
  return *graph = std::malloc(1), std::fprintf(out(), "capture_end\n"), 0;
}
CUresult cuGraphInstantiateWithFlags(void** exec, void*, unsigned long long) {
  return *exec = std::malloc(1), std::fprintf(out(), "graph_instantiate\n"), 0;
}
CUresult cuGraphLaunch(void*, void*) {
  return std::fprintf(out(), "graph_launch\n"), 0;
}
CUresult cuGraphExecDestroy(void* e) { return std::free(e), 0; }
CUresult cuGraphDestroy(void* g) { return std::free(g), 0; }

}  // extern "C"
