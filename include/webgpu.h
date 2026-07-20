#pragma once

// Own WebGPU backend (M10) — the browser GPU backend, mirroring metal.h and
// cuda.h. Kernels are hand-written WGSL (kernels/tensorlib_webgpu.wgsl),
// compiled at first use from a committed C-string .inc. No vendor library.
//
// Memory: a persistent host/device MIRROR per allocation, copied wholesale
// from cuda.h. A WebGPU storage buffer has no CPU-dereferenceable pointer, but
// gpu::alloc(bytes, float** contents) must hand one back (storage.h keeps it as
// storage::ptr, and every ref::/accel::/cpu:: path then treats it as ordinary
// memory). So each allocation is a device buffer (`native`, what kernels bind)
// paired with a malloc'd host buffer (`contents`), and a per-allocation dirty
// state drives lazy copies: H2D before a kernel reads a host-dirty buffer, D2H
// before the CPU reads a device-dirty one (array::raw()/data() →
// gpu::sync_to_host). This is why array.h and storage.h need no changes.
//
// Async: flush() and sync_to_host() keep their SYNCHRONOUS signatures. The
// instance is created with the TimedWaitAny feature, which makes
// wgpuInstanceWaitAny(timeout > 0) a legal blocking wait; emdawnwebgpu
// implements it by suspending through Asyncify/JSPI. The suspend surface is
// exactly two call sites — OnSubmittedWorkDone and MapAsync. Uploads
// (WriteBuffer) are queued, never awaited. See spike/webgpu/README.md.
//
// Batching: one command encoder accumulates dispatches and flush() submits and
// waits once, as metal.h does. This matters more here than on Metal — a
// dispatch+wait has a ~0.6-1.0 ms fixed floor in the browser, so a flush per op
// would be dominated by it.
//
// Real implementation is gated on TENSORLIB_WEBGPU && __EMSCRIPTEN__; a plain
// build gets the stubs below. If the link lacks JSPI, or JS handed in no
// device, CreateInstance/device acquisition fails and available() stays false —
// every op then routes to CPU, which is also the Safari fallback.

#include <cstdint>

#include "metal.h"  // reuse tl::metal::kop (platform-independent op enum)
#include "types.h"

namespace tl {
namespace webgpu {

using kop = tl::metal::kop;

#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)

}  // namespace webgpu
}  // namespace tl

// emdawnwebgpu declares emscripten_webgpu_get_device() in webgpu.h itself,
// not in emscripten/html5_webgpu.h (that is the old built-in binding's home).
#include <webgpu/webgpu_cpp.h>

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace tl {
namespace webgpu {

inline const char* wgsl_source_() {
  static const char* src =
#include "tensorlib_webgpu_wgsl.inc"
      ;
  return src;
}

// Uniform params, laid out to match the WGSL Params struct (16 x 4 bytes).
struct gemm_params {
  uint32_t M, N, K;
  uint32_t lda, ldb, ldc;
  uint32_t a_off, b_off, c_off;
  uint32_t ta, tb;
  float scale, offset;
  uint32_t pad0, pad1, pad2;
};

// Dynamic uniform offsets must be a multiple of the adapter's
// minUniformBufferOffsetAlignment; 256 is the spec's guaranteed-safe maximum.
constexpr uint64_t kUniformSlot = 256;
constexpr uint32_t kUniformSlots = 256;  // one flush can batch this many

struct context {
  wgpu::Instance instance;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::BindGroupLayout bgl;
  wgpu::ComputePipeline sgemm;
  wgpu::Buffer uniforms;  // ring of kUniformSlots x kUniformSlot bytes
  bool ready = false;
  bool pending = false;

  // The open encoder. Created lazily on the first dispatch after a flush, so
  // an idle backend submits nothing.
  wgpu::CommandEncoder enc;
  uint32_t slot = 0;  // next free uniform ring slot

  // Host/device mirror per allocation, keyed by the opaque handle alloc()
  // returns as `native`. Views sharing a storage share the key, so one dirty
  // state serves every view. `where` tracks which copy is live.
  enum loc { HOST, DEVICE, BOTH };
  struct mirror {
    float* host = nullptr;   // CPU-side buffer (storage.ptr)
    wgpu::Buffer dev;        // device buffer
    size_t bytes = 0;
    loc where = HOST;
  };
  std::unordered_map<void*, mirror> mirrors;

  // Size-keyed free lists (like Metal's MTLBuffer pool and CUDA's). Repeated
  // alloc/free of identical shapes is the common case, and per-dispatch
  // allocation would compound the fixed dispatch floor.
  std::unordered_map<size_t, std::vector<std::pair<wgpu::Buffer, float*>>> pool;
  std::unordered_map<size_t, std::vector<wgpu::Buffer>> staging_pool;

  static context& get() {
    static auto* c = new context();  // leaked: outlives all storage deleters
    return *c;
  }

  context() {
    // TimedWaitAny is what makes WaitAny(timeout > 0) legal. emdawnwebgpu
    // refuses to create the instance without JSPI/Asyncify, so a link missing
    // it fails loudly here rather than deadlocking later.
    wgpu::InstanceFeatureName features[] = {
        wgpu::InstanceFeatureName::TimedWaitAny};
    wgpu::InstanceDescriptor idesc = {};
    idesc.requiredFeatureCount = 1;
    idesc.requiredFeatures = features;
    instance = wgpu::CreateInstance(&idesc);
    if (!instance) return;

    // No adapter/device round-trip on this side: JS already did it and passed
    // the result as Module.preinitializedWebGPUDevice.
    device = wgpu::Device::Acquire(emscripten_webgpu_get_device());
    if (!device) return;
    queue = device.GetQueue();

    wgpu::ShaderSourceWGSL wgsl = {};
    wgsl.code = wgsl_source_();
    wgpu::ShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl;
    wgpu::ShaderModule mod = device.CreateShaderModule(&smd);
    if (!mod) return;

    // Explicit layout rather than GetBindGroupLayout(0): the auto-generated
    // one has no dynamic offset on the uniform binding, which the ring needs.
    wgpu::BindGroupLayoutEntry be[4] = {};
    for (int i = 0; i < 3; ++i) {
      be[i].binding = i;
      be[i].visibility = wgpu::ShaderStage::Compute;
      be[i].buffer.type = i == 2 ? wgpu::BufferBindingType::Storage
                                 : wgpu::BufferBindingType::ReadOnlyStorage;
    }
    be[3].binding = 3;
    be[3].visibility = wgpu::ShaderStage::Compute;
    be[3].buffer.type = wgpu::BufferBindingType::Uniform;
    be[3].buffer.hasDynamicOffset = true;
    be[3].buffer.minBindingSize = sizeof(gemm_params);
    wgpu::BindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 4;
    bgld.entries = be;
    bgl = device.CreateBindGroupLayout(&bgld);

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pl = device.CreatePipelineLayout(&pld);

    wgpu::ComputePipelineDescriptor pd = {};
    pd.layout = pl;
    pd.compute.module = mod;
    pd.compute.entryPoint = "sgemm";
    sgemm = device.CreateComputePipeline(&pd);
    if (!sgemm) return;

    wgpu::BufferDescriptor ud = {};
    ud.size = kUniformSlot * kUniformSlots;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = device.CreateBuffer(&ud);
    if (!uniforms) return;

    ready = true;
  }

  // Blocking wait on a single future — the one place anything suspends.
  bool wait(wgpu::Future f) {
    return instance.WaitAny(f, UINT64_MAX) == wgpu::WaitStatus::Success;
  }

  mirror* mirror_(void* native) {
    auto it = mirrors.find(native);
    return it == mirrors.end() ? nullptr : &it->second;
  }

  // A kernel is about to READ this buffer: ensure the device copy is current.
  //
  // WriteBuffer executes in queue order, i.e. ahead of anything still sitting
  // in the unsubmitted encoder. That is safe precisely because a buffer in
  // HOST state has no encoded command touching it: a pending kernel write
  // would have set DEVICE, and a pending kernel read would have come through
  // here and set BOTH.
  void device_read_(void* native) {
    mirror* m = mirror_(native);
    if (m && m->where == HOST) {
      queue.WriteBuffer(m->dev, 0, m->host, m->bytes);
      m->where = BOTH;
    }
  }

  // A kernel is about to WRITE this buffer: it becomes the live copy.
  void device_write_(void* native) {
    if (mirror* m = mirror_(native)) m->where = DEVICE;
  }

  wgpu::Buffer staging_(size_t bytes) {
    auto it = staging_pool.find(bytes);
    if (it != staging_pool.end() && !it->second.empty()) {
      wgpu::Buffer b = it->second.back();
      it->second.pop_back();
      return b;
    }
    wgpu::BufferDescriptor d = {};
    d.size = bytes;
    d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    return device.CreateBuffer(&d);
  }
};

inline bool available() { return context::get().ready; }
inline bool pending() { return context::get().pending; }

// End the batch: submit the accumulated encoder and block until the GPU
// finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  c.pending = false;
  c.slot = 0;
  if (!c.enc) return;
  wgpu::CommandBuffer cmds = c.enc.Finish();
  c.enc = nullptr;
  c.queue.Submit(1, &cmds);
  c.wait(c.queue.OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}));
}

// Mirror allocation: a device buffer paired with a host buffer (returned via
// `contents`). They are DISTINCT memory — the dirty state copies between them
// on demand. The returned handle is an opaque token, not a pointer to
// anything dereferenceable; it is only ever a key back into `mirrors`.
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  size_t nb = bytes > 0 ? (size_t)bytes : 4;
  nb = (nb + 3) & ~size_t(3);  // WebGPU buffer sizes must be 4-byte multiples

  wgpu::Buffer dev;
  float* host = nullptr;
  auto it = c.pool.find(nb);  // reuse a recycled buffer of this exact size
  if (it != c.pool.end() && !it->second.empty()) {
    dev = it->second.back().first;
    host = it->second.back().second;
    it->second.pop_back();
  } else {
    wgpu::BufferDescriptor d = {};
    d.size = nb;
    d.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
              wgpu::BufferUsage::CopySrc;
    dev = c.device.CreateBuffer(&d);
    if (!dev) return nullptr;
    host = static_cast<float*>(std::malloc(nb));
    if (!host) return nullptr;
  }

  // The token has to be unique and stable for the allocation's lifetime; the
  // host pointer is both, and malloc will not hand out the same address twice
  // while it is live.
  void* token = host;
  c.mirrors[token] = context::mirror{host, dev, nb, context::HOST};
  if (contents) *contents = host;
  return token;
}

inline void release(void* buf, int64_t, float*) {
  auto& c = context::get();
  if (!c.ready || !buf) return;
  auto it = c.mirrors.find(buf);
  if (it == c.mirrors.end()) return;
  c.pool[it->second.bytes].push_back({it->second.dev, it->second.host});
  c.mirrors.erase(it);
}

// Reconcile a buffer for a CPU access: flush pending kernels, then D2H if the
// device holds the live copy. for_write invalidates the device copy (the host
// is about to mutate it). No-op for heap storages / unknown handles.
inline void sync_to_host(void* native, bool for_write) {
  auto& c = context::get();
  if (!c.ready || !native) return;
  context::mirror* m = c.mirror_(native);
  if (!m) return;
  if (c.pending) flush();
  if (m->where == context::DEVICE) {
    // No CPU-visible pointer to read from: copy device -> a MapRead staging
    // buffer, map it (the second and last suspend point), memcpy out.
    wgpu::Buffer stg = c.staging_(m->bytes);
    wgpu::CommandEncoder e = c.device.CreateCommandEncoder();
    e.CopyBufferToBuffer(m->dev, 0, stg, 0, m->bytes);
    wgpu::CommandBuffer cmds = e.Finish();
    c.queue.Submit(1, &cmds);

    bool ok = false;
    if (c.wait(stg.MapAsync(wgpu::MapMode::Read, 0, m->bytes,
                            wgpu::CallbackMode::WaitAnyOnly,
                            [&ok](wgpu::MapAsyncStatus s, wgpu::StringView) {
                              ok = (s == wgpu::MapAsyncStatus::Success);
                            })) &&
        ok) {
      if (const void* src = stg.GetConstMappedRange(0, m->bytes))
        std::memcpy(m->host, src, m->bytes);
      stg.Unmap();
      m->where = context::BOTH;
    }
    c.staging_pool[m->bytes].push_back(stg);
  }
  if (for_write) m->where = context::HOST;
}

// out = op(a) @ op(b) * scale + offset. Offsets are BYTE offsets into the
// storage; the kernel wants elements, so they must be 4-aligned (they always
// are for f32 views).
inline bool gemm(void* a, int64_t ao, int64_t lda, bool ta, void* b, int64_t bo,
                 int64_t ldb, bool tb, void* out, int64_t oo, int64_t m,
                 int64_t n, int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready) return false;
  if (m <= 0 || n <= 0 || k <= 0) return false;
  if ((ao % 4) || (bo % 4) || (oo % 4)) return false;
  // Every operand must be a tracked mirror: an untracked pointer is a heap
  // storage with no device buffer to bind, so this op belongs on the CPU.
  context::mirror* ma = c.mirror_(a);
  context::mirror* mb = c.mirror_(b);
  context::mirror* mo = c.mirror_(out);
  if (!ma || !mb || !mo) return false;
  if (c.slot >= kUniformSlots) flush();  // ring exhausted; start a new batch

  c.device_read_(a);
  c.device_read_(b);
  c.device_write_(out);

  gemm_params p = {};
  p.M = (uint32_t)m;
  p.N = (uint32_t)n;
  p.K = (uint32_t)k;
  p.lda = (uint32_t)lda;
  p.ldb = (uint32_t)ldb;
  p.ldc = (uint32_t)n;  // the eval seam always hands us a contiguous output
  p.a_off = (uint32_t)(ao / 4);
  p.b_off = (uint32_t)(bo / 4);
  p.c_off = (uint32_t)(oo / 4);
  p.ta = ta ? 1u : 0u;
  p.tb = tb ? 1u : 0u;
  p.scale = scale;
  p.offset = offset;

  const uint32_t off = c.slot++ * (uint32_t)kUniformSlot;
  c.queue.WriteBuffer(c.uniforms, off, &p, sizeof(p));

  wgpu::BindGroupEntry e[4] = {};
  e[0].binding = 0; e[0].buffer = ma->dev; e[0].size = ma->bytes;
  e[1].binding = 1; e[1].buffer = mb->dev; e[1].size = mb->bytes;
  e[2].binding = 2; e[2].buffer = mo->dev; e[2].size = mo->bytes;
  e[3].binding = 3; e[3].buffer = c.uniforms; e[3].size = sizeof(gemm_params);
  wgpu::BindGroupDescriptor bgd = {};
  bgd.layout = c.bgl;
  bgd.entryCount = 4;
  bgd.entries = e;
  wgpu::BindGroup bg = c.device.CreateBindGroup(&bgd);

  if (!c.enc) c.enc = c.device.CreateCommandEncoder();
  wgpu::ComputePassEncoder pass = c.enc.BeginComputePass();
  pass.SetPipeline(c.sgemm);
  pass.SetBindGroup(0, bg, 1, &off);
  pass.DispatchWorkgroups((uint32_t)((n + 63) / 64), (uint32_t)((m + 63) / 64),
                          1);
  pass.End();

  c.pending = true;
  return true;
}

// ---- Not ported yet (Phase 3+). Returning false routes the op to CPU, which
// is why each phase lands in a working state.
inline bool binary(kop, void*, int64_t, void*, int64_t, void*, int64_t, int64_t,
                   float, float) {
  return false;
}
inline bool binary_bcast(kop, void*, int64_t, int64_t, int64_t, void*, int64_t,
                         int64_t, int64_t, void*, int64_t, int64_t, int64_t,
                         float, float) {
  return false;
}
inline bool unary(kop, void*, int64_t, void*, int64_t, int64_t, float, float) {
  return false;
}
inline bool row_op(kop, void*, int64_t, void*, int64_t, int64_t, int64_t, float,
                   float) {
  return false;
}
inline bool gemv_f32(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool gemv_bf16(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool attn_decode(void*, void*, void*, void*, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float) {
  return false;
}
inline bool rope(void*, void*, int64_t, int64_t, int64_t, int64_t, float) {
  return false;
}
inline bool gemv_q4(void*, void*, void*, void*, int64_t, int64_t, int64_t) {
  return false;
}

#else  // !(TENSORLIB_WEBGPU && __EMSCRIPTEN__) — stubs, as in metal.h

inline bool available() { return false; }
inline bool pending() { return false; }
inline void flush() {}
inline void* alloc(int64_t, float**) { return nullptr; }
inline void release(void*, int64_t, float*) {}
inline void sync_to_host(void*, bool) {}
inline bool binary(kop, void*, int64_t, void*, int64_t, void*, int64_t, int64_t,
                   float, float) {
  return false;
}
inline bool binary_bcast(kop, void*, int64_t, int64_t, int64_t, void*, int64_t,
                         int64_t, int64_t, void*, int64_t, int64_t, int64_t,
                         float, float) {
  return false;
}
inline bool unary(kop, void*, int64_t, void*, int64_t, int64_t, float, float) {
  return false;
}
inline bool gemm(void*, int64_t, int64_t, bool, void*, int64_t, int64_t, bool,
                 void*, int64_t, int64_t, int64_t, int64_t, float, float) {
  return false;
}
inline bool row_op(kop, void*, int64_t, void*, int64_t, int64_t, int64_t, float,
                   float) {
  return false;
}
inline bool gemv_f32(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool gemv_bf16(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool attn_decode(void*, void*, void*, void*, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float) {
  return false;
}
inline bool rope(void*, void*, int64_t, int64_t, int64_t, int64_t, float) {
  return false;
}
inline bool gemv_q4(void*, void*, void*, void*, int64_t, int64_t, int64_t) {
  return false;
}

#endif

// Every CPU-side buffer read funnels through array::raw()/data(), which call
// this: one choke point makes mixed CPU/GPU graphs safe.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace webgpu
}  // namespace tl
