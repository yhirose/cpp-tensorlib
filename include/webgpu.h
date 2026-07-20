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
#include <cstdio>
#include <cstring>
#include <string>
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

// Uniform params, laid out to match the WGSL Params struct. One struct serves
// every kernel family (see the comment on Params in the .wgsl): unused fields
// cost a few bytes of a 256-byte slot, and it keeps one uniform ring and one
// bind group layout for the whole backend.
struct params {
  uint32_t M, N, K;
  uint32_t lda, ldb, ldc;
  uint32_t a_off, b_off, c_off;
  uint32_t ta, tb;
  uint32_t ars, acs, brs, bcs;
  uint32_t op;
  float scale, offset;
  uint32_t pad0, pad1, pad2, pad3, pad4, pad5;
};

// WGSL gives a uniform-address-space struct align 16, so Params is 96 bytes
// there. This must agree: the bind group's minBindingSize comes from sizeof
// here, and a short one fails validation on every dispatch.
static_assert(sizeof(params) == 96, "params must match the WGSL Params size");

// Which operation within a family, matching the OP_* constants in the WGSL.
// Families have separate numbering, so this is only meaningful alongside the
// entry point it is passed to.
inline uint32_t kernel_op_(kop op) {
  switch (op) {
    case kop::add: case kop::badd: return 0;
    case kop::sub: case kop::bsub: return 1;
    case kop::mul: case kop::bmul: return 2;
    case kop::div: case kop::bdiv: return 3;
    case kop::pow_: case kop::bpow: return 4;

    case kop::exp_: return 0;
    case kop::log_: return 1;
    case kop::sqrt_: return 2;
    case kop::sigmoid: return 3;
    case kop::relu: return 4;
    case kop::affine: return 5;

    case kop::row_sum: return 0;
    case kop::row_max: return 1;
    default: return 0;
  }
}

// WebGPU guarantees maxComputeWorkgroupsPerDimension >= 65535. Anything past
// that returns false and falls to CPU rather than silently truncating.
constexpr int64_t kMaxWorkgroups = 65535;

// Dynamic uniform offsets must be a multiple of the adapter's
// minUniformBufferOffsetAlignment; 256 is the spec's guaranteed-safe maximum.
constexpr uint64_t kUniformSlot = 256;
constexpr uint32_t kUniformSlots = 256;  // one flush can batch this many

struct context {
  wgpu::Instance instance;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::BindGroupLayout bgl;
  wgpu::PipelineLayout play;
  wgpu::ShaderModule mod;
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

  // Compute pipelines, built on first use and keyed by WGSL entry point.
  // Pipeline creation is not free, and every entry point shares one layout,
  // so this is a pure cache.
  std::unordered_map<std::string, wgpu::ComputePipeline> pipelines;
  // Per-entry-point dispatch census. Unlike the native backends, this one has
  // no test runner that fails when it is absent: the browser suite passes
  // whether or not the GPU engages, because every unported op falls back to
  // CPU. So the harness reports these counts, and a family reading zero means
  // the backend quietly stopped doing the work. Cheap enough to always keep.
  std::unordered_map<std::string, long> dispatch_counts;

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
    mod = device.CreateShaderModule(&smd);
    // CreateShaderModule hands back an INVALID object, not null, when the WGSL
    // fails to compile — and so does every pipeline built from it, and every
    // dispatch then silently does nothing. (That cost a debugging session: one
    // bad literal made the whole suite read zeros, including ops that had been
    // working.) Ask for the compilation log and refuse to come up ready.
    bool compiled = false;
    if (mod) {
      wait(mod.GetCompilationInfo(
          wgpu::CallbackMode::WaitAnyOnly,
          [&compiled](wgpu::CompilationInfoRequestStatus st,
                      const wgpu::CompilationInfo* info) {
            compiled = st == wgpu::CompilationInfoRequestStatus::Success;
            if (!info) return;
            for (size_t i = 0; i < info->messageCount; ++i) {
              const auto& m = info->messages[i];
              if (m.type != wgpu::CompilationMessageType::Error) continue;
              compiled = false;
              std::fprintf(stderr, "tensorlib webgpu: WGSL error at %llu:%llu: %.*s\n",
                           (unsigned long long)m.lineNum,
                           (unsigned long long)m.linePos,
                           (int)m.message.length, m.message.data);
            }
          }));
    }
    if (!compiled) return;

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
    be[3].buffer.minBindingSize = sizeof(params);
    wgpu::BindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 4;
    bgld.entries = be;
    bgl = device.CreateBindGroupLayout(&bgld);
    if (!bgl) return;

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &bgl;
    play = device.CreatePipelineLayout(&pld);
    if (!play) return;

    wgpu::BufferDescriptor ud = {};
    ud.size = kUniformSlot * kUniformSlots;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = device.CreateBuffer(&ud);
    if (!uniforms) return;

    // Build every pipeline up front rather than on first use. Lazily, a WGSL
    // error would surface as an op quietly falling back to CPU forever; here
    // it makes available() false, which the harness reports.
    for (const char* ep : {"sgemm", "ew_binary", "ew_unary", "ew_bcast",
                           "softmax", "row_reduce"}) {
      if (!pipeline_(ep)) return;
    }

    ready = true;
  }

  wgpu::ComputePipeline pipeline_(const char* entry) {
    auto it = pipelines.find(entry);
    if (it != pipelines.end()) return it->second;
    wgpu::ComputePipelineDescriptor pd = {};
    pd.layout = play;
    pd.compute.module = mod;
    pd.compute.entryPoint = entry;
    wgpu::ComputePipeline p = device.CreateComputePipeline(&pd);
    if (p) pipelines[entry] = p;
    return p;
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

  // The one place a dispatch is encoded. Every op differs only in which
  // pipeline, which params and what grid — keeping the bind group, uniform
  // ring and encoder bookkeeping in a single copy is the same discipline
  // metal_kernels.metal applies to its kernel bodies.
  //
  // `b` may be the same mirror as `a` (a unary or reduce kernel binds its one
  // input twice): two read-only bindings may alias. `out` is always a fresh
  // allocation from the evaluator, so a writable binding never does.
  bool encode_(const char* entry, mirror* a, mirror* b, mirror* out,
               const params& p, int64_t gx, int64_t gy) {
    if (gx <= 0 || gy <= 0) return false;
    if (gx > kMaxWorkgroups || gy > kMaxWorkgroups) return false;
    wgpu::ComputePipeline pipe = pipeline_(entry);
    if (!pipe) return false;
    if (slot >= kUniformSlots) flush_();  // ring exhausted; start a new batch

    const uint32_t off = slot++ * (uint32_t)kUniformSlot;
    queue.WriteBuffer(uniforms, off, &p, sizeof(p));

    wgpu::BindGroupEntry e[4] = {};
    e[0].binding = 0; e[0].buffer = a->dev;   e[0].size = a->bytes;
    e[1].binding = 1; e[1].buffer = b->dev;   e[1].size = b->bytes;
    e[2].binding = 2; e[2].buffer = out->dev; e[2].size = out->bytes;
    e[3].binding = 3; e[3].buffer = uniforms; e[3].size = sizeof(params);
    wgpu::BindGroupDescriptor bgd = {};
    bgd.layout = bgl;
    bgd.entryCount = 4;
    bgd.entries = e;
    wgpu::BindGroup bg = device.CreateBindGroup(&bgd);

    if (!enc) enc = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = enc.BeginComputePass();
    pass.SetPipeline(pipe);
    pass.SetBindGroup(0, bg, 1, &off);
    pass.DispatchWorkgroups((uint32_t)gx, (uint32_t)gy, 1);
    pass.End();

    pending = true;
    dispatch_counts[entry]++;
    return true;
  }

  void flush_();  // defined below, once flush() is in scope

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

inline void context::flush_() { flush(); }

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

// Shared host-side prologue for every op: resolve the operand mirrors and
// stage the lazy copies. Returns false when any operand is untracked — that
// means a heap storage with no device buffer, so the op belongs on the CPU.
// `b` may be null for one-input kernels, which then bind `a` twice.
inline bool operands_(context& c, void* a, void* b, void* out,
                      context::mirror** ma, context::mirror** mb,
                      context::mirror** mo) {
  *ma = c.mirror_(a);
  *mb = b ? c.mirror_(b) : *ma;
  *mo = c.mirror_(out);
  if (!*ma || !*mb || !*mo) return false;
  c.device_read_(a);
  if (b) c.device_read_(b);
  c.device_write_(out);
  return true;
}

// Byte offsets must be 4-aligned to convert to the element offsets the
// kernels index with. They always are for f32 views; anything else falls to
// the CPU rather than silently truncating.
inline bool elem_off_(int64_t byte_off, uint32_t* out) {
  if (byte_off % 4) return false;
  *out = (uint32_t)(byte_off / 4);
  return true;
}

// out = op(a) @ op(b) * scale + offset.
inline bool gemm(void* a, int64_t ao, int64_t lda, bool ta, void* b, int64_t bo,
                 int64_t ldb, bool tb, void* out, int64_t oo, int64_t m,
                 int64_t n, int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || m <= 0 || n <= 0 || k <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(bo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, b, out, &ma, &mb, &mo)) return false;

  p.M = (uint32_t)m;
  p.N = (uint32_t)n;
  p.K = (uint32_t)k;
  p.lda = (uint32_t)lda;
  p.ldb = (uint32_t)ldb;
  p.ldc = (uint32_t)n;  // the eval seam always hands us a contiguous output
  p.ta = ta ? 1u : 0u;
  p.tb = tb ? 1u : 0u;
  p.scale = scale;
  p.offset = offset;
  return c.encode_("sgemm", ma, mb, mo, p, (n + 63) / 64, (m + 63) / 64);
}

// Contiguous elementwise binary over n elements.
inline bool binary(kop op, void* a, int64_t ao, void* b, int64_t bo, void* out,
                   int64_t oo, int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(bo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, b, out, &ma, &mb, &mo)) return false;

  p.M = (uint32_t)n;
  p.op = kernel_op_(op);
  p.scale = scale;
  p.offset = offset;
  return c.encode_("ew_binary", ma, mb, mo, p, (n + 255) / 256, 1);
}

inline bool unary(kop op, void* a, int64_t ao, void* out, int64_t oo, int64_t n,
                  float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, nullptr, out, &ma, &mb, &mo)) return false;

  p.b_off = p.a_off;  // the kernel binds its one input twice
  p.M = (uint32_t)n;
  p.op = kernel_op_(op);
  p.scale = scale;
  p.offset = offset;
  return c.encode_("ew_unary", ma, mb, mo, p, (n + 255) / 256, 1);
}

// Rank-2 broadcast binary: out[r,c] = f(a[r*ars + c*acs], b[r*brs + c*bcs])
// into a contiguous [m,n] output. One stride-parameterized kernel covers every
// rank-2 broadcast, which keeps bias/gamma/beta chains on the GPU — falling
// back mid-graph would cost a full submit-and-wait.
inline bool binary_bcast(kop op, void* a, int64_t ao, int64_t ars, int64_t acs,
                         void* b, int64_t bo, int64_t brs, int64_t bcs,
                         void* out, int64_t oo, int64_t m, int64_t n,
                         float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || m <= 0 || n <= 0) return false;
  // Broadcast strides are non-negative here (broadcast_strides only ever
  // zeroes an axis); a negative one would wrap as u32 in the kernel.
  if (ars < 0 || acs < 0 || brs < 0 || bcs < 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(bo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, b, out, &ma, &mb, &mo)) return false;

  p.M = (uint32_t)m;
  p.N = (uint32_t)n;
  p.ars = (uint32_t)ars;
  p.acs = (uint32_t)acs;
  p.brs = (uint32_t)brs;
  p.bcs = (uint32_t)bcs;
  p.op = kernel_op_(op);
  p.scale = scale;
  p.offset = offset;
  return c.encode_("ew_bcast", ma, mb, mo, p, (n + 31) / 32, (m + 7) / 8);
}

// Row-wise op over the last axis: softmax writes rows x cols; row_sum/row_max
// write one value per row, with the affine epilogue. One workgroup per row.
inline bool row_op(kop op, void* in, int64_t io, void* out, int64_t oo,
                   int64_t rows, int64_t cols, float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || rows <= 0 || cols <= 0) return false;
  params p = {};
  if (!elem_off_(io, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, in, nullptr, out, &ma, &mb, &mo)) return false;

  p.b_off = p.a_off;
  p.M = (uint32_t)rows;
  p.N = (uint32_t)cols;
  p.op = kernel_op_(op);
  p.scale = scale;
  p.offset = offset;
  const char* entry = op == kop::softmax ? "softmax" : "row_reduce";
  return c.encode_(entry, ma, mb, mo, p, rows, 1);
}

// ---- Not ported yet. Returning false routes the op to CPU, which is why each
// phase lands in a working state.
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
