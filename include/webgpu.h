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
// Async: flush() and sync_to_host() keep their SYNCHRONOUS signatures — the one
// thing that had to hold for array.h's eval seam to survive a browser backend,
// so it was measured in a standalone probe before any of this was written. The
// instance is created with the TimedWaitAny feature, which makes
// wgpuInstanceWaitAny(timeout > 0) a legal blocking wait; emdawnwebgpu
// implements it by suspending through Asyncify/JSPI, and the suspend surface is
// exactly two call sites — OnSubmittedWorkDone and MapAsync — not the whole
// interpreter. Uploads (WriteBuffer) are queued, never awaited. Which of the two
// suspend mechanisms to link is a build-time fork, not a backend concern: see
// test/wasm/build.sh, which takes JSPI and says why.
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
using cmp_op = tl::metal::cmp_op;
using unary_ext_op = tl::metal::unary_ext_op;

#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)

}  // namespace webgpu
}  // namespace tl

#include <emscripten/em_asm.h>

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
    case kop::tanh_: return 6;
    case kop::sin_: return 7;
    case kop::cos_: return 8;

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
constexpr uint64_t kUniformSlotBytes = 256;
// One flush can batch this many dispatches; past it, encode_ forces a blocking
// flush mid-graph. At 96 bytes of payload per 256-byte slot the ring is pure
// device memory (1 MB here) with no binding-size implication, so it is sized to
// put that forced stall well beyond any graph the backend is aimed at.
constexpr uint32_t kUniformSlotCount = 4096;

// Every WGSL entry point. The context prebuilds a pipeline for each and the
// browser harness asserts each one dispatched — both need the same list, and a
// new kernel missing from either loses a guarantee silently.
inline constexpr const char* kEntryPoints[] = {
    "sgemm",        "ew_binary",   "ew_unary",     "ew_bcast",
    "softmax",      "row_reduce",  "pad",          "fold",
    "index_select", "index_add",  "scatter_axis", "ew_bcast_nd",
    "where_nd",     "cmp",        "clamp_",       "sum_to"};

// emscripten_webgpu_get_device() does not report "no device" — it hands
// Module.preinitializedWebGPUDevice straight to importJsDevice, which reads
// .queue off it (library_webgpu.js), so an absent device leaves as a JS
// TypeError on the first dispatch instead of as a null return. Its assert()
// says as much, and a release link compiles that assert out. Ask JS directly,
// so the case the header documents — JS handed in no device, available()
// stays false, every op routes to CPU — is the one that actually happens.
//
// EM_ASM_INT, not EM_JS: EM_JS defines a named extern "C" symbol, and in a
// header-only library every translation unit that includes this file would
// define it again — two TUs (the CI suite links main_wasm.cpp with
// test_array.cpp) is a duplicate-symbol link error. EM_ASM's snippet is
// per-call-site section data the linker accepts from any number of TUs, and
// this runs once at context construction, where the JS-call overhead EM_JS
// exists to avoid is irrelevant.
inline bool has_preinitialized_device_() {
  return EM_ASM_INT({ return Module["preinitializedWebGPUDevice"] ? 1 : 0; }) != 0;
}

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

  // The open encoder and its compute pass. Created lazily on the first
  // dispatch after a flush, so an idle backend submits nothing. One pass spans
  // the whole batch, as metal.h keeps one MTLComputeCommandEncoder: WebGPU
  // orders dispatches within a pass and makes each one's writes visible to the
  // next, so chained ops stay correct, while a pass boundary per dispatch would
  // serialize independent ones and give back what the batching is for.
  wgpu::CommandEncoder enc;
  wgpu::ComputePassEncoder pass;
  uint32_t slot = 0;  // next free uniform ring slot

  // pad_/fold_'s per-call shape metadata ring — same idea as the uniform
  // ring above (queue.WriteBuffer runs ahead of whatever is still sitting in
  // the unsubmitted encoder, so two calls sharing one buffer before a flush
  // would have the second's write stomp the first dispatch's not-yet-
  // executed read), kept as real context state rather than a second,
  // independent static-local ring so flush() resets both counters together
  // (see meta_ring_/meta_reserve_slot_ below, defined once alloc() is in
  // scope, same reason flush_() is forward-declared here).
  void* meta_ring_tok = nullptr;
  float* meta_ring_host = nullptr;
  uint32_t meta_slot = 0;
  void* meta_ring_(float** host_out);
  uint32_t meta_reserve_slot_();

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

  // Compute pipelines, keyed by WGSL entry point. Every one is built in the
  // constructor and they all share a layout, so by the time encode_ runs this
  // is a pure lookup and pipeline_'s create branch is unreachable.
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
    if (!has_preinitialized_device_()) return;
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
    // Bindings 4 (D) and 5 (E) are a third and fourth read-only operand —
    // binary_bcast_nd and where_nd need more real buffers (operands + N-D
    // shape/stride meta) than A/B/C alone can hold; see tensorlib_webgpu.wgsl's
    // comment on D/E for which kernel binds what there.
    wgpu::BindGroupLayoutEntry be[6] = {};
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
    for (int i = 4; i < 6; ++i) {
      be[i].binding = i;
      be[i].visibility = wgpu::ShaderStage::Compute;
      be[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    wgpu::BindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 6;
    bgld.entries = be;
    bgl = device.CreateBindGroupLayout(&bgld);
    if (!bgl) return;

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &bgl;
    play = device.CreatePipelineLayout(&pld);
    if (!play) return;

    wgpu::BufferDescriptor ud = {};
    ud.size = kUniformSlotBytes * kUniformSlotCount;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = device.CreateBuffer(&ud);
    if (!uniforms) return;

    // Build every pipeline up front rather than on first use. Lazily, a WGSL
    // error would surface as an op quietly falling back to CPU forever; here
    // it makes available() false, which the harness reports.
    for (const char* ep : kEntryPoints) {
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
  // allocation from the evaluator, so a writable binding never does. `d`/`e`
  // are the optional third/fourth read-only operand (binary_bcast_nd/
  // where_nd's extra tensor operand and/or N-D shape/stride meta); every
  // other kernel leaves them null, which binds them to `a` -- unused by that
  // kernel's WGSL, but bind group validation requires every declared binding
  // be present regardless of which ones the active entry point reads.
  bool encode_(const char* entry, mirror* a, mirror* b, mirror* out,
               const params& p, int64_t gx, int64_t gy, mirror* d = nullptr,
               mirror* e = nullptr) {
    if (gx <= 0 || gy <= 0) return false;
    if (gx > kMaxWorkgroups || gy > kMaxWorkgroups) return false;
    wgpu::ComputePipeline pipe = pipeline_(entry);
    if (!pipe) return false;
    if (slot >= kUniformSlotCount) flush_();  // ring exhausted; new batch

    const uint32_t off = slot++ * (uint32_t)kUniformSlotBytes;
    queue.WriteBuffer(uniforms, off, &p, sizeof(p));

    mirror* md = d ? d : a;
    mirror* me = e ? e : a;
    wgpu::BindGroupEntry bge[6] = {};
    bge[0].binding = 0; bge[0].buffer = a->dev;    bge[0].size = a->bytes;
    bge[1].binding = 1; bge[1].buffer = b->dev;    bge[1].size = b->bytes;
    bge[2].binding = 2; bge[2].buffer = out->dev;  bge[2].size = out->bytes;
    bge[3].binding = 3; bge[3].buffer = uniforms;  bge[3].size = sizeof(params);
    bge[4].binding = 4; bge[4].buffer = md->dev;   bge[4].size = md->bytes;
    bge[5].binding = 5; bge[5].buffer = me->dev;   bge[5].size = me->bytes;
    wgpu::BindGroupDescriptor bgd = {};
    bgd.layout = bgl;
    bgd.entryCount = 6;
    bgd.entries = bge;
    wgpu::BindGroup bg = device.CreateBindGroup(&bgd);

    if (!enc) {
      enc = device.CreateCommandEncoder();
      pass = enc.BeginComputePass();
    }
    pass.SetPipeline(pipe);
    pass.SetBindGroup(0, bg, 1, &off);
    pass.DispatchWorkgroups((uint32_t)gx, (uint32_t)gy, 1);

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
  c.meta_slot = 0;
  if (!c.enc) return;
  c.pass.End();
  c.pass = nullptr;
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
      if (const void* src = stg.GetConstMappedRange(0, m->bytes)) {
        std::memcpy(m->host, src, m->bytes);
        m->where = context::BOTH;
      }
      stg.Unmap();
    }
    // Only a completed memcpy makes the host copy current. Declaring BOTH on a
    // failed readback would leave stale bytes permanently believed live, and no
    // later sync_to_host would retry — so say so loudly instead, as the WGSL
    // compile failure above does.
    if (m->where == context::DEVICE) {
      std::fprintf(stderr, "tensorlib webgpu: readback of %zu bytes failed\n",
                   m->bytes);
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

// A ring, not one reused buffer: queue.WriteBuffer runs ahead of whatever is
// still sitting in the unsubmitted encoder (see device_read_'s comment
// above), so two pad/fold calls batched into the same unflushed pass would
// have the second call's metadata write stomp the first dispatch's
// not-yet-executed read of the same buffer — the exact hazard the uniform
// ring above (kUniformSlotCount) already exists to avoid for Params, just for
// a second resource. One slot comfortably covers the rank-8 cap (pad needs
// 2*rank <= 16 words, fold (rank-1)+rank <= 15, binary_bcast_nd 3*rank <= 24,
// where_nd's own [out_shape, cond_strides, a_strides, b_strides] 4*rank <=
// 32 -- the largest of the four, which is what sizes this), which is why
// this state lives on `context` (meta_ring_tok/meta_ring_host/meta_slot)
// right beside `slot` instead of as a second, independent ring: flush()
// resets both counters together, the way it already resets `slot`.
inline constexpr size_t kMetaSlotWords = 32;
inline constexpr size_t kMetaSlotCount = 4096;

// Rank cap for pad_/fold_'s GPU dispatch — matches cuda.h's own
// kPadFoldMaxRank (not unified with it: the two backends derive their caps
// from different physical constraints, kMetaSlotWords here vs a fixed-size
// on-stack index array there) and kernels/tensorlib_webgpu.wgsl's
// kPadFoldMaxRank, which the WGSL side needs as its own `const` since a
// shader can't see a host-side C++ constant.
inline constexpr int kPadFoldMaxRank = 8;

// Allocated once, sized for the whole ring — through the same alloc() pool
// every tensor buffer uses (by the time pad()/fold() below can run, alloc()
// is already defined above). The token is opaque to eval_one's storage
// layer, so nothing else could mistake it for a live array.
inline void* context::meta_ring_(float** host_out) {
  if (!meta_ring_tok) {
    meta_ring_tok = alloc(
        static_cast<int64_t>(kMetaSlotCount * kMetaSlotWords * 4),
        &meta_ring_host);
  }
  if (!meta_ring_tok) return nullptr;
  *host_out = meta_ring_host;
  return meta_ring_tok;
}

// This call's word offset into the ring, advancing like the uniform ring's
// `slot` and forcing the same flush-then-reset on wraparound.
inline uint32_t context::meta_reserve_slot_() {
  if (meta_slot >= kMetaSlotCount) {
    flush();
    meta_slot = 0;
  }
  return (meta_slot++) * static_cast<uint32_t>(kMetaSlotWords);
}

// Reserve one meta-ring slot: returns the u32 write pointer for this call's
// slot (already offset into the ring) via `host_words`, and this slot's word
// offset via `word_off_out`; the return value is the ring's own opaque
// token, or null if the ring couldn't be allocated. Shared by pad()/fold()/
// binary_bcast_nd()/where_nd() below, which differ only in how many words
// they fill in and with what.
inline void* reserve_meta_(context& c, uint32_t* word_off_out,
                           uint32_t** host_words) {
  float* ring_host = nullptr;
  void* ring_tok = c.meta_ring_(&ring_host);
  if (!ring_tok) return nullptr;
  *word_off_out = c.meta_reserve_slot_();
  *host_words = reinterpret_cast<uint32_t*>(ring_host) + *word_off_out;
  return ring_tok;
}

// Upload a filled meta-ring slot and mark it live -- the other half of
// reserve_meta_ above, split from it so the caller can fill `host_words` (the
// pointer reserve_meta_ handed back) in between.
inline context::mirror* commit_meta_(context& c, void* ring_tok,
                                     uint32_t word_off, const uint32_t* words,
                                     size_t word_count) {
  context::mirror* mm = c.mirror_(ring_tok);
  // This call's slot only, at its own byte offset — an unconditional
  // WriteBuffer, not device_read_'s dirty-flag check, since the ring's
  // mirror never legitimately settles into a single steady HOST/DEVICE state
  // (each slot is written once, read once, never again).
  c.queue.WriteBuffer(mm->dev, word_off * 4, words, word_count * 4);
  mm->where = context::BOTH;
  return mm;
}

// Gather-style pad/fold (M11): unlike CUDA's scatter+atomicAdd, WGSL has no
// float atomicAdd, so both dispatch one invocation per OUTPUT element and
// have it read (pad) or sum (fold) whatever cells of `a` map to it — no
// output cell is ever written by two invocations, so unlike cuda.h's pad/fold
// neither needs a pre-zeroed buffer. `a_shape`/`out_shape` (length `rank`,
// `rank-1` for fold's `out_shape`) ride the otherwise-unused second storage
// binding as bit-reinterpreted u32 — WGSL's fixed Params uniform (used by
// every other kernel here) has no room for a variable-length array, and
// WriteBuffer is a raw byte copy regardless of the binding's declared type.
inline bool pad(void* a_native, int64_t ao, void* out_native, int64_t oo,
                const int64_t* a_shape, const int64_t* out_shape, int rank,
                int axis, int64_t before, int64_t n, int64_t out_n) {
  (void)n;
  auto& c = context::get();
  if (!c.ready || rank <= 0 || rank > kPadFoldMaxRank) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror* ma = c.mirror_(a_native);
  context::mirror* mo = c.mirror_(out_native);
  if (!ma || !mo) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);

  uint32_t word_off, *raw;
  void* ring_tok = reserve_meta_(c, &word_off, &raw);
  if (!ring_tok) return false;
  for (int d = 0; d < rank; d++) raw[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++) raw[rank + d] = static_cast<uint32_t>(a_shape[d]);
  context::mirror* mm =
      commit_meta_(c, ring_tok, word_off, raw, 2 * static_cast<size_t>(rank));

  p.M = static_cast<uint32_t>(out_n);
  p.b_off = word_off;
  p.pad0 = static_cast<uint32_t>(rank);
  p.pad1 = static_cast<uint32_t>(axis);
  p.pad2 = static_cast<uint32_t>(before);
  return c.encode_("pad", ma, mm, mo, p, (out_n + 255) / 256, 1);
}

// unfold's inverse. Each output element sums over the bounded range of
// window indices `w` whose window covers it (`w*step + k == out coordinate`
// along `axis`, `0 <= k < win`) — the gather twin of cuda.h's scatter+
// atomicAdd fold, needed because WGSL has no float atomicAdd. `a`'s own
// strides aren't part of the metadata: `a` is contiguous (gpu_fold_'s
// contract), so the kernel derives them from `a_shape` itself.
inline bool fold(void* a_native, int64_t ao, void* out_native, int64_t oo,
                 const int64_t* a_shape, const int64_t* out_shape, int rank,
                 int axis, int64_t step, int64_t n, int64_t out_n) {
  (void)n;
  auto& c = context::get();
  if (!c.ready || rank <= 0 || rank > kPadFoldMaxRank) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror* ma = c.mirror_(a_native);
  context::mirror* mo = c.mirror_(out_native);
  if (!ma || !mo) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);

  int out_rank = rank - 1;
  uint32_t word_off, *raw;
  void* ring_tok = reserve_meta_(c, &word_off, &raw);
  if (!ring_tok) return false;
  for (int d = 0; d < out_rank; d++) raw[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++) raw[out_rank + d] = static_cast<uint32_t>(a_shape[d]);
  context::mirror* mm =
      commit_meta_(c, ring_tok, word_off, raw,
                   static_cast<size_t>(out_rank) + static_cast<size_t>(rank));

  p.M = static_cast<uint32_t>(out_n);
  p.b_off = word_off;
  p.pad0 = static_cast<uint32_t>(rank);
  p.pad1 = static_cast<uint32_t>(axis);
  p.pad2 = static_cast<uint32_t>(step);
  return c.encode_("fold", ma, mm, mo, p, (out_n + 255) / 256, 1);
}

// Row gather along axis 0: out[i] = a[indices[row(i)]] (a, indices
// contiguous). A = a, B = idx, C = out; p.pad0 = row_size.
inline bool index_select(void* a_native, int64_t ao, void* idx_native,
                         int64_t idxo, void* out_native, int64_t oo,
                         int64_t row_size, int64_t k) {
  auto& c = context::get();
  int64_t n = k * row_size;
  if (!c.ready || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(idxo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a_native, idx_native, out_native, &ma, &mb, &mo)) {
    return false;
  }
  p.M = static_cast<uint32_t>(n);
  p.pad0 = static_cast<uint32_t>(row_size);
  return c.encode_("index_select", ma, mb, mo, p, (n + 255) / 256, 1);
}

// index_select's dual, rewritten as a gather: WGSL has no float atomicAdd,
// the same gap pad/fold above work around, so this sums over every source
// row matching each OUTPUT row instead of scattering into a pre-zeroed
// buffer -- no zeroing needed. A = idx, B = values, C = out; p.pad0 =
// row_size, p.pad1 = k (number of source rows to scan).
inline bool index_add(void* idx_native, int64_t idxo, void* values_native,
                      int64_t vo, void* out_native, int64_t oo,
                      int64_t row_size, int64_t k, int64_t out_n) {
  auto& c = context::get();
  if (!c.ready || out_n <= 0) return false;
  params p = {};
  if (!elem_off_(idxo, &p.a_off) || !elem_off_(vo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, idx_native, values_native, out_native, &ma, &mb, &mo)) {
    return false;
  }
  p.M = static_cast<uint32_t>(out_n);
  p.pad0 = static_cast<uint32_t>(row_size);
  p.pad1 = static_cast<uint32_t>(k);
  return c.encode_("index_add", ma, mb, mo, p, (out_n + 255) / 256, 1);
}

// One-hot scatter into a new trailing axis, as a gather: out[pos,k] =
// values[pos] where indices[pos] == k, else 0. Every output element reads,
// never writes twice, so -- like index_select above -- no zeroing needed.
// A = idx, B = values, C = out; p.pad0 = size.
inline bool scatter_to_axis(void* idx_native, int64_t idxo,
                            void* values_native, int64_t vo, void* out_native,
                            int64_t oo, int64_t n, int64_t size) {
  auto& c = context::get();
  int64_t out_n = n * size;
  if (!c.ready || out_n <= 0) return false;
  params p = {};
  if (!elem_off_(idxo, &p.a_off) || !elem_off_(vo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, idx_native, values_native, out_native, &ma, &mb, &mo)) {
    return false;
  }
  p.M = static_cast<uint32_t>(out_n);
  p.pad0 = static_cast<uint32_t>(size);
  return c.encode_("scatter_axis", ma, mb, mo, p, (out_n + 255) / 256, 1);
}

// N-D broadcast binary: generalizes binary_bcast() above to any rank (a
// Transformer's [N,S,D] LayerNorm broadcasting a [N,S,1] mean, rank 3).
// a_strides/b_strides are the broadcast strides (0 on a broadcast axis)
// array.h computes host-side via the same broadcast_strides() the CPU
// oracle uses. A = a, B = b, D = meta [out_shape(rank), a_strides(rank),
// b_strides(rank)] (a and b already fill A/B, unlike pad/fold where B was
// free for this); p.pad0 = rank, p.pad3 = meta's word offset into D.
inline bool binary_bcast_nd(kop op, void* a_native, int64_t ao,
                            const int64_t* a_strides, void* b_native,
                            int64_t bo, const int64_t* b_strides,
                            void* out_native, int64_t oo,
                            const int64_t* out_shape, int rank, int64_t n,
                            float scale, float offset) {
  auto& c = context::get();
  if (!c.ready || rank <= 0 || rank > kPadFoldMaxRank || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(bo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a_native, b_native, out_native, &ma, &mb, &mo)) {
    return false;
  }

  uint32_t word_off, *raw;
  void* ring_tok = reserve_meta_(c, &word_off, &raw);
  if (!ring_tok) return false;
  for (int d = 0; d < rank; d++) raw[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++) {
    raw[rank + d] = static_cast<uint32_t>(a_strides[d]);
  }
  for (int d = 0; d < rank; d++) {
    raw[2 * rank + d] = static_cast<uint32_t>(b_strides[d]);
  }
  context::mirror* mm =
      commit_meta_(c, ring_tok, word_off, raw, 3 * static_cast<size_t>(rank));

  p.M = static_cast<uint32_t>(n);
  p.op = kernel_op_(op);
  p.pad0 = static_cast<uint32_t>(rank);
  p.pad3 = word_off;
  p.scale = scale;
  p.offset = offset;
  return c.encode_("ew_bcast_nd", ma, mb, mo, p, (n + 255) / 256, 1, mm);
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch. A = cond, B = a,
// D = b, E = meta [out_shape(rank), cond_strides(rank), a_strides(rank),
// b_strides(rank)] (cond/a/b fill A/B/D, leaving E free for meta); p.pad0 =
// rank, p.pad3 = b's element offset into D, p.pad4 = meta's word offset
// into E. Bypasses operands_() (built for two real operands) since this one
// needs three, the same way pad()/fold() above do their own mirror lookups.
inline bool where_nd(void* cond_native, int64_t co, const int64_t* c_strides,
                     void* a_native, int64_t ao, const int64_t* a_strides,
                     void* b_native, int64_t bo, const int64_t* b_strides,
                     void* out_native, int64_t oo, const int64_t* out_shape,
                     int rank, int64_t n) {
  auto& c = context::get();
  if (!c.ready || rank <= 0 || rank > kPadFoldMaxRank || n <= 0) return false;
  params p = {};
  uint32_t b_elem_off;
  if (!elem_off_(co, &p.a_off) || !elem_off_(ao, &p.b_off) ||
      !elem_off_(bo, &b_elem_off) || !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror* mcond = c.mirror_(cond_native);
  context::mirror* ma = c.mirror_(a_native);
  context::mirror* mb = c.mirror_(b_native);
  context::mirror* mo = c.mirror_(out_native);
  if (!mcond || !ma || !mb || !mo) return false;
  c.device_read_(cond_native);
  c.device_read_(a_native);
  c.device_read_(b_native);
  c.device_write_(out_native);

  uint32_t word_off, *raw;
  void* ring_tok = reserve_meta_(c, &word_off, &raw);
  if (!ring_tok) return false;
  for (int d = 0; d < rank; d++) raw[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++) {
    raw[rank + d] = static_cast<uint32_t>(c_strides[d]);
  }
  for (int d = 0; d < rank; d++) {
    raw[2 * rank + d] = static_cast<uint32_t>(a_strides[d]);
  }
  for (int d = 0; d < rank; d++) {
    raw[3 * rank + d] = static_cast<uint32_t>(b_strides[d]);
  }
  context::mirror* mm =
      commit_meta_(c, ring_tok, word_off, raw, 4 * static_cast<size_t>(rank));

  p.M = static_cast<uint32_t>(n);
  p.pad0 = static_cast<uint32_t>(rank);
  p.pad3 = b_elem_off;
  p.pad4 = word_off;
  return c.encode_("where_nd", mcond, ma, mo, p, (n + 255) / 256, 1, mb, mm);
}

// sum_to (un-broadcast a gradient): gather, mirrors cuda.h's tl_sum_to and
// metal.h's own sum_to -- one invocation per OUTPUT element sums every `a`
// element that broadcasts onto it, so no atomics (unlike index_add). Only
// one real tensor operand (`a`), so -- like pad/fold above -- B is free for
// the meta ring: [a_shape(rank), a_strides(rank), acc(rank)].
inline bool sum_to(void* a_native, int64_t ao, const int64_t* a_shape,
                   const int64_t* a_strides, const int64_t* acc, int rank,
                   int64_t out_n, int64_t reduced_n, void* out_native,
                   int64_t oo) {
  auto& c = context::get();
  if (!c.ready || rank <= 0 || rank > kPadFoldMaxRank) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror* ma = c.mirror_(a_native);
  context::mirror* mo = c.mirror_(out_native);
  if (!ma || !mo) return false;
  c.device_read_(a_native);
  c.device_write_(out_native);

  uint32_t word_off, *raw;
  void* ring_tok = reserve_meta_(c, &word_off, &raw);
  if (!ring_tok) return false;
  for (int d = 0; d < rank; d++) {
    raw[d] = static_cast<uint32_t>(a_shape[d]);
  }
  for (int d = 0; d < rank; d++) {
    raw[rank + d] = static_cast<uint32_t>(a_strides[d]);
  }
  for (int d = 0; d < rank; d++) {
    raw[2 * rank + d] = static_cast<uint32_t>(acc[d]);
  }
  context::mirror* mm =
      commit_meta_(c, ring_tok, word_off, raw, 3 * static_cast<size_t>(rank));

  p.M = static_cast<uint32_t>(out_n);
  p.b_off = word_off;
  p.pad0 = static_cast<uint32_t>(rank);
  p.pad1 = static_cast<uint32_t>(reduced_n);
  return c.encode_("sum_to", ma, mm, mo, p, (out_n + 255) / 256, 1);
}

// gt/lt/ge/le/eq/ne (array.h's comparison ops, ReLU/LeakyReLU/Clip's
// backward gate): same-shape only (bstride=1) or a scalar b (bstride=0) --
// the two shapes array.h's gpu_compare_ ever dispatches. p.ars (unused by
// this family) carries bstride. Own entry point/op numbering, not folded
// into ew_binary's binary_op -- it returns a bool-as-float mask rather than
// composing with scale/offset.
inline bool compare(cmp_op op, void* a, int64_t ao, void* b, int64_t bo,
                    void* out, int64_t oo, int64_t n, int64_t bstride) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(bo, &p.b_off) ||
      !elem_off_(oo, &p.c_off)) {
    return false;
  }
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, b, out, &ma, &mb, &mo)) return false;
  p.M = static_cast<uint32_t>(n);
  p.ars = static_cast<uint32_t>(bstride);
  switch (op) {
    case cmp_op::gt: p.op = 0; break;
    case cmp_op::lt: p.op = 1; break;
    case cmp_op::ge: p.op = 2; break;
    case cmp_op::le: p.op = 3; break;
    case cmp_op::eq: p.op = 4; break;
    case cmp_op::ne: p.op = 5; break;
  }
  return c.encode_("cmp", ma, mb, mo, p, (n + 255) / 256, 1);
}

// tanh_/sin_/cos_ (RoPE's trig, RNN/LSTM's tanh): plain elementwise, same
// shape as exp_/sqrt_ -- unary() above already does exactly this dispatch
// through ew_unary, just keyed by a kop array.h doesn't see directly.
inline bool unary_ext(unary_ext_op op, void* a, int64_t ao, void* out,
                      int64_t oo, int64_t n, float scale, float offset) {
  kop k;
  switch (op) {
    case unary_ext_op::tanh_: k = kop::tanh_; break;
    case unary_ext_op::sin_: k = kop::sin_; break;
    case unary_ext_op::cos_: k = kop::cos_; break;
  }
  // Qualified: unqualified unary(...) is ambiguous here -- kop is really
  // tl::metal::kop, so ADL pulls in metal.h's non-Apple unary() stub
  // alongside this namespace's own.
  return tl::webgpu::unary(k, a, ao, out, oo, n, scale, offset);
}

// clamp(x, lo, hi): Clip's forward. No epilogue -- p.scale/p.offset carry
// lo/hi instead (mirrors cuda.h's/metal.h's own clamp).
inline bool clamp(void* a, int64_t ao, void* out, int64_t oo, int64_t n,
                  float lo, float hi) {
  auto& c = context::get();
  if (!c.ready || n <= 0) return false;
  params p = {};
  if (!elem_off_(ao, &p.a_off) || !elem_off_(oo, &p.c_off)) return false;
  context::mirror *ma, *mb, *mo;
  if (!operands_(c, a, nullptr, out, &ma, &mb, &mo)) return false;
  p.b_off = p.a_off;  // the kernel binds its one input twice
  p.M = static_cast<uint32_t>(n);
  p.scale = lo;
  p.offset = hi;
  return c.encode_("clamp_", ma, mb, mo, p, (n + 255) / 256, 1);
}
inline bool concat_part(void*, int64_t, void*, int64_t, const int64_t*,
                        const int64_t*, int, int, int64_t, int64_t) {
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
inline bool pad(void*, int64_t, void*, int64_t, const int64_t*,
                const int64_t*, int, int, int64_t, int64_t, int64_t) {
  return false;
}
inline bool fold(void*, int64_t, void*, int64_t, const int64_t*,
                 const int64_t*, int, int, int64_t, int64_t, int64_t) {
  return false;
}
inline bool index_select(void*, int64_t, void*, int64_t, void*, int64_t,
                         int64_t, int64_t) {
  return false;
}
inline bool index_add(void*, int64_t, void*, int64_t, void*, int64_t, int64_t,
                      int64_t, int64_t) {
  return false;
}
inline bool scatter_to_axis(void*, int64_t, void*, int64_t, void*, int64_t,
                            int64_t, int64_t) {
  return false;
}
inline bool binary_bcast_nd(kop, void*, int64_t, const int64_t*, void*,
                            int64_t, const int64_t*, void*, int64_t,
                            const int64_t*, int, int64_t, float, float) {
  return false;
}
inline bool where_nd(void*, int64_t, const int64_t*, void*, int64_t,
                     const int64_t*, void*, int64_t, const int64_t*, void*,
                     int64_t, const int64_t*, int, int64_t) {
  return false;
}
inline bool sum_to(void*, int64_t, const int64_t*, const int64_t*,
                   const int64_t*, int, int64_t, int64_t, void*, int64_t) {
  return false;
}
inline bool compare(cmp_op, void*, int64_t, void*, int64_t, void*, int64_t,
                    int64_t, int64_t) {
  return false;
}
inline bool unary_ext(unary_ext_op, void*, int64_t, void*, int64_t, int64_t,
                      float, float) {
  return false;
}
inline bool clamp(void*, int64_t, void*, int64_t, int64_t, float, float) {
  return false;
}
inline bool concat_part(void*, int64_t, void*, int64_t, const int64_t*,
                        const int64_t*, int, int, int64_t, int64_t) {
  return false;
}

#endif

// ---- Ops with no WGSL kernel yet (the LLM decode path). Outside the #if/#else
// on purpose: both branches would define them identically, and returning false
// is the whole implementation either way — it routes the op to CPU, which is
// why each porting phase lands in a working state.
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

// Every CPU-side buffer read funnels through array::raw()/data(), which call
// this: one choke point makes mixed CPU/GPU graphs safe.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace webgpu
}  // namespace tl
