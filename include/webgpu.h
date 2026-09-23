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

#include "gpu_abi.h"  // the op vocabulary and the launch contract
#include "profile.h"
#include "shape.h"  // tl::contiguous_strides_into (concat_part's meta)
#include "types.h"

#if defined(TENSORLIB_WEBGPU) && defined(__EMSCRIPTEN__)

#include <emscripten/em_asm.h>

// emdawnwebgpu declares emscripten_webgpu_get_device() in webgpu.h itself,
// not in emscripten/html5_webgpu.h (that is the old built-in binding's home).
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace tl {
namespace webgpu {

using kop = gpu::kop;
using cmp_op = gpu::cmp_op;
using unary_ext_op = gpu::unary_ext_op;
using scalar_op = gpu::scalar_op;

inline const char* wgsl_source_() {
  static const char* src =
#include "tensorlib_webgpu_wgsl.inc"
      ;
  return src;
}

// The kernel table: a kernel id's WGSL entry point and the OP its family
// selects the operation by (-1: the entry point has no OP), or no entry for an
// id this backend has no kernel for. The counterpart of metal.h's and cuda.h's
// kernel_name_; the OP values are the case labels of the family's switch in
// the .wgsl.
struct kernel {
  const char* entry = nullptr;
  int op = -1;
};
inline kernel kernel_(kop k) {
  switch (k) {
    case kop::add: return {"ew_binary", 0};
    case kop::sub: return {"ew_binary", 1};
    case kop::mul: return {"ew_binary", 2};
    case kop::div: return {"ew_binary", 3};
    case kop::pow_: return {"ew_binary", 4};

    case kop::exp_: return {"ew_unary", 0};
    case kop::log_: return {"ew_unary", 1};
    case kop::sqrt_: return {"ew_unary", 2};
    case kop::sigmoid: return {"ew_unary", 3};
    case kop::relu: return {"ew_unary", 4};
    case kop::affine: return {"ew_unary", 5};
    case kop::tanh_: return {"ew_unary", 6};
    case kop::sin_: return {"ew_unary", 7};
    case kop::cos_: return {"ew_unary", 8};

    case kop::badd: return {"ew_bcast", 0};
    case kop::bsub: return {"ew_bcast", 1};
    case kop::bmul: return {"ew_bcast", 2};
    case kop::bdiv: return {"ew_bcast", 3};
    case kop::bpow: return {"ew_bcast", 4};

    case kop::gt_: return {"cmp", 0};
    case kop::lt_: return {"cmp", 1};
    case kop::ge_: return {"cmp", 2};
    case kop::le_: return {"cmp", 3};
    case kop::eq_: return {"cmp", 4};
    case kop::ne_: return {"cmp", 5};

    case kop::pow_s_: return {"ew_scalar", 0};
    case kop::gt_s_: return {"ew_scalar", 1};
    case kop::lt_s_: return {"ew_scalar", 2};
    case kop::ge_s_: return {"ew_scalar", 3};
    case kop::le_s_: return {"ew_scalar", 4};
    case kop::eq_s_: return {"ew_scalar", 5};
    case kop::ne_s_: return {"ew_scalar", 6};

    case kop::row_sum: return {"row_reduce", 0};
    case kop::row_max: return {"row_reduce", 1};
    case kop::softmax: return {"softmax", -1};
    case kop::clamp_: return {"clamp_", -1};
    case kop::layer_norm_: return {"layer_norm", -1};
    case kop::index_select: return {"index_select", -1};
    default: return {};
  }
}

// The uniform a kernel reads (tensorlib_webgpu.wgsl's header): its views'
// element offsets, one slot a view, then its params struct.
constexpr size_t kMaxViews = 8;
constexpr size_t kViewsBytes = kMaxViews * sizeof(uint32_t);

// WebGPU guarantees maxComputeWorkgroupsPerDimension >= 65535. Anything past
// that returns false and falls to CPU rather than silently truncating.
constexpr int64_t kMaxWorkgroups = 65535;

// A uniform binding's offset must be a multiple of the adapter's
// minUniformBufferOffsetAlignment; 256 is the spec's guaranteed-safe maximum.
constexpr uint64_t kUniformSlotBytes = 256;
// One flush can batch this many launches; past it, launch_ forces a blocking
// flush mid-graph. The ring is pure device memory (1 MB here) with no
// binding-size implication, so it is sized to put that forced stall well
// beyond any graph the backend is aimed at.
constexpr uint32_t kUniformSlotCount = 4096;

// Every WGSL entry point. The context prebuilds a pipeline for each (with its
// OP at the default; the family's other operations are the same code under
// another constant, built on first use) and the browser harness asserts each
// one dispatched — both need the same list, and a new kernel missing from
// either loses a guarantee silently.
inline constexpr const char* kEntryPoints[] = {
    "sgemm",        "ew_binary",   "ew_unary",     "ew_bcast",
    "softmax",      "row_reduce",  "pad",          "fold",
    "index_select", "index_add",  "scatter_axis", "ew_bcast_nd",
    "where_nd",     "cmp",        "clamp_",       "sum_to",
    "concat_part",  "rope",        "ew_scalar",    "layer_norm"};

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

// Byte offsets must be 4-aligned to convert to the element offsets the
// kernels index with. They always are for f32 views; anything else falls to
// the CPU rather than silently truncating.
inline bool elem_off_(int64_t byte_off, uint32_t* out) {
  if (byte_off % 4) return false;
  *out = (uint32_t)(byte_off / 4);
  return true;
}

struct context {
  wgpu::Instance instance;
  wgpu::Device device;
  wgpu::Queue queue;
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

  // The N-D kernels' per-call shape metadata ring — same idea as the uniform
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

  // Host/device mirror per allocation and its size-keyed free list (shared
  // shape with cuda.h; `gpu::mirror_table`, gpu_abi.h), keyed by the opaque
  // handle alloc() returns as `native`. Views sharing a storage share the
  // key, so one dirty state serves every view.
  using mirror = gpu::mirror_table<wgpu::Buffer>::entry;
  gpu::mirror_table<wgpu::Buffer> mt;

  // Mapped-for-readback staging buffers (sync_to_host's D2H), a distinct pool
  // from the mirror table's device buffers: these are never bound as a
  // kernel operand, only mapped.
  std::unordered_map<size_t, std::vector<wgpu::Buffer>> staging_pool;

  // Compute pipelines, keyed by WGSL entry point and OP, each with the bind
  // group layout it took from its entry point (an auto layout: the WGSL's
  // declarations are the kernel's side of the ABI).
  struct pipeline {
    wgpu::ComputePipeline pipe;
    wgpu::BindGroupLayout layout;
    const char* entry;  // for the census and the profile row
  };
  std::unordered_map<std::string, pipeline> pipelines;
  // Each kernel id's pipeline once dispatch has resolved it, so a shared op's
  // launch indexes an array where the map lookup would build a key.
  // unordered_map never moves its values, so the pointers stay good.
  std::array<const pipeline*, gpu::kKopCount> kop_pipelines{};
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

    wgpu::BufferDescriptor ud = {};
    ud.size = kUniformSlotBytes * kUniformSlotCount;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniforms = device.CreateBuffer(&ud);
    if (!uniforms) return;

    // Build every pipeline up front rather than on first use. Lazily, a WGSL
    // error would surface as an op quietly falling back to CPU forever; here
    // it makes available() false, which the harness reports.
    for (const char* ep : kEntryPoints) {
      if (!pipeline_(ep, -1)) return;
    }

    ready = true;
  }

  // `entry` with OP set to `op`. 0 is OP's default in the WGSL, so it and -1
  // (no OP) name the one pipeline the constructor prebuilt. No layout is
  // given, so the pipeline takes the auto layout of what its entry point
  // declares.
  const pipeline* pipeline_(const char* entry, int op) {
    std::string key = entry;
    if (op > 0) key += '#' + std::to_string(op);
    auto it = pipelines.find(key);
    if (it != pipelines.end()) return &it->second;
    wgpu::ConstantEntry constant = {};
    constant.key = "OP";
    constant.value = op;
    wgpu::ComputePipelineDescriptor pd = {};
    pd.compute.module = mod;
    pd.compute.entryPoint = entry;
    if (op > 0) {
      pd.compute.constantCount = 1;
      pd.compute.constants = &constant;
    }
    wgpu::ComputePipeline p = device.CreateComputePipeline(&pd);
    if (!p) return nullptr;
    return &(pipelines[key] = pipeline{p, p.GetBindGroupLayout(0), entry});
  }

  // Blocking wait on a single future — the one place anything suspends.
  bool wait(wgpu::Future f) {
    return instance.WaitAny(f, UINT64_MAX) == wgpu::WaitStatus::Success;
  }

  mirror* mirror_(void* native) { return mt.find(native); }

  // A kernel is about to touch this buffer as `a`: bring the host copy up if
  // residency says so.
  //
  // WriteBuffer executes in queue order, i.e. ahead of anything still sitting
  // in the unsubmitted encoder. That is safe precisely because a buffer whose
  // live bytes are the host's has no encoded command touching it: a pending
  // kernel write would have made the device copy live, and a pending kernel
  // read would have come through here and uploaded already.
  void before_kernel_(mirror& m, gpu::access a) {
    if (m.live.before_kernel(a)) queue.WriteBuffer(m.dev, 0, m.host, m.bytes);
  }

  // The one place a kernel is launched, the shared ops' and this backend's own
  // alike: view i bound whole at binding i, then the uniform — the views'
  // element offsets and the params — at binding n (tensorlib_webgpu.wgsl's
  // header). Nothing here knows a kernel. Declines, so the op falls back to
  // the CPU, for a view with no device buffer or an offset that is not a
  // whole float.
  //
  // Two inputs may be the same buffer: read-only bindings may alias. A buffer
  // bound as an output may not be bound again in the same launch — WebGPU
  // invalidates the whole encoder for it, and with it every launch batched
  // since the last flush — so no caller hands one over twice.
  bool launch_(const pipeline* p, const gpu::arg* args, size_t n,
               const void* params, size_t params_bytes, uint32_t gx,
               uint32_t gy) {
    if (!p || n == 0 || n > kMaxViews ||
        kViewsBytes + params_bytes > kUniformSlotBytes) {
      return false;
    }
    if (gx == 0 || gy == 0 || gx > kMaxWorkgroups || gy > kMaxWorkgroups) {
      return false;
    }
    mirror* views[kMaxViews];
    uint32_t offs[kMaxViews] = {};
    for (size_t i = 0; i < n; i++) {
      views[i] = mirror_(args[i].s.buf);
      if (!views[i] || !elem_off_(args[i].s.off, &offs[i])) return false;
    }

    if (slot >= kUniformSlotCount) flush_();  // ring exhausted; new batch
    for (size_t i = 0; i < n; i++) before_kernel_(*views[i], args[i].a);

    const uint32_t off = slot++ * (uint32_t)kUniformSlotBytes;
    unsigned char u[kUniformSlotBytes];
    std::memcpy(u, offs, kViewsBytes);
    std::memcpy(u + kViewsBytes, params, params_bytes);
    queue.WriteBuffer(uniforms, off, u, kViewsBytes + params_bytes);

    wgpu::BindGroupEntry bge[kMaxViews + 1] = {};
    for (size_t i = 0; i < n; i++) {
      bge[i].binding = static_cast<uint32_t>(i);
      bge[i].buffer = views[i]->dev;
      bge[i].size = views[i]->bytes;
    }
    bge[n].binding = static_cast<uint32_t>(n);
    bge[n].buffer = uniforms;
    bge[n].offset = off;
    bge[n].size = kUniformSlotBytes;
    wgpu::BindGroupDescriptor bgd = {};
    bgd.layout = p->layout;
    bgd.entryCount = n + 1;
    bgd.entries = bge;
    wgpu::BindGroup bg = device.CreateBindGroup(&bgd);

    if (!enc) {
      enc = device.CreateCommandEncoder();
      pass = enc.BeginComputePass();
    }
    pass.SetPipeline(p->pipe);
    pass.SetBindGroup(0, bg);
    pass.DispatchWorkgroups(gx, gy, 1);

    pending = true;
    dispatch_counts[p->entry]++;
    gpu::launched(p->entry);  // counted, untimed
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

// The ops this backend runs its own way: a different algorithm, several
// kernels, or a kernel whose ABI is its own. gpu_ops.h forwards to whichever of
// these exist (TL_GPU_DETECT_OWN) and answers false for the rest, so a backend
// declares what it has and nothing else. Defined below, among their helpers.
struct own {
  static bool binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
                              gpu::span b, const int64_t* b_strides,
                              gpu::span out, const int64_t* out_shape, int rank,
                              int64_t n, float scale, float offset);
  static bool where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
                       const int64_t* a_strides, gpu::span b,
                       const int64_t* b_strides, gpu::span out,
                       const int64_t* out_shape, int rank, int64_t n);
  static bool sum_to(gpu::span a, const int64_t* a_shape,
                     const int64_t* a_strides, const int64_t* acc, int rank,
                     int64_t out_n, int64_t reduced_n, gpu::span out);
  static bool pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                  const int64_t* out_shape, int rank, int axis, int64_t before,
                  int64_t n, int64_t out_n);
  static bool fold(gpu::span a, gpu::span out, const int64_t* a_shape,
                   const int64_t* out_shape, int rank, int axis, int64_t step,
                   int64_t n, int64_t out_n);
  static bool concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
                          const int64_t* out_shape, int rank, int axis,
                          int64_t before, int64_t n);
  static bool index_add(gpu::span idx, gpu::span values, gpu::span out,
                        int64_t row_size, int64_t k, int64_t out_n);
  static bool scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                              int64_t n, int64_t size);
  static bool gemm(gpu::span a, int64_t lda, bool ta, gpu::span b, int64_t ldb,
                   bool tb, gpu::span out, int64_t m, int64_t n, int64_t k,
                   float scale, float offset);
  static bool rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                   int64_t D, int64_t pos, float base, gpu::span bias = {});
};
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
  profile::detail::blocked waiting;
  c.wait(c.queue.OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}));
}

inline void context::flush_() { flush(); }

// Mirror allocation: a device buffer paired with a host buffer (returned via
// `contents`). They are DISTINCT memory — the dirty state copies between them
// on demand. The returned handle is an opaque token, not a pointer to
// anything dereferenceable; it is only ever a key back into `mirrors`.
// `host_fill` (the host writes it first) needs nothing here: the host copy is
// its own malloc, which kernels never write and an upload copies when queued.
inline void* alloc(int64_t bytes, float** contents, bool host_fill = false) {
  auto& c = context::get();
  if (!c.ready) return nullptr;
  size_t nb = bytes > 0 ? (size_t)bytes : 4;
  nb = (nb + 3) & ~size_t(3);  // WebGPU buffer sizes must be 4-byte multiples

  wgpu::Buffer dev;
  float* host = nullptr;
  if (!c.mt.take(nb, dev, host)) {  // no recycled buffer of this exact size
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
  c.mt.insert(token, dev, host, nb, host_fill);
  if (contents) *contents = host;
  return token;
}

inline void release(void* buf, int64_t, float*) {
  auto& c = context::get();
  if (!c.ready || !buf) return;
  c.mt.release(buf);
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
  if (m->live.needs_download()) {
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
        m->live.downloaded();
      }
      stg.Unmap();
    }
    // Only a completed memcpy makes the host copy current. Declaring it so on a
    // failed readback would leave stale bytes permanently believed live, and no
    // later sync_to_host would retry — so say so loudly instead, as the WGSL
    // compile failure above does.
    if (m->live.needs_download()) {
      std::fprintf(stderr, "tensorlib webgpu: readback of %zu bytes failed\n",
                   m->bytes);
    }
    c.staging_pool[m->bytes].push_back(stg);
  }
  if (for_write) m->live.host_wrote();
}

// The device core's one way to run a kernel for the shared ops: the id's
// entry point and OP from the kernel table, and the rest as the shared op
// handed it — views, params, grid — with no per-kernel code.
inline bool dispatch(kop k, const gpu::arg* args, size_t n, const void* params,
                     size_t params_bytes, const gpu::grid& g) {
  auto& c = context::get();
  if (!c.ready) return false;
  const context::pipeline*& p = c.kop_pipelines[static_cast<size_t>(k)];
  if (!p) {
    const kernel kn = kernel_(k);
    if (!kn.entry) return false;
    p = c.pipeline_(kn.entry, kn.op);
  }
  return c.launch_(p, args, n, params, params_bytes, g.gx, g.gy);
}

namespace detail_ {

// This backend's own kernels' params: 4-byte fields, which
// tensorlib_webgpu.wgsl declares in the same order after the views' offsets.
struct gemm_params {
  uint32_t m, n, k, lda, ldb, ldc, ta, tb;
  float scale, offset;
};
struct pad_params { uint32_t n, rank, axis, before; };
struct fold_params { uint32_t n, rank, axis, step; };
struct sum_to_params { uint32_t n, rank, reduced_n; };
struct concat_params { uint32_t n, rank, shift; };
struct bcast_nd_params {
  uint32_t n, rank;
  float scale, offset;
};
struct where_nd_params { uint32_t n, rank; };
struct index_add_params { uint32_t n, row_size, k; };
struct scatter_axis_params { uint32_t n, size; };
struct rope_params {
  uint32_t n, t, d, pos, half;
  float base;
};

inline uint32_t u32(int64_t v) { return static_cast<uint32_t>(v); }

// An own op's launch, through the same launch_ as the shared ops.
template <class P>
inline bool launch_own_(const char* entry, int op,
                        std::initializer_list<gpu::arg> args, const P& p,
                        const gpu::grid& g) {
  auto& c = context::get();
  if (!c.ready) return false;
  return c.launch_(c.pipeline_(entry, op), args.begin(), args.size(), &p,
                   sizeof(P), g.gx, g.gy);
}

}  // namespace detail_

// out = op(a) @ op(b) * scale + offset.
inline bool own::gemm(gpu::span a, int64_t lda, bool ta, gpu::span b,
                      int64_t ldb, bool tb, gpu::span out, int64_t m, int64_t n,
                      int64_t k, float scale, float offset) {
  using detail_::u32;
  if (m <= 0 || n <= 0 || k <= 0) return false;
  // ldc = n: the eval seam always hands us a contiguous output.
  detail_::gemm_params p{u32(m), u32(n), u32(k), u32(lda), u32(ldb), u32(n),
                         ta ? 1u : 0u, tb ? 1u : 0u, scale, offset};
  return detail_::launch_own_("sgemm", -1,
                              {gpu::in(a), gpu::in(b), gpu::out(out)}, p,
                              gpu::grid{u32((n + 63) / 64), u32((m + 63) / 64)});
}

// A ring, not one reused buffer: queue.WriteBuffer runs ahead of whatever is
// still sitting in the unsubmitted encoder (see before_kernel_'s comment
// above), so two N-D calls batched into the same unflushed pass would have
// the second call's metadata write stomp the first launch's not-yet-executed
// read of the same buffer — the exact hazard the uniform ring above
// (kUniformSlotCount) already exists to avoid, just for a second resource.
// One slot comfortably covers the rank-8 cap (pad needs 2*rank <= 16 words,
// fold (rank-1)+rank <= 15, binary_bcast_nd 3*rank <= 24, where_nd's own
// [out_shape, cond_strides, a_strides, b_strides] 4*rank <= 32 -- the largest,
// which is what sizes this), which is why this state lives on `context`
// (meta_ring_tok/meta_ring_host/meta_slot) right beside `slot` instead of as
// a second, independent ring: flush() resets both counters together, the way
// it already resets `slot`.
inline constexpr size_t kMetaSlotWords = 32;
inline constexpr size_t kMetaSlotCount = 4096;

// Rank cap for the N-D kernels — matches cuda.h's own kPadFoldMaxRank (not
// unified with it: the two backends derive their caps from different physical
// constraints, kMetaSlotWords here vs a fixed-size on-stack index array
// there) and kernels/tensorlib_webgpu.wgsl's kPadFoldMaxRank, which the WGSL
// side needs as its own `const` since a shader can't see a host-side C++
// constant.
inline constexpr int kPadFoldMaxRank = 8;

// Allocated once, sized for the whole ring — through the same alloc() pool
// every tensor buffer uses. The token is opaque to eval_one's storage layer,
// so nothing else could mistake it for a live array.
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
// `slot`. A full ring — either ring: the launch this slot is for comes next,
// and were launch_ to flush between them, later calls in the new batch would
// be handed this slot again and rewrite it before the launch had read it —
// flushes first, so a slot and its launch always share a batch.
inline uint32_t context::meta_reserve_slot_() {
  if (meta_slot >= kMetaSlotCount || slot >= kUniformSlotCount) {
    flush();
    meta_slot = 0;
  }
  return (meta_slot++) * static_cast<uint32_t>(kMetaSlotWords);
}

// A run of `len` shape or stride values, one u32 word each in a meta slot.
struct meta_run {
  const int64_t* p;
  int len;
};

// `runs`, back to back, in a fresh slot of the meta ring, as a view an N-D
// kernel reads like any other input; a null span when they overflow a slot
// or the ring could not be allocated.
inline gpu::span meta_(std::initializer_list<meta_run> runs) {
  uint32_t words[kMetaSlotWords];
  size_t count = 0;
  for (const meta_run& r : runs) {
    if (r.len < 0 || count + r.len > kMetaSlotWords) return {};
    for (int d = 0; d < r.len; d++) words[count++] = detail_::u32(r.p[d]);
  }
  auto& c = context::get();
  float* ring_host = nullptr;
  void* ring_tok = c.meta_ring_(&ring_host);
  if (!ring_tok) return {};
  const uint32_t word_off = c.meta_reserve_slot_();
  std::memcpy(reinterpret_cast<uint32_t*>(ring_host) + word_off, words,
              count * 4);
  // This slot only, at its own byte offset, and unconditionally: the ring's
  // residency never settles into one live copy (each slot is written once,
  // read once, never again), so the host and device copies are both live
  // after it rather than one or the other.
  context::mirror* mm = c.mirror_(ring_tok);
  c.queue.WriteBuffer(mm->dev, word_off * 4, words, count * 4);
  mm->live.uploaded();
  return {ring_tok, static_cast<int64_t>(word_off) * 4};
}

// Gather-style pad/fold (M11): unlike CUDA's scatter+atomicAdd, WGSL has no
// float atomicAdd, so both dispatch one invocation per OUTPUT element and
// have it read (pad) or sum (fold) whatever cells of `a` map to it — no
// output cell is ever written by two invocations, so unlike cuda.h's pad/fold
// neither needs a pre-zeroed buffer. `a_shape`/`out_shape` (length `rank`,
// `rank-1` for fold's `out_shape`) ride the meta ring.
inline bool own::pad(gpu::span a, gpu::span out, const int64_t* a_shape,
                     const int64_t* out_shape, int rank, int axis,
                     int64_t before, int64_t n, int64_t out_n) {
  using detail_::u32;
  (void)n;
  if (rank <= 0 || rank > kPadFoldMaxRank || out_n <= 0) return false;
  gpu::span meta = meta_({{out_shape, rank}, {a_shape, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "pad", -1, {gpu::in(a), gpu::in(meta), gpu::out(out)},
      detail_::pad_params{u32(out_n), u32(rank), u32(axis), u32(before)},
      gpu::policy::flat(out_n));
}

// unfold's inverse. Each output element sums over the bounded range of
// window indices `w` whose window covers it (`w*step + k == out coordinate`
// along `axis`, `0 <= k < win`) — the gather twin of cuda.h's scatter+
// atomicAdd fold, needed because WGSL has no float atomicAdd. `a`'s own
// strides aren't part of the metadata: `a` is contiguous (gpu_fold_'s
// contract), so the kernel derives them from `a_shape` itself.
inline bool own::fold(gpu::span a, gpu::span out, const int64_t* a_shape,
                      const int64_t* out_shape, int rank, int axis,
                      int64_t step, int64_t n, int64_t out_n) {
  using detail_::u32;
  (void)n;
  if (rank <= 0 || rank > kPadFoldMaxRank || out_n <= 0) return false;
  const int out_rank = rank - 1;
  gpu::span meta = meta_({{out_shape, out_rank}, {a_shape, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "fold", -1, {gpu::in(a), gpu::in(meta), gpu::out(out)},
      detail_::fold_params{u32(out_n), u32(rank), u32(axis), u32(step)},
      gpu::policy::flat(out_n));
}

// index_select's dual, rewritten as a gather: WGSL has no float atomicAdd,
// the same gap pad/fold above work around, so this sums over every source
// row matching each OUTPUT row instead of scattering into a pre-zeroed
// buffer -- no zeroing needed.
inline bool own::index_add(gpu::span idx, gpu::span values, gpu::span out,
                           int64_t row_size, int64_t k, int64_t out_n) {
  using detail_::u32;
  if (out_n <= 0) return false;
  return detail_::launch_own_(
      "index_add", -1, {gpu::in(idx), gpu::in(values), gpu::out(out)},
      detail_::index_add_params{u32(out_n), u32(row_size), u32(k)},
      gpu::policy::flat(out_n));
}

// One-hot scatter into a new trailing axis, as a gather: out[pos,k] =
// values[pos] where indices[pos] == k, else 0. Every output element reads,
// never writes twice, so -- like index_select -- no zeroing needed.
inline bool own::scatter_to_axis(gpu::span idx, gpu::span values, gpu::span out,
                                 int64_t n, int64_t size) {
  using detail_::u32;
  const int64_t out_n = n * size;
  if (out_n <= 0) return false;
  return detail_::launch_own_(
      "scatter_axis", -1, {gpu::in(idx), gpu::in(values), gpu::out(out)},
      detail_::scatter_axis_params{u32(out_n), u32(size)},
      gpu::policy::flat(out_n));
}

// N-D broadcast binary: generalizes binary_bcast() to any rank (a
// Transformer's [N,S,D] LayerNorm broadcasting a [N,S,1] mean, rank 3).
// a_strides/b_strides are the broadcast strides (0 on a broadcast axis)
// array.h computes host-side via the same broadcast_strides() the CPU oracle
// uses. `op` is the rank-2 broadcast id (badd..bpow), whose OP this kernel
// shares.
inline bool own::binary_bcast_nd(kop op, gpu::span a, const int64_t* a_strides,
                                 gpu::span b, const int64_t* b_strides,
                                 gpu::span out, const int64_t* out_shape,
                                 int rank, int64_t n, float scale,
                                 float offset) {
  using detail_::u32;
  if (op < kop::badd || op > kop::bpow) return false;
  if (rank <= 0 || rank > kPadFoldMaxRank || n <= 0) return false;
  gpu::span meta =
      meta_({{out_shape, rank}, {a_strides, rank}, {b_strides, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "ew_bcast_nd", kernel_(op).op,
      {gpu::in(a), gpu::in(b), gpu::in(meta), gpu::out(out)},
      detail_::bcast_nd_params{u32(n), u32(rank), scale, offset},
      gpu::policy::flat(n));
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch, each operand
// through its own broadcast strides.
inline bool own::where_nd(gpu::span cond, const int64_t* c_strides, gpu::span a,
                          const int64_t* a_strides, gpu::span b,
                          const int64_t* b_strides, gpu::span out,
                          const int64_t* out_shape, int rank, int64_t n) {
  using detail_::u32;
  if (rank <= 0 || rank > kPadFoldMaxRank || n <= 0) return false;
  gpu::span meta = meta_({{out_shape, rank}, {c_strides, rank},
                          {a_strides, rank}, {b_strides, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "where_nd", -1,
      {gpu::in(cond), gpu::in(a), gpu::in(b), gpu::in(meta), gpu::out(out)},
      detail_::where_nd_params{u32(n), u32(rank)}, gpu::policy::flat(n));
}

// sum_to (un-broadcast a gradient): gather, mirrors cuda.h's tl_sum_to and
// metal.h's own sum_to -- one invocation per OUTPUT element sums every `a`
// element that broadcasts onto it, so no atomics (unlike index_add).
inline bool own::sum_to(gpu::span a, const int64_t* a_shape,
                        const int64_t* a_strides, const int64_t* acc, int rank,
                        int64_t out_n, int64_t reduced_n, gpu::span out) {
  using detail_::u32;
  if (rank <= 0 || rank > kPadFoldMaxRank || out_n <= 0) return false;
  gpu::span meta = meta_({{a_shape, rank}, {a_strides, rank}, {acc, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "sum_to", -1, {gpu::in(a), gpu::in(meta), gpu::out(out)},
      detail_::sum_to_params{u32(out_n), u32(rank), u32(reduced_n)},
      gpu::policy::flat(out_n));
}

// concat_part (Tensor.concat along an arbitrary axis, KV-cache append):
// scatters `a` (this part) into `out` at a flat element shift along one
// axis -- see kernels/tensorlib_webgpu.wgsl's own concat_part for why this
// is its own entry rather than reusing pad's (that one dispatches over
// OUTPUT elements for its zero border; concat has no border and wants the
// much smaller SOURCE element count instead). out_strides are computed
// host-side, as cuda.h's upload_pad_fold_meta_ does.
inline bool own::concat_part(gpu::span a, gpu::span out, const int64_t* a_shape,
                             const int64_t* out_shape, int rank, int axis,
                             int64_t before, int64_t n) {
  using detail_::u32;
  if (rank <= 0 || rank > kPadFoldMaxRank || n <= 0) return false;
  int64_t out_strides[kPadFoldMaxRank];
  contiguous_strides_into(out_shape, rank, out_strides);
  gpu::span meta = meta_({{a_shape, rank}, {out_strides, rank}});
  if (!meta) return false;
  return detail_::launch_own_(
      "concat_part", -1, {gpu::in(a), gpu::in(meta), gpu::out(out)},
      detail_::concat_params{u32(n), u32(rank), u32(before * out_strides[axis])},
      gpu::policy::flat(n));
}

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors cuda.h's/metal.h's own rope. x is [rows, D]
// contiguous (rows = H*T); dispatched flat over rows*(D/2), one invocation a
// rotated pair.
inline bool own::rope(gpu::span x, gpu::span out, int64_t rows, int64_t T,
                      int64_t D, int64_t pos, float base, gpu::span bias) {
  using detail_::u32;
  if (D <= 0 || (D & 1) || bias.buf) return false;  // no fused bias
  const int64_t half = D / 2;
  const int64_t n = rows * half;
  if (n <= 0) return false;
  return detail_::launch_own_(
      "rope", -1, {gpu::in(x), gpu::out(out)},
      detail_::rope_params{u32(n), u32(T), u32(D), u32(pos), u32(half), base},
      gpu::policy::flat(n));
}

// What a model may ask of this backend beyond the kernel contract (gpu.h),
// and the graph-capture group it names: none of it here, so each answers
// false or does nothing and a decoder takes its host-position path.
// What the shared launch policy (gpu_ops.h) may assume of this backend's
// kernels.
struct traits {
  // A [rows, cols] elementwise kernel reads its cell from a 2-D thread
  // position rather than a flat index.
  static constexpr bool cells_2d = true;
  // A launch's tl::profile row is counted, not timed: WebGPU has no per-launch
  // device time to stamp it with.
  static constexpr bool times_launches = false;
  // Workgroups that keep the device busy. Untuned: the browser hides the
  // device, and no op here splits against it yet.
  static constexpr int64_t fill_groups = 64;
};

struct caps {
  static constexpr bool graph_capture = false;
  static constexpr bool row_gemv = false;
  static constexpr bool bf16_gemm = false;
};
using graph_exec = void*;
inline bool graph_available() { return false; }
inline bool capture_begin() { return false; }
inline graph_exec capture_end() { return nullptr; }
inline bool graph_launch(graph_exec) { return false; }
inline void graph_destroy(graph_exec) {}
inline void upload(void*, const float*, int64_t) {}
inline void upload_u32(void*, unsigned) {}
inline int64_t attn_dpos_partials_bytes(int64_t, int64_t, int64_t) { return 0; }

// Every CPU-side buffer read funnels through array::raw()/data(), which call
// this: one choke point makes mixed CPU/GPU graphs safe.
inline void cpu_barrier() {
  if (pending()) flush();
}

}  // namespace webgpu
}  // namespace tl

#endif  // TENSORLIB_WEBGPU && __EMSCRIPTEN__
