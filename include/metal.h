#pragma once

// Metal GPU backend (macOS).
//
// Depends only on storage-free primitives (raw MTLBuffer handles + byte
// offsets), so storage.h can build on it without a cycle. Design points,
// informed by silarray:
//   - Unified memory: when a device exists, every tl::storage buffer is a
//     shared-mode MTLBuffer from a size-keyed pool — CPU and GPU read the
//     same bytes, no residency tracking, no transfers.
//   - One long-lived command buffer/encoder: dispatches accumulate without
//     committing; flush() (end + commit + waitUntilCompleted) runs when the
//     graph evaluation finishes or a CPU-side read needs the data.
//   - Kernels JIT-compile once from the #embed'd MSL source on first GPU
//     dispatch. Editing metal_kernels.metal requires rebuilding the host.
//
// On non-Apple builds everything is an inline stub returning false/null, so
// callers carry no platform conditionals.

#include <cstdint>

#ifdef __APPLE__

#include <objc.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void* MTLCreateSystemDefaultDevice(void);
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

#endif

namespace tl {
namespace metal {

enum class kop {
  add, sub, mul, div, pow_, exp_, log_, sqrt_, sigmoid, relu, affine,
  badd, bsub, bmul, bdiv, bpow,  // rank-2 broadcast binary (strided operands)
  sgemm32, sgemm32x64, sgemm64x32, sgemm64,
  steel, steel32x64, steel_ta, steel_tb, steel32x64_ta, steel32x64_tb,
  softmax, row_sum, row_max, pad, fold,
  index_select, index_add, scatter_axis,
  badd_nd, bsub_nd, bmul_nd, bdiv_nd, bpow_nd,  // N-D broadcast binary
  where_nd,
  gt_, lt_, ge_, le_, eq_, ne_,  // comparisons -- cmp_op maps onto these
  tanh_, sin_, cos_,             // unary_ext_op maps onto these
  clamp_, sum_to_,               // dedicated ops, mirroring cuda.h's own
  concat_part_, rope_            // ditto -- Tensor.concat / RoPE's own dispatch
};

// Comparisons (gt/lt/ge/le/eq/ne) are deliberately NOT kop values: kop is
// called unconditionally through this file's own pso_()-based binary(),
// which throws rather than declining an enum value it has no MSL kernel
// for -- fine for ops every backend already implements, not for a
// CUDA-only addition landing ahead of its Metal/WebGPU kernels. compare()
// below is its own small vocabulary so an unimplemented backend can just
// return false, same as index_select/index_add/scatter_axis/sum_to.
enum class cmp_op { gt, lt, ge, le, eq, ne };

// tanh_/sin_/cos_: same reasoning as cmp_op above -- a CUDA-only addition,
// so its own vocabulary rather than a new kop. clamp (2 node-specific
// scalars, no epilogue) gets its own dedicated function below instead of
// an enum value, same as index_select/sum_to's own dedicated functions.
enum class unary_ext_op { tanh_, sin_, cos_ };

#ifdef __APPLE__

struct mtl_size {
  unsigned long w, h, d;
};

struct context {
  objc::id device = nullptr;
  objc::id queue = nullptr;
  objc::id library = nullptr;
  objc::id cb = nullptr;   // command buffer (while pending)
  objc::id enc = nullptr;  // compute encoder (while pending)
  void* pool = nullptr;    // autorelease pool for the pending batch
  bool pending = false;
  // Free-list by byte size; contents pointers cached so a pool hit costs no
  // objc round trip (tiny-tensor workloads allocate per op).
  std::unordered_map<int64_t, std::vector<std::pair<void*, float*>>> free_bufs;
  std::unordered_map<int, objc::id> psos;

  static context& get() {
    static auto* c = new context();  // leaked: outlives all storage deleters
    return *c;
  }

  context() {
    device = MTLCreateSystemDefaultDevice();
    if (device) queue = objc::send(device, "newCommandQueue");
  }

  static const char* msl_source_() {
    static const char src[] = {
#embed "metal_kernels.metal"
        , '\0'};
    return src;
  }

  static const char* kernel_name_(kop op) {
    switch (op) {
      case kop::add: return "add_";
      case kop::sub: return "sub_";
      case kop::mul: return "mul_";
      case kop::div: return "div_";
      case kop::pow_: return "pow_";
      case kop::badd: return "badd_";
      case kop::bsub: return "bsub_";
      case kop::bmul: return "bmul_";
      case kop::bdiv: return "bdiv_";
      case kop::bpow: return "bpow_";
      case kop::exp_: return "exp_";
      case kop::log_: return "log_";
      case kop::sqrt_: return "sqrt_";
      case kop::sigmoid: return "sigmoid_";
      case kop::relu: return "relu_";
      case kop::affine: return "affine_";
      case kop::sgemm32: return "sgemm_32_";
      case kop::sgemm32x64: return "sgemm_32x64_";
      case kop::sgemm64x32: return "sgemm_64x32_";
      case kop::sgemm64: return "sgemm_64_";
      case kop::steel: return "sgemm_steel_";
      case kop::steel32x64: return "sgemm_steel_32x64_";
      case kop::steel_ta: return "sgemm_steel_ta_";
      case kop::steel_tb: return "sgemm_steel_tb_";
      case kop::steel32x64_ta: return "sgemm_steel_32x64_ta_";
      case kop::steel32x64_tb: return "sgemm_steel_32x64_tb_";
      case kop::softmax: return "softmax_";
      case kop::row_sum: return "row_sum_";
      case kop::row_max: return "row_max_";
      case kop::pad: return "pad_";
      case kop::fold: return "fold_";
      case kop::index_select: return "index_select_";
      case kop::index_add: return "index_add_";
      case kop::scatter_axis: return "scatter_axis_";
      case kop::badd_nd: return "badd_nd_";
      case kop::bsub_nd: return "bsub_nd_";
      case kop::bmul_nd: return "bmul_nd_";
      case kop::bdiv_nd: return "bdiv_nd_";
      case kop::bpow_nd: return "bpow_nd_";
      case kop::where_nd: return "where_nd_";
      case kop::gt_: return "gt_";
      case kop::lt_: return "lt_";
      case kop::ge_: return "ge_";
      case kop::le_: return "le_";
      case kop::eq_: return "eq_";
      case kop::ne_: return "ne_";
      case kop::tanh_: return "tanh_";
      case kop::sin_: return "sin_";
      case kop::cos_: return "cos_";
      case kop::clamp_: return "clamp_";
      case kop::sum_to_: return "sum_to_";
      case kop::concat_part_: return "concat_part_";
      case kop::rope_: return "rope_";
    }
    return "";
  }

  objc::id pso_(kop op) {
    auto it = psos.find(static_cast<int>(op));
    if (it != psos.end()) return it->second;
    if (!library) {
      objc::id err = nullptr;
      auto src = objc::send(objc::cls("NSString"), "stringWithUTF8String:",
                            msl_source_());
      library = objc::send(device, "newLibraryWithSource:options:error:", src,
                           static_cast<objc::id>(nullptr), &err);
      if (!library) {
        throw std::runtime_error("tl::metal: MSL compile failed: " +
                                 objc::error_str(err));
      }
    }
    auto name = objc::send(objc::cls("NSString"), "stringWithUTF8String:",
                           kernel_name_(op));
    auto fn = objc::send(library, "newFunctionWithName:", name);
    objc::id err = nullptr;
    auto pso = objc::send(device, "newComputePipelineStateWithFunction:error:",
                          fn, &err);
    if (!pso) {
      throw std::runtime_error("tl::metal: PSO creation failed for " +
                               std::string(kernel_name_(op)) + ": " +
                               objc::error_str(err));
    }
    psos[static_cast<int>(op)] = pso;
    return pso;
  }

  void ensure_encoder_() {
    if (cb) return;
    pool = objc_autoreleasePoolPush();
    cb = objc::send(queue, "commandBuffer");
    enc = objc::send(cb, "computeCommandEncoder");
    pending = true;
  }
};

inline bool available() { return context::get().device != nullptr; }

inline bool pending() { return context::get().pending; }

// End the batch: commit and block until the GPU finishes (MLX-style eval).
inline void flush() {
  auto& c = context::get();
  if (!c.pending) return;
  objc::send(c.enc, "endEncoding");
  objc::send(c.cb, "commit");
  objc::send(c.cb, "waitUntilCompleted");
  objc_autoreleasePoolPop(c.pool);
  c.cb = c.enc = nullptr;
  c.pool = nullptr;
  c.pending = false;
}

// Pooled shared-mode MTLBuffer. Returns null when no device (caller falls
// back to heap). `bytes` is the pool key — pass the same value to release.
inline void* alloc(int64_t bytes, float** contents) {
  auto& c = context::get();
  if (!c.device) return nullptr;
  auto it = c.free_bufs.find(bytes);
  if (it != c.free_bufs.end() && !it->second.empty()) {
    auto [buf, ptr] = it->second.back();
    it->second.pop_back();
    *contents = ptr;
    return buf;
  }
  // MTLResourceStorageModeShared = 0
  void* buf = objc::send(c.device, "newBufferWithLength:options:",
                         static_cast<unsigned long>(bytes), 0ul);
  if (!buf) return nullptr;
  *contents = static_cast<float*>(objc::send(buf, "contents"));
  return buf;
}

inline void release(void* buf, int64_t bytes, float* contents) {
  context::get().free_bufs[bytes].emplace_back(buf, contents);
}

namespace detail_ {

struct ew_params {
  float scale;
  float offset;
  uint32_t n;
};

inline void dispatch_(objc::id enc, objc::id pso, const ew_params& p,
                      unsigned long params_index) {
  objc::send(enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), params_index);
  unsigned long groups = (p.n + 255ul) / 256ul;
  using dispatch_fn = void (*)(objc::id, objc::sel_t, mtl_size, mtl_size);
  reinterpret_cast<dispatch_fn>(objc_msgSend)(
      enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
      mtl_size{groups, 1, 1}, mtl_size{256, 1, 1});
  (void)pso;
}

}  // namespace detail_

// Contiguous elementwise dispatches; offsets in bytes. Epilogue (scale,
// offset) applies inside the kernel. Encodes without committing.
inline bool binary(kop op, void* a, int64_t ao, void* b, int64_t bo, void* out,
                   int64_t oo, int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  objc::send(c.enc, "setBuffer:offset:atIndex:", a,
             static_cast<unsigned long>(ao), 0ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", b,
             static_cast<unsigned long>(bo), 1ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", out,
             static_cast<unsigned long>(oo), 2ul);
  detail_::dispatch_(c.enc, pso,
                     {scale, offset, static_cast<uint32_t>(n)}, 3ul);
  return true;
}

inline bool unary(kop op, void* a, int64_t ao, void* out, int64_t oo,
                  int64_t n, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  objc::send(c.enc, "setBuffer:offset:atIndex:", a,
             static_cast<unsigned long>(ao), 0ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", out,
             static_cast<unsigned long>(oo), 1ul);
  detail_::dispatch_(c.enc, pso,
                     {scale, offset, static_cast<uint32_t>(n)}, 2ul);
  return true;
}

namespace detail_ {
struct ew_bcast_params {
  float scale;
  float offset;
  uint32_t M, N;
  uint32_t ars, acs, brs, bcs;  // per-operand row/col strides (elements)
};
}  // namespace detail_

// Rank-2 broadcast binary: out[r,c] = f(a[r*ars+c*acs], b[r*brs+c*bcs]) into a
// contiguous [M,N] output. One kernel covers every rank-2 broadcast (row
// vector, column vector, per-row scalar), keeping bias/gamma/beta chains on
// the GPU instead of a CPU fallback that drains the pipeline. Encodes without
// committing, like binary().
inline bool binary_bcast(kop op, void* a, int64_t ao, int64_t ars, int64_t acs,
                         void* b, int64_t bo, int64_t brs, int64_t bcs,
                         void* out, int64_t oo, int64_t m, int64_t n,
                         float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  objc::send(c.enc, "setBuffer:offset:atIndex:", a,
             static_cast<unsigned long>(ao), 0ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", b,
             static_cast<unsigned long>(bo), 1ul);
  objc::send(c.enc, "setBuffer:offset:atIndex:", out,
             static_cast<unsigned long>(oo), 2ul);
  detail_::ew_bcast_params p{
      scale,
      offset,
      static_cast<uint32_t>(m),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(ars),
      static_cast<uint32_t>(acs),
      static_cast<uint32_t>(brs),
      static_cast<uint32_t>(bcs)};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  using fn = void (*)(objc::id, objc::sel_t, mtl_size, mtl_size);
  reinterpret_cast<fn>(objc_msgSend)(
      c.enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
      mtl_size{(static_cast<unsigned long>(n) + 31ul) / 32ul,
               (static_cast<unsigned long>(m) + 7ul) / 8ul, 1},
      mtl_size{32, 8, 1});
  return true;
}

namespace detail_ {

struct gemm_params {
  uint32_t M, N, K, lda, ldb, trans_a, trans_b;
  uint32_t a_fast, b_fast;  // float4 loader eligibility, verified host-side
  float scale, offset;
};

struct reduce_params {
  uint32_t rows, cols;
  float scale, offset;
};

inline void set_buf_(objc::id enc, void* buf, int64_t off, unsigned long idx) {
  objc::send(enc, "setBuffer:offset:atIndex:", buf,
             static_cast<unsigned long>(off), idx);
}

inline void dispatch_grid_(objc::id enc, mtl_size grid, mtl_size tg) {
  using fn = void (*)(objc::id, objc::sel_t, mtl_size, mtl_size);
  reinterpret_cast<fn>(objc_msgSend)(
      enc, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
      grid, tg);
}

}  // namespace detail_

// C(m,n) = (A @ B) * scale + offset. lda/ldb are row strides; trans flags
// let a transposed view be read in place. Buffers are raw MTLBuffers; byte
// offsets fold the view offset in. Encodes without committing.
inline bool gemm(void* a, int64_t ao, int64_t lda, bool ta, void* b,
                 int64_t bo, int64_t ldb, bool tb, void* out, int64_t oo,
                 int64_t m, int64_t n, int64_t k, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  // Dispatch ladder. STEEL (BN=64 bands) covers NN and single-transposed
  // shapes with enough width (the transposing loader reads the view in
  // place); BM band follows silarray (M < 97 → 32×64 tiles). TT or narrow
  // shapes take the simple-tile family, which reads transposed views in
  // place. Gates are provisional pending a full census vs PyTorch-MPS.
  bool steel = !(ta && tb) && m >= 16 && n >= 48 && k >= 16;
  kop kk_;
  unsigned long bm, bn;
  uint32_t fast_a, fast_b;  // STEEL reuses the a_fast slot for swizzle_log
  unsigned long gx, gy;
  if (steel) {
    bool band32 = m < 97;
    kk_ = band32 ? (ta ? kop::steel32x64_ta
                       : tb ? kop::steel32x64_tb : kop::steel32x64)
                 : (ta ? kop::steel_ta : tb ? kop::steel_tb : kop::steel);
    bm = band32 ? 32 : 64;
    bn = 64;
    unsigned long tiles_n = (static_cast<unsigned long>(n) + bn - 1) / bn;
    unsigned long tiles_m = (static_cast<unsigned long>(m) + bm - 1) / bm;
    uint32_t swizzle_log = 0;  // threadgroup swizzle for L2 reuse
    while ((tiles_n >> (swizzle_log + 1)) >= 1 && swizzle_log < 3)
      swizzle_log++;
    fast_a = swizzle_log;
    fast_b = 0;
    gx = tiles_n << swizzle_log;
    gy = (tiles_m + ((1ul << swizzle_log) - 1)) >> swizzle_log;
  } else {
    kk_ = m >= 64 ? kop::sgemm64x32 : kop::sgemm32;
    bm = m >= 64 ? 64 : 32;
    bn = 32;
    // float4 loader eligibility: row-major operand only (Apple GPUs handle
    // the unaligned vector loads; transposed operands use the strided path).
    fast_a = !ta ? 1u : 0u;
    fast_b = !tb ? 1u : 0u;
    gx = (static_cast<unsigned long>(n) + bn - 1) / bn;
    gy = (static_cast<unsigned long>(m) + bm - 1) / bm;
  }
  auto pso = c.pso_(kk_);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a, ao, 0ul);
  detail_::set_buf_(c.enc, b, bo, 1ul);
  detail_::set_buf_(c.enc, out, oo, 2ul);
  detail_::gemm_params p{static_cast<uint32_t>(m),   static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k),   static_cast<uint32_t>(lda),
                         static_cast<uint32_t>(ldb), ta ? 1u : 0u,
                         tb ? 1u : 0u,               fast_a,
                         fast_b,                     scale,
                         offset};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  detail_::dispatch_grid_(c.enc, {gx, gy, 1}, {128, 1, 1});
  return true;
}

// Row-wise op over the last axis (cols): softmax writes rows×cols; row_sum/
// row_max write one value per row (rows), with the affine epilogue.
inline bool row_op(kop op, void* in, int64_t io, void* out, int64_t oo,
                   int64_t rows, int64_t cols, float scale, float offset) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, in, io, 0ul);
  detail_::set_buf_(c.enc, out, oo, 1ul);
  detail_::reduce_params p{static_cast<uint32_t>(rows),
                           static_cast<uint32_t>(cols), scale, offset};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  detail_::dispatch_grid_(c.enc, {static_cast<unsigned long>(rows), 1, 1},
                          {256, 1, 1});
  return true;
}

// Rank cap for pad_/fold_'s GPU dispatch — matches cuda.h's kPadFoldMaxRank
// and metal_kernels.metal's own copy (an MSL kernel can't see a host-side
// C++ constant), and bounds pad_fold_params' fixed-size arrays.
inline constexpr int kPadFoldMaxRank = 8;

namespace detail_ {
struct pad_fold_params {
  uint32_t out_shape[kPadFoldMaxRank];  // pad: length rank; fold: rank-1
  uint32_t a_shape[kPadFoldMaxRank];    // length rank
  uint32_t rank;
  uint32_t axis;
  int32_t shift;  // pad: `before`; fold: `step`
  uint32_t n;     // output element count (dispatch bound)
};

// Shared by pad()/fold() below: the two differ only in which kop to run and
// how many leading dims `out_shape` has (rank for pad, rank-1 for fold,
// since fold's own out has one fewer axis than `a`) — everything else, down
// to the dispatch grid, is identical.
inline bool dispatch_pad_fold_(kop op, void* a_native, int64_t ao,
                               void* out_native, int64_t oo,
                               const int64_t* a_shape,
                               const int64_t* out_shape, int rank,
                               int out_rank, int axis, int64_t shift,
                               int64_t out_n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  set_buf_(c.enc, a_native, ao, 0ul);
  set_buf_(c.enc, out_native, oo, 1ul);
  pad_fold_params p{};
  for (int d = 0; d < out_rank; d++)
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  for (int d = 0; d < rank; d++)
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
  p.rank = static_cast<uint32_t>(rank);
  p.axis = static_cast<uint32_t>(axis);
  p.shift = static_cast<int32_t>(shift);
  p.n = static_cast<uint32_t>(out_n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(out_n) + 255ul) / 256ul;
  dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
}  // namespace detail_

// Gather-style pad/fold (im2col), mirroring kernels/tensorlib_webgpu.wgsl's
// pad/fold: one invocation per OUTPUT element reads (pad) or sums (fold)
// whatever cells of `a` map to it, so — unlike cuda.h's scatter+atomicAdd —
// no output cell is ever written by two invocations, and no pre-zeroed
// buffer or atomics are needed (Metal's device-memory atomics are int/uint
// only, the same gap WGSL has). `a` is required contiguous by array.h's
// gpu_pad_/gpu_fold_, so the metal_kernels.metal side derives a_strides from
// a_shape rather than have them uploaded. Encodes without committing, like
// every other dispatch above.
inline bool pad(void* a_native, int64_t ao, void* out_native, int64_t oo,
                const int64_t* a_shape, const int64_t* out_shape, int rank,
                int axis, int64_t before, int64_t n, int64_t out_n) {
  (void)n;
  return detail_::dispatch_pad_fold_(kop::pad, a_native, ao, out_native, oo,
                                     a_shape, out_shape, rank, rank, axis,
                                     before, out_n);
}

// unfold's inverse: the gather twin of fold — see pad() above for why this
// is a gather rather than a scatter+atomicAdd. `out_shape` has rank-1 dims
// (fold's own out has one fewer axis than `a`); `a`'s last dim is the
// sliding window (size a_shape[rank-1]), and `a`'s `axis` dim is the window
// count.
inline bool fold(void* a_native, int64_t ao, void* out_native, int64_t oo,
                 const int64_t* a_shape, const int64_t* out_shape, int rank,
                 int axis, int64_t step, int64_t n, int64_t out_n) {
  (void)n;
  return detail_::dispatch_pad_fold_(kop::fold, a_native, ao, out_native, oo,
                                     a_shape, out_shape, rank, rank - 1, axis,
                                     step, out_n);
}

namespace detail_ {
struct gather_params {
  uint32_t row_size;
  uint32_t n;
};
struct index_add_params {
  uint32_t row_size;
  uint32_t k;
  uint32_t n;
};
struct scatter_axis_params {
  uint32_t size;
  uint32_t n;
};

// Shared by index_select()/index_add()/scatter_to_axis() below: all three
// are a gather into `out` from two source buffers plus a small params blob,
// differing only in which kop/buffers/params struct they use and how `n`
// (the dispatch bound) is derived -- mirrors dispatch_pad_fold_ above, which
// extracts the same kind of shared tail for pad()/fold().
inline bool dispatch_gather3_(kop op, void* buf0, int64_t off0, void* buf1,
                              int64_t off1, void* out_native, int64_t oo,
                              const void* params, unsigned long params_size,
                              uint32_t n) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(op);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  set_buf_(c.enc, buf0, off0, 0ul);
  set_buf_(c.enc, buf1, off1, 1ul);
  set_buf_(c.enc, out_native, oo, 2ul);
  objc::send(c.enc, "setBytes:length:atIndex:", params, params_size, 3ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
}  // namespace detail_

// Row gather along axis 0: out[i] = a[indices[row(i)]] (a, indices
// contiguous). One thread per output element, no write conflicts.
inline bool index_select(void* a_native, int64_t ao, void* idx_native,
                         int64_t idxo, void* out_native, int64_t oo,
                         int64_t row_size, int64_t k) {
  detail_::gather_params p{static_cast<uint32_t>(row_size),
                           static_cast<uint32_t>(k * row_size)};
  return detail_::dispatch_gather3_(kop::index_select, a_native, ao,
                                    idx_native, idxo, out_native, oo, &p,
                                    sizeof(p), p.n);
}

// index_select's dual, rewritten as a gather: Metal's device-memory atomics
// are int/uint only (no float atomicAdd), the same gap pad_/fold_ above work
// around, so this sums over every source row matching each OUTPUT row
// instead of scattering into a pre-zeroed buffer -- no zeroing needed.
inline bool index_add(void* idx_native, int64_t idxo, void* values_native,
                      int64_t vo, void* out_native, int64_t oo,
                      int64_t row_size, int64_t k, int64_t out_n) {
  detail_::index_add_params p{static_cast<uint32_t>(row_size),
                              static_cast<uint32_t>(k),
                              static_cast<uint32_t>(out_n)};
  return detail_::dispatch_gather3_(kop::index_add, idx_native, idxo,
                                    values_native, vo, out_native, oo, &p,
                                    sizeof(p), p.n);
}

// One-hot scatter into a new trailing axis, as a gather: out[pos,k] =
// values[pos] where indices[pos] == k, else 0. Every output element reads,
// never writes twice, so — like index_select above — no zeroing needed.
inline bool scatter_to_axis(void* idx_native, int64_t idxo,
                            void* values_native, int64_t vo, void* out_native,
                            int64_t oo, int64_t n, int64_t size) {
  detail_::scatter_axis_params p{static_cast<uint32_t>(size),
                                 static_cast<uint32_t>(n * size)};
  return detail_::dispatch_gather3_(kop::scatter_axis, idx_native, idxo,
                                    values_native, vo, out_native, oo, &p,
                                    sizeof(p), p.n);
}

namespace detail_ {
struct bcast_nd_params {
  uint32_t out_shape[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t b_strides[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t n;
  float scale;
  float offset;
};
struct where_nd_params {
  uint32_t out_shape[kPadFoldMaxRank];
  uint32_t c_strides[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t b_strides[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t n;
};

// binary_bcast_nd's incoming `op` is one of the rank-2 kop values (badd etc,
// shared with binary_bcast() above -- array.h's gpu_binary_bcast_nd_ passes
// the same `bk` either kernel would take); map it to its own PSO/kernel name
// here rather than caching the N-D kernel under the rank-2 op's slot in
// context::psos, which pso_() keys by this same enum value.
inline kop to_nd_(kop op) {
  switch (op) {
    case kop::badd: return kop::badd_nd;
    case kop::bsub: return kop::bsub_nd;
    case kop::bmul: return kop::bmul_nd;
    case kop::bdiv: return kop::bdiv_nd;
    case kop::bpow: return kop::bpow_nd;
    default: return op;
  }
}
}  // namespace detail_

// N-D broadcast binary: generalizes binary_bcast() above to any rank (a
// Transformer's [N,S,D] LayerNorm broadcasting a [N,S,1] mean, rank 3).
// a_strides/b_strides are the broadcast strides (0 on a broadcast axis)
// array.h computes host-side via the same broadcast_strides() the CPU
// oracle uses -- mirrors cuda.h's own binary_bcast_nd exactly.
inline bool binary_bcast_nd(kop op, void* a_native, int64_t ao,
                            const int64_t* a_strides, void* b_native,
                            int64_t bo, const int64_t* b_strides,
                            void* out_native, int64_t oo,
                            const int64_t* out_shape, int rank, int64_t n,
                            float scale, float offset) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto pso = c.pso_(detail_::to_nd_(op));
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a_native, ao, 0ul);
  detail_::set_buf_(c.enc, b_native, bo, 1ul);
  detail_::set_buf_(c.enc, out_native, oo, 2ul);
  detail_::bcast_nd_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.b_strides[d] = static_cast<uint32_t>(b_strides[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.n = static_cast<uint32_t>(n);
  p.scale = scale;
  p.offset = offset;
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

// N-D broadcast ternary select: Tensor.where's GPU dispatch. Same flat-index
// decode as binary_bcast_nd above, one more operand -- mirrors cuda.h's own
// where_nd exactly.
inline bool where_nd(void* cond_native, int64_t co, const int64_t* c_strides,
                     void* a_native, int64_t ao, const int64_t* a_strides,
                     void* b_native, int64_t bo, const int64_t* b_strides,
                     void* out_native, int64_t oo, const int64_t* out_shape,
                     int rank, int64_t n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto pso = c.pso_(kop::where_nd);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, cond_native, co, 0ul);
  detail_::set_buf_(c.enc, a_native, ao, 1ul);
  detail_::set_buf_(c.enc, b_native, bo, 2ul);
  detail_::set_buf_(c.enc, out_native, oo, 3ul);
  detail_::where_nd_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
    p.c_strides[d] = static_cast<uint32_t>(c_strides[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.b_strides[d] = static_cast<uint32_t>(b_strides[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 4ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
namespace detail_ {
inline kop to_cmp_(cmp_op op) {
  switch (op) {
    case cmp_op::gt: return kop::gt_;
    case cmp_op::lt: return kop::lt_;
    case cmp_op::ge: return kop::ge_;
    case cmp_op::le: return kop::le_;
    case cmp_op::eq: return kop::eq_;
    case cmp_op::ne: return kop::ne_;
  }
  return kop::gt_;
}
inline kop to_unary_ext_(unary_ext_op op) {
  switch (op) {
    case unary_ext_op::tanh_: return kop::tanh_;
    case unary_ext_op::sin_: return kop::sin_;
    case unary_ext_op::cos_: return kop::cos_;
  }
  return kop::tanh_;
}
struct cmp_params {
  uint32_t n;
  uint32_t bstride;
};
struct clamp_params {
  float lo, hi;
  uint32_t n;
};
struct sum_to_params {
  uint32_t a_shape[kPadFoldMaxRank];
  uint32_t a_strides[kPadFoldMaxRank];
  uint32_t acc[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t out_n;
  uint32_t reduced_n;
};
}  // namespace detail_

// gt/lt/ge/le/eq/ne (array.h's comparison ops, ReLU/LeakyReLU/Clip's
// backward gate): same-shape only (bstride=1) or a scalar b (bstride=0) --
// the two shapes array.h's gpu_compare_ ever dispatches. cmp_op stays its
// own vocabulary at the array.h boundary (see this file's cmp_op comment);
// mapped onto a dedicated kop slot here so it shares pso_()'s caching, same
// idea as binary_bcast_nd's to_nd_ above. Mirrors cuda.h's own compare().
inline bool compare(cmp_op op, void* a, int64_t ao, void* b, int64_t bo,
                    void* out, int64_t oo, int64_t n, int64_t bstride) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(detail_::to_cmp_(op));
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a, ao, 0ul);
  detail_::set_buf_(c.enc, b, bo, 1ul);
  detail_::set_buf_(c.enc, out, oo, 2ul);
  detail_::cmp_params p{static_cast<uint32_t>(n),
                        static_cast<uint32_t>(bstride)};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 3ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
// tanh_/sin_/cos_ (RoPE's trig, RNN/LSTM's tanh): plain elementwise, same
// shape as exp_/sqrt_ above -- unary() already does exactly this dispatch,
// just keyed by a kop array.h doesn't see directly.
inline bool unary_ext(unary_ext_op op, void* a, int64_t ao, void* out,
                      int64_t oo, int64_t n, float scale, float offset) {
  return unary(detail_::to_unary_ext_(op), a, ao, out, oo, n, scale, offset);
}
// clamp(x, lo, hi): Clip's forward. No epilogue -- lo/hi occupy the role
// scale/offset play elsewhere (mirrors cuda.h's own clamp).
inline bool clamp(void* a, int64_t ao, void* out, int64_t oo, int64_t n,
                  float lo, float hi) {
  auto& c = context::get();
  if (!c.device) return false;
  auto pso = c.pso_(kop::clamp_);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a, ao, 0ul);
  detail_::set_buf_(c.enc, out, oo, 1ul);
  detail_::clamp_params p{lo, hi, static_cast<uint32_t>(n)};
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
// sum_to (un-broadcast a gradient): gather, mirrors cuda.h's tl_sum_to --
// one thread per OUTPUT element sums every `a` element that broadcasts
// onto it, so no atomics (unlike index_add).
inline bool sum_to(void* a, int64_t ao, const int64_t* a_shape,
                   const int64_t* a_strides, const int64_t* acc, int rank,
                   int64_t out_n, int64_t reduced_n, void* out, int64_t oo) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  auto pso = c.pso_(kop::sum_to_);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a, ao, 0ul);
  detail_::set_buf_(c.enc, out, oo, 1ul);
  detail_::sum_to_params p{};
  for (int d = 0; d < rank; d++) {
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
    p.a_strides[d] = static_cast<uint32_t>(a_strides[d]);
    p.acc[d] = static_cast<uint32_t>(acc[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.out_n = static_cast<uint32_t>(out_n);
  p.reduced_n = static_cast<uint32_t>(reduced_n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(out_n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}
namespace detail_ {
struct concat_part_params {
  uint32_t out_strides[kPadFoldMaxRank];
  uint32_t a_shape[kPadFoldMaxRank];
  uint32_t rank;
  uint32_t shift;
  uint32_t n;
};
}  // namespace detail_

// concat_part (Tensor.concat along an arbitrary axis, KV-cache append):
// writes `a` (contiguous, this part) into `out` at `shift` (already
// before*out_strides[axis], a flat element offset) -- no zeroing, no
// bounds check, since concat's parts exhaustively cover `out` with no
// border. cuda.h's own concat_part reuses its tl_pad kernel body for this
// (writing a source at an axis-shifted offset is exactly what pad already
// does per element), but this file's pad_ above is a gather dispatched
// over OUTPUT elements (needed for pad's zero border) rather than the much
// smaller SOURCE (this part's own) element count concat wants, so this
// gets its own small kernel instead of reusing pad_'s PSO.
inline bool concat_part(void* a, int64_t ao, void* out, int64_t oo,
                        const int64_t* a_shape, const int64_t* out_shape,
                        int rank, int axis, int64_t before, int64_t n) {
  auto& c = context::get();
  if (!c.device || rank <= 0 || rank > kPadFoldMaxRank) return false;
  int64_t out_strides[kPadFoldMaxRank];
  int64_t acc = 1;
  for (int d = rank - 1; d >= 0; d--) {
    out_strides[d] = acc;
    acc *= out_shape[d];
  }
  auto pso = c.pso_(kop::concat_part_);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, a, ao, 0ul);
  detail_::set_buf_(c.enc, out, oo, 1ul);
  detail_::concat_part_params p{};
  for (int d = 0; d < rank; d++) {
    p.out_strides[d] = static_cast<uint32_t>(out_strides[d]);
    p.a_shape[d] = static_cast<uint32_t>(a_shape[d]);
  }
  p.rank = static_cast<uint32_t>(rank);
  p.shift = static_cast<uint32_t>(before * out_strides[axis]);
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

namespace detail_ {
struct rope_params {
  uint32_t T, D, pos;
  uint32_t half_;
  float base;
  uint32_t n;
};
}  // namespace detail_

// RoPE (rotary position embedding), half-split (GPT-NeoX / HF-llama)
// convention -- mirrors tensorlib_cuda.cu's own tl_rope exactly (no fused
// bias: array.h's gpu_rope_ never passes one). x is [rows, D] contiguous
// (rows = H*T: a [H,T,D] tensor flattened, or [H,D] with T=1); row r's
// head-dim vector sits at position `pos + (r % T)`; pairs (j, j+D/2)
// rotate by angle = position * base^(-2j/D). Dispatched flat over
// rows*(D/2) (CUDA instead grids by row, blocks by D/2 -- this file's own
// kernels are all flat-1D, same as sum_to/compare/unary_ext above).
inline bool rope(void* x, void* out, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base) {
  auto& c = context::get();
  if (!c.device || D <= 0 || (D & 1)) return false;
  int64_t half = D / 2;
  int64_t n = rows * half;
  auto pso = c.pso_(kop::rope_);
  c.ensure_encoder_();
  objc::send(c.enc, "setComputePipelineState:", pso);
  detail_::set_buf_(c.enc, x, 0, 0ul);
  detail_::set_buf_(c.enc, out, 0, 1ul);
  detail_::rope_params p{};
  p.T = static_cast<uint32_t>(T);
  p.D = static_cast<uint32_t>(D);
  p.pos = static_cast<uint32_t>(pos);
  p.half_ = static_cast<uint32_t>(half);
  p.base = base;
  p.n = static_cast<uint32_t>(n);
  objc::send(c.enc, "setBytes:length:atIndex:", static_cast<const void*>(&p),
             static_cast<unsigned long>(sizeof(p)), 2ul);
  unsigned long groups = (static_cast<unsigned long>(n) + 255ul) / 256ul;
  detail_::dispatch_grid_(c.enc, {groups, 1, 1}, {256, 1, 1});
  return true;
}

#else  // !__APPLE__ — stubs so callers carry no platform conditionals

inline bool available() { return false; }
inline bool pending() { return false; }
inline void flush() {}
inline void* alloc(int64_t, float**) { return nullptr; }
inline void release(void*, int64_t, float*) {}
inline bool binary(kop, void*, int64_t, void*, int64_t, void*, int64_t,
                   int64_t, float, float) {
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
inline bool row_op(kop, void*, int64_t, void*, int64_t, int64_t, int64_t,
                   float, float) {
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
inline bool rope(void*, void*, int64_t, int64_t, int64_t, int64_t, float) {
  return false;
}

#endif

// ---- LLM decode ops with no MSL kernel yet (M7/M8/M9) -----------------------
// Outside the #if/#else on purpose: both branches would define them identically,
// and returning false is the whole implementation either way — it sends the
// evaluator down the widen-to-F32 CPU fallback. Keeps the gpu:: facade
// symmetric with CUDA, so the one platform #ifdef stays the namespace alias.
inline bool gemv_f32(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool gemv_bf16(void*, void*, void*, int64_t, int64_t) { return false; }
inline bool attn_decode(void*, void*, void*, void*, int64_t, int64_t, int64_t,
                        int64_t, int64_t, float) {
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

// Host↔device coherence hook (see cuda.h). Metal is genuinely unified memory —
// the CPU and GPU see the same MTLBuffer bytes — so there is nothing to copy;
// cpu_barrier (the flush) is the only synchronization needed. A no-op here keeps
// the eval seam backend-agnostic (the CUDA device-mirror does the real work).
inline void sync_to_host(void*, bool) {}

}  // namespace metal

// tl::gpu_available() and the tl::gpu facade (which backend this build uses)
// live in gpu.h — one place, so array.h's eval seam stays #ifdef-free.

}  // namespace tl
