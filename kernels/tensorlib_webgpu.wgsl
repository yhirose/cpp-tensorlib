// WebGPU compute kernels for cpp-tensorlib (M10). Hand-written WGSL, no
// vendor library — the same stance the Metal and CUDA backends take.
//
// Compiled at first use from the C string in tensorlib_webgpu_wgsl.inc, which
// kernels/gen_wgsl_inc.sh generates from this file. The generated .inc is
// committed because the wasm build is a flat emcc line that never runs CMake
// (the CUDA backend's PTX goes through bin2c for the same reason).
//
// View offsets arrive as ELEMENT offsets in the params block and are folded
// into the indexing here, rather than as bind-group binding offsets: WebGPU
// requires those to be 256-byte aligned, which an arbitrary view offset is
// not. So every binding covers its whole buffer. (CUDA instead folds offsets
// host-side into the pointer it passes, which WebGPU has no equivalent of.)

// One Params struct and one bind group layout serve every kernel here, rather
// than the per-family structs metal_kernels.metal uses. WebGPU's bind group
// ceremony is heavy enough that a second layout would buy nothing: the fields
// each family ignores cost 4 bytes of a 256-byte uniform slot. Kernels that
// take one input (unary, the row reductions) get A bound to B as well — two
// read-only bindings may alias, and the output is always a fresh allocation,
// so no writable binding ever aliases a readable one.
struct Params {
  M : u32,        // gemm rows | elementwise element count | reduce rows
  N : u32,        // gemm cols | reduce cols
  K : u32,
  lda : u32,
  ldb : u32,
  ldc : u32,
  a_off : u32,
  b_off : u32,
  c_off : u32,
  ta : u32,
  tb : u32,
  ars : u32,      // broadcast: per-operand row/col strides, in elements
  acs : u32,
  brs : u32,
  bcs : u32,
  op : u32,       // which operation, within the entry point's family
  scale : f32,
  offset : f32,
  // A uniform-address-space struct has align 16, so its size rounds up to a
  // multiple of 16. Pad explicitly to 96 bytes so the host struct (which the
  // bind group's minBindingSize comes from) matches exactly — a short
  // minBindingSize fails bind group validation for every dispatch.
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  _pad3 : u32,
  _pad4 : u32,
  _pad5 : u32,
};

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             p : Params;
// A third and fourth read-only operand, for kernels A/B alone can't cover:
// binary_bcast_nd's two real tensor operands (a, b) already fill A and B, so
// its shape/stride meta needs D; where_nd's three real operands (cond, a, b)
// fill A, B and D, so its meta needs E. Every other entry point ignores
// these (webgpu.h's encode_ binds them to A when a kernel has no use for
// them — bind group validation requires every declared binding be present
// regardless of which bindings the active entry point actually reads).
@group(0) @binding(4) var<storage, read>       D : array<f32>;
@group(0) @binding(5) var<storage, read>       E : array<f32>;

// ---- sgemm: C(M,N) = op(A)(M,K) @ op(B)(K,N) * scale + offset
//
// 64x64 workgroup tile, 16x16 = 256 invocations, each holding a 4x4 register
// accumulator. Mirrors the shape of the Metal sgemm_64_ kernel; MMA intrinsics
// have no WGSL equivalent, so the inner product is plain FMA over registers.
// Measured at ~580-630 GF/s for n=1024 on an M1 Pro (see spike/webgpu).

const BM : u32 = 64u;
const BN : u32 = 64u;
const BK : u32 = 16u;
const TM : u32 = 4u;
const TN : u32 = 4u;
const THREADS : u32 = 256u;

var<workgroup> As : array<f32, 1024>;  // BM * BK
var<workgroup> Bs : array<f32, 1024>;  // BK * BN

// Row/column strides for a possibly-transposed operand: transposing swaps
// which axis walks by the leading dimension (cf. metal_kernels.metal:119-120).
fn a_index(m : u32, k : u32) -> u32 {
  let rs = select(p.lda, 1u, p.ta == 1u);
  let cs = select(1u, p.lda, p.ta == 1u);
  return p.a_off + m * rs + k * cs;
}

fn b_index(k : u32, n : u32) -> u32 {
  let rs = select(p.ldb, 1u, p.tb == 1u);
  let cs = select(1u, p.ldb, p.tb == 1u);
  return p.b_off + k * rs + n * cs;
}

@compute @workgroup_size(16, 16, 1)
fn sgemm(@builtin(workgroup_id) wg : vec3<u32>,
         @builtin(local_invocation_id) lid : vec3<u32>) {
  let m_base = wg.y * BM;
  let n_base = wg.x * BN;
  let tid = lid.y * 16u + lid.x;

  var acc : array<f32, 16>;  // TM * TN, zero-initialized

  let n_tiles = (p.K + BK - 1u) / BK;
  for (var kt : u32 = 0u; kt < n_tiles; kt = kt + 1u) {
    let k0 = kt * BK;

    // Stage A tile (BM x BK) and B tile (BK x BN), 4 elements per invocation
    // each. Out-of-range reads are zero-filled so the tail tile needs no
    // special case in the inner loop.
    for (var s : u32 = 0u; s < 4u; s = s + 1u) {
      let i = tid + s * THREADS;

      let am = m_base + i / BK;
      let ak = k0 + i % BK;
      let a_ok = am < p.M && ak < p.K;
      As[i] = select(0.0, A[a_index(am, ak)], a_ok);

      let bk = k0 + i / BN;
      let bn = n_base + i % BN;
      let b_ok = bk < p.K && bn < p.N;
      Bs[i] = select(0.0, B[b_index(bk, bn)], b_ok);
    }

    workgroupBarrier();

    for (var kk : u32 = 0u; kk < BK; kk = kk + 1u) {
      var av : array<f32, 4>;
      var bv : array<f32, 4>;
      for (var i : u32 = 0u; i < TM; i = i + 1u) {
        av[i] = As[(lid.y * TM + i) * BK + kk];
      }
      for (var j : u32 = 0u; j < TN; j = j + 1u) {
        bv[j] = Bs[kk * BN + lid.x * TN + j];
      }
      for (var i : u32 = 0u; i < TM; i = i + 1u) {
        for (var j : u32 = 0u; j < TN; j = j + 1u) {
          acc[i * TN + j] = fma(av[i], bv[j], acc[i * TN + j]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var i : u32 = 0u; i < TM; i = i + 1u) {
    let m = m_base + lid.y * TM + i;
    if (m >= p.M) { continue; }
    for (var j : u32 = 0u; j < TN; j = j + 1u) {
      let n = n_base + lid.x * TN + j;
      if (n >= p.N) { continue; }
      C[p.c_off + m * p.ldc + n] = acc[i * TN + j] * p.scale + p.offset;
    }
  }
}

// ---- Elementwise, broadcast and row reductions
//
// WGSL has neither templates nor a preprocessor, so the per-op variants that
// metal_kernels.metal generates from a macro would have to be copy-pasted
// here — exactly the edge-tile bug class that file's header warns against.
// Instead the operation is a uniform field and each family is ONE entry point
// that switches on it. The branch is uniform across the dispatch and these
// kernels are memory-bound, so it costs nothing measurable; what it buys is a
// single copy of every bounds check and epilogue.
//
// Op codes are assigned by kernel_op_() in webgpu.h.
const OP_ADD : u32 = 0u;
const OP_SUB : u32 = 1u;
const OP_MUL : u32 = 2u;
const OP_DIV : u32 = 3u;
const OP_POW : u32 = 4u;

const OP_EXP     : u32 = 0u;
const OP_LOG     : u32 = 1u;
const OP_SQRT    : u32 = 2u;
const OP_SIGMOID : u32 = 3u;
const OP_RELU    : u32 = 4u;
const OP_AFFINE  : u32 = 5u;
const OP_TANH    : u32 = 6u;
const OP_SIN     : u32 = 7u;
const OP_COS     : u32 = 8u;

const OP_ROW_SUM : u32 = 0u;
const OP_ROW_MAX : u32 = 1u;

const OP_GT : u32 = 0u;
const OP_LT : u32 = 1u;
const OP_GE : u32 = 2u;
const OP_LE : u32 = 3u;
const OP_EQ : u32 = 4u;
const OP_NE : u32 = 5u;

// Identity for a max reduction. WGSL has no -inf literal, and the decimal
// spelling of f32::lowest rounds just past the representable range ("cannot be
// represented as 'f32'"), so this is the largest round number safely inside
// it. Threads whose row is shorter than the workgroup contribute this.
const NEG_HUGE : f32 = -3.4e38;

// Shared by the contiguous and the broadcast binary: same five operations,
// only the addressing differs.
fn binary_op(op : u32, av : f32, bv : f32) -> f32 {
  switch (op) {
    case 1u: { return av - bv; }
    case 2u: { return av * bv; }
    case 3u: { return av / bv; }
    case 4u: { return pow(av, bv); }
    default: { return av + bv; }
  }
}

fn unary_op(op : u32, v : f32) -> f32 {
  switch (op) {
    case 1u: { return log(v); }
    case 2u: { return sqrt(v); }
    case 3u: { return 1.0 / (1.0 + exp(-v)); }
    case 4u: { return max(v, 0.0); }
    case 5u: { return v; }
    case 6u: { return tanh(v); }
    case 7u: { return sin(v); }
    case 8u: { return cos(v); }
    default: { return exp(v); }
  }
}

// Elementwise comparison: out = (a OP b) ? 1.0 : 0.0. No scale/offset -- masks
// don't compose with the affine epilogue, so ew_cmp below skips it (unlike
// ew_binary/ew_unary). Own family/entry point, not folded into binary_op,
// since it returns a bool-as-float rather than composing with `bv`.
fn cmp_op(op : u32, av : f32, bv : f32) -> f32 {
  switch (op) {
    case 1u: { return select(0.0, 1.0, av < bv); }
    case 2u: { return select(0.0, 1.0, av >= bv); }
    case 3u: { return select(0.0, 1.0, av <= bv); }
    case 4u: { return select(0.0, 1.0, av == bv); }
    case 5u: { return select(0.0, 1.0, av != bv); }
    default: { return select(0.0, 1.0, av > bv); }
  }
}

// Contiguous elementwise binary over p.M elements.
@compute @workgroup_size(256, 1, 1)
fn ew_binary(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let v = binary_op(p.op, A[p.a_off + i], B[p.b_off + i]);
  C[p.c_off + i] = fma(v, p.scale, p.offset);
}

@compute @workgroup_size(256, 1, 1)
fn ew_unary(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let v = unary_op(p.op, A[p.a_off + i]);
  C[p.c_off + i] = fma(v, p.scale, p.offset);
}

// Elementwise comparison over p.M elements: out = (a OP b) ? 1.0 : 0.0
// (no epilogue -- masks don't compose with scale/offset). p.ars carries the
// bstride webgpu.h's compare() receives: 1 for a same-shape b, 0 for a
// scalar b (the concrete ReLU-style masked-gate shape array.h's `x > 0.0f`
// produces) -- an unused field for this family, repurposed rather than
// widening Params.
@compute @workgroup_size(256, 1, 1)
fn cmp(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let bv = B[p.b_off + i * p.ars];
  C[p.c_off + i] = cmp_op(p.op, A[p.a_off + i], bv);
}

// clamp(x, lo, hi): Clip's forward. No affine epilogue -- p.scale/p.offset
// carry lo/hi instead (a dedicated entry, same as this family's clamp_ in
// metal_kernels.metal/tl_clamp in tensorlib_cuda.cu). Named clamp_, not
// clamp, so the entry point doesn't shadow WGSL's builtin of that name.
@compute @workgroup_size(256, 1, 1)
fn clamp_(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  C[p.c_off + i] = clamp(A[p.a_off + i], p.scale, p.offset);
}

// Rank-2 broadcast binary: out[r,c] = f(a[r*ars + c*acs], b[r*brs + c*bcs])
// into a contiguous [M,N] output. Per-operand strides express every rank-2
// broadcast (row vector, column vector, scalar) in one kernel, which keeps
// bias/gamma/beta chains on the GPU — a CPU fallback mid-graph costs a full
// submit-and-wait, and that is dearer here than on Metal.
@compute @workgroup_size(32, 8, 1)
fn ew_bcast(@builtin(global_invocation_id) gid : vec3<u32>) {
  let c = gid.x;
  let r = gid.y;
  if (c >= p.N || r >= p.M) { return; }
  let av = A[p.a_off + r * p.ars + c * p.acs];
  let bv = B[p.b_off + r * p.brs + c * p.bcs];
  C[p.c_off + r * p.N + c] = fma(binary_op(p.op, av, bv), p.scale, p.offset);
}

// ---- Row reductions over the last axis: one workgroup per row, 256
// invocations, workgroup-scratch tree reduction. p.N (cols) may exceed the
// invocation count, so each thread strides over the row first.

const T : u32 = 256u;
var<workgroup> scratch : array<f32, 256>;

fn reduce_op(op : u32, acc : f32, v : f32) -> f32 {
  if (op == OP_ROW_MAX) { return max(acc, v); }
  return acc + v;
}

// Tree-reduce one value per invocation through `scratch` and broadcast the
// result to the whole workgroup. Call only from uniform control flow — it
// barriers. The trailing barrier is what makes `scratch` safe to reuse for a
// second reduction in the same entry point (softmax's max phase, then its sum
// phase): without it, a fast invocation could overwrite scratch[0] before a
// slow one has read it.
fn tree_reduce(op : u32, lid : u32, v : f32) -> f32 {
  scratch[lid] = v;
  workgroupBarrier();
  for (var s : u32 = T / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) { scratch[lid] = reduce_op(op, scratch[lid], scratch[lid + s]); }
    workgroupBarrier();
  }
  let r = scratch[0];
  workgroupBarrier();
  return r;
}

// Numerically stable softmax (subtract the row max). Applying an affine
// epilogue to a softmax is not meaningful, so scale/offset are ignored here,
// as they are on Metal.
@compute @workgroup_size(256, 1, 1)
fn softmax(@builtin(workgroup_id) wg : vec3<u32>,
           @builtin(local_invocation_index) lid : u32) {
  let row = wg.x;
  let src = p.a_off + row * p.N;
  let dst = p.c_off + row * p.N;

  var m = NEG_HUGE;
  for (var c : u32 = lid; c < p.N; c = c + T) { m = max(m, A[src + c]); }
  let row_max = tree_reduce(OP_ROW_MAX, lid, m);

  var sum = 0.0;
  for (var c : u32 = lid; c < p.N; c = c + T) { sum = sum + exp(A[src + c] - row_max); }
  let inv = 1.0 / tree_reduce(OP_ROW_SUM, lid, sum);

  for (var c : u32 = lid; c < p.N; c = c + T) {
    C[dst + c] = exp(A[src + c] - row_max) * inv;
  }
}

// One value per row, with the affine epilogue.
@compute @workgroup_size(256, 1, 1)
fn row_reduce(@builtin(workgroup_id) wg : vec3<u32>,
              @builtin(local_invocation_index) lid : u32) {
  let row = wg.x;
  let src = p.a_off + row * p.N;

  var acc = select(0.0, NEG_HUGE, p.op == OP_ROW_MAX);
  for (var c : u32 = lid; c < p.N; c = c + T) {
    acc = reduce_op(p.op, acc, A[src + c]);
  }
  let r = tree_reduce(p.op, lid, acc);
  if (lid == 0u) { C[p.c_off + row] = r * p.scale + p.offset; }
}

// ---- im2col's pad/fold (M11)
//
// Both dispatch one invocation per OUTPUT element and gather from A, rather
// than CUDA's scatter+atomicAdd: WGSL has no float atomicAdd. A gather needs
// no pre-zeroed output (every C cell is written exactly once, by exactly one
// invocation) and no atomics — the tradeoff is fold's small bounded loop over
// the window indices that could cover a given output cell, in place of one
// atomicAdd per source element.
//
// Neither family's shape metadata fits the fixed Params uniform (a
// variable-length array has no home there), so it rides B as
// bit-reinterpreted u32 — B's declared type is array<f32>, but WriteBuffer on
// the host side is a raw byte copy regardless, and bitcast<u32> reads it back
// correctly. p._pad0/_pad1/_pad2 (otherwise-unused Params padding) carry
// rank/axis/before-or-step; p.M is the output element count (webgpu.h's
// pad()/fold() dispatch over out_n, not a's own size).
//
// Rank cap — matches webgpu.h's own kPadFoldMaxRank (a shader can't see a
// host-side C++ constant, so this is its own copy) and bounds every
// fixed-size local array below.
const kPadFoldMaxRank : u32 = 8u;

// Shared by both: row-major decode of a dispatch-global thread id `i` against
// a shape held in B at word offset `base`, into a fixed-size local array.
// `rank8` bounds the loop so callers can pass either a full-rank or a
// (rank-1)-length shape.
fn decode_idx(i : u32, base : u32, rank8 : u32,
             out_idx : ptr<function, array<u32, kPadFoldMaxRank>>) {
  var rem = i;
  for (var d : i32 = i32(rank8) - 1; d >= 0; d = d - 1) {
    let dim = bitcast<u32>(B[base + u32(d)]);
    (*out_idx)[u32(d)] = rem % dim;
    rem = rem / dim;
  }
}

// Row-major strides of a contiguous tensor whose shape is held in B at word
// offset `base`, length `rank` — used to address `A`, which pad_/fold_'s GPU
// dispatch (array.h's gpu_pad_/gpu_fold_) requires to be contiguous.
fn a_strides_from_shape(base : u32, rank : u32,
                        a_shape : ptr<function, array<u32, kPadFoldMaxRank>>,
                        out_strides : ptr<function, array<u32, kPadFoldMaxRank>>) {
  var acc : u32 = 1u;
  for (var d : i32 = i32(rank) - 1; d >= 0; d = d - 1) {
    let dim = bitcast<u32>(B[base + u32(d)]);
    (*a_shape)[u32(d)] = dim;
    (*out_strides)[u32(d)] = acc;
    acc = acc * dim;
  }
}

// B layout at word offset p.b_off: [out_shape(rank), a_shape(rank)] — webgpu.h
// reserves a fresh ring slot per call (see meta_reserve_slot_), so two pad/
// fold calls batched into the same unflushed pass never share one offset.
@compute @workgroup_size(256, 1, 1)
fn pad(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let rank = p._pad0;
  let axis = p._pad1;
  let before = i32(p._pad2);

  var out_idx : array<u32, kPadFoldMaxRank>;
  decode_idx(i, p.b_off, rank, &out_idx);
  var a_shape : array<u32, kPadFoldMaxRank>;
  var a_strides : array<u32, kPadFoldMaxRank>;
  a_strides_from_shape(p.b_off + rank, rank, &a_shape, &a_strides);

  var src : u32 = 0u;
  var in_bounds = true;
  for (var d : u32 = 0u; d < rank; d = d + 1u) {
    var c : i32 = i32(out_idx[d]);
    if (d == axis) {
      c = c - before;
      if (c < 0 || c >= i32(a_shape[d])) { in_bounds = false; }
    }
    // Clamped even out of range: A[src] is still read below (WGSL's select
    // evaluates both operands), so src must stay in bounds regardless of
    // in_bounds — only the select, not the address, decides the result.
    let cc = clamp(c, 0, i32(a_shape[d]) - 1);
    src = src + u32(cc) * a_strides[d];
  }
  C[p.c_off + i] = select(0.0, A[p.a_off + src], in_bounds);
}

// B layout at word offset p.b_off: [out_shape(rank-1), a_shape(rank)] — same
// per-call ring slot as pad above. a's last dim is the sliding window (size
// a_shape[rank-1]); a's `axis` dim (size a_shape[axis]) is the window count.
@compute @workgroup_size(256, 1, 1)
fn fold(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let rank = p._pad0;
  let axis = p._pad1;
  let step = i32(p._pad2);
  let out_rank = rank - 1u;

  var out_idx : array<u32, kPadFoldMaxRank>;
  decode_idx(i, p.b_off, out_rank, &out_idx);
  var a_shape : array<u32, kPadFoldMaxRank>;
  var a_strides : array<u32, kPadFoldMaxRank>;
  a_strides_from_shape(p.b_off + out_rank, rank, &a_shape, &a_strides);

  let win : i32 = i32(a_shape[rank - 1u]);
  let nwin : i32 = i32(a_shape[axis]);
  let j : i32 = i32(out_idx[axis]);

  var w_min : i32 = 0;
  if (j - win + 1 > 0) { w_min = (j - win + 1 + step - 1) / step; }
  var w_max : i32 = j / step;
  if (w_max > nwin - 1) { w_max = nwin - 1; }

  var sum : f32 = 0.0;
  for (var w : i32 = w_min; w <= w_max; w = w + 1) {
    let k = j - w * step;
    if (k < 0 || k >= win) { continue; }
    var src : u32 = 0u;
    for (var d : u32 = 0u; d < out_rank; d = d + 1u) {
      let coord = select(out_idx[d], u32(w), d == axis);
      src = src + coord * a_strides[d];
    }
    src = src + u32(k) * a_strides[rank - 1u];
    sum = sum + A[p.a_off + src];
  }
  C[p.c_off + i] = sum;
}

// sum_to: sum `a` down to a smaller broadcast-target shape (the dual of
// broadcast_to that every arithmetic op's backward uses to un-broadcast a
// gradient). Gather, not scatter: one invocation per OUTPUT element sums
// every `a` element that broadcasts onto it, so -- like pad/fold above --
// no write conflict and no atomics. B layout at word offset p.b_off:
// [a_shape(rank), a_strides(rank), acc(rank)] -- acc is a's shape
// broadcast-aligned against the output's own strides (array.h's
// broadcast_strides(target, out.strides(), a.shape())), 0 on a reduced
// axis. p._pad0 = rank, p._pad1 = reduced_n (product of a_shape over
// exactly the zero-acc axes; 1 if there are none).
@compute @workgroup_size(256, 1, 1)
fn sum_to(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = gid.x;
  if (t >= p.M) { return; }
  let rank = p._pad0;
  let reduced_n = p._pad1;

  var base : u32 = 0u;
  var red_axis : array<u32, kPadFoldMaxRank>;
  var red_count : u32 = 0u;
  for (var d : u32 = 0u; d < rank; d = d + 1u) {
    let a_shape_d = bitcast<u32>(B[p.b_off + d]);
    let acc_d = bitcast<u32>(B[p.b_off + 2u * rank + d]);
    if (acc_d != 0u) {
      let a_strides_d = bitcast<u32>(B[p.b_off + rank + d]);
      let idx = (t / acc_d) % a_shape_d;
      base = base + idx * a_strides_d;
    } else {
      red_axis[red_count] = d;
      red_count = red_count + 1u;
    }
  }
  var sum : f32 = 0.0;
  for (var r : u32 = 0u; r < reduced_n; r = r + 1u) {
    var rem = r;
    var off = base;
    for (var k : i32 = i32(red_count) - 1; k >= 0; k = k - 1) {
      let d = red_axis[u32(k)];
      let dim = bitcast<u32>(B[p.b_off + d]);
      let coord = rem % dim;
      rem = rem / dim;
      let stride = bitcast<u32>(B[p.b_off + rank + d]);
      off = off + coord * stride;
    }
    sum = sum + A[p.a_off + off];
  }
  C[p.c_off + t] = sum;
}

// ---- Embedding-table lookup (index_select/index_add) and pooling-style
// one-hot scatter (scatter_to_axis) -- the WGSL counterparts of
// tl_index_select/tl_index_add/tl_scatter_axis in kernels/tensorlib_cuda.cu.
//
// index_select and scatter_axis are gathers already (every output element
// is written by exactly one invocation), so they port the CUDA kernel body
// directly. index_add is CUDA's one true scatter+atomicAdd here -- repeated
// indices really do collide -- and WGSL has no float atomicAdd, the same gap
// pad/fold above work around. So index_add is a gather too: one invocation
// per OUTPUT element, summing over every source row whose index matches it,
// in place of scattering into a pre-zeroed buffer.

// A = a (table), B = idx, C = out. p._pad0 = row_size.
@compute @workgroup_size(256, 1, 1)
fn index_select(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let row_size = p._pad0;
  let row = i / row_size;
  let col = i % row_size;
  let src_row = u32(B[p.b_off + row] + 0.5);
  C[p.c_off + i] = A[p.a_off + src_row * row_size + col];
}

// A = idx, B = values, C = out. p._pad0 = row_size, p._pad1 = k (source rows).
@compute @workgroup_size(256, 1, 1)
fn index_add(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let row_size = p._pad0;
  let k_count = p._pad1;
  let row = i / row_size;
  let col = i % row_size;
  var sum : f32 = 0.0;
  for (var k : u32 = 0u; k < k_count; k = k + 1u) {
    let idx_row = u32(A[p.a_off + k] + 0.5);
    if (idx_row == row) {
      sum = sum + B[p.b_off + k * row_size + col];
    }
  }
  C[p.c_off + i] = sum;
}

// A = idx, B = values, C = out. p._pad0 = size (the new trailing axis).
// out[pos, k] = values[pos] where idx[pos] == k, else 0 -- gather, so no
// zeroing needed, unlike CUDA's pre-zeroed scatter.
@compute @workgroup_size(256, 1, 1)
fn scatter_axis(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let size = p._pad0;
  let pos = i / size;
  let k = i % size;
  let dst_k = u32(A[p.a_off + pos] + 0.5);
  C[p.c_off + i] = select(0.0, B[p.b_off + pos], dst_k == k);
}

// ---- N-D broadcast binary (any rank) and N-D broadcast ternary select
// (Tensor.where's masking) -- the WGSL counterparts of tl_b*_nd/tl_where_nd
// in kernels/tensorlib_cuda.cu. Same flat-index decode as pad/fold above,
// against strides the host supplies (broadcast_strides(), 0 on a broadcast
// axis) rather than derived from a shape -- inlined per kernel rather than
// shared via decode_idx, since that helper always reads from B and here the
// meta buffer is D or E (WGSL has no templates to parameterize over which
// storage binding to read).

// A = a, B = b, D = meta [out_shape(rank), a_strides(rank), b_strides(rank)],
// E unused. p._pad0 = rank, p._pad3 = meta's word offset into D.
@compute @workgroup_size(256, 1, 1)
fn ew_bcast_nd(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let rank = p._pad0;
  let base = p._pad3;
  var rem = i;
  var a_off : u32 = 0u;
  var b_off : u32 = 0u;
  for (var d : i32 = i32(rank) - 1; d >= 0; d = d - 1) {
    let dim = bitcast<u32>(D[base + u32(d)]);
    let coord = rem % dim;
    rem = rem / dim;
    a_off = a_off + coord * bitcast<u32>(D[base + rank + u32(d)]);
    b_off = b_off + coord * bitcast<u32>(D[base + 2u * rank + u32(d)]);
  }
  let av = A[p.a_off + a_off];
  let bv = B[p.b_off + b_off];
  C[p.c_off + i] = fma(binary_op(p.op, av, bv), p.scale, p.offset);
}

// A = cond, B = a, D = b, E = meta [out_shape(rank), cond_strides(rank),
// a_strides(rank), b_strides(rank)]. p._pad0 = rank, p._pad3 = b's element
// offset into D, p._pad4 = meta's word offset into E.
@compute @workgroup_size(256, 1, 1)
fn where_nd(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.M) { return; }
  let rank = p._pad0;
  let base = p._pad4;
  var rem = i;
  var c_off : u32 = 0u;
  var a_off : u32 = 0u;
  var b_off : u32 = 0u;
  for (var d : i32 = i32(rank) - 1; d >= 0; d = d - 1) {
    let dim = bitcast<u32>(E[base + u32(d)]);
    let coord = rem % dim;
    rem = rem / dim;
    c_off = c_off + coord * bitcast<u32>(E[base + rank + u32(d)]);
    a_off = a_off + coord * bitcast<u32>(E[base + 2u * rank + u32(d)]);
    b_off = b_off + coord * bitcast<u32>(E[base + 3u * rank + u32(d)]);
  }
  let cv = A[p.a_off + c_off];
  let av = B[p.b_off + a_off];
  let bv = D[p._pad3 + b_off];
  C[p.c_off + i] = select(bv, av, cv != 0.0);
}
