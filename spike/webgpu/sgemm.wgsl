// Tiled SGEMM: C(M,N) = A(M,K) @ B(K,N), row-major, no transpose.
//
// 64x64 threadgroup tile, 16x16 = 256 invocations, each holding a 4x4
// register accumulator. Mirrors the shape of the Metal sgemm_64_ kernel so the
// measured numbers are comparable; MMA intrinsics have no WGSL equivalent, so
// the inner product is plain FMA over registers.

struct Params {
  M : u32,
  N : u32,
  K : u32,
  scale : f32,
  offset : f32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             p : Params;

const BM : u32 = 64u;
const BN : u32 = 64u;
const BK : u32 = 16u;
const TM : u32 = 4u;
const TN : u32 = 4u;
const THREADS : u32 = 256u;

var<workgroup> As : array<f32, 1024>;  // BM * BK
var<workgroup> Bs : array<f32, 1024>;  // BK * BN

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

      let ar = i / BK;
      let ac = i % BK;
      let am = m_base + ar;
      let ak = k0 + ac;
      As[i] = select(0.0, A[am * p.K + ak], am < p.M && ak < p.K);

      let br = i / BN;
      let bc = i % BN;
      let bk = k0 + br;
      let bn = n_base + bc;
      Bs[i] = select(0.0, B[bk * p.N + bn], bk < p.K && bn < p.N);
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
      C[m * p.N + n] = acc[i * TN + j] * p.scale + p.offset;
    }
  }
}
