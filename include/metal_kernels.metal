// MSL kernel source, #embed'd into metal.h and JIT-compiled on first GPU
// use. Editing this file changes nothing until the host binary is rebuilt.
//
// Every kernel applies the graph's affine epilogue (out = op * scale +
// offset) so fused scalar chains cost zero extra dispatches. The host
// dispatches ceil(n/256) threadgroups; the `i >= p.n` bound uses the same
// ceiling arithmetic (shared-template rule from silarray's edge-tile bugs:
// op bodies live in one macro, never copy-pasted per variant).

#include <metal_stdlib>
using namespace metal;

struct ew_params {
  float scale;
  float offset;
  uint n;
};

#define EW_BINARY(name, expr)                                  \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device const float* b [[buffer(1)]],        \
                   device float* out [[buffer(2)]],            \
                   constant ew_params& p [[buffer(3)]],        \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    out[i] = fma(expr, p.scale, p.offset);                     \
  }

EW_BINARY(add_, a[i] + b[i])
EW_BINARY(sub_, a[i] - b[i])
EW_BINARY(mul_, a[i] * b[i])
EW_BINARY(div_, a[i] / b[i])

#define EW_UNARY(name, expr)                                   \
  kernel void name(device const float* a [[buffer(0)]],        \
                   device float* out [[buffer(1)]],            \
                   constant ew_params& p [[buffer(2)]],        \
                   uint i [[thread_position_in_grid]]) {       \
    if (i >= p.n) return;                                      \
    out[i] = fma(expr, p.scale, p.offset);                     \
  }

EW_UNARY(exp_, exp(a[i]))
EW_UNARY(log_, log(a[i]))
EW_UNARY(sqrt_, sqrt(a[i]))
EW_UNARY(sigmoid_, 1.0f / (1.0f + exp(-a[i])))
EW_UNARY(relu_, max(a[i], 0.0f))
EW_UNARY(affine_, a[i])
