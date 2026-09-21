# GPU backends

How the GPU layer is put together, how to add an op to it, and how to add a
backend. The code is `include/gpu.h`, `gpu_abi.h`, `gpu_ops.h`, `gpu_null.h`
and one header per backend (`metal.h`, `cuda.h`, `webgpu.h`).

## Layers

```
array.h / storage.h / kv_cache.h / models     name no backend; call tl::gpu
──────────────────────────────────────────────
gpu_ops.h    every op, written once            shared
gpu_abi.h    span, access, grid, kernel ABI,
             params structs, launch policy
──────────────────────────────────────────────
device core  lifecycle, memory, dispatch,      one per backend:
             own, traits, caps                 metal / cuda / webgpu / null
kernels      .metal / .cu / .wgsl
```

`tl::gpu` is a namespace holding the shared layer plus a using-directive for
the selected backend. A name the shared layer declares (an op) is found
first; anything else (`alloc`, `flush`, `caps`) falls through to the backend.
`gpu.h` selects exactly one backend header by its gate and includes only that
one; a build that fits none gets `gpu_null.h`.

`array.h` and `storage.h` never name a backend, and neither does an embedder:
everything goes through `tl::gpu` or the array API.

## Views: `gpu::span`

```cpp
struct span { void* buf; int64_t off; };   // off in bytes
```

Every op takes its buffers as spans. `buf` is whatever the backend's `alloc`
returned and means nothing outside that backend: an `MTLBuffer` handle on
Metal, a device address on CUDA, a key into a mirror table on WebGPU. The
offset travels beside the handle rather than inside it, so a view is
expressible on every backend, and arithmetic on a handle is not expressible at
all. `s.at(bytes)` slices a view; `array::device_span()` and
`storage::device_span()` produce one.

Each view a kernel takes is tagged with how the kernel touches it:

| access  | meaning | what a mirrored backend does |
|---------|---------|------------------------------|
| `in`    | read | upload first if the host holds the live copy |
| `out`   | written | the device copy becomes the live one; upload first only if the host had filled the buffer |
| `inout` | read, then written | upload, then the device copy is the live one |

A unified-memory backend ignores the tag. No op states residency by hand.

When to copy is decided once, by `gpu::residency` (`gpu_abi.h`): a mirrored
backend keeps one per allocation beside its two copies, asks it
`before_kernel(access)` and `before_host(for_write)`, and does the copying. An
allocation starts `none` (nobody has filled it) unless `alloc` was told the host
fills it. That is what makes `out` safe on a view: a view may cover part of its
buffer, so an output into a buffer whose live bytes are the host's has to bring
them up or lose the rest, while an output into a fresh buffer, which is nearly
every output, has nothing to bring.

## The kernel ABI

An op hands the backend a kernel id, an ordered list of views, a params
struct, and a grid:

```cpp
inline bool rmsnorm_res(span x, span delta, span w, span xout, span hout,
                        int64_t n, float eps, int64_t rows = 1) {
  if (n <= 0 || rows <= 0) return false;
  return launch(kop::add_rmsnorm_,
                {in(x), in(delta), in(w), out(xout), out(hout)},
                rmsnorm_params{static_cast<uint32_t>(n), eps},
                policy::one_group_per_row(rows));
}
```

For a kernel id, two things are the same on every backend:

- **the views**, in the order the kernel declares its buffers;
- **the params struct**: a run of 4-byte fields (`uint32_t`, `int32_t`,
  `float`) in the order the kernel takes its scalars. The structs live in
  `gpu_abi.h`; the MSL and WGSL sources declare the same layouts on their side.

That is what lets a backend realize a launch with no per-kernel host code:

- **Metal** binds view *i* at buffer index *i* (`setBuffer:offset:atIndex:`)
  and the params at index *n* (`setBytes`).
- **CUDA** builds `cuLaunchKernel`'s `argv` as the views' device addresses
  followed by the params' fields, four bytes apiece. Its kernels take their
  pointers first and 4-byte scalars after (116 of the 117 do;
  `tools/cuda_trace/gen_kernel_sigs.py` reads this off the `.cu`).
- **WebGPU**'s kernels predate the ABI: every entry point reads one 96-byte
  uniform layout, a family picks its operation by number, and the bind group is
  fixed (A and B read, C written, D and E read). Its core carries a `marshal_`
  from the canonical params into that layout, per kernel id. This stays inside
  `webgpu.h`; a backend whose kernels follow the ABI needs none.

Where two backends' kernels disagreed, the CUDA kernel's order is canonical,
because the `.cu` is the one source no development machine here can run, and
the MSL side can be reordered and tested locally.

## Two kinds of op

**A single-kernel op** is one kernel with the same buffers everywhere. It is a
function in `gpu_ops.h` over `launch`, as above, and exists on every backend
that has a kernel for its id. A backend without one returns false from
`dispatch`, and the evaluator falls back to the CPU.

**A backend-own op** is one whose algorithm differs by backend: a scatter into a
zeroed buffer on CUDA against a gather on Metal and WebGPU; N-D shape metadata
uploaded to a device buffer against a params block; one kernel against a split
pass and a combine. Forcing these into one kernel ABI would mean rewriting
kernels whose differences are deliberate. Instead a backend declares the op as
a static member of its `own` struct, with the signature the op has in
`gpu_ops.h`, and writes it over its own `dispatch`:

```cpp
// gpu_ops.h — the signature, once
TL_GPU_DETECT_OWN(rope)
template <class Own = own>
inline bool rope(span x, span o, int64_t rows, int64_t T, int64_t D,
                 int64_t pos, float base, span bias = {}) {
  if constexpr (detail::owns_rope<Own>::value) {
    return Own::rope(x, o, rows, T, D, pos, base, bias);
  } else {
    return false;
  }
}

// metal.h — declared in `struct own`, defined among its helpers
inline bool own::rope(gpu::span x, gpu::span out, ...) { ... }
```

`gpu_ops.h` detects the member and forwards to it, or answers false. A backend
declares what it has and nothing else: there are no stubs. A member whose
signature drifts from the shared one is a compile error, not a silent fallback.

Prefer the single-kernel form. Reach for `own` when the kernels genuinely
differ, not to avoid reordering a params struct.

## Launch policy

The shapes ops launch in live in `gpu::policy` (`gpu_abi.h`), shared host code:
`flat`, `flat_rows`, `one_group_per_row`, `row_reduce`, `per_head`, `cells_2d`.
A `grid` is groups x threads-per-group plus the bytes of per-group scratch a
reduction needs where the backend sizes it at launch (CUDA's shared memory;
Metal and WGSL size theirs in the kernel).

What differs between backends' kernels comes in through the backend's `traits`.
Today that is one fact: whether a rank-2 elementwise kernel reads its cell from
a 2-D thread position or from a flat index (`traits::cells_2d`).

## Adding an op

1. If every backend can run it as one kernel with the same buffers: add its
   params struct to `gpu_abi.h`, the function to `gpu_ops.h`, the kernel to each
   backend's source with that layout, and the id to each backend's kernel table
   (`kernel_name_` in `metal.h` and `cuda.h`, `marshal_` in `webgpu.h`).
2. Otherwise add the detecting wrapper to `gpu_ops.h` and the member to the
   `own` struct of each backend that implements it.
3. A backend that gets neither simply declines the op. Nothing else changes.

## Adding a backend

1. Copy `gpu_null.h`. It is everything `gpu.h` asks of a backend, with nothing
   filled in: lifecycle (`available`, `pending`, `flush`, `cpu_barrier`), memory
   (`alloc`, `release`, `sync_to_host`, `upload`), `dispatch`, `own`, `traits`,
   `caps` and the graph-capture plumbing.
2. Gate the header whole on the platform it builds for, and add one branch to
   the selection in `gpu.h`.
3. Write `dispatch` against the kernel ABI, and kernels that follow it. Start
   with the elementwise, broadcast, reduction, GEMM, copy and index families:
   they close the array surface. Every op the backend has no kernel for falls
   back to the CPU, so the suite passes from the first kernel on.
4. Declare in `own` whatever the backend runs its own way.
5. Run the suite in `--gpu` and `--auto` mode and check `gpu::census`: the
   suite's oracle comparisons pass whether or not the GPU engaged.

No existing backend's file is touched.

## What is not shared yet

- The mirror table (handle to host copy, device copy, size, `residency`) and
  the buffer pool exist twice, in `cuda.h` and `webgpu.h`. The state machine
  itself is shared.
- `tl::profile` hooks sit in each backend's launch path.
- `kop` still lists kernel ids only one backend has (Metal's GEMM tiles and
  attention variants).
- Launch policy inside the own ops (CUDA's split-K and tile choices, Metal's
  GEMM ladder) is still the backend's.
- There is no generic composition under the fused ops, so a backend without the
  model-path kernels reports `caps::model_path = false` rather than running them
  slowly. WebGPU is in that position.

## Verifying a change

All four run on a development Mac.

| what | command |
|------|---------|
| Metal, and the CPU | `cmake --build build && ctest --test-dir build` |
| a real model on Metal | `build/tensorlib_check_qwen` (greedy tokens against a numpy oracle) |
| WebGPU | `test/wasm/build.sh && deno run --allow-all test/wasm/deno_run.js` |
| CUDA's host side | `tools/cuda_trace/compare.sh <base-ref>` |
| no backend | a Linux build without `TENSORLIB_CUDA`: `gpu.h` selects `gpu_null.h` |

No machine here runs CUDA kernels, and CI compiles them without running them.
`tools/cuda_trace` puts a stand-in `libcuda.so.1` in front of the backend (it
`dlopen`s the driver, so nothing in `cuda.h` knows), builds the suite and the
CUDA checkers on Linux in a container, and records every launch: kernel, grid,
block, shared-memory bytes and each argument, with pointers printed as
(allocation, byte offset) so two runs diff. What a change to the backend's host
side must preserve is that the same kernels get the same arguments, and that is
what the trace holds. `bench/cuda/check/trace_sweep.cpp` reaches the kernels the
suite does not, so all 117 appear. `TL_CUDA_TRACE_CHECK=1 tools/cuda_trace/run.sh
<dir>` only compiles: every test, checker, bench and model driver against the
CUDA branch of the headers. Whether the kernels compute the right thing is
still a `ctest` on NVIDIA hardware.

`gpu::census(kernel)` counts a shared op's launches and `gpu::ops_run()` every
op that ran on the device, shared or backend-own, since `gpu::census_reset()`.
An op that declines falls back to the CPU and the result is still right, so a
test that wants to know the device was reached has to ask. The suite has two
tests built on this: every op on views at non-zero offsets, inside buffers with
sentinels on both sides, against a plain host loop; and one graph per op family
through the evaluator in GPU mode, where the census has to move.
