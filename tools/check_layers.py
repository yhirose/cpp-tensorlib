#!/usr/bin/env python3
"""The layers (include/gpu_abi.h) as a check: core <- nn <- llm, never upward.

  tools/check_layers.py            the include graph, and the GPU layer compiled
                                   with each upper layer left out
  tools/check_layers.py --nvcc     also the CUDA kernels, core alone and core + nn

1. Includes. Every file of the library, its kernels and its tests has a layer,
   and a file includes (or #embeds) only files of its own layer or below. The
   one way up is a binder: a file that joins the layers for its users, at its
   end (gpu.h, array.h, the backend headers, tensorlib_cuda.cu). Those edges
   are listed below; any other upward edge fails.
2. Compiles. gpu.h with the llm layer's ops left out, and with the nn
   layer's too, must still compile (the host and the null backends, which
   keep every layer in one file): a lower layer that called up would not.
3. --nvcc: the CUDA kernels of the core layer alone, then with the nn layer,
   must compile (each .cuh builds only on the ones before it).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = {"core": 0, "nn": 1, "llm": 2}

# Every layered file. A file of the library that is missing here is an error,
# so a new header has to say where it stands.
FILES = {
    "core": """include/gpu_abi.h include/gpu_ops.h include/gpu.h include/cuda.h
               include/metal.h include/webgpu.h include/gpu_host.h include/gpu_null.h
               include/array.h include/tensorlib.h include/storage.h include/types.h
               include/shape.h include/profile.h include/cpu.h include/cpu_threadpool.h
               include/objc.h include/metal_kernels.metal kernels/tensorlib_cuda.cu
               kernels/cuda_core.cuh test/test_util.h test/test_basic.cpp
               test/test_array.cpp""",
    "nn": """include/gpu_abi_nn.h include/gpu_ops_nn.h include/cuda_nn.h include/metal_nn.h
             include/array_nn.h include/ggml_quant.h include/metal_kernels_nn.metal
             kernels/cuda_nn.cuh test/test_nn.cpp""",
    "llm": """include/gpu_abi_llm.h include/gpu_ops_llm.h include/cuda_llm.h
              include/kv_cache.h include/gguf.h include/tokenizer.h
              include/tokenizer_data.h include/metal_kernels_llm.metal
              kernels/cuda_llm.cuh test/test_llm.cpp""",
}
LAYER_OF = {f: lay for lay, fs in FILES.items() for f in fs.split()}

# The binders' upward edges: each joins the layers above its own at its end.
UP = {
    ("include/gpu.h", "include/gpu_ops_nn.h"), ("include/gpu.h", "include/gpu_ops_llm.h"),
    ("include/array.h", "include/array_nn.h"),
    ("include/cuda.h", "include/cuda_nn.h"), ("include/cuda.h", "include/cuda_llm.h"),
    ("include/metal.h", "include/metal_nn.h"),
    ("include/metal.h", "include/metal_kernels_nn.metal"),
    ("include/metal.h", "include/metal_kernels_llm.metal"),
    ("kernels/tensorlib_cuda.cu", "kernels/cuda_nn.cuh"),
    ("kernels/tensorlib_cuda.cu", "kernels/cuda_llm.cuh"),
    # The reference backend states every kernel id in one plain file.
    ("include/gpu_host.h", "include/gpu_abi_nn.h"),
    ("include/gpu_host.h", "include/gpu_abi_llm.h"),
}


def includes(path):
    out = []
    for line in open(os.path.join(ROOT, path), errors="replace"):
        m = re.match(r'^\s*#\s*(?:include|embed)\s+[<"]([^>"]+)[>"]', line)
        if m:
            out.append(m.group(1))
    return out


def resolve(src, name):
    for d in (os.path.dirname(src), "include", "kernels", "test"):
        cand = os.path.normpath(os.path.join(d, name))
        if os.path.exists(os.path.join(ROOT, cand)):
            return cand
    return None


def check_includes():
    errors = []
    listed = set(LAYER_OF)
    for d, exts in (("include", (".h", ".metal")), ("kernels", (".cu", ".cuh")),
                    ("test", (".h", ".cpp"))):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            p = d + "/" + f
            if f.endswith(exts) and p not in listed and f not in ("doctest.h",):
                errors.append(f"{p}: has no layer (add it to FILES in {__file__})")
    for src, lay in sorted(LAYER_OF.items()):
        if not os.path.exists(os.path.join(ROOT, src)):
            errors.append(f"{src}: listed but missing")
            continue
        for name in includes(src):
            dst = resolve(src, name)
            if dst not in LAYER_OF:
                continue
            if LAYERS[LAYER_OF[dst]] > LAYERS[lay] and (src, dst) not in UP:
                errors.append(f"{src} ({lay}) includes {dst} ({LAYER_OF[dst]})")
    return errors


def compile_cxx(tmp, keep, backend):
    # gpu.h with the op layers above `keep` dropped.
    inc = os.path.join(tmp, "include")
    shutil.rmtree(inc, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "include"), inc)
    gpu = open(os.path.join(inc, "gpu.h")).read()
    drop = {"core": ["gpu_ops_nn.h", "gpu_ops_llm.h"], "nn": ["gpu_ops_llm.h"]}[keep]
    for h in drop:
        gpu, n = re.subn(r'^#include "' + re.escape(h) + r'"\n', "", gpu, flags=re.M)
        assert n == 1, h
    open(os.path.join(inc, "gpu.h"), "w").write(gpu)
    src = os.path.join(tmp, "t.cpp")
    open(src, "w").write('#include "gpu.h"\nint main() { return tl::gpu::census(tl::gpu::kop::add) > 0; }\n')
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    flags = ["-std=c++20", "-fsyntax-only", "-I" + inc, src] + (["-D" + backend] if backend else [])
    r = subprocess.run([cxx] + flags, capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr.strip().splitlines()[:5]


def compile_nvcc(tmp, keep):
    cu = open(os.path.join(ROOT, "kernels/tensorlib_cuda.cu")).read()
    upto = {"core": ["cuda_core.cuh"], "nn": ["cuda_core.cuh", "cuda_nn.cuh"]}[keep]
    for h in ("cuda_core.cuh", "cuda_nn.cuh", "cuda_llm.cuh"):
        if h not in upto:
            cu, n = re.subn(r'^#include "' + re.escape(h) + r'"\n', "", cu, flags=re.M)
            assert n == 1, h
    src = os.path.join(ROOT, "kernels", "_layer_check.cu")  # beside the .cuh it includes
    try:
        open(src, "w").write(cu)
        r = subprocess.run(["nvcc", "-arch=sm_80", "-ptx", src, "-o", os.path.join(tmp, "k.ptx")],
                           capture_output=True, text=True)
    finally:
        os.remove(src)
    return None if r.returncode == 0 else r.stderr.strip().splitlines()[:5]


def main():
    errors = check_includes()
    with tempfile.TemporaryDirectory() as tmp:
        for keep in ("core", "nn"):
            for backend in ("TENSORLIB_HOST_GPU", None):
                err = compile_cxx(tmp, keep, backend)
                if err:
                    errors.append(f"gpu.h up to {keep} ({backend or 'null backend'}) does not compile:")
                    errors += ["  " + e for e in err]
        if "--nvcc" in sys.argv:
            for keep in ("core", "nn"):
                err = compile_nvcc(tmp, keep)
                if err:
                    errors.append(f"the CUDA kernels up to {keep} do not compile:")
                    errors += ["  " + e for e in err]
    if errors:
        print("check_layers: " + "\n".join(errors), file=sys.stderr)
        return 1
    print("check_layers: every include goes down a layer or through a binder; "
          "the GPU layer compiles up to core and up to nn" +
          ("; so do the CUDA kernels" if "--nvcc" in sys.argv else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
