#!/usr/bin/env python3
"""No CUDA kernel has a stack frame, checked from the PTX, without a GPU.

  tools/check_cuda_frames.py build/tensorlib_cuda.ptx [ARCH]

check_cuda (bench/cuda/check) holds the same on a GPU, from what the driver
JIT-compiled: an array a kernel keeps in local memory is read through it, on
every call where it is a working set (the Q4_K scales once cost the decode
GEMV a fifth of its speed). ptxas, which comes with nvcc, reports each
function's frame as it compiles the PTX for an arch (ARCH, the RTX 3090's
sm_86 by default), so a frame fails here on every pull request, not only
where check_cuda runs.
"""

import os
import re
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ptx, arch = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sm_86"
    r = subprocess.run(["ptxas", "-arch=" + arch, "-v", ptx, "-o", os.devnull],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out)
        return 1
    entries = len(re.findall(r"Compiling entry function '", out))
    # "Function properties for NAME", then its frame and spills on the next line.
    props = re.findall(
        r"Function properties for (\S+)\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores", out)
    if entries == 0 or len(props) < entries:  # ptxas's wording changed: fail, not pass
        print(f"ptxas -v listed {entries} kernels, the frames of {len(props)}")
        return 1
    framed = [f"{name} ({frame} B frame, {spill} B spilled)"
              for name, frame, spill in props if int(frame) or int(spill)]
    print(f"{entries} kernels for {arch}: " + ("framed: " + ", ".join(framed) if framed else "none framed"))
    return 1 if framed else 0


if __name__ == "__main__":
    sys.exit(main())
