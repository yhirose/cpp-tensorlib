#!/usr/bin/env python3
"""Compare the files tensorlib_dump_logits wrote (bench/models/dump_logits.cpp)
before and after a change.

    tools/compare_logits.py BEFORE.bin [BEFORE2.bin ...] -- AFTER.bin [AFTER2.bin ...]

Per section: identical in every pair, or the largest |difference| relative to
1 + the largest |logit| between a before run and an after run, and whether the
top token agrees. Some paths are not bit-reproducible (the GEMVs that split K
add their partial sums with atomics, in no fixed order), so a section that
differs is held to the spread the same code shows between its own runs: the
largest difference within the before runs and within the after runs. Give two
or more runs of a side for that spread to mean something; with one run of each,
any difference fails. Exit 0 when every section is identical, or differs across
the change no more than 1.5x the code's own spread.
"""
import itertools
import struct
import sys

import numpy as np


def read(path):
    out = {}
    with open(path, "rb") as f:
        b = f.read()
    o = 0
    while o < len(b):
        (n,) = struct.unpack_from("<I", b, o)
        o += 4
        name = b[o:o + n].decode()
        o += n
        (count,) = struct.unpack_from("<q", b, o)
        o += 8
        out[name] = np.frombuffer(b, np.float32, count, o)
        o += 4 * count
    return out


def rel(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b)) / (1 + np.max(np.abs(b))))


def same(a, b):
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def main():
    args = sys.argv[1:]
    if "--" not in args:
        args = [args[0], "--"] + args[1:]  # BEFORE AFTER
    cut = args.index("--")
    before = [read(p) for p in args[:cut]]
    after = [read(p) for p in args[cut + 1:]]
    ok = True
    names = list(before[0])
    if any(r.keys() != before[0].keys() for r in before + after):
        print("the runs do not have the same sections")
        return 1
    for name in names:
        cross = [(a[name], b[name]) for b in before for a in after]
        if all(same(x, y) for x, y in cross):
            print(f"  {name:36s} identical")
            continue
        d = max(rel(x, y) for x, y in cross)
        top = all(int(np.argmax(x)) == int(np.argmax(y)) for x, y in cross)
        own = [rel(x[name], y[name]) for side in (before, after)
               for x, y in itertools.combinations(side, 2)]
        spread = max(own, default=0.0)
        within = spread > 0 and d <= 1.5 * spread
        ok &= within
        print(f"  {name:36s} differs {d:.2e}, top token {'same' if top else 'DIFFERENT'}"
              f" | own runs {spread:.2e} {'within' if within else 'BEYOND'}")
    print("OK" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
