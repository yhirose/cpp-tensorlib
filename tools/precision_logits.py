#!/usr/bin/env python3
"""How far each forward path's logits are from the f32 reference, in the files
tensorlib_dump_logits wrote (bench/models/dump_logits.cpp).

    tools/precision_logits.py DUMP.bin [OTHER.bin ...]   (LABEL=FILE names a file)

The reference is the f32 build's imperative step (f32 weights, activations,
sums and KV cache): a section build/path/what (what: prompt, or stepN) is held
to f32/imperative/what. Every other section of every file (another tool's
logits for the same tokens in the same format, tools/llama_logits.cpp's, too)
is reported by the KL divergence of its distribution from the reference's
(nats, as llama.cpp's perplexity --kl-divergence reports it), whether its top
token is the reference's, and its largest |difference| relative to 1 + the
largest |logit|. Per path, the mean over the prompt and the steps; a table a
file, the reference the first file's that has one (the same model's dumps under
other settings, say, which leave the f32 build alone).
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I
from compare_logits import read, rel  # noqa: E402


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def report(sections, ref, ref_lp):
    rows = defaultdict(list)
    for name, v in sections.items():
        path, what = name.rsplit("/", 1)
        if path == "f32/imperative" or path == "tokens" or what not in ref:
            continue
        lp = ref_lp[what]
        kl = float(np.sum(np.exp(lp) * (lp - log_softmax(v))))
        top = int(np.argmax(v)) == int(np.argmax(ref[what]))
        rows[path].append((kl, top, rel(v, ref[what])))
    print(f"  {'path':28s} {'KL mean':>9s} {'KL max':>9s} {'top-1':>7s} {'max |d|':>9s}")
    for path, r in rows.items():
        kls = [x[0] for x in r]
        print(f"  {path:28s} {np.mean(kls):9.2e} {max(kls):9.2e} {sum(x[1] for x in r):3d}/{len(r):<3d}"
              f" {max(x[2] for x in r):9.2e}")


def main():
    files = []
    for arg in sys.argv[1:]:
        label, _, path = arg.rpartition("=")
        files.append((label or path, read(path)))
    ref = {}
    for _, sections in files:
        ref = {name.rsplit("/", 1)[1]: v for name, v in sections.items()
               if name.startswith("f32/imperative/")}
        if ref:
            break
    if "prompt" not in ref:
        print("no f32/imperative sections: the reference is missing")
        return 1
    ref_lp = {what: log_softmax(v) for what, v in ref.items()}
    for label, sections in files:
        print(label)
        report(sections, ref, ref_lp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
