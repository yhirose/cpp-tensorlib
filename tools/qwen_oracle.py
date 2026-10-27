#!/usr/bin/env python3
"""Tight numpy oracle for bench/models/check_qwen.cpp.

Reads a Qwen2.5 / Qwen3 F16 GGUF with its own minimal parser and runs the decoder
in float32 numpy on the SAME weights the C++ loads, so the two must agree to
~1e-4 at every checkpoint. (Not torch on the HF weights: those are bf16, the GGUF
is F16, and the two drift apart within a few tokens.) Writes a C++ header with
the checkpoints the check compares: last-token embedding, layer-0 residual, final
norm, top-5 logits, and the greedy continuation.

  python3 -I tools/qwen_oracle.py MODEL.gguf NAMESPACE > bench/models/X_oracle_data.h

The prompt is fixed token ids (the chat template around one user sentence), so
no tokenizer is involved; Qwen2.5 and Qwen3 differ only in the system prompt.
"""
import mmap
import struct
import sys

import numpy as np

USER = [35127, 752, 264, 2805, 16800, 311, 3460, 4128, 4119, 13]  # "Give me a short introduction to large language models."
QWEN2_SYS = [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364,
             14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198]
N_GREEDY = 30


class Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0]

    def string(self):
        n = self.take("Q")
        s = bytes(self.b[self.o:self.o + n]).decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                   10: "Q", 11: "q", 12: "d"}
        if t in scalars:
            return self.take(scalars[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"gguf value type {t}")


def load_gguf(path):
    f = open(path, "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    r = Reader(buf)
    assert bytes(buf[:4]) == b"GGUF"
    r.o = 4
    r.take("I")  # version
    n_tensors, n_kv = r.take("Q"), r.take("Q")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.take("I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("Q") for _ in range(r.take("I"))]
        ty, off = r.take("I"), r.take("Q")
        infos.append((name, dims, ty, off))
    align = kv.get("general.alignment", 32)
    base = (r.o + align - 1) // align * align
    tensors = {}
    for name, dims, ty, off in infos:
        assert ty in (0, 1), f"{name}: ggml type {ty} (F32/F16 only)"
        dt = np.float32 if ty == 0 else np.float16
        n = int(np.prod(dims))
        a = np.frombuffer(buf, dtype=dt, count=n, offset=base + off)
        tensors[name] = a.reshape(dims[::-1]).astype(np.float32)  # GGML dims are reversed
    return kv, tensors


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, base):  # x [H, D], half-split pairs (i, i + D/2)
    d = x.shape[-1]
    ang = pos * base ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


class Model:
    def __init__(self, path):
        kv, self.t = load_gguf(path)
        arch = kv["general.architecture"]
        g = lambda k: kv[f"{arch}.{k}"]
        self.nl, self.ne = g("block_count"), g("embedding_length")
        self.nh, self.nkv = g("attention.head_count"), g("attention.head_count_kv")
        self.hd = kv.get(f"{arch}.attention.key_length", self.ne // self.nh)
        self.eps, self.base = g("attention.layer_norm_rms_epsilon"), g("rope.freq_base")
        self.arch = arch
        self.k = [[] for _ in range(self.nl)]
        self.v = [[] for _ in range(self.nl)]
        self.lm = self.t.get("output.weight", self.t["token_embd.weight"])

    def layer(self, l, x, pos):
        t, p = self.t, f"blk.{l}."
        h = rmsnorm(x, t[p + "attn_norm.weight"], self.eps)
        q, k, v = (h @ t[p + n + ".weight"].T for n in ("attn_q", "attn_k", "attn_v"))
        if p + "attn_q.bias" in t:
            q, k, v = q + t[p + "attn_q.bias"], k + t[p + "attn_k.bias"], v + t[p + "attn_v.bias"]
        q, k, v = q.reshape(self.nh, self.hd), k.reshape(self.nkv, self.hd), v.reshape(self.nkv, self.hd)
        if p + "attn_q_norm.weight" in t:  # Qwen3: per-head q/k RMSNorm before rope
            q = rmsnorm(q, t[p + "attn_q_norm.weight"], self.eps)
            k = rmsnorm(k, t[p + "attn_k_norm.weight"], self.eps)
        q, k = rope(q, pos, self.base), rope(k, pos, self.base)
        self.k[l].append(k)
        self.v[l].append(v)
        K, V = np.stack(self.k[l], 1), np.stack(self.v[l], 1)  # [kv, T, D]
        g = self.nh // self.nkv
        out = np.empty_like(q)
        for hh in range(self.nh):
            s = K[hh // g] @ q[hh] / np.sqrt(self.hd)
            e = np.exp(s - s.max())
            out[hh] = (e / e.sum()) @ V[hh // g]
        x = x + out.reshape(-1) @ t[p + "attn_output.weight"].T
        h2 = rmsnorm(x, t[p + "ffn_norm.weight"], self.eps)
        gt, up = h2 @ t[p + "ffn_gate.weight"].T, h2 @ t[p + "ffn_up.weight"].T
        with np.errstate(over="ignore"):  # exp(-gt) -> inf, silu -> 0: the right limit
            act = gt / (1 + np.exp(-gt))
        return x + (act * up) @ t[p + "ffn_down.weight"].T

    def step(self, tok, pos):
        x = self.t["token_embd.weight"][tok]
        emb, l0 = x.copy(), None
        for l in range(self.nl):
            x = self.layer(l, x, pos)
            if l == 0:
                l0 = x.copy()
        fn = rmsnorm(x, self.t["output_norm.weight"], self.eps)
        return emb, l0, fn, fn @ self.lm.T


def arr(name, ctype, a):
    a = np.asarray(a).reshape(-1)
    body = ",".join(repr(float(x)) if ctype == "float" else str(int(x)) for x in a)
    return f"inline const {ctype} {name}[{len(a)}] = {{{body}}};"


def main():
    path, ns = sys.argv[1], sys.argv[2]
    m = Model(path)
    prompt = (QWEN2_SYS if m.arch == "qwen2" else []) + [151644, 872, 198] + USER + [151645, 198, 151644, 77091, 198]
    pos = 0
    for tok in prompt:
        emb, l0, fn, logits = m.step(tok, pos)
        pos += 1
    top5 = np.argsort(-logits)[:5]
    greedy, nxt = [], int(np.argmax(logits))
    for _ in range(N_GREEDY):
        greedy.append(nxt)
        nxt = int(np.argmax(m.step(nxt, pos)[3]))
        pos += 1
    print("#pragma once")
    print(f"// Generated by tools/qwen_oracle.py from {path.split('/')[-1]} ({m.arch}); do not edit.")
    print("// numpy forward on the SAME F16 weights; C++ must match to ~1e-4 + greedy exactly.")
    print(f"namespace {ns} {{")
    print(arr("emb_last", "float", emb))
    print(arr("l0_last", "float", l0))
    print(arr("fnorm_last", "float", fn))
    print(arr("top5_ids", "int", top5))
    print(arr("top5_vals", "float", logits[top5]))
    print(arr("greedy30", "int", greedy))
    print(arr("prompt_ids", "int", prompt))
    print("}")


if __name__ == "__main__":
    main()
