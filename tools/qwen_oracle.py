#!/usr/bin/env python3
"""Tight numpy oracle for bench/models/check_qwen.cpp (Qwen2.5, Qwen3, Qwen3.5, Gemma 3, Llama 3.2).

Reads a Qwen2.5 / Qwen3 / Qwen3.5 F16 (or BF16) GGUF with its own minimal parser and runs the decoder
in float32 numpy on the SAME weights the C++ loads, so the two must agree to
~1e-4 at every checkpoint. (Not torch on the HF weights: those are bf16, the GGUF
is F16, and the two drift apart within a few tokens.) Writes a C++ header with
the checkpoints the check compares: last-token embedding, layer-0 residual, final
norm, top-5 logits, and the greedy continuation.

A Gemma 3 GGUF is block-quantized (Q4_K / Q6_K / Q5_0 / Q8_0): its weights are
read with the gguf package (llama.cpp's gguf-py) and dequantized to f32, all at
once when that copy is under KEEP_ALL_BYTES (Gemma 3 1B's 4 GB), else when a step
reads them, layer by layer, so no f32 copy of the whole model is held (the 4B's
16 GB). Its local layers attend the last `sliding_window` positions.

A Llama GGUF's q and k rows are as llama.cpp reads them: the converter permuted
them for its NORM rope, which turns adjacent pairs. This forward turns them so,
on the file's rows, where the C++ reads them back in the model's order and turns
halves (qwen.h's wpart::norm_hd): two independent routes to the same numbers.
Llama 3's rope_freqs.weight divides the pairs' angles, as llama.cpp's freq factors.

  python3 -I tools/qwen_oracle.py MODEL.gguf NAMESPACE [LLAMA_TOKENIZE] > bench/models/X_oracle_data.h

The prompt is fixed token ids (the chat template around one user sentence), so
no tokenizer is involved; Qwen2.5 and Qwen3 differ only in the system prompt.
Given llama.cpp's llama-tokenize, the user turn is LONG_USER instead, tokenized
by it: a prompt past a sliding window (Gemma 3 1B's 512), so the oracle holds
the window.
"""
import math
import mmap
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I
from tokenizer_oracle import tokenize  # noqa: E402

USER35 = [33963, 728, 264, 2716, 16308, 310, 3349, 3992, 3983, 13]  # same sentence, Qwen3.5 vocab
USER = [35127, 752, 264, 2805, 16800, 311, 3460, 4128, 4119, 13]  # "Give me a short introduction to large language models."
LLAMA_USER = [36227, 757, 264, 2875, 17219, 311, 3544, 4221, 4211, 13]  # same sentence, Llama 3 vocab
QWEN2_SYS = [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364,
             14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198]
GEMMA_USER = [46762, 786, 496, 2822, 12650, 531, 2455, 5192, 4681, 236761]  # same sentence, Gemma 3 vocab
N_GREEDY = 30
KEEP_ALL_BYTES = 8 << 30

# The long user turn: 760 tokens of prose (Gemma 3) and a question on its beginning.
LONG_USER = """The lighthouse at Gray Point stands on a narrow spit of granite where two currents meet. It was first lit in the autumn of 1841, after a winter in which three cargo ships ran aground within sight of the village harbor. The first keeper, a former ship's carpenter named Elias Brandt, kept a logbook from the night the lamp was lit, and his successors kept it after him for one hundred and twelve years. The volumes, bound in canvas and stored in a dry room beneath the lantern gallery, record the weather twice a day, every ship that passed, the oil burned each night, and the small events of a life lived far from town.

Brandt's entries are short and practical. He notes the direction of the wind, the height of the swell, and the hours when fog closed over the point. In the spring of 1843 he writes that the lamp's wick had to be trimmed eleven times in one night because the oil was poor, and that he had written to the harbor board to complain. Three weeks later he records, with obvious satisfaction, the arrival of a better grade of oil and a new set of brass tools. Most of his entries end with the same phrase: all well at the light.

His daughter, Margarethe, took over the logbook when his eyesight began to fail. Her handwriting is smaller and neater, and her entries are longer. She describes the birds that struck the lantern glass during migration, the seals that rested on the rocks at low tide, and the children from the village who rowed out on summer afternoons to ask for a tour of the tower. She also kept careful records of the fog signal, a bell that had to be rung by hand every thirty seconds whenever visibility fell below a mile. During one week in November 1867, she and her husband rang the bell in shifts for ninety hours without a break.

The logbook changes again at the end of the century. A clockwork mechanism replaced the hand-rung bell in 1889, and a kerosene lamp with a larger lens replaced the old oil lamp in 1896. The entries from these years are filled with notes about adjustments, broken springs, and the slow work of learning a new machine. The keepers, now employed by the national lighthouse service rather than the harbor board, were also required to send monthly reports to an inspector in the city, and the logbook begins to read more like an official record than a diary.

Electricity reached Gray Point in 1931, carried along poles that followed the coast road. The keeper at the time, Thomas Okafor, wrote that the new lamp was so bright that fishermen ten miles out could read their charts by it, which was surely an exaggeration, but his pride is easy to understand. With electric light and an automatic fog horn, the work of the station became lighter, and the families who lived there had more time for gardens, for teaching their children, and for the long walks along the cliffs that several keepers mention.

The last entry in the logbook was written on the first of March, 1953, the day the station was fully automated. The final keeper, a quiet man named Samuel Reyes, wrote only a few lines. He listed the weather, noted that a fishing boat had passed to the north at dawn, and then, after a blank line, added the words his predecessors had written so many times before: all well at the light. The logbooks are now kept in the village museum, where visitors can read them in a glass case near the door.

Who kept the logbook first, and what phrase did the keepers often write? Answer briefly."""


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


def load_gguf(path, read=True):
    """(metadata, name -> f32 array); with read=False, (metadata, the ggml types)."""
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
    if not read:
        return kv, [ty for _, _, ty, _ in infos]
    align = kv.get("general.alignment", 32)
    base = (r.o + align - 1) // align * align
    tensors = {}
    for name, dims, ty, off in infos:
        assert ty in (0, 1, 30), f"{name}: ggml type {ty} (F32/F16/BF16 only)"
        n = int(np.prod(dims))
        if ty == 30:  # bf16: the high half of an f32
            u = np.frombuffer(buf, dtype=np.uint16, count=n, offset=base + off)
            a = (u.astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.frombuffer(buf, dtype=np.float32 if ty == 0 else np.float16, count=n,
                              offset=base + off)
        tensors[name] = a.reshape(dims[::-1]).astype(np.float32)  # GGML dims are reversed
    return kv, tensors


class Dequantized:
    """name -> f32 array, dequantized from the GGUF's blocks: each once and kept when
    the whole f32 copy is under KEEP_ALL_BYTES, else on each read (small tensors and
    the embedding kept)."""

    def __init__(self, path):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.dequantize = dequantize
        r = GGUFReader(path)
        self.info = {t.name: t for t in r.tensors}
        self.kv = {k: f.contents() for k, f in r.fields.items() if not k.startswith("tokenizer.")}
        self.kept = {}
        self.keep_all = sum(int(t.n_elements) for t in r.tensors) * 4 < KEEP_ALL_BYTES

    def __contains__(self, name):
        return name in self.info

    def get(self, name, default=None):
        return self[name] if name in self.info else default

    def __getitem__(self, name):
        if name in self.kept:
            return self.kept[name]
        t = self.info[name]
        a = np.asarray(self.dequantize(t.data, t.tensor_type), np.float32)
        a = a.reshape([int(d) for d in t.shape[::-1]])
        if self.keep_all or a.size < 1 << 20 or name == "token_embd.weight":
            self.kept[name] = a
        return a


def is_quantized(path):  # any weight not F32 / F16 / BF16 (Gemma 3's: Q4_K / Q6_K)
    return any(ty not in (0, 1, 30) for ty in load_gguf(path, read=False)[1])


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, base, rot=None, factors=None, norm=False):
    """x [H, D]: its first `rot` dims turn as pairs (j, j + rot/2), or with `norm`
    (llama.cpp's NORM) (2j, 2j + 1), by pos * base^(-2j/rot) / factors[j]."""
    rot = rot or x.shape[-1]
    ang = pos * base ** (-np.arange(0, rot, 2, dtype=np.float64) / rot)
    if factors is not None:
        ang = ang / factors.astype(np.float64)
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    ia, ib = (slice(0, rot, 2), slice(1, rot, 2)) if norm else (slice(0, rot // 2), slice(rot // 2, rot))
    a, b, out = x[..., ia], x[..., ib], x.copy()
    out[..., ia], out[..., ib] = a * c - b * s, a * s + b * c
    return out


def gelu(x):  # tanh form: ggml's, and PyTorch's approximate='tanh'
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * x * (1 + 0.044715 * x * x)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    with np.errstate(over="ignore"):  # exp(-x) -> inf, silu -> 0: the right limit
        return x / (1 + np.exp(-x))


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


class Model:
    def __init__(self, path):
        # Gemma 3's GGUFs are block-quantized; Qwen's are read whole as F16 / BF16.
        if is_quantized(path):
            self.t = Dequantized(path)
            kv = self.t.kv
        else:
            kv, self.t = load_gguf(path)
        arch = kv["general.architecture"]
        g = lambda k: kv[f"{arch}.{k}"]
        self.nl, self.ne = g("block_count"), g("embedding_length")
        self.nh, self.nkv = g("attention.head_count"), g("attention.head_count_kv")
        self.hd = kv.get(f"{arch}.attention.key_length", self.ne // self.nh)
        self.eps, self.base = g("attention.layer_norm_rms_epsilon"), g("rope.freq_base")
        self.arch = arch
        self.rot = kv.get(f"{arch}.rope.dimension_count", self.hd)
        # Qwen3.5: every `interval`-th layer is full attention, the rest Gated DeltaNet.
        self.interval = kv.get(f"{arch}.full_attention_interval", 1)
        if arch == "qwen35":
            self.ssm_heads = g("ssm.time_step_rank")
            self.ssm_dk = g("ssm.state_size")
            self.ssm_conv = g("ssm.conv_kernel")
            self.conv_state = [None] * self.nl  # [conv-1, C] last inputs
            self.state = [None] * self.nl       # [heads, dk, dv]
        # RoPE: a linear scaling on the global layers; Gemma 3's local layers
        # (sliding window) turn at their own base, unscaled. As llama.cpp
        # (src/models/gemma3.cpp, load_swa_pattern); see qwen.h's Config.
        linear = kv.get(f"{arch}.rope.scaling.type") == "linear"
        self.scale = 1 / g("rope.scaling.factor") if linear else 1.0
        self.local = [False] * self.nl
        self.window = kv.get(f"{arch}.attention.sliding_window", 0) if arch == "gemma3" else 0
        if self.window > 0:
            pat = kv.get(f"{arch}.attention.sliding_window_pattern", 6)
            self.local = ([bool(b) for b in pat] if isinstance(pat, list)
                          else [pat == 0 or l % pat < pat - 1 for l in range(self.nl)])
            self.base_local = kv.get(f"{arch}.rope.freq_base_swa", 10000.0)
        # Llama: NORM rope on the file's q / k rows, and its per-pair factors on the
        # global layers (llama.cpp's LLM_ARCH_LLAMA; qwen.h's rope_of).
        self.norm = arch == "llama"
        self.factors = self.t.get("rope_freqs.weight")
        self.k = [[] for _ in range(self.nl)]
        self.v = [[] for _ in range(self.nl)]
        self.lm = self.t.get("output.weight", self.t["token_embd.weight"])

    def is_delta(self, l):
        return self.arch == "qwen35" and (l + 1) % self.interval != 0

    def delta(self, l, h):
        """Gated DeltaNet, one token: conv -> l2norm -> delta-rule state update -> gated norm."""
        t, p = self.t, f"blk.{l}."
        H, dk = self.ssm_heads, self.ssm_dk
        qkv = h @ t[p + "attn_qkv.weight"].T
        z = (h @ t[p + "attn_gate.weight"].T).reshape(H, dk)
        if self.conv_state[l] is None:
            self.conv_state[l] = np.zeros((self.ssm_conv - 1, qkv.shape[0]), np.float32)
            self.state[l] = np.zeros((H, dk, dk), np.float32)
        win = np.concatenate([self.conv_state[l], qkv[None]], 0)  # [conv, C], oldest first
        self.conv_state[l] = win[1:]
        mixed = silu((win * t[p + "ssm_conv1d.weight"].T).sum(0))
        q, k, v = (mixed[i * H * dk:(i + 1) * H * dk].reshape(H, dk) for i in range(3))
        q, k = l2norm(q) / math.sqrt(dk), l2norm(k)
        beta = sigmoid(h @ t[p + "ssm_beta.weight"].T)
        a = h @ t[p + "ssm_alpha.weight"].T + t[p + "ssm_dt.bias"]
        decay = np.exp(t[p + "ssm_a"] * np.logaddexp(0, a))  # ssm_a = -exp(A_log) < 0
        S = self.state[l] * decay[:, None, None]
        delta = (v - np.einsum("hkv,hk->hv", S, k)) * beta[:, None]
        S = S + np.einsum("hk,hv->hkv", k, delta)
        self.state[l] = S
        o = np.einsum("hkv,hk->hv", S, q)
        o = rmsnorm(o, t[p + "ssm_norm.weight"], self.eps) * silu(z)
        return o.reshape(-1) @ t[p + "ssm_out.weight"].T

    def layer(self, l, x, pos):
        t, p = self.t, f"blk.{l}."
        h = rmsnorm(x, t[p + "attn_norm.weight"], self.eps)
        if self.is_delta(l):
            x = x + self.delta(l, h)
            return self.mlp(l, x)
        q, k, v = (h @ t[p + n + ".weight"].T for n in ("attn_q", "attn_k", "attn_v"))
        gate = None
        if self.arch == "qwen35":  # q|gate interleaved per head
            q = q.reshape(self.nh, 2 * self.hd)
            q, gate = q[:, : self.hd], q[:, self.hd:]
            q = q.reshape(-1)
        if p + "attn_q.bias" in t:
            q, k, v = q + t[p + "attn_q.bias"], k + t[p + "attn_k.bias"], v + t[p + "attn_v.bias"]
        q, k, v = q.reshape(self.nh, self.hd), k.reshape(self.nkv, self.hd), v.reshape(self.nkv, self.hd)
        if p + "attn_q_norm.weight" in t:  # Qwen3: per-head q/k RMSNorm before rope
            q = rmsnorm(q, t[p + "attn_q_norm.weight"], self.eps)
            k = rmsnorm(k, t[p + "attn_k_norm.weight"], self.eps)
        base, p_, f = ((self.base_local, pos, None) if self.local[l]
                       else (self.base, pos * self.scale, self.factors))
        q, k = (rope(y, p_, base, self.rot, f, self.norm) for y in (q, k))
        self.k[l].append(k)
        self.v[l].append(v)
        # A local layer reads the last `window` positions, this one the last of them
        # (llama.cpp's is_masked_swa; gpu::attn_window_lo).
        lo = max(0, pos + 1 - self.window) if self.local[l] else 0
        K, V = np.stack(self.k[l][lo:], 1), np.stack(self.v[l][lo:], 1)  # [kv, T, D]
        g = self.nh // self.nkv
        out = np.empty_like(q)
        for hh in range(self.nh):
            s = K[hh // g] @ q[hh] / math.sqrt(self.hd)
            e = np.exp(s - s.max())
            out[hh] = (e / e.sum()) @ V[hh // g]
        if gate is not None:
            out = out * sigmoid(gate)
        o = out.reshape(-1) @ t[p + "attn_output.weight"].T
        if p + "post_ffw_norm.weight" in t:  # Gemma 3: a norm after the attention too
            o = rmsnorm(o, t[p + "post_attention_norm.weight"], self.eps)
        return self.mlp(l, x + o)

    def mlp(self, l, x):
        t, p = self.t, f"blk.{l}."
        # Qwen3.5 names the pre-FFN norm post_attention_norm.
        n = p + ("ffn_norm.weight" if p + "ffn_norm.weight" in t else "post_attention_norm.weight")
        h2 = rmsnorm(x, t[n], self.eps)
        gt, up = h2 @ t[p + "ffn_gate.weight"].T, h2 @ t[p + "ffn_up.weight"].T
        o = ((gelu if self.arch == "gemma3" else silu)(gt) * up) @ t[p + "ffn_down.weight"].T
        if p + "post_ffw_norm.weight" in t:  # Gemma 3
            o = rmsnorm(o, t[p + "post_ffw_norm.weight"], self.eps)
        return x + o

    def step(self, tok, pos, logits=True):
        x = self.t["token_embd.weight"][tok]
        emb, l0 = x.copy(), None
        if self.arch == "gemma3":
            x = x * np.float32(np.sqrt(np.float32(self.ne)))
        for l in range(self.nl):
            x = self.layer(l, x, pos)
            if l == 0:
                l0 = x.copy()
        # float32 throughout: under numpy 2 a numpy float64 scalar (np.sqrt(2 / np.pi))
        # times a float32 array is float64, and every GEMV after it widens its weights.
        assert x.dtype == np.float32, x.dtype
        fn = rmsnorm(x, self.t["output_norm.weight"], self.eps)
        return emb, l0, fn, fn @ self.lm.T if logits else None


def arr(name, ctype, a):
    a = np.asarray(a).reshape(-1)
    body = ",".join(repr(float(x)) if ctype == "float" else str(int(x)) for x in a)
    return f"inline const {ctype} {name}[{len(a)}] = {{{body}}};"


def main():
    path, ns = sys.argv[1], sys.argv[2]
    m = Model(path)
    long = tokenize(sys.argv[3], path, LONG_USER) if len(sys.argv) > 3 else None
    if m.arch == "gemma3":  # <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        prompt = [2, 105, 2364, 107] + (long or GEMMA_USER) + [106, 107, 105, 4368, 107]
    elif m.arch == "llama":  # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...
        prompt = [128000, 128006, 882, 128007, 271] + (long or LLAMA_USER) + [128009, 128006, 78191, 128007, 271]
    elif m.arch == "qwen35":  # thinking template: ends with "<think>\n"
        prompt = [248045, 846, 198] + (long or USER35) + [248046, 198, 248045, 74455, 198, 248068, 198]
    else:
        prompt = ((QWEN2_SYS if m.arch == "qwen2" else []) + [151644, 872, 198] + (long or USER) +
                  [151645, 198, 151644, 77091, 198])
    total, t0 = len(prompt) + N_GREEDY, time.perf_counter()

    def step(tok, pos, logits=True):  # a run takes minutes: its progress on stderr
        out = m.step(tok, pos, logits)
        if pos == 0 or (pos + 1) % 25 == 0 or pos + 1 == total:
            print(f"{pos + 1}/{total} tokens, {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
        return out

    pos = 0
    for tok in prompt:  # the logits of the last prompt token only (the lm head is the largest read)
        emb, l0, fn, logits = step(tok, pos, pos == len(prompt) - 1)
        pos += 1
    top5 = np.argsort(-logits)[:5]
    greedy, nxt = [], int(np.argmax(logits))
    for _ in range(N_GREEDY):
        greedy.append(nxt)
        nxt = int(np.argmax(step(nxt, pos)[3]))
        pos += 1
    print("#pragma once")
    print(f"// Generated by tools/qwen_oracle.py from {path.split('/')[-1]} ({m.arch}); do not edit.")
    print("// numpy float32 forward on the SAME weights (F16, or blocks dequantized exactly); C++ must")
    print(f"// match to ~1e-4 + greedy exactly. Prompt: {len(prompt)} tokens"
          f"{' (LONG_USER, tokenized by llama-tokenize)' if long else ''}.")
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
