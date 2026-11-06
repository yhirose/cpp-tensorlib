#!/usr/bin/env python3
"""Tight numpy oracle for bench/models/check_qwen.cpp (Qwen2.5, Qwen3, Qwen3.5, Gemma 3, Gemma 4,
Llama 3.2, PLaMo 3).

Reads a Qwen2.5 / Qwen3 / Qwen3.5 F16 (or BF16) GGUF with its own minimal parser and runs the decoder
in float32 numpy on the SAME weights the C++ loads, so the two must agree to
~1e-4 at every checkpoint. (Not torch on the HF weights: those are bf16, the GGUF
is F16, and the two drift apart within a few tokens.) Writes a C++ header with
the checkpoints the check compares: last-token embedding, layer-0 residual, final
norm, top-5 logits, and the greedy continuation.

A block-quantized GGUF (Gemma 3's, or a Q4_K_M of the others) has its weights
read with the gguf package (llama.cpp's gguf-py) and dequantized to f32, all at
once when that copy is under KEEP_ALL_BYTES (Gemma 3 1B's 4 GB), else when a step
reads them, layer by layer, so no f32 copy of the whole model is held (the 4B's
16 GB). Gemma 3's local layers attend the last `sliding_window` positions.

A Llama GGUF's q and k rows are as llama.cpp reads them: the converter permuted
them for its NORM rope, which turns adjacent pairs. This forward turns them so,
on the file's rows, where the C++ reads them back in the model's order and turns
halves (qwen.h's wpart::norm_hd): two independent routes to the same numbers.
Llama 3's rope_freqs.weight divides the pairs' angles, as llama.cpp's freq factors.

PLaMo 3 (its Translate models) as PFN's modeling_plamo.py: q|k|v in one tensor and
gate|up in one, post norms (their offsets already in the GGUF's weights), a
sliding window on 7 layers of 8, and YaRN on the global ones, unrounded
(truncate false): the pairs' frequencies blended toward 1/factor, cos and sin
times 0.1 ln(factor) + 1. Its prompt is a translation (PFML3) of LONG_USER, four
times over: 2,542 tokens, past the 2048-position window.

Gemma 4 (E2B) as HF's modeling_gemma4.py, which llama.cpp's gemma4.cpp follows: heads of
256 on the local layers and 512 on the global ones, which turn only their first 64 pairs
(rope_freqs.weight: 1 for those, 1e30 for the rest), v RMS-normalized without a weight,
an attention scale of 1, the per-layer embeddings (a token's own row of
per_layer_token_embd, gathered, never the whole table, plus a projection of its
embedding), the last 20 layers reading the KV of the last layer of their kind before
them, each layer's output times its layer_output_scale, and the logits capped at 30
by tanh.

The forward takes a run of tokens at once (the prompt, then one a step): each
weight is read once a run, so a model whose f32 copy is not kept (the 2B's 10.7
GB) dequantizes it once for the prompt rather than once a token. Qwen3.5's
DeltaNet layers still go token by token within the run.

  python3 -I tools/qwen_oracle.py MODEL.gguf NAMESPACE [LLAMA_TOKENIZE] > bench/models/X_oracle_data.h

(PLaMo 3 needs LLAMA_TOKENIZE: its prompt is text.)

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

    def rows(self, name, ids):
        """Rows `ids` of a 2-D tensor, only those dequantized (Gemma 4's per-layer
        embeddings: 9.4 GB as f32 whole)."""
        t = self.info[name]
        return np.asarray(self.dequantize(t.data[np.asarray(ids)], t.tensor_type), np.float32)

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


def rope(x, pos, base, rot=None, factors=None, norm=False, mscale=1.0):
    """x [T, H, D], token t at position pos[t]: its first `rot` dims turn as pairs
    (j, j + rot/2), or with `norm` (llama.cpp's NORM) (2j, 2j + 1), by
    pos * base^(-2j/rot) / factors[j], cos and sin times mscale (YaRN's)."""
    rot = rot or x.shape[-1]
    ang = np.asarray(pos, np.float64)[:, None] * base ** (-np.arange(0, rot, 2, dtype=np.float64) / rot)
    if factors is not None:
        ang = ang / factors.astype(np.float64)
    c = (np.cos(ang) * mscale).astype(np.float32)[:, None, :]
    s = (np.sin(ang) * mscale).astype(np.float32)[:, None, :]
    ia, ib = (slice(0, rot, 2), slice(1, rot, 2)) if norm else (slice(0, rot // 2), slice(rot // 2, rot))
    a, b, out = x[..., ia], x[..., ib], x.copy()
    out[..., ia], out[..., ib] = a * c - b * s, a * s + b * c
    return out


def yarn_divisors(base, rot, factor, orig_ctx, beta_fast, beta_slow, truncate):
    """YaRN's pair divisors, as PFN's RotaryEmbedding._compute_buffers (and HF's
    _compute_yarn_parameters): pair j's frequency times g_j = ramp_j / factor +
    (1 - ramp_j), ramp_j rising from 0 at `low` to 1 at `high`."""
    dim = lambda beta: rot * math.log(orig_ctx / (beta * 2 * math.pi)) / (2 * math.log(base))
    low, high = dim(beta_fast), dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rot - 1)
    ramp = np.clip((np.arange(rot // 2, dtype=np.float64) - low) / max(high - low, 0.001), 0, 1)
    return 1 / (ramp / factor + (1 - ramp))


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
        # Gemma 4: the local layers' heads and RoPE have their own width, the MLP's
        # width is per layer, and the last `shared` layers keep no KV of their own.
        self.hd_local = kv.get(f"{arch}.attention.key_length_swa", self.hd)
        self.rot_local = kv.get(f"{arch}.rope.dimension_count_swa", self.hd_local)
        self.shared = kv.get(f"{arch}.attention.shared_kv_layers", 0)
        self.ple = kv.get(f"{arch}.embedding_length_per_layer_input", 0)
        self.softcap = kv.get(f"{arch}.final_logit_softcapping", 0.0) if arch == "gemma4" else 0.0
        # Qwen3.5: every `interval`-th layer is full attention, the rest Gated DeltaNet.
        self.interval = kv.get(f"{arch}.full_attention_interval", 1)
        if arch == "qwen35":
            self.ssm_heads = g("ssm.time_step_rank")
            self.ssm_dk = g("ssm.state_size")
            self.ssm_conv = g("ssm.conv_kernel")
            self.conv_state = [None] * self.nl  # [conv-1, C] last inputs
            self.state = [None] * self.nl       # [heads, dk, dv]
        # RoPE: a linear scaling (Gemma 3) or YaRN (PLaMo 3) on the global layers;
        # the local layers (sliding window) turn at their own base, unscaled. As
        # llama.cpp (src/models/gemma3.cpp and plamo3.cpp, load_swa_pattern with
        # their default patterns, 6 and 8); see qwen.h's Config.
        scaling = kv.get(f"{arch}.rope.scaling.type", "none")
        self.scale = 1 / g("rope.scaling.factor") if scaling == "linear" else 1.0
        self.local = [False] * self.nl
        self.window = kv.get(f"{arch}.attention.sliding_window", 0)
        if self.window > 0:
            pat = kv.get(f"{arch}.attention.sliding_window_pattern", 8 if arch == "plamo3" else 6)
            self.local = ([bool(b) for b in pat] if isinstance(pat, list)
                          else [pat == 0 or l % pat < pat - 1 for l in range(self.nl)])
            self.base_local = kv.get(f"{arch}.rope.freq_base_swa", 10000.0)
        # Llama: NORM rope on the file's q / k rows, and its per-pair factors on the
        # global layers (llama.cpp's LLM_ARCH_LLAMA; qwen.h's rope_of).
        self.norm = arch == "llama"
        self.factors = self.t.get("rope_freqs.weight")
        self.mscale = 1.0
        if scaling == "yarn":  # PLaMo 3
            factor = g("rope.scaling.factor")
            self.factors = yarn_divisors(self.base, self.rot, factor, g("rope.scaling.original_context_length"),
                                         kv.get(f"{arch}.rope.scaling.yarn_beta_fast", 32.0),
                                         kv.get(f"{arch}.rope.scaling.yarn_beta_slow", 1.0),
                                         bool(kv.get(f"{arch}.rope.scaling.yarn_truncate", True)))
            self.mscale = 0.1 * math.log(factor) + 1
        self.k = [np.zeros((0, self.nkv, self.head_dim(l)), np.float32) for l in range(self.nl)]
        self.v = [np.zeros((0, self.nkv, self.head_dim(l)), np.float32) for l in range(self.nl)]
        self.lm = self.t.get("output.weight", self.t["token_embd.weight"])

    def head_dim(self, l):
        return self.hd_local if self.local[l] else self.hd

    def kv_source(self, l):
        """The layer whose KV layer l reads: its own, or (Gemma 4's last `shared`)
        the last layer of its kind before them (llama.cpp's kv_source)."""
        first = self.nl - self.shared
        if l < first:
            return l
        return max(j for j in range(first) if self.local[j] == self.local[l])

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

    def w(self, p, base):
        """A layer tensor, named with .weight or bare (PLaMo 3's post norms); None if neither."""
        for n in (p + base + ".weight", p + base):
            if n in self.t:
                return self.t[n]
        return None

    def layer(self, l, x, pos):
        """x [T, NE], token t at position pos[t]. The mixer's run-long temporaries
        (scores [T, ctx] a head) are gone before the MLP's weights are read."""
        h = rmsnorm(x, self.t[f"blk.{l}.attn_norm.weight"], self.eps)
        if self.is_delta(l):
            x = x + np.stack([self.delta(l, r) for r in h])
        else:
            x = x + self.attn(l, h, pos)
        del h
        return self.mlp(l, x)

    def attn(self, l, h, pos):
        if self.arch == "gemma4":
            return self.attn4(l, h, pos)
        t, p = self.t, f"blk.{l}."
        T, Q, KV = len(h), self.nh * self.hd, self.nkv * self.hd
        if p + "attn_qkv.weight" in t:  # PLaMo 3: q|k|v in one
            qkv = h @ t[p + "attn_qkv.weight"].T
            q, k, v = qkv[:, :Q], qkv[:, Q:Q + KV], qkv[:, Q + KV:]
        else:
            q, k, v = (h @ t[p + n + ".weight"].T for n in ("attn_q", "attn_k", "attn_v"))
        gate = None
        if self.arch == "qwen35":  # q|gate interleaved per head
            q = q.reshape(T, self.nh, 2 * self.hd)
            q, gate = q[..., : self.hd].reshape(T, Q), q[..., self.hd:].reshape(T, Q)
        if p + "attn_q.bias" in t:
            q, k, v = q + t[p + "attn_q.bias"], k + t[p + "attn_k.bias"], v + t[p + "attn_v.bias"]
        q, k, v = q.reshape(T, self.nh, self.hd), k.reshape(T, self.nkv, self.hd), v.reshape(T, self.nkv, self.hd)
        if p + "attn_q_norm.weight" in t:  # Qwen3, PLaMo 3: per-head q/k RMSNorm before rope
            q = rmsnorm(q, t[p + "attn_q_norm.weight"], self.eps)
            k = rmsnorm(k, t[p + "attn_k_norm.weight"], self.eps)
        base, p_, f, ms = ((self.base_local, pos, None, 1.0) if self.local[l]
                           else (self.base, pos * self.scale, self.factors, self.mscale))
        q, k = (rope(y, p_, base, self.rot, f, self.norm, ms) for y in (q, k))
        self.k[l] = np.concatenate([self.k[l], k])
        self.v[l] = np.concatenate([self.v[l], v])
        out = self.attend(l, q, pos, np.float32(math.sqrt(self.hd))).reshape(T, Q)
        if gate is not None:
            out = out * sigmoid(gate)
        o = out @ t[p + "attn_output.weight"].T
        if self.w(p, "post_ffw_norm") is not None:  # Gemma 3, PLaMo 3: a norm after the attention too
            o = rmsnorm(o, self.w(p, "post_attention_norm"), self.eps)
        return o

    def attend(self, l, q, pos, div):
        """q [T, NH, D] over layer l's cache (its KV source's), the scores over `div`: causal, and a local
        layer reads the last `window` positions, this one the last of them
        (llama.cpp's is_masked_swa; gpu::attn_window_lo)."""
        K, V = self.k[self.kv_source(l)], self.v[self.kv_source(l)]  # [ctx, kv, D], position = row
        kp = np.arange(len(K))[None, :]
        live = kp <= pos[:, None]
        if self.local[l]:
            live &= kp > pos[:, None] - self.window
        g = self.nh // self.nkv
        out = np.empty_like(q)
        for hh in range(self.nh):
            sc = np.where(live, q[:, hh] @ K[:, hh // g].T / div, -np.inf)
            e = np.exp(sc - sc.max(-1, keepdims=True))
            out[:, hh] = (e / e.sum(-1, keepdims=True)) @ V[:, hh // g]
        return out

    def attn4(self, l, h, pos):
        """Gemma 4: q, k RMS-normalized by their weights and v by none, before the
        RoPE; an attention scale of 1; a layer past the KV-sharing line projects only
        q and reads its source's cache, which already holds this run."""
        t, p = self.t, f"blk.{l}."
        T, hd = len(h), self.head_dim(l)
        q = rmsnorm((h @ t[p + "attn_q.weight"].T).reshape(T, self.nh, hd), t[p + "attn_q_norm.weight"], self.eps)
        base, f = (self.base_local, None) if self.local[l] else (self.base, self.factors)
        rot = self.rot_local if self.local[l] else self.rot
        q = rope(q, pos, base, rot, f)
        if self.kv_source(l) == l:
            k = h @ t[p + "attn_k.weight"].T
            v = h @ t[p + "attn_v.weight"].T if p + "attn_v.weight" in t else k  # 12B/31B global: v is k
            k = rmsnorm(k.reshape(T, self.nkv, hd), t[p + "attn_k_norm.weight"], self.eps)
            v = rmsnorm(v.reshape(T, self.nkv, hd), np.float32(1), self.eps)
            self.k[l] = np.concatenate([self.k[l], rope(k, pos, base, rot, f)])
            self.v[l] = np.concatenate([self.v[l], v])
        o = self.attend(l, q, pos, np.float32(1)).reshape(T, self.nh * hd) @ t[p + "attn_output.weight"].T
        return rmsnorm(o, t[p + "post_attention_norm.weight"], self.eps)

    def mlp(self, l, x):
        t, p = self.t, f"blk.{l}."
        # Qwen3.5 names the pre-FFN norm post_attention_norm.
        w = self.w(p, "ffn_norm")
        h2 = rmsnorm(x, w if w is not None else self.w(p, "post_attention_norm"), self.eps)
        if p + "ffn_gate.weight" in t:
            gt, up = h2 @ t[p + "ffn_gate.weight"].T, h2 @ t[p + "ffn_up.weight"].T
        else:  # PLaMo 3: gate|up in one
            gt, up = np.split(h2 @ t[p + "ffn_up.weight"].T, 2, axis=-1)
        del h2
        a = (gelu if self.arch in ("gemma3", "gemma4") else silu)(gt) * up
        del gt, up  # the [T, 2 FF] product, before ffn_down is dequantized
        o = a @ t[p + "ffn_down.weight"].T
        post = self.w(p, "post_ffw_norm")
        if post is not None:  # Gemma 3, PLaMo 3
            o = rmsnorm(o, post, self.eps)
        return x + o

    def per_layer_inputs(self, toks, x):
        """Gemma 4's per-layer embeddings [T, NL, PLE] (HF's get_per_layer_inputs and
        project_per_layer_inputs): the token's row of per_layer_token_embd times
        sqrt(PLE), plus the embedding x (already times sqrt(NE)) projected, divided by
        sqrt(NE) and RMS-normalized a layer's slice at a time, the sum over sqrt(2)."""
        T, n = len(toks), self.ple
        tok = self.t.rows("per_layer_token_embd.weight", toks).reshape(T, self.nl, n)
        tok = tok * np.float32(math.sqrt(n))
        proj = (x @ self.t["per_layer_model_proj.weight"].T) * np.float32(1 / math.sqrt(self.ne))
        proj = rmsnorm(proj.reshape(T, self.nl, n), self.t["per_layer_proj_norm.weight"], self.eps)
        return (proj + tok) * np.float32(1 / math.sqrt(2))

    def layer_tail(self, l, x, ple):
        """Gemma 4's end of a layer: the per-layer embedding's gated projection added
        (post-normed), then the whole residual times the layer's scale."""
        t, p = self.t, f"blk.{l}."
        if ple is not None:
            g = gelu(x @ t[p + "inp_gate.weight"].T) * ple[:, l]
            x = x + rmsnorm(g @ t[p + "proj.weight"].T, t[p + "post_norm.weight"], self.eps)
        s = t.get(p + "layer_output_scale.weight")
        return x if s is None else x * s

    def run(self, toks, pos0):
        """The tokens at pos0.. through the model: the last one's embedding, layer-0
        residual, final norm and logits."""
        x = self.t["token_embd.weight"][np.asarray(toks)]
        emb, l0 = x[-1].copy(), None
        if self.arch in ("gemma3", "gemma4"):
            x = x * np.float32(np.sqrt(np.float32(self.ne)))
        ple = self.per_layer_inputs(toks, x) if self.ple else None
        pos = pos0 + np.arange(len(toks))
        for l in range(self.nl):
            x = self.layer(l, x, pos)
            if self.arch == "gemma4":
                x = self.layer_tail(l, x, ple)
            if l == 0:
                l0 = x[-1].copy()
        # float32 throughout: under numpy 2 a numpy float64 scalar (np.sqrt(2 / np.pi))
        # times a float32 array is float64, and every GEMV after it widens its weights.
        assert x.dtype == np.float32, x.dtype
        fn = rmsnorm(x[-1], self.t["output_norm.weight"], self.eps)
        logits = fn @ self.lm.T
        if self.softcap:  # Gemma 4: tanh(logits / cap) * cap
            logits = np.tanh(logits / np.float32(self.softcap)) * np.float32(self.softcap)
        return emb, l0, fn, logits


def arr(name, ctype, a):
    a = np.asarray(a).reshape(-1)
    body = ",".join(repr(float(x)) if ctype == "float" else str(int(x)) for x in a)
    return f"inline const {ctype} {name}[{len(a)}] = {{{body}}};"


def main():
    path, ns = sys.argv[1], sys.argv[2]
    if load_gguf(path, read=False)[0]["general.architecture"] in ("plamo3", "gemma4") and len(sys.argv) < 4:
        sys.exit("plamo3, gemma4: the prompt is text, so LLAMA_TOKENIZE is needed")  # before the model loads
    m = Model(path)
    long = (tokenize(sys.argv[3], path, LONG_USER) if len(sys.argv) > 3 and m.arch not in ("plamo3", "gemma4")
            else None)
    if m.arch == "plamo3":  # a translation into Japanese (PFML3, the GGUF's chat template)
        pfml = ("<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang"
                "<|plamo:val|>en<|plamo:msg|>" + "\n\n".join([LONG_USER] * 4) +
                "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>")
        prompt = tokenize(sys.argv[3], path, pfml)
    elif m.arch == "gemma3":  # <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        prompt = [2, 105, 2364, 107] + (long or GEMMA_USER) + [106, 107, 105, 4368, 107]
    elif m.arch == "gemma4":  # the chat template, no thinking: <bos><|turn>user\n...<turn|>\n<|turn>model\n
        prompt = tokenize(sys.argv[3], path, "<bos><|turn>user\n" + LONG_USER + "<turn|>\n<|turn>model\n")
    elif m.arch == "llama":  # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...
        prompt = [128000, 128006, 882, 128007, 271] + (long or LLAMA_USER) + [128009, 128006, 78191, 128007, 271]
    elif m.arch == "qwen35":  # thinking template: ends with "<think>\n"
        prompt = [248045, 846, 198] + (long or USER35) + [248046, 198, 248045, 74455, 198, 248068, 198]
    else:
        prompt = ((QWEN2_SYS if m.arch == "qwen2" else []) + [151644, 872, 198] + (long or USER) +
                  [151645, 198, 151644, 77091, 198])
    total, t0 = len(prompt) + N_GREEDY, time.perf_counter()

    def run(toks, pos):  # a run takes minutes: its progress on stderr
        out = m.run(toks, pos)
        print(f"{pos + len(toks)}/{total} tokens, {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
        return out

    emb, l0, fn, logits = run(prompt, 0)
    pos = len(prompt)
    top5 = np.argsort(-logits)[:5]
    greedy, nxt = [], int(np.argmax(logits))
    for _ in range(N_GREEDY):
        greedy.append(nxt)
        nxt = int(np.argmax(run([nxt], pos)[3]))
        pos += 1
    print("#pragma once")
    print(f"// Generated by tools/qwen_oracle.py from {path.split('/')[-1]} ({m.arch}); do not edit.")
    print("// numpy float32 forward on the SAME weights (F16, or blocks dequantized exactly); C++ must")
    how = (" (a PFML3 translation of LONG_USER four times over, tokenized by llama-tokenize)" if m.arch == "plamo3"
           else " (LONG_USER in the chat template, tokenized by llama-tokenize)" if m.arch == "gemma4"
           else " (LONG_USER, tokenized by llama-tokenize)" if long else "")
    print(f"// match to ~1e-4 + greedy exactly. Prompt: {len(prompt)} tokens{how}.")
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
