#!/usr/bin/env python3
"""Independent NumPy FP64 Qwen3.8 (qwen35) forward pass for teacher-forced token sequences.

Written from the pinned HF modeling semantics plus the pinned GGUF converter mapping
(docs/research/qwen35-execution.md). It does not import or execute llama.cpp/ggml or
PyTorch. Weights are decoded here with vectorized decoders written from the quant
specs and cross-checked against the earlier independent scalar decoders. Every
decoded weight is exact in FP64; all arithmetic is FP64.
Development/test oracle only; never part of zerv.
"""
import hashlib
import json
import math
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path

# One BLAS thread per process (set before numpy loads OpenBLAS): the FP64 reference runs its
# row work in a pool of worker processes (qwen35_reference.Projector); OpenBLAS would otherwise
# start one thread per CPU in each of them (22 workers x 24 threads measured on 2026-09-26).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT/"docs/bench/data/2026-09-22-gguf-qwen38/container.json"
METADATA = ROOT/"docs/research/2026-09-22/qwen38-gguf-metadata.json"
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
TYPES = {0: (1, 4), 2: (32, 18), 3: (32, 20), 13: (256, 176), 14: (256, 210)}


def f16(raw, at):
    return np.ascontiguousarray(raw[:, at:at+2]).view("<f2")[:, 0].astype(np.float64)


def dequant(kind, raw):
    """raw: uint8 [n_blocks*block_bytes] -> float64 values in element order."""
    elements, width = TYPES[kind]
    if kind == 0:
        return np.frombuffer(raw.tobytes(), "<f4").astype(np.float64)
    b = raw.reshape(-1, width)
    if kind in (2, 3):
        d = f16(b, 0)
        qs = b[:, 4 if kind == 3 else 2:].astype(np.int64)
        q = np.concatenate([qs & 15, qs >> 4], axis=1)
        if kind == 2:
            return ((q - 8) * d[:, None]).ravel()
        return (q * d[:, None] + f16(b, 2)[:, None]).ravel()
    if kind == 13:
        d, dmin = f16(b, 0), f16(b, 2)
        s = b[:, 4:16].astype(np.int64)
        qh, qs = b[:, 16:48].astype(np.int64), b[:, 48:176].astype(np.int64)
        out = np.empty((b.shape[0], 256))
        for g in range(8):
            if g < 4:
                sc, mn = s[:, g] & 63, s[:, g+4] & 63
            else:
                sc = (s[:, g+4] & 15) | ((s[:, g-4] >> 6) << 4)
                mn = (s[:, g+4] >> 4) | ((s[:, g] >> 6) << 4)
            low = (qs[:, (g//2)*32:(g//2)*32+32] >> (4*(g % 2))) & 15
            q = low | (((qh >> g) & 1) << 4)
            out[:, g*32:g*32+32] = (d*sc)[:, None]*q - (dmin*mn)[:, None]
        return out.ravel()
    if kind == 14:
        ql, qh = b[:, :128].astype(np.int64), b[:, 128:192].astype(np.int64)
        sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.int64)
        d = f16(b, 208)
        out = np.empty((b.shape[0], 256))
        for h in range(2):
            for g in range(4):
                low = (ql[:, h*64+(g % 2)*32:h*64+(g % 2)*32+32] >> (4*(g//2))) & 15
                high = (qh[:, h*32:h*32+32] >> (2*g)) & 3
                q = (low | (high << 4)) - 32
                scale = np.repeat(sc[:, h*8+g*2:h*8+g*2+2], 16, axis=1)
                out[:, h*128+g*32:h*128+g*32+32] = d[:, None]*scale*q
        return out.ravel()
    raise ValueError("unsupported type")


class Weights:
    def __init__(self, model, verify_hash=True):
        if verify_hash:
            h = hashlib.sha256()
            with open(model, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
            if h.hexdigest() != MODEL_SHA: raise ValueError("model hash mismatch")
        inv = json.loads(INVENTORY.read_text())
        self.data_offset = inv["data_offset"]
        self.tensors = {t["name"]: t for t in inv["tensors"]}
        self.meta = json.loads(METADATA.read_text())
        self.mm = np.memmap(model, dtype=np.uint8, mode="r")
        self.path = str(model)

    def raw(self, name, first_row=0, rows=None):
        t = self.tensors[name]
        k = t["dims"][0]
        total_rows = t["size"] // (k // TYPES[t["type"]][0] * TYPES[t["type"]][1])
        rows = total_rows - first_row if rows is None else rows
        row_bytes = t["size"] // total_rows
        start = self.data_offset + t["offset"] + first_row*row_bytes
        return t["type"], k, np.asarray(self.mm[start:start+rows*row_bytes])

    def matrix(self, name, first_row=0, rows=None):
        kind, k, raw = self.raw(name, first_row, rows)
        return dequant(kind, raw).reshape(-1, k)

    def vector(self, name):
        return self.matrix(name).ravel()


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * w


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


_WORKER = {}


def _worker_init(model, shm_name, capacity):
    _WORKER["W"] = Weights(model, verify_hash=False)
    _WORKER["shm"] = shared_memory.SharedMemory(name=shm_name)
    _WORKER["x"] = np.ndarray((capacity,), dtype=np.float64, buffer=_WORKER["shm"].buf)


def _worker_task(task):
    name, first, rows, t, k = task
    x = _WORKER["x"][:t*k].reshape(t, k)
    return x @ _WORKER["W"].matrix(name, first, rows).T


class Projector:
    """Exact FP64 x @ W.T with W decoded and multiplied in row chunks by worker
    processes (the system BLAS is single-threaded reference BLAS)."""

    def __init__(self, model, workers=None, capacity=512*17408):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 2)
        self.capacity = capacity
        self.shm = shared_memory.SharedMemory(create=True, size=capacity*8)
        self.x = np.ndarray((capacity,), dtype=np.float64, buffer=self.shm.buf)
        self.pool = multiprocessing.get_context("fork").Pool(self.workers, _worker_init, (str(model), self.shm.name, capacity))

    def close(self):
        self.pool.close(); self.pool.join()
        self.shm.close(); self.shm.unlink()

    def __call__(self, W, name, x):
        t, k = x.shape
        if t*k > self.capacity: raise ValueError("projection input too large")
        rows = W.tensors[name]["dims"][1]
        self.x[:t*k] = x.ravel()
        step = max(1, math.ceil(rows/(self.workers*4)))
        parts = self.pool.map(_worker_task, [(name, f, min(step, rows-f), t, k) for f in range(0, rows, step)])
        return np.concatenate(parts, axis=1)


class LayerWeights:
    def __init__(self, W, il, projector):
        self.W, self.prefix, self.projector = W, f"blk.{il}.", projector

    def vec(self, name):
        return self.W.vector(self.prefix+name)

    def mat(self, name):
        return self.W.matrix(self.prefix+name)

    def mm(self, x, name):
        return self.projector(self.W, self.prefix+name, x)


def forward_many(weights, sequences, captures, eps=None, projector=None):
    """sequences: list of token lists; captures: list of name-base sets (None = all).
    One weight decode per layer serves every sequence. Returns per sequence
    (captured {name: [T, n]}, logits [T, vocab], final state dict)."""
    W = weights
    eps = float(np.float32(W.meta["qwen35.attention.layer_norm_rms_epsilon"])) if eps is None else eps
    base = float(W.meta["qwen35.rope.freq_base"])
    runs = []
    for tokens, capture in zip(sequences, captures):
        T = len(tokens)
        ang = np.arange(T, dtype=np.float64)[:, None]*(base ** (-np.arange(32, dtype=np.float64)*2/64))[None, :]
        runs.append(dict(T=T, capture=capture, out={}, state={}, cos=np.cos(ang), sin=np.sin(ang),
                         x=np.stack([W.matrix("token_embd.weight", t, 1)[0] for t in tokens])))
    for run in runs:
        keep(run, "model.input_embed", run["x"])
    own = projector is None
    # The widest projection input is T x 17408 (ffn_down); size the shared buffer for the
    # longest sequence (at least the historical 512 rows).
    projector = projector or Projector(W.path, capacity=max(512, max(r["T"] for r in runs))*17408)
    try:
        for il in range(64):
            L = LayerWeights(W, il, projector)
            for run in runs:
                run["x"] = layer(L, il, run, eps)
        results = _finish(W, runs, eps, projector)
    finally:
        if own: projector.close()
    return results


def _finish(W, runs, eps, projector):
    hw = W.vector("output_norm.weight")
    for run in runs:
        run["hn"] = rmsnorm(run["x"], hw, eps)
        keep(run, "result_norm", run["hn"])
        run["logits"] = projector(W, "output.weight", run["hn"])
    return [(run["out"], run["logits"], run["state"]) for run in runs]


def forward(weights, tokens, capture=None, eps=None):
    out, logits, _ = forward_many(weights, [tokens], [capture], eps)[0]
    return out, logits


def keep(run, name, value):
    if run["capture"] is None or name.rsplit("-", 1)[0] in run["capture"]:
        run["out"][name] = np.array(value, dtype=np.float64, copy=True).reshape(run["T"], -1)


def rope(v, cos, sin):  # v: [T, heads, 256]; NEOX pairs (i, i+32) over the first 64 dims
    r = v.copy()
    a, b = v[:, :, :32], v[:, :, 32:64]
    r[:, :, :32] = a*cos[:, None, :] - b*sin[:, None, :]
    r[:, :, 32:64] = a*sin[:, None, :] + b*cos[:, None, :]
    return r


def layer(L, il, run, eps):
    T, x = run["T"], run["x"]
    k_ = lambda name, value: keep(run, f"{name}-{il}", value)
    h = rmsnorm(x, L.vec("attn_norm.weight"), eps)
    k_("attn_norm", h)
    if il % 4 == 3:
        qf = L.mm(h, "attn_q.weight")
        k = L.mm(h, "attn_k.weight")
        v = L.mm(h, "attn_v.weight")
        k_("Qcur_full", qf); k_("Kcur", k); k_("Vcur", v)
        qf = qf.reshape(T, 24, 512)
        q, gate = qf[:, :, :256], qf[:, :, 256:].reshape(T, 6144)
        q = rmsnorm(q, L.vec("attn_q_norm.weight"), eps)
        k = rmsnorm(k.reshape(T, 4, 256), L.vec("attn_k_norm.weight"), eps)
        k_("Qcur_normed", q); k_("Kcur_normed", k)
        q, k = rope(q, run["cos"], run["sin"]), rope(k, run["cos"], run["sin"])
        k_("Qcur", q); k_("Kcur_roped", k)
        v = v.reshape(T, 4, 256)
        run["state"][f"k-{il}"], run["state"][f"v-{il}"] = k, v
        att = np.empty((T, 24, 256))
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for hq in range(24):
            kv = hq // 6
            s = (q[:, hq, :] @ k[:, kv, :].T) / 16.0 + mask
            s = np.exp(s - s.max(axis=1, keepdims=True))
            s /= s.sum(axis=1, keepdims=True)
            att[:, hq, :] = s @ v[:, kv, :]
        att = att.reshape(T, 6144)
        k_("attn_pregate", att)
        g = sigmoid(gate)
        k_("gate_sigmoid", g)
        att = att*g
        k_("attn_gated", att)
        a = L.mm(att, "attn_output.weight")
        k_("attn_output", a)
    else:
        mixed = L.mm(h, "attn_qkv.weight")
        z = L.mm(h, "attn_gate.weight")
        k_("linear_attn_qkv_mixed", mixed); k_("z", z)
        beta_raw = L.mm(h, "ssm_beta.weight")
        alpha = L.mm(h, "ssm_alpha.weight")
        k_("beta", beta_raw); k_("alpha", alpha)
        beta = sigmoid(beta_raw)
        sp = np.logaddexp(0.0, alpha + L.vec("ssm_dt.bias"))
        gdecay = L.vec("ssm_a")*sp
        k_("beta_sigmoid", beta); k_("a_softplus", sp); k_("gate", gdecay)
        wconv = L.mat("ssm_conv1d.weight")  # [10240, 4]; tap 3 = current token
        padded = np.concatenate([np.zeros((3, 10240)), mixed])
        run["state"][f"conv-{il}"] = padded[-3:].T.copy()  # [10240, 3] oldest..newest
        conv = sum(padded[j:j+T]*wconv[:, j] for j in range(4))
        k_("conv_output_raw", conv)
        conv = conv*sigmoid(conv)
        k_("conv_output_silu", conv)
        q = conv[:, :2048].reshape(T, 16, 128)
        k = conv[:, 2048:4096].reshape(T, 16, 128)
        v = conv[:, 4096:].reshape(T, 48, 128)
        q = q/np.sqrt(np.sum(q*q, axis=-1, keepdims=True)+1e-6)
        k = k/np.sqrt(np.sum(k*k, axis=-1, keepdims=True)+1e-6)
        k_("q_conv_predelta", q); k_("k_conv_predelta", k)
        q = q/np.sqrt(128.0)
        S = np.zeros((48, 128, 128))  # S[hv, j(value), i(key)]
        o = np.empty((T, 48, 128))
        kh = np.arange(48) % 16  # tiled V-head order after conversion
        for t in range(T):
            S *= np.exp(gdecay[t])[:, None, None]
            kt, qt = k[t, kh], q[t, kh]
            d = (v[t] - np.einsum("hji,hi->hj", S, kt))*beta[t][:, None]
            S += d[:, :, None]*kt[:, None, :]
            o[t] = np.einsum("hji,hi->hj", S, qt)
        run["state"][f"ssm-{il}"] = S
        k_("attn_output", o)
        zz = z.reshape(T, 48, 128)
        fo = (rmsnorm(o, L.vec("ssm_norm.weight"), eps)*(zz*sigmoid(zz))).reshape(T, 6144)
        k_("final_output", fo)
        a = L.mm(fo, "ssm_out.weight")
        k_("linear_attn_out", a)
    r = x + a
    k_("attn_residual", r)
    h2 = rmsnorm(r, L.vec("post_attention_norm.weight"), eps)
    k_("attn_post_norm", h2)
    fg = L.mm(h2, "ffn_gate.weight")
    fu = L.mm(h2, "ffn_up.weight")
    k_("ffn_gate", fg); k_("ffn_up", fu)
    sw = fg*sigmoid(fg)*fu
    k_("ffn_swiglu", sw)
    f = L.mm(sw, "ffn_down.weight")
    k_("ffn_out", f)
    out = r + f
    k_("l_out", out)
    return out
