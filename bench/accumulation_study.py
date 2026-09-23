#!/usr/bin/env python3
"""FP32 emulation study: accuracy of the projection accumulation schemes on real data.

Inputs: real Q4_0 weights of the pinned model (blk.0.ffn_gate: K=5120; blk.8.ffn_down:
K=17408), dequantized exactly, and real activations (the block-09 FP64 reference
attn_post_norm-0 / ffn_swiglu-8, rounded to FP32). For every (weight row, token) the
exact FP64 dot product is the reference; each scheme's FP32 result is compared as a
relative error |y - ref| / sum|w x|.

Schemes (all in FP32; fma emulated as FP32(FP64(a)*FP64(b)+FP64(c))):
  gemm      prefill GEMM (gemm.comp): part = fma chain over each 32-k tile in k order,
            total += part per tile.
  gemm3     candidate: gemm with a third level (mid += part; total += mid every 8 tiles).
  matvec    decode matvec (matvec.comp Q4_0, 64 lanes): lane = 4 per 32-block, 4
            consecutive k per lane half (low nibbles -> a, high -> z), blocks ascending
            per lane; product rounded, then added (strict mul+add); v=a+z;
            (v.x+v.y)+(v.z+v.w); 64-lane halving tree.
  matvec_fma candidate: the same traversal with fused multiply-add.
Usage: accumulation_study.py MODEL.gguf --oracle-dir DIR [--rows N] [--output JSON]
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests/reference"))
from compare_model import load_reference  # noqa: E402

f32, f64 = np.float32, np.float64


def gguf_tensor(path, name):
    """Minimal GGUF v3 reader: returns (type, dims, raw bytes) of one tensor."""
    b = Path(path).open("rb")
    head = b.read(24)
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", head)
    assert magic == 0x46554747 and version == 3

    def rstr():
        (n,) = struct.unpack("<Q", b.read(8)); return b.read(n).decode()

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_value(t):
        if t == 8: rstr()
        elif t == 9:
            (et,) = struct.unpack("<I", b.read(4)); (n,) = struct.unpack("<Q", b.read(8))
            if et == 8:
                for _ in range(n): rstr()
            else: b.seek(n*sizes[et], 1)
        else: b.seek(sizes[t], 1)
    alignment = 32
    for _ in range(n_kv):
        key = rstr(); (t,) = struct.unpack("<I", b.read(4))
        if key == "general.alignment":
            (alignment,) = struct.unpack("<I", b.read(4))
        else: skip_value(t)
    infos = {}
    for _ in range(n_tensors):
        n = rstr(); (nd,) = struct.unpack("<I", b.read(4))
        dims = struct.unpack("<"+"Q"*nd, b.read(8*nd)); (t,) = struct.unpack("<I", b.read(4)); (off,) = struct.unpack("<Q", b.read(8))
        infos[n] = (t, dims, off)
    data = (b.tell() + alignment - 1)//alignment*alignment
    t, dims, off = infos[name]
    assert t == 2, "Q4_0 only"
    K, M = dims[0], dims[1]
    b.seek(data + off)
    raw = np.frombuffer(b.read(M*K//32*18), dtype=np.uint8).reshape(M, K//32, 18)
    return raw, K, M


def dequant_q4_0(raw, rows):
    blocks = raw[:rows]
    d = blocks[:, :, :2].copy().view(np.float16).astype(np.float64)[..., 0]
    q = blocks[:, :, 2:]
    lo = (q & 15).astype(np.float64) - 8; hi = (q >> 4).astype(np.float64) - 8
    w = np.concatenate([lo, hi], axis=2) * d[:, :, None]   # [rows, blocks, 32], k order in block
    return w.reshape(rows, -1)                             # exact in FP32 (d*(q-8))


def fma(a, b, c):
    return (a.astype(f64)*b.astype(f64) + c.astype(f64)).astype(f32)


def gemm(W, X, levels=2):
    M, K = W.shape; T = X.shape[0]
    total = np.zeros((M, T), f32); mid = np.zeros((M, T), f32)
    for t0 in range(0, K, 32):
        part = np.zeros((M, T), f32)
        for k in range(t0, t0+32):
            part = fma(W[:, k:k+1], X[None, :, k], part)
        if levels == 2:
            total = (total + part).astype(f32)
        else:
            mid = (mid + part).astype(f32)
            if (t0//32) % 8 == 7 or t0+32 >= K:
                total = (total + mid).astype(f32); mid = np.zeros((M, T), f32)
    return total


def matvec(W, X, fused):
    M, K = W.shape; T = X.shape[0]; nblocks = K//32
    lanes = np.zeros((64, 8, M, T), f32)  # per lane: a.xyzw, z.xyzw
    for lane in range(64):
        j = (lane % 4)*4
        for blk in range(lane//4, nblocks, 16):
            for c in range(4):
                for half, slot in ((0, c), (16, 4+c)):
                    k = blk*32 + half + j + c
                    w = W[:, k:k+1]; x = X[None, :, k]
                    if fused:
                        lanes[lane, slot] = fma(w, x, lanes[lane, slot])
                    else:
                        lanes[lane, slot] = (lanes[lane, slot] + (w.astype(f32)*x.astype(f32)).astype(f32)).astype(f32)
    v = (lanes[:, 0:4] + lanes[:, 4:8]).astype(f32)
    s = ((v[:, 0] + v[:, 1]).astype(f32) + (v[:, 2] + v[:, 3]).astype(f32)).astype(f32)  # [64, M, T]
    stride = 32
    while stride:
        s[:stride] = (s[:stride] + s[stride:2*stride]).astype(f32); stride //= 2
    return s[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    ref = load_reference(a.oracle_dir/"short-nothink", "fp64-reference")
    report = {}
    for tensor, act in (("blk.0.ffn_gate.weight", "attn_post_norm-0"), ("blk.8.ffn_down.weight", "ffn_swiglu-8")):
        raw, K, M = gguf_tensor(a.model, tensor)
        W = dequant_q4_0(raw, a.rows)
        assert np.all(W.astype(f32).astype(f64) == W), "dequantized weights must be exact in FP32"
        W32 = W.astype(f32)
        X = ref[act][:, :K].astype(f32)  # [tokens, K]
        exact = W.astype(f64) @ X.astype(f64).T
        scale = np.abs(W.astype(f64)) @ np.abs(X.astype(f64)).T
        res = {}
        for name, y in (("gemm", gemm(W32, X, 2)), ("gemm3", gemm(W32, X, 3)), ("matvec", matvec(W32, X, False)), ("matvec_fma", matvec(W32, X, True))):
            e = np.abs(y.astype(f64) - exact)/scale
            res[name] = dict(mean=float(e.mean()), median=float(np.median(e)), p90=float(np.quantile(e, 0.9)), max=float(e.max()))
        report[tensor] = dict(K=K, rows=a.rows, tokens=int(X.shape[0]), activations=act, errors=res)
        print(tensor, f"K={K}", json.dumps({k: {kk: f"{vv:.3e}" for kk, vv in v.items()} for k, v in res.items()}))
    if a.output: a.output.write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
