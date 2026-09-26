#!/usr/bin/env python3
"""FP32 emulation study: accuracy of decode-attention summation orders on real tensors.

Isolates the arithmetic of the attention kernel from upstream error propagation. Inputs
are the block-09 FP64 reference captures, rounded to FP32: keys/values
(Kcur_roped, Vcur) of the long-think case and queries (Qcur) of the short-nothink case,
same layer; the query/key pairing is from different contexts, so the score spread is
realistic but not the model's own. For each layer, query row and key count n, it compares
against FP64 attention on the same FP32 inputs:
  fused:  the pre-13b kernel (acc sequential over all keys; exp-sum via 256-thread tree)
  split:  13b (per-64-key chunk sequential acc and exp-sum; chunks combined in order)
  split_tree: 13b with a halving-tree chunk exp-sum (candidate refinement)
  split_best: split_tree + 4 interleaved P.V accumulators per chunk + chunks combined
              in blocks of 8 (candidate refinement)
Key counts up to 221 use one layer's real keys; longer contexts (1000, 3000) concatenate
the real keys/values of all 16 attention layers for the same KV head.
Scores and exponentials are shared (identical arithmetic in all variants); FMA emulated
as FP32(FP64(a) + FP64(b)*FP64(c)).

Usage: attention_order_study.py --oracle-dir DIR [--output JSON]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests/reference"))
from compare_model import load_reference  # noqa: E402

f32 = np.float32


def fma(acc, a, b):
    return (acc.astype(np.float64) + a.astype(np.float64)*b.astype(np.float64)).astype(f32)


def tree256(values):
    """wg_sum over 256 threads: thread-strided sequential local sums, then halving tree."""
    red = np.zeros((256,)+values.shape[1:], dtype=f32)
    for j in range(values.shape[0]): red[j % 256] = (red[j % 256] + values[j]).astype(f32)
    s = 128
    while s > 0:
        red[:s] = (red[:s] + red[s:2*s]).astype(f32)
        s //= 2
    return red[0]


def tree(values):
    """Halving tree over a zero-padded 64-slot array (the kernel's shared-memory pairing k, k+s)."""
    v = np.zeros((64,)+values.shape[1:], dtype=f32)
    v[:values.shape[0]] = values
    s = 32
    while s > 0:
        v[:s] = (v[:s] + v[s:2*s]).astype(f32)
        s //= 2
    return v[0]


def blocked(parts):
    """Sequential over blocks of 8 items; each block summed sequentially first."""
    total = np.zeros(parts[0].shape, dtype=f32)
    for b in range(0, len(parts), 8):
        g = np.zeros(parts[0].shape, dtype=f32)
        for x in parts[b:b+8]: g = (g + x).astype(f32)
        total = (total + g).astype(f32)
    return total


def attention(q, K, V):
    """q [heads,256] (6 query heads of one KV head), K/V [n,256]. Returns dict of outputs [heads,256]."""
    n = K.shape[0]
    s = np.zeros((n, q.shape[0]), dtype=f32)
    for d in range(256): s = fma(s, np.repeat(K[:, d:d+1], q.shape[0], 1), np.repeat(q[None, :, d], n, 0))
    s = (s*f32(1/16)).astype(f32)
    m = s.max(0)
    e = np.exp((s - m).astype(f32)).astype(f32)  # [n, heads]
    out = {}
    # fused (pre-13b)
    acc = np.zeros((q.shape[0], 256), dtype=f32)
    for j in range(n): acc = fma(acc, np.repeat(e[j][:, None], 256, 1), np.repeat(V[j][None, :], q.shape[0], 0))
    total = tree256(e)
    out["fused"] = (acc/total[:, None]).astype(f32)
    heads = q.shape[0]
    ev = lambda j: np.repeat(e[j][:, None], 256, 1)  # noqa: E731
    vv = lambda j: np.repeat(V[j][None, :], heads, 0)  # noqa: E731
    for variant in ("split", "split_tree", "split_best"):
        parts, sums = [], []
        for c in range(0, n, 64):
            keys = range(c, min(n, c+64))
            if variant == "split_best":
                lanes = [np.zeros((heads, 256), dtype=f32) for _ in range(4)]
                for j in keys: lanes[j % 4] = fma(lanes[j % 4], ev(j), vv(j))
                part = ((lanes[0] + lanes[1]).astype(f32) + (lanes[2] + lanes[3]).astype(f32)).astype(f32)
            else:
                part = np.zeros((heads, 256), dtype=f32)
                for j in keys: part = fma(part, ev(j), vv(j))
            if variant == "split":
                csum = np.zeros(heads, dtype=f32)
                for j in keys: csum = (csum + e[j]).astype(f32)
            else:
                csum = tree(e[c:min(n, c+64)])
            parts.append(part)
            sums.append(csum)
        if variant == "split_best":
            acc, total = blocked(parts), blocked(sums)
        else:
            acc = np.zeros((heads, 256), dtype=f32)
            total = np.zeros(heads, dtype=f32)
            for part, csum in zip(parts, sums):
                acc = (acc + part).astype(f32)
                total = (total + csum).astype(f32)
        out[variant] = (acc/total[:, None]).astype(f32)
    # FP64 truth on the same FP32 inputs
    s64 = K.astype(np.float64) @ q.astype(np.float64).T / 16
    p64 = np.exp(s64 - s64.max(0))
    p64 /= p64.sum(0)
    out["fp64"] = (p64.T @ V.astype(np.float64))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    long = load_reference(a.oracle_dir/"long-think", "fp64-reference")
    short = load_reference(a.oracle_dir/"short-nothink", "fp64-reference")
    variants = ("fused", "split", "split_tree", "split_best")
    layers = list(range(3, 64, 4))
    K_cat = np.concatenate([long[f"Kcur_roped-{L}"] for L in layers]).astype(f32).reshape(-1, 4, 256)
    V_cat = np.concatenate([long[f"Vcur-{L}"] for L in layers]).astype(f32).reshape(-1, 4, 256)
    summary = {}
    for regime, cases in (("real<=221", [(L, qi, n) for L in layers for qi, n in ((5, 64), (11, 100), (23, 150), (31, 200), (47, 221))]),
                          ("concat-long", [(L, qi, n) for L in layers[::3] for qi, n in ((13, 1000), (40, 3000))])):
        errors = {k: [] for k in variants}
        wins = {"split_vs_fused": 0, "fused_vs_split": 0, "best_vs_split": 0, "split_vs_best": 0}
        for layer, qi, n in cases:
            Q_all = short[f"Qcur-{layer}"].astype(f32).reshape(-1, 24, 256)
            for g in range(4):
                K = long[f"Kcur_roped-{layer}"].astype(f32).reshape(-1, 4, 256)[:n, g] if n <= 221 else K_cat[:n, g]
                V = long[f"Vcur-{layer}"].astype(f32).reshape(-1, 4, 256)[:n, g] if n <= 221 else V_cat[:n, g]
                o = attention(Q_all[qi, 6*g:6*g+6], K, V)
                for h in range(6):
                    ref = o["fp64"][h]
                    e = {k: float(np.linalg.norm(o[k][h].astype(np.float64)-ref)/np.linalg.norm(ref)) for k in variants}
                    for k in variants: errors[k].append(e[k])
                    wins["split_vs_fused"] += e["split"] < e["fused"]
                    wins["fused_vs_split"] += e["fused"] < e["split"]
                    wins["best_vs_split"] += e["split_best"] < e["split"]
                    wins["split_vs_best"] += e["split"] < e["split_best"]
        summary[regime] = {k: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(np.max(v))) for k, v in errors.items()}
        summary[regime]["pairwise"] = wins
    print(json.dumps(summary, indent=1))
    if a.output: a.output.write_text(json.dumps(summary, indent=1)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
