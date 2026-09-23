#!/usr/bin/env python3
"""FP32 emulation study: decode-attention accuracy vs the score dot-product accumulation.

Real tensors from the block-09 FP64 reference (short-nothink case, every attention layer):
Qcur (roped query), Kcur_roped, Vcur, rounded to FP32. For each layer and query position
t (keys 0..t), the pre-gate attention output of each KV head's 6 query heads is compared
with FP64 attention on the same FP32 inputs (normalized L2 per head).

Score variants (everything after the scores is the adopted 13b split-K arithmetic,
emulated as in bench/attention_order_study.py):
  seq256   decode kernel today: s = fma chain over d = 0..255, then * 1/16
  twolevel as the prefill GEMM: part = fma chain over each 32-d tile, total += part
FMA emulated as FP32(FP64(a)*FP64(b)+FP64(c)).
Usage: score_accumulation_study.py --oracle-dir DIR [--output JSON]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests/reference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_model import load_reference  # noqa: E402
from attention_order_study import tree, blocked  # noqa: E402

f32, f64 = np.float32, np.float64


def fma(a, b, c):
    """FP32 fused multiply-add a*b + c (single rounding)."""
    return (a.astype(f64)*b.astype(f64) + c.astype(f64)).astype(f32)


def scores(q, K, mode):
    """q [H,256], K [n,256] -> [n,H] scaled scores."""
    n, H = K.shape[0], q.shape[0]
    if mode == "seq256":
        s = np.zeros((n, H), f32)
        for d in range(256): s = fma(np.repeat(K[:, d:d+1], H, 1), np.repeat(q[None, :, d], n, 0), s)
    else:
        s = np.zeros((n, H), f32)
        for d0 in range(0, 256, 32):
            part = np.zeros((n, H), f32)
            for d in range(d0, d0+32): part = fma(np.repeat(K[:, d:d+1], H, 1), np.repeat(q[None, :, d], n, 0), part)
            s = (s + part).astype(f32)
    return (s*f32(1/16)).astype(f32)


def split_attention(s, V):
    """13b adopted orders (tree chunk sums, 4 interleaved accumulators, blocked-8 combine)."""
    n, H = s.shape
    m = s.max(0)
    e = np.exp((s - m).astype(f32)).astype(f32)
    parts, sums = [], []
    for c in range(0, n, 64):
        keys = range(c, min(n, c+64))
        lanes = [np.zeros((H, 256), f32) for _ in range(4)]
        for j in keys: lanes[j % 4] = fma(np.repeat(e[j][:, None], 256, 1), np.repeat(V[j][None, :], H, 0), lanes[j % 4])
        parts.append(((lanes[0] + lanes[1]).astype(f32) + (lanes[2] + lanes[3]).astype(f32)).astype(f32))
        sums.append(tree(e[c:min(n, c+64)]))
    return (blocked(parts)/blocked(sums)[:, None]).astype(f32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    ref = load_reference(a.oracle_dir/"short-nothink", "fp64-reference")
    errors = {"seq256": [], "twolevel": []}
    wins = 0; total = 0
    for layer in range(3, 64, 4):
        Q = ref[f"Qcur-{layer}"].astype(f32).reshape(-1, 24, 256)
        K = ref[f"Kcur_roped-{layer}"].astype(f32).reshape(-1, 4, 256)
        V = ref[f"Vcur-{layer}"].astype(f32).reshape(-1, 4, 256)
        for t in range(0, Q.shape[0], 3):
            for g in range(4):
                q = Q[t, 6*g:6*g+6]; k = K[:t+1, g]; v = V[:t+1, g]
                s64 = k.astype(f64) @ q.astype(f64).T / 16
                p64 = np.exp(s64 - s64.max(0)); p64 /= p64.sum(0)
                o64 = p64.T @ v.astype(f64)
                e = {}
                for mode in errors:
                    o = split_attention(scores(q, k, mode), v)
                    e[mode] = np.linalg.norm(o.astype(f64) - o64, axis=1)/np.linalg.norm(o64, axis=1)
                    errors[mode].extend(e[mode].tolist())
                wins += int(np.sum(e["twolevel"] < e["seq256"])); total += len(e["seq256"])
    summary = {k: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(np.max(v))) for k, v in errors.items()}
    summary["twolevel_better"] = f"{wins}/{total}"
    print(json.dumps(summary, indent=1))
    if a.output: a.output.write_text(json.dumps(summary, indent=1)+"\n")


if __name__ == "__main__":
    main()
