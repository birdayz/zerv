#!/usr/bin/env python3
"""Research (block 13g): accuracy of exact-integer WMMA prefill projections with
fixed-point activation limbs, compared with the shipped FP32 GEMM, on real data.

Inputs are as in bench/accumulation_study.py: real Q4_0 weights, dequantized exactly,
and real activations from the FP64 reference, rounded to FP32. For each
(weight row, token), the exact FP64 dot product of the FP32 inputs is the reference.
Errors are relative, |y - ref| / sum|w x|; `rel_to_y` is |y - ref| / |ref|.

Schemes:
  gemm       the shipped prefill GEMM order (FP32 fma chain per 32-k tile, then
             total += part).
  limbsL     the candidate (L = 2, 3, 4).
             - Each token row is scaled by a power of two s so that
               max|x|/s < 2^(8L-1).
             - x/s is rounded to the nearest integer X (8L-bit signed fixed point)
               and split into L signed 8-bit limbs.
             - s8 x s8 -> s32 WMMA products of the Q4_0 integers (q-8) with each limb
               are exact.
             - Per 32-block the exact integer total T = sum_i 2^(8i) sum_k (q-8) l_i
               is rounded once to FP32; then acc = fma(T, d, acc) in FP32; y = acc*s.
  fp16acc    the fp16 WMMA path for reference: x rounded to fp16, products exact,
             FP32 accumulation per 16-k (IEEE model; the real hardware is worse,
             docs/research/coopmat-prefill.md).
Usage: limb_study.py MODEL.gguf --oracle-dir DIR [--rows N] [--output JSON]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests/reference"))
from accumulation_study import gguf_tensor, dequant_q4_0, gemm, fma, f32, f64  # noqa: E402
from compare_model import load_reference  # noqa: E402


def limbs(raw, X, rows, L):
    """Emulates the exact-integer limb scheme for Q4_0 weights."""
    blocks = raw[:rows]
    d = blocks[:, :, :2].copy().view(np.float16).astype(np.float64)[..., 0]           # [M, nb]
    q = blocks[:, :, 2:]
    qi = np.concatenate([(q & 15).astype(np.int64) - 8, (q >> 4).astype(np.int64) - 8], axis=2)  # [M, nb, 32]
    T, K = X.shape
    bits = 8*L - 1
    mx = np.max(np.abs(X.astype(f64)), axis=1)                                           # [T]
    e = np.ceil(np.log2(np.where(mx > 0, mx, 1.0))) - bits
    s = np.exp2(e)                                                                       # power of two
    Xi = np.rint(X.astype(f64) / s[:, None]).astype(np.int64)                            # |Xi| <= 2^bits
    assert np.all(np.abs(Xi) <= 2**bits)
    Xb = Xi.reshape(T, K//32, 32)
    # Exact integer block totals (the limb decomposition is exact, so T is the same
    # as the product with the full fixed-point integer).
    tot = np.einsum("mbk,tbk->mtb", qi, Xb)                                              # [M, T, nb] int64
    acc = np.zeros((rows, T), f32)
    for b in range(K//32):
        acc = fma(tot[:, :, b].astype(f32), d[:, b:b+1].astype(f32), acc)
    return (acc.astype(f64) * s[None, :]).astype(f32), float(np.max(np.abs(X.astype(f64) - Xi*s[:, None]) / np.maximum(np.abs(X.astype(f64)), 1e-300)))


def fp16acc(W32, X):
    Xh = X.astype(np.float16).astype(f32)
    M, K = W32.shape; T = X.shape[0]
    total = np.zeros((M, T), f32)
    for k0 in range(0, K, 16):
        part = np.zeros((M, T), f32)
        for k in range(k0, k0+16):
            part = fma(W32[:, k:k+1], Xh[None, :, k], part)
        total = (total + part).astype(f32)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = {}
    for case in ("short-nothink",):  # the oracle stores these intermediate activations only for this case
        ref = load_reference(a.oracle_dir/case, "fp64-reference")
        for tensor, act in (("blk.0.ffn_gate.weight", "attn_post_norm-0"), ("blk.8.ffn_down.weight", "ffn_swiglu-8")):
            raw, K, M = gguf_tensor(a.model, tensor)
            W = dequant_q4_0(raw, a.rows)
            W32 = W.astype(f32)
            X = ref[act][:, :K].astype(f32)
            exact = W.astype(f64) @ X.astype(f64).T
            scale = np.abs(W.astype(f64)) @ np.abs(X.astype(f64)).T
            peak = np.max(np.abs(X.astype(f64)), axis=1); rms = np.sqrt(np.mean(X.astype(f64)**2, axis=1))
            res = {}
            outs = [("gemm", gemm(W32, X, 2))]
            for L in (2, 3, 4):
                y, _ = limbs(raw, X, a.rows, L); outs.append((f"limbs{L}", y))
            outs.append(("fp16acc", fp16acc(W32, X)))
            for name, y in outs:
                err = np.abs(y.astype(f64) - exact)
                e = err/scale; ry = err/np.maximum(np.abs(exact), 1e-300)
                res[name] = dict(mean=float(e.mean()), median=float(np.median(e)), p99=float(np.quantile(e, 0.99)), max=float(e.max()),
                                 rel_to_y_median=float(np.median(ry)), rel_to_y_p99=float(np.quantile(ry, 0.99)))
            key = f"{case}/{tensor}"
            report[key] = dict(K=K, rows=a.rows, tokens=int(X.shape[0]), activations=act,
                               peak_over_rms=dict(median=float(np.median(peak/rms)), max=float(np.max(peak/rms))), errors=res)
            print(key, f"K={K} T={X.shape[0]} peak/rms median {np.median(peak/rms):.1f} max {np.max(peak/rms):.1f}")
            for name, v in res.items(): print(f"   {name:8s}", " ".join(f"{k}={vv:.2e}" for k, vv in v.items()))
    if a.output: a.output.write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
