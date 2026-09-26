#!/usr/bin/env python3
"""Research (block 13g): characterize the numerics of the device's f16 x f16 -> f32
cooperative-matrix multiply-add (RDNA3 v_wmma_f32_16x16x16_f16 on RADV) against exact
rational arithmetic. Each experiment builds 16x16x16 cases, runs them through
zerv-coopmat-probe, and compares every output element with the exact value
round-to-nearest-even to f32 (the value an exact dot product would produce).
Writes a JSON report; raw inputs and outputs go under the work directory."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)


def run(probe, spv, work, name, A, B, C):
    """A: (n,16,16) f16 values [i][k]; B: (n,16,16) [k][j]; C: (n,16,16) f32. Returns D (n,16,16) f32."""
    n = A.shape[0]
    raw = struct.pack("<I", n) + A.astype(np.float16).tobytes() + np.ascontiguousarray(B.astype(np.float16).transpose(0, 2, 1)).tobytes() + C.astype(np.float32).tobytes()
    inp, out = work/f"{name}.in", work/f"{name}.out"
    inp.write_bytes(raw)
    subprocess.run([str(probe), str(spv), str(inp), str(out)], check=True)
    return np.frombuffer(out.read_bytes(), dtype=np.float32).reshape(n, 16, 16)


def exact(A, B, C):
    """Exact D as Fractions -> rounded to f32 (numpy float64 is exact enough? no: use Fractions)."""
    n = A.shape[0]
    Af = A.astype(np.float16).astype(np.float64); Bf = B.astype(np.float16).astype(np.float64); Cf = C.astype(np.float32).astype(np.float64)
    out = np.empty((n, 16, 16), dtype=np.float64)
    exact_frac = [[[None]*16 for _ in range(16)] for _ in range(n)]
    for c in range(n):
        for i in range(16):
            for j in range(16):
                s = Fraction(Cf[c, i, j])
                for k in range(16): s += Fraction(Af[c, i, k]) * Fraction(Bf[c, k, j])
                exact_frac[c][i][j] = s
                out[c, i, j] = float(s)  # correctly rounded to f64
    return out, exact_frac


def to_f32_rne(values64):
    return values64.astype(np.float32)  # f64 -> f32 is RNE; double rounding is harmless here only if f64 was exact


def ulp32(x):
    x = np.abs(x.astype(np.float32))
    return np.spacing(x)


def summarize(name, D, E64, note):
    E32 = E64.astype(np.float32)
    err = D.astype(np.float64) - E64
    exact_match = int(np.sum(D == E32))
    total = D.size
    ulps = np.abs(D.astype(np.float64) - E32.astype(np.float64)) / np.maximum(ulp32(E32).astype(np.float64), np.finfo(np.float32).smallest_subnormal)
    return dict(experiment=name, note=note, elements=total, equal_to_rounded_exact=exact_match,
                max_abs_err=float(np.max(np.abs(err))), max_ulps=float(np.max(ulps)), mean_abs_err=float(np.mean(np.abs(err))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, help="default: //bench:zerv-coopmat-probe, built")
    p.add_argument("--spv", type=Path, default=ROOT/"bench/coopmat/probe.spv")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.probe is None: a.probe = zerv_build.binary("zerv-coopmat-probe")
    a.work.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260923)
    report = dict(probe=str(a.probe), experiments=[])
    Z = lambda n: np.zeros((n, 16, 16))

    # E1: a single nonzero product per output (k = (i+j) % 16), C = 0.
    n = 8
    A, B = Z(n), Z(n)
    for c in range(n):
        for i in range(16):
            A[c, i, :] = 0
        for i in range(16):
            for j in range(16): pass
    A = rng.choice([-1, 1], size=(n, 16, 16)) * rng.uniform(0.5, 2, size=(n, 16, 16))
    B = Z(n); 
    for c in range(n):
        for j in range(16): B[c, (j*3 + c) % 16, j] = rng.uniform(-2, 2)
    C = Z(n)
    D = run(a.probe, a.spv, a.work, "e1", A, B, C); E, _ = exact(A, B, C)
    report["experiments"].append(summarize("single_product", D, E, "one nonzero term per output; product of two f16 needs <=22 bits"))

    # E2: a single product plus a nonzero accumulator C.
    C = rng.uniform(-4, 4, size=(n, 16, 16)).astype(np.float32)
    D = run(a.probe, a.spv, a.work, "e2", A, B, C); E, _ = exact(A, B, C)
    report["experiments"].append(summarize("single_product_plus_c", D, E, "one term + f32 C"))

    # E3: small integers (exact everywhere), C = 0 and C = 1.
    A = rng.integers(-8, 8, size=(n, 16, 16)).astype(np.float64)
    B = rng.integers(-8, 9, size=(n, 16, 16)).astype(np.float64) * 0.25
    for cval in (0.0, 1.0):
        C = np.full((n, 16, 16), cval)
        D = run(a.probe, a.spv, a.work, f"e3_c{int(cval)}", A, B, C); E, _ = exact(A, B, C)
        report["experiments"].append(summarize(f"small_integers_c{int(cval)}", D, E, "every partial sum exact in f32"))

    # E4: all ones (sum = 16 exactly), C = 0.
    A = np.ones((1, 16, 16)); B = np.ones((1, 16, 16)); C = Z(1)
    D = run(a.probe, a.spv, a.work, "e4", A, B, C); E, _ = exact(A, B, C)
    report["experiments"].append(summarize("all_ones", D, E, "sum of 16 ones") | dict(values=sorted(set(D.ravel().tolist()))))

    # E5: two terms: 1*1 + 1*x for x in small powers of two (alignment / guard bits).
    xs = [2.0**-e for e in range(0, 25)]
    n = len(xs)
    A = Z(n); B = Z(n); C = Z(n)
    for c, x in enumerate(xs):
        A[c, :, 0] = 1; B[c, 0, :] = 1
        A[c, :, 1] = 1; B[c, 1, :] = x if x >= 2.0**-24 else 0
    D = run(a.probe, a.spv, a.work, "e5", A, B, C); E, _ = exact(A, B, C)
    rows = [dict(x=x, got=float(D[c, 0, 0]), exact=float(E[c, 0, 0])) for c, x in enumerate(xs)]
    report["experiments"].append(summarize("one_plus_small", D, E, "1 + x (two products)") | dict(rows=rows))

    # E6: C = 1, single product of x (A=x, B=1): accumulator alignment.
    n = len(xs)
    A = Z(n); B = Z(n); C = np.ones((n, 16, 16))
    for c, x in enumerate(xs):
        A[c, :, 0] = x; B[c, 0, :] = 1
    D = run(a.probe, a.spv, a.work, "e6", A, B, C); E, _ = exact(A, B, C)
    rows = [dict(x=x, got=float(D[c, 0, 0]), exact=float(E[c, 0, 0])) for c, x in enumerate(xs)]
    report["experiments"].append(summarize("c_one_plus_small_product", D, E, "C=1 + x") | dict(rows=rows))

    # E7: random normal-ish data like activations x weights, C = 0: error vs exact.
    n = 64
    A = rng.integers(-8, 8, size=(n, 16, 16)).astype(np.float64)
    B = rng.normal(0, 1, size=(n, 16, 16)).astype(np.float16).astype(np.float64)
    C = Z(n)
    D = run(a.probe, a.spv, a.work, "e7", A, B, C); E, _ = exact(A, B, C)
    # Compare with sequential f32 fma in k order.
    seq = np.zeros((n, 16, 16), dtype=np.float32)
    for k in range(16):
        seq = (seq.astype(np.float64) + A[:, :, k:k+1] * B[:, k:k+1, :]).astype(np.float32)
    s = summarize("random_int_weights_normal_x", D, E, "q in [-8,7] x N(0,1) f16")
    s["sequential_fma_f32"] = summarize("seq", seq, E, "")
    report["experiments"].append(s)

    # E8: rounding mode. Random two- and many-term sums whose exact value needs more than
    # 24 bits; classify each output as RNE / toward -inf / toward +inf / toward 0.
    def classify(D, fracs):
        counts = dict(rne=0, down=0, up=0, zero=0, exact=0, other=0, total=0)
        for c in range(D.shape[0]):
            for i in range(16):
                for j in range(16):
                    e = fracs[c][i][j]; g = Fraction(float(D[c, i, j])); counts["total"] += 1
                    f = np.float32(float(e))  # may double-round; recompute neighbours exactly
                    lo = np.float32(f); 
                    if Fraction(float(lo)) > e: lo = np.nextafter(lo, np.float32(-np.inf))
                    hi = np.nextafter(lo, np.float32(np.inf)) if Fraction(float(lo)) != e else lo
                    if Fraction(float(lo)) == e:
                        counts["exact" if g == e else "other"] += 1; continue
                    dn, up = Fraction(float(lo)), Fraction(float(hi))
                    nearest = dn if (e - dn) < (up - e) or ((e - dn) == (up - e) and int(np.float32(lo).view(np.uint32)) % 2 == 0) else up
                    tz = dn if e > 0 else up
                    if g == nearest: counts["rne"] += 1
                    elif g == dn: counts["down"] += 1
                    elif g == up: counts["up"] += 1
                    else: counts["other"] += 1
                    if g == tz and g != nearest: counts["zero"] += 1
        return counts
    n = 32
    A = rng.uniform(-2, 2, size=(n, 16, 16)); B = rng.uniform(-2, 2, size=(n, 16, 16))
    mask = np.zeros((n, 16, 16)); mask[:, :, :2] = 1
    A = A*mask
    C = Z(n)
    D = run(a.probe, a.spv, a.work, "e8a", A, B, C); E, F = exact(A, B, C)
    report["experiments"].append(summarize("two_terms_rounding", D, E, "a1*b1 + a2*b2, random") | dict(rounding=classify(D, F)))
    A = rng.uniform(-2, 2, size=(n, 16, 16))
    D = run(a.probe, a.spv, a.work, "e8b", A, B, C); E, F = exact(A, B, C)
    report["experiments"].append(summarize("sixteen_terms_rounding", D, E, "16 random terms") | dict(rounding=classify(D, F)))
    Cr = rng.uniform(-8, 8, size=(n, 16, 16)).astype(np.float32)
    D = run(a.probe, a.spv, a.work, "e8c", A, B, Cr); E, F = exact(A, B, Cr)
    report["experiments"].append(summarize("sixteen_terms_plus_c_rounding", D, E, "16 random terms + C") | dict(rounding=classify(D, F)))
    # Single products, split by sign.
    A = rng.uniform(0.5, 2, size=(n, 16, 16)) * mask[:, :, :] ; A[:, :, 1] = 0
    Bp = rng.uniform(0.5, 2, size=(n, 16, 16))
    D = run(a.probe, a.spv, a.work, "e9p", A, Bp, C); E, F = exact(A, Bp, C)
    report["experiments"].append(summarize("single_product_positive", D, E, "a*b > 0, exactly representable") | dict(rounding=classify(D, F)))
    D = run(a.probe, a.spv, a.work, "e9n", A, -Bp, C); E, F = exact(A, -Bp, C)
    report["experiments"].append(summarize("single_product_negative", D, E, "a*b < 0, exactly representable") | dict(rounding=classify(D, F)))

    a.output.write_text(json.dumps(report, indent=1) + "\n")
    for e in report["experiments"]:
        print(e["experiment"], {k: v for k, v in e.items() if k in ("elements", "equal_to_rounded_exact", "max_abs_err", "max_ulps", "mean_abs_err")})
        if "rows" in e:
            for r in e["rows"]: print("   ", r)
        if "values" in e: print("   values", e["values"])
        if "rounding" in e: print("   rounding", e["rounding"])
        if "sequential_fma_f32" in e: print("   seq fma:", {k: v for k, v in e["sequential_fma_f32"].items() if k in ("equal_to_rounded_exact", "max_abs_err", "max_ulps", "mean_abs_err")})


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
