#!/usr/bin/env python3
"""Bitwise gate of a gemm_f16x pipeline binary against the SPIR-V kernel (research tool,
docs/bench/2026-09-24-gemm-f16x-isa.md).

  sweep.py SPV BIN OUT.jsonl

Runs pipeline_binary_lab race ... reps=0 (check only) over every model Q4_0 shape and a set of
edge cases: K = 64 (one stage), odd and even stage counts, row tails (n < plan rows, n = 1,
n = 257), misaligned Q4_0 blocks (abase = 2), an X offset, f16-subnormal scales, several seeds.
Every output of the plan (including rows >= n, which read row n - 1) must be bit-identical.
Exit status 1 if any configuration differs.
"""
import json, pathlib, subprocess, sys

LAB = pathlib.Path(__file__).resolve().parents[2] / "third_party/isa-lab/pipeline_binary_lab"
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (12288, 5120), (5120, 6144)]
EDGE = ["m=4096 k=64 rows=256", "m=4096 k=128 rows=256 n=17", "m=4096 k=192 rows=512 n=300",
        "m=4096 k=320 rows=512 n=1", "m=4096 k=512 rows=768 n=255 abase=2", "m=4096 k=256 rows=512 n=257",
        "m=4096 k=640 rows=256 sub=500 seed=11", "m=128 k=5120 rows=256 xbase=8", "m=4096 k=1088 rows=256 abase=2 xbase=24"]


def configs():
    for m, k in SHAPES:
        yield f"m={m} k={k} rows=512 seed=1"
        yield f"m={m} k={k} rows=768 n=700 abase=2 xbase=8 sub=20 seed=2"
    for m, k in SHAPES[:2]:
        yield f"m={m} k={k} rows=256 n=129 sub=50 seed=3"
    yield from EDGE


def main():
    spv, binary, out = sys.argv[1:4]
    bad = 0
    with open(out, "w") as f:
        for c in configs():
            r = subprocess.run([str(LAB), "race", spv, binary, "reps=0"] + c.split(), capture_output=True, text=True, check=True)
            d = json.loads(r.stdout.splitlines()[0])
            f.write(json.dumps(d) + "\n")
            bad += d["different"] != 0
            print(f"{c:55s} values {d['values']:>9d} different {d['different']}")
    print("PASS" if not bad else f"FAIL: {bad} configurations differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
