#!/usr/bin/env python3
"""Compile GEMM lab variants with the pinned glslc/spirv-val flags of tools/compile_model.py.
Usage: build.py SOURCE.comp OUTDIR NAME=DEF1,DEF2 [NAME=...]  (FORMAT/BLOCK_BYTES from --format)"""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[2]/"tools"))
import zerv_build  # noqa: E402  (the source-built shader tools)

FORMATS = {"q4_0": (2, 18), "q4_1": (3, 20), "q5_k": (13, 176)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("variants", nargs="+", help="name=DEF,DEF (empty after = for the base)")
    p.add_argument("--format", default="q4_0", choices=FORMATS)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    fmt, width = FORMATS[a.format]
    glslc, spirv_val = zerv_build.shader_tools()
    for v in a.variants:
        name, _, defs = v.partition("=")
        out = a.outdir/f"{name}-{a.format}.spv"
        cmd = [glslc, "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}"]
        cmd += [f"-D{d}" for d in defs.split(",") if d]
        subprocess.run(cmd + [str(a.source), "-o", str(out)], check=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.1", str(out)], check=True)
        print(out)


if __name__ == "__main__":
    main()
