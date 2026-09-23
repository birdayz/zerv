#!/usr/bin/env python3
"""Rebuild owned Vulkan1.1 model-operator modules with pinned offline tools; never runtime code."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from compile_matvec import PINS, sha

ROOT = Path(__file__).resolve().parents[1]
KERNELS = {"embed": 1, "norm": 2, "qkprep": 3, "conv": 5, "delta": 6, "swiglu": 7, "zero": 8, "reduce": 9,
           "embed_b": 10, "qk_b": 11, "softmax": 12, "gate": 13, "conv_b": 14, "delta_b": 15,
           "attn_scores": 16, "attn_pv": 17, "attn_combine": 18, "gnorm_b": 19}
# gemm variants: name -> (FORMAT, BLOCK_BYTES, PAYLOAD_OFFSET, A_MCONTIG)
GEMM = {"gemm_f32_k": (0, 4, 0, 0), "gemm_f32_m": (0, 4, 0, 1), "gemm_q4_0": (2, 18, 2, 0), "gemm_q4_1": (3, 20, 4, 0),
        "gemm_q5_k": (13, 176, 0, 0), "gemm_q6_k": (14, 210, 0, 0)}
# Wide-tile (256x64, XW=16) modules of the quantized formats (block 13h).
GEMM_WIDE = {name+"_w": GEMM[name] for name in ("gemm_q4_0", "gemm_q4_1", "gemm_q5_k", "gemm_q6_k")}
# f16 WMMA prefill GEMM (block 14, explicit --prefill-precision f16): name -> (FORMAT, BLOCK_BYTES)
GEMM_F16 = {"gemm_f16_q4_0": (2, 18), "gemm_f16_q4_1": (3, 20), "gemm_f16_q5_k": (13, 176)}
# wave32 f16 GEMM on f16 X (block 16b): name -> (FORMAT, BLOCK_BYTES)
GEMM_F16X = {"gemm_f16x_q4_0": (2, 18)}
# f16-mode producers writing an f16 copy of their output (block 16b): name -> KERNEL
KERNELS_H = {"norm_h": 2, "swiglu_h": 7, "gate_h": 13}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): p.error("fresh directory required")
    for tool, digest in PINS.items():
        path = shutil.which(tool)
        if not path or sha(path) != digest: raise ValueError("tool pin mismatch: "+tool)
    a.output_dir.mkdir(parents=True)
    source = ROOT/"src/model/model.comp"
    gemm = ROOT/"src/model/gemm.comp"
    gemm_f16 = ROOT/"src/model/gemm_f16.comp"
    gemm_f16x = ROOT/"src/model/gemm_f16x.comp"
    manifest = dict(source_sha256=sha(source), gemm_source_sha256=sha(gemm), gemm_f16_source_sha256=sha(gemm_f16),
                    gemm_f16x_source_sha256=sha(gemm_f16x), tools=PINS, modules={})
    for name, (fmt, width, payload, mcontig) in list(GEMM.items()) + list(GEMM_WIDE.items()):
        output = a.output_dir/(name+".spv")
        wide = ["-DXW=16"] if name in GEMM_WIDE else []
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        f"-DPAYLOAD_OFFSET={payload}", f"-DA_MCONTIG={mcontig}"] + wide + [str(gemm), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (fmt, width) in GEMM_F16.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        str(gemm_f16), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (fmt, width) in GEMM_F16X.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        str(gemm_f16x), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, kernel in list(KERNELS.items()) + list(KERNELS_H.items()):
        output = a.output_dir/(name+".spv")
        f16out = ["-DF16OUT"] if name in KERNELS_H else []
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DKERNEL={kernel}"] + f16out + [str(source), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
