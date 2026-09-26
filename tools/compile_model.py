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
           "embed_b": 10, "qk_b": 11, "gate": 13, "conv_b": 14, "delta_b": 15,
           "attn_scores": 16, "attn_pv": 17, "attn_combine": 18, "gnorm_b": 19,
           "rowcopy": 20, "argmax_a": 21, "argmax_b": 22, "copy2d": 23, "attn_gmax": 24, "attn_cblock": 25}
# gemm variants: name -> (FORMAT, BLOCK_BYTES, PAYLOAD_OFFSET, A_MCONTIG)
GEMM = {"gemm_f32_k": (0, 4, 0, 0), "gemm_f32_m": (0, 4, 0, 1), "gemm_q4_0": (2, 18, 2, 0), "gemm_q4_1": (3, 20, 4, 0),
        "gemm_q5_k": (13, 176, 0, 0), "gemm_q6_k": (14, 210, 0, 0), "gemm_q8_0": (8, 34, 2, 0)}
# Wide-tile (256x64, XW=16) modules of the quantized formats (block 13h).
GEMM_WIDE = {name+"_w": GEMM[name] for name in ("gemm_q4_0", "gemm_q4_1", "gemm_q5_k", "gemm_q6_k", "gemm_q8_0")}
# f16 WMMA prefill GEMM (block 14, explicit --prefill-precision f16): name -> (FORMAT, BLOCK_BYTES)
GEMM_F16 = {"gemm_f16_q4_0": (2, 18), "gemm_f16_q4_1": (3, 20), "gemm_f16_q5_k": (13, 176)}
# wave32 f16 GEMM on f16 X (block 16b): name -> (FORMAT, BLOCK_BYTES)
GEMM_F16X = {"gemm_f16x_q4_0": (2, 18)}
# Batched decode v2 (block 18e, gemm_f16d.comp): wave32 streaming tiles, gemm_f16n arithmetic.
GEMM_F16D = {"gemm_f16d_q4_0": (2, 18)}
# Batched decode tile (block 18e, -DSMALLN=1): 128 x 16, the same per-element arithmetic.
GEMM_F16N = {"gemm_f16n_q4_0": (2, 18), "gemm_f16n_q4_1": (3, 20), "gemm_f16n_q5_k": (13, 176)}
# Short-prompt tile (block 18c.2, -DSMALLM=1): 32 x 128, the same per-element arithmetic.
GEMM_F16M = {"gemm_f16m_q4_0": (2, 18), "gemm_f16m_q4_1": (3, 20), "gemm_f16m_q5_k": (13, 176)}
# Fused attention tile (flash.comp RW, GROUPS; runtime.zig flash_rows / flash_groups).
FLASH_DEFINES = ["-DRW=8", "-DGROUPS=6", "-DOU=4"]
# f16-mode producers writing an f16 copy of their output (block 16b): name -> KERNEL
KERNELS_H = {"norm_h": 2, "swiglu_h": 7, "gate_h": 13}
# f16 KV cache variants of the KV kernels (block 17c, -DKV16): name -> KERNEL
KERNELS_KV16 = {"qkprep_kv16": 3, "qk_b_kv16": 11, "attn_scores_kv16": 16, "attn_pv_kv16": 17}
# DeltaNet kernels: the default modules are built with -DSTATE_OUT (separate store offset,
# no address spill); the `_legacy` modules without it (Options.delta_state_out = false).
STATE_OUT = {"delta", "delta_b"}
KERNELS_LEGACY = {"delta_legacy": 6, "delta_b_legacy": 15}
# Packed multi-sequence prefill (docs/specs/concurrent.md, "18d.1 design"; parallel mode):
# name -> (KERNEL, extra defines).
KERNELS_PACKED = {"qk_p": (11, []), "qk_p_kv16": (11, ["-DKV16"]), "conv_p": (14, []), "delta_p": (15, ["-DSTATE_OUT=1"])}
FLASH_PACKED = {"attn_flash_p": [], "attn_flash_p_kv16": ["-DKV16"]}


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
    gemm_f16d = ROOT/"src/model/gemm_f16d.comp"
    flash = ROOT/"src/model/flash.comp"
    manifest = dict(source_sha256=sha(source), gemm_source_sha256=sha(gemm), gemm_f16_source_sha256=sha(gemm_f16),
                    gemm_f16x_source_sha256=sha(gemm_f16x), gemm_f16d_source_sha256=sha(gemm_f16d), flash_source_sha256=sha(flash), tools=PINS, modules={})
    # Fused causal prefill attention (block 16a).
    for name, kv16 in (("attn_flash", []), ("attn_flash_kv16", ["-DKV16"])):
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute"] + FLASH_DEFINES + kv16 + [str(flash), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
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
    for name, (fmt, width) in GEMM_F16M.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", "-DSMALLM=1",
                        str(gemm_f16), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (fmt, width) in GEMM_F16N.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", "-DSMALLN=1",
                        str(gemm_f16), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (fmt, width) in GEMM_F16D.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        str(gemm_f16d), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (fmt, width) in GEMM_F16X.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        str(gemm_f16x), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, kernel in list(KERNELS.items()) + list(KERNELS_H.items()) + list(KERNELS_KV16.items()) + list(KERNELS_LEGACY.items()):
        output = a.output_dir/(name+".spv")
        f16out = ["-DF16OUT"] if name in KERNELS_H else ["-DKV16"] if name in KERNELS_KV16 else ["-DSTATE_OUT=1"] if name in STATE_OUT else []
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DKERNEL={kernel}"] + f16out + [str(source), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, (kernel, extra) in KERNELS_PACKED.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DKERNEL={kernel}", "-DPACKED=1"] + extra + [str(source), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    for name, extra in FLASH_PACKED.items():
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", "-DPACKED=1"] + FLASH_DEFINES + extra + [str(flash), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
