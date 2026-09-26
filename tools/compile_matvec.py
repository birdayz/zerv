#!/usr/bin/env python3
"""Rebuild owned Vulkan1.1 matvec modules with pinned offline tools; never runtime code.

The tools are glslc and spirv-val built from source by Bazel (MODULE.bazel):
`bazel run //tools:compile_matvec -- --output-dir DIR` (the build itself runs this script as
//src/matvec:generated_shaders)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel runs it from its runfiles
MAX_ROWS = 5  # matvec.max_rows
# Per exact row count: (GROUP weight rows per workgroup = matvec.rows_groups, CB blocks per
# chunk); re-tuned for FMA accumulation in docs/bench/2026-09-24-fma-matvec.md (interleaved
# in-model race; first tuned in docs/bench/2026-09-24-spec-verify.md).
ROWS_CONFIG = {1: (1, 4), 2: (3, 2), 3: (3, 3), 4: (3, 2), 5: (3, 3)}
# The separate-accumulation modules keep the table tuned for them (17b.1).
ROWS_CONFIG_SEPARATE = {1: (1, 4), 2: (2, 2), 3: (2, 2), 4: (3, 2), 5: (2, 4)}
# Fused verify FFN input (block 17c, matvec_rows.comp with SWIGLU): per exact row count,
# (GROUP gate rows per workgroup, i.e. 2 GROUP weight rows with up = matvec.swigluRowsGroups,
# CB blocks per chunk), or None: no fused module for that count (the verify pass records
# the separate path). Tuned by the in-model race tools/race_swiglu_rows.py
# (docs/bench/2026-09-24-verify-fusion.md); the separate-accumulation modules use the same
# table (correctness-tested, speed not tuned). Count 5: no group beat the separate path.
# K-quant formats (Q5_K, Q6_K) only up to count 2: at counts 3-4 their 4-weight-row modules
# spill VGPRs, and ACO's VGPR-to-LDS spilling miscompiled the 5-row case
# (docs/bench/2026-09-24-verify-fusion.md); no shipped matvec module may spill
# (tools/check_shader_spills.py).
SWIGLU_ROWS_CONFIG = {1: (1, 4), 2: (1, 2), 3: (2, 3), 4: (2, 2), 5: None}
SWIGLU_ROWS_CONFIG_SEPARATE = {1: (1, 4), 2: (1, 2), 3: (2, 3), 4: (2, 2), 5: None}
SWIGLU_ROWS_CONFIG_KQUANT = {1: (1, 4), 2: (1, 2), 3: None, 4: None, 5: None}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_args(p):
    """The shader tool options shared with compile_model.py. The tools are always given (Bazel
    passes its source-built ones): a host glslc is never picked up from PATH."""
    p.add_argument("--output-dir", type=Path, required=True, help="fresh or empty directory")
    p.add_argument("--glslc", type=Path, required=True, help="glslc executable (@shaderc//:glslc)")
    p.add_argument("--spirv-val", type=Path, required=True, help="spirv-val executable (@spirv_tools//:spirv-val)")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel compiles")
    p.add_argument("--quiet", action="store_true", help="do not print the manifest")


class Batch:
    """glslc + spirv-val runs, executed in parallel; manifest entries in the order added (the
    output is independent of --jobs)."""

    def __init__(self, p, a):
        # `bazel run` starts in the runfiles; a relative output directory means the caller's.
        a.output_dir = Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / a.output_dir
        if a.output_dir.exists() and any(a.output_dir.iterdir()): p.error("fresh or empty directory required")
        self.tools = {"glslc": str(a.glslc.absolute()), "spirv-val": str(a.spirv_val.absolute())}
        a.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs, self.items = a.jobs, []

    def identity(self):
        """The tools' version lines (their pinned source revisions; see MODULE.bazel)."""
        def version(tool):
            out = subprocess.run([self.tools[tool], "--version"], check=True, capture_output=True, text=True).stdout
            return [line for line in out.splitlines() if line and not line.startswith(("Target", "Targets", " "))]
        return {tool: version(tool) for tool in self.tools}

    def add(self, module, output, args, meta):
        """glslc ARGS -o OUTPUT, validated; manifest entry MODULE with its hash, size and META."""
        self.items.append((module, output, args, meta))

    def _one(self, item):
        _, output, args, _ = item
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([self.tools["glslc"], *args, "-o", str(output)], check=True)
        subprocess.run([self.tools["spirv-val"], "--target-env", "vulkan1.1", str(output)], check=True)

    def run(self, manifest):
        with ThreadPoolExecutor(self.jobs) as pool: list(pool.map(self._one, self.items))
        for module, output, _, meta in self.items:
            manifest["modules"][module] = dict(sha256=sha(output), bytes=output.stat().st_size, **meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    tool_args(p)
    a = p.parse_args()
    batch = Batch(p, a)
    manifest = dict(source_sha256=sha(ROOT/"src/matvec/matvec.comp"), tools=batch.identity(), modules={})
    manifest["rows_source_sha256"] = sha(ROOT/"src/matvec/matvec_rows.comp")
    # Two accumulation modes (docs/specs/matvec-push.md): fma (default, top directory) and
    # separate (`separate/`, the pre-FMA arithmetic with its own verify table).
    for prefix, accum_fma, rows_config, swiglu_config in (("", 1, ROWS_CONFIG, SWIGLU_ROWS_CONFIG), ("separate/", 0, ROWS_CONFIG_SEPARATE, SWIGLU_ROWS_CONFIG_SEPARATE)):
        (a.output_dir/prefix).mkdir(parents=True, exist_ok=True)
        accum = [f"-DACCUM_FMA={accum_fma}"]
        for name, fmt, width, payload, lanes in (("f32", 0, 4, 0, 256), ("f32_small", 0, 4, 0, 64), ("q4_0", 2, 18, 2, 64), ("q4_1", 3, 20, 4, 64), ("q5_k", 13, 176, 48, 64), ("q6_k", 14, 210, 0, 64), ("q4_1_aligned", 3, 20, 4, 64), ("q5_k_aligned", 13, 176, 48, 64)):
            output = a.output_dir/(prefix+name+".spv")
            batch.add(prefix+name, output, ["--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={payload}", f"-DLANES={lanes}", f"-DALIGNED_WORDS={int(name.endswith('_aligned'))}"] + accum + [str(ROOT/"src/matvec/matvec.comp")], dict(lanes=lanes, rows_per_group=1, aligned_words=name.endswith("_aligned"), accum_fma=bool(accum_fma)))
        # Fused gate/up/swiglu decode modules (block 17c): matvec.comp with SWIGLU.
        for name, fmt, width, payload, lanes in (("q4_0", 2, 18, 2, 64), ("q4_1", 3, 20, 4, 64), ("q5_k", 13, 176, 48, 64), ("q6_k", 14, 210, 0, 64), ("q4_1_aligned", 3, 20, 4, 64), ("q5_k_aligned", 13, 176, 48, 64)):
            module = prefix+"swiglu_"+name
            output = a.output_dir/(module+".spv")
            batch.add(module, output, ["--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={payload}", f"-DLANES={lanes}", f"-DALIGNED_WORDS={int(name.endswith('_aligned'))}", "-DSWIGLU=1"] + accum + [str(ROOT/"src/matvec/matvec.comp")], dict(lanes=lanes, rows_per_group=1, aligned_words=name.endswith("_aligned"), swiglu=True, accum_fma=bool(accum_fma)))
        # Multi-row modules (block 17b, speculative verification): one per exact input row
        # count 1..MAX_ROWS with its (GROUP weight rows per workgroup, CB blocks per chunk).
        for rows in range(1, MAX_ROWS+1):
            group, chunk = rows_config[rows]
            for name, fmt, width, payload, lanes in (("f32", 0, 4, 0, 256), ("f32_small", 0, 4, 0, 64), ("q4_0", 2, 18, 2, 64), ("q4_1", 3, 20, 4, 64), ("q5_k", 13, 176, 48, 64), ("q6_k", 14, 210, 0, 64), ("q4_1_aligned", 3, 20, 4, 64), ("q5_k_aligned", 13, 176, 48, 64), ("q8_0", 8, 34, 2, 64)):
                module = f"{prefix}rows{rows}_{name}"
                output = a.output_dir/(module+".spv")
                batch.add(module, output, ["--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={payload}", f"-DLANES={lanes}", f"-DALIGNED_WORDS={int(name.endswith('_aligned'))}", f"-DROWS={rows}", f"-DGROUP={group}", f"-DCB={chunk}"] + accum + [str(ROOT/"src/matvec/matvec_rows.comp")], dict(lanes=lanes, rows_per_group=group, blocks_per_chunk=chunk, input_rows=rows, aligned_words=name.endswith("_aligned"), accum_fma=bool(accum_fma)))
            # Fused verify FFN input modules (quantized formats with a single-row SWIGLU module).
            for name, fmt, width, payload, lanes in (("q4_0", 2, 18, 2, 64), ("q4_1", 3, 20, 4, 64), ("q5_k", 13, 176, 48, 64), ("q6_k", 14, 210, 0, 64), ("q4_1_aligned", 3, 20, 4, 64), ("q5_k_aligned", 13, 176, 48, 64)):
                config_row = (SWIGLU_ROWS_CONFIG_KQUANT if fmt in (13, 14) else swiglu_config)[rows]
                if config_row is None: continue
                group, chunk = config_row
                module = f"{prefix}rows{rows}_swiglu_{name}"
                output = a.output_dir/(module+".spv")
                batch.add(module, output, ["--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={payload}", f"-DLANES={lanes}", f"-DALIGNED_WORDS={int(name.endswith('_aligned'))}", f"-DROWS={rows}", f"-DGROUP={group}", f"-DCB={chunk}", "-DSWIGLU=1"] + accum + [str(ROOT/"src/matvec/matvec_rows.comp")], dict(lanes=lanes, rows_per_group=group, blocks_per_chunk=chunk, input_rows=rows, aligned_words=name.endswith("_aligned"), swiglu=True, accum_fma=bool(accum_fma)))
    batch.run(manifest)
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not a.quiet: print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
