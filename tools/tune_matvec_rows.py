#!/usr/bin/env python3
"""Per-count GROUP/CB sweep of the multi-row matvec (speculative verification; block 17b.1
procedure, docs/bench/2026-09-24-spec-verify.md), checked in for block 17c's FMA re-tune.

For each row count R and (GROUP, CB) pair: compile src/matvec/matvec_rows.comp into a lab
directory (same defines as tools/compile_matvec.py except GROUP/CB), run
`zerv-matvec-rows-bench MODEL --spv-dir DIR --group G R` (every benched role over all its
layers, bitwise check against the shipped single-row modules), and sum the multi-row
GPU medians over the roles. The grid runs PASSES times, alternating direction (D6: separate
runs drift ±3–4% with the thermal state), and each (R, G, CB) is scored by its mean sum.
Usage: tools/tune_matvec_rows.py --output DIR [--counts 2,3,4,5] [--grid 1:4,2:2,...] [--passes 2]"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_matvec import ROWS_CONFIG, sha  # noqa: E402
from check_shader_spills import stats as shader_stats  # noqa: E402
import zerv_build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf"
# The benched formats (name, FORMAT, BLOCK_BYTES, PAYLOAD_OFFSET, LANES), as compile_matvec.py.
FORMATS = (("f32", 0, 4, 0, 256), ("q4_0", 2, 18, 2, 64), ("q4_1", 3, 20, 4, 64), ("q5_k", 13, 176, 48, 64), ("q6_k", 14, 210, 0, 64))
GRID = "1:4,2:2,2:3,2:4,3:2,3:3,4:2"


def compile_variant(directory, rows, group, chunk, glslc, spirv_val):
    directory.mkdir(parents=True)
    for name, fmt, width, payload, lanes in FORMATS:
        out = directory/(name+".spv")
        subprocess.run([glslc, "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}",
                        f"-DPAYLOAD_OFFSET={payload}", f"-DLANES={lanes}", "-DALIGNED_WORDS=0", f"-DROWS={rows}", f"-DGROUP={group}", f"-DCB={chunk}",
                        str(ROOT/"src/matvec/matvec_rows.comp"), "-o", str(out)], check=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.1", str(out)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--counts", default="2,3,4,5")
    p.add_argument("--grid", default=GRID)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--samples", type=int, default=9)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    work = ROOT/"third_party/matvec-rows-tune"/out.name; work.mkdir(parents=True, exist_ok=False)
    glslc, spirv_val = zerv_build.shader_tools()
    bench = work/"zerv-matvec-rows-bench"; bench.write_bytes(zerv_build.binary("zerv-matvec-rows-bench").read_bytes()); bench.chmod(0o755)
    counts = [int(c) for c in a.counts.split(",")]
    grid = [tuple(int(v) for v in item.split(":")) for item in a.grid.split(",")]
    variants = [(r, g, c) for r in counts for g, c in grid]
    for r, g, c in variants: compile_variant(work/f"r{r}g{g}c{c}", r, g, c, glslc, spirv_val)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, bench_sha256=sha(bench), model=str(MODEL),
                    rows_source_sha256=sha(ROOT/"src/matvec/matvec_rows.comp"), shipped_rows_config=ROWS_CONFIG, runs=[])
    raw = (out/"raw.jsonl").open("w")
    sums = {}
    invalid = {}
    for pass_index in range(a.passes):
        order = variants if pass_index % 2 == 0 else list(reversed(variants))
        for r, g, c in order:
            cmd = [str(bench), str(MODEL), "--spv-dir", str(work/f"r{r}g{g}c{c}"), "--group", str(g), "--samples", str(a.samples), str(r)]
            # Shader statistics (cache off so every module compiles): a variant whose modules
            # spill VGPRs is invalid. ACO's LDS spills can miscompile, scratch spills are slow
            # (docs/bench/2026-09-24-aco-lds-spill.md).
            env = dict(os.environ, MESA_SHADER_CACHE_DISABLE="true", RADV_DEBUG="shaderstats")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            if res.returncode: raise SystemExit(f"r{r}g{g}c{c} failed:\n{res.stderr[-2000:]}")
            lines = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
            # A variant whose rows are not bitwise equal to the single-row module is invalid
            # (recorded, never chosen): verify must equal decode.
            bad = [line["role"] for line in lines if line["bitwise_mismatches"]]
            if any(s["spilled_vgprs"] for s in shader_stats(res.stderr)): bad.append("vgpr-spill")
            if bad: invalid.setdefault((r, g, c), set()).update(bad)
            total = sum(line["multi_ns_median"] for line in lines)
            single = sum(line["single_ns_median"] for line in lines)
            sums.setdefault((r, g, c), []).append(total)
            for line in lines: raw.write(json.dumps(dict(line, pass_index=pass_index, cb=c))+"\n")
            raw.flush()
            manifest["runs"].append(dict(cmd=cmd, pass_index=pass_index, multi_sum_ns=total, single_sum_ns=single))
            print(f"pass {pass_index} rows {r} G {g} CB {c}: multi {total/1e3:.0f} us (single-row x1: {single/1e3:.0f} us)" + (f" MISMATCH {bad}" if bad else ""), flush=True)
    raw.close()
    best = {}
    for r in counts:
        scored = sorted((sum(v)/len(v), g, c) for (rr, g, c), v in sums.items() if rr == r and (rr, g, c) not in invalid)
        best[r] = dict(group=scored[0][1], chunk=scored[0][2], mean_us=scored[0][0]/1e3,
                       table=[dict(group=g, chunk=c, mean_us=s/1e3) for s, g, c in scored])
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), best=best,
                    invalid={f"r{r}g{g}c{c}": sorted(v) for (r, g, c), v in invalid.items()})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    print(json.dumps({r: (b["group"], b["chunk"], round(b["mean_us"])) for r, b in best.items()}))


if __name__ == "__main__":
    main()
