#!/usr/bin/env python3
"""In-model race of fused verify FFN tables (block 17c; matvec_rows.comp with SWIGLU).

For each variant GROUP:CB (applied to every row count 2..5; count 1 keeps 1:4), build a
`zerv-spec-check` binary with that table (tools/compile_matvec.py SWIGLU_ROWS_CONFIG and
matvec.swigluRowsGroups edited in place, modules compiled into a fresh lab directory and
copied into src/matvec/shaders), then restore the sources and rebuild. The binaries run
interleaved (D6: separate runs drift with the thermal state), PASSES rounds, alternating
direction; each run is the full spec-check (11 bitwise cases must pass) whose verify+commit
timings per row count are recorded. Output: DIR/runs.jsonl, DIR/manifest.json.
Before racing, every variant runs the full GPU fixture suite (all formats, not only this
model's) under tools/check_shader_spills.py; variants that fail it (wrong results or VGPR
spills in LDS) are recorded in the manifest and not raced. The K-quant rows of the table
(SWIGLU_ROWS_CONFIG_KQUANT) are not varied.
Grid item `off`: the first built variant's binary run with verify fusion off (the unfused
reference, interleaved with the rest).
Usage: tools/race_swiglu_rows.py --output DIR [--grid off,1:2,1:4,2:2,2:3] [--passes 3]"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import zerv_build  # noqa: E402
MODEL = ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf"
COMPILE = ROOT/"tools/compile_matvec.py"
MATVEC = ROOT/"src/matvec/root.zig"
SHADERS = ROOT/"src/matvec/shaders"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_table(group, chunk):
    table = {1: (1, 4)} | {n: (group, chunk) for n in range(2, 6)}
    text = COMPILE.read_text()
    text, n = re.subn(r"SWIGLU_ROWS_CONFIG = \{[^}]*\}", "SWIGLU_ROWS_CONFIG = "+repr(table), text)
    assert n == 1
    COMPILE.write_text(text)
    text = MATVEC.read_text()
    groups = ", ".join(str(table[n][0]) for n in range(1, 6))
    text, n = re.subn(r"(pub fn swigluRowsGroups.*?\.fma => \.)\{[^}]*\}", r"\g<1>{ "+groups+" }", text, flags=re.S)
    assert n == 1
    MATVEC.write_text(text)


def build(lab):
    subprocess.run([sys.executable, str(COMPILE), "--output-dir", str(lab)], check=True, stdout=subprocess.DEVNULL)
    for spv in lab.glob("rows*_swiglu_*.spv"): shutil.copy(spv, SHADERS/spv.name)
    return zerv_build.binary("zerv-spec-check")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid", default="1:2,1:4,2:2,2:3")
    p.add_argument("--passes", type=int, default=3)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    work = ROOT/"third_party/swiglu-rows-race"/out.name; work.mkdir(parents=True, exist_ok=False)
    saved = {path: path.read_bytes() for path in (COMPILE, MATVEC)}
    saved_spv = {spv: spv.read_bytes() for spv in SHADERS.glob("rows*_swiglu_*.spv")}
    variants = [tuple(int(v) for v in item.split(":")) if item != "off" else "off" for item in a.grid.split(",")]
    built = [v for v in variants if v != "off"]
    binaries = {}
    validation = {}
    try:
        for g, c in built:
            set_table(g, c)
            spec_check = build(work/f"spv-g{g}c{c}")
            # The spill gate over the ReleaseFast GPU tests (a Bazel test: it reruns because
            # the shaders changed); its test log holds the gate's summary.
            gate = subprocess.run([zerv_build.BAZEL, "test", "--test_output=summary", "//tests:gpu_spills"], cwd=ROOT,
                                  capture_output=True, text=True)
            testlogs = ROOT/"bazel-testlogs/tests/gpu_spills"
            shutil.copy(testlogs/"test.outputs/shaderstats.txt", work/f"gpu-test-g{g}c{c}.txt")
            validation[f"g{g}c{c}"] = dict(returncode=gate.returncode, summary=(testlogs/"test.log").read_text().strip().splitlines()[-1:])
            print(f"g{g}c{c} validation:", validation[f"g{g}c{c}"], flush=True)
            if gate.returncode != 0: continue  # 1: VGPR spills in LDS; 2: a GPU test failed
            binary = work/f"zerv-spec-check-g{g}c{c}"
            shutil.copy(spec_check, binary)
            binaries[(g, c)] = binary
    finally:
        for path, data in saved.items(): path.write_bytes(data)
        for spv, data in saved_spv.items(): spv.write_bytes(data)
        zerv_build.build("zerv-spec-check")
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, model=str(MODEL), passes=a.passes,
                    binaries={f"g{g}c{c}": sha(b) for (g, c), b in binaries.items()}, validation=validation)
    variants = [v for v in variants if v == "off" or v in binaries]
    built = [v for v in built if v in binaries]
    if not built: raise SystemExit("no variant passed validation")
    manifest["off_binary"] = f"g{built[0][0]}c{built[0][1]}"
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (out/"runs.jsonl").open("w") as runs:
        for pass_index in range(a.passes):
            order = variants if pass_index % 2 == 0 else variants[::-1]
            for variant in order:
                if variant == "off":
                    name = "off"
                    result = subprocess.run([str(binaries[built[0]]), str(MODEL), "f32", "fused", "fma", "separate"], capture_output=True, text=True)
                else:
                    name = f"g{variant[0]}c{variant[1]}"
                    result = subprocess.run([str(binaries[variant]), str(MODEL)], capture_output=True, text=True)
                lines = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                ok = sum(1 for line in lines if line.get("ok") is True)
                timings = {("step" if t["timing"] == "step" else str(t["rows"])): t["ms_median"] for t in lines if "timing" in t}
                record = dict(variant=name, pass_index=pass_index, returncode=result.returncode, ok_cases=ok, timings=timings)
                runs.write(json.dumps(record)+"\n"); runs.flush()
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
