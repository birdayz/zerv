#!/usr/bin/env python3
"""Correctness-gated matvec shader experiment, retaining source/modules/raw outputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"tests/reference"), str(ROOT/"tools"), str(ROOT/"bench")]
import zerv_build  # noqa: E402
from generate_gpu_matvec import TYPES, metrics, sha
from run_gpu_matvec import timings
from run_gpu_driver import gpu_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--define", action="append", default=[])
    p.add_argument("--format-rows", action="append", default=[], help="FORMAT=ROWS override for dispatch rows per group")
    p.add_argument("--aligned", action="store_true", help="separate natural-alignment experiment; never replaces the offset2 stress case")
    p.add_argument("--corpus", type=Path, default=ROOT/"docs/bench/data/2026-09-22-gpu-matvec-repeat")
    p.add_argument("--cpu", type=int, default=10)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    artifact = ROOT/"third_party/matvec-dfs"/out.name; artifact.mkdir(parents=True, exist_ok=False)
    m = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, rows=a.rows, defines=a.define,
             cpu=a.cpu, aligned=a.aligned, environment={k: v for k, v in os.environ.items() if k.startswith(("RADV_", "MESA_", "VK_", "GGML_"))}, gpu_before=gpu_snapshot(), commands=[], checks=[], source_sha256=sha(a.source), tools=PINS, modules={})
    def run(cmd):
        cmd = list(map(str, cmd)); m["commands"].append(cmd)
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=300)
        with (out/"commands.log").open("a") as f: f.write(json.dumps(cmd)+"\n"+r.stdout+r.stderr)
        r.check_returncode(); return r
    try:
        rows = {fmt: a.rows for fmt in TYPES}
        for item in a.format_rows:
            fmt, value = item.split("=")
            if fmt not in rows: raise ValueError("unknown format override")
            rows[fmt] = int(value)
        m["format_rows"] = rows
        if not all(1 <= r <= 32 for r in rows.values()) or a.cpu not in os.sched_getaffinity(0): raise ValueError("bad configuration")
        glslc, spirv_val = zerv_build.shader_tools()
        source = out/"candidate.comp"; shutil.copyfile(a.source, source)
        for fmt, (code, _, width) in TYPES.items():
            spv = artifact/(fmt+".spv")
            run([glslc, "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={code}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={4 if fmt == 'q4_1' else 2}", *["-D"+d for d in a.define], source, "-o", spv])
            run([spirv_val, "--target-env", "vulkan1.1", spv])
            m["modules"][fmt] = dict(path=str(spv), sha256=sha(spv))
        run(zerv_build.build_command("zerv-gpu-matvec-bench"))
        native = artifact/"native"; shutil.copy2(zerv_build.path("zerv-gpu-matvec-bench"), native)
        m["native_sha256"] = sha(native)
        os.sched_setaffinity(0, {a.cpu})
        extra = ["--aligned"] if a.aligned else []
        if len(set(rows.values())) == 1:
            run([native, "--fixtures", artifact, str(a.rows), *extra])
        else:
            for fmt, value in rows.items():
                run([native, "--fixtures", artifact, str(value), fmt, *extra])
        m["fixtures"] = "48 cases, replay/guards/changed input passed"
        baseline = json.loads((a.corpus/"manifest.json").read_text())
        summary = {}
        for case in baseline["workloads"]:
            path = Path(case["case_path"]); n = case["rows"]; fmt = case["format"]
            if sha(path) != case["input_sha256"]: raise ValueError("case changed")
            golden = path.with_name(case["name"]+"-golden.output")
            record = next(r for r in baseline["checks"] if r["case"] == case["name"] and r["engine"] == "reference-f32-golden")
            if sha(golden) != record["output_sha256"]: raise ValueError("independent golden changed")
            raw = golden.read_bytes()
            ideal = struct.unpack_from("<"+"d"*n, raw); sums = struct.unpack_from("<"+"d"*n, raw, n*8)
            output = artifact/(case["name"]+".output")
            proc = run([native, path, output, case["iterations"], artifact/(fmt+".spv"), str(rows[fmt]), *extra])
            values = timings(proc.stdout, case["iterations"])
            (out/(case["name"]+".jsonl")).write_text(proc.stdout)
            actual = struct.unpack("<"+"f"*n, output.read_bytes())
            check = metrics(actual, ideal, sums)
            m["checks"].append(dict(case=case["name"], input_sha256=case["input_sha256"], output_sha256=sha(output), metrics=check))
            profile_output = artifact/(case["name"]+".profile.output")
            profiled = run([native, path, profile_output, 0, artifact/(fmt+".spv"), str(rows[fmt]), *extra, "--timestamps"])
            if profile_output.read_bytes() != output.read_bytes(): raise ValueError("profiled output differs")
            (out/(case["name"]+".profile.jsonl")).write_text(profiled.stdout)
            gpu = [json.loads(line)["gpu_ns"] for line in profiled.stdout.splitlines()]
            if len(gpu) != 32: raise ValueError("missing GPU timestamp trials")
            summary[case["name"]] = dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values), stdev_ns=statistics.stdev(values), gpu_median_ns=statistics.median(gpu), gpu_min_ns=min(gpu))
            print(case["name"], round(statistics.median(values)/1000, 3), round(statistics.median(gpu)/1000, 3), flush=True)
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        m["status"] = "passed"
    except Exception as error:
        m.update(status="failed", error=str(error)); raise
    finally:
        m["finished_at"] = datetime.now(timezone.utc).isoformat(); m["gpu_after"] = gpu_snapshot()
        (out/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
