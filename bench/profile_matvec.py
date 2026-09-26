#!/usr/bin/env python3
"""Diagnostic GPU timestamps and raw-weight streaming probe for block08c; not inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"tests/reference"), str(ROOT/"tools"), str(ROOT/"bench")]
import zerv_build  # noqa: E402
from generate_gpu_matvec import metrics, sha
from compile_matvec import PINS
from run_gpu_driver import gpu_snapshot


def records(text):
    rows = [json.loads(line) for line in text.splitlines() if "profile_trial" in line]
    if len(rows) != 32 or [r["profile_trial"] for r in rows] != list(range(32)): raise ValueError("missing profile trials")
    return rows


def xor_rows(case_path, rows, row_bytes):
    raw = np.memmap(case_path, dtype="<u4", mode="r", offset=32, shape=(rows, row_bytes//4))
    out = np.empty(rows, dtype="<u4")
    for start in range(0, rows, 16384):
        out[start:start+16384] = np.bitwise_xor.reduce(np.asarray(raw[start:start+16384]), axis=1)
    return out.tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=ROOT/"docs/bench/data/2026-09-22-gpu-matvec-repeat")
    p.add_argument("--cpu", type=int, default=10)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    artifact = ROOT/"third_party/matvec-push"/out.name; artifact.mkdir(parents=True, exist_ok=False)
    m = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, cpu=a.cpu, gpu_before=gpu_snapshot(),
             environment={k: v for k, v in os.environ.items() if k.startswith(("RADV_", "MESA_", "VK_"))}, commands=[], checks=[],
             warning="stream is a raw XOR read diagnostic omitting matvec arithmetic; never a matvec/inference result")
    def run(cmd):
        cmd = list(map(str, cmd)); m["commands"].append(cmd)
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
        with (out/"commands.log").open("a") as f: f.write(json.dumps(cmd)+"\n"+r.stdout+r.stderr)
        r.check_returncode(); return r
    try:
        for name, digest in PINS.items():
            if sha(Path(shutil.which(name))) != digest: raise ValueError("tool changed")
        stream = artifact/"stream.spv"
        run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", ROOT/"bench/matvec_stream.comp", "-o", stream])
        run(["spirv-val", "--target-env", "vulkan1.1", stream])
        run(zerv_build.build_command("zerv-gpu-matvec-bench"))
        native = artifact/"native"; shutil.copy2(zerv_build.path("zerv-gpu-matvec-bench"), native)
        m.update(native_sha256=sha(native), stream_sha256=sha(stream), stream_source_sha256=sha(ROOT/"bench/matvec_stream.comp"),
                 production_shader_manifest_sha256=sha(ROOT/"src/matvec/shaders/manifest.json"))
        os.sched_setaffinity(0, {a.cpu})
        corpus = json.loads((a.corpus/"manifest.json").read_text()); summary = {}
        for case in corpus["workloads"]:
            path = Path(case["case_path"]); n = case["rows"]
            if sha(path) != case["input_sha256"]: raise ValueError("case changed")
            golden = path.with_name(case["name"]+"-golden.output")
            record = next(r for r in corpus["checks"] if r["case"] == case["name"] and r["engine"] == "reference-f32-golden")
            if sha(golden) != record["output_sha256"]: raise ValueError("golden changed")
            raw = golden.read_bytes(); ideal = struct.unpack_from("<"+"d"*n, raw); sums = struct.unpack_from("<"+"d"*n, raw, 8*n)
            row_bytes = case["tensor"]["size"]//n
            xor = xor_rows(path, n, row_bytes)
            for mode in ("stress", "aligned", "stream"):
                output = artifact/f"{case['name']}-{mode}.output"
                cmd = [native, path, output, 0]
                if mode == "stream": cmd += [stream, 1]
                if mode != "stress": cmd.append("--aligned")
                cmd.append("--timestamps")
                proc = run(cmd); rows = records(proc.stdout)
                (out/f"{case['name']}-{mode}.jsonl").write_text(proc.stdout)
                data = output.read_bytes()
                if mode == "stream":
                    if data != xor: raise ValueError("stream XOR mismatch")
                    check = dict(xor_rows_sha256=sha(output))
                else:
                    check = metrics(struct.unpack("<"+"f"*n, data), ideal, sums)
                m["checks"].append(dict(case=case["name"], mode=mode, output_sha256=sha(output), check=check))
                gpu = [r["gpu_ns"] for r in rows]; wall = [r["wall_ns"] for r in rows]
                s = dict(gpu_median_ns=statistics.median(gpu), gpu_min_ns=min(gpu), gpu_max_ns=max(gpu), wall_median_ns=statistics.median(wall),
                         bytes=case["tensor"]["size"], weight_gbps=case["tensor"]["size"]/statistics.median(gpu))
                summary[f"{case['name']}/{mode}"] = s
                print(case["name"], mode, round(s["gpu_median_ns"]/1000, 3), round(s["wall_median_ns"]/1000, 3), round(s["weight_gbps"], 1), flush=True)
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        m["artifact_hashes"] = {str(x): sha(x) for x in sorted(artifact.iterdir()) if x.is_file() and x.suffix != ".output"}
        m["status"] = "passed"
    except Exception as error:
        m.update(status="failed", error=str(error)); raise
    finally:
        m["finished_at"] = datetime.now(timezone.utc).isoformat(); m["gpu_after"] = gpu_snapshot()
        (out/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")


if __name__ == "__main__": main()
