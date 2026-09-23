#!/usr/bin/env python3
"""External-reference GPU op timestamps via ggml's GGML_VK_PERF_LOGGER; diagnostic only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"tests/reference")]
from generate_gpu_matvec import metrics, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", type=Path, default=ROOT/"docs/bench/data/2026-09-22-matvec-final-repeat")
    p.add_argument("--iterations", type=int, default=64)
    p.add_argument("--cpu", type=int, default=10)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    run = json.loads((a.run/"manifest.json").read_text())
    binary = Path(run["reference"]["path"])
    m = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, reference_sha256=sha(binary),
             note="GGML_VK_PERF_LOGGER inserts timestamp writes and extra barriers; its per-op GPU interval is diagnostic, not submit-to-completion time", checks=[])
    try:
        if sha(binary) != run["reference"]["sha256"]: raise ValueError("reference changed")
        os.sched_setaffinity(0, {a.cpu}); summary = {}
        for case in run["workloads"]:
            path = Path(case["case_path"]); n = case["rows"]
            if sha(path) != case["input_sha256"]: raise ValueError("case changed")
            env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_VK_")}
            env.update(GGML_VK_DISABLE_MMVQ="1", GGML_VK_PERF_LOGGER="1")
            output = ROOT/"third_party/matvec-push"/(out.name+"-"+case["name"]+".output")
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(binary), str(path), str(output), str(a.iterations)]
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=600)
            (out/(case["name"]+".stderr")).write_text(proc.stderr)
            if proc.returncode or "AMD Radeon RX 7900 XTX" not in proc.stderr: raise ValueError("reference failed")
            values = [float(v) for v in re.findall(r"^MUL_MAT_VEC[^:]*: 1 x ([0-9.]+) us", proc.stderr, re.M)]
            if len(values) < a.iterations: raise ValueError("missing perf-logger records")
            timed = values[-a.iterations:]
            raw = output.read_bytes()
            ideal = struct.unpack_from("<"+"d"*n, raw); sums = struct.unpack_from("<"+"d"*n, raw, 8*n)
            actual = struct.unpack_from("<"+"f"*n, raw, 16*n)
            m["checks"].append(dict(case=case["name"], command=cmd, records=len(values), metrics=metrics(actual, ideal, sums), output_sha256=sha(output)))
            summary[case["name"]] = dict(gpu_median_ns=1000*statistics.median(timed), gpu_min_ns=1000*min(timed), gpu_max_ns=1000*max(timed), samples=len(timed))
            print(case["name"], round(statistics.median(timed), 3), flush=True)
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        m["status"] = "passed"
    except Exception as error:
        m.update(status="failed", error=str(error)); raise
    finally:
        m["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")


if __name__ == "__main__": main()
