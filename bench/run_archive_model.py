#!/usr/bin/env python3
"""Gate archive addressing on real model state and full-vocabulary continuation rows.
RAM adapter or production disk owner, not HTTP serving. Defaults to 257 and 80000 prefix tokens.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build as build


def planned_cases(tokens, cadence):
    return [(n, mode) for i, n in enumerate(tokens)
            for mode in ((["unit", "chunk"] if i % 2 == 0 else ["chunk", "unit"]) if cadence == "both" else [cadence])]


def validate_prefill(row, cadence):
    if row["exact_packed_rows"] != 2 or (row["prefill_source_quanta"] > 0) != (cadence == "unit"):
        raise RuntimeError("packed prefill cadence/rows not exercised")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--scratch-dir", type=Path, help="enable production disk owner; existing operator-prepared directory")
    p.add_argument("--direct-alignment", type=int, default=0)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demand", action="store_true", help="production disk-to-staging prefetch, reuse, stop and handoff")
    mode.add_argument("--prepare", action="store_true", help="leased preparation cancel/live-hit/host-restore exactness")
    mode.add_argument("--source", action="store_true", help="leased mixed cache source, canonical tails and live source reuse")
    mode.add_argument("--pressure", action="store_true", help="production pressure selection/poll/discard over the mixed-source gate")
    p.add_argument("--prefill", action="store_true", help="pressure capture between packed prefill units; exact solo rows")
    p.add_argument("--prefill-cadence", choices=["unit", "chunk", "both"], default="unit", help="matched diagnostic counterfactual; both alternates order")
    p.add_argument("--prepare-window", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--chunk-mib", type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument("--tokens", type=int, nargs="+", default=[257, 80000])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.prepare and (a.prefill or min(a.tokens) <= 128): p.error("--prepare needs >128 tokens and cannot combine with --prefill")
    if a.demand and (not a.scratch_dir or a.prefill or min(a.tokens) <= 128 or a.prepare_window > 2): p.error("--demand needs disk, >128 tokens, window1/2 and no --prefill")
    if a.prepare_window != 1 and not (a.prepare or a.demand): p.error("--prepare-window requires --prepare")
    if a.chunk_mib != 1 and not a.scratch_dir: p.error("--chunk-mib requires disk")
    if a.prefill_cadence != "unit" and not a.prefill: p.error("--prefill-cadence requires --prefill")
    if a.prefill: a.pressure = True
    if a.pressure: a.source = True
    if a.scratch_dir is not None and not a.scratch_dir.is_dir(): p.error("scratch directory does not exist")
    if a.source and (not a.scratch_dir or min(a.tokens) <= 128): p.error("--source needs disk and prefixes >128 tokens")
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv, commands=[], status="running", scope="model archive addressing and production disk owner" if a.scratch_dir else "RAM model archive addressing, not disk/serving")
    try:
        for name, cmd in [("cpu", build.test_command()), ("host_gpu", build.host_gpu_test_command())]:
            m["commands"].append(cmd)
            with (a.output / (name + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary("zerv-archive-model-check")
        m.update(build=build.provenance(), binary_sha256=build.sha(binary), model=dict(path=str(a.model), sha256=build.sha(a.model)))
        sources = [*sorted((ROOT / "src").rglob("*.zig")), ROOT / "bench/archive_model_check.zig", Path(__file__).resolve()]
        m["sources"] = {str(f.relative_to(ROOT)): build.sha(f) for f in sources}
        results = []
        for index, (tokens, cadence) in enumerate(planned_cases(a.tokens, a.prefill_cadence)):
            cmd = [str(binary), str(a.model.resolve()), str(tokens)]
            if a.scratch_dir: cmd += [str(a.scratch_dir.resolve()), str(a.direct_alignment)]
            if a.demand: cmd += ["demand"]
            elif a.prepare:
                if not a.scratch_dir: cmd += ["-", "0"]
                cmd += ["prepare"]
            elif a.prefill: cmd += ["prefill-chunk" if cadence == "chunk" else "prefill"]
            elif a.source: cmd += ["pressure" if a.pressure else "source"]
            elif a.scratch_dir: cmd += ["disk"]
            if a.scratch_dir or a.prepare: cmd += [str(a.chunk_mib)]
            if a.prepare or a.demand: cmd += [str(a.prepare_window)]
            m["commands"].append(cmd)
            log_path = a.output / f"case-{index}-{tokens}.log"
            with log_path.open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env=build.host_vulkan_env(), stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
            if len(rows) != 1 or not rows[0]["exact_state"] or rows[0]["exact_vocab_rows"] != (8 if a.prepare else 4) or rows[0]["disk"] != bool(a.scratch_dir): raise RuntimeError("missing exactness result")
            if a.prepare and (not rows[0]["preparation"] or rows[0]["aborts"] != 2 or rows[0]["concurrent_read_rows"] != int(bool(a.scratch_dir)) or rows[0]["window"] != a.prepare_window or rows[0]["first_pages"] != min(a.prepare_window, (tokens + 127) // 128 - 1) or rows[0]["committed"] != rows[0]["first_pages"] + rows[0]["packed_pages"] or rows[0]["freed"] < rows[0]["packed_pages"] or rows[0]["exact_packed_rows"] != 2): raise RuntimeError("preparation ownership not exercised")
            if a.demand and (not rows[0]["demand"] or rows[0]["window"] != a.prepare_window or rows[0]["handoffs"] != 2 or rows[0]["cancellations"] != 4 or not rows[0]["source_conflict"] or not rows[0]["admission_drain"] or not rows[0]["corrupt_miss"] or not rows[0]["foreground_preemption"] or rows[0]["exact_packed_rows"] != 1): raise RuntimeError("demand ownership not exercised")
            if a.source and (not rows[0]["source"] or not rows[0]["source_cpu_quanta"] or not rows[0]["source_gpu_quanta"]): raise RuntimeError("mixed source path not exercised")
            if a.pressure and not rows[0]["pressure"]: raise RuntimeError("pressure owner not exercised")
            if a.prefill:
                validate_prefill(rows[0], cadence)
                rows[0]["prefill_cadence"] = cadence
            rows[0]["chunk_mib"] = a.chunk_mib
            results += rows
        m.update(status="passed", results=results)
    except BaseException as e:
        m.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}")
    main()
