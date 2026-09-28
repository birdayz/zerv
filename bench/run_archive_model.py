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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--scratch-dir", type=Path, help="enable production disk owner; existing operator-prepared directory")
    p.add_argument("--direct-alignment", type=int, default=0)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--source", action="store_true", help="leased mixed cache source, canonical tails and live source reuse")
    mode.add_argument("--pressure", action="store_true", help="production pressure selection/poll/discard over the mixed-source gate")
    p.add_argument("--tokens", type=int, nargs="+", default=[257, 80000])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
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
        for index, tokens in enumerate(a.tokens):
            cmd = [str(binary), str(a.model.resolve()), str(tokens)]
            if a.scratch_dir: cmd += [str(a.scratch_dir.resolve()), str(a.direct_alignment)]
            if a.source: cmd += ["pressure" if a.pressure else "source"]
            m["commands"].append(cmd)
            log_path = a.output / f"case-{index}-{tokens}.log"
            with log_path.open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env=build.host_vulkan_env(), stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
            rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
            if len(rows) != 1 or not rows[0]["exact_state"] or rows[0]["exact_vocab_rows"] != 4 or rows[0]["disk"] != bool(a.scratch_dir): raise RuntimeError("missing exactness result")
            if a.source and (not rows[0]["source"] or not rows[0]["source_cpu_quanta"] or not rows[0]["source_gpu_quanta"]): raise RuntimeError("mixed source path not exercised")
            if a.pressure and not rows[0]["pressure"]: raise RuntimeError("pressure owner not exercised")
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
