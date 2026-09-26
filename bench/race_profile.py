#!/usr/bin/env python3
"""Interleaved A/B of `zerv-model-profile` binaries on one workload (docs/performance.md).

Each round runs every engine once in order, then once in reverse (ABBA...), so drift and
thermal state hit all engines alike. Every run's raw JSONL and stderr are kept; the
summary gives per-engine median/min/max of decode step GPU time, decode attention phases,
and prefill totals (all chunks) with their attention phases.

  bench/race_profile.py --engine old=PATH --engine new=PATH --rounds 2 \
      --output docs/bench/data/NEW --args 32768 512 30000 32 f16@native f32

An engine may carry its own profiler arguments after `|` (replacing --args), e.g. one
binary at several KV page sizes: --engine "p128=PATH|32768 512 30000 32 f16@native f32@page=128".
"""
import shlex
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(lines):
    """Per-run numbers from the profiler's JSON lines."""
    prefill = [d for d in lines if d["kind"] == "prefill"]
    decode = [d for d in lines if d["kind"] == "decode"]
    if len(decode) != 1 or not prefill:
        raise SystemExit("unexpected profiler output")
    dp = decode[0]["phases_ms"]
    return {
        "decode_gpu_ms": decode[0]["gpu_ms"],
        "decode_qkprep_ms": dp.get("qkprep", 0.0),
        "decode_scores_ms": dp.get("scores", 0.0),
        "decode_pv_ms": dp.get("pv", 0.0),
        "decode_attention_ms": dp.get("attention", 0.0),
        "prefill_gpu_ms": sum(d["gpu_ms"] for d in prefill),
        "prefill_qkprep_ms": sum(d["phases_ms"].get("qkprep", 0.0) for d in prefill),
        "prefill_attention_ms": sum(d["phases_ms"].get("attention", 0.0) for d in prefill),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--engine", action="append", required=True, help="NAME=PATH[|ARGS] of a zerv-model-profile binary")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--output", type=Path, required=True, help="fresh directory")
    p.add_argument("--args", nargs="+", required=True, help="CONTEXT CHUNK PROMPT_TOKENS DECODE_STEPS [PRECISION [KV]] (engines without their own)")
    a = p.parse_args()
    if len(a.engine) < 2 or any("=" not in e for e in a.engine):
        p.error("two or more --engine NAME=PATH[|ARGS]")
    engines, engine_args = [], {}
    for e in a.engine:
        n, spec = e.split("=", 1)
        path, _, extra = spec.partition("|")
        engines.append((n, path))
        engine_args[n] = shlex.split(extra) if extra else a.args
    if len({n for n, _ in engines}) != len(engines):
        p.error("engine names must differ")
    if a.output.exists():
        p.error("fresh output directory required")
    a.output.mkdir(parents=True)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, model=str(a.model),
                    model_bytes=a.model.stat().st_size, profile_args=a.args, rounds=a.rounds,
                    engines={n: dict(path=path, sha256=sha(path), args=engine_args[n]) for n, path in engines}, runs=[])
    results = {n: [] for n, _ in engines}
    for r in range(a.rounds):
        for n, path in engines + engines[::-1]:
            i = len(results[n])
            cmd = [path, str(a.model), *engine_args[n]]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            (a.output/f"{n}-{i}.jsonl").write_text(run.stdout)
            (a.output/f"{n}-{i}.stderr").write_text(run.stderr)
            if run.returncode:
                raise SystemExit(f"{n} run {i} failed:\n{run.stderr[-2000:]}")
            m = metrics([json.loads(line) for line in run.stdout.splitlines() if line.strip()])
            results[n].append(m)
            manifest["runs"].append(dict(engine=n, index=i, round=r, cmd=cmd, **m))
            print(json.dumps(dict(engine=n, index=i, **m)), flush=True)
    summary = {}
    for n, runs in results.items():
        summary[n] = {k: dict(median=statistics.median(x[k] for x in runs), min=min(x[k] for x in runs),
                              max=max(x[k] for x in runs)) for k in runs[0]}
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), summary=summary)
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    base = engines[0][0]
    for k in results[base][0]:
        cells = []
        for n, _ in engines:
            s = summary[n][k]
            cells.append(f"{n} {s['median']:.4f} [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"{k}: " + "; ".join(cells))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
