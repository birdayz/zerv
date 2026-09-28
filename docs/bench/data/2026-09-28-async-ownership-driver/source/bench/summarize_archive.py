#!/usr/bin/env python3
"""Validate and summarize the integrated archive evidence (no model/server execution).
 tools/py bench/summarize_archive.py --serving DIR --component DIR --output FILE
"""
import argparse
import json
from pathlib import Path
import statistics
import sys


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), stdev=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def summarize(serving, component):
    manifest = json.loads((serving / "manifest.json").read_text())
    if manifest.get("status") != "passed": raise ValueError("serving run incomplete or failed")
    rows = [json.loads(s) for s in (serving / "raw.jsonl").read_text().splitlines()]
    if any(r.get("error") or not r.get("usage") or not r.get("output_sha256") for r in rows): raise ValueError("failed/incomplete response")
    root = Path(__file__).resolve().parents[1]
    workload = json.loads((root / next(iter(manifest["workload"]))).read_text())
    expected = {(rnd, level, workload["conversations"][i]["name"], t) for rnd in range(manifest["rounds"])
                for level in manifest["levels"] for i in range(level) for t in range(len(workload["conversations"][i]["turns"]))}
    by_engine = {}
    for name in manifest["engines"]:
        selected = [r for r in rows if r["engine"] == name]
        keys = {(r["round"], r["level"], r["conversation"], r["turn"]) for r in selected}
        if keys != expected or len(selected) != len(expected): raise ValueError(f"missing/duplicate responses: {name}")
        by_engine[name] = selected
    native = {name: selected for name, selected in by_engine.items() if name.startswith("zerv")}
    reference = {}
    for selected in native.values():
        for r in selected:
            key = (r["level"], r["conversation"], r["turn"])
            want = reference.setdefault(key, r["output_sha256"])
            if want != r["output_sha256"]: raise ValueError(f"native output mismatch: {key}, {r['engine']}, round {r['round']}")
    measured = json.loads((serving / "summary.json").read_text())
    out = dict(native_identical=sum(map(len, native.values())), serving={}, component={})
    for name, trials in measured.items():
        item = {k: stats([r[k] for r in trials]) for k in ("wall_s", "aggregate_tok_s", "ttft_p50_ms", "ttft_p95_ms")}
        item["gap_p99_ms"] = stats([r["stream_gap_ms"]["p99"] for r in trials])
        for t in range(len(workload["conversations"][0]["turns"])):
            item[f"turn_{t}_ttft_p50_ms"] = stats([r["turns"][t]["ttft_p50_ms"] for r in trials])
        item["resources"] = manifest["engines"][name]["resources"]
        out["serving"][name] = item
    cm = json.loads((component / "manifest.json").read_text())
    if cm["status"] != "passed" or any(not r["disk"] or not r["exact_state"] or r["exact_vocab_rows"] != 4 for r in cm["results"]):
        raise ValueError("component did not pass disk exactness")
    short = [r for r in cm["results"] if r["prefix"] == 257][1:]  # first fresh-process trial warms hardware
    if short:
        out["component"]["short_after_warmup"] = {k.removesuffix("_ns") + "_ms": stats([r[k] / 1e6 for r in short]) for k in ("disk_write_ns", "restore_ns")}
    out["component"]["long"] = [r for r in cm["results"] if r["prefix"] == 80000]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serving", type=Path, required=True)
    p.add_argument("--component", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.serving, a.component)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['native_identical']} native turns identical; complete runs only")
    for name, s in result["serving"].items():
        print(name, "wall", s["wall_s"], "turn1", s["turn_1_ttft_p50_ms"], "gap99", s["gap_p99_ms"])
    print("component", result["component"])


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable: sys.exit(f"run it with tools/py {sys.argv[0]}")
    main()
