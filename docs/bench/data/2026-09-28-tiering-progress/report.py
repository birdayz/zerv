"""Validate/print D.0a evidence; no model or server execution.
Run with tools/py, from the repository root, after the serving and paired model runs.
"""
import json
from pathlib import Path
import re
import sys

if "/bazel-out/" not in sys.executable:
    raise SystemExit("run with tools/py")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "bench"))
from summarize_archive import summarize, stats

base = ROOT / "docs/bench/data/2026-09-28-progress-serving"
component = ROOT / "docs/bench/data/2026-09-28-progress-model"
paired = ROOT / "docs/bench/data/2026-09-28-progress-paired"
out = summarize(base, component)
reference = {}
for line in (ROOT / "docs/bench/data/2026-09-28-async-archive-serving/raw.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["engine"].startswith("zerv"):
        key = row["level"], row["conversation"], row["turn"]
        identity = row["output_sha256"], row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"]
        assert reference.setdefault(key, identity) == identity
counts = {}
for line in (base / "raw.jsonl").read_text().splitlines():
    row = json.loads(line)
    assert not row.get("error") and row["usage"]["completion_tokens"] > 0
    key = row["level"], row["conversation"], row["turn"]
    equal = (row["output_sha256"], row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"]) == reference[key]
    if row["engine"].startswith("zerv"):
        assert equal, (row["engine"], key, row["round"])
    count = counts.setdefault(row["engine"], [0, 0])
    count[0] += int(equal)
    count[1] += 1
assert out["native_identical"] == 96
print("Text AND prompt/generated token identity vs historical native baseline:", json.dumps(counts))
for engine, measured in out["serving"].items():
    print(engine)
    for key in ["wall_s", "turn_0_ttft_p50_ms", "turn_1_ttft_p50_ms", "aggregate_tok_s", "gap_p99_ms"]:
        print(" ", key, json.dumps(measured[key]))
    print(" resources", json.dumps(measured["resources"]))
    if "prefix-cache-disk-dir" in engine:
        for resource in measured["resources"]:
            log = (base / resource["log"]).read_text()
            for label in ["disk prefix archive", "disk source retention"]:
                match = re.findall(r"zerv: " + label + r": (.*)", log)
                assert len(match) == 1
                print(" round", resource["round"], label, match[0])
cm = json.loads((component / "manifest.json").read_text())
assert cm["status"] == "passed"
assert all(r["prefill_source_quanta"] > 0 and r["exact_packed_rows"] == 2 for r in cm["results"])
short = [r for r in cm["results"] if r["prefix"] == 257][1:]
assert len(short) == 5
for key in ["capture_ns", "disk_write_ns", "restore_ns"]:
    print("component", key, "ms", json.dumps(stats([r[key] / 1e6 for r in short])))
print("long", json.dumps(out["component"]["long"]))
pm = json.loads((paired / "manifest.json").read_text())
assert pm["status"] == "passed" and len(pm["results"]) == 12
assert all(r["exact_state"] and r["exact_vocab_rows"] == 4 and r["exact_packed_rows"] == 2 for r in pm["results"])
for r in pm["results"]:
    assert (r["prefill_source_quanta"] > 0) == (r["prefill_cadence"] == "unit")
for mode in ["unit", "chunk"]:
    trials = [r for r in pm["results"] if r["prefill_cadence"] == mode][1:]
    assert len(trials) == 5
    print("paired", mode, json.dumps(stats([r["disk_write_ns"] / 1e6 for r in trials])))
deltas = []
for i in range(2, 12, 2):
    pair = {r["prefill_cadence"]: r["disk_write_ns"] / 1e6 for r in pm["results"][i:i+2]}
    deltas.append(pair["unit"] - pair["chunk"])
print("paired unit-minus-chunk ms", json.dumps(stats(deltas)))
out["identity_vs_baseline"] = counts
out["paired_write_delta_ms"] = stats(deltas)
(base / "validated-summary.json").write_text(json.dumps(out, indent=2) + "\n")
