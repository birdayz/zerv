"""Reproduce report cells and compare all native HTTP outputs with the eager baseline."""
import json
from pathlib import Path
import sys

if "/bazel-out/" not in sys.executable:
    raise SystemExit("run with tools/py")

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "validated-summary.json").read_text())
baseline = ROOT / "docs/bench/data/2026-09-28-disk-serving-fixed/raw.jsonl"
want = {}
for line in baseline.read_text().splitlines():
    row = json.loads(line)
    if row["engine"].startswith("zerv"):
        key = row["level"], row["conversation"], row["turn"]
        assert want.setdefault(key, row["output_sha256"]) == row["output_sha256"]
native = 0
for line in (HERE / "raw.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["engine"].startswith("zerv"):
        key = row["level"], row["conversation"], row["turn"]
        assert row["output_sha256"] == want[key], key
        native += 1
assert native == 72
print(f"All {native} new native turns also match the earlier synchronous baseline.")
for name, value in summary["serving"].items():
    fields = []
    for key, scale in (("turn_0_ttft_p50_ms", 1000), ("turn_1_ttft_p50_ms", 1000), ("wall_s", 1), ("aggregate_tok_s", 1), ("gap_p99_ms", 1)):
        s = value[key]
        fields.append(f"{s['mean']/scale:.3f} ± {s['stdev']/scale:.3f}")
    print(name, " | ".join(fields), "peak VRAM", max(r["vram_peak"] for r in value["resources"]))
print(json.dumps(summary["component"], indent=2))
