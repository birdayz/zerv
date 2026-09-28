#!/usr/bin/env python3
"""Measure snapshot residency bookkeeping, not I/O or serving.

  tools/py bench/run_residency.py --output NEW_DIRECTORY
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build as build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    uname=list(os.uname()), cpuinfo=Path("/proc/cpuinfo").read_text(),
                    status="running", commands=[], scope="metadata only; no physical I/O")
    try:
        command = build.test_command()
        manifest["commands"].append(command)
        with (args.output / "tests.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary("zerv-residency-bench")
        manifest.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = [ROOT / "bench/residency.zig", Path(__file__), ROOT / "tools/zerv_build.py",
                   ROOT / "src/session/residency.zig", ROOT / "tests/residency.zig",
                   ROOT / "tests/reference/generate_residency_fixture.py",
                   *sorted((ROOT / "tests/fixtures/residency").glob("*"))]
        manifest["source_sha256"] = {str(f.relative_to(ROOT)): build.sha(f) for f in sources}
        manifest["commands"].append([str(binary)])
        with (args.output / "native.log").open("w") as log:
            subprocess.run([str(binary)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=600, check=True)
        rows = [json.loads(line) for line in (args.output / "native.log").read_text().splitlines()
                if line.startswith("{")]
        if len(rows) != 30 or not all(row["exact"] for row in rows):
            raise RuntimeError("missing trials or failed state checks")
        (args.output / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = []
        for count in (64, 256, 1024):
            for hot in (1, 8):
                rs = [r for r in rows if r["entries"] == count and r["hot"] == hot]
                if len(rs) != 5 or {r["trial"] for r in rs} != set(range(5)):
                    raise RuntimeError("incomplete trial set")
                result = dict(entries=count, hot=hot, n=len(rs), metadata_bytes=rs[0]["metadata_bytes"])
                for metric in ("save_spill_ns", "restore_lease_demote_ns", "drop_ns"):
                    values = [r[metric] / (r["cycles"] * count) for r in rs]
                    result[metric + "_per_record"] = dict(mean=statistics.mean(values),
                        stdev=statistics.stdev(values), min=min(values), max=max(values))
                summary.append(result)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
    except BaseException as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
