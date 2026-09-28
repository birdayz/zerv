#!/usr/bin/env python3
"""CPU/disk archive measurement; no GPU/serving claim. Uses 256 MiB temporary disk.
 tools/py bench/run_archive.py --scratch-dir DIR --direct-alignment 4096 --output NEW_DIR
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build as build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--direct-alignment", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.scratch_dir.is_dir() or a.direct_alignment not in (0, 4096):
        p.error("existing scratch-dir required; this harness supports reported or 4096 alignment")
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv, uname=list(os.uname()),
             status="running", commands=[], scope="CPU archive, not GPU or serving")
    try:
        command = build.test_command(); m["commands"].append(command)
        with (a.output / "tests.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary("zerv-archive-bench")
        m.update(build=build.provenance(), binary_sha256=build.sha(binary))
        files = ["src/session/archive.zig", "src/storage/root.zig", "bench/archive.zig", "bench/run_archive.py", "tests/archive.zig", "tests/fixtures/archive/oracle.json", "tests/reference/generate_archive_fixture.py"]
        m["source_sha256"] = {f: build.sha(ROOT / f) for f in files}
        with tempfile.TemporaryDirectory(prefix="zerv-archive-", dir=a.scratch_dir) as directory:
            command = [str(binary), directory, str(a.direct_alignment)]; m["commands"].append(command)
            with (a.output / "native.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        rows = [json.loads(line) for line in (a.output / "native.log").read_text().splitlines() if line.startswith("{")]
        if len(rows) != 5 or not all(r["exact"] for r in rows): raise RuntimeError("missing or inexact trial")
        (a.output / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = {}
        for op in ("read", "write"):
            values = [r["bytes"] / r[op + "_ns"] for r in rows]
            summary[op + "_GBps"] = dict(mean=statistics.mean(values), stdev=statistics.stdev(values), min=min(values), max=max(values), n=5)
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        m["status"] = "passed"
    except BaseException as e:
        m.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
