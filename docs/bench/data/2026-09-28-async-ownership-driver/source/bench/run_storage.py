#!/usr/bin/env python3
"""Measure the bounded disk worker vs synchronous positional OS I/O.

  tools/py bench/run_storage.py --scratch-dir DIR --output NEW_DIR [--mib 1024]
      [--direct-alignment BYTES]

Uses at most --mib MiB temporary disk space; no filesystem-attribute changes.
Prepare the directory according to deployment guidance; on the target btrfs host,
use an explicitly prepared NOCOW directory and --direct-alignment 4096.
This is a CPU/disk component test, not a GPU or serving benchmark.
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mib", type=int, default=1024)
    p.add_argument("--direct-alignment", type=int, default=0,
                   help="explicit memory/offset alignment; 0 requires filesystem-reported values")
    args = p.parse_args()
    if args.direct_alignment < 0 or args.direct_alignment > 64 << 20 or (args.direct_alignment & (args.direct_alignment - 1)):
        p.error("direct-alignment must be 0 or a power of two <= 64 MiB")
    if not args.scratch_dir.is_dir() or not 8 <= args.mib <= 4096 or args.mib % 8:
        p.error("scratch-dir must exist; mib must be a multiple of 8 in 8..4096")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    scratch_dir=str(args.scratch_dir.resolve()), uname=list(os.uname()),
                    mountinfo=Path("/proc/self/mountinfo").read_text(),
                    meminfo=Path("/proc/meminfo").read_text(),
                    bytes=args.mib << 20, chunk_mib=8, depths=[1, 8, 8, 1],
                    configured_alignment=args.direct_alignment,
                    status="running", commands=[])
    records = []
    try:
        command = build.test_command()
        manifest["commands"].append(command)
        with (args.output / "tests.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary("zerv-storage-bench")
        manifest.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = [ROOT / "bench/storage.zig", Path(__file__), ROOT / "tools/zerv_build.py",
                   ROOT / "src/storage/root.zig", ROOT / "tests/storage.zig",
                   *sorted((ROOT / "tests/fixtures/storage").glob("*"))]
        manifest["source_sha256"] = {str(f.relative_to(ROOT)): build.sha(f) for f in sources}
        with tempfile.TemporaryDirectory(prefix="zerv-worker-", dir=args.scratch_dir) as temp:
            for index, depth in enumerate(manifest["depths"]):
                command = [str(binary), temp, str(args.mib), "8", str(depth), str(args.direct_alignment)]
                manifest["commands"].append(command)
                log_path = args.output / f"run-{index}-d{depth}.log"
                with log_path.open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=600, check=True)
                rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
                if len(rows) != 10 or not all(r["exact"] for r in rows):
                    raise RuntimeError("missing trials or failed byte gate")
                records.extend(dict(run=index, worker_depth=depth, **r) for r in rows)
        summary = []
        for depth in (1, 8):
            for path in ("worker", "pread-pwrite"):
                rs = [r for r in records if r["worker_depth"] == depth and r["path"] == path]
                row = dict(worker_depth=depth, path=path, n=len(rs))
                for op in ("read", "write"):
                    rates = [r["bytes"] / r[op + "_ns"] for r in rs]
                    row[op + "_GBps"] = dict(mean=statistics.mean(rates), min=min(rates), max=max(rates),
                                              stdev=statistics.stdev(rates))
                summary.append(row)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
    except BaseException as e:
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (args.output / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
