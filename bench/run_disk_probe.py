#!/usr/bin/env python3
"""Bounded NVMe/imported-buffer experiment, NOT a serving/NVMe-cache benchmark.

  tools/py bench/run_disk_probe.py --scratch-dir DIR --output NEW_DIR

Uses 1 GiB of temporary disk space by default, never mounts/formats a disk or changes
filesystem attributes. On btrfs prepare a NOCOW scratch directory explicitly before use.
Tests the production GPU driver first, then depth 1/8/8/1 (ABBA), each with a warmup and
three alternating buffer-order trials. Logs all trials, failures, hashes and build identity.
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
    args = p.parse_args()
    if not args.scratch_dir.is_dir() or not 8 <= args.mib <= 4096 or args.mib % 8:
        p.error("scratch-dir must exist; mib must be a multiple of 8 in 8..4096")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    scratch_dir=str(args.scratch_dir.resolve()), uname=list(os.uname()),
                    mountinfo=Path("/proc/self/mountinfo").read_text(),
                    meminfo=Path("/proc/meminfo").read_text(),
                    thp=Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text(),
                    bytes=args.mib << 20, chunk_mib=8, depths=[1, 8, 8, 1],
                    status="running", commands=[])
    records = []
    try:
        with (args.output / "host-gate.log").open("w") as log:
            command = build.host_gpu_test_command()
            manifest["commands"].append(command)
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary("zerv-disk-probe")
        manifest.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = [ROOT / "bench/disk_probe.zig", Path(__file__), ROOT / "tools/zerv_build.py",
                   *sorted((ROOT / "src/gpu").glob("*.zig"))]
        manifest["source_sha256"] = {str(f.relative_to(ROOT)): build.sha(f) for f in sources}
        with tempfile.TemporaryDirectory(prefix="zerv-probe-", dir=args.scratch_dir) as temp:
            for index, depth in enumerate(manifest["depths"]):
                file = Path(temp) / f"probe-{index}.bin"
                command = [str(binary), str(file), str(args.mib), "8", str(depth)]
                manifest["commands"].append(command)
                log_path = args.output / f"probe-{index}-d{depth}.log"
                with log_path.open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=600, check=True)
                rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
                if len(rows) != 6 or not all(r["exact"] for r in rows):
                    raise RuntimeError("missing trials or failed byte gate")
                records.extend(dict(run=index, depth=depth, **r) for r in rows)
                file.unlink()
        summary = []
        for depth in (1, 8):
            for buffer in ("vulkan-host", "imported-anon"):
                rs = [r for r in records if r["depth"] == depth and r["buffer"] == buffer]
                row = dict(depth=depth, buffer=buffer, n=len(rs))
                for op in ("read", "write", "upload", "download"):
                    rates = [r["bytes"] / r[op + "_ns"] for r in rs]  # decimal GB/s
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
