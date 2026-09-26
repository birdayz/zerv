#!/usr/bin/env python3
"""Rebuild the pre-DFS matvec from its hash-verified checked-in source snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from rebuild_tokenizer_baseline import verified_sources
from run_gpu_matvec import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"docs/bench/data/2026-09-22-gpu-matvec-repeat")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run, dest = a.run.resolve(), a.output.resolve()
    if not dest.is_relative_to(ROOT/"third_party"): p.error("output must be under third_party")
    saved = json.loads((run/"manifest.json").read_text())
    # The archived tree builds with its own build.zig, using the Zig of this repository's
    # Bazel toolchain (the same executable, sha256 2317bbb9..., as the archived runs).
    sys.path.insert(0, str(ROOT/"tools")); import zerv_build  # noqa: E402
    zig = zerv_build.binary("zig", config=None)
    if saved["status"] != "passed" or sha(zig) != saved["zig_sha256"]: raise ValueError("baseline/compiler mismatch")
    sources = verified_sources(run, saved)
    required = {Path("src/matvec/root.zig"), Path("src/matvec/matvec.comp"), Path("bench/gpu_matvec.zig")}
    if not required.issubset({p for p, _ in sources}): raise ValueError("incomplete matvec snapshot")
    dest.mkdir(parents=True, exist_ok=False)
    m = dict(status="running", restored_from=str(run), original_manifest_sha256=sha(run/"manifest.json"),
             started_at=datetime.now(timezone.utc).isoformat(), sources=saved["sources"], commands=[], zig_sha256=sha(zig))
    try:
        for relative, source in sources:
            target = dest/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        commands = [[sys.executable, str(dest/"source/tools/compile_matvec.py"), "--output-dir", str(dest/"shaders")]]
        commands += [[str(zig), "build", *args, "--summary", "all"] for args in
                     (("test", "gpu-test"), ("test", "gpu-test", "gpu-matvec-bench-build", "-Doptimize=ReleaseFast", "-Dcpu=native"))]
        for cmd in commands:
            m["commands"].append(cmd)
            result = subprocess.run(cmd, cwd=dest/"source", capture_output=True, text=True, timeout=600)
            with (dest/"commands.log").open("a") as f: f.write(json.dumps(cmd)+"\n"+result.stdout+result.stderr)
            result.check_returncode()
        for module in (dest/"shaders").iterdir():
            if module.read_bytes() != (dest/"source/src/matvec/shaders"/module.name).read_bytes():
                raise ValueError("baseline shader rebuild mismatch")
        native = dest/"source/zig-out/bin/zerv-gpu-matvec-bench"
        m.update(status="passed", native_binary=str(native), native_binary_sha256=sha(native))
        print(native)
    except Exception as error:
        m.update(status="failed", error=str(error)); raise
    finally:
        m["finished_at"] = datetime.now(timezone.utc).isoformat()
        (dest/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
