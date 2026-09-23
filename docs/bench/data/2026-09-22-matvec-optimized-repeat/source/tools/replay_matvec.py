#!/usr/bin/env python3
"""Fresh-source matvec replay: verify/restore research, regenerate goldens/SPIR-V, run gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--restore-sources", action="store_true", help="download missing pinned small research files; never replace mismatches/install packages/download weights")
    a = p.parse_args()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, sources={}, commands=[])

    def run(command):
        command = list(map(str, command)); manifest["commands"].append(command)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
        with (out/"commands.log").open("a") as f: f.write(json.dumps(command)+"\n"+result.stdout+result.stderr)
        result.check_returncode()

    try:
        ledger = json.loads((ROOT/"docs/research/2026-09-22/gpu-matvec-sources.json").read_text())
        seen = set()
        for source in ledger:
            path = (ROOT/source["local_path"]).resolve()
            if not path.is_relative_to(ROOT/"third_party") or path in seen: raise ValueError("unsafe/duplicate research path")
            seen.add(path)
            if not path.exists():
                if not a.restore_sources: raise ValueError("missing research file; opt in with --restore-sources: "+str(path))
                if not source["url"].startswith("https://raw.githubusercontent.com/ggml-org/ggml/"+source["revision"]+"/"): raise ValueError("unexpected research origin")
                with urllib.request.urlopen(source["url"], timeout=30) as response: data = response.read(8*1024*1024+1)
                if len(data) > 8*1024*1024 or hashlib.sha256(data).hexdigest() != source["sha256"]: raise ValueError("download hash mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as f: f.write(data)
            if sha(path) != source["sha256"]: raise ValueError("existing research hash mismatch: "+str(path))
            manifest["sources"][source["local_path"]] = source["sha256"]
        run([sys.executable, ROOT/"tools/compile_matvec.py", "--output-dir", out/"shaders"])
        for path in (out/"shaders").iterdir():
            if path.read_bytes() != (ROOT/"src/matvec/shaders"/path.name).read_bytes(): raise ValueError("owned shader replay mismatch")
        run([sys.executable, ROOT/"tests/reference/generate_gpu_matvec.py", "--model", a.model.resolve(), "--work-dir", out/"oracle", "--output", out/"matvec.json"])
        if (out/"matvec.json").read_bytes() != (ROOT/"tests/fixtures/gpu/matvec.json").read_bytes(): raise ValueError("independent golden replay mismatch")
        manifest["fixture_sha256"] = sha(out/"matvec.json")
        zig = ROOT/".tools/zig-x86_64-linux-0.16.0/zig"
        manifest["zig_sha256"] = sha(zig)
        run([zig, "fmt", "--check", ROOT/"build.zig", ROOT/"src", *sorted((ROOT/"bench").glob("*.zig")), *sorted((ROOT/"tools").glob("*.zig")), *sorted((ROOT/"tests").glob("*.zig"))])
        run([zig, "build", "test", "gpu-test", "--summary", "all"])
        run([zig, "build", "test", "gpu-test", "-Doptimize=ReleaseFast", "--summary", "all"])
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])
        manifest["status"] = "passed"
        print("verified independent fixture, six SPIR-V modules, research sources and native/Python suites")
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__": main()
