#!/usr/bin/env python3
"""Rebuild an archived tokenizer baseline without relying on its ignored binary."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
sys.path.insert(0, str(ROOT))
from bench.run_tokenizer import sha


def verified_sources(run, manifest):
    result = []
    for relative, digest in manifest["sources"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe snapshot path")
        source = run / "source" / path
        if source.is_symlink() or sha(source) != digest:
            raise ValueError(f"snapshot changed: {relative}")
        result.append((path, source))
    if not {Path("build.zig"), Path(".zig-version"), Path("bench/tokenizer.zig")}.issubset({p for p, _ in result}):
        raise ValueError("incomplete source snapshot")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="fresh directory beneath third_party")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--zig", type=Path, help="default: the Zig of this repository's Bazel toolchain (//bazel:zig)")
    a = p.parse_args()
    if a.zig is None:
        # The archived tree builds with its own build.zig; the toolchain's Zig is the same
        # executable (sha256 2317bbb9...) as the archived runs used.
        sys.path.insert(0, str(ROOT / "tools")); import zerv_build  # noqa: E402
        a.zig = zerv_build.binary("zig", config=None)
    run, dest, zig, model = [x.resolve() for x in (a.run, a.output, a.zig, a.model)]
    if not dest.is_relative_to(ROOT / "third_party"):
        p.error("rebuild artifacts must be under ignored third_party")
    saved = json.loads((run / "manifest.json").read_text())
    if saved.get("status") != "passed" or sha(zig) != saved["zig_sha256"] or sha(model) != saved["model_sha256"]:
        raise ValueError("baseline/compiler/model identity mismatch")
    if sha(run / "native-corpus.json") != saved["native_corpus_sha256"]:
        raise ValueError("baseline corpus changed")
    sources = verified_sources(run, saved)
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(saved, status="running", restored_from=str(run), commands=[],
                    restored_at=datetime.now(timezone.utc).isoformat(),
                    note="Rebuilt from verified archived sources; original binary was not read. Binary hash may change with build path.")
    try:
        for relative, source in sources:
            target = dest / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(run / "native-corpus.json", dest / "native-corpus.json")
        commands = [[str(zig), "build", "test", "--summary", "all"],
                    [str(zig), "build", "test", "tokenizer-bench-build", "-Doptimize=ReleaseFast", "-Dcpu=native", "--summary", "all"],
                    [str(dest / "source/zig-out/bin/zerv-tokenizer-bench"), str(model), str(dest / "native-corpus.json")]]
        for i, command in enumerate(commands):
            manifest["commands"].append(command)
            result = subprocess.run(command, cwd=dest / "source", capture_output=True, text=True)
            (dest / f"{i}.stdout").write_text(result.stdout)
            (dest / f"{i}.stderr").write_text(result.stderr)
            result.check_returncode()
        expected = json.loads((run / "native-validation.json").read_text())
        if json.loads(result.stdout) != expected:
            raise ValueError("rebuilt baseline failed the original full-corpus check")
        (dest / "native-validation.json").write_text(result.stdout)
        native = dest / "source/zig-out/bin/zerv-tokenizer-bench"
        manifest.update(status="passed", native_binary=str(native), native_binary_sha256=sha(native))
        print(json.dumps(dict(status="passed", baseline_run=str(dest), checks=expected), indent=2))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
