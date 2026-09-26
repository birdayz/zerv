#!/usr/bin/env python3
"""Committed generated files equal what Bazel regenerates (bazel/shaders.bzl, bazel/native.bzl).

  check_generated.py GENERATED COMMITTED [--subset] [--hint TEXT]   compare (a Bazel test)
  check_generated.py GENERATED COMMITTED --update                   write GENERATED over COMMITTED

GENERATED is the regenerated directory; COMMITTED its committed counterpart, relative to the
repository root (the runfiles root in a test, the workspace for `bazel run`). The comparison
covers every file both ways (missing, extra and differing files fail); with --subset,
COMMITTED may hold further files that Bazel does not generate. --hint says how to regenerate.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys


def files(root):
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("generated", type=Path)
    p.add_argument("committed")
    p.add_argument("--update", action="store_true")
    p.add_argument("--subset", action="store_true")
    p.add_argument("--hint", default="")
    a = p.parse_args()
    new = files(a.generated)
    if not new: raise SystemExit(f"{a.generated}: nothing generated")
    if a.update:
        target = Path(os.environ["BUILD_WORKSPACE_DIRECTORY"]) / a.committed
        for rel, path in files(target).items():
            if rel not in new and not a.subset: path.unlink(); print("removed", a.committed + "/" + rel)
        for rel, path in new.items():
            out = target / rel
            if out.exists() and out.read_bytes() == path.read_bytes(): continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out); print("wrote", a.committed + "/" + rel)
        return 0
    old = files(Path(a.committed))
    missing = sorted(new.keys() - old.keys())
    extra = [] if a.subset else sorted(old.keys() - new.keys())
    changed = sorted(rel for rel in new.keys() & old.keys() if new[rel].read_bytes() != old[rel].read_bytes())
    for label, names in (("not committed", missing), ("committed but not generated", extra), ("differs", changed)):
        for rel in names: print(f"{label}: {a.committed}/{rel}")
    if missing or extra or changed:
        print(f"{a.committed} is not what its sources and tools produce. If the change is intended: {a.hint}")
        return 1
    print(f"{a.committed}: {len(new)} files equal the regenerated ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
