#!/usr/bin/env python3
"""Run a repository Python script with the pinned interpreter and packages (hermetic).

  tools/py SCRIPT [ARGS...]      (= bazel run //tools:py -- SCRIPT [ARGS...])

The harnesses and reference generators (bench/, tools/, tests/reference/) run from the
source tree, as with `python3 SCRIPT`, but under rules_python's interpreter with the
hash-locked packages of requirements_lock.txt, never the host's Python
(docs/specs/hermetic-build.md). The working directory is the caller's; child processes that
start `sys.executable` (reference workers) get the same interpreter and packages.
"""
import os
import sys


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    os.chdir(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()))
    script = os.path.abspath(sys.argv[1])
    # The packages are on this process's path (runfiles); children inherit them.
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in sys.path[1:] if p)
    os.environ["ZERV_HERMETIC_PYTHON"] = "1"
    os.execv(sys.executable, [sys.executable, script, *sys.argv[2:]])


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
