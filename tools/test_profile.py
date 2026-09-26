#!/usr/bin/env python3
"""Per-test wall times of the unit test binaries (docs/development.md, "Tests in parallel").

  tools/test_profile.py [--optimize Debug|ReleaseFast] [--top 20] [name ...]

Builds the Zig unit test binaries with Bazel (//tests:NAME, or //tests:NAME_release_fast),
then runs each (or the named ones) alone, one at a time so the times are not skewed by the
others, in its runfiles tree as `bazel test` does, and times every test from the runner's
per-test lines. Prints the slowest tests and each binary's run time. Development tool; no
GPU. Bazel's own per-target times: `bazel test --test_summary=detailed`.
"""
import argparse, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402


def unit_tests():
    """The CPU Zig unit tests (zig_test targets without the gpu tag)."""
    query = 'kind("zig_test", //tests:all) except attr(tags, "\\bgpu\\b", //tests:all)'
    labels = zerv_build.bazel("query", query, capture=True).stdout.split()
    return [label.rsplit(":", 1)[1] for label in labels]


def label(name, optimize):
    return f"//tests:{name}" + ("_release_fast" if optimize == "ReleaseFast" else "")


def build(names, optimize):
    """The unit test binaries, in parallel (`bazel build`)."""
    t0 = time.monotonic()
    r = subprocess.run([zerv_build.BAZEL, "build", *(label(n, optimize) for n in names)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0: raise SystemExit(f"build failed\n{r.stderr[-3000:]}")
    return time.monotonic() - t0


def run(exe):
    """Per-test durations: the runner prints 'i/n name...' before a test and 'OK' after it."""
    t0 = time.monotonic()
    p = subprocess.Popen([str(exe)], cwd=exe.parent / (exe.name + ".runfiles/_main"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    times, current, start = [], None, t0
    buf = ""
    while True:
        ch = p.stdout.read(1)
        if not ch: break
        buf += ch
        m = re.search(r"(\d+)/(\d+) ([^\n]*?)\.\.\.$", buf)
        if m:
            current, start, buf = m.group(3), time.monotonic(), ""
        elif buf.endswith("\n"):
            if current is not None and ("OK" in buf or "FAIL" in buf or "SKIP" in buf):
                times.append((time.monotonic() - start, current, "FAIL" not in buf))
                current = None
            buf = ""
    rc = p.wait()
    return times, time.monotonic() - t0, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--optimize", default="Debug", choices=["Debug", "ReleaseFast"])
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    names = a.names or unit_tests()
    build_s = build(names, a.optimize)
    print(f"build (bazel build, all binaries in parallel): {build_s:.2f} s")
    rows, binaries, failed = [], [], False
    for name in names:
        times, run_s, rc = run(ROOT / "bazel-bin/tests" / label(name, a.optimize).rsplit(":", 1)[1])
        failed |= rc != 0
        binaries.append((run_s, name, run_s, rc))
        rows += [(d, f"{name}: {test}", ok) for d, test, ok in times]
    print(f"slowest tests ({a.optimize}):")
    for d, test, ok in sorted(rows, reverse=True)[:a.top]:
        print(f"  {d:7.2f} s  {test}{'' if ok else '  FAILED'}")
    print("binaries (run alone):")
    for total, name, r, rc in sorted(binaries, reverse=True):
        print(f"  {r:6.2f} s  {name:16s}{'' if rc == 0 else f'  exit {rc}'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
