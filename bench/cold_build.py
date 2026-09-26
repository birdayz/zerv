#!/usr/bin/env python3
"""Cold, no-op and incremental build+check times: the pre-Bazel build (`zig build check`) against
Bazel (`bazel test`), each on a clean checkout (docs/bench/2026-09-26-bazel-build.md).

  tools/py bench/cold_build.py --baseline-tree DIR --bazel-tree DIR --output DIR [--trials 3]

Per trial and system, in alternating order (ABBA over trials):

  cold         every cache empty: baseline: fresh ZIG_GLOBAL_CACHE_DIR and ZIG_LOCAL_CACHE_DIR;
               Bazel: fresh output base, disk cache off, fresh Zig caches of rules_zig and
               hermetic_cc_toolchain (they live outside the output base). Downloads are not part
               of "cold": the repository cache stays (the baseline's Zig is installed too).
  noop         the same command again
  incremental  after appending a comment line to EDIT (the same file in both trees), restored after

Commands: baseline `zig build check` (AGENTS.md's required checks before Bazel: zig fmt, unit
tests Debug + ReleaseFast, Python tests; its Zig is the toolchain's, byte-identical to the
baseline's pinned one); Bazel `bazel test //...` (the required checks now, which also rebuild the
shader tools and check every generated file, build every executable and check linkage), and
`matched`: `bazel test --build_tests_only` of the same kinds of checks as the baseline (the
generated-file, linkage and no-C++ checks excluded).

Records wall time, CPU time (children's rusage; for Bazel also its server's including reaped
action processes), the load average before each step, cache sizes after the cold step, exit
status and the tests executed. The trees are prepared by the caller (a checkout each); their
revisions are read from their .git files and every tracked-looking source is hashed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import host_info  # noqa: E402  (tools/host_info.py: the host, recorded without host tools)
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)

EDIT = "src/session/root.zig"
NOT_MATCHED = ["//src/matvec:generated_shaders_test", "//src/model:generated_shaders_test", "//src/model:native_code_test",
               "//src/model:native_code_test_radv_test", "//src/gpu:vk_zig_test", "//tests:vulkan_fixtures_test",
               "//tests:test_linkage", "//src:no_cc"]
TICK = os.sysconf("SC_CLK_TCK")


def tree_hash(tree):
    h = hashlib.sha256()
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree)
        if p.is_file() and not p.is_symlink() and rel.parts[0] not in (".git", ".zig-cache", "zig-out", "third_party", "models") \
                and not rel.parts[0].startswith("bazel-"):
            h.update(str(rel).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try: total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError: pass
    return total


def remove(path):
    if not path.exists(): return
    for dirpath, dirs, _ in os.walk(path):
        for d in dirs:
            try: os.chmod(os.path.join(dirpath, d), 0o755)
            except OSError: pass
    shutil.rmtree(path)


def timed(cmd, cwd, env, log):
    """Runs cmd; returns (wall s, children CPU s, exit status, output)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    with log.open("a") as f: f.write(json.dumps(cmd) + "\n" + r.stdout + "\n")
    return wall, cpu, r.returncode, r.stdout


def server_cpu(pid):
    """CPU seconds of a Bazel server and the action processes it has reaped."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return sum(int(x) for x in fields[11:15]) / TICK


def executed(output):
    line = next((l for l in output.splitlines() if l.startswith("Executed ")), "")
    return line.strip()


class Edit:
    def __init__(self, tree):
        self.path = tree / EDIT
        self.original = self.path.read_bytes()

    def __enter__(self):
        self.path.write_bytes(self.original + b"// cold_build.py incremental edit\n")

    def __exit__(self, *exc):
        self.path.write_bytes(self.original)


def baseline(tree, work, log, zig):
    caches = work / "zig-cache"
    env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(caches / "global"), ZIG_LOCAL_CACHE_DIR=str(caches / "local"))
    cmd = [str(zig), "build", "check"]
    out = {}
    for step in ("cold", "noop", "incremental"):
        load = Path("/proc/loadavg").read_text().split()[0]
        if step == "incremental":
            with Edit(tree): wall, cpu, status, text = timed(cmd, tree, env, log)
        else:
            wall, cpu, status, text = timed(cmd, tree, env, log)
        out[step] = dict(wall_s=wall, cpu_s=cpu, status=status, load1_before=float(load))
        if step == "cold": out[step]["cache_bytes"] = size(caches)
        print(f"  baseline {step:12s} {wall:8.1f} s wall {cpu:8.1f} s CPU exit {status}", flush=True)
        if status: raise SystemExit(f"baseline {step} failed; see {log}")
    remove(caches)
    return out


def bazel(tree, work, log, matched):
    base = work / ("ob-matched" if matched else "ob")
    zc = work / "zig-caches"
    startup = [zerv_build.BAZEL, f"--output_base={base}"]
    flags = ["--disk_cache=", "--symlink_prefix=/", f"--sandbox_add_mount_pair={zc}",
             f"--repo_env=RULES_ZIG_CACHE_PREFIX={zc / 'rules_zig'}", f"--repo_env=HERMETIC_CC_TOOLCHAIN_CACHE_PREFIX={zc / 'hermetic_cc'}"]
    if matched: flags.append("--build_tests_only")
    targets = ["--", "//..."] + ([f"-{t}" for t in NOT_MATCHED] if matched else [])
    cmd = startup + ["test"] + flags + targets
    zc.mkdir(parents=True)
    env = dict(os.environ)
    env.pop("BAZELISK_SKIP_WRAPPER", None)
    out, pid = {}, None
    for step in ("cold", "noop", "incremental"):
        load = Path("/proc/loadavg").read_text().split()[0]
        cpu0 = server_cpu(pid) if pid else 0.0
        if step == "incremental":
            with Edit(tree): wall, cpu, status, text = timed(cmd, tree, env, log)
        else:
            wall, cpu, status, text = timed(cmd, tree, env, log)
        if pid is None:
            pid = int(subprocess.run(startup + ["info", "server_pid"], cwd=tree, env=env, capture_output=True, text=True, check=True).stdout)
        cpu += server_cpu(pid) - cpu0
        out[step] = dict(wall_s=wall, cpu_s=cpu, status=status, load1_before=float(load), tests=executed(text))
        if step == "cold": out[step].update(output_base_bytes=size(base), zig_cache_bytes=size(zc))
        name = "matched" if matched else "bazel"
        print(f"  {name:8s} {step:12s} {wall:8.1f} s wall {cpu:8.1f} s CPU exit {status} {out[step]['tests']}", flush=True)
        if status: raise SystemExit(f"{name} {step} failed; see {log}")
    subprocess.run(startup + ["shutdown"], cwd=tree, env=env, check=True)
    remove(base); remove(zc)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-tree", type=Path, required=True)
    p.add_argument("--bazel-tree", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, help="scratch for caches and output bases (default: OUTPUT/../cold-build-work)")
    p.add_argument("--trials", type=int, default=3)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    work = (a.work or out.parent / (out.name + "-work")).resolve(); work.mkdir(parents=True, exist_ok=False)
    btree, ztree = a.baseline_tree.resolve(strict=True), a.bazel_tree.resolve(strict=True)
    for tree in (btree, ztree):
        if not (tree / EDIT).is_file(): raise SystemExit(f"{tree}: no {EDIT}")
    busy = host_info.processes(r"(^|/)(zerv(-[a-z0-9-]+)?|llama-server)( |$)|vllm serve")
    load = float(Path("/proc/loadavg").read_text().split()[0])
    if busy or load > 2: raise SystemExit(f"machine busy (load {load}): " + "; ".join(busy))
    zig = zerv_build.binary("zig", config=None)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, cpu=host_info.cpu(),
                    baseline=dict(tree=str(btree), revision=zerv_build.source_revision(btree), sources_sha256=tree_hash(btree),
                                  command="zig build check", zig=str(zig), zig_sha256=zerv_build.sha(zig.resolve())),
                    bazel=dict(tree=str(ztree), revision=zerv_build.source_revision(ztree), sources_sha256=tree_hash(ztree),
                               command="bazel test //...", matched_excluded=NOT_MATCHED),
                    edit=EDIT, trials=[])
    log = out / "commands.log"
    for trial in range(a.trials):
        print(f"trial {trial}", flush=True)
        order = ["baseline", "bazel", "matched"] if trial % 2 == 0 else ["matched", "bazel", "baseline"]
        record = dict(trial=trial, order=order)
        for system in order:
            record[system] = (baseline(btree, work / f"t{trial}-baseline", log, zig) if system == "baseline"
                              else bazel(ztree, work / f"t{trial}-{system}", log, system == "matched"))
        manifest["trials"].append(record)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    summary = {}
    for system in ("baseline", "bazel", "matched"):
        for step in ("cold", "noop", "incremental"):
            walls = sorted(t[system][step]["wall_s"] for t in manifest["trials"])
            cpus = sorted(t[system][step]["cpu_s"] for t in manifest["trials"])
            summary[f"{system}/{step}"] = dict(wall_s=walls, cpu_s=cpus, median_wall_s=walls[len(walls) // 2])
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: round(v["median_wall_s"], 1) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
