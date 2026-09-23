#!/usr/bin/env python3
"""Render the fixed official-template golden corpus, native vs precompiled Jinja."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/chat-template.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(rows, expected):
    if len(rows) != 7:
        raise ValueError("expected seven trials")
    keys = {"trial", "iterations", "renders", "bytes", "elapsed_ns", "output_sha256"}
    seen = set()
    for row in rows:
        if set(row) != keys or any(type(row[k]) is not int for k in keys - {"output_sha256"}):
            raise ValueError("bad schema")
        if row["trial"] not in range(7) or row["trial"] in seen or row["iterations"] != 100 or row["elapsed_ns"] <= 0:
            raise ValueError("bad trial")
        seen.add(row["trial"])
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError("output mismatch")


def reference_worker(config):
    sys.path.insert(0, str(ROOT / "tests/reference"))
    from generate_chat_goldens import environment
    template = environment().from_string(json.loads(config.read_text())["chat_template"])
    cases = [c for c in json.loads(FIXTURE.read_text())["cases"] if "output" in c["official"]]
    prepared = [dict(messages=c["messages"], **dict({"add_generation_prompt": True}, **c["options"])) for c in cases]
    outputs = [template.render(**kw) for kw in prepared]
    if outputs != [c["official"]["output"] for c in cases]:
        raise ValueError("Jinja/golden mismatch")
    encoded = "".join(outputs).encode()
    for trial in range(-3, 7):
        iterations = 1 if trial < 0 else 100
        start = time.perf_counter_ns()
        for _ in range(iterations):
            for kw in prepared:
                template.render(**kw)
        elapsed = time.perf_counter_ns() - start
        if trial >= 0:
            print(json.dumps(dict(trial=trial, iterations=iterations, renders=len(cases), bytes=len(encoded),
                                  elapsed_ns=elapsed, output_sha256=hashlib.sha256(encoded).hexdigest())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--zig", type=Path, default=ROOT / ".tools/zig-x86_64-linux-0.16.0/zig")
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--reference-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference_worker:
        reference_worker(args.config)
        return
    if args.output is None:
        p.error("--output required")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("unavailable CPU")
    args.output.mkdir(parents=True, exist_ok=False)
    config, zig = args.config.resolve(strict=True), args.zig.resolve(strict=True)
    corpus = json.loads(FIXTURE.read_text())
    if sha(config) != corpus["config_sha256"]:
        raise ValueError("config differs from reference fixture")
    output = "".join(c["official"]["output"] for c in corpus["cases"] if "output" in c["official"]).encode()
    expected = dict(renders=sum("output" in c["official"] for c in corpus["cases"]), bytes=len(output),
                    output_sha256=hashlib.sha256(output).hexdigest())
    commands = []

    def run(command):
        cmd = list(map(str, command))
        commands.append(cmd)
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        with (args.output / "commands.log").open("a") as log:
            log.write(json.dumps(cmd) + "\n" + result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    run([zig, "build", "test"])
    run([zig, "build", "test", "chat-bench-build", "-Doptimize=ReleaseFast"])
    os.sched_setaffinity(0, {args.cpu})
    observations = {"native": [], "jinja": []}
    for round_id in range(3):
        for engine in (("native", "jinja") if round_id % 2 == 0 else ("jinja", "native")):
            cmd = [ROOT / "zig-out/bin/zerv-chat-bench", FIXTURE] if engine == "native" else [
                sys.executable, Path(__file__).resolve(), "--reference-worker", "--config", config]
            raw = run(cmd)
            (args.output / f"{round_id}-{engine}.jsonl").write_text(raw)
            rows = [json.loads(line) for line in raw.splitlines()]
            validate(rows, expected)
            observations[engine].extend(r["elapsed_ns"] / (r["iterations"] * r["renders"]) for r in rows)
    summary = {engine: dict(median_ns_per_render=statistics.median(v), min_ns_per_render=min(v),
                           max_ns_per_render=max(v), stdev_ns_per_render=statistics.stdev(v), trials=len(v))
               for engine, v in observations.items()}
    summary["native_over_jinja"] = summary["native"]["median_ns_per_render"] / summary["jinja"]["median_ns_per_render"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = [ROOT / "build.zig"]
    for directory in ("src", "bench", "tools", "tests"):
        paths += sorted(path for path in (ROOT / directory).rglob("*")
                        if path.is_file() and path.suffix in (".zig", ".py", ".json", ".gguf"))
    for path in paths:
        saved = args.output / "source" / path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
    manifest = dict(host=platform.uname()._asdict(), python=sys.version, packages=run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
                    zig_version=run([zig, "version"]).strip(), zig_sha256=sha(zig),
                    binary_sha256=sha(ROOT / "zig-out/bin/zerv-chat-bench"),
                    config_sha256=sha(config), sources={str(f.relative_to(ROOT)): sha(f) for f in paths},
                    commands=commands, cpu=args.cpu, warmups=3, rounds=3, trials=7, iterations=100,
                    corpus=expected, caveat="Specialized native text-only rendering vs general Jinja interpreter; not tokenization or serving.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
