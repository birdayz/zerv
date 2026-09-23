#!/usr/bin/env python3
"""Correctness-gated native Unicode-9 NFC vs pinned HF normalizer; not serving."""
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
FIXTURE = ROOT / "tests/fixtures/nfc9/manifest.json"
ITERATIONS = {"ascii": 1000, "multilingual": 100, "marks": 10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_cases(corpus):
    result = {}
    for case in corpus["workloads"]:
        name, text, output = case["name"], case["text"].encode(), case["output"].encode()
        if name not in ITERATIONS or name in result:
            raise ValueError("invalid workload set")
        result[name] = dict(input_bytes=len(text), output_bytes=len(output),
                            output_sha256=hashlib.sha256(output).hexdigest())
    if set(result) != set(ITERATIONS):
        raise ValueError("missing workload")
    return result


def validate(rows, expected):
    if set(expected) != set(ITERATIONS) or len(rows) != 7 * len(expected):
        raise ValueError("expected seven trials for every workload")
    keys = {"workload", "trial", "iterations", "input_bytes", "output_bytes", "elapsed_ns", "output_sha256"}
    seen = set()
    for row in rows:
        if set(row) != keys or any(type(row[k]) is not int for k in keys - {"workload", "output_sha256"}):
            raise ValueError("bad schema")
        name = row["workload"]
        if not isinstance(name, str) or name not in expected:
            raise ValueError("unexpected workload")
        pair = (name, row["trial"])
        if row["trial"] not in range(7) or pair in seen or row["iterations"] != ITERATIONS[name] or row["elapsed_ns"] <= 0:
            raise ValueError("bad trial")
        seen.add(pair)
        if any(row[k] != v for k, v in expected[name].items()):
            raise ValueError("output mismatch")


def oracle():
    import tokenizers
    from tokenizers.normalizers import NFC
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("reference requires tokenizers==0.22.2")
    return tokenizers, NFC()


def reference_worker():
    _, nfc = oracle()
    corpus = json.loads(FIXTURE.read_text())
    expected = expected_cases(corpus)
    for case in corpus["workloads"]:
        name, text = case["name"], case["text"]
        if nfc.normalize_str(text) != case["output"]:
            raise ValueError("HF/golden mismatch")
        for trial in range(-3, 7):
            iterations = 1 if trial < 0 else ITERATIONS[name]
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result = nfc.normalize_str(text)
            elapsed = time.perf_counter_ns() - start
            if result != case["output"]:
                raise ValueError("HF/golden mismatch after timing")
            if trial >= 0:
                print(json.dumps(dict(workload=name, trial=trial, iterations=iterations,
                                      elapsed_ns=elapsed, **expected[name])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zig", type=Path, default=ROOT / ".tools/zig-x86_64-linux-0.16.0/zig")
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--reference-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference_worker:
        reference_worker()
        return
    if args.output is None:
        p.error("--output required")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("unavailable CPU")
    tokenizers, _ = oracle()
    args.output.mkdir(parents=True, exist_ok=False)
    zig = args.zig.resolve(strict=True)
    corpus = json.loads(FIXTURE.read_text())
    expected = expected_cases(corpus)
    for path, key in [(ROOT / "tests/reference/generate_nfc.py", "generator_sha256"),
                      (ROOT / "src/text/data/nfc9.bin", "data_sha256"),
                      (FIXTURE.parent / "cases.bin", "golden_sha256")]:
        if sha(path) != corpus[key]:
            raise ValueError(f"provenance mismatch: {path}")
    commands = []

    def run(command):
        cmd = list(map(str, command))
        commands.append(cmd)
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        with (args.output / "commands.log").open("a") as log:
            log.write(json.dumps(cmd) + "\n" + result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    run([zig, "build", "test", "--summary", "all"])
    run([zig, "build", "test", "nfc-bench-build", "-Doptimize=ReleaseFast", "--summary", "all"])
    os.sched_setaffinity(0, {args.cpu})
    observations = {name: {"native": [], "hf": []} for name in expected}
    for round_id in range(3):
        for engine in (("native", "hf") if round_id % 2 == 0 else ("hf", "native")):
            cmd = [ROOT / "zig-out/bin/zerv-nfc-bench", FIXTURE] if engine == "native" else [
                sys.executable, Path(__file__).resolve(), "--reference-worker"]
            raw = run(cmd)
            (args.output / f"{round_id}-{engine}.jsonl").write_text(raw)
            rows = [json.loads(line) for line in raw.splitlines()]
            validate(rows, expected)
            for row in rows:
                observations[row["workload"]][engine].append(row["elapsed_ns"] / row["iterations"])
    summary = {}
    for name, engines in observations.items():
        stats = {engine: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v),
                              stdev_ns=statistics.stdev(v), trials=len(v)) for engine, v in engines.items()}
        stats["native_over_hf"] = stats["native"]["median_ns"] / stats["hf"]["median_ns"]
        summary[name] = stats
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = [ROOT / "build.zig"]
    for directory in ("src", "bench", "tools", "tests"):
        paths += sorted(path for path in (ROOT / directory).rglob("*")
                        if path.is_file() and path.suffix in (".zig", ".py", ".json", ".gguf", ".bin", ".txt"))
    for path in paths:
        saved = args.output / "source" / path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
    extensions = sorted(Path(tokenizers.__file__).parent.glob("*.so"))
    if not extensions:
        raise ValueError("missing reference extension binary")
    manifest = dict(host=platform.uname()._asdict(), cpu_model=Path("/proc/cpuinfo").read_text().split("model name\t: ", 1)[1].splitlines()[0],
                    python=sys.version, packages=run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
                    oracle_extensions={str(f): sha(f) for f in extensions},
                    zig_version=run([zig, "version"]).strip(), zig_sha256=sha(zig),
                    binary_sha256=sha(ROOT / "zig-out/bin/zerv-nfc-bench"), fixture_sha256=sha(FIXTURE),
                    sources={str(f.relative_to(ROOT)): sha(f) for f in paths}, commands=commands,
                    cpu=args.cpu, warmups=3, rounds=3, trials=7, iterations=ITERATIONS, corpus=expected,
                    caveat="Native validated UTF-8/preallocated buffers vs HF Python str/allocating API; no equivalent llama-server NFC operation; not tokenization or serving.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
