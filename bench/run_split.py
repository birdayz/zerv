#!/usr/bin/env python3
"""Correctness-gated native Qwen regex splitting vs HF Split; not BPE or serving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
FIXTURE = ROOT / "tests/fixtures/tokenizer-split/manifest.json"
ITERATIONS = {"ascii": 100, "multilingual": 100, "whitespace": 100}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_cases(corpus):
    result = {}
    for case in corpus["workloads"]:
        name, text = case["name"], case["text"].encode()
        output = struct.pack("<" + "I" * len(case["ends"]), *case["ends"])
        if name not in ITERATIONS or name in result:
            raise ValueError("invalid workload set")
        result[name] = dict(input_bytes=len(text), pieces=len(case["ends"]),
                            output_sha256=hashlib.sha256(output).hexdigest())
    if set(result) != set(ITERATIONS):
        raise ValueError("missing workload")
    return result


def validate(rows, expected):
    if set(expected) != set(ITERATIONS) or len(rows) != 7 * len(expected):
        raise ValueError("expected seven trials for every workload")
    keys = {"workload", "trial", "iterations", "input_bytes", "pieces", "elapsed_ns", "output_sha256"}
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
    from tokenizers import Regex, pre_tokenizers
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("reference requires tokenizers==0.22.2")
    pattern = json.loads(FIXTURE.read_text())["pattern"]
    return tokenizers, pre_tokenizers.Split(Regex(pattern), "isolated")


def checked_ends(result, text):
    if "".join(piece for piece, _ in result) != text:
        raise ValueError("non-covering reference split")
    end, ends = 0, []
    for piece, _ in result:
        if not piece:
            raise ValueError("empty reference piece")
        end += len(piece.encode())
        ends.append(end)
    return ends


def reference_worker():
    _, splitter = oracle()
    corpus = json.loads(FIXTURE.read_text())
    expected = expected_cases(corpus)
    for case in corpus["workloads"]:
        name, text = case["name"], case["text"]
        if checked_ends(splitter.pre_tokenize_str(text), text) != case["ends"]:
            raise ValueError("HF/golden mismatch")
        for trial in range(-3, 7):
            iterations = 1 if trial < 0 else ITERATIONS[name]
            start = time.perf_counter_ns()
            for _ in range(iterations):
                result = splitter.pre_tokenize_str(text)
            elapsed = time.perf_counter_ns() - start
            if checked_ends(result, text) != case["ends"]:
                raise ValueError("HF/golden mismatch after timing")
            if trial >= 0:
                print(json.dumps(dict(workload=name, trial=trial, iterations=iterations,
                                      elapsed_ns=elapsed, **expected[name])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    corpus = json.loads(FIXTURE.read_text())
    expected = expected_cases(corpus)
    for path, key in [(ROOT / "tests/reference/generate_split_goldens.py", "generator_sha256"),
                      (ROOT / "src/tokenizer/data/classes16.bin", "data_sha256"),
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

    run(zerv_build.test_command())
    run(zerv_build.build_command("zerv-split-bench"))
    native = zerv_build.path("zerv-split-bench")
    os.sched_setaffinity(0, {args.cpu})
    observations = {name: {"native": [], "hf": []} for name in expected}
    for round_id in range(3):
        for engine in (("native", "hf") if round_id % 2 == 0 else ("hf", "native")):
            cmd = [native, FIXTURE] if engine == "native" else [
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
    paths = zerv_build.build_files()
    for directory in ("src", "bench", "tools", "tests"):
        paths += sorted(path for path in (ROOT / directory).rglob("*")
                        if path.is_file() and path.suffix in (".zig", ".py", ".c", ".json", ".gguf", ".bin", ".txt"))
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
                    **zerv_build.provenance(),
                    binary_sha256=sha(native), fixture_sha256=sha(FIXTURE),
                    sources={str(f.relative_to(ROOT)): sha(f) for f in paths}, commands=commands,
                    cpu=args.cpu, warmups=3, rounds=3, trials=7, iterations=ITERATIONS, corpus=expected,
                    caveat="Native validated UTF-8/borrowed slices vs HF Python str/allocated strings and offsets; no equivalent llama-server isolated-split endpoint; not BPE or serving.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
