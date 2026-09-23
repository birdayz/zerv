#!/usr/bin/env python3
"""Independent warm GGUF parse/free comparison. No production reference dependency."""
import argparse
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/reference"))
from gguf_oracle import Oracle, InitParams


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate(records, container):
    if len(records) != 7:
        raise ValueError("expected exactly seven trials")
    keys = {"trial", "iterations", "elapsed_ns", "metadata", "tensors", "data_offset"}
    seen = set()
    for row in records:
        if set(row) != keys or any(type(v) is not int for v in row.values()):
            raise ValueError("invalid trial schema")
        if row["trial"] not in range(7) or row["trial"] in seen:
            raise ValueError("duplicate or invalid trial")
        seen.add(row["trial"])
        if row["iterations"] != 10 or row["elapsed_ns"] <= 0:
            raise ValueError("invalid trial timing")
        if (row["metadata"], row["tensors"], row["data_offset"]) != (
                len(container["metadata"]), len(container["tensors"]), container["data_offset"]):
            raise ValueError("container identity mismatch")


def reference_worker(model, library):
    oracle = Oracle(library)
    load = oracle.bind("gguf_init_from_buffer", C.c_void_p, [C.c_void_p, C.c_size_t, InitParams])
    count_metadata = oracle.bind("gguf_get_n_kv", C.c_int64, [C.c_void_p])
    count_tensors = oracle.bind("gguf_get_n_tensors", C.c_int64, [C.c_void_p])
    data_start = oracle.bind("gguf_get_data_offset", C.c_size_t, [C.c_void_p])
    with Path(model).open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
        byte = C.c_char.from_buffer(mapped)
        pointer = C.addressof(byte)
        try:
            for trial in range(-3, 7):
                iterations = 1 if trial < 0 else 10
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    ctx = load(pointer, len(mapped), InitParams(True, None))
                    if not ctx:
                        raise RuntimeError("reference rejected mapped container")
                    try:
                        metadata, tensors, offset = count_metadata(ctx), count_tensors(ctx), data_start(ctx)
                    finally:
                        oracle.free(ctx)
                elapsed = time.perf_counter_ns() - start
                if trial >= 0:
                    print(json.dumps(dict(trial=trial, iterations=iterations, elapsed_ns=elapsed,
                                          metadata=metadata, tensors=tensors, data_offset=offset)))
        finally:
            del byte


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zig", type=Path, default=ROOT / ".tools/zig-x86_64-linux-0.16.0/zig")
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", type=int)
    p.add_argument("--reference-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference_worker:
        reference_worker(args.model, args.library)
        return
    if args.output is None:
        p.error("--output required")
    args.output.mkdir(parents=True, exist_ok=False)
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if args.cpu is None else args.cpu
    if cpu not in allowed:
        raise ValueError("CPU not in allowed affinity")
    model, zig, library = (x.resolve(strict=True) for x in (args.model, args.zig, args.library))
    commands = []

    def run(command):
        command = list(map(str, command))
        commands.append(command)
        done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        with (args.output / "commands.log").open("a") as log:
            log.write(json.dumps(command) + "\n" + done.stdout + done.stderr)
        done.check_returncode()
        return done.stdout

    run([zig, "build", "test"])
    run([zig, "build", "test", "inspect-build", "gguf-bench-build", "-Doptimize=ReleaseFast"])
    model_hash = sha(model)
    oracle = Oracle(library)
    reference = oracle.inspect(model)
    native = json.loads(run([ROOT / "zig-out/bin/zerv-inspect", model]))
    if native != reference:
        (args.output / "mismatch.json").write_text(json.dumps(dict(native=native, reference=reference), indent=2))
        raise ValueError("native/reference container inspection mismatch")
    (args.output / "container.json").write_text(json.dumps(reference, indent=2) + "\n")
    # Build outside pinned measurement; both worker processes inherit this affinity.
    os.sched_setaffinity(0, {cpu})
    observations = {"native": [], "reference": []}
    for round_id in range(3):
        for engine in (("native", "reference") if round_id % 2 == 0 else ("reference", "native")):
            command = [ROOT / "zig-out/bin/zerv-gguf-bench", model] if engine == "native" else [
                sys.executable, Path(__file__).resolve(), "--reference-worker", "--model", model, "--library", library]
            raw = run(command)
            (args.output / f"{round_id}-{engine}.jsonl").write_text(raw)
            records = [json.loads(line) for line in raw.splitlines()]
            validate(records, reference)
            observations[engine].extend(row["elapsed_ns"] / row["iterations"] for row in records)
    summary = {engine: dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
                           stdev_ns=statistics.stdev(values), trials=len(values)) for engine, values in observations.items()}
    summary["native_over_reference"] = summary["native"]["median_ns"] / summary["reference"]["median_ns"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [Path("build.zig")]
    for directory in ("src", "bench", "tools", "tests"):
        sources += [path.relative_to(ROOT) for path in (ROOT / directory).rglob("*")
                    if path.is_file() and path.suffix in (".zig", ".py", ".json", ".gguf", ".bin", ".txt")]
    for path in sources:
        saved = args.output / "source" / path
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((ROOT / path).read_bytes())
    manifest = dict(model=dict(path=str(model), size=model.stat().st_size, sha256=model_hash),
                    oracle=oracle.identity, zig=dict(path=str(zig), sha256=sha(zig), version=run([zig, "version"]).strip()),
                    host=platform.uname()._asdict(), python=sys.version, allowed_cpus=allowed, effective_cpu=cpu,
                    warmups=3, trials=7, iterations=10, rounds=3, commands=commands,
                    sources={str(path): sha(ROOT / path) for path in sorted(sources)},
                    binaries={str(path.relative_to(ROOT)): sha(path) for path in [ROOT / "zig-out/bin/zerv-inspect", ROOT / "zig-out/bin/zerv-gguf-bench"]},
                    caveats=["Warm header/page cache; no weight reads or GPU upload timed.",
                             "Native borrows metadata and validates UTF-8 and complete payload bounds; reference copies metadata.",
                             "Native page_allocator vs reference allocator. Reference ctypes call/getter overhead included."])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
