#!/usr/bin/env python3
"""Rebuild, verify and compare the native CPU decoder with an external ggml oracle."""
import argparse
import ctypes
from datetime import datetime, timezone
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
VALUES = 5120 * 2048
ITERATIONS = 16
TRIALS = 5
ROUNDS = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_reference(library, fmt):
    width = 18 if fmt == "q4_0" else 34
    encoded = bytearray(VALUES // 32 * width)
    for block in range(VALUES // 32):
        offset = block * width
        struct.pack_into("<H", encoded, offset, 0x3000 + block % 0x1000)
        encoded[offset + 2:offset + width] = bytes((block * 37 + j * 19) % 256 for j in range(width - 2))
    source = (ctypes.c_char * len(encoded)).from_buffer(encoded)
    output = (ctypes.c_float * VALUES)()
    if ctypes.addressof(source) % 2 or ctypes.addressof(output) % 4:
        raise RuntimeError("unaligned oracle storage")
    function = getattr(library, "dequantize_row_" + fmt)
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    function.restype = None
    function(source, output, VALUES)
    expected = {"input_sha256": hashlib.sha256(encoded).hexdigest(), "output_sha256": hashlib.sha256(output).hexdigest()}
    return function, source, output, expected


def reference_trials(cases):
    records = []
    for fmt, (function, source, output, expected) in cases.items():
        for _ in range(3):
            function(source, output, VALUES)
        for trial in range(TRIALS):
            start = time.perf_counter_ns()
            for _ in range(ITERATIONS):
                function(source, output, VALUES)
            elapsed = time.perf_counter_ns() - start
            records.append({"format": fmt, "trial": trial, "iterations": ITERATIONS,
                            "values_per_call": VALUES, "elapsed_ns": elapsed,
                            "input_sha256": expected["input_sha256"],
                            "output_sha256": hashlib.sha256(output).hexdigest()})
    return records


def validate(records, cases):
    required = {(fmt, trial) for fmt in cases for trial in range(TRIALS)}
    if len(records) != len(required):
        raise RuntimeError("incorrect trial count")
    for row in records:
        if set(row) != {"format", "trial", "iterations", "values_per_call", "elapsed_ns", "input_sha256", "output_sha256"}:
            raise RuntimeError("invalid result schema")
        if any(type(row[name]) is not int for name in ("trial", "iterations", "values_per_call", "elapsed_ns")):
            raise RuntimeError("non-integer timing/count field")
        key = (row["format"], row["trial"])
        if key not in required:
            raise RuntimeError("duplicate or unexpected trial")
        required.remove(key)
        if row["iterations"] != ITERATIONS or row["values_per_call"] != VALUES:
            raise RuntimeError("workload mismatch")
        if type(row["elapsed_ns"]) is not int or row["elapsed_ns"] <= 0:
            raise RuntimeError("invalid elapsed time")
        for name, digest in cases[row["format"]][3].items():
            if row[name] != digest:
                raise RuntimeError(f"{row['format']} {name} differs from external oracle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.byteorder != "little" or ctypes.sizeof(ctypes.c_float) != 4:
        parser.error("requires little-endian IEEE binary32 C floats")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU is outside this process's allowed affinity")
    reference = args.reference.resolve(strict=True)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv, "cwd": str(ROOT), "platform": platform.platform(), "python": platform.python_version(),
                "cpu": args.cpu, "initial_affinity": sorted(os.sched_getaffinity(0)),
                "values_per_call": VALUES, "iterations": ITERATIONS, "trials": TRIALS, "rounds": ROUNDS,
                "warmup_calls": 3, "cache_policy": "reused buffers; warm; no cache flushing", "commands": [],
                "reference_path": str(reference), "reference_sha256": sha(reference)}

    def command(name, argv):
        manifest["commands"].append(argv)
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        (destination / (name + ".stdout")).write_text(completed.stdout)
        (destination / (name + ".stderr")).write_text(completed.stderr)
        completed.check_returncode()
        return completed.stdout

    try:
        manifest.update(zerv_build.provenance())
        command("test", zerv_build.test_command())
        command("build-bench", zerv_build.build_command("zerv-quant-bench"))
        native = zerv_build.path("zerv-quant-bench")
        manifest["native_sha256"] = sha(native)
        paths = zerv_build.build_files()
        for directory in ("src", "bench", "tests"):
            paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        manifest["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
        command("cpu", ["lscpu"])
        command("git-status", ["git", "status", "--short"])
        command("reference-dependencies", ["ldd", str(reference)])
        os.sched_setaffinity(0, {args.cpu})
        manifest["measured_affinity"] = sorted(os.sched_getaffinity(0))
        governor = Path(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor")
        manifest["governor"] = governor.read_text().strip() if governor.exists() else None
        library = ctypes.CDLL(str(reference))
        for symbol in ("ggml_version", "ggml_commit"):
            function = getattr(library, symbol)
            function.argtypes, function.restype = [], ctypes.c_char_p
            manifest[symbol] = function().decode()
        cases = {fmt: prepare_reference(library, fmt) for fmt in ("q4_0", "q8_0")}
        manifest["workload_hashes"] = {fmt: case[3] for fmt, case in cases.items()}
        results = []
        for round_index in range(ROUNDS):
            order = ("native", "reference") if round_index % 2 == 0 else ("reference", "native")
            for engine in order:
                if engine == "native":
                    raw = command(f"native-{round_index}", [str(native)])
                    records = [json.loads(line) for line in raw.splitlines()]
                else:
                    records = reference_trials(cases)
                    (destination / f"reference-{round_index}.json").write_text(json.dumps(records, indent=2) + "\n")
                validate(records, cases)
                for record in records:
                    results.append(dict(record, engine=engine, round=round_index))
                print("verified", engine, "round", round_index, flush=True)
        (destination / "trials.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
        summary = {}
        for engine in ("native", "reference"):
            for fmt in cases:
                samples = [row["elapsed_ns"] / ITERATIONS / 1e6 for row in results if row["engine"] == engine and row["format"] == fmt]
                summary[engine + "/" + fmt] = {"samples": len(samples), "median_ms_per_call": statistics.median(samples),
                                               "min_ms_per_call": min(samples), "max_ms_per_call": max(samples),
                                               "stdev_ms_per_call": statistics.stdev(samples)}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
