#!/usr/bin/env python3
"""Matched actual-Qwen quant decoder timings; external C oracle loops, no Python timing."""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
import host_info  # noqa: E402  (tools/host_info.py: the host, recorded without host tools)
sys.path.insert(0, str(ROOT / "tests/reference"))
import gguf_oracle  # noqa: E402
from gguf_oracle import Oracle  # noqa: E402

ROWS = {1: 10000, 64: 256, 5120: 4}
PROFILES = {
    "q4_1": dict(type=3, columns=17408, block_elements=32, block_bytes=20, tensor="blk.0.ffn_down.weight", decoder="dequantize_row_q4_1"),
    "q5_k": dict(type=13, columns=6144, block_elements=256, block_bytes=176, tensor="blk.0.ssm_out.weight", decoder="dequantize_row_q5_K"),
    "q6_k": dict(type=14, columns=5120, block_elements=256, block_bytes=210, tensor="output.weight", decoder="dequantize_row_q6_K", tensor_rows=248320, workloads={1: 10000, 64: 256, 4096: 4}),
}
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate(records, checks, fmt="q4_1"):
    columns = PROFILES[fmt]["columns"]
    workloads = PROFILES[fmt].get("workloads", ROWS)
    fields = {"format", "rows", "values", "trial", "iterations", "elapsed_ns", "input_sha256", "output_sha256"}
    unseen = {(rows, trial) for rows in workloads for trial in range(7)}
    if len(records) != len(unseen):
        raise ValueError("incorrect trial count")
    for r in records:
        if set(r) != fields or any(type(r[k]) is not int for k in ("rows", "values", "trial", "iterations", "elapsed_ns")):
            raise ValueError("invalid record schema")
        key = r["rows"], r["trial"]
        if r["format"] != fmt or key not in unseen or r["elapsed_ns"] <= 0:
            raise ValueError("invalid/duplicate trial")
        unseen.remove(key)
        if r["values"] != columns * r["rows"] or r["iterations"] != workloads[r["rows"]]:
            raise ValueError("wrong shape/iterations")
        if any(r[k] != value for k, value in checks[r["rows"]].items()):
            raise ValueError("independent output/input mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--format", choices=PROFILES, default="q4_1")
    p.add_argument("--library", type=Path, help="ggml library for the Python oracle; default: built from source (@ggml)")
    p.add_argument("--cpu", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    profile = PROFILES[a.format]
    workloads = profile.get("workloads", ROWS)
    tensor_rows = profile.get("tensor_rows", 5120)
    columns, block_elements, block_bytes = [profile[k] for k in ("columns", "block_elements", "block_bytes")]
    if a.cpu not in os.sched_getaffinity(0):
        p.error("unavailable CPU")
    dest = a.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    host=platform.uname()._asdict(), python=sys.version, commands=[], cpu=a.cpu,
                    initial_affinity=sorted(os.sched_getaffinity(0)), warmups=3, trials=7, rounds=3,
                    format=a.format, workloads=workloads, cache="warm readonly model input and reused output; no cache flushing",
                    boundary="in-process synchronous row decode; all allocations/hash/logging outside timing; native additionally prevalidates finite fields")

    def run(cmd):
        cmd = list(map(str, cmd)); manifest["commands"].append(cmd)
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        with (dest / "commands.log").open("a") as log:
            log.write(json.dumps(cmd) + "\n" + result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    try:
        model = a.model.resolve(strict=True)
        library, manifest["library_build"] = gguf_oracle.library(a.library)
        if sys.byteorder != "little" or C.sizeof(C.c_float) != 4:
            raise ValueError("unsupported oracle host")
        manifest["model_sha256"], manifest["library_sha256"] = sha(model), sha(library)
        if manifest["model_sha256"] != MODEL_SHA:
            raise ValueError("model identity mismatch")
        manifest.update(zerv_build.provenance())
        run(zerv_build.test_command())
        run(zerv_build.build_command("zerv-model-quant-bench"))
        artifacts = ROOT / "third_party/model-quant-bench" / dest.name
        artifacts.mkdir(parents=True, exist_ok=False)
        native = artifacts / "native"
        shutil.copy2(zerv_build.path("zerv-model-quant-bench"), native)
        manifest["native_sha256"] = sha(native)
        if host_info.elf_dynamic(native) is not None:
            raise ValueError("native acquired a dynamic dependency")
        # The reference: ggml (the same pinned source as the library) linked in statically.
        built, manifest["reference"] = zerv_build.oracle("oracle_model_quant_bench")
        reference = artifacts / "reference"
        shutil.copy2(built, reference)
        manifest["reference_sha256"] = sha(reference)
        manifest["reference_dependencies"] = host_info.library_record(reference)
        if any(name.startswith("libggml") for name in manifest["reference_dependencies"]):
            raise ValueError("the reference loads a ggml library")
        manifest["cpu_info"] = host_info.cpu()
        oracle = Oracle(library)
        manifest["oracle"] = oracle.identity
        inventory = oracle.inspect(model, samples=False)
        tensor = next(t for t in inventory["tensors"] if t["name"] == profile["tensor"])
        if tensor["type"] != profile["type"] or tensor["dims"] != [columns, tensor_rows, 1, 1] or tensor["size"] != columns * tensor_rows // block_elements * block_bytes:
            raise ValueError("unexpected benchmark tensor")
        offset = inventory["data_offset"] + tensor["offset"]
        manifest["tensor"], manifest["absolute_offset"] = tensor, offset
        decode = oracle.bind(profile["decoder"], None, [C.c_void_p, C.POINTER(C.c_float), C.c_int64])
        checks = {}
        with model.open("rb") as f:
            for rows in workloads:
                values = columns * rows
                f.seek(offset)
                encoded = f.read(values // block_elements * block_bytes)
                if len(encoded) != values // block_elements * block_bytes:
                    raise ValueError("short model read")
                source = C.create_string_buffer(encoded)
                output = (C.c_float * values)()
                if C.addressof(source) % 2 or C.addressof(output) % 4:
                    raise ValueError("unaligned oracle")
                decode(source, output, values)
                checks[rows] = dict(input_sha256=hashlib.sha256(encoded).hexdigest(), output_sha256=hashlib.sha256(output).hexdigest())
                del source, output, encoded
        manifest["independent_hashes"] = checks
        paths = zerv_build.build_files()
        for directory in ("src", "bench", "tools", "tests"):
            paths += sorted(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in (".zig", ".py", ".c", ".json", ".bin", ".gguf", ".txt"))
        manifest["sources"] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        for path in paths:
            snapshot = dest / "source" / path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
        os.sched_setaffinity(0, {a.cpu})
        manifest["measured_affinity"] = sorted(os.sched_getaffinity(0))
        governor = Path(f"/sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor")
        manifest["governor"] = governor.read_text().strip() if governor.exists() else None
        observations = {}
        for round_id in range(3):
            for engine in (("native", "ggml") if round_id % 2 == 0 else ("ggml", "native")):
                raw = run([native, model, tensor["name"]] if engine == "native" else [reference, model, str(offset), a.format])
                (dest / f"{round_id}-{engine}.jsonl").write_text(raw)
                records = [json.loads(line) for line in raw.splitlines()]
                validate(records, checks, a.format)
                for row in records:
                    observations.setdefault(f'{row["rows"]}/{engine}', []).append(row["elapsed_ns"] / row["iterations"])
                print("verified", engine, round_id, flush=True)
        summary = {k: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v), stdev_ns=statistics.stdev(v), trials=len(v)) for k, v in observations.items()}
        (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
