#!/usr/bin/env python3
"""Rebuild/verify native and independent C Vulkan driver work; matched submit/fence loops."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
sys.path.insert(0, str(ROOT / "tests/reference"))
from generate_vulkan_goldens import PINS

WORKLOADS = {("affine", 65): 1000, ("affine", 5120): 1000, ("affine", 1048576): 100,
             ("roundtrip", 256): 1000, ("roundtrip", 1048576): 100, ("roundtrip", 67108864): 10}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate(records, goldens, timed):
    expected = {(c["kind"], c["count"]): c for c in goldens["cases"] if not timed or (c["kind"], c["count"]) in WORKLOADS}
    unseen = {(k, n, trial) for k, n in WORKLOADS for trial in range(7)} if timed else set()
    timings = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid record")
        if record.get("kind") == "timing":
            if set(record) != {"kind", "workload", "count", "trial", "iterations", "elapsed_ns"} or any(type(record[k]) is not int for k in ("count", "trial", "iterations", "elapsed_ns")):
                raise ValueError("invalid timing schema")
            key = record["workload"], record["count"], record["trial"]
            if key not in unseen or record["iterations"] != WORKLOADS[key[:2]] or record["elapsed_ns"] <= 0:
                raise ValueError("unexpected/duplicate/invalid timing")
            unseen.remove(key); timings.append(record)
        else:
            if type(record.get("count")) is not int or type(record.get("bytes")) is not int:
                raise ValueError("invalid correctness schema")
            key = record.get("kind"), record.get("count")
            if key not in expected or record != expected[key]:
                raise ValueError("independent correctness mismatch/duplicate")
            del expected[key]
    if expected or unseen:
        raise ValueError("missing checks/trials")
    return timings


def metadata(stderr):
    lines = stderr.strip().splitlines()
    if not lines or not lines[0].startswith("device="):
        raise ValueError("missing device identity")
    allocations = {}
    for line in lines[1:]:
        if not line.startswith("allocation "):
            raise ValueError("unexpected driver diagnostics")
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        if set(fields) != {"kind", "count", "bytes", "allocation_sizes", "memory_types"}:
            raise ValueError("invalid allocation record")
        key = fields["kind"], int(fields["count"])
        sizes = [int(x) for x in fields["allocation_sizes"].split(",")]
        types = [int(x) for x in fields["memory_types"].split(",")]
        if key in allocations or len(sizes) != 4 or len(types) != 4 or any(s < int(fields["bytes"]) for s in sizes) or any(not 0 <= t < 32 for t in types):
            raise ValueError("invalid/duplicate allocation layout")
        allocations[key] = fields
    return lines[0], allocations


def gpu_snapshot():
    result = {}
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if re.fullmatch(r"card[0-9]+", device.parent.name) is None:
            continue
        fields = {}
        for name in ("vendor", "device", "gpu_busy_percent", "mem_info_vram_total", "mem_info_vram_used", "mem_info_gtt_used"):
            p = device / name
            if p.exists():
                fields[name] = p.read_text().strip()
        for sensor in device.glob("hwmon/hwmon*/temp1_input"):
            fields[str(sensor.relative_to(device))] = sensor.read_text().strip()
        result[str(device)] = fields
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=int, default=10)
    a = p.parse_args()
    if a.cpu not in os.sched_getaffinity(0):
        p.error("CPU unavailable")
    dest = a.output.resolve(); dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    host=platform.uname()._asdict(), python=sys.version, commands=[], cpu=a.cpu,
                    initial_affinity=sorted(os.sched_getaffinity(0)), warmups=3, trials=7, rounds=3,
                    budget_bytes=512*1024*1024, boundary="pre-recorded commands, in-process submit+finite fence wait; no allocation/compilation/host fill/hash inside timing",
                    correctness="full output including sentinel tail after timed workload; independent scalar golden hashes; no per-call readback for affine",
                    gpu_before=None)

    def run(command):
        command = list(map(str, command)); manifest["commands"].append(command)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        with (dest / "commands.log").open("a") as log:
            log.write(json.dumps(command)+"\n"+result.stdout+result.stderr)
        result.check_returncode()
        return result

    try:
        manifest["gpu_before"] = gpu_snapshot()
        for path, digest in PINS.items():
            if sha(path) != digest:
                raise ValueError("tool/driver identity changed: " + path)
        manifest["tool_driver_hashes"] = PINS
        for entry in json.loads((ROOT / "docs/research/2026-09-22/vulkan-sources.json").read_text()):
            if sha(ROOT / entry["local_path"]) != entry["sha256"]:
                raise ValueError("research header/spec changed")
        goldens = json.loads((ROOT / "tests/fixtures/gpu/dispatch.json").read_text())
        if sha(ROOT / "tests/fixtures/gpu/affine.spv") != goldens["shader_sha256"] or sha(ROOT / "tests/reference/vulkan_driver.c") != goldens["reference_source_sha256"]:
            raise ValueError("shader/reference differs from independently validated fixture")
        manifest["fixture_sha256"] = sha(ROOT / "tests/fixtures/gpu/dispatch.json")
        manifest["shader_sha256"] = goldens["shader_sha256"]
        manifest.update(zerv_build.provenance())
        run(zerv_build.test_command(*zerv_build.GPU_TESTS))
        run(zerv_build.build_command("zerv-gpu-driver-bench"))
        artifact = ROOT / "third_party/gpu-driver-bench" / dest.name
        artifact.mkdir(parents=True, exist_ok=False)
        native, reference = artifact / "native", artifact / "reference"
        shutil.copy2(zerv_build.path("zerv-gpu-driver-bench"), native)
        cc = Path(shutil.which("cc")).resolve(strict=True)
        manifest["cc_sha256"], manifest["cc_version"] = sha(cc), run([cc, "--version"]).stdout
        run([cc, "-std=c11", "-O3", "-march=native", "-Wall", "-Wextra", "-Werror",
             "-I"+str(ROOT / "third_party/vulkan/1.4.354/include"), ROOT / "tests/reference/vulkan_driver.c", "-lvulkan", "-lcrypto", "-o", reference])
        manifest["native_sha256"], manifest["reference_sha256"] = sha(native), sha(reference)
        manifest["native_elf"] = run(["readelf", "-d", native]).stdout
        needed = re.findall(r"Shared library: \[([^]]+)\]", manifest["native_elf"])
        if not needed or set(needed)-{"libvulkan.so.1", "libc.so.6", "ld-linux-x86-64.so.2"}:
            raise ValueError("native acquired a non-system runtime dependency")
        for engine, binary in (("native", native), ("reference", reference)):
            deps = run(["ldd", binary]).stdout
            manifest[engine+"_dependencies"] = deps
            manifest[engine+"_dependency_hashes"] = {path: sha(path) for path in re.findall(r"=> (/\S+)", deps)}
        manifest["vulkaninfo"] = run(["vulkaninfo", "--summary"]).stdout
        (dest / "vulkaninfo.txt").write_text(run(["vulkaninfo"]).stdout)
        manifest["lscpu"] = run(["lscpu"]).stdout
        sources = zerv_build.build_files()
        for directory in ("src", "bench", "tools", "tests"):
            sources += sorted(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in (".zig", ".py", ".c", ".json", ".bin", ".gguf", ".txt", ".comp", ".spv"))
        manifest["sources"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
        for path in sources:
            copy = dest / "source" / path.relative_to(ROOT); copy.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, copy)
        os.sched_setaffinity(0, {a.cpu}); manifest["measured_affinity"] = sorted(os.sched_getaffinity(0))
        governor = Path(f"/sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor")
        manifest["governor"] = governor.read_text().strip() if governor.exists() else None
        module = ROOT / "tests/fixtures/gpu/affine.spv"
        devices, allocations = {}, {}
        for engine, command in (("native", [native]), ("reference", [reference, module])):
            result = run(command); (dest / f"{engine}-checks.jsonl").write_text(result.stdout)
            validate([json.loads(line) for line in result.stdout.splitlines()], goldens, False)
            devices[engine], allocations[engine] = metadata(result.stderr)
        if devices["native"] != devices["reference"] or allocations["native"] != allocations["reference"]:
            raise ValueError("native/reference device/queue/allocation mismatch")
        if set(allocations["native"]) != {(c["kind"], c["count"]) for c in goldens["cases"]}:
            raise ValueError("missing allocation metadata")
        manifest["selected_devices"] = devices
        manifest["allocations"] = list(allocations["native"].values())
        observations = {}
        for round_id in range(3):
            for engine in (("native", "reference") if round_id % 2 == 0 else ("reference", "native")):
                result = run([native, "--bench"] if engine == "native" else [reference, module, "--bench"])
                identity, layout = metadata(result.stderr)
                if identity != devices[engine] or layout != {key: value for key, value in allocations[engine].items() if key in WORKLOADS}:
                    raise ValueError("device/queue/allocations changed during timing")
                (dest / f"{round_id}-{engine}.jsonl").write_text(result.stdout)
                records = validate([json.loads(line) for line in result.stdout.splitlines()], goldens, True)
                for record in records:
                    observations.setdefault(f'{record["workload"]}/{record["count"]}/{engine}', []).append(record["elapsed_ns"] / record["iterations"])
                print("verified", engine, round_id, flush=True)
        summary = {k: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v), stdev_ns=statistics.stdev(v), trials=len(v)) for k, v in observations.items()}
        (dest / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        manifest["status"] = "passed"
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["gpu_after"] = gpu_snapshot()
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
