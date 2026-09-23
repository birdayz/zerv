#!/usr/bin/env python3
"""Rebuild, check and time full real-model resident matvecs; no serving speed claim."""
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
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests/reference"))
from generate_gpu_matvec import TYPES, PINS, MODEL_SHA, Oracle, build_oracle, encode_case, metrics, run_oracle, sha
from run_gpu_driver import gpu_snapshot
from generate_vulkan_goldens import PINS as DRIVER_PINS


def timings(text, iterations):
    records = [json.loads(line) for line in text.splitlines()]
    if len(records) != 7: raise ValueError("missing trials")
    for i, r in enumerate(records):
        if set(r) != {"trial", "iterations", "elapsed_ns"} or any(type(v) is not int for v in r.values()) or r["trial"] != i or r["iterations"] != iterations or r["elapsed_ns"] <= 0:
            raise ValueError("invalid trial")
    return [r["elapsed_ns"]/iterations for r in records]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=int, default=10)
    p.add_argument("--baseline-run", type=Path, help="hash-verified rebuild_matvec_baseline.py output")
    p.add_argument("--aligned", action="store_true", help="also measure naturally aligned native views, retaining the original offset2 stress case")
    a = p.parse_args()
    if a.cpu not in os.sched_getaffinity(0): p.error("CPU unavailable")
    dest = a.output.resolve(); dest.mkdir(parents=True, exist_ok=False)
    artifact = ROOT/"third_party/gpu-matvec-bench"/dest.name
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=platform.uname()._asdict(), cpu=a.cpu,
                    initial_affinity=sorted(os.sched_getaffinity(0)), warmups=3, trials=7, rounds=3, seed="index LCG32 xor 0xa5a5a5a5, signed low11 /1024",
                    commands=[], workloads=[], checks=[], gpu_before=gpu_snapshot(), environment={k:v for k,v in os.environ.items() if k.startswith(("GGML_", "VK_", "MESA_", "RADV_", "DRI_"))},
                    boundary="native reusable submit/fence vs external one-node synchronous graph; graph internals are not equivalent host overhead",
                    precision="native and reference-f32 retain FP32 input; reference-default is separately measured Q8_1 activation quantization where selected, not matched precision",
                    alignment="native/native-baseline: weight offset2 (F32 offset4), input aligned4 after weights+16; native-aligned: weight offset0, same input placement rule; reference: ggml aligned tensor allocation")

    def run(cmd):
        cmd = list(map(str, cmd)); manifest["commands"].append(cmd)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        with (dest/"commands.log").open("a") as f: f.write(json.dumps(cmd)+"\n"+r.stdout+r.stderr)
        r.check_returncode(); return r

    try:
        if sha(a.model) != MODEL_SHA: raise ValueError("model mismatch")
        manifest["model"] = dict(path=str(a.model.resolve()), sha256=MODEL_SHA)
        for path, digest in dict(PINS, **DRIVER_PINS).items():
            if sha(Path(path)) != digest: raise ValueError("pin mismatch: "+path)
        manifest["pins"] = dict(PINS, **DRIVER_PINS)
        fixture = json.loads((ROOT/"tests/fixtures/gpu/matvec.json").read_text())
        for name, digest in fixture["sources"].items():
            if sha(ROOT/"tests/reference"/name) != digest: raise ValueError("fixture source changed")
        manifest["fixture_sha256"] = sha(ROOT/"tests/fixtures/gpu/matvec.json")
        artifact.mkdir(parents=True, exist_ok=False)
        # Rebuild shader bytes and require equality; do not silently overwrite native modules.
        run([sys.executable, ROOT/"tools/compile_matvec.py", "--output-dir", artifact/"shaders"])
        for path in (artifact/"shaders").iterdir():
            if path.read_bytes() != (ROOT/"src/matvec/shaders"/path.name).read_bytes(): raise ValueError("shader replay mismatch")
        zig = ROOT/".tools/zig-x86_64-linux-0.16.0/zig"
        manifest["zig_sha256"] = sha(zig)
        if run([zig, "version"]).stdout.strip() != "0.16.0": raise ValueError("Zig version mismatch")
        run([zig, "build", "test", "gpu-test", "--summary", "all"])
        run([zig, "build", "test", "gpu-test", "gpu-matvec-bench-build", "-Doptimize=ReleaseFast", "-Dcpu=native", "--summary", "all"])
        native = artifact/"native"; shutil.copy2(ROOT/"zig-out/bin/zerv-gpu-matvec-bench", native)
        baseline_native = None
        if a.baseline_run:
            baseline_record = a.baseline_run.resolve()/"manifest.json"
            baseline = json.loads(baseline_record.read_text())
            baseline_native = Path(baseline["native_binary"])
            if baseline["status"] != "passed" or baseline["zig_sha256"] != sha(zig) or sha(baseline_native) != baseline["native_binary_sha256"]:
                raise ValueError("baseline rebuild changed")
            for relative, digest in baseline["sources"].items():
                if sha(a.baseline_run/"source"/relative) != digest: raise ValueError("baseline source changed")
            manifest["baseline"] = dict(manifest_path=str(baseline_record), manifest_sha256=sha(baseline_record), binary_path=str(baseline_native), binary_sha256=sha(baseline_native))
        reference = build_oracle(artifact)
        shutil.copyfile(artifact/"build.json", dest/"reference-build.json")
        for name, binary in (("native", native), ("reference", reference)):
            deps = run(["ldd", binary]).stdout
            manifest[name] = dict(path=str(binary), sha256=sha(binary), dependencies=deps,
                                  dependency_hashes={path: sha(Path(path)) for path in re.findall(r"=> (/\S+)", deps)})
        needed = re.findall(r"Shared library: \[([^]]+)\]", run(["readelf", "-d", native]).stdout)
        if not needed or set(needed)-{"libvulkan.so.1", "libc.so.6", "ld-linux-x86-64.so.2"}: raise ValueError("non-system native dependency")
        manifest["vulkaninfo"] = run(["vulkaninfo", "--summary"]).stdout
        manifest["lscpu"] = run(["lscpu"]).stdout
        sources = [ROOT/"build.zig", ROOT/".zig-version"]
        for directory in ("src", "bench", "tools", "tests"):
            sources += sorted(p for p in (ROOT/directory).rglob("*") if p.is_file() and p.suffix in (".zig", ".py", ".c", ".json", ".bin", ".gguf", ".txt", ".comp", ".spv"))
        manifest["sources"] = {str(path.relative_to(ROOT)): sha(path) for path in sources}
        for path in sources:
            copy = dest/"source"/path.relative_to(ROOT); copy.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, copy)
        os.sched_setaffinity(0, {a.cpu}); manifest["measured_affinity"] = sorted(os.sched_getaffinity(0))
        governor = Path(f"/sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor")
        manifest["governor"] = governor.read_text().strip() if governor.exists() else None
        oracle = Oracle("/usr/lib/libggml-base.so.0.24.0")
        inv = oracle.inspect(a.model, samples=False)
        seen = set(); cases = []
        with a.model.open("rb") as model:
            for tensor in inv["tensors"]:
                k, rows = tensor["dims"][:2]
                if tensor["name"].startswith("blk.64.") or tensor["name"] == "token_embd.weight" or rows == 1 or k == 4: continue
                key = tensor["type"], k, rows
                if key in seen: continue
                seen.add(key)
                fmt = next(name for name, layout in TYPES.items() if layout[0] == tensor["type"])
                case = dict(name=f"{fmt}-{k}-{rows}", format=fmt, columns=k, rows=rows, tensor=tensor,
                            iterations=10 if rows > 65535 else 256 if rows <= 48 else 64)
                model.seek(inv["data_offset"]+tensor["offset"]); packed = model.read(tensor["size"])
                xv = [((((i*1664525+1013904223) & 0xffffffff) ^ 0xa5a5a5a5) & 2047)-1024 for i in range(k)]
                x = struct.pack("<"+"f"*k, *(v/1024 for v in xv))
                path = artifact/(case["name"]+".case")
                path.write_bytes(encode_case(case, packed, x))
                del packed
                case.update(case_path=str(path), input_sha256=sha(path))
                manifest["workloads"].append(case); cases.append(case)
        if len(cases) != 11: raise ValueError("unexpected real-model dense shapes")
        observations = {}
        for case in cases:
            path = Path(case["case_path"]); name = case["name"]; n = case["rows"]
            # Establish the full-row independent reference before timed native comparisons.
            golden_path = artifact/(name+"-golden.output")
            ideal, sums, gpu = run_oracle(reference, path, golden_path)
            baseline_metrics = metrics(gpu, ideal, sums)
            manifest["checks"].append(dict(case=name, engine="reference-f32-golden", metrics=baseline_metrics, output_sha256=sha(golden_path)))
            for round_id in range(3):
                order = ("native", "reference-f32", "reference-default")
                if a.aligned: order = ("native", "native-aligned", "reference-f32", "reference-default")
                if baseline_native: order = ("native-baseline", *order)
                if round_id % 2: order = tuple(reversed(order))
                for engine in order:
                    prefix = f"{name}-{round_id}-{engine}"; output = artifact/(prefix+".output")
                    if engine in ("native", "native-baseline", "native-aligned"):
                        extra = ["--aligned"] if engine == "native-aligned" else []
                        proc = run([baseline_native if engine == "native-baseline" else native, path, output, case["iterations"], *extra])
                        if "AMD Radeon RX 7900 XTX" not in proc.stderr: raise ValueError("wrong native device")
                        output.with_suffix(".stdout").write_text(proc.stdout); output.with_suffix(".stderr").write_text(proc.stderr)
                        raw = output.read_bytes()
                        if len(raw) != n*4: raise ValueError("wrong native output shape")
                        actual = struct.unpack("<"+"f"*n, raw)
                    else:
                        dots, absums, actual = run_oracle(reference, path, output, case["iterations"], default=engine.endswith("default"))
                        if dots != ideal or absums != sums: raise ValueError("CPU baseline changed")
                    values = timings(output.with_suffix(".stdout").read_text(), case["iterations"])
                    stats = metrics(actual, ideal, sums, enforce=engine != "reference-default")
                    record = dict(case=name, round=round_id, engine=engine, metrics=stats, output_sha256=sha(output), matched_fp32=engine != "reference-default")
                    manifest["checks"].append(record)
                    for suffix in (".stdout", ".stderr"):
                        shutil.copyfile(output.with_suffix(suffix), dest/(prefix+suffix))
                    observations.setdefault(name+"/"+engine, []).extend(values)
                    print("verified", prefix, "median_us", statistics.median(values)/1000, "max_error_over_sumabs", stats["max_error_over_sumabs"], flush=True)
        summary = {key: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v), stdev_ns=statistics.stdev(v), trials=len(v)) for key, v in observations.items()}
        (dest/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        manifest["artifact_hashes"] = {str(p): sha(p) for p in sorted(artifact.iterdir()) if p.is_file()}
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=str(error)); raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["gpu_after"] = gpu_snapshot()
        (dest/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__": main()
