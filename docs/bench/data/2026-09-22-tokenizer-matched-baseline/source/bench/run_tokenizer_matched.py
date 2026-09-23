#!/usr/bin/env python3
"""Matched direct-call zerv/libllama tokenizer comparison; no HTTP timings."""
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
sys.path.insert(0, str(ROOT))
from bench import run_tokenizer as common

REV = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
LIBRARY = Path("/usr/lib/libllama.so.0.4.1")
LIBRARY_SHA = "c352cb4b1f5456dffbc4483ba1e0be7a547b21a0f7e63462ab8fb333f51245e1"
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"


def corpus_bytes(corpus, hf):
    def field(data):
        if len(data) > 1048576:
            raise ValueError("oversized corpus field")
        return struct.pack("<I", len(data)) + data

    def record(name, case, text):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid workload name")
        ids = case["ids"]
        if len(ids) > 1048576 or any(type(id_) is not int or not 0 <= id_ < 248320 for id_ in ids):
            raise ValueError("invalid corpus IDs")
        return (field(name.encode()) + field(text.encode()) + struct.pack("<I", len(ids)) +
                struct.pack("<" + "I" * len(ids), *ids) + field(bytes.fromhex(case["raw_hex"])))

    groups = [corpus[k] for k in ("cases", "decode_cases", "workloads")]
    if any(not group or len(group) > 100000 for group in groups):
        raise ValueError("invalid corpus count")
    data = bytearray(b"ZTBC" + struct.pack("<IIII", 1, *map(len, groups)))
    normalized_count = 0
    for i, case in enumerate(groups[0]):
        text = hf.normalizer.normalize_str(case["text"])
        if hf.encode(text, add_special_tokens=False).ids != case["ids"]:
            raise ValueError("normalizing fixture changed official IDs")
        normalized_count += text != case["text"]
        data += record(f"encode-{i}", case, text)
    for i, case in enumerate(groups[1]):
        data += record(f"decode-{i}", case, "")
    names = set()
    for case in groups[2]:
        if case["name"] in names:
            raise ValueError("duplicate workload")
        names.add(case["name"])
        if hf.normalizer.normalize_str(case["text"]) != case["text"]:
            raise ValueError("timed inputs must already be NFC")
        if hf.encode(case["text"], add_special_tokens=False).ids != case["ids"]:
            raise ValueError("timed IDs differ from independent oracle")
        data += record(case["name"], case, case["text"])
    return bytes(data), normalized_count


def reference_rows(raw):
    rows = []
    for line in raw.splitlines():
        row = json.loads(line)
        if "output_sha256" in row or not isinstance(row.get("output_hex"), str):
            raise ValueError("invalid reference output")
        data = bytes.fromhex(row.pop("output_hex"))
        row["output_sha256"] = hashlib.sha256(data).hexdigest()
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--zig", type=Path, default=ROOT / ".tools/zig-x86_64-linux-0.16.0/zig")
    args = p.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        p.error("unavailable CPU")
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    host=platform.uname()._asdict(), python=sys.version, commands=[], cpu=args.cpu,
                    initial_affinity=sorted(os.sched_getaffinity(0)), warmups=3, trials=7, rounds=3,
                    iterations={k: v for k, v in common.ITERATIONS.items() if k != "http"},
                    boundary="Direct in-process: encode owned output allocation/free; decode preallocated raw buffer. Common NFC input domain. No HTTP/Python/JSON in timed loops.",
                    allocator="zerv smp_allocator; libllama C++/libc plus C owned-result malloc/free",
                    cache="warm vocab; reused fixed inputs; no native cross-request result cache")

    def run(argv):
        argv = list(map(str, argv))
        manifest["commands"].append(argv)
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        with (dest / "commands.log").open("a") as log:
            log.write(json.dumps(argv) + "\n" + result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    try:
        zig, model, config = [path.resolve(strict=True) for path in (args.zig, args.model, args.tokenizer)]
        corpus = json.loads(common.FIXTURE.read_text())
        if common.sha(config) != corpus["tokenizer_sha256"]:
            raise ValueError("tokenizer identity mismatch")
        for path, key in [(ROOT / "tests/reference/generate_tokenizer_goldens.py", "generator_sha256"),
                          (ROOT / "tests/reference/tokenizer_pieces.c", "oracle_adapter_sha256"),
                          (common.FIXTURE.parent / "model.bin", "model_data_sha256")]:
            if common.sha(path) != corpus[key]:
                raise ValueError(f"fixture provenance mismatch: {path}")
        header = Path("/usr/include/llama.h")
        if header.read_bytes() != (ROOT / f"third_party/llama.cpp/{REV}/include/llama.h").read_bytes():
            raise ValueError("reference header mismatch")
        if common.sha(LIBRARY) != LIBRARY_SHA:
            raise ValueError("reference library differs from actual llama-server")
        manifest["reference_version"] = run(["llama-server", "--version"]).strip()
        manifest["reference_library_sha256"] = common.sha(LIBRARY)
        manifest["reference_header_sha256"] = common.sha(header)
        manifest["model_sha256"] = common.sha(model)
        manifest["model_bytes"] = model.stat().st_size
        if manifest["model_sha256"] != MODEL_SHA:
            raise ValueError("model hash mismatch")
        manifest["zig_version"] = run([zig, "version"]).strip()
        if manifest["zig_version"] != (ROOT / ".zig-version").read_text().strip():
            raise ValueError("compiler version mismatch")
        manifest["zig_sha256"] = common.sha(zig)
        run([zig, "build", "test", "--summary", "all"])
        run([zig, "build", "test", "tokenizer-bench-build", "-Doptimize=ReleaseFast", "-Dcpu=native", "--summary", "all"])
        native = ROOT / "zig-out/bin/zerv-tokenizer-bench"
        # A repeatable baseline binary, not just a digest of a later-overwritten build.
        artifacts = ROOT / "third_party/tokenizer-matched" / dest.name
        artifacts.mkdir(parents=True, exist_ok=False)
        shutil.copy2(native, artifacts / "native-bench")
        native = artifacts / "native-bench"
        manifest["native_binary"] = str(native)
        manifest["native_binary_sha256"] = common.sha(native)
        manifest["native_elf"] = run(["readelf", "-d", native])
        if "NEEDED" in manifest["native_elf"]:
            raise ValueError("native benchmark acquired a dynamic dependency")
        compiler = Path(shutil.which("cc")).resolve(strict=True)
        manifest["cc_sha256"] = common.sha(compiler)
        manifest["cc_version"] = run([compiler, "--version"])
        reference = artifacts / "llama-bench"
        manifest["reference_adapter"] = str(reference)
        run([compiler, "-std=c11", "-O3", "-march=native", "-Wall", "-Wextra", "-Werror",
             ROOT / "tests/reference/tokenizer_bench.c", LIBRARY, "-o", reference])
        manifest["reference_adapter_sha256"] = common.sha(reference)
        manifest["reference_dependencies"] = run(["ldd", reference])
        # Confirm what the dynamic loader actually selects, not only the link input.
        resolved = re.search(r"libllama\.so[^ ]* => (\S+)", manifest["reference_dependencies"])
        if resolved is None or common.sha(Path(resolved[1])) != LIBRARY_SHA:
            raise ValueError("dynamic loader selected a different libllama")
        manifest["cpu_info"] = run(["lscpu"])
        packages, hf = common.oracle(config)
        manifest["hf_version"] = packages.__version__
        data, changed = corpus_bytes(corpus, hf)
        binary_corpus = dest / "corpus.bin"
        binary_corpus.write_bytes(data)
        manifest["corpus_sha256"] = common.sha(binary_corpus)
        manifest["normalized_correctness_cases"] = changed
        manifest["timed_workloads"] = [dict(name=c["name"], bytes=len(c["text"].encode()), tokens=len(c["ids"]),
                                             input_sha256=hashlib.sha256(c["text"].encode()).hexdigest()) for c in corpus["workloads"]]
        native_check = json.loads(run([native, model, common.FIXTURE]))
        wanted = dict(cases=len(corpus["cases"]), decode_cases=len(corpus["decode_cases"]),
                      pieces=corpus["raw_piece_count"], raw_pieces_sha256=corpus["raw_pieces_sha256"])
        if native_check != wanted:
            raise ValueError("incomplete native validation")
        (dest / "native-validation.json").write_text(json.dumps(native_check, indent=2) + "\n")
        reference_check = json.loads(run([reference, model, binary_corpus]))
        if reference_check != dict(cases=len(corpus["cases"]), decode_cases=len(corpus["decode_cases"]), workloads=len(corpus["workloads"])):
            raise ValueError("incomplete reference validation")
        (dest / "reference-validation.json").write_text(json.dumps(reference_check, indent=2) + "\n")
        paths = [ROOT / "build.zig", ROOT / ".zig-version"]
        for directory in ("src", "bench", "tools", "tests"):
            paths += sorted(path for path in (ROOT / directory).rglob("*") if path.is_file() and path.suffix in (".zig", ".py", ".c", ".json", ".gguf", ".bin", ".txt"))
        manifest["sources"] = {str(path.relative_to(ROOT)): common.sha(path) for path in paths}
        for path in paths:
            snapshot = dest / "source" / path.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
        os.sched_setaffinity(0, {args.cpu})
        manifest["measured_affinity"] = sorted(os.sched_getaffinity(0))
        governor = Path(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor")
        manifest["governor"] = governor.read_text().strip() if governor.exists() else None
        observations = {}
        for round_id in range(3):
            for engine in (("native", "llama") if round_id % 2 == 0 else ("llama", "native")):
                cmd = [native, model, common.FIXTURE, "--bench"] if engine == "native" else [reference, model, binary_corpus, "--bench"]
                raw = run(cmd)
                (dest / f"{round_id}-{engine}.jsonl").write_text(raw)
                rows = reference_rows(raw) if engine == "llama" else [json.loads(line) for line in raw.splitlines()]
                common.validate(rows, common.expected(corpus))
                for row in rows:
                    key = "/".join((row["workload"], row["operation"], engine))
                    observations.setdefault(key, []).append(row["elapsed_ns"] / row["iterations"])
                print(f"verified {engine} round {round_id}", flush=True)
        summary = {key: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v), stdev_ns=statistics.stdev(v), trials=len(v)) for key, v in observations.items()}
        (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = "passed"
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
