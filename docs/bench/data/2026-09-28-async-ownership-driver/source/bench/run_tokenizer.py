#!/usr/bin/env python3
"""Full native tokenizer gate, HF component timings and actual llama-server HTTP calls."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)
FIXTURE = ROOT / "tests/fixtures/tokenizer/manifest.json"
ITERATIONS = {"encode": 100, "decode": 1000, "http": 20}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(struct.pack("<" + "I" * len(ids), *ids)).hexdigest()


def expected(corpus, http=False):
    result = {}
    for case in corpus["workloads"]:
        result[case["name"], "encode"] = ids_hash(case["ids"])
        if not http:
            result[case["name"], "decode"] = hashlib.sha256(bytes.fromhex(case["raw_hex"])).hexdigest()
    return result


def validate(rows, checks, http=False):
    keys = {"workload", "operation", "trial", "iterations", "elapsed_ns", "output_sha256"}
    if len(rows) != 7 * len(checks):
        raise ValueError("incomplete trial set")
    seen = set()
    for row in rows:
        if set(row) != keys or any(type(row[k]) is not int for k in ("trial", "iterations", "elapsed_ns")):
            raise ValueError("invalid row schema")
        if not isinstance(row["workload"], str) or not isinstance(row["operation"], str):
            raise ValueError("invalid workload/operation")
        pair = row["workload"], row["operation"]
        key = pair + (row["trial"],)
        if pair not in checks or row["trial"] not in range(7) or key in seen or row["elapsed_ns"] <= 0:
            raise ValueError("unexpected/duplicate/invalid trial")
        seen.add(key)
        if row["output_sha256"] != checks[pair] or row["iterations"] != ITERATIONS["http" if http else pair[1]]:
            raise ValueError("incorrect result/iterations")


def oracle(path):
    import tokenizers
    from tokenizers import Tokenizer
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("requires tokenizers==0.22.2")
    return tokenizers, Tokenizer.from_file(str(path))


def connection(url):
    p = urlsplit(url)
    if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost", "::1") or p.username or p.path not in ("", "/") or p.query or p.fragment:
        raise ValueError("server must be a loopback HTTP origin")
    return http.client.HTTPConnection(p.hostname, p.port or 80, timeout=120)


def request(conn, body):
    conn.request("POST", "/tokenize", body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    data = response.read()
    if response.status != 200:
        raise ValueError(f"server HTTP {response.status}: {data[:1000]!r}")
    return json.loads(data)["tokens"]


def body(text, pieces=False):
    return json.dumps(dict(content=text, add_special=False, parse_special=True, with_pieces=pieces), ensure_ascii=False).encode()


def reference_worker(args, corpus):
    http = args.http_worker
    conn = connection(args.server) if http else None
    hf = None if http else oracle(args.tokenizer)[1]
    try:
        for case in corpus["workloads"]:
            text, ids = case["text"], case["ids"]
            prepared = body(text)
            for operation in (("encode",) if http else ("encode", "decode")):
                def perform():
                    if http:
                        return request(conn, prepared)
                    return hf.encode(text, add_special_tokens=False).ids if operation == "encode" else hf.decode(ids, skip_special_tokens=False).encode()
                want = ids if operation == "encode" else bytes.fromhex(case["raw_hex"])
                if perform() != want:
                    raise ValueError("reference workload mismatch")
                for trial in range(-3, 7):
                    iterations = 1 if trial < 0 else ITERATIONS["http" if http else operation]
                    start = time.perf_counter_ns()
                    for _ in range(iterations):
                        result = perform()
                    elapsed = time.perf_counter_ns() - start
                    if result != want:
                        raise ValueError("reference output changed")
                    if trial >= 0:
                        print(json.dumps(dict(workload=case["name"], operation=operation, trial=trial, iterations=iterations,
                                              elapsed_ns=elapsed, output_sha256=expected(corpus, http)[case["name"], operation])))
    finally:
        if conn:
            conn.close()


def compare_server(url, hf, corpus):
    conn = connection(url)
    mismatches, total, unexpected = [], 0, 0
    try:
        for index, case in enumerate(corpus["cases"]):
            tokens = request(conn, body(case["text"], True))
            ids = [t["id"] for t in tokens]
            raw = b"".join(t["piece"].encode() if isinstance(t["piece"], str) else bytes(t["piece"]) for t in tokens)
            if raw != case["text"].encode():
                raise ValueError(f"server pieces failed to reconstruct input at {index}")
            if ids != case["ids"]:
                normalized = hf.normalizer.normalize_str(case["text"])
                normalized_ids = request(conn, body(normalized)) if normalized != case["text"] else ids
                if normalized == case["text"] or normalized_ids != case["ids"]:
                    unexpected += 1
                mismatches.append(dict(index=index, text=case["text"], official=case["ids"], llama=ids,
                                       nfc_changes_input=normalized != case["text"], llama_after_nfc=normalized_ids,
                                       matches_after_nfc=normalized_ids == case["ids"]))
            total += 1
        return dict(cases=total, matched=total-len(mismatches), mismatches=mismatches, unexpected=unexpected,
                    note="llama does not NFC-normalize; raw server pieces reconstruct every original input")
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:18081")
    p.add_argument("--server-record", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--reference-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--http-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    corpus = json.loads(FIXTURE.read_text())
    if args.reference_worker or args.http_worker:
        reference_worker(args, corpus)
        return
    if args.output is None or args.model is None or args.server_record is None:
        p.error("--output, --model and --server-record required")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("unavailable CPU")
    args.output.mkdir(parents=True, exist_ok=False)
    model, config = [p.resolve(strict=True) for p in (args.model, args.tokenizer)]
    if sha(config) != corpus["tokenizer_sha256"]:
        raise ValueError("tokenizer identity mismatch")
    for path, key in [(ROOT / "tests/reference/generate_tokenizer_goldens.py", "generator_sha256"),
                      (ROOT / "tests/reference/tokenizer_pieces.c", "oracle_adapter_sha256"),
                      (FIXTURE.parent / "model.bin", "model_data_sha256")]:
        if sha(path) != corpus[key]:
            raise ValueError(f"provenance mismatch: {path}")
    packages, hf = oracle(config)
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
    run(zerv_build.build_command("zerv-tokenizer-bench"))
    native = zerv_build.path("zerv-tokenizer-bench")
    raw = run([native, model, FIXTURE])
    (args.output / "native-validation.json").write_text(raw)
    native_check = json.loads(raw)
    if native_check != dict(cases=len(corpus["cases"]), decode_cases=len(corpus["decode_cases"]), pieces=corpus["raw_piece_count"], raw_pieces_sha256=corpus["raw_pieces_sha256"]):
        raise ValueError("incomplete native verification")
    comparison = compare_server(args.server, hf, corpus)
    (args.output / "server-comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    if comparison["unexpected"]:
        raise ValueError("unexpected server differences; inspect server-comparison.json before timing")
    model_hash = sha(model)
    if model_hash != "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d":
        raise ValueError("model hash mismatch")
    os.sched_setaffinity(0, {args.cpu})
    observations = {}
    for round_id in range(3):
        for engine in (("native", "hf", "http") if round_id % 2 == 0 else ("http", "hf", "native")):
            cmd = [native, model, FIXTURE, "--bench"] if engine == "native" else [
                sys.executable, Path(__file__).resolve(), "--http-worker" if engine == "http" else "--reference-worker",
                "--tokenizer", config, "--server", args.server]
            raw = run(cmd)
            (args.output / f"{round_id}-{engine}.jsonl").write_text(raw)
            rows = [json.loads(line) for line in raw.splitlines()]
            validate(rows, expected(corpus, engine == "http"), engine == "http")
            for row in rows:
                key = "/".join((row["workload"], row["operation"], engine))
                observations.setdefault(key, []).append(row["elapsed_ns"] / row["iterations"])
    summary = {key: dict(median_ns=statistics.median(v), min_ns=min(v), max_ns=max(v), stdev_ns=statistics.stdev(v), trials=len(v)) for key, v in observations.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = zerv_build.build_files()
    for directory in ("src", "bench", "tools", "tests"):
        paths += sorted(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in (".zig", ".py", ".c", ".json", ".gguf", ".bin", ".txt"))
    for path in paths:
        dest = args.output / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    for name in ("command.txt", "manifest.json", "props.json"):
        source = args.server_record / name
        (args.output / ("server-" + name)).write_bytes(source.read_bytes())
    extensions = sorted(Path(packages.__file__).parent.glob("*.so"))
    manifest = dict(host=platform.uname()._asdict(), python=sys.version, packages=run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
                    hf_extensions={str(p): sha(p) for p in extensions}, **zerv_build.provenance(),
                    native_binary_sha256=sha(native), model_sha256=model_hash, model_bytes=model.stat().st_size,
                    tokenizer_sha256=sha(config), fixture_sha256=sha(FIXTURE), cpu=args.cpu, warmups=3, rounds=3, trials=7,
                    iterations=ITERATIONS, commands=commands, sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
                    server=args.server, server_record=str(args.server_record.resolve()),
                    caveat="Direct native/HF component API vs actual llama-server HTTP round-trip; native HTTP omitted, no serving speedup claim. HF word cache is warm after warmups; native has no word-result cache. Decode native raw/preallocated vs HF UTF-8/allocating.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
