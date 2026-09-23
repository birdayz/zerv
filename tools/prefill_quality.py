#!/usr/bin/env python3
"""Prefill-precision quality against the FP64 oracle, for llama.cpp (batched prompt path
under a named precision config) and zerv (native captures from tools/verify_model.py).
Block 14 gate 1: a matched-precision comparison needs both engines' errors on the same
tokens against the same FP64 reference.

Metrics, per oracle case:
  l_out-63: final-layer hidden state of every token, normalized L2 vs FP64
            (both engines capture it for all tokens);
  logits:   normalized L2 vs FP64 at every position llama computes (all) and at the
            positions zerv's prefill emits logits for; argmax agreement with FP64 where
            the FP64 top-1/top-2 margin exceeds the fixture's near-tie margin.
llama runs: tests/reference/llama_batch_capture.c (pinned libllama), fresh work dirs.
Usage:
  prefill_quality.py llama --config default|fp32-full|nocoopmat|nof16 --fixture F --oracle-dir D --work W --output O.json
  prefill_quality.py zerv --native CASE=DIR [...] --fixture F --oracle-dir D --output O.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests/reference"))
from compare_model import load_reference  # noqa: E402
import model_capture  # noqa: E402

MODEL = ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf"
CAPTURE_SRC = ROOT/"tests/reference/llama_batch_capture.c"
CONFIGS = {  # environment + KV type, mirroring bench/run_serving.py engines (FA on, ub512)
    "default": ({}, "f16"),
    "nof16": ({"GGML_VK_DISABLE_F16": "1"}, "f16"),
    "nocoopmat": ({"GGML_VK_DISABLE_COOPMAT": "1"}, "f16"),
    "fp32-full": ({"GGML_VK_DISABLE_MMVQ": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_DISABLE_COOPMAT": "1",
                   "GGML_VK_DISABLE_F16": "1"}, "f32"),
}
PINS = {"/usr/lib/libllama.so.0.4.1": "c352cb4b1f5456dffbc4483ba1e0be7a547b21a0f7e63462ab8fb333f51245e1",
        "/usr/lib/libggml-base.so.0.24.0": "7d9065538f5df6342613b4fa92e661d5ad8fd811c2dbe16ff0e4b62a77777073",
        "/usr/lib/ggml/libggml-vulkan.so": "d09aac86141492bdf22daad0c61b5ded720f772b3f18d8167aae1f264532979a"}


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def nl2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def summarize(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(count=int(v.size), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max()))


def score(case, oracle_dir, hidden, logits_at):
    """hidden: [T, 5120] final-layer states; logits_at: {position: logits row}."""
    work = oracle_dir/case["name"]
    ref = load_reference(work, "fp64-reference")
    T = len(case["tokens"])
    ref_logits = np.fromfile(work/"fp64-logits.bin", dtype="<f4").astype(np.float64).reshape(T, -1)
    h_ref = np.asarray(ref["l_out-63"], dtype=np.float64)
    h_err = [nl2(hidden[t].astype(np.float64), h_ref[t]) for t in range(T)]
    # Near-tie rule as tests/reference/compare_model.py: 10 x max |llama(n_batch 1) - FP64|.
    oracle_llama = np.asarray(model_capture.load(work)["logits"], dtype=np.float64)
    near_tie = 10*max(float(np.max(np.abs(oracle_llama[t]-ref_logits[t]))) for t in range(T))
    l_err, agree, ties = [], 0, 0
    for t, row in sorted(logits_at.items()):
        l_err.append(nl2(row.astype(np.float64), ref_logits[t]))
        top = np.argsort(-ref_logits[t], kind="stable")[:2]
        if ref_logits[t][top[0]] - ref_logits[t][top[1]] > near_tie:
            agree += int(np.argmax(row) == np.argmax(ref_logits[t]))
        else:
            ties += 1
    decided = len(logits_at) - ties
    return dict(tokens=T, hidden_l_out_63=summarize(h_err), logits=summarize(l_err), logit_positions=len(logits_at),
                argmax_agree=agree, argmax_decided=decided, near_ties=ties)


def run_llama(a, fixture):
    for path, digest in PINS.items():
        if sha(path) != digest: raise SystemExit("pin mismatch: "+path)
    env_extra, kv = CONFIGS[a.config]
    work = a.work.resolve()
    if work.exists() or not work.is_relative_to(ROOT/"third_party"): raise SystemExit("fresh --work under third_party required")
    work.mkdir(parents=True)
    tool = work/"llama_batch_capture"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(CAPTURE_SRC), "-o", str(tool),
                    "/usr/lib/libllama.so.0.4.1", "/usr/lib/libggml-base.so.0.24.0", "-lm"], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_", "MESA_", "VK_"))}
    env.update(env_extra)
    results = {}
    for case in fixture["cases"]:
        d = work/case["name"]; d.mkdir()
        (d/"tokens.json").write_text(json.dumps(dict(tokens=case["tokens"]))+"\n")
        (d/"names.txt").write_text("l_out-63\n")
        cmd = [str(tool), str(MODEL), str(d/"tokens.json"), str(d/"names.txt"), str(d), "4096", "2048", "512", "1", kv]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        (d/"stderr.txt").write_text(r.stderr)
        if r.returncode: raise SystemExit(f"{case['name']}: capture failed\n{r.stderr[-2000:]}")
        T = len(case["tokens"])
        logits = np.fromfile(d/"logits.bin", dtype="<f4").reshape(T, -1)
        blob = np.fromfile(d/"tensors.bin", dtype="<f4")
        hidden = np.full((T, 5120), np.nan, dtype=np.float32)
        for line in (d/"index.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["name"] != "l_out-63": continue
            rows = blob[e["offset"]//4:][:e["tokens"]*e["width"]].reshape(e["tokens"], e["width"])
            hidden[e["first_token"]:e["first_token"]+e["tokens"]] = rows
        if np.isnan(hidden).any(): raise SystemExit(case["name"]+": l_out-63 not captured for every token")
        results[case["name"]] = score(case, a.oracle_dir, hidden, {t: logits[t] for t in range(T)})
        results[case["name"]]["capture"] = json.loads(r.stdout.strip().splitlines()[-1])
        print(case["name"], json.dumps(results[case["name"]]), flush=True)
    return dict(engine="llama.cpp b29c606e28 (libllama pinned)", config=a.config, env=env_extra, kv=kv, batch=2048, ubatch=512, flash=True,
                tool_sha256=sha(tool), cases=results)


def run_zerv(a, fixture):
    natives = dict(item.split("=", 1) for item in a.native)
    results = {}
    for case in fixture["cases"]:
        d = Path(natives[case["name"]])
        cap = model_capture.load(d)
        if cap["tokens"]["tokens"] != case["tokens"]: raise SystemExit("native ran different tokens")
        T = len(case["tokens"])
        hidden = np.stack([model_capture.tensor(cap, "l_out-63", t) for t in range(T)])
        summary = json.loads((d/"native-summary.json").read_text())
        prefill = summary.get("mode") == "prefill"
        positions = summary["logit_positions"] if prefill else list(range(T))
        at = {t: np.asarray(cap["logits"][t]) for t in positions}  # T rows; only these are valid
        if prefill:  # decode continuation after the prefill, as compare_model.py
            serving = np.fromfile(d/"serving-logits.bin", dtype="<f4").reshape(-1, 248320)
            at.update({summary["serving_first_position"]+i: serving[i] for i in range(len(serving))})
        results[case["name"]] = score(case, a.oracle_dir, hidden, at)
        results[case["name"]]["mode"] = summary.get("mode"); results[case["name"]]["chunk"] = summary.get("chunk")
        print(case["name"], json.dumps(results[case["name"]]), flush=True)
    return dict(engine="zerv", natives=natives, cases=results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engine", choices=("llama", "zerv"))
    p.add_argument("--config", choices=sorted(CONFIGS), default="default")
    p.add_argument("--native", action="append", default=[])
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--work", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("fresh output required")
    fixture = json.loads(a.fixture.read_text())
    for case in fixture["cases"]:
        for name, record in case["files"].items():
            if sha(a.oracle_dir/name) != record["sha256"]: raise SystemExit("oracle file changed: "+name)
    report = run_llama(a, fixture) if a.engine == "llama" else run_zerv(a, fixture)
    report.update(fixture=str(a.fixture), fixture_sha256=sha(a.fixture), oracle_dir=str(a.oracle_dir))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
