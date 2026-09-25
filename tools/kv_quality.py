#!/usr/bin/env python3
"""Long-context KV precision quality (docs/specs/model.md, "KV precision", gate 4).

Text: the documents of bench/workloads/long-v1.json case decode-38k (repository docs
snapshot, without the needle sentence and the final question), raw tokens, no template.
For each engine and KV type: prefill PREFIX tokens, then decode STEPS tokens one at a
time, teacher-forced, and keep the next-token logits of positions PREFIX-1 .. PREFIX+STEPS-1.
  zerv:  zig-out/bin/zerv-kv-quality (built by --build)
  llama: tests/reference/llama_batch_capture.c (pinned libllama, FA on) with LOGITS_FROM,
         so its steps also use its single-token decode path; f16 twice for run-to-run noise.
Metrics per pair (reference P, candidate Q) over the STEPS + 1 rows: KL(P || Q) mean,
median, p99, max; top-1 agreement; and each run's mean NLL of the true next tokens.
Usage: tools/kv_quality.py --output DIR [--prefix 36000] [--steps 256] [--runs zerv-f32,...]"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf"
WORKLOAD = ROOT/"bench/workloads/long-v1.json"
CAPTURE_SRC = ROOT/"tests/reference/llama_batch_capture.c"
VOCAB = 248320
PINS = {"/usr/lib/libllama.so.0.4.1": "c352cb4b1f5456dffbc4483ba1e0be7a547b21a0f7e63462ab8fb333f51245e1",
        "/usr/lib/libggml-base.so.0.24.0": "7d9065538f5df6342613b4fa92e661d5ad8fd811c2dbe16ff0e4b62a77777073",
        "/usr/lib/ggml/libggml-vulkan.so": "d09aac86141492bdf22daad0c61b5ded720f772b3f18d8167aae1f264532979a"}
RUNS = ["zerv-f32", "zerv-f16", "llama-f32", "llama-f16", "llama-f16-repeat"]
PAIRS = [("zerv-f32", "zerv-f16"), ("llama-f32", "llama-f16"), ("llama-f16", "llama-f16-repeat"), ("zerv-f32", "llama-f32"), ("zerv-f32", "llama-f16")]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def document_text():
    """The docs of long-v1's decode-38k prompt: after the needle paragraph, before the question."""
    case = next(c for c in json.loads(WORKLOAD.read_text())["cases"] if c["name"] == "decode-38k")
    content = case["messages"][-1]["content"]
    start = content.index("\n\n") + 2
    end = content.rindex("\n\n")
    return content[start:end]


def log_softmax(rows):
    rows = rows.astype(np.float64)
    m = rows.max(axis=1, keepdims=True)
    return rows - m - np.log(np.exp(rows - m).sum(axis=1, keepdims=True))


def compare(p_logits, q_logits):
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    return dict(kl_mean=float(kl.mean()), kl_median=float(np.median(kl)), kl_p99=float(np.quantile(kl, 0.99)), kl_max=float(kl.max()),
                top1_agree=float((p_logits.argmax(axis=1) == q_logits.argmax(axis=1)).mean()), rows=int(len(kl)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", type=int, default=36000)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--runs", default=",".join(RUNS))
    p.add_argument("--build", action="store_true", help="build zerv-kv-quality first")
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    work = ROOT/"third_party/kv-quality"/out.name; work.mkdir(parents=True, exist_ok=False)
    if sha(MODEL) != "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d": raise SystemExit("model mismatch")
    for path, digest in PINS.items():
        if sha(path) != digest: raise SystemExit("pin mismatch: "+path)
    text = work/"text.txt"; text.write_text(document_text())
    zig = ROOT/".tools/zig-x86_64-linux-0.16.0/zig"
    if a.build: subprocess.run([str(zig), "build", "kv-quality-build", "-Doptimize=ReleaseFast", "-Dcpu=native"], cwd=ROOT, check=True)
    tool = work/"zerv-kv-quality"; tool.write_bytes((ROOT/"zig-out/bin/zerv-kv-quality").read_bytes()); tool.chmod(0o755)
    capture = work/"llama_batch_capture"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(CAPTURE_SRC), "-o", str(capture),
                    "/usr/lib/libllama.so.0.4.1", "/usr/lib/libggml-base.so.0.24.0", "-lm"], check=True)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, model_sha256=sha(MODEL), workload_sha256=sha(WORKLOAD),
                    text_sha256=sha(text), zerv_tool_sha256=sha(tool), llama_capture_sha256=sha(capture), llama_pins=PINS,
                    prefix=a.prefix, steps=a.steps, runs={})
    n = a.prefix + a.steps
    ctx = str((n + 1 + 255) // 256 * 256)
    runs = a.runs.split(",")
    tokens_json = None
    for name in sorted(runs, key=lambda r: not r.startswith("zerv")):  # zerv first: it writes the tokens
        d = work/name; d.mkdir()
        kv = name.split("-")[1]
        if name.startswith("zerv"):
            cmd = [str(tool), str(MODEL), str(text), str(d), str(a.prefix), str(a.steps), kv]
            env = None
        else:
            if tokens_json is None: raise SystemExit("a zerv run must come first (it writes the tokens)")
            (d/"names.txt").write_text("")
            cmd = [str(capture), str(MODEL), str(tokens_json), str(d/"names.txt"), str(d), ctx, "2048", "512", "1", kv, str(a.prefix)]
            env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_", "MESA_", "VK_"))}
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
        (d/"stderr.txt").write_text(r.stderr)
        if r.returncode: raise SystemExit(f"{name} failed:\n{r.stderr[-3000:]}")
        if name.startswith("zerv") and tokens_json is None: tokens_json = d/"tokens.json"
        manifest["runs"][name] = dict(cmd=cmd, stdout=r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", logits_sha256=sha(d/"logits.bin"))
        print(name, manifest["runs"][name]["stdout"], flush=True)
    tokens = json.loads(tokens_json.read_text())["tokens"]
    for name in runs:
        if name.startswith("zerv") and json.loads((work/name/"tokens.json").read_text())["tokens"] != tokens: raise SystemExit("token mismatch")
    logits = {name: np.fromfile(work/name/"logits.bin", dtype="<f4").reshape(-1, VOCAB) for name in runs}
    for name, rows in logits.items():
        if rows.shape[0] != a.steps + 1: raise SystemExit(f"{name}: {rows.shape[0]} rows")
    truth = np.asarray(tokens[a.prefix:n + 0], dtype=np.int64)  # next token of rows 0 .. steps-1
    results = dict(nll={}, pairs={})
    for name, rows in logits.items():
        lp = log_softmax(rows[:a.steps])
        results["nll"][name] = float(-lp[np.arange(a.steps), truth].mean())
    for ref, cand in PAIRS:
        if ref in logits and cand in logits: results["pairs"][f"{ref} || {cand}"] = compare(logits[ref], logits[cand])
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    (out/"results.json").write_text(json.dumps(results, indent=1)+"\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
