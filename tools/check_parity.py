#!/usr/bin/env python3
"""Output-format parity with llama-server. Each engine runs the same greedy requests
(bench/workloads/output-parity-v1.json) in JSON and SSE modes. The engines run one
after the other, never concurrently. For every case and mode, reasoning_content,
content, finish_reason and prompt/completion token counts must be equal.

Verdicts:
- `equal`: every compared field matches.
- `FORMAT-MISMATCH`: the completion token counts are equal but the text or the
  finish reason differs.
- `COUNT-MISMATCH`: the completion token counts differ. This is either a numerical
  (greedy token) divergence or a termination-semantics difference, for example a
  stop string that one engine does not honour. Inspect the texts in report.json.
Only all-`equal` passes."""
import argparse
from datetime import datetime, timezone
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_serving", ROOT/"bench/run_serving.py")
serving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serving)


def post(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    if r.status != 200: raise RuntimeError(f"HTTP {r.status}: {data[:400]}")
    j = json.loads(data)
    ch = j["choices"][0]
    return dict(reasoning=ch["message"].get("reasoning_content"), content=ch["message"].get("content"), finish=ch["finish_reason"],
                usage=dict(prompt_tokens=j["usage"]["prompt_tokens"], completion_tokens=j["usage"]["completion_tokens"]),
                reasoning_key_present="reasoning_content" in ch["message"])


def run_engine(name, cmd, env, port, workload, out):
    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
    full_env.update(env)
    log = (out/f"{name}.log").open("w")
    proc = subprocess.Popen(cmd, env=full_env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    results = {}
    try:
        serving.wait_ready(port, proc)
        for case in workload["cases"]:
            base = dict(model="qwen3.8-27b", messages=case["messages"], max_tokens=case["max_tokens"],
                        temperature=workload["temperature"], seed=workload["seed"], **case["options"])
            j = post(port, dict(base, stream=False))
            s = serving.stream_request(port, dict(base, stream=True, stream_options={"include_usage": True}))
            results[case["name"]] = dict(
                json=j,
                sse=dict(reasoning=s["reasoning"] or None, content=s["content"], finish=s["finish"],
                         usage=dict(prompt_tokens=s["usage"]["prompt_tokens"], completion_tokens=s["usage"]["completion_tokens"])))
            print(name, case["name"], "json", j["finish"], j["usage"], "sse", s["finish"], s["usage"], flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=60)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()
        time.sleep(3)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True, help="fresh directory for report.json and logs")
    p.add_argument("--workload", type=Path, default=ROOT/"bench/workloads/output-parity-v1.json")
    p.add_argument("--llama-engine", default="llama-fp32-full", help="engine name from bench/run_serving.py")
    p.add_argument("--zerv-binary", type=Path, help="use this zerv binary instead of building one")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--port", type=int, default=18097)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if serving.sha(a.model) != serving.MODEL_SHA: raise SystemExit("model mismatch")
    workload = json.loads(a.workload.read_text())
    if a.zerv_binary: source = a.zerv_binary.resolve()
    else:
        subprocess.run([str(ROOT/".tools/zig-x86_64-linux-0.16.0/zig"), "build", "server", "-Doptimize=ReleaseFast", "-Dcpu=native"], cwd=ROOT, check=True)
        source = ROOT/"zig-out/bin/zerv"
    binary = ROOT/"third_party/serving-bench"/out.name/"zerv"
    binary.parent.mkdir(parents=True, exist_ok=False); shutil.copy2(source, binary)
    table = serving.engines(a.model, a.port, a.context, binary)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, workload_sha256=serving.sha(a.workload),
                  model_sha256=serving.MODEL_SHA, zerv_sha256=serving.sha(binary), llama_server_sha256=serving.sha(serving.LLAMA_SERVER),
                  llama_version=subprocess.run([serving.LLAMA_SERVER, "--version"], capture_output=True, text=True).stderr.strip(),
                  engines={}, cases=[])
    results = {}
    for name in (a.llama_engine, "zerv"):
        report["engines"][name] = dict(cmd=table[name]["cmd"], env=table[name]["env"])
        results[name] = run_engine(name, table[name]["cmd"], table[name]["env"], a.port, workload, out)
    formatting_ok = True
    for case in workload["cases"]:
        for mode in ("json", "sse"):
            ref, got = results[a.llama_engine][case["name"]][mode], results["zerv"][case["name"]][mode]
            keys = ("reasoning", "content", "finish", "usage")
            equal = all(ref[k] == got[k] for k in keys)
            counts_differ = ref["usage"]["completion_tokens"] != got["usage"]["completion_tokens"]
            verdict = "equal" if equal else ("COUNT-MISMATCH" if counts_differ else "FORMAT-MISMATCH")
            if verdict == "FORMAT-MISMATCH": formatting_ok = False
            report["cases"].append(dict(case=case["name"], mode=mode, verdict=verdict, llama=ref, zerv=got))
            print(case["name"], mode, verdict, flush=True)
    report["formatting_ok"] = formatting_ok
    report["all_equal"] = all(c["verdict"] == "equal" for c in report["cases"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"report.json").write_text(json.dumps(report, indent=1)+"\n")
    sys.exit(0 if report["all_equal"] else 1)


if __name__ == "__main__":
    main()
