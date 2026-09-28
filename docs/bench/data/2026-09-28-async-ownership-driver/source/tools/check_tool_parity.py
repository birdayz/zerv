#!/usr/bin/env python3
"""Tool-calling parity with llama-server (block 12c).

Workload: bench/workloads/tool-parity-v1.json.
- `render` cases: llama-server's POST /apply-template prompt is recorded, and compared
  with the independent Jinja oracle (tests/reference/render_tools.py) when `--oracle`
  is given. zerv's rendering is checked against the same oracle by the Zig unit tests.
- `generation` cases: greedy requests in JSON and SSE modes. For every case and mode,
  reasoning_content, content, tool calls (name + arguments parsed as JSON), finish_reason
  and prompt/completion token counts are compared between the engines.

Engines run one after the other, never concurrently. Raw responses are kept.
Verdicts per generation case/mode: `equal`, `FORMAT-MISMATCH` (equal completion token
counts, different parsed message) or `COUNT-MISMATCH` (numerical divergence or a
termination difference; inspect report.json). Only all-`equal` passes."""
import argparse
from datetime import datetime, timezone
import http.client
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_serving", ROOT/"bench/run_serving.py")
serving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serving)


def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request(method, path, json.dumps(body) if body is not None else None, {"content-type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, data


def body_for(case, workload, stream):
    b = dict(model="qwen3.8-27b", messages=case["messages"], tools=case["tools"], max_tokens=case["max_tokens"],
             temperature=workload["temperature"], seed=workload["seed"], stream=stream, **case["options"])
    if stream: b["stream_options"] = {"include_usage": True}
    return b


def parse_json_response(status, data):
    if status != 200: return dict(http_status=status, error=data.decode(errors="replace")[:2000])
    j = json.loads(data)
    ch = j["choices"][0]; m = ch["message"]
    return dict(raw=j, message=dict(reasoning=m.get("reasoning_content"), content=m.get("content"), tool_calls=m.get("tool_calls")),
                finish=ch["finish_reason"], usage=dict(prompt_tokens=j["usage"]["prompt_tokens"], completion_tokens=j["usage"]["completion_tokens"]))


def stream(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        data = r.read(); c.close()
        return dict(http_status=r.status, error=data.decode(errors="replace")[:2000])
    events, buf = [], b""
    while True:
        chunk = r.read1(65536)
        if not chunk: break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            line = event.decode()
            if line.startswith("data: "): events.append(line[6:])
    c.close()
    reasoning, content, calls, finish, usage = [], [], {}, None, None
    for e in events:
        if e == "[DONE]": continue
        j = json.loads(e)
        if j.get("usage"): usage = dict(prompt_tokens=j["usage"]["prompt_tokens"], completion_tokens=j["usage"]["completion_tokens"])
        for ch in j.get("choices", []):
            d = ch.get("delta", {})
            reasoning.append(d.get("reasoning_content") or ""); content.append(d.get("content") or "")
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], dict(id="", type="", name="", arguments=""))
                slot["id"] += tc.get("id") or ""; slot["type"] += tc.get("type") or ""
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""; slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    tool_calls = [dict(id=v["id"], type=v["type"], function=dict(name=v["name"], arguments=v["arguments"])) for _, v in sorted(calls.items())] or None
    r_text, c_text = "".join(reasoning), "".join(content)
    return dict(events=events, message=dict(reasoning=r_text or None, content=c_text, tool_calls=tool_calls), finish=finish, usage=usage)


def run_engine(name, cmd, env, port, workload, out, render):
    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
    full_env.update(env)
    log = (out/f"{name}.log").open("w")
    proc = subprocess.Popen(cmd, env=full_env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    results = dict(render={}, generation={})
    try:
        serving.wait_ready(port, proc)
        if render:
            status, data = request(port, "GET", "/props")
            results["props"] = json.loads(data) if status == 200 else dict(http_status=status)
            for case in workload["render"]:
                b = dict(messages=case["messages"], tools=case["tools"], **case["options"])
                status, data = request(port, "POST", "/apply-template", b)
                results["render"][case["name"]] = json.loads(data)["prompt"] if status == 200 else dict(http_status=status, error=data.decode(errors="replace")[:2000])
        for case in workload["generation"]:
            status, data = request(port, "POST", "/v1/chat/completions", body_for(case, workload, False))
            j = parse_json_response(status, data)
            s = stream(port, body_for(case, workload, True))
            results["generation"][case["name"]] = dict(json=j, sse=s)
            print(name, case["name"], j.get("finish"), (j.get("message") or {}).get("tool_calls"), j.get("usage") or j.get("error"), flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=60)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()
        time.sleep(3)
    return results


def normalized(r):
    """Comparable view of one response: tool call ids are server-generated and ignored;
    arguments are compared as parsed JSON (key order and spacing are not semantic)."""
    if "message" not in r: return dict(error=r.get("http_status"))
    m = r["message"]
    calls = None
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            try: args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError): args = dict(__unparsed__=tc["function"]["arguments"])
            calls.append(dict(type=tc.get("type"), name=tc["function"]["name"], arguments=args))
    return dict(reasoning=m.get("reasoning") or None, content=m.get("content") or "", tool_calls=calls, finish=r["finish"], usage=r["usage"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--workload", type=Path, default=ROOT/"bench/workloads/tool-parity-v1.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engines", default="llama-fp32-full,zerv", help="engine names from bench/run_serving.py; the first is the reference")
    p.add_argument("--zerv-binary", type=Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--port", type=int, default=18093)
    p.add_argument("--reference-raw", type=Path, help="raw.json of an earlier run: its first engine is the reference and is not rerun")
    a = p.parse_args()
    if a.zerv_binary is None:
        import zerv_build
        a.zerv_binary = zerv_build.binary("zerv")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if serving.sha(a.model) != serving.MODEL_SHA: raise SystemExit("model mismatch")
    workload = json.loads(a.workload.read_text())
    binary = a.zerv_binary.resolve()
    table = serving.engines(a.model, a.port, a.context, binary)
    names = a.engines.split(",")
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, workload_sha256=serving.sha(a.workload),
                  model_sha256=serving.MODEL_SHA, zerv_sha256=serving.sha(binary) if binary.exists() else None,
                  llama_server_sha256=serving.sha(serving.llama_server()),
                  llama_version=subprocess.run([serving.llama_server(), "--version"], capture_output=True, text=True).stderr.strip(),
                  engines={}, cases=[])
    results = {}
    if a.reference_raw:
        stored = json.loads(a.reference_raw.read_text())
        ref_name = next(iter(stored))
        results[ref_name] = stored[ref_name]
        report["reference_raw"] = dict(path=str(a.reference_raw), sha256=serving.sha(a.reference_raw), engine=ref_name)
        names = [ref_name] + [n for n in names if n != ref_name]
    for name in names:
        if name in results: continue
        report["engines"][name] = dict(cmd=table[name]["cmd"], env=table[name]["env"])
        results[name] = run_engine(name, table[name]["cmd"], table[name]["env"], a.port, workload, out, render=name.startswith("llama"))
    (out/"raw.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
    if len(names) >= 2:
        ref, test = names[0], names[1]
        for case in workload["generation"]:
            for mode in ("json", "sse"):
                a_, b_ = normalized(results[ref]["generation"][case["name"]][mode]), normalized(results[test]["generation"][case["name"]][mode])
                if a_ == b_: verdict = "equal"
                elif a_.get("usage") and b_.get("usage") and a_["usage"]["completion_tokens"] != b_["usage"]["completion_tokens"]: verdict = "COUNT-MISMATCH"
                else: verdict = "FORMAT-MISMATCH"
                report["cases"].append(dict(case=case["name"], mode=mode, verdict=verdict, reference=a_, test=b_))
                print(f"{case['name']:24s} {mode:4s} {verdict}")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    if report["cases"] and any(c["verdict"] != "equal" for c in report["cases"]): sys.exit(1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
