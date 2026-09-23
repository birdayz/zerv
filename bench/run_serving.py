#!/usr/bin/env python3
"""Matched serving benchmark: native zerv vs tuned llama-server over the same
OpenAI Chat Completions v1 streaming client, same artifact, prompts and greedy
settings. Records raw per-request timings, output hashes/texts, VRAM and RSS.
Engines run one at a time; nothing else should use the GPU meanwhile."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
TEMPLATE = ROOT/"third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/official-template.jinja"
LLAMA_SERVER = "/usr/bin/llama-server"


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def vram_used():
    for card in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        return int(card.read_text())
    return None


def engines(model, port, context, zerv_binary):
    common = [LLAMA_SERVER, "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(context), "-np", "1", "-ngl", "99",
              "--spec-type", "none", "--no-context-shift", "--no-webui", "--jinja", "--chat-template-file", str(TEMPLATE),
              "--reasoning-format", "deepseek", "--cache-ram", "0", "-a", "qwen3.8-27b"]
    return {
        "zerv": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context)], env={}),
        # Explicit f16 prompt projections (block 14; matched to llama's fast-path arithmetic class).
        "zerv-f16": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--prefill-precision", "f16"], env={}),
        # Best observed llama.cpp Vulkan configuration family (tuned below by the sweep).
        "llama-fa-ub512": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={}),
        "llama-fa-ub256": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "256"], env={}),
        "llama-nofa-ub512": dict(cmd=common+["-fa", "off", "-b", "2048", "-ub", "512"], env={}),
        # Precision ladder (block 14 research; the pipelines used are logged via GGML_VK_PIPELINE_STATS):
        # default = f16 activations, f16 accumulation (coopmat); nocoopmat = Q8_1 activations with
        # integer dot products (exact int32 block sums); nof16 = coopmat with f32 accumulation?
        "llama-fa-ub512-nocoopmat": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        "llama-fa-ub512-nof16": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_F16": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        "llama-fa-ub512-noint-nocoopmat": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        # Partial precision control: FP32 activations into decode matvecs and FP32 KV only.
        "llama-f32": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512", "-ctk", "f32", "-ctv", "f32"], env={"GGML_VK_DISABLE_MMVQ": "1"}),
        # Full FP32 control: also no Q8_1 integer-dot prompt matmuls, no FP16/cooperative matrices.
        "llama-fp32-full": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512", "-ctk", "f32", "-ctv", "f32"],
                                env={"GGML_VK_DISABLE_MMVQ": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_DISABLE_F16": "1"}),
    }


def stream_request(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200: raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]}")
    first = last = None
    reasoning, content, usage, finish, deltas = [], [], None, None, 0
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk: break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            line = event.decode()
            if not line.startswith("data: "): continue
            data = line[6:]
            if data == "[DONE]": continue
            j = json.loads(data)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                r, c = d.get("reasoning_content") or "", d.get("content") or ""
                if r or c:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    deltas += 1
                reasoning.append(r); content.append(c)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    end = time.perf_counter()
    conn.close()
    return dict(ttft_s=(first or end)-start, total_s=end-start, last_delta_s=(last or end)-start, deltas=deltas, usage=usage, finish=finish,
                reasoning="".join(reasoning), content="".join(content))


def wait_ready(port, process, timeout=600):
    deadline = time.time()+timeout
    while time.time() < deadline:
        if process.poll() is not None: raise RuntimeError("server exited during startup")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health"); r = c.getresponse(); ok = r.status == 200; r.read(); c.close()
            if ok: return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workload", type=Path, default=ROOT/"bench/workloads/serving-v2.json")
    p.add_argument("--engines", default="zerv,llama-fa-ub512,llama-fp32-full")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--zerv-binary", type=Path, help="reuse a previously benchmarked zerv binary (no rebuild; its hash is recorded)")
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if sha(a.model) != MODEL_SHA: raise SystemExit("model mismatch")
    workload = json.loads(a.workload.read_text())
    zig = ROOT/".tools/zig-x86_64-linux-0.16.0/zig"
    build = [str(zig), "build", "server", "-Doptimize=ReleaseFast", "-Dcpu=native"]
    if a.zerv_binary:
        build = ["reused", str(a.zerv_binary.resolve()), sha(a.zerv_binary)]
        source = a.zerv_binary.resolve()
    else:
        subprocess.run(build, cwd=ROOT, check=True)
        source = ROOT/"zig-out/bin/zerv"
    artifact = ROOT/"third_party/serving-bench"/out.name; artifact.mkdir(parents=True, exist_ok=False)
    zerv_binary = artifact/"zerv"; shutil.copy2(source, zerv_binary)
    table = engines(a.model, a.port, a.context, zerv_binary)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=platform.uname()._asdict(), model_sha256=MODEL_SHA,
                    workload_sha256=sha(a.workload), zerv_sha256=sha(zerv_binary), llama_server_sha256=sha(LLAMA_SERVER),
                    llama_version=subprocess.run([LLAMA_SERVER, "--version"], capture_output=True, text=True).stderr.strip(),
                    template_sha256=sha(TEMPLATE), build=build, context=a.context, repeats=a.repeats, engines={}, vram_before=vram_used(),
                    client="python http.client streaming SSE; TTFT = first reasoning/content delta; decode rate = (completion_tokens-1)/(last_delta-first_delta)")
    raw = (out/"raw.jsonl").open("w")
    for name in a.engines.split(","):
        spec = table[name]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
        env.update(spec["env"])
        log = (out/f"{name}.log").open("w")
        base_vram = vram_used()
        t0 = time.perf_counter()
        proc = subprocess.Popen(spec["cmd"], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        peak = [base_vram or 0]
        stop = threading.Event()
        def poll():
            while not stop.is_set():
                v = vram_used()
                if v: peak[0] = max(peak[0], v)
                time.sleep(0.05)
        poller = threading.Thread(target=poll); poller.start()
        try:
            wait_ready(a.port, proc)
            load_s = time.perf_counter()-t0
            loaded_vram = vram_used()
            # Warmup: one short request per case.
            for case in workload["cases"]:
                stream_request(a.port, dict(model="qwen3.8-27b", messages=case["messages"], stream=True, max_tokens=8, temperature=0, seed=workload["seed"], **case["options"]))
            for rep in range(a.repeats):
                for case in workload["cases"]:
                    body = dict(model="qwen3.8-27b", messages=case["messages"], stream=True, stream_options={"include_usage": True},
                                max_tokens=case["max_tokens"], temperature=workload["temperature"], seed=workload["seed"], **case["options"])
                    r = stream_request(a.port, body)
                    r.update(engine=name, case=case["name"], repeat=rep, output_sha256=hashlib.sha256((r["reasoning"]+"\x00"+r["content"]).encode()).hexdigest())
                    ct = r["usage"]["completion_tokens"]
                    span = r["last_delta_s"]-r["ttft_s"]
                    r["decode_tok_s"] = (ct-1)/span if ct > 1 and span > 0 else None
                    raw.write(json.dumps(r)+"\n"); raw.flush()
                    print(name, case["name"], rep, f"ttft={r['ttft_s']*1000:.1f}ms total={r['total_s']:.2f}s tokens={r['usage']} decode={r['decode_tok_s']}", flush=True)
            status = Path(f"/proc/{proc.pid}/status").read_text()
            rss = {k: v.strip() for k, v in (line.split(":", 1) for line in status.splitlines()) if k in ("VmRSS", "VmHWM")}
            manifest["engines"][name] = dict(cmd=spec["cmd"], env=spec["env"], load_s=load_s, vram_idle_before=base_vram, vram_loaded=loaded_vram,
                                             vram_peak=peak[0], host_memory=rss)
        finally:
            stop.set(); poller.join()
            proc.send_signal(signal.SIGINT)
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            log.close()
            time.sleep(3)
    raw.close()
    rows = [json.loads(line) for line in (out/"raw.jsonl").read_text().splitlines()]
    summary = {}
    for r in rows:
        s = summary.setdefault(r["engine"], {}).setdefault(r["case"], dict(ttft_ms=[], total_s=[], decode_tok_s=[], completion_tokens=[], prompt_tokens=[], outputs=set()))
        s["ttft_ms"].append(r["ttft_s"]*1000); s["total_s"].append(r["total_s"])
        if r["decode_tok_s"]: s["decode_tok_s"].append(r["decode_tok_s"])
        s["completion_tokens"].append(r["usage"]["completion_tokens"]); s["prompt_tokens"].append(r["usage"]["prompt_tokens"]); s["outputs"].add(r["output_sha256"])
    for engine in summary.values():
        for case, s in engine.items():
            engine[case] = {k: (dict(median=statistics.median(v), min=min(v), max=max(v)) if isinstance(v, list) and v and not isinstance(v[0], str) else sorted(v) if isinstance(v, set) else v) for k, v in s.items()}
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"summary.json").write_text(json.dumps(summary, indent=1)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")


if __name__ == "__main__":
    main()
