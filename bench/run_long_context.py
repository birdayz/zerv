#!/usr/bin/env python3
"""Long-context serving benchmark: one ~29k-token prompt per request, engines one
at a time, same OpenAI Chat Completions streaming client and engine commands as
run_serving.py. Every request starts with a distinct session tag, so no engine can
reuse a cached prefix or a llama.cpp context checkpoint from an earlier request.
The same prompts (warmup + repeats) are sent to every engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

import run_serving as rs

ROOT = rs.ROOT
QUESTION = "Summarize the passage above in about 200 words."


def paragraph():
    # The single cartography paragraph that serving-v2's long-prompt case repeats 16 times.
    text = json.loads((ROOT/"bench/workloads/serving-v2.json").read_text())["cases"][3]["messages"][0]["content"].split("\n\n")[0]
    head = "The history of cartography"
    second = text.index(head, 1)
    para = text[:second]  # ends with the separating space, as in serving-v2
    assert text == para*16, "serving-v2 long prompt is no longer 16 repeats of one paragraph"
    return para


def prompt(index, repeats):
    return f"Session {index:04d}. " + paragraph()*repeats + "\n\n" + QUESTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engines", default="zerv,zerv-f16,llama-fa-ub512",
                   help="comma-separated engine names; with any `@` knob suffix (run_serving.resolve_engine) separate them with `;`")
    p.add_argument("--paragraphs", type=int, default=145, help="paragraph repeats (145 -> ~29k prompt tokens)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--context", type=int, default=29504)
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--zerv-binary", type=Path, required=True, help="previously built zerv binary (its hash is recorded)")
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if rs.sha(a.model) != rs.MODEL_SHA: raise SystemExit("model mismatch")
    zerv_binary = a.zerv_binary.resolve()
    table = rs.engines(a.model, a.port, a.context, zerv_binary)
    prompts = [prompt(i, a.paragraphs) for i in range(a.repeats+1)]
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=platform.uname()._asdict(), model_sha256=rs.MODEL_SHA,
                    zerv_sha256=rs.sha(zerv_binary), llama_server_sha256=rs.sha(rs.LLAMA_SERVER),
                    llama_version=subprocess.run([rs.LLAMA_SERVER, "--version"], capture_output=True, text=True).stderr.strip(),
                    template_sha256=rs.sha(rs.TEMPLATE), context=a.context, repeats=a.repeats, paragraphs=a.paragraphs, max_tokens=a.max_tokens,
                    prompt_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in prompts], engines={}, vram_before=rs.vram_used(),
                    client="python http.client streaming SSE; TTFT = first content delta; decode rate = (completion_tokens-1)/(last_delta-first_delta); "
                           "request 0 is a warmup with a distinct session tag; every request has a unique prefix (no prefix/checkpoint reuse)")
    raw = (out/"raw.jsonl").open("w")
    for name in (a.engines.split(";") if "@" in a.engines else a.engines.split(",")):
        spec = rs.resolve_engine(table, name, zerv_binary)  # BASE@flag=value knob runs
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
        env.update(spec["env"])
        log = (out/f"{name}.log").open("w")
        base_vram = rs.vram_used()
        t0 = time.perf_counter()
        proc = subprocess.Popen(spec["cmd"], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        peak = [base_vram or 0]
        stop = threading.Event()
        def poll():
            while not stop.is_set():
                v = rs.vram_used()
                if v: peak[0] = max(peak[0], v)
                time.sleep(0.05)
        poller = threading.Thread(target=poll); poller.start()
        try:
            rs.wait_ready(a.port, proc)
            load_s = time.perf_counter()-t0
            loaded_vram = rs.vram_used()
            for i, text in enumerate(prompts):
                body = dict(model="qwen3.8-27b", messages=[dict(role="user", content=text)], stream=True, stream_options={"include_usage": True},
                            max_tokens=a.max_tokens, temperature=0, seed=1234, chat_template_kwargs={"enable_thinking": False})
                r = rs.stream_request(a.port, body)
                r.update(engine=name, request=i, warmup=i == 0, output_sha256=hashlib.sha256((r["reasoning"]+"\x00"+r["content"]).encode()).hexdigest())
                ct = r["usage"]["completion_tokens"]
                span = r["last_delta_s"]-r["ttft_s"]
                r["decode_tok_s"] = (ct-1)/span if ct > 1 and span > 0 else None
                raw.write(json.dumps(r)+"\n"); raw.flush()
                print(name, i, f"ttft={r['ttft_s']:.2f}s total={r['total_s']:.2f}s tokens={r['usage']} decode={r['decode_tok_s']}", flush=True)
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
    for name in (a.engines.split(";") if "@" in a.engines else a.engines.split(",")):
        timed = [r for r in rows if r["engine"] == name and not r["warmup"]]
        if not timed: continue
        # llama-server's own per-request prompt/eval timings, in request order (warmup first).
        server = [dict(prompt_ms=float(m[1]), prompt_tokens=int(m[2]))
                  for m in re.finditer(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", (out/f"{name}.log").read_text())]
        summary[name] = dict(ttft_s=[r["ttft_s"] for r in timed], decode_tok_s=[r["decode_tok_s"] for r in timed],
                             ttft_median_s=statistics.median(r["ttft_s"] for r in timed),
                             decode_median_tok_s=statistics.median(r["decode_tok_s"] for r in timed if r["decode_tok_s"]),
                             prompt_tokens=sorted({r["usage"]["prompt_tokens"] for r in timed}), completion_tokens=[r["usage"]["completion_tokens"] for r in timed],
                             outputs=sorted({r["output_sha256"] for r in timed}), server_prompt_eval=server[1:] if server else None)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"summary.json").write_text(json.dumps(summary, indent=1)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
