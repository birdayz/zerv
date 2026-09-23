#!/usr/bin/env python3
"""Startup protections of the real server binary (docs/specs/serving.md, "Startup").

With server A loaded and serving:
1. B on A's port must fail at once (before loading the model): "already in use".
2. C on another port must fail before allocating: "not enough free VRAM" (A holds it).
3. A must be unaffected: /health ok, a greedy request completes, SIGINT exits 0.
Uses the GPU; nothing else may use it meanwhile."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def vram_used():
    for card in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        return int(card.read_text())
    return None


def healthy(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/health"); r = c.getresponse(); r.read(); c.close()
        return r.status == 200
    except OSError:
        return False


def attempt(cmd, log):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    log.write_text(r.stdout)
    return dict(exit_code=r.returncode, seconds=time.perf_counter()-t0, output_tail=r.stdout.strip().splitlines()[-3:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--zerv-binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="fresh directory")
    p.add_argument("--port", type=int, default=18100)
    p.add_argument("--context", type=int, default=29504)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = str(a.zerv_binary.resolve())
    base = [binary, "--model", str(a.model), "--context", str(a.context)]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, zerv_sha256=sha(binary), vram_idle=vram_used())
    log_a = (out/"a.log").open("w")
    server = subprocess.Popen(base + ["--port", str(a.port)], stdout=log_a, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 300
        while not healthy(a.port):
            if server.poll() is not None or time.time() > deadline: raise SystemExit("server A did not start")
            time.sleep(0.5)
        report["vram_a_loaded"] = vram_used()
        report["same_port"] = attempt(base + ["--port", str(a.port)], out/"b.log")
        report["vram_after_b"] = vram_used()
        report["other_port"] = attempt(base + ["--port", str(a.port + 1)], out/"c.log")
        report["vram_after_c"] = vram_used()
        report["a_healthy_after"] = healthy(a.port)
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
        body = dict(model="qwen3.8-27b", messages=[dict(role="user", content="Say OK.")], max_tokens=8, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        r = c.getresponse(); data = json.loads(r.read()); c.close()
        report["a_request"] = dict(status=r.status, content=data["choices"][0]["message"]["content"], usage=data["usage"])
    finally:
        server.send_signal(signal.SIGINT)
        try: report["a_exit_code"] = server.wait(timeout=60)
        except subprocess.TimeoutExpired: server.kill(); report["a_exit_code"] = server.wait()
        log_a.close()
    b, c_ = report["same_port"], report["other_port"]
    checks = dict(
        same_port_refused_fast=b["exit_code"] != 0 and b["seconds"] < 5 and any("already in use" in l for l in b["output_tail"]),
        same_port_no_vram=report["vram_after_b"] is not None and abs(report["vram_after_b"] - report["vram_a_loaded"]) < 256 << 20,
        other_port_refused_vram=c_["exit_code"] != 0 and any("not enough free VRAM" in l for l in c_["output_tail"]),
        a_unaffected=report["a_healthy_after"] and report["a_request"]["status"] == 200 and report["a_exit_code"] == 0,
    )
    report.update(checks=checks, passed=all(checks.values()), finished_at=datetime.now(timezone.utc).isoformat())
    (out/"report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
