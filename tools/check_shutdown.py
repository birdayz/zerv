#!/usr/bin/env python3
"""Graceful-shutdown check against the real zerv binary and model (one GPU job).

1. Drain. A SIGINT arrives while a stream is generating. New connections must then
   be refused, the stream must complete (finish_reason and [DONE]), and the process
   must exit 0 and log "zerv: stopped".
2. Idle. A SIGTERM arrives while a keep-alive connection is idle. The process must
   exit 0 within a few seconds.
3. Force. A second SIGINT arrives during a long generation. The process must exit
   130 at once.

The report also records host RSS (VmRSS/VmHWM) after load."""
import argparse
from datetime import datetime, timezone
import http.client
import importlib.util
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_serving", ROOT/"bench/run_serving.py")
serving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serving)
BODY = dict(model="qwen3.8-27b", messages=[{"role": "user", "content": "Count from 1 to 200, separated by spaces."}],
            temperature=0, chat_template_kwargs={"enable_thinking": False})


def start(binary, model, port, log):
    proc = subprocess.Popen([str(binary), "--model", str(model), "--port", str(port), "--context", "4096", "--drain-timeout", "60"],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    serving.wait_ready(port, proc)
    status = Path(f"/proc/{proc.pid}/status").read_text()
    rss = {k: v.strip() for k, v in (line.split(":", 1) for line in status.splitlines()) if k in ("VmRSS", "VmHWM")}
    return proc, rss


def refused(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
        return False
    except ConnectionRefusedError:
        return True


def stream_until_first_delta(port, body, first):
    """Streams `body`; sets `first` at the first content delta; returns the parsed stream."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    text, finish, done, buf = "", None, False, b""
    while True:
        chunk = r.read1(65536)
        if not chunk: break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            data = event.decode()[6:]
            if data == "[DONE]": done = True; continue
            for ch in json.loads(data).get("choices", []):
                text += ch["delta"].get("content") or ""
                finish = ch.get("finish_reason") or finish
                if text: first.set()
    close_header = r.getheader("connection")
    c.close()
    return dict(status=r.status, content=text, finish=finish, done=done, connection_header=close_header)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--zerv-binary", type=Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=18099)
    a = p.parse_args()
    if a.zerv_binary is None:
        import zerv_build
        a.zerv_binary = zerv_build.binary("zerv")
    if a.output.exists(): p.error("fresh output required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log = open(str(a.output)+".server.log", "w")
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), zerv_sha256=serving.sha(a.zerv_binary), scenarios={})
    ok = True

    # 1. Drain an in-flight stream.
    proc, rss = start(a.zerv_binary, a.model, a.port, log)
    report["host_memory_after_load"] = rss
    first, result = threading.Event(), {}
    worker = threading.Thread(target=lambda: result.update(stream_until_first_delta(a.port, dict(BODY, stream=True, max_tokens=96), first)))
    worker.start()
    first.wait(120)
    t0 = time.perf_counter()
    proc.send_signal(signal.SIGINT)
    time.sleep(0.2)
    new_refused = refused(a.port)
    worker.join(300)
    code = proc.wait(120)
    exit_s = time.perf_counter()-t0
    log.flush()
    stopped = "zerv: stopped" in Path(str(a.output)+".server.log").read_text()
    s1 = dict(stream=result, new_connection_refused=new_refused, exit_code=code, exit_after_signal_s=exit_s, logged_stopped=stopped)
    s1["ok"] = (result.get("status") == 200 and result.get("finish") == "length" and result.get("done") and new_refused and code == 0 and stopped)
    report["scenarios"]["drain_stream"] = s1; ok &= s1["ok"]
    print("drain_stream", s1, flush=True)
    time.sleep(3)

    # 2. SIGTERM with an idle keep-alive connection.
    proc, _ = start(a.zerv_binary, a.model, a.port, log)
    idle = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
    idle.request("GET", "/health"); idle.getresponse().read()  # keep-alive, now idle
    t0 = time.perf_counter()
    proc.send_signal(signal.SIGTERM)
    code = proc.wait(60)
    s2 = dict(exit_code=code, exit_after_signal_s=time.perf_counter()-t0)
    s2["ok"] = code == 0 and s2["exit_after_signal_s"] < 5
    idle.close()
    report["scenarios"]["idle_sigterm"] = s2; ok &= s2["ok"]
    print("idle_sigterm", s2, flush=True)
    time.sleep(3)

    # 3. A second SIGINT forces an immediate exit during a long generation.
    proc, _ = start(a.zerv_binary, a.model, a.port, log)
    first, result = threading.Event(), {}
    def aborted_stream():
        try: result.update(stream_until_first_delta(a.port, dict(BODY, stream=True, max_tokens=2000), first))
        except (http.client.HTTPException, OSError) as e: result.update(aborted=type(e).__name__)  # expected: forced exit
    worker = threading.Thread(target=aborted_stream, daemon=True)
    worker.start()
    first.wait(120)
    t0 = time.perf_counter()
    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    code = proc.wait(60)
    s3 = dict(exit_code=code, exit_after_second_signal_s=time.perf_counter()-t0-0.3)
    worker.join(10)
    s3["client"] = result.get("aborted", "completed")
    s3["ok"] = code == 130 and s3["exit_after_second_signal_s"] < 5
    report["scenarios"]["force_second_sigint"] = s3; ok &= s3["ok"]
    print("force_second_sigint", s3, flush=True)

    log.close()
    report["passed"] = bool(ok)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    a.output.write_text(json.dumps(report, indent=1)+"\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
