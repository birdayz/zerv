#!/usr/bin/env python3
"""Multi-turn benchmark: C concurrent conversations of a shared system prompt and fixed user
turns (bench/workloads/multiturn-v1.json); each conversation sends its turns one after the
other, every turn with the whole conversation so far (system prompt, earlier user turns, and
the server's own earlier answers). This is where a prefix cache pays: turn 1 of the first
conversation is cold, later turns and other conversations repeat a prefix.

  tools/py bench/run_multiturn.py --output DIR --engines "zerv-f16@parallel=8,kv-type=f16;vllm-apc;llama-fa-kvu" \\
                   [--levels 1,4,8] [--parallel 8] [--context-per-slot 16384] [--rounds 1] [--zerv-binary PATH]
                   [--reference RAW.jsonl]  (identity gate: every turn's output equals that run's)

Engines are started as in bench/run_multiuser.py (one at a time, GPU checked free).
llama-server requests keep its default prompt cache (`cache_prompt` is not sent); vLLM
`vllm-apc` has automatic prefix caching on (its default), `vllm` off. Per level every
conversation runs once (level C: conversations 0..C-1 at once). A fresh server per engine
and round; the levels run in order on it, so later levels may reuse the system prompt from
earlier ones (as a long-running server would).

Reported per level and turn index: TTFT p50/max, prompt tokens (mean), and the server's
`usage.prompt_tokens_details.cached_tokens` where it reports one. Outputs: raw.jsonl (one
line per turn), summary.json, manifest.json, each server's log.
"""
import argparse, hashlib, http.client, json, os, pathlib, statistics, subprocess, sys, threading, time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_serving as rs  # noqa: E402
import run_concurrent as rc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "bench/workloads/multiturn-v1.json"


def log_name(name, rnd):
    # Disk directory flags contain path separators; long knob lists exceed NAME_MAX.
    label = name if "/" not in name and "\\" not in name and len(name.encode()) < 200 else "engine-" + hashlib.sha256(name.encode()).hexdigest()[:16]
    return f"{label}-r{rnd}.log"


def host_memory(spec, launcher_pid):
    """Container launcher RSS is not server RSS; resolve the container's host PID."""
    pid = launcher_pid
    scope = "server-process"
    try:
        if spec.get("container"):
            scope = "container-init-process"
            pid = int(subprocess.check_output(["docker", "inspect", "--format", "{{.State.Pid}}", spec["container"]], text=True))
            if pid <= 0: raise ValueError("container has no running process")
        elif spec["cmd"][0] == "docker":
            return dict(values={}, scope="unavailable", error="container identity not supplied")
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
        values = {k: v.strip() for k, v in (line.split(":", 1) for line in status.splitlines()) if k in ("VmRSS", "VmHWM")}
        return dict(values=values, scope=scope, pid=pid)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        return dict(values={}, scope="unavailable", error=str(error))


def turn(port, body, rec):
    """One streaming turn: rec gets send and token times, usage, and the answer text."""
    conn = None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        rec["send"] = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return ""
        times, content, finishes = [], [], []
        done = False
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done = True
                continue
            if not line.startswith("data: "): continue
            j = json.loads(line[6:])
            if j.get("error"): raise RuntimeError(f"stream error: {j['error']}")
            if j.get("usage"): rec["usage"] = j["usage"]
            for ch in j.get("choices", []):
                if ch.get("finish_reason") is not None: finishes.append(ch["finish_reason"])
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if piece: times.append(time.perf_counter())
                if d.get("content"): content.append(d["content"])
        if not done: raise RuntimeError("stream ended without [DONE]")
        rec["times"] = times
        text = "".join(content)
        rec["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        rec["output_text"] = text
        rec["finish_reasons"] = finishes
        return text
    except Exception as e:  # noqa: BLE001 - recorded, the level is marked failed
        rec["error"] = f"{type(e).__name__}: {e}"
        rec.pop("times", None)
        return ""
    finally:
        if conn is not None: conn.close()


def validate_history(w):
    for conv in w["conversations"]:
        if "assistant_history" in conv:
            history = conv["assistant_history"]
            if not isinstance(history, list) or len(history) != len(conv["turns"]) - 1 or not all(isinstance(text, str) for text in history):
                raise ValueError("assistant_history must contain exactly turns-1 strings")


def conversation(port, w, conv, recs, engine, level, rnd, barrier=None):
    messages = [{"role": "system", "content": conv.get("system", w["system"])}]
    for t, user in enumerate(conv["turns"]):
        if barrier is not None and t > 0: barrier.wait()  # phased: every conversation finished turn t-1
        messages.append({"role": "user", "content": user})
        rec = dict(engine=engine, round=rnd, level=level, conversation=conv["name"], turn=t)
        body = dict(model="qwen3.8-27b", messages=messages, stream=True, stream_options={"include_usage": True},
                    max_tokens=w["max_tokens"], temperature=w["temperature"], seed=w["seed"], **w["options"])
        rec["request_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        rec["history_mode"] = "fixed" if "assistant_history" in conv else "generated"
        answer = turn(port, body, rec)
        recs.append(rec)
        if "error" in rec:
            if barrier is not None: barrier.abort()
            return
        if "assistant_history" in conv and t < len(conv["assistant_history"]): answer = conv["assistant_history"][t]
        messages.append({"role": "assistant", "content": answer})


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def level_run(port, w, level, raw, engine, rnd, phased=False, phase_idle_s=0):
    recs = []
    barrier = threading.Barrier(level, action=(lambda: time.sleep(phase_idle_s)) if phase_idle_s else None) if phased else None
    threads = [threading.Thread(target=conversation, args=(port, w, w["conversations"][i], recs, engine, level, rnd, barrier)) for i in range(level)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    for r in recs: raw.write(json.dumps(r) + "\n")
    gaps = [1000 * (b - a) for r in recs for a, b in zip(r.get("times", []), r.get("times", [])[1:])]
    completed = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in recs)
    out = dict(level=level, round=rnd, wall_s=wall, completion_tokens=completed, aggregate_tok_s=completed / wall,
               stream_gap_ms=dict(p50=pct(gaps, .5), p99=pct(gaps, .99), maximum=max(gaps) if gaps else None),
               errors=[r["error"] for r in recs if "error" in r], turns=[])
    for t in range(len(w["conversations"][0]["turns"])):
        rs_ = [r for r in recs if r["turn"] == t and r.get("times")]
        ttft = [(r["times"][0] - r["send"]) * 1000 for r in rs_]
        cached = [((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in rs_]
        out["turns"].append(dict(turn=t, n=len(rs_), ttft_p50_ms=pct(ttft, .5), ttft_max_ms=max(ttft) if ttft else None,
                                 prompt_tokens_mean=statistics.mean(r["usage"]["prompt_tokens"] for r in rs_ if r.get("usage")) if rs_ else None,
                                 cached_tokens_mean=statistics.mean(c for c in cached if c is not None) if any(c is not None for c in cached) else None))
    all_ttft = [(r["times"][0] - r["send"]) * 1000 for r in recs if r.get("times")]
    out["ttft_p50_ms"], out["ttft_p95_ms"] = pct(all_ttft, .5), pct(all_ttft, .95)
    out["completion_tokens"] = sum(r["usage"]["completion_tokens"] for r in recs if r.get("usage"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=pathlib.Path, default=ROOT / "models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--engines", required=True, help="`;`-separated engine names from run_serving.engines, with @ knobs")
    p.add_argument("--workload", type=pathlib.Path, default=WORKLOAD)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--context-per-slot", type=int, default=16384)
    p.add_argument("--levels", default="1,4,8")
    p.add_argument("--rounds", type=int, default=1, help="start every engine this many times, order alternating per round")
    p.add_argument("--warmup", action="store_true", help="one untimed short request before measured conversations")
    p.add_argument("--phased", action="store_true", help="all conversations finish turn t before any sends turn t+1 (isolates the cache from queueing behind first turns)")
    p.add_argument("--phase-idle-s", type=float, default=0, help="explicit idle time between phased turns, included in wall time (default: no pause)")
    p.add_argument("--port", type=int, default=18098)
    p.add_argument("--llama-cache-ram-mib", type=int, default=0, help="llama-server RAM prompt cache budget (0 retains the old harness baseline)")
    p.add_argument("--llama-checkpoints", type=int, help="llama-server context checkpoints per slot (omit: server default)")
    p.add_argument("--llama-server", type=pathlib.Path, help="another llama-server build to run (default: built in the graph)")
    p.add_argument("--zerv-binary", type=pathlib.Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    p.add_argument("--reference", type=pathlib.Path, help="raw.jsonl of an earlier run: every turn's output must match it (identity gate; exit 1 otherwise)")
    a = p.parse_args()
    if not 0 <= a.phase_idle_s <= 60 or (a.phase_idle_s and not a.phased): p.error("phase idle requires --phased and 0..60 seconds")
    w = json.loads(a.workload.read_text())
    validate_history(w)
    rs.require_host_gpu()
    if a.llama_server: rs.LLAMA_SERVER_OVERRIDE = str(a.llama_server.resolve(strict=True))
    if a.zerv_binary is None:
        sys.path.insert(0, str(ROOT/"tools"))
        import zerv_build
        a.zerv_binary = zerv_build.binary("zerv")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    busy = rc.gpu_busy()
    if busy: raise SystemExit("GPU busy: " + "; ".join(busy))
    levels = [int(x) for x in a.levels.split(",")]
    if max(levels) > len(w["conversations"]): raise SystemExit("more conversations requested than the workload has")
    zb = a.zerv_binary.resolve()
    table = rs.engines(a.model, a.port, a.context_per_slot * a.parallel, zb)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())),
                    model_sha256=rs.sha(a.model), zerv_sha256=rs.sha(zb), llama_server=str(rs.llama_server()), llama_server_sha256=rs.sha(rs.llama_server()),
                    vllm_image=rs.VLLM_IMAGE, vllm_model=str(rs.VLLM_MODEL.relative_to(rs.ROOT)),
                    workload={str(a.workload.resolve().relative_to(ROOT)): rs.sha(a.workload)}, levels=levels, phased=a.phased, warmup=a.warmup,
                    parallel=a.parallel, context_per_slot=a.context_per_slot, rounds=a.rounds, phase_idle_s=a.phase_idle_s, engines={})
    sys.path.insert(0, str(ROOT / "tools"))
    import zerv_build
    manifest["build"] = zerv_build.provenance()
    manifest["sources"] = {str(f.relative_to(ROOT)): rs.sha(f) for f in [*sorted((ROOT / "src").rglob("*.zig")), pathlib.Path(__file__).resolve(), ROOT / "bench/run_serving.py", ROOT / "bench/run_concurrent.py"]}
    raw = (out / "raw.jsonl").open("w")
    names = a.engines.split(";")
    summary = {n: [] for n in names}
    manifest["status"] = "running"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for rnd, name in [(r, n) for r in range(a.rounds) for n in (names if r % 2 == 0 else names[::-1])]:
        spec = dict(rs.resolve_engine(table, name, zb))
        cmd = list(spec["cmd"])
        if cmd[0] == rs.llama_server() or "/llama/bin/llama-server" in cmd:
            cmd[cmd.index("-np") + 1] = str(a.parallel)
            cmd += ["--cache-ram", str(a.llama_cache_ram_mib)]
            if a.llama_checkpoints is not None: cmd += ["--ctx-checkpoints", str(a.llama_checkpoints)]
        elif "--max-num-seqs" in cmd:
            cmd[cmd.index("--max-num-seqs") + 1] = str(a.parallel)
            cmd[cmd.index("--max-model-len") + 1] = str(a.context_per_slot)
        else:
            cmd[cmd.index("--context") + 1] = str(a.context_per_slot)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
        env.update(spec["env"])
        manifest["engines"].setdefault(name, dict(cmd=cmd, env=spec["env"], vram_loaded=[]))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        log = (out / log_name(name, rnd)).open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        peak = [0]
        stop_poll = threading.Event()
        def poll_vram():
            while not stop_poll.wait(.05): peak[0] = max(peak[0], rs.vram_used() or 0)
        poller = threading.Thread(target=poll_vram); poller.start()
        try:
            rs.wait_ready(a.port, proc, spec.get("ready_timeout", 600))
            manifest["engines"].setdefault(name, dict(cmd=cmd, env=spec["env"], vram_loaded=[]))["vram_loaded"].append(rs.vram_used())
            if a.warmup:
                warm = {}
                turn(a.port, dict(model="qwen3.8-27b", messages=[{"role": "user", "content": "Reply with OK."}], stream=True,
                                  max_tokens=8, temperature=0, **w["options"]), warm)
                if "error" in warm: raise RuntimeError(warm["error"])
            for level in levels:
                s = level_run(a.port, w, level, raw, name, rnd, a.phased, a.phase_idle_s)
                summary[name].append(s)
                print(name, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k != "turns"}), flush=True)
                for t in s["turns"]:
                    print("   ", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()}), flush=True)
            memory = host_memory(spec, proc.pid)
            manifest["engines"][name].setdefault("resources", []).append(dict(round=rnd, log=log_name(name, rnd), vram_peak=peak[0], host_memory=memory["values"], host_memory_details=memory))
        except BaseException as e:
            manifest.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=datetime.now(timezone.utc).isoformat())
            raw.flush()
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
            raise
        finally:
            stop_poll.set(); poller.join()
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if spec.get("stop"): subprocess.run(spec["stop"], capture_output=True)
            log.close()
            time.sleep(3)
    raw.close()
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "failed" if any(s["errors"] for series in summary.values() for s in series) else "passed"
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if rc.gpu_busy(): print("WARNING: a server is still running after the benchmark", file=sys.stderr)
    if manifest["status"] != "passed": raise SystemExit("serving request failures: see raw.jsonl")
    if a.reference:
        key = lambda r: (r["level"], r["conversation"], r["turn"])
        ref = {key(r): r.get("output_sha256") for r in map(json.loads, a.reference.read_text().splitlines())}
        mine = [json.loads(l) for l in (out / "raw.jsonl").read_text().splitlines()]
        bad = [key(r) for r in mine if r.get("output_sha256") is None or ref.get(key(r)) != r["output_sha256"]]
        missing = [k for k in ref if k[0] in levels and k not in {key(r) for r in mine}]  # levels this run covers
        print(f"identity gate: {len(mine) - len(bad)}/{len(mine)} turns equal the reference" + (f", mismatches {bad}" if bad else "") + (f", missing {missing}" if missing else ""))
        if bad or missing: sys.exit(1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
