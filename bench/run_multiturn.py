#!/usr/bin/env python3
"""Multi-turn benchmark: C concurrent conversations of a shared system prompt and fixed user
turns (bench/workloads/multiturn-v1.json); each conversation sends its turns one after the
other, every turn with the whole conversation so far (system prompt, earlier user turns, and
the server's own earlier answers). This is where a prefix cache pays: turn 1 of the first
conversation is cold, later turns and other conversations repeat a prefix.

  tools/py bench/run_multiturn.py --output DIR --engines "zerv-f16@parallel=8,kv-type=f16;vllm-apc;llama-fa-kvu" \\
                   [--levels 1,4,8] [--parallel 8] [--context-per-slot 16384] [--rounds 1] [--zerv-binary PATH]

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


def turn(port, body, rec):
    """One streaming turn: rec gets send and token times, usage, and the answer text."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        rec["send"] = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return ""
        times, content = [], []
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            j = json.loads(line[6:])
            if j.get("usage"): rec["usage"] = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if piece: times.append(time.perf_counter())
                if d.get("content"): content.append(d["content"])
        conn.close()
        rec["times"] = times
        text = "".join(content)
        rec["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        return text
    except Exception as e:  # noqa: BLE001 - recorded, the level is marked failed
        rec["error"] = f"{type(e).__name__}: {e}"
        rec.pop("times", None)
        return ""


def conversation(port, w, conv, recs, engine, level, rnd):
    messages = [{"role": "system", "content": w["system"]}]
    for t, user in enumerate(conv["turns"]):
        messages.append({"role": "user", "content": user})
        rec = dict(engine=engine, round=rnd, level=level, conversation=conv["name"], turn=t)
        body = dict(model="qwen3.8-27b", messages=messages, stream=True, stream_options={"include_usage": True},
                    max_tokens=w["max_tokens"], temperature=w["temperature"], seed=w["seed"], **w["options"])
        answer = turn(port, body, rec)
        recs.append(rec)
        if "error" in rec: return
        messages.append({"role": "assistant", "content": answer})


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def level_run(port, w, level, raw, engine, rnd):
    recs = []
    threads = [threading.Thread(target=conversation, args=(port, w, w["conversations"][i], recs, engine, level, rnd)) for i in range(level)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    for r in recs: raw.write(json.dumps(r) + "\n")
    out = dict(level=level, round=rnd, wall_s=wall, errors=[r["error"] for r in recs if "error" in r], turns=[])
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
    p.add_argument("--port", type=int, default=18098)
    p.add_argument("--llama-server", type=pathlib.Path, help="another llama-server build to run (default: built in the graph)")
    p.add_argument("--zerv-binary", type=pathlib.Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    a = p.parse_args()
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
    w = json.loads(a.workload.read_text())
    levels = [int(x) for x in a.levels.split(",")]
    if max(levels) > len(w["conversations"]): raise SystemExit("more conversations requested than the workload has")
    zb = a.zerv_binary.resolve()
    table = rs.engines(a.model, a.port, a.context_per_slot * a.parallel, zb)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())),
                    model_sha256=rs.sha(a.model), zerv_sha256=rs.sha(zb), llama_server=str(rs.llama_server()), llama_server_sha256=rs.sha(rs.llama_server()),
                    vllm_image=rs.VLLM_IMAGE, vllm_model=str(rs.VLLM_MODEL.relative_to(rs.ROOT)),
                    workload={str(a.workload.resolve().relative_to(ROOT)): rs.sha(a.workload)}, levels=levels,
                    parallel=a.parallel, context_per_slot=a.context_per_slot, rounds=a.rounds, engines={})
    raw = (out / "raw.jsonl").open("w")
    names = a.engines.split(";")
    summary = {n: [] for n in names}
    for rnd, name in [(r, n) for r in range(a.rounds) for n in (names if r % 2 == 0 else names[::-1])]:
        spec = dict(rs.resolve_engine(table, name, zb))
        cmd = list(spec["cmd"])
        if cmd[0] == rs.llama_server() or "/llama/bin/llama-server" in cmd:
            cmd[cmd.index("-np") + 1] = str(a.parallel)
        elif "--max-num-seqs" in cmd:
            cmd[cmd.index("--max-num-seqs") + 1] = str(a.parallel)
            cmd[cmd.index("--max-model-len") + 1] = str(a.context_per_slot)
        else:
            cmd[cmd.index("--context") + 1] = str(a.context_per_slot)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
        env.update(spec["env"])
        log = (out / (f"{name}.log" if a.rounds == 1 else f"{name}-r{rnd}.log")).open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            rs.wait_ready(a.port, proc, spec.get("ready_timeout", 600))
            manifest["engines"].setdefault(name, dict(cmd=cmd, env=spec["env"], vram_loaded=[]))["vram_loaded"].append(rs.vram_used())
            for level in levels:
                s = level_run(a.port, w, level, raw, name, rnd)
                summary[name].append(s)
                print(name, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k != "turns"}), flush=True)
                for t in s["turns"]:
                    print("   ", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()}), flush=True)
        finally:
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
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if rc.gpu_busy(): print("WARNING: a server is still running after the benchmark", file=sys.stderr)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
