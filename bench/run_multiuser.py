#!/usr/bin/env python3
"""Multi-user latency benchmark (block 18): what each user sees while others share the server.

  run_multiuser.py --output DIR --engines "zerv-f16@parallel=8,kv-type=f16;llama-fa-ub512" \\
                   [--parallel 8] [--context-per-slot 8192] [--reps 3] [--rounds 1] [--zerv-binary PATH]

Engines run one at a time (GPU checked free before and after). With --rounds R every engine is
started R times, the order alternating per round (ABBA), and runs all scenarios each time
(steady once per level, interference and queue --reps times). llama-server engines get
`-np P -c P*context-per-slot`; zerv engines must set `@parallel=P` themselves and get
`--context context-per-slot`.

Scenarios (fixed prompts, greedy, fixed seed; every stream records the arrival time of every
token-bearing SSE delta):

steady      C = 1, 2, 4, 8 closed-loop clients, short chat prompts (HyperQwen real prompts),
            256 tokens each, 2 requests per client. Aggregate tok/s, TTFT, inter-token gaps.
interference  P - 2 users stream long answers (short prompts); once each has 20 tokens, a
            4,936-token prompt arrives (needle-long-6k, 64 tokens out), and 100 ms later a short
            prompt (280 tokens, 32 out). Both get a slot (P - 2 + 2 = P). Measured per repetition: the running users' inter-token
            gaps while the long prompt is prefilled (its send to its first token) against their
            gaps before it; the long prompt's TTFT; the short prompt's TTFT (head-of-line
            blocking behind the long one).

queue       P + 2 users: P stream long answers, 2 more arrive and wait for a slot. Their TTFT
            (queueing included) and what the running users see.

Token gaps: a delta may carry several tokens (a server holds back text that could start a
stop string or completes a UTF-8 sequence); a request with fewer deltas than tokens is left
out of the gap statistics (both engines, same rule; the count is reported). Reported per
level: p50/p99/max gap, the share of gaps above 1.5x the median ("stalled") and their mean,
and per-user decode rates (slowest and median: fairness).

Outputs: raw.jsonl (one line per stream with its token times), summary.json, manifest.json
(commands, binary and workload hashes, host), and each server's log.
"""
import argparse, hashlib, http.client, json, os, pathlib, statistics, subprocess, sys, threading, time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_serving as rs  # noqa: E402
import run_concurrent as rc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHORT = ROOT / "bench/workloads/hyperqwen-real-v1-greedy.json"
LONG = ROOT / "bench/workloads/long-6k-v1.json"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def stream(port, body, rec, started=None):
    """POST a streaming chat request; rec gets send time, token times, usage and output, or
    `error` (HTTP status or a connection failure; the scenario then reports the run as failed)."""
    try:
        _stream(port, body, rec, started)
    except Exception as e:  # noqa: BLE001 - recorded, the run is marked failed
        rec["error"] = f"{type(e).__name__}: {e}"
        rec.pop("times", None)
        if started is not None: started.release()


def _stream(port, body, rec, started):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    rec["send"] = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        rec["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
        conn.close()
        return
    times, text = [], []
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            rec["usage"] = j["usage"]
        for ch in j.get("choices", []):
            d = ch.get("delta", {})
            piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
            if piece:
                times.append(time.perf_counter())
                text.append(piece)
                if started is not None and len(times) == 20:
                    started.release()
    conn.close()
    rec["times"] = times
    rec["output_sha256"] = hashlib.sha256("".join(text).encode()).hexdigest()


# Request fields for every engine; main() sets cache_prompt=False unless --prompt-cache on
# (llama-server reuses a slot's KV for a repeated prefix; zerv --parallel N has no prefix cache).
EXTRA = {}


def body(case, workload, max_tokens):
    return dict(model="qwen3.8-27b", messages=case["messages"], stream=True, stream_options={"include_usage": True},
                max_tokens=max_tokens, temperature=workload["temperature"], seed=workload["seed"], **EXTRA, **case.get("options", {}))


def exact(r):
    """Every delta carried one token (see the module doc)."""
    return r.get("usage") and r.get("times") and r["usage"]["completion_tokens"] == len(r["times"])


def gap_stats(gs, prefix):
    gs = sorted(gs)
    if not gs: return {}
    base = gs[len(gs) // 2]
    stalled = [g for g in gs if g > 1.5 * base]
    return {f"{prefix}p50_ms": base, f"{prefix}p99_ms": pct(gs, .99), f"{prefix}max_ms": gs[-1],
            f"{prefix}stalled_pct": 100 * len(stalled) / len(gs), f"{prefix}stalled_mean_ms": statistics.mean(stalled) if stalled else 0.0}


def rate(r):
    ts = r["times"]
    return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None


def gaps(times, lo=None, hi=None):
    """Inter-token gaps (ms) whose later token falls in [lo, hi)."""
    out = []
    for a, b in zip(times, times[1:]):
        if (lo is None or b >= lo) and (hi is None or b < hi):
            out.append((b - a) * 1000)
    return out


def steady(port, short, level, raw, engine, rnd=0):
    recs = [dict() for _ in range(level * 2)]

    def client(i):
        for n in range(2):
            case = short["cases"][(i + n * level) % len(short["cases"])]
            r = recs[i * 2 + n]
            r.update(engine=engine, scenario="steady", round=rnd, level=level, client=i, case=case["name"])
            stream(port, body(case, short, 256), r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client, args=(i,)) for i in range(level)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    for r in recs: raw.write(json.dumps(r) + "\n")
    errors = [r["error"] for r in recs if "error" in r]
    ttft = [(r["times"][0] - r["send"]) * 1000 for r in recs if r.get("times")]
    itl = [g for r in recs if exact(r) for g in gaps(r["times"])]
    rates = sorted(x for x in (rate(r) for r in recs if r.get("times")) if x)
    tokens = sum(r["usage"]["completion_tokens"] for r in recs if r.get("usage"))
    return dict(level=level, round=rnd, errors=errors, aggregate_tok_s=tokens / wall, ttft_p50_ms=pct(ttft, .5), ttft_p95_ms=pct(ttft, .95),
                **gap_stats(itl, "itl_"), itl_excluded_requests=sum(1 for r in recs if r.get("times") and not exact(r)),
                user_tok_s_min=rates[0] if rates else None, user_tok_s_median=pct(rates, .5))


def interference(port, short, long_w, users, raw, engine, rep):
    bg = [dict(engine=engine, scenario="interference", rep=rep, role="background", case=short["cases"][i % 7]["name"]) for i in range(users)]
    started = threading.Semaphore(0)
    threads = [threading.Thread(target=stream, args=(port, body(short["cases"][i % 7], short, 512), bg[i], started)) for i in range(users)]
    for t in threads: t.start()
    for _ in range(users): started.acquire()
    long_rec = dict(engine=engine, scenario="interference", rep=rep, role="long", case=long_w["cases"][0]["name"])
    short_rec = dict(engine=engine, scenario="interference", rep=rep, role="short", case=short["cases"][7]["name"])
    tl = threading.Thread(target=stream, args=(port, body(long_w["cases"][0], long_w, 64), long_rec))
    ts = threading.Thread(target=stream, args=(port, body(short["cases"][7], short, 32), short_rec))
    tl.start()
    time.sleep(0.1)
    ts.start()
    for t in threads + [tl, ts]: t.join()
    for r in bg + [long_rec, short_rec]: raw.write(json.dumps(r) + "\n")
    errors = [r["error"] for r in bg + [long_rec, short_rec] if "error" in r]
    if errors: return dict(rep=rep, errors=errors)
    lo, hi = long_rec["send"], long_rec["times"][0]
    ex = [r for r in bg if exact(r)]
    during = [g for r in ex for g in gaps(r["times"], lo, hi)]
    before = [g for r in ex for g in gaps(r["times"], None, lo)]
    tokens_during = sum(1 for r in bg for t in r["times"] if lo <= t < hi)
    return dict(rep=rep, errors=[], long_prompt_tokens=long_rec["usage"]["prompt_tokens"], short_prompt_tokens=short_rec["usage"]["prompt_tokens"],
                long_ttft_ms=(hi - lo) * 1000, short_ttft_ms=(short_rec["times"][0] - short_rec["send"]) * 1000,
                bg_itl_before_p50_ms=pct(before, .5), bg_itl_during_p50_ms=pct(during, .5), bg_itl_during_p99_ms=pct(during, .99),
                bg_itl_during_max_ms=max(during) if during else None, bg_excluded_requests=len(bg) - len(ex),
                bg_tokens_during=tokens_during, bg_tok_s_during=tokens_during / (hi - lo) if hi > lo else None)


def queue(port, short, users, extra, raw, engine, rep):
    """`users` stream 512 tokens; once all have 20 tokens `extra` more arrive (no slot free)."""
    bg = [dict(engine=engine, scenario="queue", rep=rep, role="running", case=short["cases"][i % 7]["name"]) for i in range(users)]
    started = threading.Semaphore(0)
    threads = [threading.Thread(target=stream, args=(port, body(short["cases"][i % 7], short, 512), bg[i], started)) for i in range(users)]
    for t in threads: t.start()
    for _ in range(users): started.acquire()
    late = [dict(engine=engine, scenario="queue", rep=rep, role="queued", case=short["cases"][(i + 3) % 7]["name"]) for i in range(extra)]
    lt = [threading.Thread(target=stream, args=(port, body(short["cases"][(i + 3) % 7], short, 64), late[i])) for i in range(extra)]
    for t in lt: t.start()
    for t in threads + lt: t.join()
    for r in bg + late: raw.write(json.dumps(r) + "\n")
    errors = [r["error"] for r in bg + late if "error" in r]
    if errors: return dict(rep=rep, errors=errors)
    first_end = min(r["times"][-1] for r in bg)
    return dict(rep=rep, errors=[], queued_ttft_ms=[(r["times"][0] - r["send"]) * 1000 for r in late],
                queued_ttft_after_first_free_ms=[(r["times"][0] - first_end) * 1000 for r in late],
                **gap_stats([g for r in bg if exact(r) for g in gaps(r["times"])], "running_itl_"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=pathlib.Path, default=ROOT / "models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--engines", required=True, help="`;`-separated engine names from run_serving.engines, with @ knobs")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--context-per-slot", type=int, default=8192)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--rounds", type=int, default=1, help="start every engine this many times, order alternating per round")
    p.add_argument("--prompt-cache", choices=("on", "off"), default="off", help="off: cache_prompt=false in every request (cold prefill)")
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--port", type=int, default=18098)
    p.add_argument("--zerv-binary", type=pathlib.Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    a = p.parse_args()
    if a.zerv_binary is None:
        sys.path.insert(0, str(ROOT/"tools"))
        import zerv_build
        a.zerv_binary = zerv_build.binary("zerv")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if a.prompt_cache == "off": EXTRA["cache_prompt"] = False
    busy = rc.gpu_busy()
    if busy: raise SystemExit("GPU busy: " + "; ".join(busy))
    short, long_w = json.loads(SHORT.read_text()), json.loads(LONG.read_text())
    zb = a.zerv_binary.resolve()
    table = rs.engines(a.model, a.port, a.context_per_slot * a.parallel, zb)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())),
                    model_sha256=rs.sha(a.model), zerv_sha256=rs.sha(zb), llama_server_sha256=rs.sha(rs.LLAMA_SERVER), vllm_image=rs.VLLM_IMAGE, vllm_model=str(rs.VLLM_MODEL.relative_to(rs.ROOT)),
                    workloads={str(SHORT.relative_to(ROOT)): rs.sha(SHORT), str(LONG.relative_to(ROOT)): rs.sha(LONG)},
                    parallel=a.parallel, context_per_slot=a.context_per_slot, reps=a.reps, rounds=a.rounds, prompt_cache=a.prompt_cache, engines={})
    raw = (out / "raw.jsonl").open("w")
    names = a.engines.split(";")
    summary = {n: dict(steady=[], interference=[], queue=[]) for n in names}
    for rnd, name in [(r, n) for r in range(a.rounds) for n in (names if r % 2 == 0 else names[::-1])]:
        spec = dict(rs.resolve_engine(table, name, zb))
        cmd = list(spec["cmd"])
        if cmd[0] == str(rs.LLAMA_SERVER) or "/llama/bin/llama-server" in cmd:
            cmd[cmd.index("-np") + 1] = str(a.parallel)
        elif "--max-num-seqs" in cmd:  # vLLM: shared paged KV pool, per-request context
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
            warm = {}
            stream(a.port, body(short["cases"][0], short, 16), warm)
            res = summary[name]
            for level in [int(x) for x in a.levels.split(",")]:
                s = steady(a.port, short, level, raw, name, rnd)
                res["steady"].append(s)
                print(name, "steady", json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()}), flush=True)
            for rep in range(rnd * a.reps, (rnd + 1) * a.reps):
                s = interference(a.port, short, long_w, a.parallel - 2, raw, name, rep)
                res["interference"].append(s)
                print(name, "interference", json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()}), flush=True)
            for rep in range(rnd * a.reps, (rnd + 1) * a.reps):
                s = queue(a.port, short, a.parallel, 2, raw, name, rep)
                res["queue"].append(s)
                print(name, "queue", json.dumps({k: (round(v, 1) if isinstance(v, float) else [round(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else v) for k, v in s.items()}), flush=True)
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
    main()
