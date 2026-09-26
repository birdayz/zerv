#!/usr/bin/env python3
"""Concurrent serving benchmark (block 18): aggregate throughput and per-request latency with C
simultaneous clients, over the same OpenAI Chat Completions v1 streaming client as
run_serving.py.

  run_concurrent.py --output DIR --engines A,B --concurrency 1,2,4,8 [--parallel N]
                    [--workload bench/workloads/hyperqwen-real-v1-greedy.json] [--max-tokens 512]
                    [--requests-per-client 2] [--context-per-slot 4096] [--zerv-binary PATH]

Closed loop: each of C client threads sends its next request as soon as the previous one ends;
prompts are taken round-robin from the workload (client i starts at prompt i). Every level starts
with one untimed warm-up request. Engines run one at a time (the GPU is checked to be free).
`--parallel N` gives llama-server engines `-np N` and `-c N * context-per-slot` (llama.cpp splits
the context evenly between slots). zerv engines get `--context N * context-per-slot` and serve one
request at a time, unless the engine sets its own `@parallel=M` (block 18c): then `--context` is
context-per-slot, the per-request maximum, and M requests decode together.

`--reference RAW` (a raw.jsonl of solo runs, e.g. concurrency 1 with enough requests per client to
cover every case) is the batch-invariance gate: every request's output_sha256 must equal the
reference's for its case; mismatches are listed in the summary and fail the run (exit 1).

Per level: aggregate output tokens / wall time, mean and percentile TTFT, per-request decode rate
((completion_tokens - 1) / (last delta - first delta)). Raw per-request lines in raw.jsonl,
summary.json, manifest.json (commands, hashes, host).
"""
import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys, threading, time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_serving as rs  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import host_info  # noqa: E402  (tools/host_info.py: the host, recorded without host tools)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def ancestors():
    """This process and its parents (their command lines may name a zerv binary)."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(str(pid))
        try:
            pid = int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def gpu_busy():
    found = host_info.processes(r"(^|/)(zerv(-[a-z0-9-]+)?|llama-server)( |$)|vllm serve|VLLM::")
    mine = ancestors()
    return [l for l in found if l.split()[0] not in mine]


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def run_level(port, workload, level, per_client, max_tokens, engine, raw):
    cases = workload["cases"]
    lock = threading.Lock()
    results, errors = [], []

    def client(i):
        for n in range(per_client):
            case = cases[(i + n * level) % len(cases)]
            body = dict(model="qwen3.8-27b", messages=case["messages"], stream=True, stream_options={"include_usage": True}, max_tokens=max_tokens,
                        temperature=workload["temperature"], seed=workload["seed"], **case.get("options", {}))
            try:
                r = rs.stream_request(port, body)
            except Exception as e:  # recorded, the level is marked failed
                with lock:
                    errors.append(f"client {i} request {n}: {e!r}")
                return
            ct = r["usage"]["completion_tokens"] if r["usage"] else 0
            span = r["last_delta_s"] - r["ttft_s"]
            r.update(engine=engine, concurrency=level, client=i, request=n, case=case["name"],
                     decode_tok_s=(ct - 1) / span if ct > 1 and span > 0 else None,
                     output_sha256=hashlib.sha256((r["reasoning"] + "\x00" + r["content"]).encode()).hexdigest())
            with lock:
                results.append(r)
                raw.write(json.dumps(r) + "\n")
                raw.flush()

    threads = [threading.Thread(target=client, args=(i,)) for i in range(level)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    tokens = sum(r["usage"]["completion_tokens"] for r in results if r["usage"])
    ttft = [r["ttft_s"] for r in results]
    dec = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    return dict(concurrency=level, requests=len(results), errors=errors, wall_s=wall, completion_tokens=tokens,
                aggregate_tok_s=tokens / wall if wall > 0 else None,
                _results=results, ttft_mean_s=statistics.mean(ttft) if ttft else None, ttft_p50_s=percentile(ttft, 0.5), ttft_p95_s=percentile(ttft, 0.95),
                decode_per_request_median_tok_s=statistics.median(dec) if dec else None, decode_per_request_min_tok_s=min(dec) if dec else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=pathlib.Path, default=ROOT / "models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--workload", type=pathlib.Path, default=ROOT / "bench/workloads/hyperqwen-real-v1-greedy.json")
    p.add_argument("--engines", required=True, help="engine names from run_serving.engines; `;`-separated with @ knobs")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--parallel", type=int, default=0, help="llama-server -np (default: the largest concurrency)")
    p.add_argument("--context-per-slot", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--requests-per-client", type=int, default=2)
    p.add_argument("--port", type=int, default=18097)
    p.add_argument("--zerv-binary", type=pathlib.Path, help="zerv binary (default: //src:zerv built with Bazel, --config=release)")
    p.add_argument("--reference", type=pathlib.Path, help="raw.jsonl of solo runs: per-case output_sha256 gate")
    a = p.parse_args()
    rs.require_host_gpu()
    if a.zerv_binary is None:
        sys.path.insert(0, str(ROOT/"tools"))
        import zerv_build
        a.zerv_binary = zerv_build.binary("zerv")
    reference = None
    if a.reference:
        reference = {}
        for line in a.reference.read_text().splitlines():
            r = json.loads(line)
            if reference.setdefault(r["case"], r["output_sha256"]) != r["output_sha256"]:
                raise SystemExit(f"reference is not deterministic for {r['case']}")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    busy = gpu_busy()
    if busy: raise SystemExit("GPU busy: " + "; ".join(busy))
    levels = [int(x) for x in a.concurrency.split(",")]
    parallel = a.parallel or max(levels)
    workload = json.loads(a.workload.read_text())
    zerv_binary = a.zerv_binary.resolve()
    table = rs.engines(a.model, a.port, a.context_per_slot * parallel, zerv_binary)
    names = a.engines.split(";") if "@" in a.engines else a.engines.split(",")
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())),
                    model_sha256=rs.sha(a.model), workload_sha256=rs.sha(a.workload), zerv_sha256=rs.sha(zerv_binary),
                    llama_server_sha256=rs.sha(rs.llama_server()), vllm_image=rs.VLLM_IMAGE, vllm_model=str(rs.VLLM_MODEL.relative_to(rs.ROOT)), levels=levels, parallel=parallel, max_tokens=a.max_tokens,
                    requests_per_client=a.requests_per_client, context_per_slot=a.context_per_slot, engines={},
                    client="closed loop, python http.client streaming SSE (run_serving.stream_request)")
    raw = (out / "raw.jsonl").open("w")
    summary = {}
    gate_failed = False
    for name in names:
        spec = dict(rs.resolve_engine(table, name, zerv_binary))
        cmd = list(spec["cmd"])
        if cmd[0] == rs.llama_server() or "/llama/bin/llama-server" in cmd:
            cmd[cmd.index("-np") + 1] = str(parallel)
        elif "--parallel" in cmd:
            cmd[cmd.index("--context") + 1] = str(a.context_per_slot)
        elif "--max-num-seqs" in cmd:  # vLLM: shared paged KV pool, per-request context
            cmd[cmd.index("--max-num-seqs") + 1] = str(parallel)
            cmd[cmd.index("--max-model-len") + 1] = str(a.context_per_slot)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "RADV_"))}
        env.update(spec["env"])
        log = (out / f"{name}.log").open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            rs.wait_ready(a.port, proc, spec.get("ready_timeout", 600))
            manifest["engines"][name] = dict(cmd=cmd, env=spec["env"], vram_loaded=rs.vram_used())
            summary[name] = []
            for level in levels:
                warm = workload["cases"][0]
                rs.stream_request(a.port, dict(model="qwen3.8-27b", messages=warm["messages"], stream=True, stream_options={"include_usage": True}, max_tokens=32,
                                               temperature=workload["temperature"], seed=workload["seed"], **warm.get("options", {})))
                s = run_level(a.port, workload, level, a.requests_per_client, a.max_tokens, name, raw)
                if reference is not None:
                    s["mismatches"] = [dict(client=r["client"], request=r["request"], case=r["case"]) for r in s.pop("_results")
                                       if reference.get(r["case"]) != r["output_sha256"]]
                    gate_failed = gate_failed or bool(s["mismatches"])
                else:
                    s.pop("_results")
                summary[name].append(s)
                print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("errors", "mismatches")}), "errors" if s["errors"] else "",
                      f"mismatches {len(s['mismatches'])}" if "mismatches" in s else "", flush=True)
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
    if gpu_busy(): print("WARNING: a server is still running after the benchmark", file=sys.stderr)
    if reference is not None:
        print("batch-invariance gate:", "FAILED" if gate_failed else "passed", flush=True)
        if gate_failed: sys.exit(1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
