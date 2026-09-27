#!/usr/bin/env python3
"""Matched serving benchmark: native zerv vs tuned llama-server over the same
OpenAI Chat Completions v1 streaming client, same artifact, prompts and greedy
settings. Records raw per-request timings, output hashes/texts, VRAM and RSS.
Engines run one at a time; nothing else should use the GPU meanwhile."""
import argparse
from datetime import datetime, timezone
import functools
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

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
TEMPLATE = ROOT/"third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/official-template.jinja"


LLAMA_SERVER_OVERRIDE = None  # --llama-server: another build to compare (recorded by hash)


def llama_server():
    """llama-server with the Vulkan backend, built in the graph from the pinned llama.cpp and
    ggml (@llama_cpp//:llama-server; docs/specs/hermetic-build.md phase 5): the path of the
    built executable (Bazel builds it on first use), unless --llama-server names another."""
    return LLAMA_SERVER_OVERRIDE or _built_llama_server()


def require_host_gpu():
    """The gate of every serving benchmark: the GPU tests on the host's driver, which the
    benchmark measures on (docs/specs/hermetic-build.md, "Host-driver GPU tests"; cached while
    driver and code are unchanged)."""
    sys.path.insert(0, str(ROOT/"tools"))
    import zerv_build
    zerv_build.test_host_gpu()


@functools.cache
def _built_llama_server():
    sys.path.insert(0, str(ROOT/"tools"))
    import zerv_build
    return str(zerv_build.binary("llama-server", config=None))


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def vram_used():
    for card in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        return int(card.read_text())
    return None


# vLLM competitor (docs/bench/2026-09-25-vllm.md): the project's official ROCm image, pinned by
# digest (v0.30.0, built for gfx1100), and RedHatAI's W4A16 checkpoint (vLLM's own quantization
# org; not the GGUF's weights: speed comparisons only, quality separately). Downloads were
# hash-verified and malware-scanned before first use.
VLLM_IMAGE = "vllm/vllm-openai-rocm@sha256:2e7da1ad1c66836802072588adea75f9f4991da5f9545b4318e91d422c22ce6a"
VLLM_MODEL = ROOT/"models/RedHatAI/Qwen3.8-27B-INT4/c063053e004e9783631651df95cf55d0bbf88b32"
VLLM_CACHE = ROOT/"third_party/vllm/cache"


def vllm_engine(port, context, extra=(), prefix_cache=False):
    """`docker run` of vLLM with the GPU device nodes and nothing else: unprivileged user, no
    capabilities, no new privileges, weights and template read-only, compile caches in
    third_party/vllm/cache, the API on 127.0.0.1 only, no telemetry and no Hub access. Same
    chat template file as llama-server (byte-identical to the checkpoint's). `--max-num-seqs`
    and `--max-model-len` are the concurrency and per-request context (run_multiuser sets
    them); prefix caching is off (cold prefill, like the other engines' runs) unless
    `prefix_cache` (vLLM's default: automatic prefix caching, engine `vllm-apc`)."""
    name = f"zerv-bench-vllm-{port}"
    VLLM_CACHE.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}", "--device", "/dev/kfd", "--device", "/dev/dri/renderD128",
           "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--shm-size", "16g", "-p", f"127.0.0.1:{port}:{port}",
           "-v", f"{VLLM_MODEL}:/model:ro", "-v", f"{TEMPLATE}:/template.jinja:ro", "-v", f"{VLLM_CACHE}:/cache",
           "-e", "HOME=/cache", "-e", "USER=bench", "-e", "LOGNAME=bench", "-e", "HF_HOME=/cache/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1", VLLM_IMAGE,
           "/model", "--served-model-name", "qwen3.8-27b", "--host", "0.0.0.0", "--port", str(port),
           "--max-model-len", str(context), "--max-num-seqs", "1", "--gpu-memory-utilization", "0.95",
           "--chat-template", "/template.jinja", "--limit-mm-per-prompt", '{"image":0,"video":0}', "--reasoning-parser", "qwen3",
           *([] if prefix_cache else ["--no-enable-prefix-caching"]), *extra]
    return dict(cmd=cmd, env={}, stop=["docker", "rm", "-f", name], ready_timeout=1800)


# llama.cpp-RDNA3-7900xtx-opt (github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt @ 15995a12, MIT):
# built from source for gfx1100 in the pinned vLLM ROCm image by tools/build_competitor_rdna3.py
# (pinned source archive and image digest, no network, our uid). Run in that image with the
# same restrictions as vLLM: our uid, no capabilities, GPU device nodes only, build and model
# read-only, API on 127.0.0.1.
RDNA3_BUILD = ROOT/"third_party/competitors/rdna3-15995a12"


def rdna3_engine(model, port, context, extra):
    name = f"zerv-bench-rdna3-{port}"
    cmd = ["docker", "run", "--rm", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}", "--device", "/dev/kfd", "--device", "/dev/dri/renderD128",
           "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "-p", f"127.0.0.1:{port}:{port}",
           "-v", f"{RDNA3_BUILD}:/llama:ro", "-v", f"{Path(model).resolve()}:/model.gguf:ro", "-v", f"{TEMPLATE}:/template.jinja:ro",
           "-e", "HOME=/tmp", "-e", "LD_LIBRARY_PATH=/llama/bin:/opt/rocm/lib", "--entrypoint", "/llama/bin/llama-server", VLLM_IMAGE,
           "-m", "/model.gguf", "--host", "0.0.0.0", "--port", str(port), "-c", str(context), "-np", "1", "-ngl", "99",
           "--no-context-shift", "--no-webui", "--jinja", "--chat-template-file", "/template.jinja",
           "--reasoning-format", "deepseek", "--cache-ram", "0", "-a", "qwen3.8-27b", "-fa", "on", *extra]
    return dict(cmd=cmd, env={}, stop=["docker", "rm", "-f", name], ready_timeout=900)


def engines(model, port, context, zerv_binary):
    common = [llama_server(), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(context), "-np", "1", "-ngl", "99",
              "--spec-type", "none", "--no-context-shift", "--no-webui", "--jinja", "--chat-template-file", str(TEMPLATE),
              "--reasoning-format", "deepseek", "--cache-ram", "0", "-a", "qwen3.8-27b"]
    return {
        # Plain decode (speculation is the server default since 2026-09-24; these keep it off).
        "zerv": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--spec-draft", "0"], env={}),
        # Explicit f16 prompt projections (block 14; matched to llama's fast-path arithmetic class).
        "zerv-f16": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--prefill-precision", "f16", "--spec-draft", "0"], env={}),
        # Lossless MTP speculative decoding (block 17b), N drafts per step; the verify count
        # is adaptive (the server default) or every draft (`-fixed`).
        **{f"zerv-spec{n}": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--spec-draft", str(n)], env={})
           for n in (1, 2, 3, 4)},
        **{f"zerv-spec{n}-fixed": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--spec-draft", str(n), "--spec-policy", "fixed"], env={})
           for n in (1, 2, 3, 4)},
        **{f"zerv-f16-spec{n}": dict(cmd=[str(zerv_binary), "--model", str(model), "--port", str(port), "--context", str(context), "--prefill-precision", "f16", "--spec-draft", str(n)], env={})
           for n in (1, 2, 3, 4)},
        # Best observed llama.cpp Vulkan configuration family (tuned below by the sweep).
        "llama-fa-ub512": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={}),
        # Smaller prompt batch per server step: less stall for running requests (multi-user).
        "llama-fa-b512": dict(cmd=common+["-fa", "on", "-b", "512", "-ub", "512"], env={}),
        # One KV buffer shared by all slots (-kvu): each sequence may use the whole -c.
        "llama-fa-kvu": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512", "-kvu"], env={}),
        "vllm": vllm_engine(port, context),
        "vllm-apc": vllm_engine(port, context, prefix_cache=True),
        "rdna3": rdna3_engine(model, port, context, ["--spec-type", "none", "-b", "2048", "-ub", "512"]),
        "rdna3-b512": rdna3_engine(model, port, context, ["--spec-type", "none", "-b", "512", "-ub", "512"]),
        "rdna3-mtp3": rdna3_engine(model, port, context, ["--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "3", "-b", "2048", "-ub", "512"]),
        "vllm-mtp3": vllm_engine(port, context, ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']),
        # Prefill chunks of 512 tokens (default 2048): shorter stalls for running requests.
        "vllm-b512": vllm_engine(port, context, ["--max-num-batched-tokens", "512"]),
        "llama-fa-ub256": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "256"], env={}),
        "llama-nofa-ub512": dict(cmd=common+["-fa", "off", "-b", "2048", "-ub", "512"], env={}),
        # Precision ladder (block 14 research; the pipelines used are logged via GGML_VK_PIPELINE_STATS):
        # default = f16 activations, f16 accumulation (coopmat); nocoopmat = Q8_1 activations with
        # integer dot products (exact int32 block sums); nof16 = coopmat with f32 accumulation?
        "llama-fa-ub512-nocoopmat": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        "llama-fa-ub512-nof16": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_F16": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        "llama-fa-ub512-noint-nocoopmat": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512"], env={"GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_PIPELINE_STATS": "matmul"}),
        # Speculative decoding (block 17 competitor): the GGUF's MTP (nextn) layer as drafter,
        # N draft tokens per step (llama.cpp default N = 3), or n-gram lookup (no draft model).
        **{f"llama-fa-ub512-mtp{n}": dict(cmd=common[:common.index("--spec-type")]+common[common.index("--spec-type")+2:]
                                          +["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "-fa", "on", "-b", "2048", "-ub", "512"], env={})
           for n in (1, 2, 3, 4)},
        "llama-fa-ub512-ngram": dict(cmd=common[:common.index("--spec-type")]+common[common.index("--spec-type")+2:]
                                     +["--spec-type", "ngram-mod", "-fa", "on", "-b", "2048", "-ub", "512"], env={}),
        # Partial precision control: FP32 activations into decode matvecs and FP32 KV only.
        "llama-f32": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512", "-ctk", "f32", "-ctv", "f32"], env={"GGML_VK_DISABLE_MMVQ": "1"}),
        # Full FP32 control: also no Q8_1 integer-dot prompt matmuls, no FP16/cooperative matrices.
        "llama-fp32-full": dict(cmd=common+["-fa", "on", "-b", "2048", "-ub", "512", "-ctk", "f32", "-ctv", "f32"],
                                env={"GGML_VK_DISABLE_MMVQ": "1", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_DISABLE_COOPMAT": "1", "GGML_VK_DISABLE_F16": "1"}),
    }


# A server that stops sending (e.g. llama-server's ngram-mod stalled mid-generation on
# 2026-09-24 and the old 3600 s socket timeout kept the run waiting for about an hour) must
# fail the request, not hang the benchmark. `stall_s`: longest silence between bytes (covers
# the longest prefill); `limit_s`: whole request.
STALL_S = 300
LIMIT_S = 1800


class RequestStalled(RuntimeError):
    pass


def resolve_engine(table, name, zerv_binary):
    """An engine by name. `BASE@flag=value,flag2=value2` appends `--flag value ...` to a zerv
    engine's command (knob A/B runs, e.g. `zerv-f16@embedding-memory=device`)."""
    base, _, extra = name.partition("@")
    spec = dict(table[base])
    if extra:
        if spec["cmd"][0] != str(zerv_binary): raise SystemExit(f"{name}: @flags apply to zerv engines only")
        flags = []
        for item in extra.split(","):
            flag, sep, value = item.partition("=")
            if not sep or not flag or not value: raise SystemExit(f"{name}: expected flag=value, got {item!r}")
            flags += [f"--{flag}", value]
        spec["cmd"] = spec["cmd"]+flags
    return spec


def stream_request(port, body, stall_s=None, limit_s=None):
    stall_s, limit_s = stall_s or STALL_S, limit_s or LIMIT_S
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=stall_s)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    try:
        resp = conn.getresponse()
    except TimeoutError as e:
        conn.close()
        raise RequestStalled(f"no response headers within {stall_s} s") from e
    if resp.status != 200: raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]}")
    first = last = None
    reasoning, content, usage, finish, deltas, timings = [], [], None, None, 0, None
    buf = b""
    while True:
        try:
            chunk = resp.read1(65536)
        except TimeoutError as e:
            conn.close()
            raise RequestStalled(f"no data for {stall_s} s after {time.perf_counter()-start:.1f} s") from e
        if time.perf_counter()-start > limit_s:
            conn.close()
            raise RequestStalled(f"request exceeded {limit_s} s")
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
            if j.get("timings"): timings = j["timings"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                r, c = d.get("reasoning_content") or d.get("reasoning") or "", d.get("content") or ""  # vLLM >= 0.11 streams `reasoning`
                if r or c:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    deltas += 1
                reasoning.append(r); content.append(c)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    end = time.perf_counter()
    conn.close()
    result = dict(ttft_s=(first or end)-start, total_s=end-start, last_delta_s=(last or end)-start, deltas=deltas, usage=usage, finish=finish,
                  reasoning="".join(reasoning), content="".join(content))
    # llama-server's speculative counters for this request (its final chunk's timings).
    if timings and "draft_n" in timings:
        result["spec"] = dict(drafted=timings["draft_n"], verified=timings["draft_n"], accepted=timings["draft_n_accepted"])
    return result


def spec_counters(metrics_text):
    """zerv's speculative counters from a /metrics body (None when absent)."""
    names = {"zerv_spec_verifies_total": "verifies", 'zerv_spec_draft_tokens_total{stage="drafted"}': "drafted",
             'zerv_spec_draft_tokens_total{stage="verified"}': "verified", 'zerv_spec_draft_tokens_total{stage="accepted"}': "accepted"}
    found = {}
    for line in metrics_text.splitlines():
        key, _, value = line.rpartition(" ")
        if key in names: found[names[key]] = int(value)
    return found if len(found) == len(names) else None


def zerv_spec(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        return spec_counters(resp.read().decode()) if resp.status == 200 else None
    except OSError:
        return None
    finally:
        conn.close()


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
    p.add_argument("--engines", default="zerv,llama-fa-ub512,llama-fp32-full",
                   help="comma-separated engine names; with any `@` knob suffix (see resolve_engine) separate them with `;`")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--zerv-binary", type=Path, help="reuse a previously benchmarked zerv binary (no rebuild; its hash is recorded)")
    p.add_argument("--llama-server", type=Path, help="another llama-server build to run (default: built in the graph; its hash is recorded)")
    p.add_argument("--rdna3-build", type=Path, help="another build directory of the RDNA3 fork (bin/llama-server; default: tools/build_competitor_rdna3.py's)")
    p.add_argument("--no-prompt-cache", action="store_true", help="cold prefill every request: zerv --prefix-cache-slots 0, llama cache_prompt=false")
    p.add_argument("--stall-timeout", type=float, default=STALL_S, help="fail a request after this many seconds without data")
    p.add_argument("--request-timeout", type=float, default=LIMIT_S, help="fail a request after this many seconds in total")
    a = p.parse_args()
    require_host_gpu()
    global LLAMA_SERVER_OVERRIDE, RDNA3_BUILD
    if a.llama_server: LLAMA_SERVER_OVERRIDE = str(a.llama_server.resolve(strict=True))
    if a.rdna3_build: RDNA3_BUILD = a.rdna3_build.resolve(strict=True)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if sha(a.model) != MODEL_SHA: raise SystemExit("model mismatch")
    # The competitors' chat template: the official template, pinned by the chat fixture.
    if sha(TEMPLATE) != json.loads((ROOT/"tests/fixtures/chat-template.json").read_text())["official_template_sha256"]:
        raise SystemExit(f"{TEMPLATE}: not the pinned official template")
    workload = json.loads(a.workload.read_text())
    if a.zerv_binary:
        build = ["reused", str(a.zerv_binary.resolve()), sha(a.zerv_binary)]
        source = a.zerv_binary.resolve()
    else:
        sys.path.insert(0, str(ROOT/"tools")); import zerv_build  # noqa: E402  (lazy: tests import this module)
        build = zerv_build.build_command("zerv")
        source = zerv_build.binary("zerv")
    # Keyed by the output's parent and name (the output directory itself must be fresh).
    artifact = ROOT/"third_party/serving-bench"/out.parent.name/out.name; artifact.mkdir(parents=True, exist_ok=False)
    zerv_binary = artifact/"zerv"; shutil.copy2(source, zerv_binary)
    table = engines(a.model, a.port, a.context, zerv_binary)
    if a.no_prompt_cache:
        for spec in table.values():
            if spec["cmd"][0] == str(zerv_binary): spec["cmd"] = spec["cmd"]+["--prefix-cache-slots", "0"]
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, host=platform.uname()._asdict(), model_sha256=MODEL_SHA,
                    workload_sha256=sha(a.workload), zerv_sha256=sha(zerv_binary), llama_server_sha256=sha(llama_server()), vllm_image=VLLM_IMAGE, vllm_model=str(VLLM_MODEL.relative_to(ROOT)),
                    llama_version=subprocess.run([llama_server(), "--version"], capture_output=True, text=True).stderr.strip(),
                    template_sha256=sha(TEMPLATE), build=build, context=a.context, repeats=a.repeats, engines={}, vram_before=vram_used(),
                    rdna3_build=dict(path=str(RDNA3_BUILD), llama_server_sha256=sha(RDNA3_BUILD/"bin/llama-server"))
                    if (RDNA3_BUILD/"bin/llama-server").exists() else None,
                    client="python http.client streaming SSE; TTFT = first reasoning/content delta; decode rate = (completion_tokens-1)/(last_delta-first_delta)")
    raw = (out/"raw.jsonl").open("w")
    for name in a.engines.split(";" if "@" in a.engines else ","):
        spec = resolve_engine(table, name, zerv_binary)
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
            wait_ready(a.port, proc, spec.get("ready_timeout", 600))
            load_s = time.perf_counter()-t0
            loaded_vram = vram_used()
            # Warmup: one short request per case.
            nocache = dict(cache_prompt=False) if a.no_prompt_cache else {}
            for case in workload["cases"]:
                stream_request(a.port, dict(model="qwen3.8-27b", messages=case["messages"], stream=True, max_tokens=8, temperature=0, seed=workload["seed"], **nocache, **case["options"]))
            for rep in range(a.repeats):
                for case in workload["cases"]:
                    body = dict(model="qwen3.8-27b", messages=case["messages"], stream=True, stream_options={"include_usage": True},
                                max_tokens=case["max_tokens"], temperature=workload["temperature"], seed=workload["seed"], **nocache, **case["options"])
                    before = zerv_spec(a.port) if spec["cmd"][0] == str(zerv_binary) else None
                    try:
                        r = stream_request(a.port, body, a.stall_timeout, a.request_timeout)
                    except RequestStalled as e:
                        # Recorded, and the engine is abandoned (its state is unknown).
                        failure = dict(engine=name, case=case["name"], repeat=rep, error=str(e))
                        raw.write(json.dumps(failure)+"\n"); raw.flush()
                        manifest.setdefault("failures", []).append(failure)
                        print(name, case["name"], rep, "FAILED:", e, flush=True)
                        raise
                    after = zerv_spec(a.port) if before else None
                    if before and after and after["verifies"] > before["verifies"]:
                        r["spec"] = {k: after[k]-before[k] for k in after}
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
        except RequestStalled:
            manifest["engines"][name] = dict(cmd=spec["cmd"], env=spec["env"], failed=True)
        finally:
            stop.set(); poller.join()
            proc.send_signal(signal.SIGINT)
            try: proc.wait(timeout=60)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            if spec.get("stop"): subprocess.run(spec["stop"], capture_output=True)
            log.close()
            time.sleep(3)
    raw.close()
    rows = [r for r in (json.loads(line) for line in (out/"raw.jsonl").read_text().splitlines()) if "error" not in r]
    summary = {}
    for r in rows:
        s = summary.setdefault(r["engine"], {}).setdefault(r["case"], dict(ttft_ms=[], total_s=[], decode_tok_s=[], completion_tokens=[], prompt_tokens=[], outputs=set()))
        s["ttft_ms"].append(r["ttft_s"]*1000); s["total_s"].append(r["total_s"])
        if r["decode_tok_s"]: s["decode_tok_s"].append(r["decode_tok_s"])
        s["completion_tokens"].append(r["usage"]["completion_tokens"]); s["prompt_tokens"].append(r["usage"]["prompt_tokens"]); s["outputs"].add(r["output_sha256"])
        if r.get("spec"):
            acc = s.setdefault("spec", dict(drafted=0, verified=0, accepted=0))
            for k in acc: acc[k] += r["spec"][k]
    for engine in summary.values():
        for case, s in engine.items():
            engine[case] = {k: (dict(median=statistics.median(v), min=min(v), max=max(v)) if isinstance(v, list) and v and not isinstance(v[0], str) else sorted(v) if isinstance(v, set) else v) for k, v in s.items()}
            if "spec" in s: s["spec"]["acceptance"] = s["spec"]["accepted"]/s["spec"]["verified"] if s["spec"]["verified"] else None
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out/"summary.json").write_text(json.dumps(summary, indent=1)+"\n")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
