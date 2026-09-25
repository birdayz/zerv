#!/usr/bin/env python3
"""Largest context per memory configuration: start `zerv --context max` with each knob set,
record the context the server chose, its VRAM accounting and the VRAM actually in use,
and serve one short request to show it works. One server at a time; nothing else should
use the GPU (docs/specs/model.md, `--context max`).

Usage: bench/context_max.py --output DIR [--binary zig-out/bin/zerv] [--configs NAME,...]"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_serving  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Knob sets, from the fastest defaults to the most memory-saving.
CONFIGS = {
    "default": [],
    "no-spec": ["--spec-draft", "0"],
    "host-snapshots": ["--prefix-cache-memory", "host"],
    "no-prefix-cache": ["--prefix-cache-slots", "0"],
    "no-prefix-cache-no-spec": ["--prefix-cache-slots", "0", "--spec-draft", "0"],
    "host-snapshots-chunk256": ["--prefix-cache-memory", "host", "--prefill-chunk", "256"],
    "device-embedding": ["--embedding-memory", "device"],
    # No reserve for other processes (the default leaves 1024 MiB, as llama's --fit-target).
    "default-reserve0": ["--vram-reserve-mib", "0"],
    "no-prefix-cache-no-spec-reserve0": ["--prefix-cache-slots", "0", "--spec-draft", "0", "--vram-reserve-mib", "0"],
    # f16 KV cache (block 17c).
    "kv-f16": ["--kv-type", "f16"],
    "kv-f16-host-snapshots": ["--kv-type", "f16", "--prefix-cache-memory", "host"],
    "kv-f16-no-prefix-cache-no-spec": ["--kv-type", "f16", "--prefix-cache-slots", "0", "--spec-draft", "0"],
    "kv-f16-reserve0": ["--kv-type", "f16", "--vram-reserve-mib", "0"],
    "kv-f16-no-prefix-cache-no-spec-reserve0": ["--kv-type", "f16", "--prefix-cache-slots", "0", "--spec-draft", "0", "--vram-reserve-mib", "0"],
}


def parse_log(text):
    """The resolved context and the VRAM line of a zerv start log."""
    context = re.search(r"context max = (\d+) tokens", text)
    vram = re.search(r"VRAM: needed (\d+) MiB of (\d+) MiB free", text)
    return dict(context=int(context.group(1)) if context else None,
                needed_mib=int(vram.group(1)) if vram else None, free_mib=int(vram.group(2)) if vram else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT/"zig-out/bin/zerv")
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--port", type=int, default=18093)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary_sha = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    results = []
    for name in a.configs.split(","):
        cmd = [str(a.binary), "--model", str(a.model), "--port", str(a.port), "--context", "max"] + CONFIGS[name]
        log_path = out/f"{name}.log"
        idle = run_serving.vram_used()
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            try:
                run_serving.wait_ready(a.port, proc)
                loaded = run_serving.vram_used()
                r = run_serving.stream_request(a.port, dict(model="qwen3.8-27b", messages=[{"role": "user", "content": "Say OK."}],
                                                            stream=True, max_tokens=8, temperature=0,
                                                            chat_template_kwargs={"enable_thinking": False}), 120, 300)
                ok = bool(r["content"] or r["reasoning"])
            finally:
                proc.send_signal(signal.SIGINT)
                try: proc.wait(timeout=60)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        entry = dict(config=name, cmd=cmd, **parse_log(log_path.read_text()), vram_idle_mib=(idle or 0) >> 20,
                     vram_loaded_mib=(loaded or 0) >> 20, served=ok, reply=r["content"])
        results.append(entry)
        print(json.dumps(entry), flush=True)
        time.sleep(3)
    manifest = dict(finished_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv, binary_sha256=binary_sha, results=results)
    (out/"results.json").write_text(json.dumps(manifest, indent=1)+"\n")


if __name__ == "__main__":
    main()
