#!/usr/bin/env python3
"""Session/serving gate: the native server's greedy chat output for every oracle case
must equal the libllama greedy continuation decoded independently (llama raw pieces),
with identical prompt/completion token counts, in both non-streaming and SSE modes.
The expected reasoning/content split is an independent Python rendering of
llama-server's output pipeline (docs/research/output-parsing.md)."""
import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).absolute().parents[1]  # not resolved: Bazel tests import it from their runfiles
EOS = {248046, 248044}
THINK_END = 248069
# GGUF token type 3 (control): rendered as nothing with special=false.
CONTROL = set(range(248044, 248058)) | set(range(248060, 248066)) | set(range(248070, 248077))
WS = " \t\n\v\f\r"  # C isspace
DELIMITERS = ("</think>", "<tool_call>")


def render(table, tokens):
    """llama_token_to_piece(special=false) per token, then UTF-8 with replacement of
    invalid sequences and an incomplete tail dropped (not flushed)."""
    raw = b"".join(b"" if t in CONTROL else table[t] for t in tokens)
    return codecs.getincrementaldecoder("utf-8")("replace").decode(raw, final=False)


def split_reference(text, thinking):
    """(reasoning or None, content) as the Qwen3-Coder PEG parser maps `text`."""
    if not thinking: return None, text.lstrip(WS)
    text = text.lstrip(WS)
    found = [(text.find(d), d) for d in DELIMITERS if d in text]
    if found:
        at, d = min(found)
        rest = text[at+len(d):] if d == "</think>" else text[at:]
        return text[:at] or None, rest.lstrip(WS)
    for k in range(min(len(text), max(map(len, DELIMITERS))-1), 0, -1):
        if any(len(d) > k and d.startswith(text[-k:]) for d in DELIMITERS):
            text = text[:-k]
            break
    return text or None, ""


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def pieces_file(model, directory):
    """llama.cpp's raw token pieces of the model (the source-built adapter,
    //tests:oracle_tokenizer_pieces), written into `directory`; checked against the tokenizer
    fixture's hash by the caller."""
    import zerv_build
    adapter, _ = zerv_build.oracle("oracle_tokenizer_pieces")
    path = directory/"pieces.bin"
    subprocess.run([str(adapter), str(model), str(path)], check=True, stdout=subprocess.DEVNULL)
    return path


def pieces(path):
    raw = path.read_bytes()
    count, = struct.unpack_from("<I", raw); at = 4; out = []
    for _ in range(count):
        n, = struct.unpack_from("<I", raw, at); out.append(raw[at+4:at+4+n]); at += 4+n
    return out


def request(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    if r.status != 200: raise SystemExit(f"HTTP {r.status}: {data[:400]}")
    if not body.get("stream"): return json.loads(data)
    reasoning, content, finish, usage = "", "", None, None
    for event in data.decode().split("\n\n"):
        if not event.startswith("data: ") or event == "data: [DONE]": continue
        j = json.loads(event[6:])
        usage = j.get("usage") or usage
        for ch in j["choices"]:
            reasoning += ch["delta"].get("reasoning_content") or ""; content += ch["delta"].get("content") or ""
            finish = ch.get("finish_reason") or finish
    return dict(choices=[dict(message=dict(reasoning_content=reasoning, content=content), finish_reason=finish)], usage=usage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT/"models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=18095)
    a = p.parse_args()
    if a.output.exists(): p.error("fresh output required")
    pieces_path = pieces_file(a.model, Path(tempfile.mkdtemp()))
    table = pieces(pieces_path)
    if sha(pieces_path) != json.loads((ROOT/"tests/fixtures/tokenizer/manifest.json").read_text())["raw_pieces_sha256"]: raise SystemExit("pieces changed")
    if table[THINK_END] != b"</think>": raise SystemExit("unexpected </think> id")
    fixture = json.loads((ROOT/"tests/fixtures/model/qwen38-oracle.json").read_text())
    import zerv_build
    binary = zerv_build.binary("zerv")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log = open(str(a.output)+".server.log", "w")
    proc = subprocess.Popen([str(binary), "--model", str(a.model), "--port", str(a.port), "--context", "4096"], stdout=log, stderr=subprocess.STDOUT)
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), server_sha256=sha(binary), fixture_sha256=sha(ROOT/"tests/fixtures/model/qwen38-oracle.json"),
                  pieces_sha256=sha(pieces_path), cases=[], passed=True)
    try:
        for _ in range(600):
            try:
                c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=2); c.request("GET", "/ready"); ok = c.getresponse().status == 200; c.close()
                if ok: break
            except OSError: time.sleep(0.5)
        else: raise SystemExit("server not ready")
        for case in fixture["cases"]:
            continuation = case["tokens"][case["n_prompt"]:]
            generated = []
            for tok in continuation:
                generated.append(tok)
                if tok in EOS: break
            finish = "stop" if generated[-1] in EOS else "length"
            thinking = case["prompt"].endswith("<think>\n")
            body_tokens = [t for t in generated if t not in EOS]
            reasoning, content = split_reference(render(table, body_tokens), thinking)
            expected = dict(reasoning=reasoning, content=content, finish=finish, prompt_tokens=case["n_prompt"], completion_tokens=len(generated))
            for stream in (False, True):
                body = dict(model="qwen3.8-27b", messages=case["messages"], temperature=0, max_tokens=case["generate"], stream=stream, **case["options"])
                if stream: body["stream_options"] = {"include_usage": True}
                if "enable_thinking" in case["options"]:
                    body.pop("enable_thinking"); body["chat_template_kwargs"] = {"enable_thinking": case["options"]["enable_thinking"]}
                r = request(a.port, body)
                msg = r["choices"][0]["message"]
                # reasoning_content is omitted when empty (as llama-server does); SSE accumulates "".
                got = dict(reasoning=msg.get("reasoning_content") or None, content=msg["content"], finish=r["choices"][0]["finish_reason"],
                           prompt_tokens=r["usage"]["prompt_tokens"], completion_tokens=r["usage"]["completion_tokens"])
                ok = got == expected
                report["passed"] &= ok
                report["cases"].append(dict(name=case["name"], stream=stream, ok=ok, expected=expected, got=got))
                print(case["name"], "stream" if stream else "json", "OK" if ok else "MISMATCH", flush=True)
                if not ok: print(" expected", expected, "\n got", got)
    finally:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1)+"\n")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
