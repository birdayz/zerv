#!/usr/bin/env python3
"""Independent oracle for tool-calling prompts: the official Qwen3.8 template rendered
by the pinned Jinja2 with Hugging Face transformers' chat-template environment
(`tojson` = json.dumps(ensure_ascii=False), loopcontrols; transformers commit
935f7ab4f432dfeccfe60b190c546463f3b1e895, src/transformers/utils/chat_template_utils.py).

Request → template mapping (docs/specs/tool-calling.md), mirroring llama-server:
- tools are normalized to {"type": "function", "function": {"name", "description"
  (default ""), "parameters" (default {})}};
- assistant tool_call arguments given as a JSON string are parsed (object key order
  kept); "" means no arguments;
- null content renders as ""; text content parts are concatenated.

Usage: render_tools.py --output tests/fixtures/chat-tools.json (refuses overwrite), or
render_tools.py --compare RAW.json to diff the oracle against llama-server's recorded
/apply-template prompts (tools/check_tool_parity.py raw.json)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/"third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json"
WORKLOAD = ROOT/"bench/workloads/tool-parity-v1.json"


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])

    def fail(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = fail
    return env


def text(content):
    if content is None: return ""
    if isinstance(content, str): return content
    return "".join(part["text"] for part in content)


def template_inputs(case):
    tools = [dict(type="function", function=dict(name=t["function"]["name"], description=t["function"].get("description", ""),
                                                 parameters=t["function"].get("parameters") or {})) for t in case.get("tools") or []]
    messages = []
    for m in case["messages"]:
        out = dict(role=m["role"], content=text(m.get("content")))
        if m.get("reasoning_content") is not None: out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str): args = json.loads(args) if args != "" else ""
                calls.append(dict(function=dict(name=tc["function"]["name"], arguments=args)))
            out["tool_calls"] = calls
        messages.append(out)
    options = dict(case.get("options") or {})
    kwargs = dict(options.pop("chat_template_kwargs", {}) or {})
    effort = options.get("reasoning_effort")
    if effort is not None: kwargs.setdefault("reasoning_effort", "xhigh" if effort == "high" else effort)
    return messages, tools, kwargs


def render(template, case):
    messages, tools, kwargs = template_inputs(case)
    try:
        return dict(output=template.render(messages=messages, tools=tools or None, add_generation_prompt=True, **kwargs))
    except (ValueError, TypeError) as error:
        return dict(error=str(error))


def corpus():
    workload = json.loads(WORKLOAD.read_text())
    cases = [dict(name=c["name"], messages=c["messages"], tools=c["tools"], options=c["options"]) for c in workload["render"]]
    weather = workload["generation"][2]["tools"]
    cases += [
        dict(name="x-empty-tools", messages=[dict(role="user", content="hi")], tools=[], options={}),
        dict(name="x-tool-first", messages=[dict(role="tool", content="t"), dict(role="user", content="q")], tools=weather, options={}),
        dict(name="x-tool-last", messages=[dict(role="user", content="q"), dict(role="assistant", content=None, tool_calls=[
            dict(id="1", type="function", function=dict(name="get_weather", arguments="{\"city\": \"Paris\"}"))]), dict(role="tool", content=[dict(type="text", text=" a"), dict(type="text", text="b ")])],
             tools=weather, options={}),
        dict(name="x-args-empty-string", messages=[dict(role="user", content="q"), dict(role="assistant", content="", tool_calls=[
            dict(id="1", type="function", function=dict(name="get_weather", arguments=""))]), dict(role="tool", content="r")], tools=weather, options={}),
        dict(name="x-args-values", messages=[dict(role="user", content="q"), dict(role="assistant", content="x", tool_calls=[
            dict(id="1", type="function", function=dict(name="f", arguments=json.dumps({
                "s": "a\"b\\c\n\u00e9\u2028\u007f\u0001", "i": -12, "big": 123456789012345678901234567890, "f": [0.1, 1.5, 100.0, 1e16, 1e-05, 0.0001, -0.0, 3.14159265358979],
                "b": [True, False, None], "o": {"k": {"z": []}, "e": {}}, "u": "\U0001f600"})))])], tools=weather, options={}),
        dict(name="x-tool-response-user", messages=[dict(role="user", content="q"), dict(role="assistant", content="", tool_calls=[
            dict(id="1", type="function", function=dict(name="get_weather", arguments="{}"))]), dict(role="user", content="<tool_response>x</tool_response>")],
             tools=weather, options={}),
        dict(name="x-args-not-object", messages=[dict(role="user", content="q"), dict(role="assistant", content="", tool_calls=[
            dict(id="1", type="function", function=dict(name="f", arguments="[1, 2]"))])], tools=weather, options={}),
    ]
    return cases


def numbers():
    """JSON number literals and what json.dumps(json.loads(literal)) prints."""
    import random
    import struct
    rng = random.Random(38)
    literals = ["0", "-0", "7", "-12", "123456789012345678901234567890", "-9223372036854775809", "0.0", "-0.0", "1.0", "1E5", "1e-5",
                "1e16", "1e15", "9999999999999998.0", "0.0001", "0.00001", "1e22", "1e-7", "2.5", "0.1", "100.0", "1.5e300", "5e-324",
                "1.7976931348623157e308", "0.30000000000000004", "123456789.123", "1e400", "-1e400", "3.14159265358979", "1E+2"]
    for _ in range(400):
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if x != x or x in (float("inf"), float("-inf")): continue
        literals.append(rng.choice([repr(x), "%.17g" % x, "%.3e" % x]).replace("inf", "1e999"))
    for _ in range(200):
        literals.append(repr(rng.uniform(-1e6, 1e6)))
        literals.append(repr(rng.randint(-10**6, 10**6) / 1000))
    return [dict(literal=s, python=json.dumps(json.loads(s))) for s in literals]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--output", type=Path)
    g.add_argument("--compare", type=Path)
    a = p.parse_args()
    source = json.loads(CONFIG.read_text())["chat_template"]
    template = environment().from_string(source)
    if a.compare:
        raw = json.loads(a.compare.read_text())
        llama = next(v for k, v in raw.items() if k.startswith("llama"))["render"]
        cases = {c["name"]: c for c in corpus()}
        for name, prompt in llama.items():
            ours = render(template, cases[name])
            if isinstance(prompt, dict): print(f"{name:32s} llama error {prompt}"); continue
            if ours.get("output") == prompt: print(f"{name:32s} equal"); continue
            o = ours.get("output") or ""
            i = next((k for k in range(min(len(o), len(prompt))) if o[k] != prompt[k]), min(len(o), len(prompt)))
            print(f"{name:32s} DIFF at {i}: oracle {o[i-40:i+60]!r}\n{'':33s}llama  {prompt[i-40:i+60]!r}")
        return
    cases = corpus()
    for case in cases: case["expected"] = render(template, case)
    result = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=sys.version,
                  jinja2=jinja2.__version__, config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                  workload_sha256=hashlib.sha256(WORKLOAD.read_bytes()).hexdigest(), cases=cases, numbers=numbers())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("rendered", len(cases), "cases;", sum("error" in c["expected"] for c in cases), "errors")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
