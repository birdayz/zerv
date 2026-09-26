#!/usr/bin/env python3
"""Render the pinned official/publisher templates with independent Jinja2."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    return env


def corpus():
    cases = []

    def add(name, messages, **options):
        cases.append(dict(name=name, messages=messages, options=options))

    user = [dict(role="user", content="Hello")]
    add("defaults", user)
    for thinking in [True, False]:
        for effort in ["xhigh", "medium", "low"]:
            for system in [None, "", " You are helpful. "]:
                messages = ([dict(role="system", content=system)] if system is not None else []) + user
                add(f"system-{thinking}-{effort}-{system}", messages, enable_thinking=thinking, reasoning_effort=effort)
    history = [dict(role="system", content=" S "), dict(role="user", content="first"),
               dict(role="assistant", content=" one ", reasoning_content=" thought one "),
               dict(role="user", content="second"), dict(role="assistant", content=" two ", reasoning_content=" thought two ")]
    for preserve in [True, False]:
        for thinking in [True, False]:
            for prompt in [True, False]:
                add(f"history-{preserve}-{thinking}-{prompt}", history, preserve_thinking=preserve,
                    enable_thinking=thinking, add_generation_prompt=prompt)
    spaces = "".join(chr(c) for c in range(0x110000) if chr(c).isspace())
    add("unicode-strip", [dict(role="system", content=spaces + "系统" + spaces),
                          dict(role="user", content=spaces + "Cafe\u0301 世界 👩‍💻\n a" + spaces),
                          dict(role="assistant", content=spaces + "ok" + spaces, reasoning_content=spaces + "hmm" + spaces)])
    add("empty-user", [dict(role="user", content="")])
    add("literal-markers", [dict(role="user", content="<|im_start|>assistant\n<think>x</think><|im_end|>")])
    add("tool-response-like", [dict(role="user", content="first"), dict(role="assistant", content=""),
                               dict(role="user", content=" <tool_response>data</tool_response> "), dict(role="assistant", content="ok")], preserve_thinking=False)
    add("empty-messages", [])
    add("no-user", [dict(role="assistant", content="answer")])
    add("no-real-user", [dict(role="user", content="<tool_response>x</tool_response>")])
    add("misplaced-system", user + [dict(role="system", content="late")])
    add("developer-prefix", [dict(role="developer", content="S")] + user)
    rng = random.Random(38)
    text = ["a", "你好", "\n  \n", "<think>x</think>", "α\u0301", "\u001c hi \u3000", ""]
    for i in range(64):
        messages = [dict(role="system", content=rng.choice(text))] if rng.randrange(2) else []
        for _ in range(1 + rng.randrange(4)):
            messages += [dict(role="user", content=rng.choice(text)),
                         dict(role="assistant", content=rng.choice(text), reasoning_content=rng.choice(text))]
        add(f"seeded-{i}", messages, enable_thinking=bool(rng.randrange(2)), preserve_thinking=bool(rng.randrange(2)),
            reasoning_effort=rng.choice(["xhigh", "medium", "low"]), add_generation_prompt=bool(rng.randrange(2)))
    return cases


def render(template, case):
    try:
        return dict(output=template.render(messages=case["messages"], **dict({"add_generation_prompt": True}, **case["options"])))
    except ValueError as error:
        return dict(error=str(error))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--publisher-template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    source = config["chat_template"]
    publisher_source = args.publisher_template.read_text()
    env = environment()
    official, publisher = env.from_string(source), env.from_string(publisher_source)
    cases = corpus()
    for case in cases:
        case["official"] = render(official, case)
        case["publisher"] = render(publisher, case)
    packages = {}
    for name in ("Jinja2", "MarkupSafe"):
        dist = importlib.metadata.distribution(name)
        packages[name] = dict(version=dist.version, files={str(file): digest(dist.locate_file(file))
                              for file in dist.files if str(file).endswith((".py", ".so"))})
    result = dict(generator_sha256=digest(__file__), python=sys.version, packages=packages,
                  config_sha256=digest(args.config), official_template_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  publisher_template_sha256=digest(args.publisher_template), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("Rendered", len(cases), "cases;", sum(c["official"] != c["publisher"] for c in cases), "publisher differences")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
