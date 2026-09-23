#!/usr/bin/env python3
"""Render one official-template prompt (stdin JSON {messages, options}) with the pinned Jinja2."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_chat_goldens import environment

ROOT = Path(__file__).resolve().parents[2]
chat = json.loads((ROOT/"tests/fixtures/chat-template.json").read_text())
for name, package in chat["packages"].items():
    dist = importlib.metadata.distribution(name)
    if dist.version != package["version"]: raise SystemExit("package version mismatch: "+name)
    for file, digest in package["files"].items():
        if hashlib.sha256(Path(dist.locate_file(file)).read_bytes()).hexdigest() != digest: raise SystemExit("package file mismatch: "+file)
config = ROOT/"third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json"
if hashlib.sha256(config.read_bytes()).hexdigest() != chat["config_sha256"]: raise SystemExit("template config mismatch")
case = json.load(sys.stdin)
template = environment().from_string(json.loads(config.read_text())["chat_template"])
sys.stdout.write(template.render(messages=case["messages"], add_generation_prompt=True, **case["options"]))
