#!/usr/bin/env python3
"""Independent HF short-piece/threshold fixtures and broader fixed benchmark inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import tokenizers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists; generate separately and review")
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("requires pinned tokenizers 0.22.2")
    hf = tokenizers.Tokenizer.from_file(str(a.tokenizer))
    rng = random.Random(0xB0E16)
    texts = [chr(i) + chr(j) for i in range(128) for j in range(128)]
    for n in range(1, 66):
        texts.extend(["a" * n, "ab" * n, " " + "x" * n, "界" * n, "é" * n, "e\u0301" * n])
        for _ in range(16):
            texts.append("".join(rng.choice("aabcdeXYZ_'界é\u0301012 \r\n<>|") for _ in range(n)))
    texts = list(dict.fromkeys(texts))

    def case(text):
        ids = hf.encode(text, add_special_tokens=False).ids
        raw = hf.decode(ids, skip_special_tokens=False).encode()
        return dict(text=text, ids=ids, raw_hex=raw.hex())

    # Long ordinary pieces exercise the heap path, not just short-piece tuning.
    workloads = [dict(case("a" * 4096), name="long_word"),
                 dict(case("漢字東京大学自然言語処理" * 64), name="unicode_long")]
    unique = []
    for i in range(96):
        word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8 + i % 25))
        unique.append(f"let {word} = values[{i}]; // item_{i}\n")
    workloads.append(dict(case("".join(unique)), name="unique_code"))
    for workload in workloads:
        if hf.normalizer.normalize_str(workload["text"]) != workload["text"]:
            raise ValueError("benchmark input changes under NFC")
    result = dict(schema_version=1, generator_sha256=sha(Path(__file__)), tokenizer_sha256=sha(a.tokenizer),
                  tokenizers_version=tokenizers.__version__, seed=0xB0E16,
                  cases=[case(text) for text in texts], workloads=workloads)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, ensure_ascii=True, separators=(",", ":"))
        f.write("\n")
    print(json.dumps(dict(cases=len(texts), workloads=[dict(name=w["name"], bytes=len(w["text"].encode()), tokens=len(w["ids"])) for w in workloads], sha256=sha(a.output)), indent=2))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
