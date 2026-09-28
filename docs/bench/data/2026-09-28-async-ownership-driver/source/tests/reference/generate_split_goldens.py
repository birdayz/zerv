#!/usr/bin/env python3
"""UCD16 classifier data and independent HF Qwen split goldens, never runtime code."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct
import sys
import tokenizers
from tokenizers import Regex, pre_tokenizers


def sha(data):
    return hashlib.sha256(data).hexdigest()


def properties(ucd):
    flags = bytearray(0x110000)
    first = None
    for line in (ucd / "UnicodeData.txt").read_text().splitlines():
        f = line.split(";")
        cp, value = int(f[0], 16), {"L": 1, "M": 2, "N": 4}.get(f[2][0], 0)
        if f[1].endswith(", First>"):
            if first is not None:
                raise ValueError("nested UCD range")
            first = cp, value
        elif f[1].endswith(", Last>"):
            if first is None or first[1] != value:
                raise ValueError("invalid UCD range")
            flags[first[0]:cp+1] = bytes([value]) * (cp+1-first[0])
            first = None
        else:
            flags[cp] = value
    if first is not None:
        raise ValueError("unterminated UCD range")
    for line in (ucd / "PropList.txt").read_text().splitlines():
        f = line.split("#")[0].strip().split(";")
        if len(f) == 2 and f[1].strip() == "White_Space":
            ends = [int(x, 16) for x in f[0].strip().split("..")]
            for cp in range(ends[0], ends[-1]+1):
                flags[cp] |= 8
    return flags


def byte_ends(splitter, text):
    pieces = [piece for piece, _ in splitter.pre_tokenize_str(text)]
    if "".join(pieces) != text or any(not piece for piece in pieces):
        raise ValueError("non-covering/empty reference split")
    total, ends = 0, []
    for piece in pieces:
        total += len(piece.encode())
        ends.append(total)
    return ends


def generate(a):
    if a.data.exists() or a.fixtures.exists():
        raise ValueError("output already exists")
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("requires tokenizers==0.22.2")
    config = json.loads(a.tokenizer.read_text())
    split = config["pre_tokenizer"]["pretokenizers"][0]
    if split["type"] != "Split" or split["behavior"] != "Isolated" or split["invert"]:
        raise ValueError("unsupported official pretokenizer")
    pattern = split["pattern"]["Regex"]
    splitter = pre_tokenizers.Split(Regex(pattern), "isolated")
    flags = properties(a.ucd)
    classifiers = [pre_tokenizers.Split(Regex(p), "isolated") for p in (r"\p{L}", r"\p{M}", r"\p{N}", r"\s")]
    digest = hashlib.sha256()
    counts = [0] * 4
    for cp in range(0x110000):
        if 0xd800 <= cp <= 0xdfff:
            continue
        observed = 0
        for i, classifier in enumerate(classifiers):
            if len(classifier.pre_tokenize_str("#" + chr(cp) + "#")) == 3:
                observed |= 1 << i
                counts[i] += 1
        if observed != flags[cp]:
            raise ValueError(f"UCD/HF mismatch at {cp:x}")
        digest.update(bytes([observed]))
    rows = [(0, flags[0])]
    for cp in range(1, len(flags)):
        if flags[cp] != flags[cp-1]:
            rows.append((cp, flags[cp]))
    rows.append((0x110000, 0))
    data = struct.pack("<4sI", b"U16C", len(rows)) + b"".join(struct.pack("<II", *row) for row in rows)
    texts = ["".join(chars) for length in range(6) for chars in itertools.product("a1 \t\r\n!\u0301", repeat=length)]
    folded = set()
    for line in (a.ucd / "CaseFolding.txt").read_text().splitlines():
        f = line.split("#")[0].strip().split(";")
        if len(f) >= 3:
            folded.add(int(f[0], 16))
    for cp in sorted(folded):
        c = chr(cp)
        texts += ["'" + c + tail for tail in ("X", "eX", "lX")]
        texts += [prefix + c + "X" for prefix in ("'r", "'v", "'l")]
    texts += ["'sX 'tX 'reX 'veX 'mX 'llX 'dX", "'Sx 'TX 'REX 'VeX 'mX 'lLX 'DX", "'ſt", "\u001cX\u0085Y"]
    rng = random.Random(20260922)
    pool = list("aA1!? 'srevl\t\r\n\u0301\u0323é世界①Ⅷ👩\u200d💻")
    pool += [chr(cp) for start, _ in rows for cp in range(max(0, start-1), min(0x110000, start+2))
             if not 0xd800 <= cp <= 0xdfff]
    for _ in range(1024):
        texts.append("".join(rng.choice(pool) for _ in range(rng.randrange(1, 80))))
    texts += [c["official"]["output"] for c in json.loads(a.chat.read_text())["cases"] if "output" in c["official"]]
    workloads = {"ascii": "Hello, world! we're testing 123.\r\n" * 512,
                 "multilingual": "Cafe\u0301 世界 किताब 'ſt ①Ⅷ👩‍💻\n" * 256,
                 "whitespace": "\t " * 8192 + "x"}
    texts += list(workloads.values()) + ["\u0301\u0323" * 8192, " \t\r\n" * 8192 + " X"]
    golden = bytearray()
    for text in texts:
        encoded, ends = text.encode(), byte_ends(splitter, text)
        golden.extend(struct.pack("<II", len(encoded), len(ends)) + encoded)
        golden.extend(struct.pack("<" + "I" * len(ends), *ends))
    extension = next(Path(tokenizers.__file__).parent.glob("*.so"))
    manifest = dict(unicode_version="16.0.0", pattern=pattern, tokenizers_version=tokenizers.__version__, python=sys.version,
                    oracle_binary_sha256=sha(extension.read_bytes()), generator_sha256=sha(Path(__file__).read_bytes()),
                    tokenizer_sha256=sha(a.tokenizer.read_bytes()), chat_sha256=sha(a.chat.read_bytes()),
                    data_sha256=sha(data), golden_sha256=sha(golden), records=len(texts), rows=len(rows),
                    property_sha256=digest.hexdigest(), property_counts=counts, scalar_count=0x110000-2048,
                    seed=20260922, sources={name: sha((a.ucd/name).read_bytes())
                                           for name in ("UnicodeData.txt", "PropList.txt", "CaseFolding.txt")},
                    workloads=[dict(name=name, text=text, ends=byte_ends(splitter, text)) for name, text in workloads.items()])
    a.fixtures.mkdir(parents=True)
    a.data.parent.mkdir(parents=True, exist_ok=True)
    a.data.write_bytes(data)
    (a.fixtures / "cases.bin").write_bytes(golden)
    (a.fixtures / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Wrote", len(texts), "HF cases;", len(rows), "property ranges;", len(data), "table bytes;", digest.hexdigest())


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ucd", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--chat", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    generate(p.parse_args())
