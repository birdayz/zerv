#!/usr/bin/env python3
"""Research-only exhaustive UCD16/HF regex property and contraction-fold probe."""
import hashlib
import json
from pathlib import Path
import sys
import tokenizers
from tokenizers import Regex, pre_tokenizers

ucd = Path(sys.argv[1])
output = Path(sys.argv[2])
if output.exists():
    raise ValueError("output exists")
if tokenizers.__version__ != "0.22.2":
    raise ValueError("unexpected oracle version")
flags = bytearray(0x110000)
first = None
for line in (ucd / "UnicodeData.txt").read_text().splitlines():
    fields = line.split(";")
    cp = int(fields[0], 16)
    value = {"L": 1, "M": 2, "N": 4}.get(fields[2][0], 0)
    if fields[1].endswith(", First>"):
        first = cp, value
    elif fields[1].endswith(", Last>"):
        if first is None or first[1] != value:
            raise ValueError("invalid UCD range")
        flags[first[0]:cp+1] = bytes([value]) * (cp+1-first[0])
        first = None
    else:
        flags[cp] = value
if first is not None:
    raise ValueError("unterminated UCD range")
for line in (ucd / "PropList.txt").read_text().splitlines():
    fields = line.split("#")[0].strip().split(";")
    if len(fields) == 2 and fields[1].strip() == "White_Space":
        ends = [int(x, 16) for x in fields[0].strip().split("..")]
        for cp in range(ends[0], ends[-1]+1):
            flags[cp] |= 8
folds = {}
for line in (ucd / "CaseFolding.txt").read_text().splitlines():
    fields = [f.strip() for f in line.split("#")[0].split(";")]
    if len(fields) >= 3 and fields[1] in ("C", "S"):
        folds[int(fields[0], 16)] = int(fields[2], 16)
patterns = [r"\p{L}", r"\p{M}", r"\p{N}", r"\s", r"(?i:[strevmld])"]
splitters = [pre_tokenizers.Split(Regex(pattern), "isolated") for pattern in patterns]
counts = [0] * 5
mismatches = []
fold_members = []
hash_ref = hashlib.sha256()
for cp in range(0x110000):
    if 0xd800 <= cp <= 0xdfff:
        continue
    text = "#" + chr(cp) + "#"
    observed = [len(split.pre_tokenize_str(text)) == 3 for split in splitters]
    expected = [bool(flags[cp] & (1 << i)) for i in range(4)] + [chr(folds.get(cp, cp)) in "strevmld"]
    counts = [n + value for n, value in zip(counts, observed)]
    if observed != expected:
        mismatches.append(dict(cp=hex(cp), expected=expected, observed=observed))
    bits = sum(1 << i for i, match in enumerate(observed[:4]) if match)
    hash_ref.update(bytes([bits]))
    if observed[-1]:
        fold_members.append(hex(cp))
result = dict(tokenizers_version=tokenizers.__version__, python=sys.version, scalars=0x110000-2048,
              patterns=patterns, counts=counts, mismatches=mismatches, fold_members=fold_members,
              property_sha256=hash_ref.hexdigest(),
              sources={name: hashlib.sha256((ucd/name).read_bytes()).hexdigest()
                       for name in ("UnicodeData.txt", "PropList.txt", "CaseFolding.txt")},
              probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if mismatches:
    raise ValueError("UCD/HF property mismatch")
