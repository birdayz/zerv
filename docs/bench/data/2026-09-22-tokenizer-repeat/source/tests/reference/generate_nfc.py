#!/usr/bin/env python3
"""Generate normative Unicode-9 data and independent HF NFC goldens; never runtime code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tokenizers
from tokenizers.normalizers import NFC


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generate(ucd, data_path, fixtures):
    if data_path.exists() or fixtures.exists():
        raise ValueError("output already exists")
    decompositions, ccc = {}, {}
    for line in (ucd / "UnicodeData.txt").read_text().splitlines():
        fields = line.split(";")
        cp, cls = int(fields[0], 16), int(fields[3])
        if cls:
            ccc[cp] = cls
        if fields[5] and not fields[5].startswith("<"):
            decompositions[cp] = [int(x, 16) for x in fields[5].split()]
    exclusions = set()
    for line in (ucd / "DerivedNormalizationProps.txt").read_text().splitlines():
        fields = line.split("#")[0].strip().split(";")
        if len(fields) > 1 and fields[1].strip() == "Full_Composition_Exclusion":
            ends = [int(x, 16) for x in fields[0].strip().split("..")]
            exclusions.update(range(ends[0], ends[-1] + 1))

    def expand(cp, ancestors=()):
        if cp in ancestors:
            raise ValueError("decomposition cycle")
        return [x for part in decompositions[cp] for x in expand(part, ancestors + (cp,))] if cp in decompositions else [cp]

    composition = {}
    for cp, parts in decompositions.items():
        if cp not in exclusions:
            if len(parts) != 2:
                raise ValueError("non-pair primary composite")
            key = parts[0] << 21 | parts[1]
            if key in composition:
                raise ValueError("duplicate composition")
            composition[key] = cp
    data = bytearray(struct.pack("<4sIII", b"NFC9", len(decompositions), len(ccc), len(composition)))
    for cp in sorted(decompositions):
        parts = expand(cp)
        if len(parts) > 4:
            raise ValueError("decomposition exceeds layout")
        data.extend(struct.pack("<6I", cp, len(parts), *(parts + [0] * (4 - len(parts)))))
    for cp, cls in sorted(ccc.items()):
        data.extend(struct.pack("<II", cp, cls))
    for key, cp in sorted(composition.items()):
        data.extend(struct.pack("<QI", key, cp))
    nfc = NFC()
    golden = bytearray()
    records = 0
    for line in (ucd / "NormalizationTest.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("@"):
            continue
        columns = ["".join(chr(int(x, 16)) for x in col.split()) for col in line.split(";")[:5]]
        for i, expected in [(0, columns[1]), (1, columns[1]), (2, columns[1]), (3, columns[3]), (4, columns[3])]:
            if nfc.normalize_str(columns[i]) != expected:
                raise ValueError("HF/Unicode-9 conformance mismatch")
            a, b = columns[i].encode(), expected.encode()
            golden.extend(struct.pack("<II", len(a), len(b)) + a + b)
            records += 1
    fingerprints = {}
    for mode in ("scalar", "mark_context"):
        digest = hashlib.sha256()
        for cp in range(0x110000):
            if 0xd800 <= cp <= 0xdfff:
                continue
            text = chr(cp) if mode == "scalar" else "[\u0301" + chr(cp) + "\u0323"
            out = nfc.normalize_str(text).encode()
            digest.update(struct.pack("<I", len(out)) + out)
        fingerprints[mode] = digest.hexdigest()
    # Performance corpus, including an adversarial long nonstarter run.
    workloads = {"ascii": "Hello, world! 0123456789\n" * 512,
                 "multilingual": "Cafe\u0301 α\u0301 각 किताब 世界 👩‍💻\n" * 256,
                 "marks": "[" + "\u0301\u0323\u0315\u0327" * 2048}
    manifest = dict(unicode_version="9.0.0", tokenizers_version=tokenizers.__version__, python=sys.version,
                    generator_sha256=sha(Path(__file__).read_bytes()), data_sha256=sha(data),
                    golden_sha256=sha(golden), records=records, fingerprints=fingerprints,
                    sources={p.name: sha(p.read_bytes()) for p in sorted(ucd.glob("*.txt"))},
                    workloads=[dict(name=name, text=text, output=nfc.normalize_str(text)) for name, text in workloads.items()])
    fixtures.mkdir(parents=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(data)
    (fixtures / "cases.bin").write_bytes(golden)
    (fixtures / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Wrote", records, "independent records; table bytes", len(data), "fingerprints", fingerprints)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ucd", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    a = p.parse_args()
    generate(a.ucd, a.data, a.fixtures)
