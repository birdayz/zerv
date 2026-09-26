#!/usr/bin/env python3
"""Official HF IDs and independent llama raw-byte pieces; no native runtime use."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import tokenizers
from tokenizers import Tokenizer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generate(a):
    if a.output.exists():
        raise ValueError("output exists")
    if tokenizers.__version__ != "0.22.2":
        raise ValueError("requires tokenizers 0.22.2")
    config = json.loads(a.tokenizer.read_text())
    hf = Tokenizer.from_file(str(a.tokenizer))
    vocab = config["model"]["vocab"]
    if len(vocab) != 248044 or len(config["added_tokens"]) != 33:
        raise ValueError("unexpected vocabulary")
    alphabet = {b: chr(b) for b in list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))}
    next_cp = 256
    for b in range(256):
        if b not in alphabet:
            alphabet[b] = chr(next_cp)
            next_cp += 1
    reverse = {c: b for b, c in alphabet.items()}
    if a.pieces is None:
        # The source-built llama.cpp adapter (//tests:oracle_tokenizer_pieces) over the model.
        sys.path.insert(0, str(Path(__file__).absolute().parents[2] / "tools"))
        import subprocess
        import tempfile
        import zerv_build
        a.adapter, adapter_build = zerv_build.oracle("oracle_tokenizer_pieces")
        a.pieces = Path(tempfile.mkdtemp()) / "pieces.bin"
        subprocess.run([str(a.adapter), str(a.model), str(a.pieces)], check=True, stdout=subprocess.DEVNULL)
    else:
        adapter_build = None
    raw = a.pieces.read_bytes()
    count = struct.unpack_from("<I", raw)[0]
    if count != 248320:
        raise ValueError("wrong oracle vocabulary count")
    pieces, at = [], 4
    for _ in range(count):
        n = struct.unpack_from("<I", raw, at)[0]
        at += 4
        pieces.append(raw[at:at+n])
        at += n
    if at != len(raw):
        raise ValueError("malformed oracle pieces")
    expected, kinds = [b""] * count, [3] * count
    for text, id_ in vocab.items():
        expected[id_] = bytes(reverse[c] for c in text)
        kinds[id_] = 1
    for entry in config["added_tokens"]:
        if any(entry[k] for k in ("normalized", "single_word", "lstrip", "rstrip")):
            raise ValueError("unsupported added-token flags")
        expected[entry["id"]] = entry["content"].encode()
        kinds[entry["id"]] = 2
    if pieces != expected:
        raise ValueError("raw llama/official byte-alphabet mismatch")
    for id_, piece in enumerate(pieces[:248077]):
        if hf.decode([id_], skip_special_tokens=False) != piece.decode(errors="replace"):
            raise ValueError(f"HF piece decode mismatch at {id_}")
    pairs = [line.split(" ") for line in config["model"]["merges"]]
    binary = bytearray(struct.pack("<4sII", b"ZBPE", count, len(pairs)))
    for kind, piece in zip(kinds, pieces):
        binary.extend(struct.pack("<IB3x", len(piece), kind) + piece)
    for left, right in pairs:
        binary.extend(struct.pack("<III", vocab[left], vocab[right], vocab[left+right]))
    text_cases = ["", "Hello, world!", "e\u0301", "é", "Cafe\u0301 α\u0301 각 किताब 世界 👩‍💻",
                  "[\u0301\u07fd\u0323", "[\u0301\U0001e08f\u0323", " a\r\n  b\t", "'ſt 'S I'VE we're",
                  "a" * 4096, "abacaba" * 1024, "\u0301\u0323" * 512, "[PAD248077]", "\0x\0"]
    text_cases += [chr(cp) for cp in range(256)]
    for entry in config["added_tokens"]:
        text = entry["content"]
        text_cases += [text, "e\u0301" + text + "\u0301x", " " + text + text + " ", text[:-1]]
    produced = {l+r for l, r in pairs} | set(reverse)
    text_cases += [piece.decode(errors="replace") for text, id_ in vocab.items()
                   if text not in produced for piece in [pieces[id_]]]
    live = set(reverse)
    forward = []
    for rank, (left, right) in enumerate(pairs):
        if left not in live or right not in live:
            forward.append(rank)
            text_cases.append(pieces[vocab[left+right]].decode(errors="replace"))
        live.add(left+right)
    chat = json.loads(a.chat.read_text())
    text_cases += [c["official"]["output"] for c in chat["cases"] if "output" in c["official"]]
    rng = random.Random(20260922)
    alphabet_text = list(" abcdefXYZ123!?'\t\r\n\u0301\u0323é世界👩‍💻①Ⅷ")
    for _ in range(512):
        text_cases.append("".join(rng.choice(alphabet_text) for _ in range(rng.randrange(1, 160))))
    workloads = {"ascii": "Hello, world! we're testing 123.\n" * 64,
                 "multilingual": "Café 世界 किताब 'ſt ①Ⅷ👩‍💻\n" * 32,
                 "chat": next(c["official"]["output"] for c in chat["cases"] if "output" in c["official"])}
    text_cases += list(workloads.values())

    def case(text):
        ids = hf.encode(text, add_special_tokens=False).ids
        decoded = b"".join(pieces[i] for i in ids)
        if decoded.decode(errors="replace") != hf.decode(ids, skip_special_tokens=False):
            raise ValueError("HF sequence decoding mismatch")
        return dict(text=text, ids=ids, raw_hex=decoded.hex())

    decode_ids = [[], [127], [102], [127, 102], [248077, 248319], list(range(248044, 248077))]
    decode_ids += [[rng.randrange(248320) for _ in range(7)] for _ in range(64)]
    extension = next(Path(tokenizers.__file__).parent.glob("*.so"))
    manifest = dict(tokenizers_version=tokenizers.__version__, python=sys.version, seed=20260922,
                    tokenizer_sha256=sha(a.tokenizer.read_bytes()), generator_sha256=sha(Path(__file__).read_bytes()),
                    model_data_sha256=sha(binary), raw_pieces_sha256=sha(raw), raw_piece_count=count,
                    oracle_adapter_sha256=sha(Path(__file__).with_name("tokenizer_pieces.c").read_bytes()),
                    oracle_binary_sha256=sha(a.adapter.read_bytes()), hf_binary_sha256=sha(extension.read_bytes()),
                    oracle_build=adapter_build,
                    forward_rank_cases=forward, cases=[case(text) for text in text_cases],
                    decode_cases=[dict(ids=ids, raw_hex=b"".join(pieces[i] for i in ids).hex()) for ids in decode_ids],
                    workloads=[dict(name=name, **case(text)) for name, text in workloads.items()])
    a.output.mkdir(parents=True)
    (a.output / "model.bin").write_bytes(binary)
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Validated", count, "raw pieces; wrote", len(text_cases), "encode cases and", len(decode_ids), "decode cases;", len(binary), "data bytes")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--chat", type=Path, required=True)
    p.add_argument("--model", type=Path, help="GGUF whose vocabulary the source-built adapter dumps (default path)")
    p.add_argument("--pieces", type=Path, help="raw pieces dumped earlier by --adapter (instead of --model)")
    p.add_argument("--adapter", type=Path, help="the adapter binary that dumped --pieces")
    p.add_argument("--output", type=Path, required=True)
    generate(p.parse_args())
