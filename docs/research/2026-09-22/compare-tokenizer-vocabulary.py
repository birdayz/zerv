#!/usr/bin/env python3
"""Research-only full official/GGUF tokenizer inventory comparison via external ggml."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/reference"))
from gguf_oracle import Oracle, InitParams

model, source, output = map(Path, sys.argv[1:4])
if output.exists():
    raise ValueError("output exists")
config = json.loads(source.read_text())
oracle = Oracle("/usr/lib/libggml-base.so.0.24.0")
ctx = oracle.load(bytes(model), InitParams(True, None))
if not ctx:
    raise ValueError("GGUF rejected")
try:
    keys = {oracle.get("key", ctx, i, result=C.c_char_p).decode(): i for i in range(oracle.get("n_kv", ctx))}
    string = oracle.bind("gguf_get_arr_str", C.c_char_p, [C.c_void_p, C.c_int64, C.c_size_t])

    def strings(key):
        count = oracle.get("arr_n", ctx, keys[key], result=C.c_size_t)
        return [string(ctx, keys[key], i).decode() for i in range(count)]

    vocab, merges = strings("tokenizer.ggml.tokens"), strings("tokenizer.ggml.merges")
    ptr = C.cast(oracle.get("arr_data", ctx, keys["tokenizer.ggml.token_type"], result=C.c_void_p), C.POINTER(C.c_int32))
    types = list(ptr[:len(vocab)])
    official = config["model"]["vocab"]
    assert len(set(vocab)) == len(vocab)
    assert all(vocab[i] == s and types[i] == 1 for s, i in official.items())
    # The official file uses the legacy string representation, not pair arrays.
    official_merges = config["model"]["merges"]
    assert all(isinstance(m, str) and m.count(" ") == 1 for m in official_merges)
    assert merges == official_merges
    pairs = [m.split(" ") for m in official_merges]
    initial = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {b: chr(b) for b in initial}
    for b in range(256):
        if b not in alphabet:
            alphabet[b] = chr(256 + len(alphabet) - len(initial))
    reverse = {v: k for k, v in alphabet.items()}
    assert len(reverse) == 256 and all(s in official for s in reverse)
    assert all(c in reverse for s in official for c in s)
    assert all(a in official and b in official and a+b in official for a, b in pairs)
    assert len({tuple(pair) for pair in pairs}) == len(merges)
    produced = {a+b for a, b in pairs} | set(reverse)
    unreachable = [dict(id=i, token=s) for s, i in official.items() if s not in produced]
    added = []
    for token in config["added_tokens"]:
        assert vocab[token["id"]] == token["content"] and types[token["id"]] in (3, 4)
        assert not any(token[k] for k in ("single_word", "lstrip", "rstrip", "normalized"))
        added.append(dict(**token, gguf_type=types[token["id"]]))
    padding = [i for i, kind in enumerate(types) if kind == 5]
    assert padding == list(range(248077, 248320))
    added_ids = {t["id"] for t in added}
    assert all(kind == 1 or i in added_ids or kind == 5 for i, kind in enumerate(types))

    def framed_hash(values):
        return hashlib.sha256(b"".join(struct.pack("<I", len(s.encode())) + s.encode() for s in values)).hexdigest()

    result = dict(official_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), oracle=oracle.identity,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_path=str(model), model_bytes=model.stat().st_size,
                  vocabulary_count=len(vocab), regular_count=len(official), merges_count=len(merges),
                  added_count=len(added), padding_count=len(padding), vocab_sha256_u32_framed=framed_hash(vocab),
                  merges_sha256_u32_framed=framed_hash(merges), all_regular_ids_equal=True, all_merges_equal=True,
                  all_regular_glyphs_valid=True, all_byte_symbols_present=True, unique_merge_pairs=True,
                  type_counts=dict(Counter(types)), max_regular_piece_bytes=max(map(len, official)),
                  total_regular_piece_bytes=sum(map(len, official)), merge_unreachable=unreachable, added_tokens=added)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("merge_unreachable", "added_tokens")}, indent=2))
    print("unreachable", len(unreachable), unreachable[:12])
finally:
    oracle.free(ctx)
