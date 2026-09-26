#!/usr/bin/env python3
"""Independent external/scalar Q5_K goldens. Not a native dependency."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

from gguf_oracle import Oracle, library as oracle_library
from generate_q4_1_goldens import EDGE, MODEL_SHA, sha

PARTNERS = [0, 0x3555, 0xfbff]


def pattern():
    scales = [0, 1, 15, 16, 31, 32, 62, 63]
    minima = scales[::-1]
    block = bytearray(176)
    struct.pack_into("<HH", block, 0, 0x3c00, 0x3800)
    for g in range(4):
        block[4+g] = scales[g] | ((scales[g+4] >> 4) << 6)
        block[8+g] = minima[g] | ((minima[g+4] >> 4) << 6)
        block[12+g] = (scales[g+4] & 15) | ((minima[g+4] & 15) << 4)
    for g in range(8):
        for lane in range(32):
            q = (lane + 5*g) % 32
            block[16+lane] |= (q >> 4) << g
            block[48+(g//2)*32+lane] |= (q & 15) << (4*(g % 2))
    return block


def scalar(packed):
    values = []
    for offset in range(0, len(packed), 176):
        block = packed[offset:offset+176]
        d, m = struct.unpack_from("<ee", block)
        s = block[4:16]
        scales = [b & 63 for b in s[:4]] + [(s[g+8] & 15) + (s[g] // 64)*16 for g in range(4)]
        minima = [b & 63 for b in s[4:8]] + [s[g+8] // 16 + (s[g+4] // 64)*16 for g in range(4)]
        for g in range(8):
            factor, minimum = d * scales[g], m * minima[g]
            for lane in range(32):
                q = block[48+(g//2)*32+lane] // (16 if g % 2 else 1) % 16
                q += (block[16+lane] // (2**g) % 2) * 16
                values.append(factor * q - minimum)
    return struct.pack("<" + "f" * len(values), *values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, help="default: the source-built ggml (@ggml//:ggml_base_so)")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    library, built = oracle_library(a.library)
    library, model = library.resolve(strict=True), a.model.resolve(strict=True)
    if sys.byteorder != "little" or C.sizeof(C.c_float) != 4 or sha(model) != MODEL_SHA:
        raise ValueError("host/model mismatch")
    oracle = Oracle(library)
    decode = oracle.bind("dequantize_row_q5_K", None, [C.c_void_p, C.POINTER(C.c_float), C.c_int64])

    def checked(packed):
        if len(packed) % 176:
            raise ValueError("incomplete blocks")
        n = len(packed)//176*256
        source, output = C.create_string_buffer(bytes(packed)), (C.c_float*n)()
        if C.addressof(source) % 2 or C.addressof(output) % 4:
            raise ValueError("unaligned oracle")
        decode(source, output, n)
        actual = C.string_at(output, n*4)
        if actual != scalar(packed):
            raise ValueError("Q5_K scalar/oracle mismatch: " + packed.hex())
        return actual

    fixed = pattern()
    result = dict(schema_version=1, generator_sha256=sha(Path(__file__)),
                  helper_sha256=sha(Path(__file__).with_name("generate_q4_1_goldens.py")),
                  oracle=dict(oracle.identity, build=built), model_sha256=MODEL_SHA, pattern_hex=fixed.hex(), partners=PARTNERS,
                  fingerprint={}, scales_fingerprint={}, examples=[])
    h = hashlib.sha256()
    finite = [n for n in range(65536) if n & 0x7c00 != 0x7c00]
    for field in range(2):
        for bits in finite:
            row = bytearray()
            for other in PARTNERS:
                block = bytearray(fixed)
                struct.pack_into("<HH", block, 0, bits if field == 0 else other, other if field == 0 else bits)
                row += block
            h.update(checked(row))
    count = 2*len(finite)*len(PARTNERS)
    result["fingerprint"] = dict(blocks=count, values=count*256, output_sha256=h.hexdigest())
    print(result["fingerprint"], flush=True)
    h = hashlib.sha256()
    for at in range(4, 16):
        row = bytearray()
        for byte in range(256):
            block = bytearray(fixed); block[at] = byte; row += block
        h.update(checked(row))
    result["scales_fingerprint"] = dict(blocks=3072, values=3072*256, output_sha256=h.hexdigest())

    def example(name, packed, **metadata):
        result["examples"].append(dict(name=name, packed_hex=packed.hex(), output_le_hex=checked(packed).hex(), **metadata))

    example("global-edges", b"".join(struct.pack("<HH", d, m)+fixed[4:] for d in EDGE for m in EDGE))
    blocks = []
    for g in range(8):
        for lane in range(32):
            block = bytearray(fixed); block[16:48] = b"\0"*32; block[16+lane] = 1 << g; blocks.append(block)
    example("isolated-high-bits", b"".join(blocks))
    rng = random.Random(0x5B10C)
    example("seeded-256", b"".join(struct.pack("<HH", rng.choice(finite), rng.choice(finite))+rng.randbytes(172) for _ in range(256)))
    inventory = oracle.inspect(model, samples=False)
    tensors = [t for t in inventory["tensors"] if t["type"] == 13]
    if len(tensors) != 48:
        raise ValueError("wrong Q5_K tensor count")
    with model.open("rb") as f:
        for t in tensors:
            if t["dims"] != [6144, 5120, 1, 1] or t["size"] != 21626880:
                raise ValueError("wrong tensor shape")
            n, row = t["size"]//176, t["dims"][0]//256
            indices = [0, 1, row-1, row, row+1, n//2, n-2, n-1]
            packed = bytearray()
            for index in indices:
                f.seek(inventory["data_offset"]+t["offset"]+index*176)
                block = f.read(176)
                if len(block) != 176:
                    raise ValueError("short read")
                packed += block
            example(t["name"], packed, tensor=t, block_indices=indices)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print("wrote", a.output, sha(a.output))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
