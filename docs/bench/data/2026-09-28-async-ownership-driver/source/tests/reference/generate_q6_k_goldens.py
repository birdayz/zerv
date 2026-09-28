#!/usr/bin/env python3
"""Independent external/scalar Q6_K goldens. Not a native dependency."""
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

SCALES = [-128, -127, -64, -33, -32, -2, -1, 0, 1, 2, 31, 32, 63, 64, 126, 127]


def pattern(phase):
    block = bytearray(210)
    struct.pack_into("<16be", block, 192, *SCALES, 1.0)
    for h in range(2):
        for g in range(4):
            for lane in range(32):
                q = phase*16 + lane % 16
                block[h*64+(g % 2)*32+lane] |= (q & 15) << (4*(g//2))
                block[128+h*32+lane] |= (q >> 4) << (g*2)
    return block


def scalar(packed):
    values = []
    for offset in range(0, len(packed), 210):
        block = packed[offset:offset+210]
        d, = struct.unpack_from("<e", block, 208)
        scales = struct.unpack_from("<16b", block, 192)
        # Decode by output index, independently from the native subgroup loop.
        for i in range(256):
            h, within = divmod(i, 128)
            g, lane = divmod(within, 32)
            low = block[h*64+(g % 2)*32+lane] // (16 if g >= 2 else 1) % 16
            high = block[128+h*32+lane] // (4**g) % 4
            factor = d * scales[i//16]
            values.append(factor * (low + high*16 - 32))
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
    decode = oracle.bind("dequantize_row_q6_K", None, [C.c_void_p, C.POINTER(C.c_float), C.c_int64])

    def checked(packed):
        if len(packed) % 210:
            raise ValueError("incomplete blocks")
        n = len(packed)//210*256
        source, output = C.create_string_buffer(bytes(packed)), (C.c_float*n)()
        if C.addressof(source) % 2 or C.addressof(output) % 4:
            raise ValueError("unaligned oracle")
        decode(source, output, n)
        actual = C.string_at(output, n*4)
        if actual != scalar(packed):
            raise ValueError("Q6_K scalar/oracle mismatch: " + packed.hex())
        return actual

    fixed = [pattern(p) for p in range(4)]
    result = dict(schema_version=1, generator_sha256=sha(Path(__file__)),
                  helper_sha256=sha(Path(__file__).with_name("generate_q4_1_goldens.py")),
                  oracle=dict(oracle.identity, build=built), model_sha256=MODEL_SHA, patterns_hex=[b.hex() for b in fixed],
                  fingerprint={}, scales_fingerprint={}, examples=[])
    h = hashlib.sha256()
    finite = [n for n in range(65536) if n & 0x7c00 != 0x7c00]
    for bits in finite:
        h.update(checked(b"".join(b[:208]+struct.pack("<H", bits) for b in fixed)))
    count = len(finite)*4
    result["fingerprint"] = dict(blocks=count, values=count*256, output_sha256=h.hexdigest())
    print(result["fingerprint"], flush=True)
    h = hashlib.sha256()
    for at in range(192, 208):
        row = bytearray()
        for byte in range(256):
            for fixed_block in fixed:
                block = bytearray(fixed_block)
                block[at] = byte
                struct.pack_into("<H", block, 208, 0x3555)
                row += block
        h.update(checked(row))
    result["scales_fingerprint"] = dict(blocks=16384, values=16384*256, output_sha256=h.hexdigest())

    def example(name, packed, **metadata):
        result["examples"].append(dict(name=name, packed_hex=packed.hex(), output_le_hex=checked(packed).hex(), **metadata))

    example("global-edges", b"".join(b[:208]+struct.pack("<H", bits) for bits in EDGE for b in fixed))
    blocks = []
    for at in range(192):
        for bit in range(8):
            block = bytearray(fixed[0])
            block[:192] = bytes(192)
            block[at] = 1 << bit
            blocks.append(block)
    example("isolated-low-and-high-bits", b"".join(blocks))
    rng = random.Random(0x6B10C)
    example("seeded-256", b"".join(rng.randbytes(208)+struct.pack("<H", rng.choice(finite)) for _ in range(256)))
    inventory = oracle.inspect(model, samples=False)
    tensors = [t for t in inventory["tensors"] if t["type"] == 14]
    if len(tensors) != 1:
        raise ValueError("wrong Q6_K tensor count")
    t = tensors[0]
    if t["name"] != "output.weight" or t["dims"] != [5120, 248320, 1, 1] or t["size"] != 1042944000:
        raise ValueError("wrong tensor shape")
    n, row = t["size"]//210, t["dims"][0]//256
    indices = sorted({k*(n-1)//63 for k in range(64)} | {1, row-1, row, row+1, n//2, n-2})
    packed = bytearray()
    with model.open("rb") as f:
        for index in indices:
            f.seek(inventory["data_offset"]+t["offset"]+index*210)
            block = f.read(210)
            if len(block) != 210:
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
