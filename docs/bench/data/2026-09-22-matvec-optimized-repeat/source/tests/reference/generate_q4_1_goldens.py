#!/usr/bin/env python3
"""External Q4_1 oracle + independent scalar goldens; never a native dependency."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import platform
import random
import struct
import sys

from gguf_oracle import Oracle

EDGE = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff]
MODEL_SHA = "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
LIB_SHA = "7d9065538f5df6342613b4fa92e661d5ad8fd811c2dbe16ff0e4b62a77777073"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def scalar(packed):
    values = []
    for at in range(0, len(packed), 20):
        d, m = struct.unpack_from("<ee", packed, at)
        data = packed[at + 4:at + 20]
        values.extend(d * (b & 15) + m for b in data)
        values.extend(d * (b >> 4) + m for b in data)
    return struct.pack("<" + "f" * len(values), *values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists; generate separately and review")
    if sys.byteorder != "little" or C.sizeof(C.c_float) != 4:
        raise ValueError("requires little-endian IEEE float")
    lib, model = a.library.resolve(strict=True), a.model.resolve(strict=True)
    if sha(lib) != LIB_SHA or sha(model) != MODEL_SHA:
        raise ValueError("oracle/model identity mismatch")
    reference = Oracle(lib)
    decode = reference.bind("dequantize_row_q4_1", None, [C.c_void_p, C.POINTER(C.c_float), C.c_int64])

    def checked(packed):
        if len(packed) % 20:
            raise ValueError("incomplete reference blocks")
        n = len(packed) // 20 * 32
        source = C.create_string_buffer(packed)
        output = (C.c_float * n)()
        if C.addressof(source) % 2 or C.addressof(output) % 4:
            raise ValueError("unaligned oracle buffer")
        decode(source, output, n)
        result = C.string_at(output, n * 4)
        if result != scalar(packed):
            raise ValueError("independent scalar/oracle mismatch: " + packed.hex())
        return result

    result = dict(schema_version=1, generator_sha256=sha(Path(__file__)), oracle=reference.identity,
                  source_commit="456172ec733a135778adcd32d00e576a58232e45", python=platform.python_version(),
                  model_sha256=MODEL_SHA, edge_fields=EDGE, fingerprint={}, examples=[])
    digest = hashlib.sha256()
    blocks = 0
    payload = bytes(j | ((15 - j) << 4) for j in range(16))
    finite = [bits for bits in range(65536) if bits & 0x7c00 != 0x7c00]
    for field in range(2):
        for bits in finite:
            packed = b"".join(struct.pack("<HH", bits, other) + payload if field == 0 else
                              struct.pack("<HH", other, bits) + payload for other in EDGE)
            digest.update(checked(packed))
            blocks += len(EDGE)
    result["fingerprint"] = dict(blocks=blocks, values=blocks * 32, output_sha256=digest.hexdigest())
    print(result["fingerprint"], flush=True)

    def example(name, packed, **metadata):
        result["examples"].append(dict(name=name, packed_hex=packed.hex(), output_le_hex=checked(packed).hex(), **metadata))

    example("all-packed-bytes", b"".join(struct.pack("<HH", EDGE[b % 11], EDGE[b // 11 % 11]) + bytes([b]) * 16 for b in range(256)))
    example("all-edge-pairs", b"".join(struct.pack("<HH", d, m) + payload for d in EDGE for m in EDGE))
    rng = random.Random(0x414)
    example("seeded-1024", b"".join(struct.pack("<HH", rng.choice(finite), rng.choice(finite)) + rng.randbytes(16) for _ in range(1024)))
    inventory = reference.inspect(model, samples=False)
    tensors = [t for t in inventory["tensors"] if t["type"] == 3]
    if len(tensors) != 8:
        raise ValueError("unexpected Q4_1 tensor count")
    with model.open("rb") as f:
        for tensor in tensors:
            if tensor["dims"] != [17408, 5120, 1, 1] or tensor["size"] != 55705600:
                raise ValueError("unexpected Q4_1 shape")
            n = tensor["size"] // 20
            row = tensor["dims"][0] // 32
            indices = [0, 1, row - 1, row, row + 1, n // 2, n - 2, n - 1]
            parts = []
            for index in indices:
                f.seek(inventory["data_offset"] + tensor["offset"] + index * 20)
                part = f.read(20)
                if len(part) != 20:
                    raise ValueError("truncated model")
                parts.append(part)
            example(tensor["name"], b"".join(parts), tensor=tensor, block_indices=indices,
                    absolute_offset=inventory["data_offset"] + tensor["offset"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("wrote", a.output, sha(a.output))


if __name__ == "__main__":
    main()
