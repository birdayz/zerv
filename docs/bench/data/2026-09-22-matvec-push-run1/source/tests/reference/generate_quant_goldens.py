#!/usr/bin/env python3
"""External oracle tool only. The native build never loads ggml or runs this file."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; generate a new file and review the diff")
    if sys.byteorder != "little" or ctypes.sizeof(ctypes.c_float) != 4:
        parser.error("oracle extraction requires little-endian 32-bit C floats")
    library_path = args.library.resolve(strict=True)
    library = ctypes.CDLL(str(library_path))

    def identity(name):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_char_p
        return function().decode("utf-8")

    def oracle(fmt, packed):
        block_bytes = 18 if fmt == "q4_0" else 34
        if len(packed) % block_bytes:
            raise ValueError("incomplete reference input")
        count = len(packed) // block_bytes * 32
        function = getattr(library, "dequantize_row_" + fmt)
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
        function.restype = None
        source = ctypes.create_string_buffer(packed)
        output = (ctypes.c_float * count)()
        if ctypes.addressof(source) % 2 or ctypes.addressof(output) % 4:
            raise RuntimeError("unaligned reference buffer")
        function(source, output, count)
        return ctypes.string_at(output, count * 4)

    def independent(fmt, packed):
        block_bytes = 18 if fmt == "q4_0" else 34
        values = []
        for offset in range(0, len(packed), block_bytes):
            scale = struct.unpack_from("<e", packed, offset)[0]
            payload = packed[offset + 2:offset + block_bytes]
            if fmt == "q4_0":
                coefficients = [byte % 16 - 8 for byte in payload]
                coefficients += [byte // 16 - 8 for byte in payload]
            else:
                coefficients = struct.unpack("<32b", payload)
            values.extend(coefficient * scale for coefficient in coefficients)
        return struct.pack("<" + "f" * len(values), *values)

    def checked_output(fmt, packed):
        actual = oracle(fmt, packed)
        expected = independent(fmt, packed)
        if actual != expected:
            raise RuntimeError(f"{fmt} oracle/scalar mismatch for {packed.hex()}")
        return actual

    result = {
        "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle": {
            "library_path": str(library_path),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "version": identity("ggml_version"),
            "commit": identity("ggml_commit"),
            "source_commit": "456172ec733a135778adcd32d00e576a58232e45",
            "source_note": "Inspected source; binary may report dirty. Binary hash is authoritative.",
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "pattern": "v1: scale_bits ascending finite binary16; q4 byte[j]=j|((15-j)<<4); q8 bytes=0..255 in 8 blocks",
        "formats": {},
        "examples": [],
    }
    for fmt in ("q4_0", "q8_0"):
        digest = hashlib.sha256()
        cases = values = 0
        for scale_bits in range(65536):
            if scale_bits & 0x7c00 == 0x7c00:
                continue
            scale = struct.pack("<H", scale_bits)
            if fmt == "q4_0":
                packed = scale + bytes(j | ((15 - j) << 4) for j in range(16))
            else:
                packed = b"".join(scale + bytes(range(start, start + 32)) for start in range(0, 256, 32))
            output = checked_output(fmt, packed)
            digest.update(output)
            cases += 1
            values += len(output) // 4
        result["formats"][fmt] = {"scale_cases": cases, "values": values, "output_sha256": digest.hexdigest()}
        print(fmt, result["formats"][fmt], flush=True)

        scales = [0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff]
        width = 16 if fmt == "q4_0" else 32
        packed = b"".join(
            struct.pack("<H", scale) + bytes((block * 37 + j * 19) % 256 for j in range(width))
            for block, scale in enumerate(scales)
        )
        result["examples"].append({
            "format": fmt,
            "packed_hex": packed.hex(),
            "output_le_hex": checked_output(fmt, packed).hex(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
