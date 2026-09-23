#!/usr/bin/env python3
"""Pinned independent GPU/CPU matvec fixtures; no native code or network required."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

from gguf_oracle import Oracle
from generate_q4_1_goldens import LIB_SHA, MODEL_SHA, sha
from generate_q5_k_goldens import scalar as scalar_q5
from generate_q6_k_goldens import scalar as scalar_q6

ROOT = Path(__file__).resolve().parents[2]
TYPES = {"f32": (0, 1, 4), "q4_0": (2, 32, 18), "q4_1": (3, 32, 20), "q5_k": (13, 256, 176), "q6_k": (14, 256, 210)}
PINS = {
    "/usr/lib/libggml-base.so.0.24.0": LIB_SHA,
    "/usr/lib/ggml/libggml-vulkan.so": "d09aac86141492bdf22daad0c61b5ded720f772b3f18d8167aae1f264532979a",
    "/usr/include/ggml-backend.h": "46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2",
    "/usr/include/ggml-alloc.h": "94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23",
    "/usr/include/ggml-vulkan.h": "7eae5dad2cc7bb4d3eca828539f441816d5fd59fdc7b224d49aa229fc7b7248c",
}


def canonical(values):
    return struct.pack("<"+"f"*len(values), *(0.0 if x == 0 else x for x in values))


def scalar(fmt, packed):
    if fmt == "f32": return packed
    if fmt == "q5_k": return scalar_q5(packed)
    if fmt == "q6_k": return scalar_q6(packed)
    stride = TYPES[fmt][2]
    result = []
    for pos in range(0, len(packed), stride):
        block = packed[pos:pos+stride]
        d, = struct.unpack_from("<e", block)
        minimum = struct.unpack_from("<e", block, 2)[0] if fmt == "q4_1" else 0
        payload = block[4 if fmt == "q4_1" else 2:]
        for c in range(32):
            q = (payload[c % 16] >> (4*(c//16))) & 15
            result.append(d*(q if fmt == "q4_1" else q-8)+minimum)
    return struct.pack("<"+"f"*len(result), *result)


def zero_block(fmt):
    if fmt == "f32": return bytearray(4)
    b = bytearray(TYPES[fmt][2])
    struct.pack_into("<e", b, 208 if fmt == "q6_k" else 0, 1)
    if fmt == "q4_0": b[2:] = bytes([0x88])*16
    if fmt == "q5_k": b[4:16] = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    if fmt == "q6_k": b[128:192] = bytes([0xaa])*64; b[192:208] = bytes([1])*16
    return b


def set_coefficient(fmt, b, c, q):
    if fmt == "f32": struct.pack_into("<f", b, 0, q); return
    if fmt in ("q4_0", "q4_1"):
        if fmt == "q4_0": q += 8
        at, shift = (2 if fmt == "q4_0" else 4)+c % 16, 4*(c//16)
    elif fmt == "q5_k":
        g, lane = divmod(c, 32)
        b[16+lane] = (b[16+lane] & ~(1 << g)) | ((q >> 4) << g)
        at, shift = 48+(g//2)*32+lane, 4*(g % 2)
    else:
        q += 32
        h, part = divmod(c, 128); g, lane = divmod(part, 32)
        hi, hs = 128+h*32+lane, 2*g
        b[hi] = (b[hi] & ~(3 << hs)) | ((q >> 4) << hs)
        at, shift = h*64+(g % 2)*32+lane, 4*(g//2)
    b[at] = (b[at] & ~(15 << shift)) | ((q & 15) << shift)


def encode_case(case, packed, x):
    _, elements, size = TYPES[case["format"]]
    if case["columns"] % elements or len(packed) != case["columns"]//elements*case["rows"]*size or len(x) != case["columns"]*4:
        raise ValueError("bad generated layout")
    return struct.pack("<8I", 0x38564d5a, 1, TYPES[case["format"]][0], case["columns"], case["rows"], len(packed), len(x), 0)+packed+x


def metrics(actual, ideal, sumabs, exact=False, enforce=True):
    if not actual or len(actual) != len(ideal) or len(actual) != len(sumabs): raise ValueError("wrong output shape")
    if not all(math.isfinite(v) for values in (actual, ideal, sumabs) for v in values): raise ValueError("nonfinite output")
    errors = [abs(a-b) for a, b in zip(actual, ideal)]
    if enforce and any(e > (0 if exact else 2e-6+2e-6*s) for e, s in zip(errors, sumabs)):
        raise ValueError("independent matvec mismatch: " + str(max(errors)))
    return dict(max_abs=max(errors), max_relative=max(e/max(abs(b), 1e-6) for e, b in zip(errors, ideal)),
                max_error_over_sumabs=max(e/max(s, 1e-6) for e, s in zip(errors, sumabs)),
                normalized_l2=math.sqrt(math.fsum(e*e for e in errors)/max(math.fsum(b*b for b in ideal), 1e-12)), nonfinite=0)


def build_oracle(directory):
    for path, digest in PINS.items():
        if sha(Path(path)) != digest: raise ValueError("pin mismatch: "+path)
    # Headers come from the installed-identical pinned research tree.
    include = ROOT/"third_party/ggml/456172ec733a135778adcd32d00e576a58232e45/include"
    for name in ("ggml.h", "ggml-backend.h", "ggml-alloc.h", "ggml-vulkan.h"):
        if sha(include/name) != sha(Path("/usr/include")/name): raise ValueError("header mismatch: "+name)
    binary = directory/"gpu-matvec-oracle"
    command = ["cc", "-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-I"+str(include), str(ROOT/"tests/reference/gpu_matvec.c"), "-o", str(binary),
               "/usr/lib/ggml/libggml-vulkan.so", "/usr/lib/libggml-base.so.0.24.0", "-lm"]
    subprocess.run(command, check=True)
    (directory/"build.json").write_text(json.dumps(dict(command=command, compiler=subprocess.check_output(["cc", "--version"], text=True), binary_sha256=sha(binary)), indent=2)+"\n")
    return binary


def run_oracle(binary, case_path, output, iterations=0, default=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_VK_")}
    if not default: env["GGML_VK_DISABLE_MMVQ"] = "1"
    proc = subprocess.run([str(binary), str(case_path), str(output), str(iterations)], env=env, text=True, capture_output=True, timeout=300)
    output.with_suffix(".stdout").write_text(proc.stdout)
    output.with_suffix(".stderr").write_text(proc.stderr)
    if proc.returncode: raise RuntimeError(f"oracle failed {proc.returncode}: {proc.stderr}")
    if "AMD Radeon RX 7900 XTX" not in proc.stderr: raise ValueError("unexpected reference device")
    raw = output.read_bytes()
    if len(raw) % 20: raise ValueError("invalid oracle output")
    n = len(raw)//20
    return struct.unpack_from("<"+"d"*n, raw), struct.unpack_from("<"+"d"*n, raw, n*8), struct.unpack_from("<"+"f"*n, raw, n*16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.work_dir.exists(): p.error("fresh output and work directory required")
    if sha(a.model) != MODEL_SHA: raise ValueError("model mismatch")
    a.work_dir.mkdir(parents=True)
    binary = build_oracle(a.work_dir.resolve())
    oracle = Oracle("/usr/lib/libggml-base.so.0.24.0")
    result = dict(schema_version=1, model_sha256=MODEL_SHA, pins=PINS, oracle_binary_sha256=sha(binary), sources={}, cases=[])
    for name in ("generate_gpu_matvec.py", "gpu_matvec.c", "gguf_oracle.py", "generate_q4_1_goldens.py", "generate_q5_k_goldens.py", "generate_q6_k_goldens.py"):
        result["sources"][name] = sha(Path(__file__).with_name(name))

    def add(case, packed, x, exact=False, expected_half=None):
        case["exact"] = exact
        raw = encode_case(case, packed, x)
        path = a.work_dir/(case["name"]+".case"); path.write_bytes(raw)
        case["input_sha256"] = hashlib.sha256(raw).hexdigest()
        ideal, sums, gpu = run_oracle(binary, path, a.work_dir/(case["name"]+".output"))
        if len(ideal) != case["rows"]: raise ValueError("oracle row mismatch")
        if expected_half is not None:
            if tuple(expected_half) != ideal or tuple(abs(x) for x in expected_half) != sums: raise ValueError("half-domain scalar mismatch")
            case["output_sha256"] = hashlib.sha256(canonical(ideal)).hexdigest()
        else:
            decoded = scalar(case["format"], packed)
            vals = struct.unpack("<"+"f"*(len(decoded)//4), decoded)
            xv = struct.unpack("<"+"f"*case["columns"], x)
            for r in range(case["rows"]):
                products = [w*v for w, v in zip(vals[r*case["columns"]:(r+1)*case["columns"]], xv)]
                dot, absolute = math.fsum(products), math.fsum(map(abs, products))
                if abs(dot-ideal[r]) > 1e-12*max(absolute, 1) or abs(absolute-sums[r]) > 1e-12*max(absolute, 1): raise ValueError("CPU/Python scalar mismatch")
            case.update(packed_hex=packed.hex(), input_hex=x.hex(), ideal=list(ideal), sumabs=list(sums), reference_hex=canonical(gpu).hex())
        case["reference_metrics"] = metrics(gpu, ideal, sums, exact)
        result["cases"].append(case)
        print(case["name"], case["reference_metrics"], flush=True)

    rng = random.Random(0x38564d)
    for fmt, (_, elements, width) in TYPES.items():
        k = 17 if fmt == "f32" else elements*3
        rows = k+1
        packed = bytearray()
        for r in range(rows):
            row = zero_block(fmt)*(k//elements)
            if r < k:
                block, lane = divmod(r, elements)
                b = row[block*width:(block+1)*width]
                # Upper bitplanes and signed Q6 scale edges are covered by seeded/real rows too.
                coefficient = 1 if fmt == "f32" else (7 if fmt == "q4_0" else 15 if fmt == "q4_1" else 31)
                set_coefficient(fmt, b, lane, coefficient)
                row[block*width:(block+1)*width] = b
            packed += row
        x = struct.pack("<"+"f"*k, *((i+1)/1024 for i in range(k)))
        add(dict(name=fmt+"-isolated-columns", kind="explicit", format=fmt, columns=k, rows=rows), packed, x, True)
        for k, rows in ((17 if fmt == "f32" else elements, 3), (5120, 17), (6144 if fmt == "q5_k" else 17408, 3)):
            if fmt == "f32":
                packed = struct.pack("<"+"f"*(k*rows), *(rng.randint(-4096, 4096)/4096 for _ in range(k*rows)))
            else:
                packed = bytearray(rng.randbytes(k//elements*rows*width))
                for pos in range(0, len(packed), width):
                    for field in ([208] if fmt == "q6_k" else [0, 2] if fmt in ("q4_1", "q5_k") else [0]):
                        struct.pack_into("<H", packed, pos+field, rng.choice([0, 1, 0x8001, 0x3555, 0xb555, 0x1400, 0x9400]))
            x = struct.pack("<"+"f"*k, *(rng.randint(-1024, 1024)/1024 for _ in range(k)))
            add(dict(name=f"{fmt}-seeded-{k}", kind="explicit", format=fmt, columns=k, rows=rows), packed, x)
        k = 5120
        packed = bytearray()
        for sign in (1, -1, 0):
            b = zero_block(fmt)
            for c in range(elements): set_coefficient(fmt, b, c, 3 if fmt != "f32" else 3*sign)
            if fmt != "f32": struct.pack_into("<e", b, 208 if fmt == "q6_k" else 0, sign)
            packed += b*(k//elements)
        x = [1.0 if c % 2 == 0 else -1.0 for c in range(k)]; x[-1] += 1/1024
        add(dict(name=fmt+"-cancellation", kind="explicit", format=fmt, columns=k, rows=3), packed, struct.pack("<"+"f"*k, *x), True)
        add(dict(name=fmt+"-zero-input", kind="explicit", format=fmt, columns=k, rows=3), packed, bytes(k*4), True)
        if fmt == "f32": continue
        for field in ([208] if fmt == "q6_k" else [0, 2] if fmt in ("q4_1", "q5_k") else [0]):
            base = zero_block(fmt)
            if field == 2:
                if fmt == "q5_k": base[8:12] = bytes([1])*4; base[12:16] = bytes([0x11])*4
            else: set_coefficient(fmt, base, 0, 1)
            finite = [bits for bits in range(65536) if bits & 0x7c00 != 0x7c00]
            packed = bytearray(); expected = []
            for bits in finite:
                b = bytearray(base); struct.pack_into("<H", b, field, bits); packed += b
                value = struct.unpack("<e", struct.pack("<H", bits))[0]
                expected.append(-value if field == 2 and fmt == "q5_k" else value)
            x = struct.pack("<"+"f"*elements, 1, *([0]*(elements-1)))
            add(dict(name=f"{fmt}-half-{field}", kind="half", format=fmt, columns=elements, rows=len(finite), base_hex=base.hex(), field=field), packed, x, True, expected)

    ramp = [(r % 257-128)/128 for r in range(65537)]
    add(dict(name="f32-dispatch-tail", kind="row_ramp", format="f32", columns=1, rows=len(ramp)),
        struct.pack("<"+"f"*len(ramp), *ramp), struct.pack("<f", 1/16), True, [v/16 for v in ramp])
    inventory = oracle.inspect(a.model, samples=False)
    seen = set()
    with a.model.open("rb") as f:
        for t in inventory["tensors"]:
            if t["name"].startswith("blk.64.") or t["name"] == "token_embd.weight" or t["dims"][1] == 1 or t["dims"][0] == 4: continue
            key = t["type"], tuple(t["dims"][:2])
            if key in seen: continue
            seen.add(key)
            fmt = next(name for name, layout in TYPES.items() if layout[0] == t["type"])
            k, rows = t["dims"][:2]; row_bytes = t["size"]//rows
            indices = [0, rows//2, rows-1]; packed = bytearray()
            for r in indices:
                f.seek(inventory["data_offset"]+t["offset"]+r*row_bytes); packed += f.read(row_bytes)
            x = struct.pack("<"+"f"*k, *(rng.randint(-1024, 1024)/1024 for _ in range(k)))
            add(dict(name=f"model-{fmt}-{k}-{rows}", kind="explicit", format=fmt, columns=k, rows=3, tensor=t, row_indices=indices), packed, x)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print("wrote", a.output, sha(a.output))


if __name__ == "__main__": main()
