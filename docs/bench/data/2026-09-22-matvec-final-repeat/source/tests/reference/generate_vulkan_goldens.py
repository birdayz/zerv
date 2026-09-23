#!/usr/bin/env python3
"""Compile/validate diagnostic shader and independently check external Vulkan execution."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    "/usr/bin/glslc": "4a4743cde357af0949cdbc07668802297327e993e548f80f4e2ee67ba9b6c74d",
    "/usr/bin/spirv-val": "02fae2475ba0f3cb4987aca3c9eb938e8543cf269244bfcb856d8e0c252c753b",
    "/usr/lib/libvulkan.so.1": "7d9f8ced1fae1d02ee7953a5a22e4efae74545e83333eedc41c53a50f51ebab6",
    "/usr/lib/libvulkan_radeon.so": "ddc11778b3e01b73d55028595a6dfc51afd8e3cb5f901fd175a74cdc9ba79248",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def input_word(i):
    if i < 3:
        return (0, 0xffffffff, 0x80000000)[i]
    return (i * 0x9e3779b9 & 0xffffffff) ^ 0xa5a5a5a5


def expected(kind, n):
    count = n+64 if kind == "affine" else n//4
    values = array("I", (input_word(i) for i in range(count)))
    ih = hashlib.sha256(values).hexdigest()
    if kind == "affine":
        for i in range(count):
            values[i] = (((values[i] * 1664525 + 1013904223) & 0xffffffff) ^ i) if i < n else 0xcdcdcdcd
    return dict(kind=kind, count=n, bytes=count*4, input_sha256=ih, output_sha256=hashlib.sha256(values).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shader-output", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.shader_output.exists() or a.work.exists():
        p.error("all outputs must be fresh")
    if sys.byteorder != "little" or array("I").itemsize != 4:
        raise ValueError("unsupported oracle host")
    for path, digest in PINS.items():
        if sha(path) != digest:
            raise ValueError("tool/driver identity mismatch: " + path)
    source = ROOT / "tests/fixtures/gpu/affine.comp"
    c = ROOT / "tests/reference/vulkan_driver.c"
    a.work.mkdir(parents=True)
    module, reference = (a.work / "affine.spv").resolve(), (a.work / "reference").resolve()
    commands = [["/usr/bin/glslc", "--target-env=vulkan1.1", "-O", str(source), "-o", str(module)],
                ["/usr/bin/spirv-val", "--target-env", "vulkan1.1", str(module)],
                ["cc", "-std=c11", "-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-I"+str(ROOT / "third_party/vulkan/1.4.354/include"), str(c), "-lvulkan", "-lcrypto", "-o", str(reference)],
                [str(reference), str(module)]]
    outputs = []
    with (a.work / "commands.log").open("x") as log:
        for command in commands:
            r = subprocess.run(command, capture_output=True, text=True)
            log.write(json.dumps(command)+"\n"+r.stdout+r.stderr); log.flush(); r.check_returncode(); outputs.append(r)
    records = [json.loads(line) for line in outputs[-1].stdout.splitlines()]
    cases = [expected("affine", n) for n in (1, 63, 64, 65, 5120, 65537, 1048576)]
    cases += [expected("roundtrip", n) for n in (256, 1048576, 67108864)]
    if records != cases:
        raise ValueError("external GPU output differs from independent CPU hashes")
    result = dict(schema_version=1, generator_sha256=sha(__file__), shader_source_sha256=sha(source),
                  reference_source_sha256=sha(c), shader_sha256=sha(module), reference_sha256=sha(reference),
                  tool_driver_sha256=PINS, device=outputs[-1].stderr.strip(), cases=cases,
                  compiler_flags=["--target-env=vulkan1.1", "-O"], validation="spirv-val --target-env vulkan1.1")
    a.shader_output.parent.mkdir(parents=True, exist_ok=True)
    with a.shader_output.open("xb") as f:
        f.write(module.read_bytes())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print("verified", len(cases), "real transfer/dispatch cases:", sha(a.output), "shader", sha(a.shader_output))


if __name__ == "__main__":
    main()
