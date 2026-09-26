#!/usr/bin/env python3
"""Check real Vulkan transfers and dispatches of the diagnostic shader with an independent C
program against independent CPU hashes; write tests/fixtures/gpu/dispatch.json.

  tools/py tests/reference/generate_vulkan_goldens.py --output FILE --work DIR

Needs the GPU. Runs on the test-only GPU runtime (the Vulkan loader and Mesa RADV built from
source, docs/specs/hermetic-build.md phase 4), never the host's Vulkan stack. The shader is the
committed tests/fixtures/gpu/affine.spv, which Bazel regenerates with the source-built glslc
and checks (//tests:vulkan_fixtures_test); the program is //tests:oracle_vulkan_driver.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]  # not resolved: Bazel tests import it from their runfiles
SHADER = "tests/fixtures/gpu/affine.spv"
SHADER_SOURCE = "tests/fixtures/gpu/affine.comp"
REFERENCE_SOURCE = "tests/reference/vulkan_driver.c"


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
    sys.path.insert(0, str(ROOT / "tools"))
    import zerv_build  # tools/zerv_build.py: Bazel builds, the GPU runtime
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.work.exists():
        p.error("all outputs must be fresh")
    if sys.byteorder != "little" or array("I").itemsize != 4:
        raise ValueError("unsupported oracle host")
    zerv_build.test("//tests:vulkan_fixtures_test")  # the committed shader is what the pinned tools produce
    reference, reference_identity = zerv_build.oracle("oracle_vulkan_driver")
    env, runtime = zerv_build.gpu_runtime()
    a.work.mkdir(parents=True)
    command = [str(reference), str(ROOT / SHADER)]
    r = subprocess.run(command, capture_output=True, text=True, env=env)
    (a.work / "commands.log").write_text(json.dumps(command) + "\n" + r.stdout + r.stderr)
    r.check_returncode()
    records = [json.loads(line) for line in r.stdout.splitlines()]
    cases = [expected("affine", n) for n in (1, 63, 64, 65, 5120, 65537, 1048576)]
    cases += [expected("roundtrip", n) for n in (256, 1048576, 67108864)]
    if records != cases:
        raise ValueError("external GPU output differs from independent CPU hashes")
    result = dict(schema_version=2, generator_sha256=sha(__file__), shader_source_sha256=sha(ROOT / SHADER_SOURCE),
                  reference_source_sha256=sha(ROOT / REFERENCE_SOURCE), shader_sha256=sha(ROOT / SHADER),
                  reference=dict(label=reference_identity["label"], sha256=reference_identity["sha256"]),
                  shader_tools=dict(glslc=zerv_build.TARGETS["glslc"], spirv_val=zerv_build.TARGETS["spirv-val"],
                                    check="//tests:vulkan_fixtures_test"),
                  gpu_runtime=runtime, device=r.stderr.strip(), cases=cases,
                  compiler_flags=["--target-env=vulkan1.1", "-O"], validation="spirv-val --target-env vulkan1.1")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print("verified", len(cases), "real transfer/dispatch cases:", sha(a.output))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
