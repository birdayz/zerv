#!/usr/bin/env python3
"""Build zerv executables with Bazel and return their paths (docs/development.md, "Bazel").

Harnesses call `binary("zerv-kv-quality")` (or `build(...)` for several) instead of running a
build tool and reading a fixed output directory: the path comes from `bazel cquery` for the
same configuration that was built, so a benchmark never measures a binary of another
configuration. Default configuration `release` (ReleaseFast, native CPU; .bazelrc).

Harnesses that log every command they run use the `*_command` forms with their own runner,
then `path(...)`. `provenance()` is the build identity a benchmark manifest records.

From a shell: `tools/zerv_build.py NAME...` builds the executables (release) and prints
their paths, e.g. `"$(tools/zerv_build.py zerv-spec-check)" MODEL`.
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAZEL = "bazelisk"

# Executable name -> Bazel label.
TARGETS = {
    "zerv": "//src:zerv",
    "zig": "//bazel:zig",
    "glslc": "@shaderc//:glslc",
    "spirv-val": "@spirv_tools//:spirv-val",
    "zerv-model-capture": "//tools:zerv-model-capture",
    "zerv-inspect": "//tools:zerv-inspect",
    **{name: f"//bench:{name}" for name in [
        "zerv-gpu-driver-bench", "zerv-gpu-matvec-bench", "zerv-gemm-bench", "zerv-decode-f16-bench",
        "zerv-matvec-rows-bench", "zerv-model-profile", "zerv-prefix-check", "zerv-kv-quality",
        "zerv-spec-check", "zerv-batch-check", "zerv-mtp-check", "zerv-kernel-chain", "zerv-coopmat-probe",
        "zerv-quant-bench", "zerv-tokenizer-bench", "zerv-split-bench", "zerv-nfc-bench", "zerv-sampler-bench",
        "zerv-chat-bench", "zerv-gguf-bench", "zerv-model-quant-bench"]},
}

# The unit tests a measurement is gated on: every test except the GPU ones (the default
# `bazel test //...`: Zig tests in both modes, format check, Python tests), and the GPU tests
# in both modes for GPU measurements.
TESTS = ("//...",)
GPU_TESTS = ("//...", "//tests:gpu", "//tests:gpu_release_fast")


def bazel(*args, capture=False):
    return subprocess.run([BAZEL, *args], cwd=ROOT, check=True, text=True, capture_output=capture)


def _config(config):
    """`--config=release` etc.; None: Bazel's default configuration (Debug)."""
    return [f"--config={config}"] if config else []


def build_command(*names, config="release"):
    return [BAZEL, "build", *_config(config), *(TARGETS[n] for n in names)]


def test_command(*labels):
    return [BAZEL, "test", *(labels or TESTS)]


def path(name, config="release", root=ROOT):
    """The absolute path of an executable built by `build_command(name, config=config)` in the
    workspace `root` (default this one; research copies are workspaces of their own)."""
    label = TARGETS[name]
    files = subprocess.run([BAZEL, "cquery", *_config(config), "--output=files", label], cwd=root, check=True,
                           text=True, capture_output=True).stdout.split()
    exe = [f for f in files if Path(f).name == label.rsplit(":", 1)[1]]
    if len(exe) != 1: raise SystemExit(f"{label}: expected one executable among {files}")
    return root / exe[0]


def build(*names, config="release"):
    """Builds the named executables; returns {name: absolute path}."""
    subprocess.run(build_command(*names, config=config), cwd=ROOT, check=True)
    return {name: path(name, config) for name in names}


def binary(name, config="release"):
    """One executable, built; its absolute path."""
    return build(name, config=config)[name]


def shader_tools():
    """(glslc, spirv-val): the shader tools built from source (MODULE.bazel), for harnesses
    that compile experimental variants; never the host's."""
    tools = build("glslc", "spirv-val", config=None)
    return str(tools["glslc"]), str(tools["spirv-val"])


def test(*labels):
    """Runs Bazel tests (default: all but the GPU tests) before a measurement."""
    subprocess.run(test_command(*labels), cwd=ROOT, check=True)


def sha(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def build_files():
    """Every file that defines how zerv is built: Bazel version and flags, module and lock
    files (toolchain and dependency pins), macros and BUILD files. A benchmark copies and
    hashes these with its sources."""
    fixed = [ROOT / name for name in (".bazelversion", ".bazelrc", ".bazelignore", "MODULE.bazel", "MODULE.bazel.lock",
                                      "requirements_lock.txt")]
    builds = [p for p in ROOT.rglob("BUILD.bazel")
              if not (p.relative_to(ROOT).parts[0].startswith(("bazel-", ".", "third_party", "models")))]
    return fixed + sorted((ROOT / "bazel").glob("*.bzl")) + sorted(builds)


def provenance():
    """Build identity for a manifest: Bazel's version, the Zig toolchain (version, hash) and
    the hash of every build definition file."""
    zig = binary("zig", config=None)
    version = subprocess.run([zig, "version"], check=True, text=True, capture_output=True).stdout.strip()
    label = bazel("version", capture=True).stdout
    return dict(bazel_version=next(l.split(": ", 1)[1] for l in label.splitlines() if l.startswith("Build label")),
                zig_version=version, zig_sha256=sha(zig.resolve()),
                build_files={str(p.relative_to(ROOT)): sha(p) for p in build_files()})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build zerv executables with Bazel; print their paths.")
    parser.add_argument("names", nargs="+", choices=sorted(TARGETS))
    parser.add_argument("--config", default="release", help="Bazel config (.bazelrc); default release")
    args = parser.parse_args()
    for name, path in build(*args.names, config=args.config).items():
        print(path)
