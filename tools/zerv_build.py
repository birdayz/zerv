"""Build zerv executables with Bazel and return their paths (docs/development.md, "Bazel").

Harnesses call `binary("zerv-kv-quality")` (or `build(...)` for several) instead of running a
build tool and reading a fixed output directory: the path comes from `bazel cquery` for the
same configuration that was built, so a benchmark never measures a binary of another
configuration. Default configuration `release` (ReleaseFast, native CPU; .bazelrc).
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Executable name -> Bazel label.
TARGETS = {
    "zerv": "//src:zerv",
    "zerv-model-capture": "//tools:zerv-model-capture",
    "zerv-inspect": "//tools:zerv-inspect",
    **{name: f"//bench:{name}" for name in [
        "zerv-gpu-driver-bench", "zerv-gpu-matvec-bench", "zerv-gemm-bench", "zerv-decode-f16-bench",
        "zerv-matvec-rows-bench", "zerv-model-profile", "zerv-prefix-check", "zerv-kv-quality",
        "zerv-spec-check", "zerv-batch-check", "zerv-mtp-check", "zerv-kernel-chain", "zerv-coopmat-probe",
        "zerv-quant-bench", "zerv-tokenizer-bench", "zerv-split-bench", "zerv-nfc-bench", "zerv-sampler-bench",
        "zerv-chat-bench", "zerv-gguf-bench", "zerv-model-quant-bench"]},
}


def bazel(*args, capture=False):
    return subprocess.run(["bazelisk", *args], cwd=ROOT, check=True, text=True, capture_output=capture)


def build(*names, config="release"):
    """Builds the named executables; returns {name: absolute path}."""
    labels = [TARGETS[n] for n in names]
    bazel("build", f"--config={config}", *labels)
    paths = {}
    for name, label in zip(names, labels):
        files = bazel("cquery", f"--config={config}", "--output=files", label, capture=True).stdout.split()
        exe = [f for f in files if Path(f).name == label.rsplit(":", 1)[1]]
        if len(exe) != 1: raise SystemExit(f"{label}: expected one executable among {files}")
        paths[name] = ROOT / exe[0]
    return paths


def binary(name, config="release"):
    """One executable, built; its absolute path."""
    return build(name, config=config)[name]


def test(*labels, config=None):
    """Runs Bazel tests (e.g. the unit tests before a measurement)."""
    bazel("test", *([f"--config={config}"] if config else []), *labels)
