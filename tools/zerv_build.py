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
import sys
import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAZEL = "bazelisk"

# Executable name -> Bazel label.
TARGETS = {
    "zerv": "//src:zerv",
    "inferencex-client": "//tools:inferencex_client",
    "inferencex-results": "//tools:inferencex_results",
    "zig": "//bazel:zig",
    # External test oracles, source-built (docs/specs/hermetic-build.md, phase 3).
    "libggml-base.so": "@ggml//:ggml_base_so",
    "libllama.so.0.4.1": "@llama_cpp//:llama_so",
    "oracle_tokenizer_pieces": "//tests:oracle_tokenizer_pieces",
    "oracle_tokenizer_bench": "//tests:oracle_tokenizer_bench",
    "oracle_model_quant_bench": "//tests:oracle_model_quant_bench",
    "oracle_gpu_matvec": "//tests:oracle_gpu_matvec",
    "oracle_vulkan_driver": "//tests:oracle_vulkan_driver",
    "oracle_model": "//tests:oracle_model",
    "oracle_llama_batch_capture": "//tests:oracle_llama_batch_capture",
    # Development tool (bench/isa_lab): RADV pipeline binaries of native kernels.
    "pipeline_binary_lab": "//bench/isa_lab:pipeline_binary_lab",
    "glslc": "@shaderc//:glslc",
    "vulkaninfo": "@vulkan_tools//:vulkaninfo",
    # Serving competitor (docs/specs/hermetic-build.md, phase 5).
    "llama-server": "//bench:llama-server",
    "spirv-val": "@spirv_tools//:spirv-val",
    "zerv-model-capture": "//tools:zerv-model-capture",
    "zerv-inspect": "//tools:zerv-inspect",
    "zerv-prompt-capture": "//tools:zerv-prompt-capture",
    **{name: f"//bench:{name}" for name in [
        "zerv-gpu-driver-bench", "zerv-gpu-matvec-bench", "zerv-gemm-bench", "zerv-decode-f16-bench",
        "zerv-matvec-rows-bench", "zerv-model-profile", "zerv-reuse-profile", "zerv-prefix-check", "zerv-kv-quality",
        "zerv-spec-check", "zerv-batch-check", "zerv-disk-probe", "zerv-storage-bench", "zerv-residency-bench", "zerv-cache-sources-bench", "zerv-pressure-bench", "zerv-preparation-bench", "zerv-archive-bench", "zerv-archive-model-check", "zerv-mtp-check", "zerv-kernel-chain", "zerv-coopmat-probe",
        "zerv-quant-bench", "zerv-tokenizer-bench", "zerv-split-bench", "zerv-nfc-bench", "zerv-sampler-bench",
        "zerv-chat-bench", "zerv-gguf-bench", "zerv-model-quant-bench"]},
}

# The unit tests a measurement is gated on: every test except the GPU ones (the default
# `bazel test //...`: Zig tests in both modes, format check, Python tests), and the GPU tests
# in both modes for GPU measurements.
TESTS = ("//...",)
GPU_TESTS = ("//...", "//tests:gpu", "//tests:gpu_release_fast")
# The GPU tests on the host's driver (production's), run by production benchmarks in their own
# Bazel invocation: --test_env applies to every test of an invocation (docs/specs/hermetic-build.md).
HOST_GPU_TESTS = ("//tests:gpu_host", "//tests:gpu_host_release_fast")


# Every Bazel invocation goes through tools/bazel (--nohome_rc --nosystem_rc). bazelisk sets
# BAZELISK_SKIP_WRAPPER for the wrapper it runs; a harness started under `bazel run` inherits it
# and would reach Bazel without the wrapper, i.e. with ~/.bazelrc.
os.environ.pop("BAZELISK_SKIP_WRAPPER", None)


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
    # A target used as a tool elsewhere in the graph also has an exec-configuration output.
    exe = [f for f in files if Path(f).name in (label.rsplit(":", 1)[1], name) and "-exec/" not in f]
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


def oracle(name):
    """A source-built test oracle (library or reference program), built in the default
    configuration; returns its path and identity (Bazel label, content hash) for manifests."""
    path = build(name, config=None)[name]
    return path, dict(label=TARGETS[name], sha256=sha(path), path=str(path.relative_to(ROOT)))


GPU_RUNTIME = "//tests:gpu_runtime"


def host_vulkan_env(base=None):
    """`base` (default os.environ) without any variable that configures a Vulkan loader or
    driver (VK_*, RADV_*, ACO_*, MESA_*, AMD_*, LD_LIBRARY_PATH): the host's Vulkan stack as
    installed."""
    return {k: v for k, v in (os.environ if base is None else base).items()
            if not k.startswith(("VK_", "RADV_", "ACO_", "MESA_", "AMD_")) and k != "LD_LIBRARY_PATH"}


def gpu_runtime(base=None):
    """The test-only GPU runtime (docs/specs/hermetic-build.md, phase 4: the source-built Vulkan
    loader and Mesa RADV), built. Returns (environment, identity): host_vulkan_env(base) plus
    the variables of tests/BUILD.bazel's GPU_ENV with absolute paths, so
    a child process loads that runtime and nothing of the host's Vulkan stack; and the runtime's
    label and file hashes for manifests."""
    env = host_vulkan_env(base)
    bazel("build", GPU_RUNTIME)
    files = sorted(ROOT / f for f in bazel("cquery", "--output=files", GPU_RUNTIME, capture=True).stdout.split())
    d = files[0].parent
    if {f.name for f in files} != {"amdgpu.ids", "libvulkan.so.1", "libvulkan_radeon.so", "radeon_icd.json"} or any(f.parent != d for f in files):
        raise SystemExit(f"{GPU_RUNTIME}: unexpected files {files}")
    env.update(LD_LIBRARY_PATH=str(d), VK_DRIVER_FILES=str(d / "radeon_icd.json"), VK_LOADER_LAYERS_DISABLE="~all~",
               AMDGPU_ASIC_ID_TABLE_PATHS=str(d), ZERV_TEST_GPU_RUNTIME="test_radv")
    return env, dict(label=GPU_RUNTIME, files={f.name: sha(f) for f in files})


def test(*labels):
    """Runs Bazel tests (default: all but the GPU tests) before a measurement."""
    subprocess.run(test_command(*labels), cwd=ROOT, check=True)


def host_gpu_test_command():
    """The host-driver GPU tests, keyed on the installed driver's identity (a cached result is
    reused only while the driver is unchanged)."""
    import host_info
    return [BAZEL, "test", f"--test_env=ZERV_HOST_VULKAN_ID={host_info.host_vulkan_id()}", *HOST_GPU_TESTS]


def test_host_gpu():
    """Runs the host-driver GPU tests: the gate of every production (host-driver) benchmark."""
    subprocess.run(host_gpu_test_command(), cwd=ROOT, check=True)


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


def source_revision(root=ROOT):
    """The checked-out commit and ref, read from the repository's files (no git executable;
    worktrees and packed refs handled). Uncommitted changes are not visible here: manifests
    hash the sources themselves. None outside a git checkout."""
    dot = root / ".git"
    if dot.is_file():
        gitdir = (root / dot.read_text().split("gitdir:", 1)[1].strip()).resolve()
    elif dot.is_dir():
        gitdir = dot
    else:
        return None
    common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve() if (gitdir / "commondir").exists() else gitdir
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref: "): return dict(commit=head, ref=None)
    ref = head[5:]
    for d in (gitdir, common):
        if (d / ref).exists(): return dict(commit=(d / ref).read_text().strip(), ref=ref)
    packed = common / "packed-refs"
    for line in packed.read_text().splitlines() if packed.exists() else []:
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref: return dict(commit=parts[0], ref=ref)
    return dict(commit=None, ref=ref)


def provenance():
    """Build identity for a manifest: Bazel's version, the Zig toolchain (version, hash), the
    checked-out revision and the hash of every build definition file."""
    zig = binary("zig", config=None)
    version = subprocess.run([zig, "version"], check=True, text=True, capture_output=True).stdout.strip()
    label = bazel("version", capture=True).stdout
    return dict(bazel_version=next(l.split(": ", 1)[1] for l in label.splitlines() if l.startswith("Build label")),
                zig_version=version, zig_sha256=sha(zig.resolve()), source_revision=source_revision(),
                build_files={str(p.relative_to(ROOT)): sha(p) for p in build_files()})


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    import argparse
    parser = argparse.ArgumentParser(description="Build zerv executables with Bazel; print their paths.")
    parser.add_argument("names", nargs="*", choices=sorted(TARGETS))
    parser.add_argument("--config", default="release", help="Bazel config (.bazelrc); default release")
    parser.add_argument("--test-host-gpu", action="store_true", help="run the GPU tests on the host's driver (HOST_GPU_TESTS)")
    args = parser.parse_args()
    if args.test_host_gpu: test_host_gpu()
    elif not args.names: parser.error("name executables to build, or --test-host-gpu")
    for name, path in (build(*args.names, config=args.config) if args.names else {}).items():
        print(path)
