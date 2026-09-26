#!/usr/bin/env python3
"""Build the llama.cpp-RDNA3-7900xtx-opt serving competitor (HIP for gfx1100) from pinned
inputs (docs/specs/hermetic-build.md, phase 5; docs/performance.md, competitor set).

  tools/py tools/build_competitor_rdna3.py [--output third_party/competitors/rdna3-15995a12]

Inputs, pinned by content: the fork's source archive (GitHub, commit 15995a12, sha256) and the
vLLM ROCm image (by digest; its ROCm 7.2.3 clang, CMake and Ninja), which also runs the result
(bench/run_serving.py). The build runs in that image with no network, as our uid, without
capabilities, the source read-only. Configuration: the fork's README build command with the
image's compiler paths, plus what the image cannot provide (no libcurl/OpenSSL: HTTPS model
downloads only) and the build number and commit that git would supply (the archive has no
.git). OUTPUT (fresh) receives bin/ (llama-server and its shared libraries), the source
archive's tree under src/, and manifest.json (inputs, command, output hashes).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fetch_hf import download  # noqa: E402  (resuming HTTPS download, no host tool)

COMMIT = "15995a12d1d530645a4f34c72afdaa30fa680149"
ARCHIVE = f"https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt/archive/{COMMIT}.tar.gz"
ARCHIVE_SHA256 = "952679dfeb5101f6c0f0afe9a244e9d70d77d38eba18bf6e49388ff4af713c32"
IMAGE = "vllm/vllm-openai-rocm@sha256:2e7da1ad1c66836802072588adea75f9f4991da5f9545b4318e91d422c22ce6a"
BUILD_NUMBER = "10823"  # `git rev-list --count` of the commit (the fork's git build, 2026-09-26)
LLVM = "/opt/rocm/lib/llvm/bin"
CMAKE = ["cmake", "-S", "/src", "-B", "/build", "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_C_COMPILER={LLVM}/clang", f"-DCMAKE_CXX_COMPILER={LLVM}/clang++", f"-DCMAKE_HIP_COMPILER={LLVM}/clang",
         "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600",
         "-DGGML_HIP=ON", "-DGGML_HIP_GRAPHS=ON", "-DAMDGPU_TARGETS=gfx1100",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF",
         f"-DLLAMA_BUILD_NUMBER={BUILD_NUMBER}", f"-DLLAMA_BUILD_COMMIT={COMMIT[:9]}"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / f"third_party/competitors/rdna3-{COMMIT[:8]}")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists(): p.error("output must be fresh")
    out.mkdir(parents=True)
    archive = out / "source.tar.gz"
    download(ARCHIVE, archive)
    if sha(archive) != ARCHIVE_SHA256: raise SystemExit(f"source archive hash mismatch: {sha(archive)}")
    with tarfile.open(archive) as t:
        t.extractall(out / "src", filter="data")
    source = out / "src" / f"llama.cpp-RDNA3-7900xtx-opt-{COMMIT}"
    build = out / "build"; build.mkdir()
    docker = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
              "--security-opt", "no-new-privileges", "-e", "HOME=/tmp", "-v", f"{source}:/src:ro", "-v", f"{build}:/build",
              "--entrypoint", "/bin/sh", IMAGE, "-c"]
    script = " ".join(f"'{c}'" if " " in c else c for c in CMAKE) + f" && cmake --build /build -j {a.jobs} --target llama-server"
    started = datetime.now(timezone.utc).isoformat()
    with (out / "build.log").open("w") as log:
        subprocess.run(docker + [script], stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / "bin").mkdir()
    binaries = sorted(p for p in (build / "bin").iterdir() if p.name == "llama-server" or ".so" in p.name)
    for b in binaries:
        (out / "bin" / b.name).write_bytes(b.read_bytes()); (out / "bin" / b.name).chmod(0o755)
    manifest = dict(started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                    source=dict(url=ARCHIVE, sha256=ARCHIVE_SHA256, commit=COMMIT), image=IMAGE,
                    command=docker + [script], build_number=BUILD_NUMBER,
                    outputs={f"bin/{b.name}": sha(out / "bin" / b.name) for b in binaries})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
