#!/usr/bin/env python3
"""Build the native gemm_f16x pipeline binary (docs/specs/prefill.md, "Native gemm_f16x machine
code" and "One binary per driver build"; docs/bench/2026-09-24-gemm-f16x-isa.md). Development
tool: needs the GPU. Never run at build time.

  build_native_gemm.py --runtime host|test --output-dir DIR [--lab PATH]

--runtime selects the driver build the binary is for: `host` the host's installed Vulkan stack
(production; src/model/native/), `test` the test-only GPU runtime built from source
(//tests:gpu_runtime; src/model/native/test_radv/). Steps: the lab (default: built by Bazel,
//bench/isa_lab:pipeline_binary_lab) compiles src/model/shaders/gemm_f16x_q4_0.spv on that
driver exactly as zerv creates it and dumps RADV's pipeline binary, its key and the driver's
global key; gen_f16x.py writes the kernel's assembly; the toolchain's Zig (`zig clang`, as
//src/model:native_code) assembles it; isa_tool.py puts the code into the binary (config
unchanged); the lab's bitwise sweep on the same driver must PASS. DIR (fresh) receives
gemm_f16x_q4_0.{bin,key,global,s}, bitwise.jsonl and manifest.json.
"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "bench/isa_lab"
sys.path.insert(0, str(ROOT / "tools"))
import zerv_build  # noqa: E402  (tools/zerv_build.py: Bazel builds and provenance)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime", choices=("host", "test"), required=True)
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--lab", type=pathlib.Path, help="default: built by Bazel")
    a = ap.parse_args()
    out = a.output_dir
    if out.exists():
        ap.error("output directory must be fresh")
    lab = a.lab or zerv_build.binary("pipeline_binary_lab", config=None)
    zig = zerv_build.binary("zig", config=None)
    if a.runtime == "test":
        env, runtime = zerv_build.gpu_runtime()
        driver = "RADV built from source (//tests:gpu_runtime, Mesa 26.2.3; valid only where the global key matches)"
    else:
        env = zerv_build.host_vulkan_env()
        driver = "RADV of the host (Mesa 26.2.3 binary layout; valid only where the global key matches)"
        try:
            runtime = dict(packages=subprocess.run(["pacman", "-Q", "mesa", "vulkan-radeon"], capture_output=True,
                                                   text=True, check=True).stdout.strip().splitlines())
        except (OSError, subprocess.CalledProcessError):
            runtime = dict(packages=None)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tools:
        # isa_tool.py assembles with `clang` from PATH: the toolchain's Zig.
        clang = pathlib.Path(tools) / "clang"
        clang.write_text(f'#!/bin/sh\nexec "{zig}" clang "$@"\n')
        clang.chmod(0o755)
        env = dict(env, PATH=f"{tools}:{env.get('PATH', '/usr/bin:/bin')}")

        def run(*cmd):
            print("+", " ".join(str(c) for c in cmd), flush=True)
            subprocess.run([str(c) for c in cmd], check=True, env=env)

        spv = ROOT / "src/model/shaders/gemm_f16x_q4_0.spv"
        base = out / "gemm_f16x_q4_0"
        run(lab, "dump", spv, out / "placeholder")
        run(sys.executable, LAB_DIR / "gen_f16x.py", f"{base}.s")
        run(sys.executable, LAB_DIR / "isa_tool.py", "asm", f"{base}.s", out / "code.bin")
        run(sys.executable, LAB_DIR / "isa_tool.py", "splice", out / "placeholder.bin", out / "code.bin", f"{base}.bin")
        (out / "gemm_f16x_q4_0.key").write_bytes((out / "placeholder.key").read_bytes())
        (out / "gemm_f16x_q4_0.global").write_bytes((out / "placeholder.global").read_bytes())
        run(sys.executable, LAB_DIR / "sweep.py", spv, f"{base}.bin", out / "bitwise.jsonl", "--lab", lab)
    for f in ("placeholder.bin", "placeholder.key", "placeholder.global", "code.bin"):
        (out / f).unlink()
    zig_version = subprocess.run([zig, "version"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(
        kernel="gemm_f16x_q4_0", driver=driver, runtime=a.runtime, driver_runtime=runtime,
        spirv_sha256=sha(spv), generator_sha256=sha(LAB_DIR / "gen_f16x.py"), isa_tool_sha256=sha(LAB_DIR / "isa_tool.py"),
        sweep_sha256=sha(LAB_DIR / "sweep.py"), lab_source_sha256=sha(LAB_DIR / "pipeline_binary_lab.c"),
        lab=dict(label=zerv_build.TARGETS["pipeline_binary_lab"], sha256=sha(lab)) if a.lab is None else dict(path=str(lab), sha256=sha(lab)),
        tools=dict(assembler=f"zig {zig_version} clang (//bazel:zig)", zig_sha256=sha(pathlib.Path(zig).resolve())),
        files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.name != "manifest.json"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
