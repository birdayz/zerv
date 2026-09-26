#!/usr/bin/env python3
"""Build the native gemm_f16x pipeline binary (docs/specs/prefill.md, "Native gemm_f16x machine
code"; docs/bench/2026-09-24-gemm-f16x-isa.md). Development tool: needs the GPU with RADV, the
lab binary (bench/isa_lab/pipeline_binary_lab.c) and clang. Never run at build time.

  build_native_gemm.py --output-dir DIR [--lab third_party/isa-lab/pipeline_binary_lab]

Steps: the lab compiles src/model/shaders/gemm_f16x_q4_0.spv exactly as zerv creates it and
dumps RADV's pipeline binary, its key and the driver's global key; gen_f16x.py writes the
kernel's assembly; clang assembles it; isa_tool.py puts the code into the binary (config
unchanged); the lab's bitwise sweep must PASS. DIR (fresh) receives gemm_f16x_q4_0.{bin,key,
global,s}, bitwise.jsonl and manifest.json.
"""
import argparse, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "bench/isa_lab"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=pathlib.Path, required=True)
    ap.add_argument("--lab", type=pathlib.Path, default=ROOT / "third_party/isa-lab/pipeline_binary_lab")
    a = ap.parse_args()
    out = a.output_dir
    if out.exists():
        ap.error("output directory must be fresh")
    out.mkdir(parents=True)
    spv = ROOT / "src/model/shaders/gemm_f16x_q4_0.spv"
    base = out / "gemm_f16x_q4_0"
    run(a.lab, "dump", spv, out / "placeholder")
    run(sys.executable, LAB_DIR / "gen_f16x.py", f"{base}.s")
    run(sys.executable, LAB_DIR / "isa_tool.py", "asm", f"{base}.s", out / "code.bin")
    run(sys.executable, LAB_DIR / "isa_tool.py", "splice", out / "placeholder.bin", out / "code.bin", f"{base}.bin")
    (out / "gemm_f16x_q4_0.key").write_bytes((out / "placeholder.key").read_bytes())
    (out / "gemm_f16x_q4_0.global").write_bytes((out / "placeholder.global").read_bytes())
    run(sys.executable, LAB_DIR / "sweep.py", spv, f"{base}.bin", out / "bitwise.jsonl")
    for f in ("placeholder.bin", "placeholder.key", "placeholder.global", "code.bin"):
        (out / f).unlink()
    tools = {}
    for name, cmd in (("clang", ["clang", "--version"]), ("mesa", ["pacman", "-Q", "mesa", "vulkan-radeon"])):
        try:
            tools[name] = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
        except (OSError, subprocess.CalledProcessError):
            tools[name] = None
    manifest = dict(
        kernel="gemm_f16x_q4_0", driver="RADV (Mesa 26.2.3 binary layout; valid only where the global key matches)",
        spirv_sha256=sha(spv), generator_sha256=sha(LAB_DIR / "gen_f16x.py"), isa_tool_sha256=sha(LAB_DIR / "isa_tool.py"),
        sweep_sha256=sha(LAB_DIR / "sweep.py"), lab_source_sha256=sha(LAB_DIR / "pipeline_binary_lab.c"), tools=tools,
        files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.name != "manifest.json"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
