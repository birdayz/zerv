#!/usr/bin/env python3
"""Rebuild owned Vulkan1.1 matvec modules with pinned offline tools; never runtime code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PINS = {"glslc": "4a4743cde357af0949cdbc07668802297327e993e548f80f4e2ee67ba9b6c74d",
        "spirv-val": "02fae2475ba0f3cb4987aca3c9eb938e8543cf269244bfcb856d8e0c252c753b"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): p.error("fresh directory required")
    for tool, digest in PINS.items():
        path = shutil.which(tool)
        if not path or sha(path) != digest: raise ValueError("tool pin mismatch: "+tool)
    a.output_dir.mkdir(parents=True)
    manifest = dict(source_sha256=sha(ROOT/"src/matvec/matvec.comp"), tools=PINS, modules={})
    for name, fmt, width, payload in (("f32", 0, 4, 0), ("q4_0", 2, 18, 2), ("q4_1", 3, 20, 4), ("q5_k", 13, 176, 48), ("q6_k", 14, 210, 0)):
        output = a.output_dir/(name+".spv")
        subprocess.run(["glslc", "--target-env=vulkan1.1", "-O", "-fshader-stage=compute", f"-DFORMAT={fmt}", f"-DBLOCK_BYTES={width}", f"-DPAYLOAD_OFFSET={payload}", str(ROOT/"src/matvec/matvec.comp"), "-o", str(output)], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
        manifest["modules"][name] = dict(sha256=sha(output), bytes=output.stat().st_size)
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
