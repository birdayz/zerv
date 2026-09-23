#!/usr/bin/env python3
"""Extract independent native ABI sizes/offsets/constants from pinned Khronos C header."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from vulkan_api import inventory

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.work.exists():
        p.error("output/work must be fresh")
    headers = ROOT / "third_party/vulkan/1.4.354"
    for source in json.loads((ROOT / "docs/research/2026-09-22/vulkan-sources.json").read_text()):
        if sha(ROOT / source["local_path"]) != source["sha256"]:
            raise ValueError("research source mismatch")
    types, _, constants = inventory(headers / "registry/vk.xml")
    lines = ['#include <vulkan/vulkan_core.h>', '#include <stddef.h>', '#include <stdio.h>', 'int main(void) {', 'puts("{\\"structs\\":[");']
    structs = sorted(n for n, t in types.items() if t.get("category") == "struct")
    for i, name in enumerate(structs):
        lines.append(f'printf("{"," if i else ""}{{\\"name\\":\\"{name}\\",\\"size\\":%zu,\\"alignment\\":%zu,\\"fields\\":{{", sizeof({name}), _Alignof({name}));')
        for j, field in enumerate(types[name].findall("member")):
            f = field.findtext("name")
            lines.append(f'printf("{"," if j else ""}\\"{f}\\":%zu", offsetof({name}, {f}));')
        lines.append('puts("}}");')
    lines.append('puts("],\\"constants\\":{");')
    for i, name in enumerate(constants):
        lines.append(f'printf("{"," if i else ""}\\"{name}\\":%lld", (long long){name});')
    lines += ['puts("}}");', 'return 0;', '}']
    a.work.mkdir(parents=True)
    c = a.work / "abi.c"; exe = (a.work / "abi").resolve()
    c.write_text("\n".join(lines) + "\n")
    command = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + str(headers / "include"), str(c), "-o", str(exe)]
    subprocess.run(command, check=True)
    result = json.loads(subprocess.check_output([exe], text=True))
    result.update(schema_version=1, generator_sha256=sha(Path(__file__)), helper_sha256=sha(Path(__file__).with_name("vulkan_api.py")),
                  header_sha256=sha(headers / "include/vulkan/vulkan_core.h"), xml_sha256=sha(headers / "registry/vk.xml"),
                  c_source_sha256=sha(c), target="Linux x86_64 SysV C ABI")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print(len(structs), "structs,", len(constants), "constants:", sha(a.output))


if __name__ == "__main__":
    main()
