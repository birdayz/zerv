#!/usr/bin/env python3
"""Independent native ABI of the scoped Vulkan API (tests/fixtures/gpu/abi.json): sizes,
alignments, field offsets and constants as the C compiler sees the pinned Khronos header.

Bazel actions (tests/BUILD.bazel, `vulkan_abi`), checked against the committed fixture by
`//tests:vulkan_abi_test`; `bazel run //tests:vulkan_abi_update` rewrites it.

  generate_vulkan_abi.py source VK_XML OUT.c
      the C program printing the ABI of every scoped structure and constant (vulkan_api.py)
  generate_vulkan_abi.py record VK_XML HEADER C_SOURCE PROBE_OUTPUT OUT.json
      the fixture: the compiled program's output plus the hashes of every input
"""
import hashlib
import json
from pathlib import Path
import sys

from vulkan_api import inventory


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source(xml):
    types, _, constants = inventory(xml)
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
    return "\n".join(lines) + "\n"


def main():
    args = sys.argv[1:]
    if args[:1] == ["source"] and len(args) == 3:
        Path(args[2]).write_text(source(args[1]))
    elif args[:1] == ["record"] and len(args) == 6:
        xml, header, c, probe, out = args[1:]
        result = json.loads(Path(probe).read_text())
        result.update(schema_version=1, generator_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name("vulkan_api.py")),
                      header_sha256=sha(header), xml_sha256=sha(xml), c_source_sha256=sha(c), target="Linux x86_64 SysV C ABI")
        with open(out, "w") as f:
            json.dump(result, f, indent=2); f.write("\n")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
