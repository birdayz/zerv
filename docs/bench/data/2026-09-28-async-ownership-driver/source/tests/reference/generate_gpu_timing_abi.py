#!/usr/bin/env python3
"""Independent header ABI for private benchmark timestamp queries (tests/fixtures/gpu/
timing-abi.json); not production math.

Bazel actions (tests/BUILD.bazel, `vulkan_abi`), checked against the committed fixture by
`//tests:vulkan_abi_test`; `bazel run //tests:vulkan_abi_update` rewrites it.

  generate_gpu_timing_abi.py source OUT.c
      the C program printing VkQueryPoolCreateInfo's layout and the timestamp constants
  generate_gpu_timing_abi.py record HEADER C_SOURCE PROBE_OUTPUT OUT.json
      the fixture: the compiled program's output plus the hashes of every input
"""
import hashlib
import json
from pathlib import Path
import sys

FIELDS = "sType pNext flags queryType queryCount pipelineStatistics".split()
CONSTANTS = "VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO VK_QUERY_TYPE_TIMESTAMP VK_QUERY_RESULT_64_BIT VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT".split()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source():
    code = ['#include <vulkan/vulkan_core.h>', '#include <stddef.h>', '#include <stdio.h>', 'int main(void) {',
            'printf("{\\"size\\":%zu,\\"alignment\\":%zu", sizeof(VkQueryPoolCreateInfo), _Alignof(VkQueryPoolCreateInfo));']
    for field in FIELDS:
        code.append(f'printf(",\\"{field}\\":%zu", offsetof(VkQueryPoolCreateInfo,{field}));')
    for name in CONSTANTS:
        code.append(f'printf(",\\"{name}\\":%u", (unsigned){name});')
    code += ['puts("}"); return 0; }']
    return "\n".join(code) + "\n"


def main():
    args = sys.argv[1:]
    if args[:1] == ["source"] and len(args) == 2:
        Path(args[1]).write_text(source())
    elif args[:1] == ["record"] and len(args) == 5:
        header, c, probe, out = args[1:]
        result = json.loads(Path(probe).read_text())
        result.update(header_sha256=sha(header), generator_sha256=sha(__file__), c_sha256=sha(c))
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
