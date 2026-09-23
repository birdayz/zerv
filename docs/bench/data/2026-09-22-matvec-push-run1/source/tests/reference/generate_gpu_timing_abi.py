#!/usr/bin/env python3
"""Independent header ABI for private benchmark timestamp queries; not production math."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADER_SHA = "55c06a17793bb1a9d752aa0906116fff9d1fb74f5dd3896777b92f124c9637aa"
FIELDS = "sType pNext flags queryType queryCount pipelineStatistics".split()
CONSTANTS = "VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO VK_QUERY_TYPE_TIMESTAMP VK_QUERY_RESULT_64_BIT VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT".split()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.work.exists() or a.output.exists(): p.error("fresh paths required")
    headers = ROOT/"third_party/vulkan/1.4.354/include"
    if sha(headers/"vulkan/vulkan_core.h") != HEADER_SHA: raise ValueError("header changed")
    a.work.mkdir(parents=True)
    code = ['#include <vulkan/vulkan_core.h>', '#include <stddef.h>', '#include <stdio.h>', 'int main(void) {',
            'printf("{\\"size\\":%zu,\\"alignment\\":%zu", sizeof(VkQueryPoolCreateInfo), _Alignof(VkQueryPoolCreateInfo));']
    for field in FIELDS:
        code.append(f'printf(",\\"{field}\\":%zu", offsetof(VkQueryPoolCreateInfo,{field}));')
    for name in CONSTANTS:
        code.append(f'printf(",\\"{name}\\":%u", (unsigned){name});')
    code += ['puts("}"); return 0; }']
    source=a.work/"abi.c"; source.write_text("\n".join(code)+"\n")
    binary=(a.work/"abi").resolve()
    command=["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I"+str(headers), str(source), "-o", str(binary)]
    subprocess.run(command, check=True)
    result=json.loads(subprocess.check_output([binary],text=True))
    result.update(header_sha256=HEADER_SHA, generator_sha256=sha(__file__), c_sha256=sha(source), command=command)
    # The ABI fixture excludes local build paths to make regeneration byte-identical.
    result.pop("command")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__": main()
