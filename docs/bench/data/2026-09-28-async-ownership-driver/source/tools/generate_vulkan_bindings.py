#!/usr/bin/env python3
"""Emit scoped Zig C-ABI declarations (src/gpu/vk.zig) from the pinned public Vulkan registry.
A Bazel action (//src/gpu:vk_zig, registry from @vulkan_headers); `//src/gpu:vk_zig_test`
checks the committed file, `bazel run //src/gpu:vk_zig_update` rewrites it. No registry
dependency at native build or run time.

  generate_vulkan_bindings.py --registry VK_XML --revision TEXT --output FILE
"""
import argparse
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/reference"))
from vulkan_api import EXTENSION_COMMANDS, OPAQUE, inventory

PRIMITIVES = dict(void="void", char="u8", uint8_t="u8", uint32_t="u32", int32_t="i32", uint64_t="u64", size_t="usize", float="f32")


def identifier(name):
    return '@"' + name + '"' if name in {"type", "error", "align", "opaque", "test", "fn", "pub", "inline"} else name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True, help="vk.xml (pinned: @vulkan_headers)")
    p.add_argument("--revision", required=True, help="the registry's release, for the file header")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output must be fresh")
    xml = a.registry
    types, functions, constant_names = inventory(xml)
    root = ET.parse(xml).getroot()
    enum_nodes = {e.get("name"): e for e in root.findall(".//enum") if e.get("value") is not None or e.get("bitpos") is not None or e.get("offset") is not None}
    # An extension's own enumerants omit `extnumber`; it is the enclosing extension's number.
    extension_number = {e: ext.get("number") for ext in root.findall("extensions/extension") for e in ext.iter("enum")}

    def constant(name):
        if name == "VK_API_VERSION_1_1":
            return (1 << 22) | (1 << 12)
        e = enum_nodes[name]
        if e.get("bitpos") is not None:
            return 1 << int(e.get("bitpos"))
        if e.get("offset") is not None:  # registry rule for extension-numbered enumerants
            value = 1000000000 + (int(e.get("extnumber") or extension_number[e]) - 1) * 1000 + int(e.get("offset"))
            return -value if e.get("dir") == "-" else value
        text = e.get("value")
        if text == "(~0U)":
            return 0xffffffff
        return int(text, 0)

    def decl(node):
        name, original = node.findtext("name"), node.findtext("type")
        declaration = (node.text or "") + "".join(("".join(c.itertext()) if c.tag != "comment" else "") + (c.tail or "") for c in node)
        before, after = declaration.split(name, 1)
        depth = before.count("*")
        base = PRIMITIVES.get(original, original)
        if depth and (original == "void" or original in OPAQUE):
            base = "?*" + ("const " if "const" in before else "") + ("anyopaque" if original == "void" else base)
            if depth == 2:
                base = "?*" + base
            elif depth != 1:
                raise ValueError("unsupported pointer depth")
        else:
            for _ in range(depth):
                base = "[*c]" + ("const " if "const" in before else "") + base
        for size in reversed(re.findall(r"\[([^]]+)\]", after)):
            base = f"[{size}]{base}"
        return identifier(name), base, depth, original

    lines = ['//! Raw system API declarations, generated from Khronos Vulkan-Headers',
             f'//! {a.revision}; see tools/generate_vulkan_bindings.py.',
             '//! No inference code, C import, or third_party build dependency.',
             'const std = @import("std");', '']
    for name in constant_names:
        lines.append(f'pub const {name} = {constant(name)};')
    for name in sorted(OPAQUE):
        lines.append(f'pub const {name} = opaque {{}};')
    lines.append('pub const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;')
    for name, node in sorted(types.items()):
        category = node.get("category")
        if category == "struct":
            lines.append(f'pub const {name} = extern struct {{')
            for field in node.findall("member"):
                n, t, depth, original = decl(field)
                if field.get("values"):
                    default = field.get("values")
                elif depth or types.get(original, ET.Element("none")).get("category") == "handle":
                    default = "null"
                else:
                    default = f"std.mem.zeroes({t})"
                lines.append(f'    {n}: {t} = {default},')
            lines.append('};')
        elif category == "handle":
            lines.append(f'pub const {name} = ?*opaque {{}};')
        elif category == "enum":
            lines.append(f'pub const {name} = i32;')
        else:
            t = node.findtext("type")
            lines.append(f'pub const {name} = {PRIMITIVES.get(t, t)};')
    for name, node in functions.items():
        params = [f'{n}: {t}' for n, t, _, _ in map(decl, node.findall("param"))]
        result = node.findtext("proto/type")
        if name in EXTENSION_COMMANDS:  # fetched with vkGetDeviceProcAddr
            lines.append(f'pub const PFN_{name} = *const fn ({", ".join(params)}) callconv(.c) {PRIMITIVES.get(result, result)};')
        else:
            lines.append(f'pub extern fn {name}({", ".join(params)}) callconv(.c) {PRIMITIVES.get(result, result)};')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(a.output)


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:  # hermetic (docs/specs/hermetic-build.md)
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned Python and packages, not {sys.executable}")
    main()
