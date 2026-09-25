#!/usr/bin/env python3
"""ISA lab helper (research tool; docs/research/native-isa-via-vulkan.md).

  asm  IN.s OUT.code         assemble gfx1100 assembly with clang, write the raw .text
  dis2s IN.dis OUT.s         turn a RADV/ACO disassembly listing (RADV_DEBUG=shaders) into
                             assembler source (comments with encodings dropped)
  splice BLOB CODE OUT [--vgprs N] [--sgprs N] [--lds N]
                             replace the machine code of a RADV pipeline binary (Mesa 26.2.3
                             layout: header 656 bytes with total_size at 624 and code_size,
                             exec_size, ir, disasm, stats, debug sizes at 632..655; then
                             stats, code, ir, disasm, debug) and optionally config fields
  info BLOB                  print the header fields
"""
import struct, subprocess, sys, tempfile, os, re

HDR, TOTAL, SIZES = 656, 624, 632
CONFIG = 4  # ac_shader_config: num_sgprs, num_vgprs, num_shared_vgprs, spilled_sgprs, spilled_vgprs, lds_size, ...

def text_section(obj):
    d = open(obj, "rb").read()
    shoff, = struct.unpack_from("<Q", d, 0x28); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
    strtab = secs[shstrndx]
    for s in secs:
        name = d[strtab[4] + s[0]:].split(b"\0", 1)[0]
        if name == b".text": return d[s[4]:s[4] + s[5]]
    raise SystemExit("no .text")

def assemble(src, out):
    with tempfile.TemporaryDirectory() as t:
        obj = os.path.join(t, "a.o")
        subprocess.run(["clang", "-target", "amdgcn-mesa-mesa3d", "-mcpu=gfx1100", "-c", "-x", "assembler", src, "-o", obj], check=True)
        open(out, "wb").write(text_section(obj))

def dis2s(src, out):
    lines = []
    for line in open(src):
        line = line.rstrip("\n")
        m = re.match(r"\s*\(then repeated (\d+) times\)", line)
        if m:  # ACO compresses runs of identical instructions
            lines.extend([lines[-1]] * int(m.group(1)))
            continue
        code = line.split(";", 1)[0].rstrip()
        if not code.strip(): continue
        lines.append(code)
    open(out, "w").write("\t.text\n" + "\n".join(lines) + "\n")

def header(d):
    total, = struct.unpack_from("<I", d, TOTAL)
    code, exe, ir, dis, st, dbg = struct.unpack_from("<6I", d, SIZES)
    return dict(total=total, code=code, exec=exe, ir=ir, disasm=dis, stats=st, debug=dbg)

def splice(blob, code_path, out, vgprs=None, sgprs=None, lds=None):
    d = bytearray(open(blob, "rb").read()); h = header(d)
    if h["total"] != len(d) or HDR + h["stats"] + h["code"] + h["ir"] + h["disasm"] + h["debug"] != len(d): raise SystemExit(f"unexpected layout {h}")
    code = open(code_path, "rb").read()
    # Pad with s_code_end (0xbf9f0000) like ACO (it appends 5 words; instruction prefetch
    # may run past the end): at least 5 words, up to a 64-byte boundary.
    words = 5 + ((-(len(code) + 20)) % 64) // 4
    code_padded = code + bytes.fromhex("00009fbf") * words
    start = HDR + h["stats"]
    new = d[:start] + code_padded + d[start + h["code"]:]
    struct.pack_into("<I", new, TOTAL, len(new))
    struct.pack_into("<2I", new, SIZES, len(code_padded), len(code))
    if sgprs is not None: struct.pack_into("<I", new, CONFIG + 0, sgprs)
    if vgprs is not None: struct.pack_into("<I", new, CONFIG + 4, vgprs)
    if lds is not None: struct.pack_into("<I", new, CONFIG + 20, lds)
    open(out, "wb").write(new)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "asm": assemble(a[1], a[2])
    elif a[0] == "dis2s": dis2s(a[1], a[2])
    elif a[0] == "info": print(header(open(a[1], "rb").read()), struct.unpack_from("<14I", open(a[1], "rb").read(), CONFIG))
    elif a[0] == "splice":
        kw = {}; rest = a[4:]
        for k, v in zip(rest[::2], rest[1::2]): kw[k.lstrip("-")] = int(v)
        splice(a[1], a[2], a[3], **kw)
    else: raise SystemExit(__doc__)
