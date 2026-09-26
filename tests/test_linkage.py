"""What the executables load at run time (docs/specs/hermetic-build.md).

Every executable is static, or dynamic with exactly the system interfaces: the Vulkan loader
(linked against the stub //src/gpu:vulkan by its SONAME) and glibc. No RPATH/RUNPATH: nothing
may make the dynamic linker find the stub (or anything else from the build) at run time.
Parses the ELF dynamic section itself (no readelf in a hermetic test).
"""
import struct
import sys
import unittest
from pathlib import Path

# glibc (libm: Debug builds call its math functions; ReleaseFast ones inline them) and Vulkan.
ALLOWED = {"libvulkan.so.1", "libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2"}
DT_NEEDED, DT_STRTAB, DT_RPATH, DT_RUNPATH = 1, 5, 15, 29


def dynamic(path):
    """(needed names, has rpath/runpath) of an ELF64 little-endian executable; None if static."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1: raise ValueError(f"{path}: not ELF64 LE")
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    loads, dyn = [], None
    for i in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1: loads.append((p_vaddr, p_offset, p_filesz))
        if p_type == 2: dyn = (p_offset, p_filesz)
    if dyn is None: return None

    def offset(vaddr):
        for v, o, size in loads:
            if v <= vaddr < v + size: return o + vaddr - v
        raise ValueError(f"{path}: address {vaddr:#x} outside the loaded segments")
    entries = [struct.unpack_from("<qQ", data, dyn[0] + i) for i in range(0, dyn[1], 16)]
    strtab = offset(next(v for t, v in entries if t == DT_STRTAB))
    name = lambda at: data[strtab + at:data.index(b"\0", strtab + at)].decode()
    return [name(v) for t, v in entries if t == DT_NEEDED], any(t in (DT_RPATH, DT_RUNPATH) for t, _ in entries)


class Linkage(unittest.TestCase):
    def test_executables_load_only_system_interfaces(self):
        paths = [Path(p) for p in sys.argv[1:]] or [Path(p) for p in PATHS]
        self.assertGreater(len(paths), 20)
        dynamic_seen = 0
        for path in paths:
            with self.subTest(executable=path.name):
                info = dynamic(path)
                if info is None: continue
                dynamic_seen += 1
                needed, run_path = info
                self.assertFalse(run_path, "RPATH/RUNPATH set")
                self.assertLessEqual(set(needed), ALLOWED, f"loads {sorted(set(needed) - ALLOWED)}")
        self.assertGreater(dynamic_seen, 10, "the Vulkan executables should be dynamic")


PATHS = []
if __name__ == "__main__":
    PATHS, sys.argv[1:] = sys.argv[1:], []
    unittest.main()
