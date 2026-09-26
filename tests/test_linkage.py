"""What the executables load at run time (docs/specs/hermetic-build.md).

Every executable is static, or dynamic with exactly the system interfaces: the Vulkan loader
(linked against the stub //src/gpu:vulkan by its SONAME) and glibc. No RPATH/RUNPATH: nothing
may make the dynamic linker find the stub (or anything else from the build) at run time.
Parses the ELF dynamic section itself (tools/host_info.py; no readelf in a hermetic test).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools"))
from host_info import elf_dynamic as dynamic  # noqa: E402

# glibc (libm: Debug builds call its math functions; ReleaseFast ones inline them) and Vulkan.
ALLOWED = {"libvulkan.so.1", "libc.so.6", "libm.so.6", "ld-linux-x86-64.so.2"}


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
