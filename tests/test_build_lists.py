"""Every Zig test file runs: it is a unit test binary in build.zig (`unit_tests`), part of
the GPU aggregate (imported by tests/gpu.zig), or a helper imported by one of those.
Since the per-file test binaries (docs/development.md, "Tests in parallel") a file missing
from the list would silently not run."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def imports(path):
    return set(re.findall(r'@import\("([a-z0-9_]+)\.zig"\)', path.read_text()))


class BuildLists(unittest.TestCase):
    def test_every_test_file_runs(self):
        build = (ROOT / "build.zig").read_text()
        block = re.search(r"const unit_tests = \[_\]UnitTest\{(.*?)\n\};", build, re.S)
        self.assertIsNotNone(block, "unit_tests list not found in build.zig")
        unit = set(re.findall(r'\.name = "([a-z0-9_]+)"', block.group(1)))
        for name in unit:
            self.assertTrue((TESTS / f"{name}.zig").is_file(), f"build.zig lists missing tests/{name}.zig")
        # Reachable: the unit roots, the GPU root, and everything they import (transitively).
        reached, todo = set(), list(unit | {"gpu"})
        while todo:
            name = todo.pop()
            if name in reached: continue
            reached.add(name)
            todo.extend(imports(TESTS / f"{name}.zig"))
        for path in sorted(TESTS.glob("*.zig")):
            with self.subTest(file=path.name):
                self.assertIn(path.stem, reached, f"{path.name} is neither a unit test in build.zig nor imported by one or by tests/gpu.zig")
        # GPU files stay out of the CPU unit list (they need the device, and run serially).
        self.assertFalse(unit & (imports(TESTS / "gpu.zig") | {"gpu"}), "a GPU test file is in the CPU unit list")

    def test_packages_are_directories_with_declared_imports(self):
        """Every src/ package is `src/NAME/root.zig`, listed in build.zig `packages`; a package
        imports other packages only by module name, and only those it declares."""
        build = (ROOT / "build.zig").read_text()
        block = re.search(r"const packages = \[_\]Package\{(.*?)\n\};", build, re.S)
        self.assertIsNotNone(block)
        declared = {m.group(1): set(re.findall(r'"([a-z0-9_]+)"', m.group(2)))
                    for m in re.finditer(r'\.name = "([a-z0-9_]+)", \.deps = &\.\{([^}]*)\}', block.group(1))}
        dirs = {p.parent.name for p in (ROOT / "src").glob("*/root.zig")}
        self.assertEqual(dirs, set(declared), "src/ package directories and build.zig `packages` differ")
        for name, deps in declared.items():
            for path in (ROOT / "src" / name).rglob("*.zig"):
                text = path.read_text()
                with self.subTest(file=str(path.relative_to(ROOT))):
                    self.assertNotIn('@import("../', text, "cross-package relative import: import the package module")
                    self.assertNotIn('@import("zerv")', text, "a package cannot import the umbrella")
                    used = set(re.findall(r'@import\("([a-z_]+)"\)', text)) & set(declared)
                    self.assertLessEqual(used, deps, f"imports undeclared packages {sorted(used - deps)}")


if __name__ == "__main__":
    unittest.main()
