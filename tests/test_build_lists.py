"""The Bazel build declares what the code uses (docs/development.md, "Bazel").

- Every Zig file under tests/ belongs to a target in tests/BUILD.bazel (a test's main or
  srcs, a helper library), so no test file silently goes unbuilt.
- Every src/ package is `src/NAME/root.zig` with a `zig_library` named NAME in
  src/NAME/BUILD.bazel, and a package imports other packages only if its BUILD lists them in
  `deps`. Zig resolves imports lazily, so an unreferenced import of an undeclared package
  compiles; this check does not depend on that.
- No harness builds with anything but Bazel (tools/zerv_build.py): no `zig build`, zig-out/
  or a Zig outside the toolchain. Exception: bench/rebuild_*_baseline.py rebuild archived
  trees, which carry their own build.zig.
- Every script (bench/, tools/) refuses to run under the host's Python: tools/py runs it with
  the pinned interpreter and packages. tests/reference/ generators are pinned by hash in the
  fixtures they produced; they get the guard when the fixtures are regenerated with the
  source-built oracles (docs/specs/hermetic-build.md, phase 3).
- The build definitions name no host path (docs/specs/hermetic-build.md): every tool,
  library and header is an external archive pinned by sha256 or built in the graph.
- Fetching an external archive runs no host tool: patches are files applied by Bazel's own
  patcher (`patches`), never `patch_cmds` or `patch_tool`, and no repository rule of ours
  executes a program.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def calls(build_text, rule):
    """(name, body) of every `rule(...)` call in a BUILD file (top-level, balanced parens)."""
    out = []
    for m in re.finditer(rf"^{rule}\(", build_text, re.M):
        depth, i = 0, m.end() - 1
        while True:
            depth += {"(": 1, ")": -1}.get(build_text[i], 0)
            if depth == 0: break
            i += 1
        body = build_text[m.end():i]
        name = re.search(r'name = "([^"]+)"', body)
        out.append((name.group(1) if name else "", body))
    return out


def strings(body, attr):
    m = re.search(rf"\b{attr} = \[(.*?)\]", body, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


class BuildLists(unittest.TestCase):
    def test_every_test_zig_file_is_built(self):
        build = (ROOT / "tests/BUILD.bazel").read_text()
        used = set()
        for rule in ("zerv_test", "zig_test", "zig_library"):
            for name, body in calls(build, rule):
                main = re.search(r'main = "([^"]+)"', body)
                used.add(main.group(1) if main else f"{name}.zig")
                used.update(strings(body, "srcs"))
                if rule == "zerv_test" and re.search(r"sha = True", body):
                    used.add("fast_sha256.zig")
        for path in sorted((ROOT / "tests").glob("*.zig")):
            with self.subTest(file=path.name):
                self.assertIn(path.name, used, f"tests/{path.name} is not built by any target in tests/BUILD.bazel")

    def test_packages_import_only_declared_packages(self):
        dirs = sorted(p.parent.name for p in (ROOT / "src").glob("*/root.zig"))
        self.assertTrue(dirs)
        for name in dirs:
            build = ROOT / "src" / name / "BUILD.bazel"
            with self.subTest(package=name):
                self.assertTrue(build.is_file(), f"src/{name} has no BUILD.bazel")
                libs = {n: b for n, b in calls(build.read_text(), "zig_library")}
                self.assertIn(name, libs, f"src/{name}/BUILD.bazel has no zig_library named {name}")
                deps = {d.removeprefix("//src/") for d in strings(libs[name], "deps")}
                for path in (ROOT / "src" / name).rglob("*.zig"):
                    text = path.read_text()
                    self.assertNotIn('@import("../', text, f"{path.relative_to(ROOT)}: cross-package relative import")
                    self.assertNotIn('@import("zerv")', text, f"{path.relative_to(ROOT)}: a package cannot import the umbrella")
                    used = set(re.findall(r'@import\("([a-z_]+)"\)', text)) & set(dirs)
                    self.assertLessEqual(used, deps, f"{path.relative_to(ROOT)} imports undeclared packages {sorted(used - deps)}")

    def test_scripts_refuse_the_host_python(self):
        scripts = sorted([*(ROOT / "bench").rglob("*.py"), *(ROOT / "tools").glob("*.py")])
        self.assertGreater(len(scripts), 50)
        for path in scripts:
            text = path.read_text()
            if '__name__ == "__main__"' not in text: continue
            with self.subTest(script=str(path.relative_to(ROOT))):
                self.assertIn('if "/bazel-out/" not in sys.executable:', text, "no host-Python guard (tools/py)")

    def test_build_definitions_name_no_host_path(self):
        files = [ROOT / name for name in ("MODULE.bazel", ".bazelrc")]
        files += sorted((ROOT / "bazel").rglob("*.bzl")) + sorted((ROOT / "bazel").rglob("*.BUILD"))
        files += sorted(p for p in ROOT.rglob("BUILD.bazel") if not p.relative_to(ROOT).parts[0].startswith("bazel-"))
        self.assertGreater(len(files), 25)
        for path in files:
            with self.subTest(file=str(path.relative_to(ROOT))):
                for line in path.read_text().splitlines():
                    code = line.split("#", 1)[0].strip()
                    if code.startswith("common --disk_cache="): continue  # where results are cached, not an input
                    self.assertNotRegex(code, r'"/(usr|opt|home|etc|lib|lib64|bin|sbin)\b|~/', f"host path: {line.strip()}")

    def test_fetching_runs_no_host_tool(self):
        module = (ROOT / "MODULE.bazel").read_text()
        self.assertIn("patches = [", module)  # the check below looks at the right file
        for attr in ("patch_cmds", "patch_tool"):
            self.assertNotRegex(module, rf"^\s*{attr}\s*=", f"{attr} runs host tools at fetch time; use patches")
        for path in sorted((ROOT / "bazel").rglob("*.bzl")):
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertNotRegex(path.read_text(), r"\.execute\(", "a repository rule executes a host program")

    def test_harnesses_build_with_bazel(self):
        scripts = sorted([*(ROOT / "bench").glob("*.py"), *(ROOT / "tools").glob("*.py")])
        self.assertGreater(len(scripts), 20)
        for path in scripts:
            if path.name.startswith("rebuild_") and path.parent.name == "bench": continue
            with self.subTest(script=str(path.relative_to(ROOT))):
                text = path.read_text()
                for marker in ("zig-out/", ".tools/zig", '"build.zig"', '"-Doptimize', '"build", "test"'):
                    self.assertFalse(marker in text, f"builds outside Bazel ({marker}); use tools/zerv_build.py")


if __name__ == "__main__":
    unittest.main()
