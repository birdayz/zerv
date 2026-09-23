import hashlib
from pathlib import Path
import tempfile
import unittest

from bench.rebuild_tokenizer_baseline import verified_sources


class BaselineRebuildTests(unittest.TestCase):
    def test_snapshot_integrity_and_required_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            names = ("build.zig", ".zig-version", "bench/tokenizer.zig")
            sources = {}
            for name in names:
                path = run / "source" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
                sources[name] = hashlib.sha256(b"fixture").hexdigest()
            manifest = {"sources": sources}
            self.assertEqual(len(verified_sources(run, manifest)), 3)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verified_sources(run, {"sources": {"build.zig": sources["build.zig"]}})
            (run / "source/build.zig").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                verified_sources(run, manifest)

    def test_path_escape_rejected(self):
        for name in ("../outside", "/outside"):
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verified_sources(Path("unused"), {"sources": {name: "ignored"}})


if __name__ == "__main__":
    unittest.main()
