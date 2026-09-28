import hashlib
import json
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).parents[1]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class VulkanFixturesTests(unittest.TestCase):
    def test_abi_provenance_and_complete_unique_records(self):
        data = json.loads((ROOT / "tests/fixtures/gpu/abi.json").read_text())
        self.assertEqual(data["generator_sha256"], sha("tests/reference/generate_vulkan_abi.py"))
        self.assertEqual(data["helper_sha256"], sha("tests/reference/vulkan_api.py"))
        self.assertEqual(len(data["structs"]), 71)
        self.assertEqual(len({s["name"] for s in data["structs"]}), 71)
        self.assertEqual(len(data["constants"]), 95)
        for s in data["structs"]:
            self.assertGreater(s["size"], 0)
            self.assertEqual(s["size"] % s["alignment"], 0)
            self.assertEqual(len(set(s["fields"].values())), len(s["fields"]))
            self.assertTrue(all(0 <= offset < s["size"] for offset in s["fields"].values()))

    def test_real_dispatch_fixture_provenance_and_guarded_cases(self):
        data = json.loads((ROOT / "tests/fixtures/gpu/dispatch.json").read_text())
        for key, path in [("generator_sha256", "tests/reference/generate_vulkan_goldens.py"),
                          ("reference_source_sha256", "tests/reference/vulkan_driver.c"),
                          ("shader_source_sha256", "tests/fixtures/gpu/affine.comp"),
                          ("shader_sha256", "tests/fixtures/gpu/affine.spv")]:
            self.assertEqual(data[key], sha(path))
        self.assertEqual({c["count"] for c in data["cases"] if c["kind"] == "affine"}, {1, 63, 64, 65, 5120, 65537, 1048576})
        self.assertEqual({c["count"] for c in data["cases"] if c["kind"] == "roundtrip"}, {256, 1048576, 67108864})
        for c in data["cases"]:
            self.assertEqual(c["bytes"], (c["count"]+64)*4 if c["kind"] == "affine" else c["count"])
            if c["kind"] == "roundtrip":
                self.assertEqual(c["input_sha256"], c["output_sha256"])
        header = struct.unpack_from("<5I", (ROOT / "tests/fixtures/gpu/affine.spv").read_bytes())
        self.assertEqual(header[0], 0x07230203)
        self.assertLessEqual(header[1], 0x10300)
        self.assertEqual(header[4], 0)


if __name__ == "__main__":
    unittest.main()
