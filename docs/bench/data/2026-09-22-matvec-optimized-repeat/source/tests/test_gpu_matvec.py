import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests/reference"))
sys.path.insert(0, str(ROOT/"bench"))
from generate_gpu_matvec import TYPES, canonical, encode_case, metrics
from run_gpu_matvec import timings


def sha(data): return hashlib.sha256(data).hexdigest()


class MatvecFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT/"tests/fixtures/gpu/matvec.json").read_text())

    def test_independent_source_provenance_and_coverage(self):
        self.assertEqual(self.fixture["schema_version"], 1)
        for name, digest in self.fixture["sources"].items():
            self.assertEqual(sha((ROOT/"tests/reference"/name).read_bytes()), digest)
        cases = self.fixture["cases"]
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({c["name"] for c in cases}), 48)
        self.assertEqual(sum(c["kind"] == "half" for c in cases), 6)
        self.assertEqual(sum("tensor" in c for c in cases), 11)
        self.assertEqual({c["format"] for c in cases}, set(TYPES))
        for fmt in TYPES:
            for suffix in ("isolated-columns", "cancellation", "zero-input"):
                self.assertTrue(next(c for c in cases if c["name"] == fmt+"-"+suffix)["exact"])

    def test_complete_input_hashes_and_independent_gpu_outputs(self):
        for c in self.fixture["cases"]:
            if c["kind"] == "explicit":
                packed, x = bytes.fromhex(c["packed_hex"]), bytes.fromhex(c["input_hex"])
                actual = struct.unpack("<"+"f"*c["rows"], bytes.fromhex(c["reference_hex"]))
                self.assertEqual(metrics(actual, c["ideal"], c["sumabs"], c["exact"]), c["reference_metrics"])
            elif c["kind"] == "half":
                base = bytes.fromhex(c["base_hex"]); field = c["field"]
                packed = bytearray(); expected = []
                for bits in range(65536):
                    if bits & 0x7c00 == 0x7c00: continue
                    packed += base[:field]+struct.pack("<H", bits)+base[field+2:]
                    value, = struct.unpack("<e", struct.pack("<H", bits))
                    expected.append(-value if c["format"] == "q5_k" and field == 2 else value)
                x = struct.pack("<f", 1)+bytes((c["columns"]-1)*4)
                self.assertEqual(len(expected), c["rows"])
                self.assertEqual(sha(canonical(expected)), c["output_sha256"])
                self.assertEqual(c["reference_metrics"]["max_abs"], 0)
            else:
                values = [(r % 257-128)/128 for r in range(65537)]
                packed = canonical(values); x = struct.pack("<f", 1/16)
                self.assertEqual(sha(canonical([v/16 for v in values])), c["output_sha256"])
            self.assertEqual(sha(encode_case(c, packed, x)), c["input_sha256"], c["name"])

    def test_owned_shader_manifest_and_capability_boundary(self):
        directory = ROOT/"src/matvec/shaders"
        manifest = json.loads((directory/"manifest.json").read_text())
        self.assertEqual(manifest["source_sha256"], sha((ROOT/"src/matvec/matvec.comp").read_bytes()))
        self.assertEqual(set(manifest["modules"]), set(TYPES) | {"f32_small"})
        for name, record in manifest["modules"].items():
            raw = (directory/(name+".spv")).read_bytes()
            self.assertEqual(sha(raw), record["sha256"])
            self.assertEqual(len(raw), record["bytes"])
            words = struct.unpack("<"+"I"*(len(raw)//4), raw)
            self.assertEqual(words[:2], (0x07230203, 0x00010300))
            pos = 5; capabilities = []; local_size = []; no_contraction = 0
            while pos < len(words):
                count, opcode = words[pos] >> 16, words[pos] & 65535
                self.assertGreater(count, 0)
                self.assertLessEqual(pos+count, len(words))
                if opcode == 17: capabilities.append(words[pos+1])
                if opcode == 16 and words[pos+2] == 17: local_size.append(words[pos+3:pos+6])
                if opcode == 71 and words[pos+2] == 42: no_contraction += 1
                pos += count
            self.assertEqual(capabilities, [1])  # Shader only, no optional Float16/Int8/subgroups.
            lanes = 256 if name == "f32" else 64
            self.assertEqual(record["lanes"], lanes)
            self.assertEqual(record["rows_per_group"], 1)
            self.assertEqual(local_size, [(lanes, 1, 1)])
            self.assertGreater(no_contraction, 0)

    def test_metrics_reject_bad_outputs_and_separate_default_precision(self):
        self.assertEqual(metrics([1.0], [1.0], [1.0], exact=True)["max_abs"], 0)
        for actual in ([math.nan], [math.inf], [], [1, 2], [1.01]):
            with self.assertRaises(ValueError): metrics(actual, [1], [1])
        with self.assertRaises(ValueError): metrics([1.0000001], [1], [1], exact=True)
        self.assertGreater(metrics([1.01], [1], [1], enforce=False)["max_abs"], 0.009)

    def test_timing_gate_rejects_missing_duplicate_invalid_trials(self):
        records = [dict(trial=i, iterations=10, elapsed_ns=10000+i) for i in range(7)]
        text = lambda rs: "\n".join(json.dumps(r) for r in rs)
        self.assertEqual(len(timings(text(records), 10)), 7)
        for altered in (records[:-1], records+[records[-1]], [records[0]]*7,
                        [dict(r, elapsed_ns=0) for r in records], [dict(r, iterations=True) for r in records],
                        [dict(r, extra=1) for r in records]):
            with self.assertRaises(ValueError): timings(text(altered), 10)


if __name__ == "__main__": unittest.main()
