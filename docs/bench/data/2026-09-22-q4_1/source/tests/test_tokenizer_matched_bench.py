import copy
import hashlib
import json
import struct
import unittest
from unittest.mock import Mock

from bench import run_tokenizer_matched as bench


class MatchedTokenizerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(text="x", ids=[87], raw_hex="78")
        self.corpus = dict(cases=[self.case], decode_cases=[self.case], workloads=[dict(self.case, name="simple")])
        self.hf = Mock()
        self.hf.normalizer.normalize_str.side_effect = lambda s: s
        self.hf.encode.return_value.ids = [87]

    def test_binary_schema_and_independent_normalization(self):
        data, changed = bench.corpus_bytes(self.corpus, self.hf)
        self.assertEqual(data[:20], b"ZTBC" + struct.pack("<IIII", 1, 1, 1, 1))
        self.assertEqual(changed, 0)
        self.assertIn(b"simple", data)
        self.corpus["cases"] = [dict(self.case, text="y")]
        self.hf.normalizer.normalize_str.side_effect = lambda s: "x"
        self.assertEqual(bench.corpus_bytes(self.corpus, self.hf)[1], 1)

    def test_bad_corpus_and_non_equivalent_timed_input(self):
        for mutate in (
            lambda c: c.update(cases=[]),
            lambda c: c["workloads"].append(c["workloads"][0]),
            lambda c: c["workloads"][0].update(name='bad"name'),
            lambda c: c["workloads"][0].update(ids=[-1]),
            lambda c: c["workloads"][0].update(text="x" * 1048577),
        ):
            corpus = copy.deepcopy(self.corpus)
            mutate(corpus)
            with self.assertRaises(ValueError):
                bench.corpus_bytes(corpus, self.hf)
        self.hf.normalizer.normalize_str.side_effect = lambda s: "changed"
        with self.assertRaisesRegex(ValueError, "already be NFC"):
            bench.corpus_bytes(self.corpus, self.hf)
        self.hf.encode.return_value.ids = [99]
        with self.assertRaisesRegex(ValueError, "official IDs"):
            bench.corpus_bytes(self.corpus, self.hf)

    def test_hash_actual_reference_results_and_apply_strict_trial_gate(self):
        checks = bench.common.expected(self.corpus)
        rows = [dict(workload=name, operation=op, trial=i, iterations=bench.common.ITERATIONS[op],
                     elapsed_ns=100, output_hex="57000000" if op == "encode" else "78")
                for name, op in checks for i in range(7)]
        raw = "\n".join(map(json.dumps, rows))
        parsed = bench.reference_rows(raw)
        bench.common.validate(parsed, checks)
        self.assertEqual(parsed[0]["output_sha256"], hashlib.sha256(struct.pack("<I", 87)).hexdigest())
        parsed[0]["output_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            bench.common.validate(parsed, checks)
        for bad in ({"output_hex": 12}, {"output_hex": "xx"}, {"output_hex": "78", "output_sha256": "spoof"}):
            with self.assertRaises(ValueError):
                bench.reference_rows(json.dumps(bad))


if __name__ == "__main__":
    unittest.main()
