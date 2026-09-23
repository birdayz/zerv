import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tokenizer_bench", ROOT / "bench/run_tokenizer.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TokenizerBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads(bench.FIXTURE.read_text())
        self.expected = bench.expected(self.corpus)
        self.rows = [dict(workload=name, operation=op, trial=i, iterations=bench.ITERATIONS[op], elapsed_ns=1000, output_sha256=digest)
                     for (name, op), digest in self.expected.items() for i in range(7)]

    def test_valid_component_and_http(self):
        bench.validate(self.rows, self.expected)
        rows = [dict(row, iterations=20) for row in self.rows if row["operation"] == "encode"]
        bench.validate(rows, bench.expected(self.corpus, True), True)

    def test_incomplete_or_duplicate(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], [self.rows[0]] * len(self.rows)):
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)

    def test_bad_fields(self):
        for key, value in [("trial", 7), ("trial", True), ("iterations", 0), ("elapsed_ns", -1),
                           ("operation", "unknown"), ("workload", []), ("output_sha256", "bad"), ("extra", 2)]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)

    def test_loopback_only(self):
        for url in ("https://127.0.0.1", "http://example.com", "http://user@127.0.0.1", "http://localhost/extra", "http://localhost?x=1"):
            with self.assertRaises(ValueError):
                bench.connection(url)
        bench.connection("http://127.0.0.1:18081").close()

    def test_server_mismatch_requires_normalized_id_equality(self):
        corpus = {"cases": [{"text": "e\u0301", "ids": [933]}]}
        hf = Mock()
        hf.normalizer.normalize_str.return_value = "é"
        for normalized_ids, unexpected in [([933], 0), ([99], 1)]:
            responses = [[{"id": 68, "piece": "e"}, {"id": 52033, "piece": "\u0301"}], normalized_ids]
            with patch.object(bench, "connection", return_value=Mock()), patch.object(bench, "request", side_effect=responses):
                result = bench.compare_server("http://localhost", hf, corpus)
            self.assertEqual(result["unexpected"], unexpected)
            self.assertEqual(result["mismatches"][0]["matches_after_nfc"], unexpected == 0)

    def test_provenance(self):
        for path, key in [(ROOT / "tests/reference/generate_tokenizer_goldens.py", "generator_sha256"),
                          (ROOT / "tests/reference/tokenizer_pieces.c", "oracle_adapter_sha256"),
                          (bench.FIXTURE.parent / "model.bin", "model_data_sha256")]:
            self.assertEqual(bench.sha(path), self.corpus[key])


if __name__ == "__main__":
    unittest.main()
