"""tools/kv_quality.py metrics (block 17c gate 4): KL, top-1 agreement and the text slice."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import kv_quality  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_identical_rows_have_zero_kl(self):
        rows = np.random.default_rng(1).normal(size=(5, 50)).astype(np.float32)
        r = kv_quality.compare(rows, rows)
        self.assertEqual(r["kl_max"], 0.0)
        self.assertEqual(r["top1_agree"], 1.0)
        self.assertEqual(r["rows"], 5)

    def test_kl_matches_a_direct_computation(self):
        rng = np.random.default_rng(2)
        p, q = rng.normal(size=(3, 7)), rng.normal(size=(3, 7))
        pp = np.exp(p) / np.exp(p).sum(axis=1, keepdims=True)
        qq = np.exp(q) / np.exp(q).sum(axis=1, keepdims=True)
        want = (pp * np.log(pp / qq)).sum(axis=1)
        r = kv_quality.compare(p, q)
        self.assertAlmostEqual(r["kl_mean"], float(want.mean()), places=12)
        self.assertAlmostEqual(r["kl_max"], float(want.max()), places=12)
        self.assertEqual(r["top1_agree"], float((p.argmax(1) == q.argmax(1)).mean()))

    def test_log_softmax_is_stable_for_large_logits(self):
        lp = kv_quality.log_softmax(np.array([[1000.0, 999.0, -1000.0]]))
        self.assertTrue(np.isfinite(lp).all())
        self.assertAlmostEqual(float(np.exp(lp).sum()), 1.0, places=12)

    def test_document_text_excludes_needle_and_question(self):
        text = kv_quality.document_text()
        self.assertTrue(text.startswith("=== docs/"))
        self.assertNotIn("ORCHID-7431", text)
        self.assertNotIn("Summarize the performance results", text)


if __name__ == "__main__":
    unittest.main()
