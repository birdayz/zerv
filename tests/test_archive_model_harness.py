"""D.0a model gate must distinguish progress from its explicit deferred counterfactual."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import run_archive_model as runner


class CadenceTests(unittest.TestCase):
    def test_matched_alternating_pairs(self):
        self.assertEqual(runner.planned_cases([257, 257, 80000], "both"),
                         [(257, "unit"), (257, "chunk"), (257, "chunk"),
                          (257, "unit"), (80000, "unit"), (80000, "chunk")])
        self.assertEqual(runner.planned_cases([257, 80000], "unit"), [(257, "unit"), (80000, "unit")])
        self.assertEqual(runner.planned_cases([257], "chunk"), [(257, "chunk")])

    def test_counterfactual_cannot_pass_positive_progress_gate(self):
        row = dict(exact_packed_rows=2, prefill_source_quanta=0)
        runner.validate_prefill(row, "chunk")
        with self.assertRaises(RuntimeError):
            runner.validate_prefill(row, "unit")
        row["prefill_source_quanta"] = 10
        runner.validate_prefill(row, "unit")
        with self.assertRaises(RuntimeError):
            runner.validate_prefill(row, "chunk")
        row["exact_packed_rows"] = 1
        with self.assertRaises(RuntimeError):
            runner.validate_prefill(row, "unit")
        with self.assertRaises(KeyError):
            runner.validate_prefill({}, "unit")

    def test_cadence_needs_prefill_and_prefill_needs_disk(self):
        for flags in [["--prefill-cadence", "both"], ["--prefill-cadence", "chunk"], ["--prefill"]]:
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["run_archive_model", "--output", "unused-output", *flags]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
