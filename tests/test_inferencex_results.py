"""Integration tests for the unchanged upstream processor, not a local metric implementation."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
BINARY = ROOT/'tools/inferencex_results'


class InferenceXResultsTests(unittest.TestCase):
    def test_executed_sources_match_pinned_upstream_hashes(self):
        rule = (ROOT/'bazel/inferencex.bzl').read_text()
        hashes = ast.literal_eval(re.search(r'_RESULTS = (\{.*?\})', rule, re.S).group(1))
        package = Path(importlib.util.find_spec('infx.results').origin).parent
        self.assertEqual(len(hashes), 12)
        for filename, expected in hashes.items():
            self.assertEqual(hashlib.sha256((package/filename).read_bytes()).hexdigest(), expected, filename)

    def invoke(self, args):
        return subprocess.run([str(BINARY), *map(str, args)],
                              env=dict(PATH=os.environ['PATH'], LANG='C.UTF-8'),
                              capture_output=True, text=True, timeout=30)

    def test_full_processor_and_collector_preserve_artifacts_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = dict(model_id='fixture', max_concurrency=1, total_token_throughput=50,
                       output_throughput=10, num_prompts=1, completed=1,
                       total_output_tokens=10, total_input_tokens=40,
                       duration=1, benchmark_start_time_unix=100, benchmark_end_time_unix=101,
                       median_tpot_ms=20, p90_tpot_ms=25, std_tpot_ms=3)
            input_path = root/'upstream.json'
            input_path.write_text(json.dumps(raw))
            original = input_path.read_bytes()
            meta = dict(RUNNER_TYPE='fixture', FRAMEWORK='fixture', PRECISION='fixture',
                        SPEC_DECODING='none', RESULT_FILENAME='upstream', ISL='40', OSL='10',
                        DISAGG='false', MODEL_PREFIX='fixture', IMAGE='fixture', TP='1',
                        EP_SIZE='1', DP_ATTENTION='false', REQUIRE_POWER='false',
                        GPU_METRICS_CSV=str(root/'missing.csv'), RECIPE_FINGERPRINT='test-config-identity')
            config = root/'metadata.json'
            config.write_text(json.dumps(meta))
            result = self.invoke(['process', root, config])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(input_path.read_bytes(), original)
            emitted = root/'agg_upstream.json'
            output = json.loads(emitted.read_text())
            self.assertEqual(output['recipe_fingerprint'], 'test-config-identity')
            self.assertEqual(output['median_intvty'], 50)
            self.assertEqual(output['p90_intvty'], 40)
            # Even upstream's questionable std conversion must not be corrected.
            self.assertEqual(output['std_intvty'], 1000/3)
            self.assertIs(type(output['power_valid']), int)
            self.assertEqual(output['power_valid'], 0)
            self.assertIn('telemetry_file_missing', output['power_invalid_reasons'])
            self.assertTrue((root/'power_validation_upstream.json').is_file())
            collection = root/'collection'
            collection.mkdir()
            (collection/'one.json').write_bytes(emitted.read_bytes())
            result = self.invoke(['collect', root, collection, 'fixture'])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root/'agg_fixture.json').read_text()), [output])
            # Let the actual upstream CLI enforce mandatory metadata and power.
            for bad in (dict(meta, REQUIRE_POWER='true'), {k: v for k, v in meta.items() if k != 'FRAMEWORK'}):
                config.write_text(json.dumps(bad))
                result = self.invoke(['process', root, config])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(input_path.read_bytes(), original)
            config.write_text(json.dumps(dict(meta, RESULT_FILENAME='../outside')))
            result = self.invoke(['process', root, config])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('basename', result.stderr)


if __name__ == '__main__':
    unittest.main()
