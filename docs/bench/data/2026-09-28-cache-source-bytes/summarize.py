"""Validate the C.2 gate manifests and summarize component trials; not serving."""
import json
from pathlib import Path
import statistics
import sys

if '/bazel-out/' not in sys.executable:
    sys.exit('run with tools/py')
root = Path(__file__).resolve().parent.parent
result = {}
for label, directory in [('source', '2026-09-28-cache-source-model'),
                         ('paused', '2026-09-28-cache-source-paused')]:
    manifest = json.loads((root / directory / 'manifest.json').read_text())
    assert manifest['status'] == 'passed'
    rows = manifest['results']
    for row in rows:
        assert row['exact_state'] and row['exact_vocab_rows'] == 4
        assert row['independent_rows'] == 2 and row['disk']
        assert row['source'] == (label == 'source')
        if label == 'source':
            assert row['source_cpu_quanta'] > 0 and row['source_gpu_quanta'] > 0
    result[label] = {'rows': rows}
    if label == 'source':
        short = [r for r in rows if r['prefix'] == 257]
        assert len(short) == 6
        result[label]['short_trials_after_first_warmup'] = {
            key: {'mean_ms': statistics.mean(r[key] / 1e6 for r in short[1:]),
                  'sample_sd_ms': statistics.stdev(r[key] / 1e6 for r in short[1:])}
            for key in ['capture_ns', 'disk_write_ns', 'restore_ns', 'max_poll_ns']}
    long = [r for r in rows if r['prefix'] == 80000]
    assert len(long) == 1 and long[0]['state_bytes'] == 5399773184
result['limitations'] = [
    'Component adapter tests, not HTTP serving.',
    'Long-prefix values are single samples, no variance claim.',
    'Source and paused paths include different interleaving/checkpoint work.',
    'device_starts includes CPU-only source acknowledgments, not just GPU submissions.',
]
out = Path(__file__).with_name('summary.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
