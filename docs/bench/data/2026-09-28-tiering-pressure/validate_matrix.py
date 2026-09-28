"""Validate complete per-engine results without hiding the original failed RDNA3 runs.
Only the two explicitly named failed competitors may be incomplete. Never edits inputs.
Run with tools/py; outputs pressure-serving/validated-summary.json.
"""
import json
from pathlib import Path
import sys

if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'bench'))
from summarize_archive import stats
base = ROOT/'docs/bench/data/2026-09-28-pressure-serving'
m = json.loads((base/'manifest.json').read_text())
rows = [json.loads(line) for line in (base/'raw.jsonl').read_text().splitlines()]
w = json.loads((ROOT/next(iter(m['workload']))).read_text())
expected = {(r, level, w['conversations'][i]['name'], t) for r in range(m['rounds'])
            for level in m['levels'] for i in range(level) for t in range(len(w['conversations'][i]['turns']))}
key = lambda r: (r['round'], r['level'], r['conversation'], r['turn'])
failed = {'rdna3', 'rdna3-b512'}
assert m['status'] == 'failed' and failed <= m['engines'].keys()
assert {r['engine'] for r in rows} == m['engines'].keys()
summary = json.loads((base/'summary.json').read_text())
out = dict(run_status='failed', native_identical=0, failed_engines={}, serving={}, component={})
reference = {}
for line in (ROOT/'docs/bench/data/2026-09-28-async-archive-serving/raw.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['engine'].startswith('zerv'):
        k = r['level'], r['conversation'], r['turn']
        v = r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']
        assert reference.setdefault(k, v) == v
for engine in m['engines']:
    selected = [r for r in rows if r['engine'] == engine]
    keys = {key(r) for r in selected}
    assert len(keys) == len(selected) and keys <= expected
    trials = summary[engine]
    assert len(trials) == m['rounds'] * len(m['levels'])
    if engine in failed:
        assert all(r.get('error') for r in selected)
        assert all(t['errors'] for t in trials)
        logs = [(base/resource['log']).read_text() for resource in m['engines'][engine]['resources']]
        assert len(logs) == 3 and all('GGML_ASSERT(ids || dst->ne[1] == 1) failed' in log for log in logs)
        out['failed_engines'][engine] = dict(attempted=len(selected), failed=len(selected), unattempted=len(expected-keys),
                                          errors=sorted({r['error'] for r in selected}), performance=None)
        continue
    assert keys == expected and all(not r.get('error') and r.get('usage') and r.get('output_sha256') for r in selected)
    assert all(r['usage']['completion_tokens'] > 0 for r in selected)
    assert all(not t['errors'] for t in trials)
    if engine.startswith('zerv'):
        for r in selected:
            k = r['level'], r['conversation'], r['turn']
            assert (r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']) == reference[k], (engine, k, r['round'])
            out['native_identical'] += 1
    item = {k: stats([t[k] for t in trials]) for k in ['wall_s', 'aggregate_tok_s', 'ttft_p50_ms', 'ttft_p95_ms']}
    item['gap_p99_ms'] = stats([t['stream_gap_ms']['p99'] for t in trials])
    for turn in [0, 1]:
        item[f'turn_{turn}_ttft_p50_ms'] = stats([t['turns'][turn]['ttft_p50_ms'] for t in trials])
    item['resources'] = m['engines'][engine]['resources']
    item['generated_tokens'] = [t['completion_tokens'] for t in trials]
    out['serving'][engine] = item
assert out['native_identical'] == 96
cm = json.loads((ROOT/'docs/bench/data/2026-09-28-pressure-model/manifest.json').read_text())
assert cm['status'] == 'passed' and all(r['exact_state'] and r['exact_vocab_rows'] == 4 for r in cm['results'])
out['component']['long'] = [r for r in cm['results'] if r['prefix'] == 80000]
(base/'validated-summary.json').write_text(json.dumps(out, indent=2)+'\n')
print('Overall run remains failed. Six complete configurations; 96 native responses/token counts exact.')
print(json.dumps(out['failed_engines'], indent=2))
