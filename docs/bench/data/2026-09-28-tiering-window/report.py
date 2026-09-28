"""Validate D.0b complete serving/model runs and print report cells (tools/py)."""
import json
from pathlib import Path
import re
import sys
if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'bench'))
from summarize_archive import summarize, stats
base = ROOT/'docs/bench/data/2026-09-28-window-serving'
out = summarize(base, ROOT/'docs/bench/data/2026-09-28-window-model-8')
assert out['native_identical'] == 120
reference = {}
for line in (ROOT/'docs/bench/data/2026-09-28-async-archive-serving/raw.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['engine'].startswith('zerv'):
        key = r['level'], r['conversation'], r['turn']
        identity = r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']
        assert reference.setdefault(key, identity) == identity
counts = {}
for line in (base/'raw.jsonl').read_text().splitlines():
    r = json.loads(line)
    assert not r.get('error') and r['usage']['completion_tokens'] > 0
    equal = (r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']) == reference[r['level'], r['conversation'], r['turn']]
    if r['engine'].startswith('zerv'):
        assert equal, (r['engine'], r['round'], r['conversation'], r['turn'])
    counter = counts.setdefault(r['engine'], [0, 0])
    counter[0] += int(equal)
    counter[1] += 1
out['identity_vs_baseline'] = counts
print('Native text+token counts exact; reference identity:', json.dumps(counts))
for name, item in out['serving'].items():
    print(name)
    for k in ['wall_s', 'turn_1_ttft_p50_ms', 'aggregate_tok_s', 'gap_p99_ms']:
        print(k, json.dumps(item[k]))
    print('resources', json.dumps(item['resources']))
    if 'prefix-cache-disk-dir' in name:
        for resource in item['resources']:
            text = (base/resource['log']).read_text()
            for label in ['disk transfer window', 'disk prefix archive', 'disk source retention']:
                found = re.findall('zerv: '+label+': (.*)', text)
                assert len(found) == 1
                print('round', resource['round'], label, found[0])
out['windows'] = {}
for c in [1, 2, 4, 8]:
    model = json.loads((ROOT/f'docs/bench/data/2026-09-28-window-model-{c}/manifest.json').read_text())
    assert model['status'] == 'passed'
    rows = model['results']
    assert len(rows) == 7
    assert all(r['chunk_mib'] == c and r['exact_state'] and r['exact_vocab_rows'] == 4 and r['exact_packed_rows'] == 2 and r['prefill_source_quanta'] > 0 for r in rows)
    short = rows[1:6]
    cells = {k.removesuffix('_ns')+'_ms': stats([r[k]/1e6 for r in short]) for k in ['capture_ns', 'disk_write_ns', 'restore_ns', 'max_poll_ns']}
    cells['long'] = rows[6]
    out['windows'][str(c)] = cells
    print('component MiB', c, json.dumps(cells))
(base/'validated-summary.json').write_text(json.dumps(out, indent=2)+'\n')
