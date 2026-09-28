"""Reproduce C.3 report cells, including cross-revision native HTTP identity.
Run: tools/py docs/bench/data/2026-09-28-tiering-pressure/report.py SERVING_DIR
First run bench/summarize_archive.py for SERVING_DIR/validated-summary.json.
"""
import json
from pathlib import Path
import re
import statistics
import sys

if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
ROOT = Path(__file__).resolve().parents[4]
serving = Path(sys.argv[1])
require_restore = '--require-disk-restore' in sys.argv[2:]
restores = 0
summary = json.loads((serving/'validated-summary.json').read_text())
manifest = json.loads((serving/'manifest.json').read_text())
failed_engines = summary.get('failed_engines', {})
assert manifest['status'] == 'passed' or (manifest['status'] == 'failed' and set(failed_engines) == {'rdna3', 'rdna3-b512'})
if failed_engines:
    print('OVERALL RUN FAILED; incomplete engines have no performance score:', json.dumps(failed_engines))
want = {}
for line in (ROOT/'docs/bench/data/2026-09-28-async-archive-serving/raw.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['engine'].startswith('zerv'):
        key = r['level'], r['conversation'], r['turn']
        identity = r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']
        assert want.setdefault(key, identity) == identity
counts = {}
for line in (serving/'raw.jsonl').read_text().splitlines():
    r = json.loads(line)
    if r['engine'] in failed_engines:
        assert r.get('error')
        continue
    assert not r.get('error') and r['usage']['completion_tokens'] > 0
    key = r['level'], r['conversation'], r['turn']
    identity = r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']
    equal = identity == want[key]
    if r['engine'].startswith('zerv'):
        assert equal, (r['engine'], key, r['round'])
    n = counts.setdefault(r['engine'], [0, 0])
    n[0] += int(equal)
    n[1] += 1
print('identity (text hash + prompt/generated token counts) vs native baseline:', json.dumps(counts))
for name, value in summary['serving'].items():
    fields = []
    for key, scale in [('turn_0_ttft_p50_ms', 1000), ('turn_1_ttft_p50_ms', 1000), ('wall_s', 1), ('aggregate_tok_s', 1), ('gap_p99_ms', 1)]:
        s = value[key]
        fields.append(f"{s['mean']/scale:.3f} ± {(s['stdev'] or 0)/scale:.3f}")
    print(name, ' | '.join(fields), 'peak VRAM', max(r['vram_peak'] for r in value['resources']))
    print(' resources', json.dumps([dict(vram_peak=r['vram_peak'], host_memory=r['host_memory'], host_memory_details=r.get('host_memory_details')) for r in value['resources']]))
    if 'prefix-cache-disk-dir' in name:
        for resource in value['resources']:
            log = (serving/resource['log']).read_text()
            for label in ['disk prefix archive', 'disk source retention', 'prefix checkpoints']:
                matches = re.findall(r'zerv: '+label+r': (.*)', log)
                if label == 'prefix checkpoints':
                    matches = [line for line in matches if ' taken,' in line]
                assert len(matches) == 1, (resource['log'], label)
                print(' round', resource['round'], label, matches[0])
                if label == 'disk prefix archive':
                    restores += int(re.match(r'\d+ writes, (\d+) restores,', matches[0]).group(1))
if require_restore:
    assert restores > 0, 'positive HTTP disk-restore gate FAILED: no disk restores'
    print('positive HTTP disk-restore gate PASSED:', restores)
cm = json.loads((ROOT/'docs/bench/data/2026-09-28-pressure-model/manifest.json').read_text())
assert cm['status'] == 'passed'
short = [r for r in cm['results'] if r['prefix'] == 257][1:]
assert len(short) == 5
for key in ['capture_ns', 'disk_write_ns', 'restore_ns']:
    values = [r[key]/1e6 for r in short]
    print(key, 'ms', statistics.mean(values), 'sample_sd', statistics.stdev(values))
print('long component', summary['component']['long'])
