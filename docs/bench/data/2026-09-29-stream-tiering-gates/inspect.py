"""Read-only gates and display of upstream-emitted metrics; tools/py this file."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

base = Path(__file__).resolve().parents[1]
raw = base / '2026-09-29-stream-tiering'
processed = base / '2026-09-29-stream-tiering-processed'
m = json.loads((raw / 'manifest.json').read_text())
p = json.loads((processed / 'manifest.json').read_text())
assert m['status'] == p['status'] == 'passed'
assert len(m['cases']) == len(p['cases']) == 24
expected = {(e, r, i, c) for e in m['engines'] for r in range(3) for i in (1024, 8192) for c in (1, 4)}
assert {(c['engine'], c['round'], c['input'], c['concurrency']) for c in m['cases']} == expected
rows = {}
signatures = {}
total = warmups = 0
for c in m['cases']:
    assert c['status'] == c['prompt_gate'] == 'passed'
    records = json.loads((raw / c['path'] / 'observer.json').read_text())
    n = 2 * c['concurrency']
    assert len(records) == n + 8
    for r in records:
        assert r['status'] == 200 and r['done'] and r['finishes'] == ['length']
        assert r['usage']['completion_tokens'] == 256
        assert r['role_s'] == r['text_times'][0]
    total += 8
    warmups += n
    signatures[c['path']] = Counter((r['request_sha256'], r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens'], tuple(r['finishes'])) for r in records[n:])
    assert 0 < c['vram_peak'] <= 24 * 1024**3
    assert 0 < int(c['host_memory']['values']['VmHWM'].split()[0]) <= 32 * 1024**2
for c in p['cases']:
    path = processed / c['name'] / 'agg_upstream.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c['result_sha256']
    assert hashlib.sha256((raw / c['name'] / 'upstream.json').read_bytes()).hexdigest() == c['input_sha256']
    rows[c['name']] = json.loads(path.read_text())
assert json.loads((processed / 'agg_inferencex-local.json').read_text()) == list(rows.values())
print('Passed matrix, raw/result hashes, collector equality, response/count/resource and role=first-text gates:', total, 'measured;', warmups, 'warmups')
for i in (1024, 8192):
    for c in (1, 4):
        names = [f'{e}-r{r}-i{i}-c{c}' for e in m['engines'] for r in range(3)]
        print('Output signatures', i, c, 'all six trials equal:', all(signatures[n] == signatures[names[0]] for n in names))
        for r in range(3):
            a, b = [signatures[f'{e}-r{r}-i{i}-c{c}'] for e in m['engines']]
            print('  paired round', r, 'matching', sum((a & b).values()), '/8')
for field in ('output_tput_per_gpu', 'median_intvty', 'p90_intvty', 'mean_ttft', 'mean_e2el'):
    print('\n', field, '(three emitted trials; arithmetic mean ± sample SD of those trial values)')
    for i in (1024, 8192):
        for c in (1, 4):
            cells=[]
            for e in m['engines']:
                vals=[rows[f'{e}-r{r}-i{i}-c{c}'][field] for r in range(3)]
                cells.append(' / '.join(f'{v:.3f}' for v in vals)+f' ({mean(vals):.3f} ± {stdev(vals):.3f})')
            print(i, c, ' | '.join(cells))
print('\nServer counters (whole lifetime, includes length gate/warmups):')
for c in m['cases']:
    print(c['path'], 'VRAM bytes', c['vram_peak'], 'host', c['host_memory']['values'])
    for line in (raw / c['path'] / 'server.log').read_text().splitlines():
        if any(s in line for s in ('disk prefix archive:', 'prefix checkpoints:', 'staging prefetch:')):
            print(line)
print('Dates', m['started_at'], m['finished_at'])
print('Model SHA', m['model_sha256'], 'binary SHA', m['native_sha256'])
print('\nMemory ranges GiB (sampled device-wide VRAM peak; server-process host high-water includes loading):')
for e in m['engines']:
    cases = [c for c in m['cases'] if c['engine'] == e]
    vram = [c['vram_peak'] / 1024**3 for c in cases]
    host = [int(c['host_memory']['values']['VmHWM'].split()[0]) / 1024**2 for c in cases]
    print(e, 'VRAM', min(vram), max(vram), 'host', min(host), max(host))
print('\nTiered throughput change relative to untiered (ratio of three-trial means):')
for i in (1024, 8192):
    for c in (1, 4):
        vals = [mean(rows[f'{e}-r{r}-i{i}-c{c}']['output_tput_per_gpu'] for r in range(3)) for e in m['engines']]
        print(i, c, 100 * (vals[0] / vals[1] - 1))
