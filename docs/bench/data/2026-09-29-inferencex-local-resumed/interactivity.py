"""Read-only postprocessing of completed runs; no inference or downloads.
Run: tools/py docs/bench/data/2026-09-29-inferencex-local-resumed/interactivity.py
"""
from collections import defaultdict
import json
from pathlib import Path
import statistics


def p90(values):
    xs = sorted(values)
    i = .9 * (len(xs) - 1)
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


assert abs(p90([1, 2, 3, 4]) - 3.7) < 1e-12
root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['status'] == 'passed' and len(manifest['cases']) == 36
groups = defaultdict(list)
for case in manifest['cases']:
    directory = root/case['path']
    if not directory.exists():
        directory = root.parent/directory.parent.name/directory.name
    upstream = json.loads((directory/'upstream.json').read_text())
    observed = json.loads((directory/'observer.json').read_text())[2*case['concurrency']:]
    assert len(observed) == 8 and upstream['output_lens'] == [256]*8
    tpots = [(r['end'] - r['text_times'][0]) / (r['usage']['completion_tokens'] - 1) for r in observed]
    assert all(t > 0 for t in tpots)
    groups[(case['input'], case['concurrency'], case['engine'])].append(dict(
        round=case['round'], raw_upstream=1000/upstream['p90_tpot_ms'],
        first_text_adjusted=1/p90(tpots)))
print('| Input | C | Engine | Raw 1000/p90_TPOT | First-text-adjusted tok/s/user |')
print('|---|---|---|---|---|')
for (length, c, engine), trials in sorted(groups.items()):
    assert sorted(t['round'] for t in trials) == [0, 1, 2]
    values = []
    for key in ('raw_upstream', 'first_text_adjusted'):
        xs = [t[key] for t in trials]
        values.append(f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}')
    print(f'| {length} | {c} | {engine} | ' + ' | '.join(values) + ' |')
