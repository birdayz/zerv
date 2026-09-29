#!/usr/bin/env python3
"""Validate recorded demand serving trials; never infer matched competitor quality."""
import hashlib
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
rows = [json.loads(line) for line in (root / 'raw.jsonl').read_text().splitlines()]
level = int(sys.argv[2]) if len(sys.argv) > 2 else 4
summary = {name: [trial for trial in trials if trial['level'] == level] for name, trials in summary.items()}
rows = [row for row in rows if row['level'] == level]
manifest = json.loads((root / 'manifest.json').read_text())
workload_path, workload_hash = next(iter(manifest['workload'].items()))
workload_bytes = pathlib.Path(workload_path).read_bytes()
assert hashlib.sha256(workload_bytes).hexdigest() == workload_hash
workload = json.loads(workload_bytes)
expected = sum(len(c['turns']) for c in workload['conversations'][:level]) * 3
native = [name for name in summary if name.startswith('zerv-')]
baseline = next((name for name in native if 'prefix-cache-demand=' not in name), native[0])
def key(row):
    return row['round'], row['level'], row['conversation'], row['turn']
def signature(row):
    return row['output_sha256'], row['usage']['prompt_tokens'], row['usage']['completion_tokens']
base = {key(row): signature(row) for row in rows if row['engine'] == baseline}
assert len(base) == expected
result = {}
for name, trials in summary.items():
    assert len(trials) == 3 and all(not trial['errors'] for trial in trials)
    records = [row for row in rows if row['engine'] == name]
    assert len(records) == expected and len({key(row) for row in records}) == expected
    matched = sum(signature(row) == base[key(row)] for row in records)
    if name in native:
        assert matched == expected, (name, matched)
    metrics = {}
    for metric, values in {
        'wall_s': [t['wall_s'] for t in trials],
        'tok_s': [t['aggregate_tok_s'] for t in trials],
        'reuse_ttft_p50_ms': [t['turns'][1]['ttft_p50_ms'] for t in trials],
        'gap_p99_ms': [t['stream_gap_ms']['p99'] for t in trials],
    }.items():
        metrics[metric] = dict(mean=statistics.mean(values), sd=statistics.stdev(values), trials=values)
    for turn in range(2, len(trials[0]['turns'])):
        values = [t['turns'][turn]['ttft_p50_ms'] for t in trials]
        metrics[f'turn_{turn + 1}_ttft_p50_ms'] = dict(mean=statistics.mean(values), sd=statistics.stdev(values), trials=values)
    logs = []
    for trial in trials:
        label = 'engine-' + hashlib.sha256(name.encode()).hexdigest()[:16] if '/' in name or '\\' in name or len(name.encode()) >= 200 else name
        text = (root / f'{label}-r{trial["round"]}.log').read_text()
        logs.append(dict(round=trial['round'], counters=[line for line in text.splitlines() if re.match(r'zerv: (staging prefetch:|disk prefix archive:|queued demand:)', line)]))
    result[name] = dict(metrics=metrics, exact_matches_to_native_off=matched, responses=len(records), logs=logs)
print(json.dumps(result, indent=2))
