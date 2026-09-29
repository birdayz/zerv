import argparse
import json
from pathlib import Path
import statistics as s
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / '2026-09-28-preparation-serving')
parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'serving-validated.json')
parser.add_argument('--clean', action='store_true')
args = parser.parse_args()
root = args.input
summary = json.loads((root / 'summary.json').read_text())
rows = [json.loads(x) for x in (root / 'raw.jsonl').read_text().splitlines()]
base = next(k for k in summary if k.startswith('zerv') and 'prepare-pages' not in k)
def key(r): return r['conversation'], r['turn']
def signature(r): return r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens']
expected = {key(r): signature(r) for r in rows if r['engine'] == base and r['round'] == 0}
assert len(expected) == 8
native = [r for r in rows if r['engine'].startswith('zerv')]
assert len(native) == 24 * sum(k.startswith('zerv') for k in summary)
assert all('error' not in r and signature(r) == expected[key(r)] for r in native)
out = dict(native_exact_responses=len(native), exploratory=not args.clean, engines={})
for name, trials in summary.items():
    assert len(trials) == 3 and all(not r['errors'] for r in trials)
    metrics = {}
    for metric in ['wall_s', 'aggregate_tok_s']:
        vals = [t[metric] for t in trials]
        metrics[metric] = dict(mean=s.mean(vals), sd=s.stdev(vals), trials=vals)
    for metric, vals in [('reuse_ttft_ms', [t['turns'][1]['ttft_p50_ms'] for t in trials]), ('gap_p99_ms', [t['stream_gap_ms']['p99'] for t in trials])]:
        metrics[metric] = dict(mean=s.mean(vals), sd=s.stdev(vals), trials=vals)
    metrics['matching_responses_and_counts'] = sum(signature(r) == expected[key(r)] for r in rows if r['engine'] == name)
    out['engines'][name] = metrics
    print(name, json.dumps(metrics))
args.output.write_text(json.dumps(out, indent=2) + '\n')
