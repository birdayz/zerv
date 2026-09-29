#!/usr/bin/env python3
"""Revalidate and summarize the predeclared local 36-point InferenceX matrix."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import sys

from inferencex_observer import validate

ENGINES = ('zerv-tiered', 'llama-fa-b512', 'rdna3-nofusion')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_sha(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def check_matrix(cases, engines=ENGINES):
    assert engines and len(engines) == len(set(engines)) and set(engines) <= set((*ENGINES, 'zerv-untiered')), 'invalid engine selection'
    expected = set(itertools.product(engines, range(3), (1024, 8192), (1, 4)))
    keys = [(c['engine'], c['round'], c['input'], c['concurrency']) for c in cases]
    assert len(keys) == len(set(keys)) and set(keys) == expected, 'incomplete or duplicate matrix'
    assert all(c['status'] == 'passed' and c['output'] == 256 for c in cases)


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), trials=values)


def percentile(values, q):
    xs = sorted(values)
    index = (len(xs) - 1) * q
    lo = int(index)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (index - lo)


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'passed'
    check_matrix(manifest['cases'])
    groups = defaultdict(list)
    artifacts = {}
    rows = []
    for c in manifest['cases']:
        directory = root / c['path']
        # Inherited paths are absolute in the original manifest; support moving the
        # complete sibling artifact directories together without rewriting raw data.
        if not directory.exists():
            directory = root.parent / directory.parent.name / directory.name
        def load(filename):
            path = directory / filename
            artifacts[str(path.relative_to(root.parent))] = sha(path)
            return json.loads(path.read_text())
        u = load('upstream.json')
        records = load('observer.json')
        gate = load('length-gate.json')
        assert len(gate) == 4 and gate[0]['done'] and gate[0]['finishes'] == ['stop']
        assert gate[0]['usage']['completion_tokens'] < 64
        assert all(validate(r, n) for r, n in zip(gate[1:], (8, 32, 64)))
        assert len(records) == 8 + 2 * c['concurrency']
        assert all(validate(r, 256) for r in records)
        assert u['completed'] == 8 and u['output_lens'] == [256] * 8
        assert u['total_output_tokens'] == 2048 and u['max_concurrency'] == c['concurrency']
        assert abs(u['output_throughput'] - 2048 / u['duration']) < 1e-8
        bodies = json.loads((root / f"requests-{c['input']}.json").read_text())
        prompts = list(map(json.loads, (root / f"prompts-{c['input']}.jsonl").read_text().splitlines()))
        sizes = {body_sha(b): len(p['tokens']) for b, p in zip(bodies, prompts)}
        measured = records[2 * c['concurrency']:]
        assert Counter(r['request_sha256'] for r in measured) == Counter(map(body_sha, bodies))
        assert all(body_sha(r['body']) == r['request_sha256'] and r['usage']['prompt_tokens'] == sizes[r['request_sha256']] for r in records)
        if c['engine'] != 'zerv-tiered':
            reference = load('reference-prompts.json')
            assert len(reference) == len(prompts)
            assert all(r['prompt'] == p['prompt'] and r['tokens'] == p['tokens'] for r, p in zip(reference, prompts))
        assert c['prompt_gate'] == 'passed'
        vram = c['vram_peak'] / 1024**3
        host = int(c['host_memory']['values']['VmHWM'].split()[0]) / 1024**2
        assert 0 < vram <= 24 and 0 < host <= 32
        first = [1000 * (r['text_times'][0] - r['send']) for r in measured]
        assert first == c['measured_first_text_ms']
        e2e = [1000 * (r['end'] - r['send']) for r in measured]
        text_gaps = [1000 * (b - a) for r in measured for a, b in zip(r['text_times'], r['text_times'][1:])]
        row = dict(engine=c['engine'], round=c['round'], input=c['input'], concurrency=c['concurrency'],
                   output_tokens_s=u['output_throughput'], wall_s=u['duration'],
                   upstream_e2e_mean_ms=u['mean_e2el_ms'], upstream_e2e_p99_ms=u['p99_e2el_ms'],
                   first_text_mean_ms=statistics.mean(first), first_text_p99_ms=percentile(first, .99),
                   observed_e2e_mean_ms=statistics.mean(e2e), observed_e2e_p99_ms=percentile(e2e, .99),
                   text_event_gap_mean_ms=statistics.mean(text_gaps),
                   vram_gib=vram, host_hwm_gib=host, actual_prompt_lengths=c['actual_prompt_lengths'],
                   upstream_scalar_metrics={k: v for k, v in u.items() if isinstance(v, (int, float))})
        if c['engine'] == 'zerv-tiered':
            log = (directory / 'server.log').read_text()
            row['tier_counters'] = [s for s in log.splitlines() if s.startswith(('zerv: disk prefix', 'zerv: prefix checkpoints', 'zerv: staging prefetch'))]
            disk = re.search(r'disk prefix archive: (\d+) writes, (\d+) restores, (\d+) bytes written, (\d+) read', log)
            assert disk
            row['disk'] = dict(zip(('writes', 'restores', 'bytes_written', 'bytes_read'), map(int, disk.groups())))
        rows.append(row)
        groups[(c['engine'], c['input'], c['concurrency'])].append(row)
    metrics = ('output_tokens_s', 'wall_s', 'upstream_e2e_mean_ms', 'upstream_e2e_p99_ms', 'first_text_mean_ms', 'first_text_p99_ms',
               'observed_e2e_mean_ms', 'observed_e2e_p99_ms', 'text_event_gap_mean_ms', 'vram_gib', 'host_hwm_gib')
    aggregate = [dict(engine=k[0], input=k[1], concurrency=k[2],
                      **{metric: stats([r[metric] for r in sorted(rs, key=lambda r: r['round'])]) for metric in metrics})
                 for k, rs in sorted(groups.items())]
    return dict(status='passed', manifest_sha256=sha(root/'manifest.json'), measured_requests=288,
                warmup_requests=sum(2*c['concurrency'] for c in manifest['cases']),
                artifacts_sha256=artifacts, aggregate=aggregate, trials=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = summarize(a.directory.resolve())
    (a.directory/'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified 36/36 points, 288 measured and 180 warmup responses; zero count/termination/resource failures.')
    print('| Raw input | C | Engine | Output tokens/s | Wall s | First text ms | VRAM GiB max | Host GiB max |')
    print('|---|---|---|---|---|---|---|---|')
    for row in result['aggregate']:
        values = [f"{row[m]['mean']:.2f} ± {row[m]['sd']:.2f}" for m in ('output_tokens_s', 'wall_s', 'first_text_mean_ms')]
        print(f"| {row['input']} | {row['concurrency']} | {row['engine']} | " + ' | '.join(values) + f" | {max(row['vram_gib']['trials']):.3f} | {max(row['host_hwm_gib']['trials']):.3f} |")
    for length in (1024, 8192):
        values = sorted({n for r in result['trials'] if r['input'] == length for n in r['actual_prompt_lengths']})
        print(f'Actual rendered prompt counts, raw {length}: {values}')
    print('Native disk totals (includes gates/warmups):', {k: sum(r['disk'][k] for r in result['trials'] if 'disk' in r) for k in ('writes', 'restores', 'bytes_written', 'bytes_read')})
    print('\n| Engine | Input | C | Upstream E2E mean ms | Upstream E2E p99 ms | First-text p99 ms | Text-event gap mean ms |')
    print('|---|---|---|---|---|---|---|')
    for r in result['aggregate']:
        values = [f"{r[m]['mean']:.2f} ± {r[m]['sd']:.2f}" for m in ('upstream_e2e_mean_ms', 'upstream_e2e_p99_ms', 'first_text_p99_ms', 'text_event_gap_mean_ms')]
        print(f"| {r['engine']} | {r['input']} | {r['concurrency']} | " + ' | '.join(values) + ' |')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        raise SystemExit('Use tools/py bench/summarize_inferencex.py DIRECTORY')
    main()
