"""Strict, predeclared D.2 structured-answer scoring; no answer repair."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def correct(text, expected):
    try:
        actual = json.loads(text)
    except (ValueError, TypeError):
        return False
    return (type(actual) is list and len(actual) == len(expected)
            and all(type(a) is type(b) and a == b for a, b in zip(actual, expected)))


def evaluate(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    path, digest = next(iter(manifest['workload'].items()))
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    w = json.loads(data)
    rows = [json.loads(line) for line in (root / 'raw.jsonl').read_text().splitlines()]
    expected = {(rnd, level, conv['name'], turn): answer
                for rnd in range(manifest['rounds']) for level in manifest['levels']
                for conv in w['conversations'][:level] for turn, answer in enumerate(conv['expected'])}
    def key(r):
        return r['round'], r['level'], r['conversation'], r['turn']
    baseline = next(name for name in manifest['engines'] if name.startswith('zerv-') and 'reuse-join=' not in name)
    base = {key(r): r for r in rows if r['engine'] == baseline}
    result = {}
    for name, engine in manifest['engines'].items():
        records = [r for r in rows if r['engine'] == name]
        failures = []
        if len(records) != len(expected) or {key(r) for r in records} != set(expected):
            failures.append(dict(reason='incomplete or duplicate coverage'))
        for r in records:
            k = key(r)
            reasons = []
            if r.get('error') or r.get('finish_reasons') != ['stop']:
                reasons.append('error or non-stop termination')
            if not correct(r.get('output_text'), expected.get(k, [])):
                reasons.append('incorrect JSON answer')
            if r.get('request_sha256') != base.get(k, {}).get('request_sha256'):
                reasons.append('input mismatch')
            if name.startswith('zerv-'):
                other = base.get(k, {})
                if (r.get('output_sha256'), r.get('usage')) != (other.get('output_sha256'), other.get('usage')):
                    # cached_tokens legitimately differ; only prompt/completion work is invariant.
                    fields = ('prompt_tokens', 'completion_tokens')
                    if r.get('output_sha256') != other.get('output_sha256') or any(r.get('usage', {}).get(f) != other.get('usage', {}).get(f) for f in fields):
                        reasons.append('native signature mismatch')
            if reasons:
                failures.append(dict(key=k, reasons=reasons, output=r.get('output_text')))
        resources = engine.get('resources', [])
        resource_pass = len(resources) == manifest['rounds'] and all(
            0 < r['vram_peak'] <= 24 * 1024**3 and
            0 < int(r['host_memory'].get('VmHWM', '0 kB').split()[0]) <= 32 * 1024**2
            for r in resources)
        result[name] = dict(responses=len(records), failures=failures, resource_pass=resource_pass,
                            resources=resources, passed=not failures and resource_pass)
    summary = json.loads((root / 'summary.json').read_text())
    metrics = {}
    for name, trials in summary.items():
        metrics[name] = {}
        for level in manifest['levels']:
            selected = [t for t in trials if t['level'] == level]
            values = {'wall_s': [t['wall_s'] for t in selected]}
            for turn in range(3):
                values[f'turn{turn + 1}_ttft_ms'] = [t['turns'][turn]['ttft_p50_ms'] for t in selected]
            metrics[name][level] = {k: dict(mean=statistics.mean(v), sd=statistics.stdev(v), trials=v) for k, v in values.items()}
    qualified = [n for n in result if not n.startswith('zerv-') and result[n]['passed']]
    parity = {}
    for name in result:
        if not name.startswith('zerv-'): continue
        ratios = {f'C{level}:{metric}': values['mean'] / min(metrics[n][level][metric]['mean'] for n in qualified)
                  for level, ms in metrics[name].items() for metric, values in ms.items()} if len(qualified) == 2 else {}
        parity[name] = dict(ratios=ratios, passed=result[name]['passed'] and bool(ratios) and all(v <= 1.05 for v in ratios.values()))
    return dict(manifest_status=manifest['status'], engines=result, metrics=metrics, parity=parity,
                quality_passed=manifest['status'] == 'passed' and all(r['passed'] for r in result.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args = p.parse_args()
    result = evaluate(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['quality_passed'] else 1)
