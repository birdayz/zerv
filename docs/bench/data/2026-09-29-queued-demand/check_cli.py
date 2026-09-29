#!/usr/bin/env python3
"""Run parser rejection checks on the exact measured binary, without model/GPU loading."""
import hashlib
import json
import pathlib
import subprocess

root = pathlib.Path('docs/bench/data/2026-09-29-demand-serving')
m = json.loads((root / 'manifest.json').read_text())
engine = next(value for name, value in m['engines'].items() if name.startswith('zerv-'))
binary = pathlib.Path(engine['cmd'][0])
assert hashlib.sha256(binary.read_bytes()).hexdigest() == m['zerv_sha256']
base = [str(binary), '--model', '/nonexistent-zerv-demand-cli-model', '--port', '0', '--parallel', '2', '--spec-draft', '0']
checks = [
    ['--prefix-cache-demand', 'unknown'],
    ['--prefix-cache-demand', 'prefetch'],
    ['--prefix-cache-prefetch-chunks', '1'],
    ['--prefix-cache-demand', 'protect', '--prefix-cache-prefetch-chunks', '1'],
    *[['--prefix-cache-demand', 'protect', *flags] for flags in [
        ['--parallel', '1'], ['--kv-pool', 'static'], ['--prefix-cache', 'flat'],
        ['--prefix-cache-slots', '0'], ['--spec-draft', '1']]],
    *[['--prefix-cache-demand', 'prefetch', '--prefix-cache-disk-dir', 'third_party/nvme-probe', '--prefix-cache-disk-mib', '8192', '--prefix-cache-prefetch-chunks', window] for window in ['0', '3', '4294967295']],
]
results = []
for flags in checks:
    p = subprocess.run(base + flags, capture_output=True, text=True, timeout=20)
    assert p.returncode != 0 and 'InvalidArguments' in p.stderr, (flags, p.stderr)
    assert 'FileNotFound' not in p.stderr
    results.append(dict(command=base + flags, returncode=p.returncode, stderr=p.stderr))
print(json.dumps(dict(binary_sha256=m['zerv_sha256'], cases=len(results), results=results), indent=2))
