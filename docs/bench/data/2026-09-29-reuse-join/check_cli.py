"""Reject incompatible reuse-join configurations before loading a model."""
import hashlib
import json
from pathlib import Path
import subprocess
m = json.loads(Path('docs/bench/data/2026-09-29-reuse-join-three-serving/manifest.json').read_text())
binary = next(v['cmd'][0] for k, v in m['engines'].items() if k.startswith('zerv-'))
assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == m['zerv_sha256']
base = [binary, '--model', '/nonexistent-reuse-join-model', '--port', '0', '--parallel', '2', '--spec-draft', '0']
cases = [
    ['--prefix-cache-reuse-join', '513'],
    ['--prefix-cache-reuse-join', '4294967295'],
    *[['--prefix-cache-reuse-join', '128', *flags] for flags in [
        ['--parallel', '1'], ['--kv-pool', 'static'], ['--prefix-cache', 'flat'],
        ['--prefix-cache-slots', '0'], ['--spec-draft', '1'],
        ['--prefill-chunk', '64'], ['--prefill-chunk', '0']]],
]
results = []
for flags in cases:
    p = subprocess.run(base + flags, capture_output=True, text=True, timeout=20)
    assert p.returncode != 0 and 'InvalidArguments' in p.stderr, (flags, p.stderr)
    assert 'FileNotFound' not in p.stderr
    results.append(dict(command=base + flags, returncode=p.returncode, stderr=p.stderr))
print(json.dumps(dict(binary_sha256=m['zerv_sha256'], cases=len(results), results=results), indent=2))
