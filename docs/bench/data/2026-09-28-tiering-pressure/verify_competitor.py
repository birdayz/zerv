"""Verify the already-installed RDNA3 build against every pinned build output; no download.
Run with tools/py before/after the serialized RDNA3 serving comparison.
"""
import hashlib
import json
from pathlib import Path
import sys

if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
ROOT = Path(__file__).resolve().parents[4]
base = ROOT/'third_party/competitors/rdna3-15995a12'
manifest = json.loads((base/'manifest.json').read_text())
assert manifest['source']['commit'] == '15995a12d1d530645a4f34c72afdaa30fa680149'
for name, expected in manifest['outputs'].items():
    with (base/name).open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    assert actual == expected, (name, expected, actual)
print(json.dumps(dict(path=str(base), verified_outputs=len(manifest['outputs']), manifest=manifest), indent=2))
