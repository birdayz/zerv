"""Independent POSIX/SHA256 transport-window goldens; run explicitly with tools/py."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if "/bazel-out/" not in sys.executable:
    raise SystemExit("run with tools/py")
root = Path(__file__).resolve().parents[2]
pattern = bytes((73*i + i//257 + 19) % 256 for i in range(257*256))
cases = []
for mib in [1, 2, 4, 8]:
    chunk = mib << 20
    length = 2*chunk + 17
    raw = (pattern * ((length + len(pattern)-1)//len(pattern)))[:length]
    padded = raw + bytes((-length) % chunk)
    with tempfile.TemporaryFile() as f:
        for at in reversed(range(0, len(padded), chunk)):
            assert os.pwrite(f.fileno(), padded[at:at+chunk], at) == chunk
        actual = os.pread(f.fileno(), len(padded), 0)
        assert actual == padded
    cases.append(dict(chunk=chunk, length=length, hashes=[hashlib.sha256(actual[at:at+chunk]).hexdigest() for at in range(0, len(actual), chunk)]))
out = dict(reference="Python POSIX positional I/O and hashlib", python=sys.version,
           generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
(root / "tests/fixtures/archive/windows.json").write_text(json.dumps(out, indent=2) + "\n")
