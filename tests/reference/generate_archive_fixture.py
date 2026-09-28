#!/usr/bin/env python3
"""Independent disk archive padded-chunk/SHA256 oracle (POSIX I/O, Python hashlib)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "tests/fixtures/storage/positional.bin"
    payload = source.read_bytes()
    cases = []
    for length in (8193, 12288, 32768, 65000):
        padded = payload[:length] + bytes((-length) % 4096)
        with tempfile.TemporaryFile() as f:
            for offset in reversed(range(0, len(padded), 4096)):
                assert os.pwrite(f.fileno(), padded[offset:offset + 4096], offset) == 4096
            actual = os.pread(f.fileno(), len(padded), 0)
            assert actual == padded
        cases.append(dict(length=length, chunk=4096, hashes=[
            hashlib.sha256(actual[o:o + 4096]).hexdigest() for o in range(0, len(actual), 4096)]))
    out = root / "tests/fixtures/archive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle.json").write_text(json.dumps(dict(reference="Python POSIX I/O + hashlib SHA256",
        python=sys.version, source_sha256=hashlib.sha256(payload).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
