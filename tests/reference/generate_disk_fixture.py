#!/usr/bin/env python3
"""Independent scratch-store oracle: positional OS I/O, not zerv/io_uring.

Regenerate explicitly: tools/py tests/reference/generate_disk_fixture.py
Ordinary native tests embed the resulting bytes; they do not run this script.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "tests/fixtures/storage"
    out.mkdir(parents=True, exist_ok=True)
    seed, block, count = 0x5EED18D6, 4096, 16
    rng = random.Random(seed)
    expected = rng.randbytes(block * count)
    order = list(range(count))
    rng.shuffle(order)
    with tempfile.TemporaryFile() as f:
        os.ftruncate(f.fileno(), len(expected))
        for i in order:
            assert os.pwrite(f.fileno(), expected[i * block:(i + 1) * block], i * block) == block
        actual = os.pread(f.fileno(), len(expected), 0)
        assert actual == expected
    (out / "positional.bin").write_bytes(actual)
    (out / "manifest.json").write_text(json.dumps(dict(
        schema=1, reference="Python os.pwrite/os.pread (kernel positional I/O)",
        python=sys.version, kernel=os.uname().release, seed=seed, block_bytes=block,
        blocks=count, write_order=order, bytes=len(actual),
        sha256=hashlib.sha256(actual).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    ), indent=2) + "\n")


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit(f"run it with tools/py {sys.argv[0]}: the pinned interpreter")
    main()
