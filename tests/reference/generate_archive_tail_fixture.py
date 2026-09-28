#!/usr/bin/env python3
"""Independent byte-coordinate archive-tail oracle (no native implementation).
Regenerate: tools/py tests/reference/generate_archive_tail_fixture.py
"""
import hashlib
import json
from pathlib import Path
import random
import sys


def canonical(case, start, size):
    # Derive the tensor coordinate of each byte, not ranges to clear.
    b, page, n, per = (case[k] for k in ('element_bytes', 'page', 'tokens', 'per_buffer'))
    npages = (n + page - 1) // page
    piece = 2 * 4 * 256 * page * b
    groups = [min(per, 16 - first) for first in range(0, 16, per)]
    result = bytearray(((start + i) * 13 + 7) % 251 + 1 for i in range(size))
    for i in range(size):
        relative = start + i - case['snapshot_bytes']
        if relative < 0:
            continue
        for layers in groups:
            group_bytes = npages * layers * piece
            if relative < group_bytes:
                break
            relative -= group_bytes
        logical_page, within_page = divmod(relative, layers * piece)
        _, within_layer = divmod(within_page, piece)
        kv, coordinate = divmod(within_layer // b, 4 * 256 * page)
        if kv == 0:  # K: head, dimension, token
            token = coordinate % page
        else:  # V: head, token, dimension
            _, within_head = divmod(coordinate, page * 256)
            token = within_head // 256
        if logical_page * page + token >= n:
            result[i] = 0
    return hashlib.sha256(result).hexdigest()


def main():
    rng = random.Random(70702)
    cases = []
    for b in (2, 4):
        for page in (128, 256):
            for per in (1, 3, 16):
                for n in (1, page - 1, page, page + 1, 2 * page + 1):
                    c = dict(element_bytes=b, page=page, tokens=n, per_buffer=per, snapshot_bytes=36)
                    npages = (n + page - 1) // page
                    piece = 2 * 4 * 256 * page * b
                    total = 36 + 16 * npages * piece
                    starts = {0, 1, 32, 34, 36, total - 4}
                    base = 36
                    for first in range(0, 16, per):
                        layers = min(per, 16 - first)
                        last_page = base + (npages - 1) * layers * piece
                        for layer in range(layers):
                            at = last_page + layer * piece
                            starts.update((at, at + (n % page) * b - 1,
                                at + piece // 2 - 4,
                                at + piece // 2 + (n % page) * 256 * b - 1))
                        base += npages * layers * piece
                        starts.add(base - 4)
                    windows = [(s, min(68, total - s)) for s in sorted(starts) if 0 <= s < total]
                    windows += [(rng.randrange(total - 4096), 4096) for _ in range(3)]
                    windows += [(36 + (npages - 1) * per * piece, min(1 << 20, per * piece))]
                    c['windows'] = [dict(offset=s, size=z, sha256=canonical(c, s, z)) for s, z in windows]
                    cases.append(c)
    out = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               seed=70702, cases=cases)
    root = Path(__file__).resolve().parents[2]
    path = root / 'tests/fixtures/archive-tails.json'
    path.write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(f'{len(cases)} layouts, {sum(len(c["windows"]) for c in cases)} windows; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        sys.exit('run with tools/py')
    main()
