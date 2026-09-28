"""Independent object-set preparation ownership goldens; explicit tools/py only."""
import hashlib
import json
from pathlib import Path
import random
import sys

if "/bazel-out/" not in sys.executable:
    raise SystemExit("run with tools/py")


def oracle(inputs):
    ids = [(3*i + inputs['shift']) % 8 for i in range(4)]
    caches = {q: ({'selected'} | ({'other'} if inputs['shared'] & (1 << i) else set()))
              for i, q in enumerate(ids)}
    live = {q: set() for q in range(8)}
    hosts = {h: ('other-cache' if h < inputs['busy'] else None) for h in range(8)}
    logical = [h % 4 if hosts[h] else 0 for h in range(8)]
    candidates = [i for i in range(inputs['first'], 4) if caches[ids[i]] == {'selected'}]
    available = [h for h in hosts if hosts[h] is None]
    budget = max(0, len(available) - inputs['reserve'])
    pairs = list(zip(candidates[:min(inputs['window'], budget)], available))
    moves = [dict(index=i, from_page=ids[i], to=h) for i, h in pairs]
    for _, h in pairs:
        hosts[h] = 'reserved'
    if inputs['hit']:
        for q in ids[:3 if inputs['hit'] == 2 else 4]:
            live[q].add('request')
        if inputs['hit'] == 2:
            fresh = next(q for q in range(8) if q not in caches)
            live[fresh].add('request')  # private copy of the partial tail
    if inputs['alias'] >= 0:
        caches[ids[inputs['alias']]].add('late-other-root')
    valid = all(caches[ids[i]] == {'selected'} for i, _ in pairs)
    status = 'none' if not pairs else 'abort' if inputs['cancel'] else 'commit' if valid else 'conflict'
    copied = freed = 0
    result = ids.copy()
    if status == 'commit':
        for i, h in pairs:
            q = ids[i]
            caches[q].remove('selected')
            copied += 1
            freed += int(not live[q])
            hosts[h] = 'selected-cache'
            logical[h] = i
            result[i] = (1 << 31) | h
    else:
        for _, h in pairs:
            hosts[h] = None
    return dict(input=inputs, moves=moves, status=status, copied=copied, freed=freed,
                pages=result, pins=[len(caches.get(q, set())) for q in range(8)],
                masks=[int(bool(live[q])) for q in range(8)],
                host_used=[hosts[h] is not None for h in range(8)], host_logical=logical)


cases = []
for first in [0, 2, 4]:
    for window in [1, 2, 4]:
        for hit in [0, 1, 2]:
            for alias in [-1, 0, 3]:
                for cancel in [False, True]:
                    cases.append(oracle(dict(first=first, window=window, hit=hit, alias=alias,
                                             cancel=cancel, busy=0, reserve=1, shared=0, shift=7)))
rng = random.Random(0xD101)
for _ in range(1200):
    cases.append(oracle(dict(first=rng.randrange(5), window=rng.choice([1, 2, 4]),
                             hit=rng.randrange(3), alias=rng.randrange(-1, 4), cancel=bool(rng.randrange(2)),
                             busy=rng.randrange(9), reserve=rng.randrange(9), shared=rng.randrange(16), shift=rng.randrange(8))))
counts = {s: sum(c['status'] == s for c in cases) for s in ['none', 'commit', 'abort', 'conflict']}
assert all(counts.values())
assert any(c['copied'] > c['freed'] > 0 for c in cases)
assert any(c['status'] == 'conflict' and c['moves'][-1]['index'] == c['input']['alias'] and len(c['moves']) > 1 for c in cases)
root = Path(__file__).resolve().parents[2]
out = dict(reference='independent named-owner sets; no native implementation', seed=0xD101,
           generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), counts=counts, cases=cases)
(root/'tests/fixtures/preparation.json').write_text(json.dumps(out, separators=(',', ':'))+'\n')
print(json.dumps(dict(cases=len(cases), outcomes=counts, generator_sha256=out['generator_sha256'])))
