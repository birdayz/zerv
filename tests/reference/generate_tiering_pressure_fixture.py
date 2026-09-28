#!/usr/bin/env python3
"""Independent declarative pressure decisions and prefix-set source/drop traces.
Run before native implementation: tools/py tests/reference/generate_tiering_pressure_fixture.py
Transfer/scheduler interleavings are separate interface gates, not simulated DMA.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
from generate_cache_source_fixture import Oracle, prefix


def decide(usage, candidates):
    slots = usage['free_slots'] <= usage['slot_headroom']
    host = usage['host_pages'] > 0 and usage['free_host_pages'] <= usage['host_headroom']
    eligible = [c for c in candidates if (slots or (host and c['has_host']))
                and (c['backed'] or c['fits'])]
    if not eligible:
        return None
    c = sorted(eligible, key=lambda c: (c['used'], c['handle']['index']))[0]
    return dict(handle=c['handle'], action='discard' if c['backed'] else 'preserve')


def trace(capacity, seed):
    o = Oracle(capacity)
    ready = set()
    pending = None
    rows = []
    def step(op):
        nonlocal pending
        decision = None
        result = 'ok'
        if op['op'] == 'take':
            o.apply(op)
        elif op['op'] == 'pressure':
            candidates = []
            for i, (tokens, used) in o.live.items():
                if o.refs(i) or any(i != j and prefix(tokens, t) for j, (t, _) in o.live.items()):
                    continue
                candidates.append(dict(handle=dict(index=i, generation=o.generations[i]), used=used,
                    has_host=False, backed=tuple(tokens) in ready, fits=len(tokens) <= op['max_tokens']))
            usage = dict(slots=capacity, free_slots=capacity-len(o.live), slot_headroom=op['headroom'],
                         host_pages=0, free_host_pages=0, host_headroom=0)
            if pending is None:
                decision = decide(usage, candidates)
                if decision:
                    i = decision['handle']['index']
                    if decision['action'] == 'discard':
                        del o.live[i]
                    else:
                        o.apply(dict(op='acquire', **decision['handle']))
                        pending = i
        elif op['op'] == 'finish':
            if pending is not None:
                if op['success']:
                    ready.add(tuple(o.live[pending][0]))
                o.apply(dict(op='release', index=pending, generation=o.generations[pending], serial=o.serials[pending]))
                pending = None
        elif op['op'] == 'discard':
            i, g = op['index'], op['generation']
            if i not in o.live or o.generations[i] != g:
                result = 'InvalidSource'
            elif o.refs(i) or any(i != j and prefix(o.live[i][0], t) for j, (t, _) in o.live.items()):
                result = 'Busy'
            else:
                del o.live[i]
        rows.append(dict(**op, result=result, decision=decision, state=o.state(), pending=pending,
                         ready=[list(t) for t in sorted(ready)]))
    seq = list(range(1, 17))
    step(dict(op='take', tokens=seq[:8]))
    step(dict(op='pressure', headroom=0, max_tokens=64))
    step(dict(op='take', tokens=seq))
    step(dict(op='pressure', headroom=capacity-1, max_tokens=64))
    step(dict(op='discard', index=0, generation=o.generations[0]))
    step(dict(op='finish', success=True))
    step(dict(op='pressure', headroom=capacity-1, max_tokens=64))
    rng = random.Random(seed)
    for _ in range(512):
        kind = rng.choices(['take', 'pressure', 'finish', 'discard'], [4, 5, 3, 2])[0]
        if kind == 'take':
            base = rng.choice([1, 21, 41, 61])
            step(dict(op=kind, tokens=list(range(base, base+rng.choice([4, 8, 12, 16])))))
        elif kind == 'pressure':
            step(dict(op=kind, headroom=rng.randrange(capacity), max_tokens=rng.choice([4, 8, 64])))
        elif kind == 'finish':
            step(dict(op=kind, success=rng.choice([False, True])))
        else:
            i = rng.randrange(capacity+1)
            g = o.generations[i] if i < capacity else 1
            step(dict(op=kind, index=i, generation=g if rng.randrange(4) else max(0, g-1)))
    step(dict(op='finish', success=False))
    return dict(capacity=capacity, seed=seed, steps=rows)


def main():
    rng = random.Random(70703)
    cases = []
    # Scalar matrix separates byte and slot pressure, tie-breaking and oversized/clean leaves.
    for index in range(2048):
        slots = rng.choice([1, 2, 3, 8, 64, 256])
        host_pages = rng.choice([0, 1, 16, 65536])
        usage = dict(slots=slots, free_slots=rng.randrange(slots+1), slot_headroom=rng.randrange(slots),
                     host_pages=host_pages, free_host_pages=rng.randrange(host_pages+1),
                     host_headroom=rng.randrange(host_pages) if host_pages else 0)
        candidates = [dict(handle=dict(index=i, generation=rng.randrange(1, 1000)),
                      used=rng.randrange(8), has_host=bool(rng.randrange(2)),
                      backed=bool(rng.randrange(2)), fits=bool(rng.randrange(2)))
                      for i in range(min(16, slots-usage['free_slots']))]
        rng.shuffle(candidates)
        cases.append(dict(usage=usage, candidates=candidates, decision=decide(usage, candidates)))
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_name('generate_cache_source_fixture.py')
    out = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               source_oracle_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases,
               traces=[trace(n, 70703+n) for n in [1, 2, 8]])
    path = root/'tests/fixtures/tiering-pressure.json'
    path.write_text(json.dumps(out, separators=(',', ':'))+'\n')
    print(f'{len(cases)} decisions, {sum(len(t["steps"]) for t in out["traces"])} transitions; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        sys.exit('run with tools/py')
    main()
