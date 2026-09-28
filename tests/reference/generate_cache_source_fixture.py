#!/usr/bin/env python3
"""Independent prefix-set source-lease oracle; never executes native code.
Regenerate: tools/py tests/reference/generate_cache_source_fixture.py
"""
import hashlib
import json
from pathlib import Path
import random
import sys


def prefix(a, b):
    return len(a) <= len(b) and b[:len(a)] == a


class Oracle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.live = {}
        self.generations = [0] * capacity
        self.serials = [0] * capacity
        self.leases = {}
        self.tick = 0

    def refs(self, i):
        return sum(i in lease[2] for lease in self.leases.values())

    def leaf(self):
        candidates = [i for i, (tokens, _) in self.live.items() if not self.refs(i)
                      and not any(i != j and prefix(tokens, other) for j, (other, _) in self.live.items())]
        return min(candidates, key=lambda i: (self.live[i][1], i)) if candidates else None

    def apply(self, op):
        kind = op['op']
        if kind == 'take':
            tokens = op['tokens']
            if any(tokens == t for t, _ in self.live.values()):
                return 'ok'
            if any(self.refs(i) and prefix(tokens, t) for i, (t, _) in self.live.items()):
                return 'ok'
            free = sorted(set(range(self.capacity)) - self.live.keys())
            i = free[0] if free else self.leaf()
            if i is None:
                return 'ok'
            self.generations[i] += 1
            self.serials[i] = 0
            self.tick += 2  # fill + touchPath in the existing cache's LRU contract
            self.live[i] = (tokens, self.tick)
            for j, (t, used) in list(self.live.items()):
                if prefix(t, tokens):
                    self.live[j] = (t, self.tick)
            return 'ok'
        if kind == 'evict':
            i = self.leaf()
            if i is None:
                return 'no'
            del self.live[i]
            return 'yes'
        i, generation = op['index'], op['generation']
        if i not in self.live or generation != self.generations[i]:
            return 'InvalidSource'
        if kind == 'acquire':
            if i in self.leases:
                return 'Busy'
            self.serials[i] += 1
            path = {j for j, (t, _) in self.live.items() if prefix(t, self.live[i][0])}
            self.leases[i] = (generation, self.serials[i], path)
            return 'ok'
        if kind == 'release':
            lease = self.leases.get(i)
            if lease is None or lease[:2] != (generation, op['serial']):
                return 'InvalidSource'
            del self.leases[i]
            return 'ok'
        raise ValueError(kind)

    def state(self):
        return [dict(tokens=self.live.get(i, ([], 0))[0], generation=self.generations[i],
                     serial=self.serials[i], active=i in self.leases, refs=self.refs(i))
                for i in range(self.capacity)]


def generate(capacity, seed):
    model = Oracle(capacity)
    rows = []
    def step(op):
        result = model.apply(op)
        rows.append(dict(**op, result=result, state=model.state(), candidate=model.leaf()))
    seq = list(range(1, 17))
    for n in (8, 16, 4, 12):
        step(dict(op='take', tokens=seq[:n]))
    for i in list(model.live):
        step(dict(op='acquire', index=i, generation=model.generations[i]))
    step(dict(op='evict'))
    step(dict(op='take', tokens=list(range(101, 109))))
    rng = random.Random(seed)
    for _ in range(768):
        kind = rng.choices(['take', 'evict', 'acquire', 'release'], [4, 2, 3, 3])[0]
        if kind == 'take':
            base = rng.choice([1, 21, 41, 61])
            step(dict(op=kind, tokens=list(range(base, base + rng.choice([4, 8, 12, 16])))))
        elif kind == 'evict':
            step(dict(op=kind))
        else:
            i = rng.randrange(capacity + 1)
            generation = model.generations[i] if i < capacity else 1
            if rng.randrange(5) == 0:
                generation = max(0, generation - 1)
            op = dict(op=kind, index=i, generation=generation)
            if kind == 'release':
                op['serial'] = model.serials[i] if i < capacity else 1
                if rng.randrange(5) == 0:
                    op['serial'] += 1
            step(op)
    for i, (g, s, _) in list(model.leases.items()):
        step(dict(op='release', index=i, generation=g, serial=s))
    while model.live:
        step(dict(op='evict'))
    return dict(capacity=capacity, seed=seed, steps=rows)


def main():
    root = Path(__file__).resolve().parents[2]
    cases = [generate(n, 70701 + n) for n in (1, 4, 8)]
    out = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               contract='docs/specs/cache-source-leases.md', cases=cases)
    target = root / 'tests/fixtures/cache-sources.json'
    target.write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(f'{sum(len(c["steps"]) for c in cases)} transitions; SHA256 {hashlib.sha256(target.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        sys.exit('run with tools/py')
    main()
