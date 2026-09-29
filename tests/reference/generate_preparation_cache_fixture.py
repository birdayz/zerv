"""Independent prefix/object-set joint preparation oracle; no native execution."""
import hashlib
import json
from pathlib import Path
import random
import sys

P = 4

def prefix(a, b):
    return len(a) <= len(b) and b[:len(a)] == a


def page_keys(tokens):
    return [('full', tuple(tokens[:end])) if end <= len(tokens) else ('tail', tuple(tokens))
            for end in range(P, len(tokens)+P, P)]


def ownership(nodes):
    keys = [page_keys(n) for n in nodes]
    first = []
    owners = {}
    for i, n in enumerate(nodes):
        ancestors = [j for j, other in enumerate(nodes) if len(other) < len(n) and prefix(other, n)]
        parent = max(ancestors, key=lambda j: len(nodes[j])) if ancestors else None
        own = 0
        if parent is not None:
            for a, b in zip(keys[i], keys[parent]):
                if a != b:
                    break
                own += 1
        first.append(own)
        for key in keys[i][own:]:
            owners.setdefault(key, set()).add(i)
    return keys, first, owners


def oracle(inp):
    nodes = [n.copy() for n in inp['nodes']]
    keys, first, owners = ownership(nodes)
    selected = inp['selected']
    moves = [k for k in keys[selected][first[selected]:] if owners[k] == {selected}]
    moves = moves[:min(inp['window'], 32-inp['reserve'])]
    held = {selected} if moves else set()
    late = inp['late']
    if late and late not in nodes and not any(len(late) < len(nodes[j]) and prefix(late, nodes[j]) for j in held):
        nodes.append(late)
    protect = inp['protect']
    if protect >= 0 and protect < len(nodes):
        assert protect != selected
        held.add(protect)
    keys, after_first, owners = ownership(nodes)
    refs = lambda i: sum(prefix(nodes[i], nodes[j]) for j in held)
    live = set(k for k in keys[selected] if k[0] == 'full') if inp['hit'] else set()
    callbacks = 0
    if not moves:
        result = 'none'
    elif inp['mode'] == 'cancel':
        result = 'cancel'
    elif refs(selected) != 1:
        result = 'Busy'
    else:
        callbacks = 1
        result = 'CommitFault' if inp['mode'] == 'fault' else 'InvalidState' if any(owners[k] != {selected} for k in moves) else 'commit'
    committed = set(moves) if result == 'commit' else set()
    if result == 'commit':
        held.remove(selected)
    return dict(input=inp, first=first[selected], planned=len(moves), result=result,
                callbacks=callbacks, copied=len(committed), freed=len(committed-live),
                nodes=[dict(tokens=n, first=after_first[i], host=[k in committed for k in keys[i]],
                            pins=[0 if k in committed else len(owners[k]) for k in keys[i]],
                            refs=refs(i), active=i in held) for i, n in enumerate(nodes)])


def main():
    a = list(range(1, 22))
    b = a[:8] + list(range(101, 114))
    graphs = [[a[:8], a[:16]], [a[:9], a[:17]], [a[:16], b[:16]],
              [a[:4], a[:8], a[:12], b[:12]], [a[:16]], [a[:8], a[:16], b[:16]]]
    cases = []
    for nodes in graphs:
        for selected, tokens in enumerate(nodes):
            late_cases = [[], tokens[:12] + [201, 202, 203, 204], tokens + [301, 302, 303, 304], tokens[:4]]
            for window in [1, 2, 4]:
                for hit in [False, True]:
                    for late in late_cases:
                        for protect in [-1] + [i for i in range(len(nodes)+1) if i != selected]:
                            for mode in ['normal', 'cancel', 'fault']:
                                cases.append(oracle(dict(nodes=nodes, selected=selected, late=late, protect=protect,
                                                         window=window, hit=hit, mode=mode, reserve=0)))
    rng = random.Random(0xD102)
    for _ in range(128):
        nodes = rng.choice(graphs)
        selected = rng.randrange(len(nodes))
        cases.append(oracle(dict(nodes=nodes, selected=selected, late=[], protect=-1,
                                 window=rng.choice([1, 2, 4]), hit=bool(rng.randrange(2)), mode='normal', reserve=32)))
    counts = {s: sum(c['result'] == s for c in cases) for s in ['none', 'commit', 'cancel', 'Busy', 'InvalidState', 'CommitFault']}
    assert all(counts.values()), counts
    assert any(0 < c['freed'] < c['copied'] for c in cases)
    assert any(c['result'] == 'InvalidState' and c['nodes'][c['input']['selected']]['refs'] == 1 for c in cases)
    root = Path(__file__).resolve().parents[2]
    target = root/'tests/fixtures/preparation-cache.json'
    out = dict(reference='independent prefix/object-owner sets', generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), counts=counts, cases=cases)
    target.write_text(json.dumps(out, separators=(',', ':'))+'\n')
    print(json.dumps(dict(cases=len(cases), outcomes=counts, fixture_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit('run with tools/py')
    main()
