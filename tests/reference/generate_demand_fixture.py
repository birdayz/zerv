"""Independent token-prefix/object ownership expectations; run with tools/py."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
root = Path(__file__).resolve().parents[2]
graphs = [
    [[1,2,3,4], [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9]],
    [[1,2,3,4,5,6,7,8], [1,2,3,4,6,7,8,9]],
    [[1,2,3], [1,2,3,4,5], [7,8,9,10]],
]
cache = []
for nodes in graphs:
    for order in itertools.permutations(nodes):
        for prompt in [[], [999], *nodes, *[n+[999] for n in nodes]]:
            matches = [i for i,n in enumerate(order) if len(n) < len(prompt) and prompt[:len(n)] == n]
            best = max(matches, key=lambda i: len(order[i]), default=-1)
            protected = [best >= 0 and len(n) <= len(order[best]) and order[best][:len(n)] == n for n in order]
            cache.append(dict(nodes=order, prompt=prompt, best=best, protected=protected))
staging = []
for chunks, window, busy, outcome in itertools.product([1,2,3], [1,2], range(7), ['take','cancel','reuse','stop','corrupt']):
    # Six optional ticket owners total; two of eight remain for real demand.
    staged = min(chunks, window, max(0, 6-busy))
    uploaded = chunks if outcome == 'take' else 0
    staging.append(dict(chunks=chunks, window=window, busy=busy, outcome=outcome,
                        staged=staged, uploaded=uploaded, final_readers=0, final_owned=0))
assert any(c['staged'] == 0 for c in staging)
assert any(c['staged'] == 2 for c in staging)
value = dict(reference='independent token prefix sets and finite ticket ownership; no native code',
             generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             cache_cases=cache, staging_cases=staging)
path = root/'tests/fixtures/queued-demand.json'
path.write_text(json.dumps(value, separators=(',', ':'))+'\n')
print(json.dumps(dict(cache_cases=len(cache), staging_cases=len(staging), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
