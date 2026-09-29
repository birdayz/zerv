"""Independent finite-domain greedy oracle: remove EOG IDs before argmax."""
import hashlib
import itertools
import json
from pathlib import Path
cases = []
for logits in itertools.product((-1, 0, 1), repeat=3):
    for mask in range(1, 7):
        eos = [i for i in range(3) if mask & (1 << i)]
        allowed = set(range(3)) - set(eos)
        best = min(allowed, key=lambda i: (-logits[i], i))
        cases.append(dict(logits=logits, eos=eos, token=best))
p = Path('tests/fixtures/ignore-eos.json')
p.write_text(json.dumps(dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases), separators=(',', ':'))+'\n')
print(len(cases), hashlib.sha256(p.read_bytes()).hexdigest())
