import json
from pathlib import Path
rows = [json.loads(line) for line in Path('docs/bench/data/2026-09-29-prompt-equivalence-final/native.jsonl').read_text().splitlines()]
a, b = rows[:2]
for field in ('prompt', 'tokens'):
    n = next((i for i, (x, y) in enumerate(zip(a[field], b[field])) if x != y), min(len(a[field]), len(b[field])))
    print(field, 'common', n, 'first length', len(a[field]), 'second length', len(b[field]))
    print('first around divergence', repr(a[field][max(0,n-80):n+100]))
    print('second around divergence', repr(b[field][max(0,n-80):n+100]))
