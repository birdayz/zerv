"""Independent boundary-set oracle for bounded reuse checkpoint suppression."""
import hashlib
import itertools
import json
from pathlib import Path
cases = []
for n in range(2, 7):
    for mask in itertools.product((False, True), repeat=n-1):
        tokens = [9] + [9 if bit else 1 for bit in mask]
        boundaries = {i for i in range(1, n) if tokens[i] == 9}
        extremes = {min(boundaries), max(boundaries)} if boundaries else set()
        for start in range(n+1):
            for threshold in (0, 1, 3, 6):
                expected = sorted(i for i in extremes if start < i < n)
                if 0 < start < n and 0 < n-start <= threshold: expected = []
                cases.append(dict(tokens=tokens, start=start, threshold=threshold, expected=expected))
out = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
path = Path('tests/fixtures/reuse-join.json')
path.write_text(json.dumps(out, separators=(',', ':')) + '\n')
print(len(cases), hashlib.sha256(path.read_bytes()).hexdigest())
