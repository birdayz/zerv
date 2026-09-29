"""Display emitted upstream fields only: no metric computation or correction.
Run with tools/py from the project root. Original artifacts are never written.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent/'run-r2'
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['status'] == 'passed' and len(manifest['cases']) == 36
rows = {}
for case in manifest['cases']:
    path = root/case['name']/'agg_upstream.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['result_sha256']
    rows[case['name']] = json.loads(path.read_text())
for field in ('p90_intvty', 'median_intvty', 'output_tput_per_gpu'):
    print(f'\n{field}: emitted values for rounds 0 / 1 / 2 (rounded for display only)')
    print('| Input | C | zerv | llama Vulkan | llama HIP |')
    print('|---|---|---|---|---|')
    for length in (1024, 8192):
        for c in (1, 4):
            cells = [' / '.join(f"{rows[f'{engine}-r{r}-i{length}-c{c}'][field]:.2f}" for r in range(3))
                     for engine in ('zerv-tiered', 'llama-fa-b512', 'rdna3-nofusion')]
            print(f'| {length} | {c} | ' + ' | '.join(cells) + ' |')
