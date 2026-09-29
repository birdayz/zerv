import json
from pathlib import Path
for filename in ['level1', 'level4', 'three-level1', 'three-level4']:
    print(filename)
    for name, row in json.loads(Path(f'docs/bench/data/2026-09-29-reuse-join/{filename}.json').read_text()).items():
        label = ('join128' if 'reuse-join=128' in name else 'off') if name.startswith('zerv') else name
        print(label, 'matches', row['exact_matches_to_native_off'], '/', row['responses'], ', '.join(f'{k}={v["mean"]:.3f} ± {v["sd"]:.3f}' for k,v in row['metrics'].items()))
