import json
from pathlib import Path
p = Path('docs/bench/data/2026-09-29-demand-pool-serving')
m = json.loads((p/'manifest.json').read_text())
for level in (1,4):
    print('C', level)
    for name, r in json.loads(Path(f'docs/bench/data/2026-09-29-demand-quality/pool-level{level}.json').read_text()).items():
        label = ('256' if 'pages=256' in name else '192') if name.startswith('zerv') else name
        print(label, r['exact_matches_to_native_off'], '/', r['responses'], {k: (round(v['mean'],3),round(v['sd'],3)) for k,v in r['metrics'].items()})
for name, e in m['engines'].items():
    print(name)
    print('resources', [(round(r['vram_peak']/1024**3,3), round(int(r['host_memory']['VmHWM'].split()[0])/1024**2,3)) for r in e['resources']])
    if not name.startswith('zerv'): continue
    for resource in e['resources']:
        text = (p/resource['log']).read_text()
        for line in text.splitlines():
            if line.startswith(('zerv: batcher:', 'zerv: disk prefix archive:', 'zerv: queued demand:')): print(resource['round'],line)
rows = list(map(json.loads, (p/'raw.jsonl').read_text().splitlines()))
for name in m['engines']:
    if not name.startswith('zerv'): continue
    reused = [r for r in rows if r['engine'] == name and r['level'] == 4 and r['turn'] > 0]
    print('zero reuse', '256' if 'pages=256' in name else '192', [(r['round'],r['conversation'],r['turn']) for r in reused if r['usage']['prompt_tokens_details']['cached_tokens']==0], '/',len(reused))
