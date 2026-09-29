import json
from pathlib import Path
s = json.loads(Path('docs/bench/data/2026-09-29-demand-quality/scores.json').read_text())
for name, e in s['engines'].items():
    label = ('join128' if 'reuse-join=' in name else 'join0') if name.startswith('zerv') else name
    print(label, e['responses'], 'failures', len(e['failures']), 'resources', e['resource_pass'])
    seen = set()
    for f in e['failures']:
        k = (tuple(f.get('key', [])[2:]), f.get('output'))
        if k not in seen: print(' ', k, f.get('reasons'))
        seen.add(k)
    for level, m in s['metrics'][name].items():
        print(level, {k: (round(v['mean'],3), round(v['sd'],3)) for k,v in m.items()})
# Prior later-turn latency and cache evidence, every round and request.
p = Path('docs/bench/data/2026-09-29-reuse-join-three-serving')
for r in map(json.loads, (p/'raw.jsonl').read_text().splitlines()):
    if r['engine'].startswith('zerv') and r['level'] == 4 and r['turn'] == 2:
        print('prior', 'join128' if 'reuse-join=' in r['engine'] else 'join0', r['round'], r['conversation'], round(1000*(r['times'][0]-r['send']),1), r['usage'])
p = Path('docs/bench/data/2026-09-29-demand-quality-serving')
rows = list(map(json.loads, (p/'raw.jsonl').read_text().splitlines()))
def key(r): return r['round'], r['level'], r['conversation'], r['turn']
def signature(r): return r['output_sha256'], r['usage']['prompt_tokens'], r['usage']['completion_tokens'], r['finish_reasons']
basename = rows[0]['engine']
base = {key(r): signature(r) for r in rows if r['engine'] == basename}
for name in dict.fromkeys(r['engine'] for r in rows):
    print('exact output/count/finish matches', name, sum(signature(r) == base[key(r)] for r in rows if r['engine'] == name))
w = json.loads(Path('bench/workloads/multiturn-demand-quality-v1.json').read_text())
for c in [w['conversations'][1], w['conversations'][2]]:
    records = [json.loads(line) for line in c['system'].splitlines()[1:]]
    i = int(c['name'][-1]); selected = [records[j] for j in (4+i,63+i,124-i)]
    print('independent sort check', c['name'], selected, 'expected', sorted(selected, key=lambda r: (r['priority'], r['id'])))
for name, e in s['engines'].items():
    print('quality resources GiB', name, 'vram', max(r['vram_peak'] for r in e['resources'])/1024**3, 'rss', max(int(r['host_memory']['VmHWM'].split()[0]) for r in e['resources'])/1024**2)
