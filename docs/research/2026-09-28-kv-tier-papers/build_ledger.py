"""Record source URLs, immutable revisions and hashes for this research inventory.
Run: tools/py docs/research/2026-09-28-kv-tier-papers/build_ledger.py
"""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parents[3]
rev = 'f8b50d5f1a2fa4d33d029ed1e82188de70b137f0'
items = []
def add(path, url, revision, role='primary source'):
    p = root / path
    items.append(dict(path=str(path), url=url, revision=revision, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, role=role))
base = root / 'third_party/mooncake' / rev
for p in sorted(base.rglob('*')):
    if not p.is_file(): continue
    rel = p.relative_to(base)
    if str(rel) == 'revision.json': url = f'https://api.github.com/repos/kvcache-ai/Mooncake/commits/{rev}'
    elif str(rel) == 'tree.json': url = f'https://api.github.com/repos/kvcache-ai/Mooncake/git/trees/{rev}?recursive=1'
    else: url = f'https://raw.githubusercontent.com/kvcache-ai/Mooncake/{rev}/{rel}'
    add(p.relative_to(root), url, rev, 'repository source/docs; inspected, not executed or linked')
for ident in ['2407.00079v4', '2312.05516v3', '2403.19708v3', '2510.09665v2']:
    base = Path('third_party/mooncake/arxiv-' + ident) if ident.startswith('2407') else Path('third_party/kv-tier-papers') / ident
    add(base / 'paper.html', f'https://arxiv.org/html/{ident}', ident)
    add(base / 'paper.txt', f'https://arxiv.org/html/{ident}', ident, 'derived via extract_html.py; equations may contain duplicate accessibility text')
add(Path('third_party/mooncake/fast25/paper.pdf'), 'https://www.usenix.org/system/files/fast25-qin.pdf', 'FAST25 published proceedings, pp.155–170')
add(Path('third_party/mooncake/fast25/paper.txt'), 'https://www.usenix.org/system/files/fast25-qin.pdf', 'FAST25 published proceedings', 'derived via extract_pdf.py')
add(Path('third_party/kv-tier-papers/discovery-2026-09-28/arxiv.xml'), 'https://export.arxiv.org/api/query?id_list=2407.00079,2312.05516,2403.19708,2510.09665', 'retrieved 2026-09-28', 'discovery metadata, not a substitute for paper text')
base = Path('third_party/kv-tier-papers/pypdf-6.1.1')
meta = json.loads((root / base / 'pypi.json').read_text())
wheel = next(x for x in meta['urls'] if x['filename'].endswith('.whl'))
add(base / 'pypi.json', 'https://pypi.org/pypi/pypdf/6.1.1/json', '6.1.1', 'research-only extractor metadata')
add(base / wheel['filename'], wheel['url'], '6.1.1', 'research-only verified pure Python wheel; no installation or production dependency')
Path(__file__).with_name('sources.json').write_text(json.dumps(dict(retrieved='2026-09-28', sources=items), indent=2) + '\n')
print(len(items), 'source/artifact hashes recorded')
