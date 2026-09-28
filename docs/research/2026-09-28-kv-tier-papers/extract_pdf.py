"""Research-only PDF extraction with pinned Python and a verified research wheel.
No installation or production dependency. Run with tools/py and a PDF path.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

root = Path(__file__).resolve().parents[3]
cache = root / 'third_party/kv-tier-papers/pypdf-6.1.1'
cache.mkdir(parents=True, exist_ok=True)
metadata = cache / 'pypi.json'
if not metadata.exists():
    metadata.write_bytes(urllib.request.urlopen('https://pypi.org/pypi/pypdf/6.1.1/json', timeout=60).read())
info = json.loads(metadata.read_text())
entry = next(u for u in info['urls'] if u['filename'].endswith('.whl'))
wheel = cache / entry['filename']
if not wheel.exists(): wheel.write_bytes(urllib.request.urlopen(entry['url'], timeout=60).read())
if hashlib.sha256(wheel.read_bytes()).hexdigest() != entry['digests']['sha256']:
    raise RuntimeError('wheel checksum mismatch')
sys.path.insert(0, str(wheel))
from pypdf import PdfReader
for filename in sys.argv[1:]:
    pdf = Path(filename)
    reader = PdfReader(pdf)
    pdf.with_suffix('.txt').write_text('\n\n'.join(f'PAGE {i + 1}\n' + p.extract_text() for i, p in enumerate(reader.pages)) + '\n')
    print(pdf, len(reader.pages), 'pages; pypdf wheel', entry['digests']['sha256'])
