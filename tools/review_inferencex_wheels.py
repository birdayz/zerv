"""Download hash-locked compatible wheels as inert data, never install or import them."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.request
import zipfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path('third_party/research-serving/inferencex-wheels-v1')
ROOT.mkdir(parents=True, exist_ok=True)
lock = Path('requirements_inferencex_lock.txt').read_text()
blocks = re.split(r'(?m)^(?=[a-zA-Z0-9_-]+==)', lock)[1:]
tags = {tag: i for i, tag in enumerate(sys_tags())}
manifest = []
for block in blocks:
    name, version = re.match(r'([^=]+)==([^\s\\]+)', block).groups()
    hashes = set(re.findall(r'--hash=sha256:([0-9a-f]{64})', block))
    url = f'https://pypi.org/pypi/{name}/{version}/json'
    with urllib.request.urlopen(url, timeout=60) as r: raw = r.read()
    (ROOT / f'{name}-{version}.json').write_bytes(raw)
    meta = json.loads(raw)
    choices = []
    for item in meta['urls']:
        if not item['filename'].endswith('.whl'): continue
        _, _, _, wheel_tags = parse_wheel_filename(item['filename'])
        scores = [tags[t] for t in wheel_tags if t in tags]
        if scores: choices.append((min(scores), item))
    assert choices, name
    item = min(choices, key=lambda x: x[0])[1]
    assert item['digests']['sha256'] in hashes
    assert item['url'].startswith('https://files.pythonhosted.org/') and item['size'] < 100 * 1024**2
    wheel = ROOT / item['filename']
    if not wheel.exists():
        with urllib.request.urlopen(item['url'], timeout=120) as r: wheel.write_bytes(r.read(100 * 1024**2 + 1))
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert digest == item['digests']['sha256']
    out = ROOT / 'unpacked' / name
    members = []
    with zipfile.ZipFile(wheel) as archive:
        assert sum(i.file_size for i in archive.infolist()) < 500 * 1024**2
        for i in archive.infolist():
            p = PurePosixPath(i.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
            assert not stat.S_ISLNK(i.external_attr >> 16)
            assert not i.filename.endswith('.pth'), (name, i.filename)
            members.append(i.filename)
        archive.extractall(out)
    manifest.append(dict(name=name, version=version, url=item['url'], sha256=digest,
                         metadata_url=url, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                         requires_dist=meta['info']['requires_dist'], files=members))
Path('docs/bench/data/2026-09-29-inferencex-security/wheels.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Downloaded and inspected archive paths for', len(manifest), 'wheels; NOT installed or executed')
