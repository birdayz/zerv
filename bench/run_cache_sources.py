#!/usr/bin/env python3
"""Repeat source-lease metadata measurements (not serving or tensor transfer).
 tools/py bench/run_cache_sources.py --output NEW_DIRECTORY
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import zerv_build as build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    uname=list(os.uname()), cpuinfo=Path('/proc/cpuinfo').read_text(),
                    status='running', commands=[], scope='metadata only, no I/O')
    try:
        command = build.test_command()
        manifest['commands'].append(command)
        with (args.output / 'tests.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary('zerv-cache-sources-bench')
        manifest.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = ['bench/cache_sources.zig', 'bench/run_cache_sources.py', 'tools/zerv_build.py',
                   'src/session/kvcache.zig', 'src/session/checkpoint.zig', 'tests/kvcache.zig',
                   'tests/reference/generate_cache_source_fixture.py', 'tests/fixtures/cache-sources.json']
        manifest['source_sha256'] = {f: build.sha(ROOT / f) for f in sources}
        manifest['commands'].append([str(binary)])
        with (args.output / 'native.log').open('w') as log:
            subprocess.run([str(binary)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        rows = [json.loads(line) for line in (args.output / 'native.log').read_text().splitlines() if line.startswith('{')]
        if len(rows) != 20 or not all(r['exact'] for r in rows):
            raise RuntimeError('missing trials or failed ownership checks')
        (args.output / 'raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary = []
        for depth in (1, 8, 64, 256):
            rs = [r for r in rows if r['depth'] == depth]
            if len(rs) != 5 or {r['trial'] for r in rs} != set(range(5)):
                raise RuntimeError('incomplete trial set')
            ns = [r['elapsed_ns'] / r['iterations'] for r in rs]
            summary.append(dict(depth=depth, n=5, ownership_bytes=rs[0]['ownership_bytes'],
                roundtrip_ns=dict(mean=statistics.mean(ns), stdev=statistics.stdev(ns), min=min(ns), max=max(ns))))
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        manifest['status'] = 'passed'
    except BaseException as e:
        manifest.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        sys.exit('run with tools/py')
    main()
