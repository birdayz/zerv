#!/usr/bin/env python3
"""Pressure-selector component: five trials after warmup, exact rotating LRU, no I/O."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import zerv_build as build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    m = dict(started=datetime.now(timezone.utc).isoformat(), argv=sys.argv, uname=list(os.uname()),
             cpuinfo=Path('/proc/cpuinfo').read_text(), status='running', commands=[], scope=__doc__)
    try:
        cmd = build.test_command()
        m['commands'].append(cmd)
        with (args.output/'tests.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build.binary('zerv-pressure-bench')
        m.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = ['bench/pressure.zig', 'bench/run_pressure.py', 'tools/zerv_build.py',
                   'src/session/pressure.zig', 'src/session/kvcache.zig', 'tests/kvcache.zig',
                   'tests/reference/generate_tiering_pressure_fixture.py', 'tests/fixtures/tiering-pressure.json']
        m['source_sha256'] = {f: build.sha(ROOT/f) for f in sources}
        m['commands'].append([str(binary)])
        with (args.output/'native.log').open('w') as log:
            subprocess.run([str(binary)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        rows = [json.loads(s) for s in (args.output/'native.log').read_text().splitlines() if s.startswith('{')]
        if len(rows) != 40 or not all(r['exact'] for r in rows):
            raise RuntimeError('missing/failed component trials')
        summary = []
        for count in [1, 8, 64, 256]:
            for host in [False, True]:
                trials = [r for r in rows if r['candidates'] == count and r['host_only'] == host]
                if len(trials) != 5 or {r['trial'] for r in trials} != set(range(5)):
                    raise RuntimeError('incomplete trials')
                values = [r['elapsed_ns']/r['iterations'] for r in trials]
                summary.append(dict(candidates=count, host_only=host, mean_ns=statistics.mean(values), sample_sd_ns=statistics.stdev(values)))
        (args.output/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        m['status'] = 'passed'
    except BaseException as e:
        m.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        sys.exit('run with tools/py')
    main()
