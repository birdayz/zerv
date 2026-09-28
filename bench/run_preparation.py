"""Preparation metadata transaction: five trials after warmup, no copies or serving."""
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
        binary = build.binary('zerv-preparation-bench')
        m.update(build=build.provenance(), binary_sha256=build.sha(binary))
        sources = ['bench/preparation.zig', 'bench/run_preparation.py', 'tools/zerv_build.py',
                   'src/model/pages.zig', 'tests/pages.zig',
                   'tests/reference/generate_preparation_fixture.py', 'tests/fixtures/preparation.json']
        m['source_sha256'] = {f: build.sha(ROOT/f) for f in sources}
        m['commands'].append([str(binary)])
        with (args.output/'native.log').open('w') as log:
            subprocess.run([str(binary)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        rows = [json.loads(s) for s in (args.output/'native.log').read_text().splitlines() if s.startswith('{')]
        if len(rows) != 135 or not all(r['exact'] for r in rows):
            raise RuntimeError('missing/failed component trials')
        summary = []
        for count in [128, 626, 2048]:
            for window in [1, 2, 4]:
                for mode in ['abort', 'commit', 'hit']:
                    trials = [r for r in rows if r['source_pages'] == count and r['window'] == window and r['mode'] == mode]
                    if len(trials) != 5 or {r['trial'] for r in trials} != set(range(5)):
                        raise RuntimeError('incomplete trials')
                    values = [r['elapsed_ns']/r['iterations'] for r in trials]
                    summary.append(dict(source_pages=count, window=window, mode=mode,
                                        mean_ns=statistics.mean(values), sample_sd_ns=statistics.stdev(values)))
        (args.output/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        m['status'] = 'passed'
    except BaseException as e:
        m.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')


if __name__ == "__main__":
    if "/bazel-out/" not in sys.executable:
        sys.exit('run with tools/py')
    main()
