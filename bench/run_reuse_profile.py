#!/usr/bin/env python3
"""Profile the first fixed-history reuse case; not a serving or competitor benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import zerv_build as build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(status='running', argv=sys.argv, commands=[])
    try:
        native = [json.loads(line) for line in a.capture.read_text().splitlines()]
        assert native[0]['prompt'].startswith('<|im_start|>')
        fixture = a.output / 'tokens.json'
        fixture.write_text(json.dumps(dict(first=native[0]['tokens'], reuse=native[1]['tokens'], boundary=native[0]['tokens'][0])))
        for command in [build.test_command(), build.host_gpu_test_command()]:
            m['commands'].append(command)
            subprocess.run(command, cwd=ROOT, check=True)
        binary = build.binary('zerv-reuse-profile')
        model = ROOT / 'models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf'
        m.update(build=build.provenance(), binary_sha256=build.sha(binary), model_sha256=build.sha(model), capture_sha256=build.sha(a.capture), fixture_sha256=build.sha(fixture), sources={f: build.sha(ROOT / f) for f in ['bench/reuse_profile.zig', 'bench/run_reuse_profile.py']})
        command = [str(binary), str(model), str(fixture.resolve())]
        m['commands'].append(command)
        with (a.output / 'native.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=build.host_vulkan_env(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        rows = [json.loads(line) for line in (a.output / 'native.log').read_text().splitlines() if line.startswith('{')]
        assert len(rows) == 5 and all(row['exact'] for row in rows)
        assert all(row['times']['restored'] == 7764 and row['times']['checkpoint_tokens'] == 7809 for row in rows)
        (a.output / 'summary.json').write_text(json.dumps({metric: dict(mean=statistics.mean(row['times'][metric] for row in rows), sd=statistics.stdev(row['times'][metric] for row in rows)) for metric in ['begin_ns', 'segment_ns', 'checkpoint_ns', 'final_ns']}, indent=2))
        m['status'] = 'passed'
    except BaseException as e:
        m.update(status='failed', error=repr(e))
        raise
    finally:
        (a.output / 'manifest.json').write_text(json.dumps(m, indent=2))


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable: sys.exit('run with tools/py')
    main()
