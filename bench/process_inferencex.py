#!/usr/bin/env python3
"""Run unchanged upstream processor/collector on original completed client artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import zerv_build as build
from summarize_inferencex import ENGINES, check_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source, out = a.source.resolve(), a.output.resolve()
    previous = json.loads((source/'manifest.json').read_text())
    assert previous['status'] == 'passed'
    check_matrix(previous['cases'], previous.get('engines', ENGINES))
    expected_count = len(previous['cases'])
    # Upstream searches this fallback even when an explicit CSV path is supplied.
    # Refuse unrelated telemetry instead of changing the processor's behavior.
    if Path('/workspace/gpu_metrics.csv').exists():
        raise RuntimeError('unrelated upstream telemetry fallback exists')
    out.mkdir(parents=True, exist_ok=False)
    collection = out/'collection-inputs'
    collection.mkdir()
    manifest = dict(status='running', source=str(source), source_manifest_sha256=sha(source/'manifest.json'),
                    revision='f437f7bfd164422036b0de7e3818f8afb5bc70d7', argv=sys.argv,
                    publication='No automatic upload/publication; git push requires user approval', cases=[],
                    adapter_sha256=sha(Path(__file__)), wrapper_sha256=sha(ROOT/'tools/inferencex_results.py'),
                    source_rule_sha256=sha(ROOT/'bazel/inferencex.bzl'))
    dump(out/'manifest.json', manifest)
    try:
        executable = build.binary('inferencex-results')
        manifest['executable'] = str(executable)
        manifest['executable_sha256'] = sha(executable)
        env = dict(PATH=os.environ['PATH'], LANG='C.UTF-8')
        identities = {'zerv-tiered': ('zerv', 'native_sha256'),
                      'zerv-untiered': ('zerv', 'native_sha256'),
                      'llama-fa-b512': ('llama.cpp-vulkan', 'llama_sha256'),
                      'rdna3-nofusion': ('llama.cpp-hip-rdna3', 'hip_sha256')}
        for c in previous['cases']:
            original = source/c['path']
            if not original.exists():
                original = source.parent/original.parent.name/original.name
            original = original/'upstream.json'
            name = f"{c['engine']}-r{c['round']}-i{c['input']}-c{c['concurrency']}"
            directory = out/name
            directory.mkdir()
            input_sha = sha(original)
            shutil.copyfile(original, directory/'upstream.json')
            framework, identity = identities[c['engine']]
            metadata = dict(RUNNER_TYPE='RX-7900-XTX', FRAMEWORK=framework, PRECISION='Q4_0;KV=f16',
                            SPEC_DECODING='none', RESULT_FILENAME='upstream', ISL=str(c['input']),
                            OSL=str(c['output']), DISAGG='false', MODEL_PREFIX='qwen3.8-27b',
                            IMAGE='binary-sha256:'+previous[identity], TP='1', PP_SIZE='1', EP_SIZE='1',
                            DP_ATTENTION='false', IS_MULTINODE='false', REQUIRE_POWER='false',
                            GPU_METRICS_CSV='gpu_metrics.csv')
            metadata['RECIPE_FINGERPRINT'] = hashlib.sha256(json.dumps(
                dict(command=c['server_command'], env=c['server_env']), sort_keys=True).encode()).hexdigest()
            dump(directory/'metadata.json', metadata)
            command = [str(executable), 'process', str(directory), str(directory/'metadata.json')]
            rec = dict(name=name, input_sha256=input_sha, source=str(original), metadata=metadata,
                       command=command, status='running')
            manifest['cases'].append(rec)
            dump(out/'manifest.json', manifest)
            with (directory/'processor.log').open('w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            rec['exit_code'] = result.returncode
            result.check_returncode()
            assert sha(original) == sha(directory/'upstream.json') == input_sha, 'input changed'
            emitted = directory/'agg_upstream.json'
            # Validate, never change, the upstream output (including its power failure).
            data = json.loads(emitted.read_text())
            assert type(data['power_valid']) is int and data['power_valid'] == 0
            assert 'telemetry_file_missing' in data['power_invalid_reasons']
            rec.update(status='passed', result_sha256=sha(emitted),
                       audit_sha256=sha(directory/'power_validation_upstream.json'))
            shutil.copyfile(emitted, collection/f'{name}.json')
            assert sha(collection/f'{name}.json') == rec['result_sha256']
            dump(out/'manifest.json', manifest)
        command = [str(executable), 'collect', str(out), str(collection), 'inferencex-local']
        manifest['collector_command'] = command
        with (out/'collector.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        manifest['collector_exit_code'] = result.returncode
        result.check_returncode()
        combined = out/'agg_inferencex-local.json'
        emitted_rows = json.loads(combined.read_text())
        expected_rows = [json.loads(path.read_text()) for path in collection.glob('*.json')]
        canonical = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
        assert len(emitted_rows) == expected_count and canonical(emitted_rows) == canonical(expected_rows)
        manifest.update(status='passed', collected_sha256=sha(combined))
        print(f'Processed and collected {expected_count} unchanged upstream results: {combined}')
        print('Power remains invalid: telemetry_file_missing (upstream behavior, not suppressed).')
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        dump(out/'manifest.json', manifest)


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable:
        raise SystemExit('Use tools/py bench/process_inferencex.py')
    main()
