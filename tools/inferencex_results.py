"""Path/environment adapter only; execute pinned upstream processing unchanged."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

FIELDS = frozenset(('RUNNER_TYPE', 'FRAMEWORK', 'PRECISION', 'SPEC_DECODING',
                    'RESULT_FILENAME', 'ISL', 'OSL', 'DISAGG', 'MODEL_PREFIX', 'IMAGE',
                    'TP', 'PP_SIZE', 'EP_SIZE', 'DP_ATTENTION', 'IS_MULTINODE',
                    'REQUIRE_POWER', 'GPU_METRICS_CSV', 'RECIPE_FINGERPRINT'))


def deny_external_actions(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.bind',
                 'subprocess.Popen', 'os.system', 'os.posix_spawn'):
        raise PermissionError(f'offline result processor blocks {event}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    process = sub.add_parser('process')
    process.add_argument('directory', type=Path)
    process.add_argument('metadata', type=Path)
    collect = sub.add_parser('collect')
    collect.add_argument('directory', type=Path)
    collect.add_argument('inputs', type=Path)
    collect.add_argument('name')
    a = p.parse_args()
    directory = a.directory.resolve(strict=True)
    env = json.loads(a.metadata.read_text()) if a.operation == 'process' else {}
    if not isinstance(env, dict) or not set(env) <= FIELDS or not all(isinstance(v, str) for v in env.values()):
        raise ValueError('metadata must contain only declared string environment fields')
    if a.operation == 'process':
        stem = env.get('RESULT_FILENAME', '')
        if not stem or Path(stem).name != stem or stem in ('.', '..'):
            raise ValueError('result filename must be a basename')
        if env.get('DISAGG') != 'false' or env.get('IS_MULTINODE', 'false') != 'false':
            raise ValueError('this local adapter supports the measured single-node path only')
        argv = ['infx.results.fixed_sequence']
    else:
        if not a.name or Path(a.name).name != a.name or a.name in ('.', '..'):
            raise ValueError('collection name must be a basename')
        argv = ['infx.results.collect_results', str(a.inputs.resolve(strict=True)), a.name]
    safe = {k: os.environ[k] for k in ('PATH', 'LANG') if k in os.environ}
    with tempfile.TemporaryDirectory(prefix='zerv-inferencex-results-') as home:
        os.environ.clear()
        os.environ.update(safe, HOME=home)
        os.environ.update(env)
        os.chdir(directory)
        sys.argv = argv
        sys.addaudithook(deny_external_actions)
        runpy.run_module(argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
