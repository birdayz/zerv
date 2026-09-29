#!/usr/bin/env python3
"""Capture full production-path prompt IDs against both recorded serving competitors."""
import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import zerv_build as build
import run_serving as rs
import run_concurrent as rc


def post(port, path, body):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    try:
        conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        data = response.read()
        if response.status != 200: raise RuntimeError((path, response.status, data[:500]))
        return json.loads(data)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serving', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.serving / 'manifest.json').read_text())
    workload_path = Path(next(iter(source['workload'])))
    assert build.sha(workload_path) == source['workload'][str(workload_path)]
    w = json.loads(workload_path.read_text())
    bodies, keys = [], []
    for conv in w['conversations'][:max(source['levels'])]:
        messages = [dict(role='system', content=conv.get('system', w['system']))]
        for turn, text in enumerate(conv['turns']):
            messages.append(dict(role='user', content=text))
            bodies.append(dict(model='qwen3.8-27b', messages=copy.deepcopy(messages), stream=True,
                               stream_options=dict(include_usage=True), max_tokens=w['max_tokens'],
                               temperature=w['temperature'], seed=w['seed'], **w['options']))
            keys.append((conv['name'], turn))
            if turn < len(conv['assistant_history']):
                messages.append(dict(role='assistant', content=conv['assistant_history'][turn]))
    fixture = args.output / 'requests.json'
    fixture.write_text(json.dumps(bodies))
    m = dict(status='running', argv=sys.argv, serving_manifest_sha256=build.sha(args.serving / 'manifest.json'),
             workload_sha256=build.sha(workload_path), request_sha256=build.sha(fixture), engines={})
    try:
        subprocess.run(build.test_command(), cwd=ROOT, check=True)
        rs.require_host_gpu()
        binary = build.binary('zerv-prompt-capture')
        model = Path('models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf').resolve()
        assert build.sha(model) == source['model_sha256']
        m.update(model_sha256=build.sha(model), binary_sha256=build.sha(binary), build=build.provenance(),
                 sources={str(p): build.sha(p) for p in [Path(__file__), Path('tools/prompt_capture.zig')]})
        command = [str(binary), str(model), str(fixture.resolve())]
        m['native_command'] = command
        with (args.output / 'native.jsonl').open('w') as output:
            subprocess.run(command, cwd=ROOT, stdout=output, check=True)
        native = [json.loads(line) for line in (args.output / 'native.jsonl').read_text().splitlines()]
        assert len(native) == len(bodies)
        raw = [json.loads(line) for line in (args.serving / 'raw.jsonl').read_text().splitlines()]
        for index, (conv, turn) in enumerate(keys):
            request_hash = hashlib.sha256(json.dumps(bodies[index], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            records = [r for r in raw if r['conversation'] == conv and r['turn'] == turn]
            assert records and all(r['request_sha256'] == request_hash and r['usage']['prompt_tokens'] == len(native[index]['tokens']) for r in records)
        for name in ('llama-fa-b512', 'rdna3-nofusion'):
            if rc.gpu_busy(): raise RuntimeError('GPU is busy')
            spec = source['engines'][name]
            command = spec['cmd']
            if name.startswith('llama'):
                assert build.sha(Path(command[0])) == source['llama_server_sha256']
            else:
                m['hip_binary_sha256'] = build.sha(rs.RDNA3_BUILD / 'bin/llama-server')
                assert m['hip_binary_sha256'] == '590c6cb61c27eae36aad6a9c2154e8c2b478d01c123d434b54648544c7229d15'  # pinned verified competitor
            port = int(command[command.index('--port') + 1])
            env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'RADV_'))}
            env.update(spec['env'])
            m['engines'][name] = dict(cmd=command, env=spec['env'])
            with (args.output / f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    rs.wait_ready(port, process, timeout=900)
                    captured = []
                    for index, body in enumerate(bodies):
                        prompt = post(port, '/apply-template', body)['prompt']
                        tokens = post(port, '/tokenize', dict(content=prompt, add_special=False, parse_special=True))['tokens']
                        captured.append(dict(index=index, prompt=prompt, tokens=tokens))
                    (args.output / f'{name}.json').write_text(json.dumps(captured))
                    assert captured == native, f'{name}: prompt/token mismatch; captures retained'
                    m['engines'][name]['exact_cases'] = len(captured)
                finally:
                    if command[0] == 'docker':
                        subprocess.run(['docker', 'rm', '-f', command[command.index('--name') + 1]], check=False)
                    else:
                        process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
        m['status'] = 'passed'
    except BaseException as e:
        m.update(status='failed', error=repr(e))
        raise
    finally:
        (args.output / 'manifest.json').write_text(json.dumps(m, indent=2))


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable: sys.exit('run with tools/py')
    main()
