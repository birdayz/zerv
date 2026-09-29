#!/usr/bin/env python3
"""Local-only pinned upstream InferenceX client versus tiered zerv and tuned llama."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import zerv_build as build
import run_serving as rs
import run_concurrent as rc
import run_multiturn as mt
from check_prompt_equivalence import post
from inferencex_observer import Observer, validate

MODEL = ROOT / 'models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf'
TOKENIZER = ROOT / 'third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
NATIVE = 'zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=8,kv-swap-mib=8192,prefix-cache-tier=host,prefix-cache-disk-dir=third_party/nvme-probe,prefix-cache-disk-mib=8192,prefix-cache-disk-entries=16,prefix-cache-disk-alignment=4096,prefix-cache-disk-chunk-mib=8,prefix-cache-demand=prefetch,prefix-cache-prefetch-chunks=2'
UNTIERED = 'zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=8,kv-swap-mib=0,prefix-cache-tier=off'
ENGINE_SPECS = {'zerv-tiered': NATIVE, 'zerv-untiered': UNTIERED,
                'llama-fa-b512': 'llama-fa-b512', 'rdna3-nofusion': 'rdna3-nofusion'}


def request_sha(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def dump(path, data): path.write_text(json.dumps(data, indent=2) + '\n')


def gate(port, directory):
    results = []
    with Observer(port) as observer:
        for count, ignore in [(64, False), (8, True), (32, True), (64, True)]:
            body = dict(model='qwen3.8-27b', messages=[dict(role='user', content='Reply with exactly OK and nothing else.')],
                        temperature=0, max_completion_tokens=count, ignore_eos=ignore, stream=True,
                        stream_options=dict(include_usage=True), chat_template_kwargs=dict(enable_thinking=False))
            rec = {}
            mt.turn(observer.port, body, rec)
            results.append(rec)
    dump(directory / 'length-gate.json', observer.records)
    assert len(observer.records) == 4
    ordinary = observer.records[0]
    assert ordinary['done'] and ordinary['finishes'] == ['stop'] and ordinary['usage']['completion_tokens'] < 64, ordinary
    for r, count in zip(observer.records[1:], [8, 32, 64]):
        assert validate(r, count), r


def resume_cases(manifest, prior_root, out, inputs, levels, output_tokens, rounds):
    prior = json.loads((prior_root / 'manifest.json').read_text())
    for field in ('model_sha256', 'native_sha256', 'llama_sha256', 'hip_sha256', 'tokenizer'):
        assert manifest[field] == prior[field], f'resume identity mismatch: {field}'
    for field in ('tools/inferencex_client.py', 'bench/inferencex_observer.py', 'bazel/inferencex.bzl', 'requirements_inferencex_lock.txt'):
        assert manifest['sources'][field] == prior['sources'][field], f'resume implementation mismatch: {field}'
    for length in inputs:
        for prefix, suffix in [('workload', 'json'), ('requests', 'json'), ('prompts', 'jsonl')]:
            filename = f'{prefix}-{length}.{suffix}'
            assert (out / filename).read_bytes() == (prior_root / filename).read_bytes(), f'resume workload mismatch: {filename}'
    inherited = []
    for case in prior['cases']:
        if case['status'] != 'passed': continue
        assert case.get('client_command'), 'cannot inherit gate-only cases as measurements'
        assert case['input'] in inputs and case['output'] == output_tokens and case['concurrency'] in levels and 0 <= case['round'] < rounds
        inherited.append(dict(case, path=str(prior_root / case['path']), inherited=True))
    keys = [(c['engine'], c['round'], c['input'], c['concurrency']) for c in inherited]
    assert len(keys) == len(set(keys)), 'duplicate inherited points'
    assert all(c['engine'] in ENGINE_SPECS for c in inherited)
    return inherited


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--inputs', default='1024,8192')
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--prompts', type=int, default=8)
    p.add_argument('--levels', default='1,4')
    p.add_argument('--gate-only', action='store_true')
    p.add_argument('--engines', default='zerv-tiered,llama-fa-b512,rdna3-nofusion', help='comma-separated measured engine labels')
    p.add_argument('--resume-from', type=Path, help='reuse passed points only after checking binaries and identical workload artifacts')
    a = p.parse_args()
    inputs = [int(v) for v in a.inputs.split(',')]
    levels = [int(v) for v in a.levels.split(',')]
    labels = a.engines.split(',')
    assert labels and len(labels) == len(set(labels)) and all(label in ENGINE_SPECS for label in labels)
    assert a.rounds > 0 and a.prompts >= max(levels) and min(levels) > 0
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', argv=sys.argv, started_at=datetime.now(timezone.utc).isoformat(),
                    official=False, publication='No automatic upload/publication; git push requires user approval', engines=labels, cases=[], preparation=[])
    def save(): dump(out / 'manifest.json', manifest)
    save()
    try:
        subprocess.run(build.test_command(), cwd=ROOT, check=True)
        rs.require_host_gpu()
        if rc.gpu_busy(): raise RuntimeError('GPU busy')
        native = build.binary('zerv')
        client = build.binary('inferencex-client')
        # Do not inherit tools/py's RUNFILES_DIR/PYTHONPATH into a different Bazel binary.
        client_env = dict(PATH=os.environ['PATH'], LANG='C.UTF-8')
        capture = build.binary('zerv-prompt-capture')
        table = rs.engines(MODEL, 18098, 24576, native)
        manifest.update(build=build.provenance(), model_sha256=build.sha(MODEL),
                        native_sha256=build.sha(native), llama_sha256=build.sha(rs.llama_server()),
                        hip_sha256=build.sha(rs.RDNA3_BUILD / 'bin/llama-server'),
                        tokenizer={f.name: build.sha(f) for f in TOKENIZER.iterdir() if f.is_file()},
                        sources={str(f.relative_to(ROOT)): build.sha(f) for f in [Path(__file__), ROOT/'bench/inferencex_observer.py', ROOT/'tools/inferencex_client.py', ROOT/'bazel/inferencex.bzl', ROOT/'requirements_inferencex_lock.txt']})
        fixtures = {}
        for length in inputs:
            config = dict(input=length, output=a.output_tokens, prompts=a.prompts, tokenizer=str(TOKENIZER), seed=42)
            config_path = out / f'workload-{length}.json'
            body_path = out / f'requests-{length}.json'
            dump(config_path, config)
            command = [str(client), '--capture-workload', str(config_path), str(body_path)]
            manifest['preparation'].append(command)
            with (out / f'prepare-{length}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=client_env, check=True, timeout=120)
            prompt_path = out / f'prompts-{length}.jsonl'
            command = [str(capture), str(MODEL), str(body_path)]
            manifest['preparation'].append(command)
            with prompt_path.open('w') as f:
                subprocess.run(command, stdout=f, cwd=ROOT, check=True, timeout=120)
            bodies = json.loads(body_path.read_text())
            prompts = list(map(json.loads, prompt_path.read_text().splitlines()))
            assert len(bodies) == len(prompts) == a.prompts
            assert all(len(row['tokens']) + a.output_tokens <= 12288 for row in prompts)
            fixtures[length] = (bodies, prompts)
        if a.resume_from:
            prior_root = a.resume_from.resolve()
            assert not a.gate_only, 'resume is for measured matrices'
            manifest['cases'] = resume_cases(manifest, prior_root, out, inputs, levels, a.output_tokens, a.rounds)
            assert all(c['engine'] in labels for c in manifest['cases']), 'resume engine selection mismatch'
            manifest['resume'] = dict(path=str(prior_root), manifest_sha256=build.sha(prior_root/'manifest.json'), reason='external execution timeout; incomplete points are not reused')
        completed = {(c['engine'], c['round'], c['input'], c['concurrency']): c for c in manifest['cases']}
        save()
        for rnd in range(1 if a.gate_only else a.rounds):
            for label in (labels if rnd % 2 == 0 else labels[::-1]):
                name = ENGINE_SPECS[label]
                points = [(inputs[0], levels[0])] if a.gate_only else [(i, c) for i in inputs for c in levels]
                for length, concurrency in points:
                    spec = dict(rs.resolve_engine(table, name, native))
                    command = list(spec['cmd'])
                    if name.startswith('zerv'):
                        command[command.index('--context') + 1] = '12288'
                        command += ['--host', '127.0.0.1']
                    else:
                        command[command.index('-np') + 1] = '2'
                        command += ['--cache-ram', '8192', '--ctx-checkpoints', '8', '-ctk', 'f16', '-ctv', 'f16']
                    inherited = completed.get((label, rnd, length, concurrency))
                    if inherited:
                        assert inherited['server_command'] == command and inherited['server_env'] == spec['env'], 'resume server configuration mismatch'
                        continue
                    if rc.gpu_busy(): raise RuntimeError('GPU busy before case')
                    directory = out / f'{label}-r{rnd}-i{length}-c{concurrency}'
                    directory.mkdir()
                    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'RADV_'))}
                    env.update(spec['env'])
                    record = dict(engine=label, round=rnd, input=length, output=a.output_tokens, concurrency=concurrency,
                                  server_command=command, server_env=spec['env'], path=str(directory.relative_to(out)), status='running')
                    manifest['cases'].append(record)
                    save()
                    peak = [0]
                    stop = threading.Event()
                    def poll():
                        while not stop.wait(.05): peak[0] = max(peak[0], rs.vram_used() or 0)
                    poller = threading.Thread(target=poll)
                    with (directory / 'server.log').open('w') as log:
                        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        poller.start()
                        try:
                            rs.wait_ready(18098, proc, timeout=900)
                            gate(18098, directory)
                            bodies, prompts = fixtures[length]
                            if not name.startswith('zerv'):
                                reference = []
                                for body, golden in zip(bodies, prompts):
                                    rendered = post(18098, '/apply-template', body)['prompt']
                                    tokens = post(18098, '/tokenize', dict(content=rendered, add_special=False, parse_special=True))['tokens']
                                    reference.append(dict(prompt=rendered, tokens=tokens))
                                    dump(directory/'reference-prompts.json', reference)
                                    assert rendered == golden['prompt'] and tokens == golden['tokens'], 'prompt/token mismatch'
                            record['prompt_gate'] = 'passed'
                            if not a.gate_only:
                                with Observer(18098) as observer:
                                    cmd = [str(client), '--model', str(TOKENIZER), '--tokenizer', str(TOKENIZER),
                                           '--served-model-name', 'qwen3.8-27b', '--backend', 'openai-chat',
                                           '--base-url', f'http://127.0.0.1:{observer.port}', '--endpoint', '/v1/chat/completions',
                                           '--dataset-name', 'random', '--random-input-len', str(length),
                                           '--random-output-len', str(a.output_tokens), '--random-range-ratio', '1.0',
                                           '--num-prompts', str(a.prompts), '--max-concurrency', str(concurrency),
                                           '--request-rate', 'inf', '--ignore-eos', '--seed', '42',
                                           '--num-warmups', str(2 * concurrency), '--percentile-metrics', 'ttft,tpot,itl,e2el',
                                           '--save-result', '--save-detailed', '--result-dir', str(directory), '--result-filename', 'upstream.json']
                                    record['client_command'] = cmd
                                    with (directory/'client.log').open('w') as clientlog:
                                        result = subprocess.run(cmd, cwd=ROOT, env=client_env, stdout=clientlog, stderr=subprocess.STDOUT, timeout=1200)
                                dump(directory/'observer.json', observer.records)
                                result.check_returncode()
                                assert len(observer.records) == a.prompts + 2 * concurrency
                                assert all(validate(r, a.output_tokens) for r in observer.records), 'response length/termination gate failed'
                                measured = observer.records[2 * concurrency:]
                                assert Counter(r['request_sha256'] for r in measured) == Counter(map(request_sha, bodies)), 'workload mismatch'
                                sizes = {request_sha(b): len(g['tokens']) for b, g in zip(bodies, prompts)}
                                assert all(r['usage']['prompt_tokens'] == sizes[r['request_sha256']] for r in observer.records), 'actual prompt length mismatch'
                                upstream = json.loads((directory/'upstream.json').read_text())
                                assert upstream['completed'] == a.prompts and upstream['output_lens'] == [a.output_tokens] * a.prompts
                                record['measured_first_text_ms'] = [1000*(r['text_times'][0]-r['send']) for r in measured]
                                record['actual_prompt_lengths'] = [r['usage']['prompt_tokens'] for r in measured]
                            record['host_memory'] = mt.host_memory(spec, proc.pid)
                            record['vram_peak'] = peak[0]
                            assert 0 < peak[0] <= 24 * 1024**3
                            assert 0 < int(record['host_memory']['values']['VmHWM'].split()[0]) <= 32 * 1024**2
                            record['status'] = 'passed'
                        finally:
                            stop.set()
                            poller.join()
                            proc.terminate()
                            try: proc.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            if spec.get('stop'): subprocess.run(spec['stop'], capture_output=True)
                            save()
                            time.sleep(3)
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    if '/bazel-out/' not in sys.executable: raise SystemExit('Use tools/py bench/run_inferencex.py')
    main()
