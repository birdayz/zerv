"""Run the unmodified reviewed InferenceX client, offline except loopback HTTP."""
import os
import runpy
import sys
import tempfile


def local_network_only(event, args):
    if event == 'socket.connect':
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
            raise PermissionError(f'benchmark blocks non-loopback connection: {address!r}')
    if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise PermissionError('benchmark blocks external DNS')
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
        raise PermissionError('benchmark client may not launch processes')


def main():
    # No proxy, HF credentials, uploader config, cloud keys or inherited API credentials.
    safe = {k: os.environ[k] for k in ('PATH', 'LANG') if k in os.environ}
    with tempfile.TemporaryDirectory(prefix='zerv-inferencex-home-') as home:
        os.environ.clear()
        os.environ.update(safe)
        os.environ.update(HOME=home, HF_HOME=home, HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                          HF_HUB_DISABLE_XET='1', USE_TORCH='0', USE_TF='0', USE_FLAX='0',
                          TOKENIZERS_PARALLELISM='false', OPENAI_API_KEY='local-benchmark')
        if '--trust-remote-code' in sys.argv or '--profile' in sys.argv:
            raise SystemExit('remote code and profiler endpoints are disabled')
        sys.addaudithook(local_network_only)
        if len(sys.argv) == 4 and sys.argv[1] == '--capture-workload':
            import json
            from pathlib import Path
            import numpy as np
            from infx.bench_serving.benchmark_serving import get_tokenizer, sample_random_requests
            config = json.loads(Path(sys.argv[2]).read_text())
            tokenizer = get_tokenizer(config['tokenizer'], trust_remote_code=False, local_files_only=True)
            np.random.seed(config['seed'])
            requests = sample_random_requests(0, config['input'], config['output'], config['prompts'], 1.0, tokenizer, num_workers=1)
            bodies = [dict(model='qwen3.8-27b', messages=[dict(role='user', content=p)], temperature=0.0,
                           max_completion_tokens=n, stream=True, stream_options=dict(include_usage=True), ignore_eos=True)
                      for p, _, n, _ in requests]
            Path(sys.argv[3]).write_text(json.dumps(bodies))
            return
        # Serial workload generation avoids subprocesses; this is outside timed serving.
        sys.argv += ['--random-num-workers', '1']
        runpy.run_module('infx.bench_serving.benchmark_serving', run_name='__main__')


if __name__ == '__main__':
    main()
