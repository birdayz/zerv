"""CPU startup parsing checks; absent model prevents any GPU initialization."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import zerv_build as build
binary = build.binary('zerv')
base = [str(binary), '--model', str(ROOT / 'third_party/nonexistent-preparation-test.gguf'), '--parallel', '2', '--spec-draft', '0']
assert not Path(base[2]).exists()
valid = ['--prefix-cache-prepare-pages', '32']
cases = [(['--prefix-cache-prepare-window-pages', '1'], 'InvalidArguments'), (['--prefix-cache-prepare-host-headroom-mib', '1'], 'InvalidArguments')]
for flags in [['--parallel', '1'], ['--kv-pool', 'static'], ['--prefix-cache', 'flat'], ['--prefix-cache-tier', 'off'], ['--spec-draft', '1'], ['--prefix-cache-slots', '0'], ['--kv-swap-mib', '0']]:
    cases.append((valid + flags, 'InvalidArguments'))
for window in [0, 3, 8, 4294967295]:
    cases.append((valid + ['--prefix-cache-prepare-window-pages', str(window)], 'InvalidArguments'))
for window in [1, 2, 4]:
    cases.append((valid + ['--prefix-cache-prepare-window-pages', str(window), '--prefix-cache-prepare-host-headroom-mib', '0'], 'FileNotFound'))
results = []
for flags, expected in cases:
    run = subprocess.run(base + flags, cwd=ROOT, text=True, capture_output=True, timeout=10)
    output = run.stdout + run.stderr
    assert run.returncode and f'error: {expected}' in output, (flags, output)
    results.append(dict(command=base + flags, expected=expected, returncode=run.returncode, output=output))
(Path(__file__).parent / 'cli-results.json').write_text(json.dumps(dict(binary_sha256=build.sha(binary), results=results), indent=2) + '\n')
print(f'{len(results)} CPU startup cases pass')
