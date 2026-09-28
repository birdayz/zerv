"""CPU-only startup rejection/acceptance gate; nonexistent model prevents GPU opening."""
import json
from pathlib import Path
import subprocess
import sys

if '/bazel-out/' not in sys.executable:
    raise SystemExit('run with tools/py')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
import zerv_build as build
binary = build.binary('zerv')
base = [str(binary), '--model', str(ROOT/'third_party/nonexistent-window-test.gguf'), '--parallel', '2']
assert not Path(base[2]).exists()
disk = ['--prefix-cache-disk-dir', str(ROOT/'third_party/nvme-probe'), '--prefix-cache-disk-mib', '8']
cases = [(['--prefix-cache-disk-chunk-mib', '1'], 'InvalidArguments')]
cases += [(disk + ['--prefix-cache-disk-chunk-mib', str(c)], 'InvalidOptions') for c in [0, 3, 16, 4294967295]]
cases += [(disk[:-1] + ['9', '--prefix-cache-disk-chunk-mib', '8'], 'InvalidOptions')]
cases += [(disk + ['--prefix-cache-disk-chunk-mib', str(c)], 'FileNotFound') for c in [1, 2, 4, 8]]
results = []
for flags, want in cases:
    result = subprocess.run(base+flags, cwd=ROOT, capture_output=True, text=True, timeout=10)
    output = result.stdout+result.stderr
    assert result.returncode != 0 and f'error: {want}' in output, (flags, result.returncode, output)
    results.append(dict(command=base+flags, expected_error=want, returncode=result.returncode, output=output))
(Path(__file__).parent/'cli-results.json').write_text(json.dumps(dict(binary_sha256=build.sha(binary), results=results), indent=2)+'\n')
print('10 startup cases pass; no model or GPU initialized')
