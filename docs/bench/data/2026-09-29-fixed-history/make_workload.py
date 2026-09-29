import hashlib
import json
from pathlib import Path
source = Path('bench/workloads/multiturn-distinct-v1.json')
w = json.loads(source.read_text())
w['description'] = 'Fixed-history control of multiturn-distinct-v1; canonical prior assistant text prevents engine output drift from changing subsequent inputs. Not a quality-equivalence assertion.'
w['parent_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
for conv in w['conversations']:
    conv['assistant_history'] = ['The project implements model serving in Zig. Correctness is checked against independent reference calculations, and performance claims require repeated benchmarks.'] * (len(conv['turns']) - 1)
Path('bench/workloads/multiturn-fixed-history-v1.json').write_text(json.dumps(w, indent=1) + '\n')
