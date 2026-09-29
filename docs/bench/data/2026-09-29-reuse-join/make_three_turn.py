import hashlib
import json
from pathlib import Path
source = Path('bench/workloads/multiturn-distinct-v1.json')
w = json.loads(source.read_text())
w['parent_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
w['description'] = 'Three-turn own-answer extension of multiturn-distinct-v1; exercises future reuse after optional intermediate checkpoint suppression. Preserve original first two turns.'
for conv in w['conversations']:
    conv['turns'].append('State the most important correctness requirement from these documents in one sentence.')
Path('bench/workloads/multiturn-reuse-three-v1.json').write_text(json.dumps(w, indent=1) + '\n')
