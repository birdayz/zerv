"""Generate the predeclared D.2 structured-answer suite; never calls a model."""
import hashlib
import json
from pathlib import Path
import random


def workload():
    rng = random.Random(2026092901)
    conversations = []
    for c in range(4):
        records = [dict(id=f"R{c}{i:03}", quantity=rng.randrange(10, 100),
                        priority=rng.randrange(1, 1000),
                        tag=''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(8)))
                   for i in range(128)]
        lookup = records[17 + c * 23]
        pair = [records[9 + c], records[119 - c]]
        order = [records[4 + c], records[63 + c], records[124 - c]]
        tasks = [
            (f'Return a JSON array containing only the tag of record {lookup["id"]}.', [lookup['tag']]),
            (f'Return a JSON array containing only the integer sum of quantities for {pair[0]["id"]} and {pair[1]["id"]}.', [sum(r['quantity'] for r in pair)]),
            ('Return a JSON array of these three record IDs sorted by ascending priority (break ties by ascending ID): ' + ', '.join(r['id'] for r in order) + '.', [r['id'] for r in sorted(order, key=lambda r: (r['priority'], r['id']))]),
        ]
        tasks = tasks[c % 3:] + tasks[:c % 3]
        system = ('Use only the following records. Answer each question with exactly one JSON array, no prose or Markdown. Quantities and priorities are integers.\n' + '\n'.join(json.dumps(r, separators=(',', ':')) for r in records))
        conversations.append(dict(name=f'records-{c}', system=system,
                                  turns=[q for q, _ in tasks], expected=[a for _, a in tasks],
                                  assistant_history=[json.dumps(a, separators=(',', ':')) for _, a in tasks[:-1]]))
    return dict(version=1, description='Predeclared held-out D.2 record lookup, addition and ordering; strict independent JSON scoring.',
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                temperature=0, seed=1234, max_tokens=64,
                options=dict(chat_template_kwargs=dict(enable_thinking=False)), system='', conversations=conversations)


if __name__ == '__main__':
    path = Path('bench/workloads/multiturn-demand-quality-v1.json')
    path.write_text(json.dumps(workload(), indent=1) + '\n')
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())
