import json
from pathlib import Path
root = Path('docs/bench/data/2026-09-29-fixed-history-serving')
rows = [json.loads(line) for line in (root / 'raw.jsonl').read_text().splitlines()]
engines = list(dict.fromkeys(row['engine'] for row in rows))
def key(row): return row['round'], row['level'], row['conversation'], row['turn']
base = {key(row): row for row in rows if row['engine'] == engines[0]}
assert len(base) == 30
result = {}
for engine in engines:
    selected = [row for row in rows if row['engine'] == engine]
    assert len(selected) == 30 and len({key(row) for row in selected}) == 30
    same_output = same_count = 0
    mismatches = []
    for row in selected:
        other = base[key(row)]
        assert row['history_mode'] == 'fixed'
        assert row['request_sha256'] == other['request_sha256'], key(row)
        assert row['usage']['prompt_tokens'] == other['usage']['prompt_tokens'], key(row)
        same_output += row['output_sha256'] == other['output_sha256']
        same_count += row['usage']['completion_tokens'] == other['usage']['completion_tokens']
        if row['output_sha256'] != other['output_sha256']:
            mismatches.append(dict(key=key(row), baseline=other['output_text'], candidate=row['output_text'], baseline_count=other['usage']['completion_tokens'], candidate_count=row['usage']['completion_tokens']))
    result[engine] = dict(identical_requests=30, identical_prompt_counts=30, identical_outputs=same_output, identical_completion_counts=same_count, mismatches=mismatches)
print(json.dumps(result, indent=2))
