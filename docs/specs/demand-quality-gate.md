# D.2 held-out structured-answer gate

2026-09-29; predeclared before fixture generation or model execution. Supporting
verification within D.2, not a new production feature. Existing paper-derived plan:
`docs/design/async-tiering.md`; direct rationale: `competitive-fixed-history.md`.
Different generated prose cannot establish equivalent answer quality from hashes.
Use objective tasks with an independent, executable answer computation instead.

## Workload and scoring contract

Four novel synthetic record tables, 128 records each, deterministic seed2026092901.
Each record has a unique ID, integer quantity and priority, and a synthetic tag.
Three requests per table: exact tag lookup, addition of quantities from two named
records, priority-ordering of three named records (ID ascending breaks ties).
Rotate task order across conversations; targets span the table. Expected answers
are JSON arrays calculated by Python directly from records, independent of inference.
Record all fixture/generator hashes. No prompt or scoring changes after viewing outputs.
No candidate filtering; every generated case belongs to the gate.

Only the table/instructions/questions are sent. Golden answers are used as canonical
assistant history for later turns on every engine; the answer to the current task is
not included in the current request. Greedy, thinking disabled, max_tokens64; raw
output must parse as exactly one JSON array matching the golden in value AND type.
Whitespace is allowed; fences, prose, bool-as-int, floats-as-int, extra elements,
wrong ordering, truncation, errors or missing responses fail. No repair or extraction.
Scorer unit tests must prove each rejection and valid whitespace handling.

Require every response correct (100%) for every engine, round, concurrency and turn.
Report individual failures, never average them away or discard a prompt. Require
identical request hashes across engines at each round/level/conversation/turn, complete
unique response coverage, and valid stop termination. Native policy off/on also needs
identical raw response hashes and usage counts. Competitor whitespace/tokenization of
answers can differ: this is scored-answer equivalence, NOT identical generated work.
Scope is only this structured retrieval/arithmetic/order suite, not general quality.

## Resources / performance acceptance

Same GGUF, parallel2, context12288 per slot, f16KV, same card/driver, no MTP. Native
192pages/eight snapshots/4096MiB host swap/8192MiB disk/16records/8MiB chunks/two
prefetch tickets. Reference8192MiB cache RAM/eight checkpoints; tuned Vulkan b512
and HIP nofusion b2048, unchanged previous tuning. Native join0 and join128.
Common hardware ceiling24GiB VRAM,32GiB server-process host high-water RSS,8GiB
scratch capacity. RSS/VRAM peaks must be reported and checked; allocation strategies
are different, not equal cache capacity. No system resource changes or new downloads.

Warmup, phased C1/C4, three alternating-order rounds, fresh server each round/engine;
use existing `run_multiturn.py`. No concurrent GPU diagnostics. Report full wall,
TTFT by turn, tail token gaps, output counts, host/GPU peaks, all trial values and SD.
Predeclared scoped parity criterion: quality passes AND mean wall and mean turn-wise
TTFT p50 each <=1.05 times the fastest qualified competitor for BOTH C1 and C4.
Three rounds are an initial gate, not strong statistical proof. High variance or
regressions require follow-up; even passing does not close full-plan acceptance.

## Later-turn diagnostic

Analyze all three prior own-answer rounds, not just the outlier: cached tokens,
per-request latency, disk-read bytes, cache misses, queue/preemption counters. Separate
observed correlations from causal evidence. Any production remedy needs a new
pre-code contract and independent test. Keep join default0 until broader benefits
are demonstrated; do not optimize against held-out quality answers.
