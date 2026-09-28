# D.1.1: bounded page-pool preparation ownership

Date **2026-09-28 UTC**, parent `df0e1b4`. Component gates closed; **not yet an
asynchronous GPU→RAM serving feature**. [Pre-code contract](../specs/tiering-preparation.md),
[paper plan](../design/async-tiering.md), [source/alias audit](../design/tiering-preparation-audit.md).

## Implemented scope

The pure model page pool now supports one generation-qualified preparation
transaction, bounded to 1/2/4 moves. It reserves host pages with a distinct,
unpublished owner tag; sources retain their pins. Only an explicit acknowledgment
of drained external ownership permits commit or abort. Submission is not completion.
All validation precedes all mutation. A new cache alias that changes a source pin
count prevents commit; a GPU hit preserves its live slot mask. Commit reports pages
copied separately from pages actually freed and preserves absolute logical indices
for suffix segments. Stale generations cannot complete a later reused reservation.
The synchronous demotion path remains unchanged. No heap allocation in this path.

No GPU command, scheduler selection, source-lease/rename callback or serving flag
is added in this increment. The caller must retain a qualified immutable cache
source; the pool does not replace radix ownership or inspect GPU fences. D.1.2
must implement and verify that adapter before using this primitive in serving.

## Independent oracle and tests

Before native changes, `tests/reference/generate_preparation_fixture.py` produced
**1,362** exact cases using sets of named cache/request owners and reserved host
objects, without importing/executing native code. Outcomes: 932 no-capacity/no-
eligible-source, 162 commits, 219 aborts, 49 late-alias commit conflicts. Fixtures
cover permuted GPU IDs, absolute suffix indices, windows, host limits/headroom,
preexisting aliases, full and partial GPU hits after planning, new cross-root pins,
and cancellation. A partial hit uses a private tail page, so copying four pages
can legitimately free only one. Native tests compare every move, checkpoint page
ID, pin, live mask, host ownership and host logical index with zero tolerance.

- Fixture SHA256: `615362657d1975722e87461e0e1e8cb3bfdc9a12aac3f4e30d34f0bc6ff0818e`.
- Generator SHA256: `e929d3b6b25994e2707d1178af0fec5199753de42ff8c6d205757ce4e41ec970`.
- Both are embedded in ordinary tests; generator integrity is checked, but golden
  regeneration is explicit and separate from builds.

Directed tests additionally cover commit/abort before acknowledgment, rejected
second owner/host resize, invisible reservations, invalid options/maps, duplicate
ack, stale generation after destination reuse, nonwrapping generation exhaustion,
mixed host/GPU repeats and validation failure on the last move before any mutation.
A final directed trace covers initially live pages, hit-then-leave while pending,
and a disjoint synchronous demotion that cannot reuse preparation's destinations;
teardown returns every GPU/host page.

**Executed:** CPU/Python/format **82/82**; pages Debug/ReleaseFast ×20, repeated again
after the additional directed trace. Negative controls both fail in both modes:

1. Clearing the GPU mask during commit fails the independent ownership fixture.
2. Removing the final exclusive-pin validation fails with
   `expected error.InvalidState, found .{ .copied = 1, .freed = 1 }`.

Both mutations were restored before component runs and final tests. Logs:
[gate directory](data/2026-09-28-preparation-ownership/). No unexplained correctness
failure occurred in the intended implementation. No GPU/kernel/driver change or
new GPU test execution in this pure bookkeeping increment; the actual DMA/device
and full-model gates remain mandatory for D.1.2. D.0b's prior gates are not claimed
as evidence for an unimplemented asynchronous demotion adapter.

## Reproducible component measurement

```sh
tools/py tests/reference/generate_preparation_fixture.py
bazel test //...
bazel test //tests:pages //tests:pages_release_fast --runs_per_test=20
tools/py bench/run_preparation.py \
  --output docs/bench/data/2026-09-28-preparation-component-1
tools/py bench/run_preparation.py \
  --output docs/bench/data/2026-09-28-preparation-component-2
```

Pinned Bazel 9.2.0/Zig 0.16.0/Python, ReleaseFast, same host as D.0b; unrelated CPU
activity remains. No GPU job, model load or host configuration change. Manifests
record exact sources, build/toolchain hashes, CPU information, commands and raw
results: [run 1](data/2026-09-28-preparation-component-1/manifest.json),
[run 2](data/2026-09-28-preparation-component-2/manifest.json).

Three shapes `(GPU pages, source pages, host pages)`:
`(192,128,512)`, `(1024,626,1024)`, `(4096,2048,4096)`; the largest is a metadata
capacity case, not a claim it fits this GPU in a particular KV format. Each has
1/2/4-page windows and abort/commit/hit-after-plan modes. Six trials each, first
warmup excluded, **20,000 iterations per trial**. Each run produces 135 measured
trials. Every iteration validates exact moves, copied/freed counts and released
host capacity; final state/invariants are checked. Commit/hit timing includes
ordinary promotion bookkeeping to reset the checkpoint for the next iteration;
hit timing also includes attach/release bookkeeping. **No bytes are copied.**

Selected cells below: microseconds per complete metadata cycle, mean ± sample SD
of five trials. All cells, including window 2, are retained in raw summaries.

| source pages | window | mode | run 1 µs | run 2 µs |
|---|---:|---|---:|---:|
| 128 | 1 | abort | 0.615 ± 0.005 | 0.614 ± 0.001 |
| 128 | 1 | commit/reset | 0.886 ± 0.006 | 0.875 ± 0.011 |
| 128 | 1 | hit/commit/reset | 1.174 ± 0.006 | 1.149 ± 0.003 |
| 128 | 4 | commit/reset | 0.996 ± 0.133 | 0.820 ± 0.011 |
| 626 | 1 | abort | 1.368 ± 0.007 | 1.356 ± 0.005 |
| 626 | 1 | commit/reset | 2.486 ± 0.007 | 2.466 ± 0.005 |
| 626 | 1 | hit/commit/reset | 3.660 ± 0.020 | 3.647 ± 0.019 |
| 626 | 4 | commit/reset | 2.753 ± 0.021 | 2.752 ± 0.017 |
| 2048 | 1 | abort | 5.066 ± 0.077 | 5.526 ± 0.023 |
| 2048 | 1 | commit/reset | 8.970 ± 0.019 | 9.878 ± 0.039 |
| 2048 | 1 | hit/commit/reset | 13.301 ± 0.040 | 13.719 ± 0.558 |
| 2048 | 4 | commit/reset | 9.134 ± 0.349 | 8.990 ± 0.015 |

These are repeatable metadata costs, not copy bandwidth, hardware overlap, or a
speedup over another engine. Scans of the full checkpoint/host capacity and reset
work contribute to the larger-shape cost; do not extrapolate a 1-page copying
latency from this table. There is no semantically equivalent llama-server primitive
for our exact hybrid radix transaction. The runnable serving milestone remains
[D.0b's measured comparison](2026-09-28-tiering-window.md), with disk still slower.

## Next gate

D.1.2: source-generation/serial validation and atomic radix finish, separate GPU
command ownership, read priority and host-only pressure policy. Resolve the joint
cache/device oracle before implementing that integration; then run real bytes,
full-vocabulary/model/device gates and tuned serving comparisons. D.2 prefetch and
the full faster-or-on-par objective remain open.
