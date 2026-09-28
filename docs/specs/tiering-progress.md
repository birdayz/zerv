# Bounded archive progress during segmented prefill (D.0a)

2026-09-28. Specification before implementation, based on C.3 commit `0b252be`.
This is the first prerequisite of [D](../design/async-tiering.md), **not** proactive
GPU demotion or queued-demand prefetch. [Research and ownership audit](../design/tiering-preparation-audit.md).

## Decision and scope

Permit one existing immutable-source maintenance quantum between completed prefill
units. Keep the existing 1 MiB tickets, eight-ticket staging budget, two reserved
read tickets, single GPU command owner and pressure policy unchanged. Larger windows
are a subsequent measured increment, not part of this causal cadence experiment.
Mooncake's source pins and LMCache's bounded start/wait ownership motivate the
lifetime contract; neither paper implements our packed hybrid-model scheduler.
Pensieve/CachedAttention motivate subsequent preparation/demand work, not an excuse
to perform unsafe cache mutations during a packed chunk.

## Interface and state transitions

`Batcher` may call optional `pollMaintenanceInChunk(stopping, reads_pending)` only
between completed units when its pack is live. Backends without the explicit hook
retain whole-chunk deferral. The ordinary maintenance callback and all foreground
restore/publication callbacks remain forbidden during a pack. A hook runs outside
the scheduler mutex; the existing deferred old-slot-release recheck applies after
it exactly as after ordinary maintenance. Stop aborts the pack before draining.

The model backend's in-chunk hook may only advance or drain an **already acquired**
immutable source. No source means a no-op: no selection, acquisition, discard,
promotion, demotion, slot selection, restore publication or new checkpoint. It
shares the ordinary running-source path, including latched cancellation, failed
incarnation suppression, release exactly once and read priority. A pending read
forbids issuing new optional transfer/disk work but does not forbid acknowledgment
or drain. A successful drain can release its lease and wake blocked admission;
ready-backed cache discard waits until an ordinary safe boundary.

The model's archive guard may exempt only a non-null immutable snapshot source
from `chunk != null`. Mutable-slot exports and every import retain that guard;
all paths retain the speculative-verification guard. `archiveSourceSubmit` still
validates snapshot identity and every pinned logical page. The cache owns the
qualified source and all ancestors until command and disk drain. Valid partial-page
bytes are immutable; unused tail bytes are canonicalized by C.2 after completion.
No model arithmetic, precision, stream format or restore behavior changes.

## Buffer/dependency audit and bounds

A source quantum reads only `snapshot_store`, `swap_kv`, and leased `kv` pages,
and writes its independent imported staging ticket. It does not select a slot or
write model `io`, activations, recurrent live state, `pack_logits`, page tables or
chunk counters. Packed prefill may write different pages or the invalid tail of a
shared final page; leases protect valid bytes. Existing same-queue compute→transfer,
transfer→transfer, transfer→host and transfer→compute dependencies must remain.
CPU-only copies still use whole-buffer `mapped()` ownership checks; never bypass
them. Commands are separate from prefill commands and retained through fence ack.
There is no new queue, hardware-overlap claim, allocation, polling loop, or larger
transfer. One scheduler callback still advances at most one archive quantum.

## Executable correctness mechanism and acceptance (before code)

There is no external implementation of this engine-specific scheduling interface.
Use the independent history function in `tests/batcher.zig` to compare each
sequence's outputs with its solo token history. Extend its delayed fake backend
with an explicit safe hook, while retaining hard failures if ordinary maintenance
or foreground I/O runs in a pack. Cover segmented and multi-member packed execution,
pending reads, mid-callback stop/cancellation, old-slot release/reuse, no-source
no-op, and three-poll cancellation drain. Require actual in-chunk progress counts;
an output-only test could pass by retaining the old deferral.

Extend `bench/archive_model_check.zig` / `bench/run_archive_model.py` with explicit
`--prefill` mode (requires pressure/disk path). Before capturing the tested prefix,
compute two independent 128-token solo-prefill vocabulary rows. While a mixed
host/GPU immutable source is held, replay those prompts as a two-member segmented
pack, advancing production in-chunk maintenance between units. Require exact
vocabulary bytes, actual new GPU source submissions while a chunk exists, and
explicit rejection of mutable archive exports/imports during the chunk. Then run
the existing poisoned/permuted full archived-state comparison and four continuation
rows at 257 and 80,000 tokens, cancel/retry and last-chunk corruption fallback.
The original pressure mode remains a cadence-isolated component counterfactual.

Independent numerical/format anchors remain mandatory: C.2's 60-layout/4,804-window
coordinate-based byte golden, C.1/C.3 object-set ownership fixtures, and the pinned
FP64/libllama model oracle (337 rows, modes 0/512, existing tolerances). Native solo
comparison isolates scheduling changes but does not replace those independent
anchors. The mechanisms/fixtures already execute at `0b252be`; no golden is to be
regenerated from the modified implementation.

Run CPU/Python/fmt, repeated batcher/archive tests in both modes, GPU/spill and host
GPU tests; a negative control restoring whole-chunk deferral must fail the new
progress gate. Preserve failure logs. Benchmark repeated component runs and fresh
serialized zero-idle HTTP rounds with identical native outputs/resources versus
C.3 and tuned Vulkan/RDNA3 nofusion llama-server. Report retention, write/restore
counts, cancellations, TTFT, token gaps, throughput and memory, including regressions.
This increment can close with a measured negative result; the user's faster-or-on-par
full-plan goal cannot. D.0 window tuning, D.1 asynchronous demotion and D.2 demand
protection/prefetch remain separately gated work.

### Stop correction exposed by the in-chunk gate (before corrective code)

The new stop/reuse trace reproduced a scheduler timeout in both modes: stop drained
running I/O and aborted the pack, but never completed a newly queued, non-running
`begin`. Such a request owns no backend borrow and must be completed with `Canceled`
before run returns. Preserve running I/O until ack; clear queued token references,
notify waiters, and drain any resulting old-slot releases before exit. Add a minimal
queued-operation stop test independent of the optional hook. The original failure
and diagnostic logs are retained under `docs/bench/data/2026-09-28-tiering-progress/`.
An existing test's plain shared `restored += 1` also lost an increment under concurrent
clients (expected 3, found 2, while backend restores was 3); use independent per-client
counters and sum only after join. This is a test data race, not lost production hits.

### Matched component counterfactual

The `--pressure` diagnostic does not execute the extra packed workload, so its wall
clock is not a cadence-only comparison. Add `--prefill-cadence unit|chunk|both`
(default unit, chunk/both require `--prefill`). Chunk mode executes the **same**
two-member 128-token prefill, but defers source progress until the pack ends. It
must have zero in-chunk quanta and exact rows/state; unit mode must have positive
in-chunk quanta. `both` alternates mode order for successive prefix cases, writes
separate results/commands, and includes one warmup pair before five measured pairs.
This is a diagnostic-only counterfactual, not a new production configuration.
Write time includes the same packed computation in both modes. Ordinary serving
comparison remains mandatory; this small component cannot prove serving speed.
