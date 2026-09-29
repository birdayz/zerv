# Proactive GPU→RAM preparation (D.1)

2026-09-28 UTC, pre-code at `df0e1b4`. [Paper review](../research/2026-09-28-kv-tier-papers.md)
(Pensieve §4.3/5, LMCache §5), [native audit](../design/tiering-preparation-audit.md).
D.0b is closed. D.1.1's pure ownership component is now implemented and verified
([report](../bench/2026-09-28-preparation-ownership.md)); D.1.2 integration is not
implemented or yet ready for code. This is not a completed asynchronous serving feature.

## Ordered scope and resolved ownership

D.1.1 first implements the **pure page-pool reservation transaction** (`src/model`),
with its independent object-set oracle and metadata measurement. It does not submit
GPU work or select cache victims. D.1.2 then integrates qualified radix leases,
atomic rename, an explicit GPU command owner and scheduler policy. No production
flag until that integration is implemented and measured. D.2 queued-demand
protection/prefetch remains separate. This division avoids combining unverified
ownership accounting with DMA/scheduling changes.

Native review at `df0e1b4`: `Pool.demotePlan/demoteCommit/demoteAbort`,
`Radix.take/segment/rename/acquireSource/releaseSource`, model
`copyPages/attachPrefix/flushRun`, and `ModelBackend.noteFailure/pollMaintenance`.
Existing synchronous demotion uses checkpoint-tagged host reservations and assumes
pins stay exactly one. That is not a safe asynchronous commit protocol: a divergent
root inserted during copying can add another owning pin without increasing the
selected node's path-ref count. A GPU hit can add live slot masks; those are valid
and must survive. Preserve the synchronous path unchanged in D.1.1.

## D.1.1 contract: one bounded preparation owner per pool

No new tensor mathematics, precision or byte layout. Pure accounting moves the
checkpoint pin, not live slot ownership. The existing GPU and host physical pages
retain their logical positions. No heap allocation: at most four move records in
the pool, one transaction at a time, a nonwrapping u64 generation. Supported window
sizes are 1/2/4 pages; this bounds bookkeeping, not eventual copy latency.

`prepare(list, first, window, host_headroom) !?u64`:

- Caller retains an immutable qualified cache source through commit/abort. `list`
  is the whole checkpoint in logical order, and `first` is the start of its owned
  segment. An ancestor's segment is not eligible. Validate the whole list, bounds,
  headroom and window before reserving anything. Error or null leaves everything
  unchanged. Active preparation rejects a second owner.
- Select in increasing logical order only GPU pages with exactly one checkpoint
  pin and no live mask. Skip host/shared/live pages. Reserve lowest free host pages,
  no more than the window or `hostFree - host_headroom`. Headroom includes existing
  reservations. No capacity returns null, not a partially started hidden job.
- A **distinct preparation host-owner tag** makes reserved destinations invisible
  to checkpoint reads/releases and unavailable to live swap/other demotions. Source
  pins stay intact. Save absolute logical indices and physical source/destination
  IDs in the owned move list; do not use suffix-relative indices at commit.
- A successful nonempty transaction increments its generation without wrapping.
  `preparedMoves(generation)` borrows that immutable list until finish/abort.

`ackPreparation(generation)` records that the external device owner has drained
**all** references. Submission is not acknowledgment. Before acknowledgment,
commit/abort reject with `PendingPreparation`, changing nothing. A failed submit
with demonstrably no borrowed span may be acknowledged then aborted; unknown GPU
ownership must fail stop rather than call this function. This pure component cannot
inspect fences; the D.1.2 adapter must establish that precondition.

`commitPreparation(generation, list) !{copied, freed}`:

- Require current generation, nonempty acknowledged transaction and original list
  length. Validate **every move before any mutation**: same source ID at its saved
  absolute index, valid GPU/source logical index, pins still exactly one, and exact
  host destination still owned by preparation. Any changed alias/pin/list fails
  atomically, leaving the acknowledged transaction available for abort.
- Transfer only that cache pin to the host page, store its original logical index,
  publish its host page ID and change host owner to checkpoint. Never clear a live
  GPU mask. `copied` counts moves; `freed` counts moved GPU pages with zero masks
  at commit, not a before/after global free-count delta.
- Successful finish makes the transaction idle; stale generation operations cannot
  acknowledge, commit or abort a later reservation, even if IDs are reused.

`abortPreparation(generation)` validates all destination ownership first, then
releases only these destinations after acknowledgment. No source page ID, pin or
mask changes. A late source alias is a valid reason to abort, not a reason to clear
its pins. Duplicate finish/abort, stale generation and generation exhaustion reject.
Initialization/host resizing while owned is forbidden by the caller's pool lifetime
contract; `setHost` must explicitly reject resizing with a preparation outstanding.

All methods are scheduler-thread-only; no locks/atomic ownership invented here.
A pool with a pending preparation cannot be destroyed/reinitialized. The external
source lease is mandatory: calling ordinary checkpoint unpin on these source pages
would violate it. D.1.1 is not permission to run the unchanged synchronous cache
callbacks concurrently without a retained source lease.

## Independent executable D.1.1 oracle, before native code

`tests/reference/generate_preparation_fixture.py` uses Python sets of named cache
and live-request owners, distinct reserved host objects and immutable logical
positions. It does not import or execute native code. Deterministic directed and
seeded cases use permuted GPU IDs, suffix starts, all supported windows, host
headroom/full capacity, preexisting aliases, GPU full/partial hits after planning,
new independent cache aliases, completion and abort. A partial hit gives its tail
a separate physical GPU page; copied-vs-freed must therefore differ correctly.
Fixtures contain exact move triples and final pins, masks, host ownership/logical
indices and checkpoint IDs. Compare native results field-for-field with zero
tolerance. Additional native lifecycle tests check no mutation before ack, delayed
cancel, invalid lists/options, stale generations after destination reuse, and
all-or-nothing validation on a late move failure. Keep the generator hash in the
fixture and test it; ordinary tests never regenerate expected values.

Executed before native code: **1,362 cases**, fixture SHA256
`615362657d1975722e87461e0e1e8cb3bfdc9a12aac3f4e30d34f0bc6ff0818e`,
generator SHA256 `e929d3b6b25994e2707d1178af0fec5199753de42ff8c6d205757ce4e41ec970`.
The post-code fixture comparison, lifecycle tests and negative controls pass;
measured metadata costs and exact commands are in the linked report.

Acceptance: full CPU/Python/format suite, pages Debug/ReleaseFast repetitions,
negative mutation (clearing a live mask or removing final exclusive-pin validation
must fail), repeatable component timings for 1/2/4-page windows and representative
source/host capacities, at least five measured trials after warmup. Benchmark exact
results inside the measured loop; report metadata-only cost, not transfer bandwidth
or a serving speedup. Existing synchronous demotion remains covered. GPU/serving
gates belong to D.1.2 once it actually submits GPU work.

## D.1.2 integration requirements (not implementation-ready yet)

Use the pool transaction only while holding the exact radix generation/serial and
its ancestors. At commit the selected node's own path refs must equal its sole
preparation lease, and the pool's pins==1 check must pass; these jointly exclude
leased descendants and newly inserted cross-root aliases. Atomic cache-owned
finish must validate, commit, rename affected descendants and release the source
without yielding or an unprotected release/commit gap. Internal-node candidates
need a separate API; do not broaden C.3's leaf-only archive-candidate contract.

A separate GPU command owner records bounded copies without selecting a request
slot or touching its logits. Keep compute→transfer, transfer→transfer,
transfer→host and later-consumer ordering; account for partial-page tails and all
KV groups. Existing `flushRun` hard-wires `swap_commands`, so cannot be reused
unchanged. The adapter must explicitly acknowledge only after fence completion,
retain reservations on cancellation, count its known owner in `noteFailure`, and
fail stop on lost/unknown ownership. Start with optional archive/preparation lanes
serialized because `swap_kv.mapped` guards the whole buffer. Foreground reads must
outrank optional issue; owned completion/drain must still run. Host-only preparation
must not accidentally require a disk owner.

Resolve headroom/selection policy, narrow cache callback APIs, queue/hit/stop
interleavings and an executable joint cache/device oracle before D.1.2 code. Then
real interleaved bytes, 257/80k full-vocabulary gates, independent FP64/libllama,
CPU/GPU/spill/host checks and repeated loaded serving versus host-only and tuned
Vulkan/HIP references. No hardware overlap or full D.1 completion claim follows
from this pure ownership component.

## D.1.2 integration contract (resolved at `efd667d`, before code)

### Cache boundary and joint oracle

Add a separate radix `preparationCandidate(index)` returning handle, LRU stamp and
absolute own-segment start for an unleased, tier-enabled GPU-owning segment. Internal
nodes are eligible; flat/tier-off caches return none. Keep archive candidates
leaf-only. Acquire the existing immutable source lease before pool planning.

`finishPreparation(lease, commit)` validates the live generation/serial, then
requires exactly one reference on the selected node (its own lease). Otherwise
return `Busy` without calling commit. The narrow callback receives the complete
mutable checkpoint list and its own-segment start; it must be non-yielding,
non-reentrant and either atomically commit the already-drained pool transaction
or return an error with no mutation. Snapshot the old own segment in existing
bounded scratch, invoke the callback, rename aliases/descendants, update demotion
statistics and release the lease in one non-yielding transition. Success consumes
the lease; **every error retains it**. Caller aborts drained pool reservations and
then releases that lease. Neither cache generation nor pool generation alone is
sufficient. No raw mutable source slice is returned to an asynchronous caller.

Before implementing this API, generate an independent prefix/object-set fixture
using exact token tuples as full-page identities and per-checkpoint partial tails.
Recompute parent relationships and segment-owning sets from prefix inclusion, not
native parent/pin arrays. Cases combine ancestors, descendants, divergent roots,
partial tails, insertion after planning, late ancestor/sibling/descendant leases,
GPU hits, cancellation and injected commit failure. Expected own-segment starts,
move counts, copied/freed counts, commit/refusal, per-node host/GPU residency and
lease refs must match the real pool/cache driven through `tests/kv_system.zig`.
Include the cross-root insertion race where own-path refs remain one but a source
pin becomes two. Simulated byte copies must preserve all live/cache content and
return all pages at teardown. Existing source/pool goldens remain mandatory.

### Device ownership and scheduler integration

A `serve` preparation owner has a separate `gpu.Commands`, borrowed model/cache,
bounded qualified source/pool IDs, a cancel latch, monotonic submission timestamp
and fixed counters. No heap allocation per preparation. Model submission records
at most four physical page copies across all KV groups, no request selection or
logits/I/O writes. Require no live packed chunk or verification at start. Use
compute→transfer and transfer→transfer ordering before copies; transfer→host,
transfer→transfer and transfer→compute dependencies after. Full physical pages use
the unchanged f16/f32 host-tier layout: selected pages initially have no slot masks,
and subsequent partial hits copy their tail privately. Do not freeze recurrent
state again: its existing cache snapshot remains protected by the source lease.

Submit without waiting. Poll the actual fence with zero timeout. On success only,
acknowledge the pool transaction. Errors with known drained ownership abort and
release; pending ownership or a deadline exceeded after one final completion check
fails stop. Device loss remains fatal. `noteFailure` must include this exact pending
command owner, not just the archive's owner, so an unrelated invalid request does
not falsely kill the engine. Unknown pending commands remain fatal.

An in-chunk maintenance call may only poll/acknowledge an existing preparation; it
must **not commit, select or rename**. Ordinary maintenance finishes or aborts it
after the chunk. Stop first aborts the pack, then drains before releasing anything.
Serialize optional archive capture and preparation until both release their sources,
including acknowledged-but-unpublished preparation. Whole-buffer `swap_kv.mapped`
guards are never bypassed. Foreground disk reads may proceed over disjoint reserved
pages; no new optional preparation while reads are pending. An owned transfer always
drains even when new issue is forbidden. A request's cancellation does not cancel
an independent cache preparation; stop does. Generation reuse tests remain mandatory.

When preparation owns potential reclaimable pages, urgent allocation returns the
existing `CacheReclaimPending` protocol rather than prematurely failing decode or
starting another synchronous eviction on that source. Finish/abort advances the
scheduler admission epoch even if a live hit prevented actual reclamation; the
retry can then use normal drop/swap fallbacks. Unrelated runnable rows continue.

### Policy and configuration

Opt-in `--prefix-cache-prepare-pages N` sets desired free GPU pages; zero/default
preserves current behavior. `--prefix-cache-prepare-window-pages 1|2|4` (default 1)
and `--prefix-cache-prepare-host-headroom-mib N` are valid only with preparation.
Resolve before execution: shared KV, parallel >1, tier-enabled radix cache, nonzero
snapshots and host store, no MTP; N cannot exceed GPU page capacity. Host reserve
rounds up to physical host pages; default is min(256 MiB, one eighth of host store),
and cannot exceed host capacity. Explicit invalid combinations reject at startup.
Report effective page thresholds/window/host reserve; no inert knobs.

Below the GPU free-page target, with host reserve available, try cold unleased
segments in LRU order, bounded by snapshot capacity. Pool planning is authoritative
about exclusive pins and live masks. Skip no-move candidates without leasing them
indefinitely. Start one window only; finish and re-evaluate pressure. Generation-
qualified suppression after a failed/conflicted optional transaction prevents busy
retry on that incarnation. No optional host eviction just to create preparation
room: live swap/read reserve outranks it. Host-only configurations work without disk.
Archive pressure handles snapshot-slot capacity, which GPU demotion cannot free.

Counters distinguish submitted/copied pages, committed pages, actually freed pages,
abort/conflict counts and source hold time. They must not present live-hit copies as
reclamation. Window size is an explicit latency/headroom tradeoff, not a default
performance claim. No second Vulkan queue or hardware-overlap claim.

### Integration acceptance

Joint oracle and delayed ownership/cancel/stop/lease tests; CPU suite and repeated
interfaces; GPU Debug/ReleaseFast/spill and host-driver gates. Extend the real model
checker to submit preparation over a cached source, make another slot hit it while
pending, run packed/independent computation, acknowledge then commit outside the
pack, and compare valid bytes, live continuation and restored full vocabulary.
Exercise host-only and host+disk, abort/retry, wrong-slot error during owned DMA,
80k mixed ancestry and partial tails, read priority and retained reservations.
Independent FP64/libllama semantics remain required. Repeat loaded HTTP with fixed
native streams/token counts, off/on preparation and tuned Vulkan/HIP references;
report negative results, memory, transfers, TTFT/gaps, throughput and variance.
Do not close D.1.2 on the CPU oracle or component alone. D.2 is still separate.

### Integration-discovered swap ordering regression (2026-09-29, before fix)

Repeated real-pool scheduler tests stalled without any preparation owner. Watchdog
snapshots show an old swapped decoder, three newer resident partial prompts and
~29,000 admission epochs in ten seconds. Observed with flat/tier-off and radix/tier-on.
The scheduler excludes a swapped slot whose swap-in already failed this epoch when
selecting the oldest. After one partial prompt is swapped out, the old decoder still
cannot fit; a newer, smaller swapped prompt is then restored into exactly the space
just freed. This cycles without accumulating enough space for the oldest decoder.

Required ordering: choose the oldest swapped waiter independent of retry eligibility.
If it already failed this epoch, time-slice another resident victim instead of letting
a younger swapped waiter consume the reclaimed pages. Preserve older-prompt priority,
epoch-gated retries and the existing minimum victim residency interval. No arithmetic,
precision or GPU layout changes. Deterministic acceptance: a 9-page swapped decoder
behind resident 4/4/2-page partial prompts in a 10-page pool must complete a row after
multiple victims leave; a smaller younger swap-in must not bypass it. Run this test
against the unfixed selection as a negative control, then both modes and repeated
real-pool tests. Preserve the original timeout/watchdog logs. This is a newly observed
integration regression, not a reopening of the retired GPU hang investigation.
