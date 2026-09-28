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
