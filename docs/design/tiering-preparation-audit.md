# Proactive preparation follow-up (D): questions exposed by C integration

2026-09-28, research/design only. Begun during C.3 verification; C.3's evaluation
gates are now closed with negative performance findings. D's pre-code research is
active. Subsequent D.0a cadence code and verification are specified in
[tiering-progress.md](../specs/tiering-progress.md); D.1/D.2 remain unimplemented.
References: [paper review](../research/2026-09-28-kv-tier-papers.md),
[ordered plan](async-tiering.md), [C source contract](../specs/cache-source-leases.md).

## Hypotheses to measure, not findings

The C.3 maintenance hook currently progresses one bounded archive quantum at a
safe scheduler boundary. At 1 MiB per quantum, busy 20 ms decode boundaries imply
an approximate **50 MiB/s issue-rate ceiling** even if the disk/DMA is much faster;
long packed chunks can give fewer boundaries. This is an estimate, not measured
serving throughput. CPU-only snapshot quanta also consume turns. Idle polling can
run faster. Record actual source hold times, bytes written, completion/cancellation
counts and loaded request timings before choosing a bigger issue window.

A cold GPU source remains pinned until disk drain; a tiny hot cache/24k KV pool may
need that capacity before a full image completes. C.3 cancels optional persistence
rather than failing/blocking live inference indefinitely. That may correctly avoid
latency while preserving few disk records. Check the serving logs: do not assume
that removing write-through alone achieves the user's speed goal.

## Native ownership details D must resolve before coding

- `Pool.demotePlan` reserves destination host pages while retaining GPU source
  pins. An asynchronous adapter needs explicit start/poll/commit/abort ownership,
  including source stability and destination reservations through cancellation.
  A prospective preparation owner can use a cache lease, but C.1 forbids renaming
  leased ancestry. Release/commit/cache-rename must therefore be one serialized,
  validated transition, not an externally observable unprotected interval.
- A GPU-only cache hit is allowed during a source lease. It can attach a formerly
  reclaimable GPU page between plan and completion. Determine whether completing
  demotion may retain that page for the live request while moving only the cache's
  pin to host. Measure **actual freed pages**, not pages copied; preparation is not
  synonymous with reclamation. Audit plan/commit assumptions and test this race.
- Partial-page valid bytes are immutable, but appended tail bytes are not. Transfer
  reads need the C.2 ordering dependencies; state restoration must still ignore
  unused tail contents or canonicalize them deliberately. No arithmetic change.
- `Buffer.mapped` guards whole buffers. Pending demotion of one range of `swap_kv`
  can forbid a CPU archive copy of another range. Do not bypass the guard with a raw
  pointer. Initially serializing the two optional owners may be preferable to a
  new range-ownership framework, but foreground reads/swaps and cancellation must
  still make progress. Verify all dependencies explicitly.
- Proactive GPU preparation must begin before hard allocation failure, while host
  space is available. Headroom must account for pending reservations and source
  leases; do not silently move the old synchronous eviction callback earlier.
- Queued-demand prefetch must outlive neither the queued request's token buffer nor
  its generation. A request can disconnect or reuse its scheduler slot while a
  prefetch is in flight. Reuse the proven deferred-release recheck and qualified
  ownership. Do not reserve the entire live request slot merely to preserve cache.
- Snapshot-slot exhaustion is separate from host-page exhaustion. GPU→RAM copying
  cannot free a recurrent snapshot slot; C's disk preservation/drop still handles
  that dimension. A device snapshot freeze would consume extra VRAM and must earn
  its cost in a separate measured comparison.

## Required D evidence

An independent plan/commit/abort model before implementation, burst demand with
partial completion, concurrent GPU hits, queued cancellation/reuse, read priority,
reserved capacity, exact 257/80k state and logits. Repeated issue-window/loaded
serving measurements against tuned compatible competitors, with fixed resources,
quality and outputs. Multiple Vulkan queues are a separate investigation requiring
cross-queue memory dependencies and measured copy/compute overlap; no such overlap
is established by C or by this note.

## C.3 smoke observation and next measurement candidates

The first C.3 two-snapshot/tier-off smoke (not a repeated comparison) is in
`docs/bench/data/2026-09-28-pressure-smoke/`: eight exact native responses and
matching prompt/generated token counts, 69.969 s wall, **zero restores**, one
completed write, 326,107,136 total written bytes, three cancellations. Sources
were held 48.267 s total / 29.161 s maximum; 305 CPU and only nine GPU quanta.
The exact same output baseline is used, so reduced write traffic must not be
presented as a win when it simply yields no useful restored prefixes.

Source review found a stronger scheduling limit than the estimate above:
`Batcher.pollBackground` defers whenever `pack != null`, just like foreground
`pollIo`. The smoke executed 156 prefill chunks / 2,496 units over about 70 s.
Thus a nominal 1 MiB per *decode* boundary estimate does not predict issue rate
while long prefills dominate. Repeated serving measurements are still running;
this is an observed negative smoke, not an isolated causal experiment.

D should explicitly evaluate two bounded alternatives before picking defaults:

1. Larger archive chunk/staging windows (e.g. 1/2/4/8 MiB per ticket), with at
   least two physically reserved read tickets, explicit RAM/rounding overhead,
   cancellation drain time and token-gap regressions. Existing tickets are 1 MiB;
   changing only an iteration count does not increase a pending GPU quantum.
2. Progress **already-owned immutable source capture** between prefill units,
   while keeping unsafe foreground restore/publication and new mutations at
   their specified boundaries. Background source copy does not select a request
   slot or modify its logits/`io`; foreground `archiveRestored` does, which is
   why removing the common guard indiscriminately would be unsafe. Audit every
   CPU mapping/barrier and pending-buffer interaction first. Validate full state
   and vocabulary after interleaved segmented/packed prefill, including partial
   source tails, cancellation and stop. C.2's interleaved decode gate alone does
   not establish segmented-prefill safety.

These are proposals for D, not implemented performance fixes. Proactive demotion
and demand prefetch remain necessary separate ownership transitions; adjusting
an archive quantum must not be mislabeled implementation of the full plan.

## Additional queue/lifetime audit (still design only)

`Cache.restore` intentionally returns a cold miss when a host-containing path is
leased: promotion would rename bytes retained by the source. This is correct for
C.1, but persisting a soon-needed host prefix can therefore force recomputation.
D's demand protection must cover hot host prefixes, not just NVMe reads. Compare
cancel/drain then promote against ordinary cold fallback; never mutate a leased
source to improve hit statistics.

The scheduler mutex alone does not make a queued prompt safe to pass through an
unlocked maintenance callback. A waiting client can cancel and free its token
buffer before that callback finishes. Candidate approaches to specify/test:

- Resolve queue hints to generation-qualified cache handles/record leases while
  the prompt is still protected, and pass only bounded metadata outside the lock.
- Or explicitly retain the request generation and token lifetime using the same
  running/deferred-release protocol as real operations. Holding only its slot
  number is insufficient.

Do not allocate/copy a context-sized token list each scheduling turn. Current disk
records already own token arrays; a leased record can retain its own immutable
key. `Archive.advanceWith` currently combines disk-read acknowledgment, digest
verification and GPU upload. A future staging-only queued read needs an explicit
no-upload phase and protected handoff to an admitted target; pretending a request
slot exists early would break `archiveMap`/`archiveRestored`. A bounded first-chunk
lookahead is not the same feature/performance claim as whole-image host prefetch.

## Proposed D.1 demotion transaction (API design, not yet an implementation contract)

Source review: `src/model/pages.zig:demotePlan/demoteCommit/demoteAbort`,
`src/model/runtime.zig:demotePages/copyPages`, and
`src/session/kvcache.zig:segment/rename/acquireSource/restore/take` at C.3's parent
`68fb7bc` plus the pressure working tree. No external runtime dependency.

Observed native details:

- `demotePlan` reserves host pages with `checkpoint_owner`; it selects GPU pages
  with exactly one cache pin and zero slot mask. Reservations already reduce
  `hostFree()`, but no cache lookup names them yet.
- `demoteCommit` clears the cache pin, preserves `logical[from]` in host metadata,
  and changes the cache's page ID. It does **not** clear a live slot mask. Thus a
  GPU hit after planning can retain that GPU page for its request; such a copy
  is not actual reclamation. Validate/count this explicitly in asynchronous code.
- GPU hits attach full prefix pages read-only; a partial page is privately copied
  before append. The initial plan requires no slot owner, but a subsequent hit is
  a valid interleaving, not grounds to free its live page.
- Radix demotion changes only a node's **own** segment; `rename` propagates changed
  page IDs into descendants. C.2 fixed host logical indices for suffix segments.
  Do not redo that bug using move-list indices as logical positions.

Candidate design to turn into a spec/oracle before implementation:

1. One preparation transaction in the optional maintenance lane, serialized with
   archive capture initially. A separate command owner is required: foreground
   swap already uses `swap_commands` and cannot reset it while a background copy
   is pending. Allocate bounded move/old-ID storage at init, not per token.
2. Resolve GPU free-page headroom and a small copy window at startup. A logical
   page here is about 8 MiB with f16 KV; four pages already copy about 32 MiB.
   Benchmark window/latency, not only transferred bytes. Host reservations must
   leave configured incoming/read headroom, with checked capacity arithmetic.
3. Select cold, unleased GPU-owning segments, including internal nodes. The C.3
   archive candidate API is intentionally leaf-only and should not silently change
   meaning. Query `reclaimable` to exclude current request-owned pages before
   acquiring the generation/serial-qualified source lease.
4. Plan and submit one bounded page-copy quantum. Keep source pins, its lease,
   destination reservations and command-buffer ownership until acknowledgment.
   Failure before submit rolls back immediately; cancel after submit drains first.
   Unknown GPU ownership remains fail-stop. No raw mapped-buffer guard bypass.
5. At completion, validate the exact source incarnation, serial, old IDs and host
   reservations before any mutation. Another descendant source lease would observe
   renamed IDs: require the selected node's path-ref count to contain only this
   preparation lease before committing its own segment. An overlapping descendant
   lease should abort optional preparation after drain, not invalidate the other
   source. Sibling leases that share an unchanged ancestor need not block it.
6. Commit model page bookkeeping and radix rename in **one scheduler-owned,
   non-yielding transition**, then release the lease. Public callers must not gain
   an unguarded mutable page slice. A narrow cache-owned finish callback can pass
   its own segment to a prevalidated model commit and then rename descendants;
   it must not fail after partially publishing IDs. Abort changes neither cache
   IDs nor snapshot incarnation. Do not expose a release-then-later-commit gap.
7. Count copied pages and actual newly free GPU pages separately. GPU hits acquired
   during the transfer retain their masks; old bytes become reusable only when no
   live mask/pin remains. Host pages become visible only after completed transfer.
   Snapshot-slot capacity is unchanged.
8. Foreground reads outrank new preparation. Urgent GPU admission waits only for
   the current bounded owner to drain, then resumes on an epoch change. Prefer a
   safe completed demotion that actually frees pages over canceling it blindly;
   ordinary hard-capacity drop/live-swap rules remain the fallback.

Dependency audit still required in the functional spec: source buffers need
compute/transfer ordering, host destinations need transfer→host visibility, and
future consumers need appropriate transfer→compute ordering. Foreground swaps use
separate physical pages because reserved host IDs and pinned GPU IDs remain
unavailable. Whole-buffer CPU mappings of `swap_kv` still forbid simultaneous
archive CPU capture; serializing the optional lane avoids that alias for now.

Required executable oracle before code: independent object-set page ownership
with GPU slot owners, cache owners, pending host reservations and qualified cache
handles. Directed + randomized start/poll/hit/leave/commit/cancel/failure/reuse,
including hit-after-plan (copied != freed), shared ancestors, descendant leases,
logical suffix indices, bounded partial windows and full host capacity. Then real
GPU bytes and model 257/80k state/vocabulary with interleaved request hits, plus
repeated loaded serving. A declaration that the operation is asynchronous is not
an overlap measurement. D.2 queued-demand protection/prefetch is separate, still
unresolved as outlined above; no D implementation readiness/completion is claimed.

## D.0a readiness decision (2026-09-28, before code)

At `0b252be`, reviewed `Batcher.pollBackground/pollIo/run`,
`ModelBackend.pollMaintenance`, `Model.archiveSourceSubmit/archiveQuantum`,
`Model.prefillPackedSegment`, and the archive model checker. Source-only capture
never touches packed live state or selects a slot. Both scheduler and model currently
reject it throughout a chunk; changing only one guard cannot make progress.
The source's existing four ordering barriers and whole-buffer mapped checks remain
necessary, including for invalid partial tails. Ordinary maintenance can discard or
select new objects, and foreground restore can publish/select a slot, so neither
is safe to enable wholesale. Decision: a separate opt-in in-chunk hook, restricted
to advancing an already leased source, plus a snapshot-only model exemption.
No new mathematics/layout/precision is introduced. The complete narrowed contract,
existing independent anchors and new executable interleaving gates are in
[the D.0a spec](../specs/tiering-progress.md). Keep tickets at 1 MiB to isolate cadence;
window tuning is not part of this first change. D.1/D.2 questions above do not block
this strictly read-only source operation and remain unresolved for their own code.

## D.1 follow-up source review during D.0a serving measurement (no D.1 code)

Re-read `Pool.demotePlan/demoteCommit/demoteAbort/checkCheckpoint`, radix
`segment/rename/makeRoom/acquireSource/releaseSource`, and model
`copyPages/flushRun/attachPrefix` at `0b252be` (unchanged by D.0a).

- Host reservations use `checkpoint_owner`, the same tag as committed cache pages;
  there is no reservation generation in the current pool. Cache IDs do not expose
  reserved destinations, so ordinary callers cannot release them, but a future
  delayed finish must retain its exact move list/transaction serial and validate
  old IDs, host ownership, GPU pins and logical indices before any commit. Existing
  `demoteCommit` is unchecked because its caller currently waits synchronously.
- `reclaimable` counts unmapped GPU pages, not exclusively pinned pages. It is only
  a candidate hint; `demotePlan` additionally requires `pins == 1`. Do not equate the
  hint with either reservable moves or actual freed pages.
- `rename` scans every live list for matching IDs at the same logical indices, not
  merely immediate children. Selected own-segment refcount == one (its own lease)
  must be proven sufficient for every renamed alias by the independent tree oracle,
  including insertion/deduplication during the pending transaction. Do not assert
  that only the directly selected node changes.
- `flushRun`/`extendRun` currently hard-wire `self.swap_commands`. A separate
  preparation owner cannot reuse these helpers unchanged. Pass the explicit command
  owner to a small copy-recording helper, with no altered synchronous semantics, or
  use a separate bounded recorder. Never reset foreground swap commands in flight.
- `attachPrefix` shares full pages but copies a partial page before append. With no
  source slot owner at planning, later GPU hits can therefore retain source pages
  without writing their valid bytes; commit must preserve their slot masks. The
  copied/freed distinction is observable, not merely defensive bookkeeping.
- Proactive host preparation should not accidentally require disk to be enabled:
  current `pollMaintenance` returns immediately without a disk owner. Its future
  optional lane needs an explicit host-only lifetime/configuration too. Archive and
  demotion remain serialized initially because of whole-buffer CPU mapping guards.

These findings refine D.1's pre-code work only. No new native D.1 API, fixture,
implementation or performance result is claimed.

## D.1 critical alias case found during D.0b measurement (pre-code)

`Radix.take` deduplicates against the longest **common token prefix**, even if
neither checkpoint is an ancestor of the other. Two root nodes `S/A` and `S/B`
can therefore own separate pins on the same leading full GPU pages without a
materialized `S` checkpoint. These are not necessarily descendant aliases.

An asynchronous demotion can plan `S/A` while pins==1, then a newly inserted
`S/B` raises those pins to two. `S/A`'s path refs can still equal its sole lease.
Calling current unchecked `demoteCommit` would incorrectly set pins to zero;
`rename` would also rewrite the other root. **Generation/serial and path-ref
validation alone are insufficient.** Before committing any move, require its GPU
source still has exactly the one selected owning pin and the expected old ID and
logical index. A changed pin count aborts the optional transaction after fence
ack; preserve all cache IDs/pins and free only its reserved destinations.
Full GPU hits that add only a slot mask remain valid and are not this race.

This refines the required independent object-set oracle: include divergent roots
with common full pages inserted after planning, leased and unleased siblings,
and partial pages copied rather than shared. Initial pins==1 avoids pre-existing
cross-root alias moves; finish-time pins==1 plus the selected-path lease/ref check
must guard newly inserted aliases too. No D.1 code is started.

D.1 error-path audit: `ModelBackend.noteFailure` currently permits exactly the
pending archive command owner (`device.pending == int(d.commands.pending)`). A new
preparation command must be counted explicitly while it owns a fence; otherwise an
unrelated rejected request operation can falsely mark the whole device fatal.
Count only actual acknowledged/owned command states, not an optimistic outstanding
job count. Retain the existing invalid-slot-while-copy-pending model test and extend
it to preparation. Unknown pending ownership and device loss must still fail stop.

## D.1.1 resolved accounting prerequisite

At `df0e1b4`, the pre-code [contract](../specs/tiering-preparation.md) and independent
named-owner fixture (1,362 cases) precede the pure page-pool transaction. It is now
implemented/verified ([report](../bench/2026-09-28-preparation-ownership.md)). One
pool-owned nonwrapping generation and at most four moves; distinct unpublished
host reservation tag; explicit external-drain acknowledgment; validate every move
before mutation, including pins still exactly one. GPU masks survive, copied and
freed counts differ correctly, and absolute logical suffix indices are retained.
A stale owner cannot release a later reservation even when host IDs are reused.

This deliberately does not submit DMA or acquire cache sources. The upcoming
D.1.2 adapter must own **both** the pool transaction generation and the radix
handle/lease serial. Neither identifier substitutes for the other. It must retain
the exact cache source until GPU acknowledgment and atomic cache-owned finish;
only then may it call the pool commit and rename affected lists without yielding.
No release-then-commit gap, no acknowledgment of a merely submitted copy, and no
use of this prerequisite as evidence that the asynchronous serving path exists.
