# Pressure-driven archive admission (C.3, specification before implementation)

2026-09-28. Specified before implementation; C.2 gates closed in `68fb7bc`.
C.3 correctness/component/serving evaluation gates are closed; immediate-turn disk
preservation remains ineffective and no speedup is claimed. [Results](../bench/2026-09-28-tiering-pressure.md),
[integration audit](../design/tiering-policy-integration.md),
[paper/code research and pinned ledger](../research/2026-09-28-kv-tier-papers.md),
[source ownership](cache-source-leases.md), [source bytes](cache-source-bytes.md).

## Scope and decisions

Single scheduler, one optional immutable-source write, existing bounded full-image
format and exact f16/f32 state. Production checkpoint creation saves a hot snapshot
but does not itself submit NVMe I/O or return PendingIo for persistence. Foreground
restores remain request-owned. Disk remains opt-in, process-local scratch, not a
persistent store. Require radix/shared KV, at least two request slots, snapshots,
and no MTP; reject unsupported options explicitly. Keep the old live-slot write
entry point only for diagnostic regression tests, not the production checkpoint
callback.

No implicit new memory allocation. Physical snapshots and host KV buffers keep
existing budgets. An in-flight source and all its ancestors remain occupied until
C.2 completion/drain. Host capacity includes live swap, but the only preservation
and discard candidates are optional cache entries. Full-image bytes are not the
same as uniquely reclaimable host bytes; never sum ancestor pages once per child.

## Capacity and selection

Expose implemented disk headroom controls for snapshot slots and host KV MiB.
Resolve these to slots/pages once at startup, with checked arithmetic. Automatic
slot headroom is one when capacity >=3 and zero for one/two-slot caches. Explicit
headroom must be less than physical capacity. Automatic host headroom is
`min(ceil(256 MiB / bytes_per_logical_page), floor(host_page_capacity / 4))`;
zero host capacity disables the byte trigger. Explicit MiB resolves by ceiling
division to whole pages and must be less than host page capacity (only zero is
accepted when no host KV exists). These are initial untuned defaults, not paper
constants. Log the effective resolved values. A headroom of zero still triggers at exhaustion;
it does not mean unlimited memory or a disabled hard-capacity policy.

At a scheduler boundary, pressure is:

- `free_snapshot_slots <= slot_headroom`, or
- host KV exists and `free_host_pages <= host_headroom_pages`.

Below both thresholds, do not start a disk write or discard a hot entry. Above a
threshold, consider unleased radix leaves, ordered by LRU then stable index. For
byte-only pressure require host pages in the leaf or its ancestry; a GPU-only
unrelated leaf cannot relieve it. Removing a zero-host leaf may expose a host
ancestor; account actual reclaimed pages only, never assumed credits.

A candidate with exact ready disk backing may be dropped immediately without a
write. Otherwise choose an eligible candidate whose complete image fits the disk
budget and start preservation. A clean/oversized oldest leaf must not prevent
trying another eligible leaf. Match ready backing using exact tokens and current
catalog state, not a stale reusable disk index. Published backing survives reads.
On successful completion re-evaluate current pressure and LRU; do not blindly drop
a formerly cold source that became hot during capture. At most one candidate
mutation/start per boundary; idle pending I/O may progress on subsequent turns.

This is best-effort caching. At hard capacity the existing eviction paths may drop
unleased optional entries before they acquire backing. They must not synchronously
start/wait for preservation. A pending source cannot be evicted, renamed or reused.
No indefinite persistence queue or full-image duplicate RAM allocation.

## Public ownership interfaces

The cache exposes bounded, read-only candidate metadata: generation-qualified
handle, LRU stamp, tokens, own host-page count and whether its ancestry includes
host pages. Metadata is borrowed only until the next cache mutation; callers must
acquire a Source before retaining bytes. Capacity queries expose used/total snapshot
slots. A generation-qualified discard operation validates that the current entry
is an unleased leaf, then uses existing tree removal. Flat caches report unsupported
rather than pretending to support source persistence.

A pure session policy accepts scalar capacities and candidate descriptions and
returns none / preserve(handle) / discard(handle). The serving adapter supplies
model-specific byte sizes, ready backing and pool usage. The policy neither calls
Vulkan/storage nor owns token buffers. The selector streams candidates without an intermediate array. The adapter is
bounded by the model's 64 physical snapshot slots; the generic radix accepts larger
configured capacities (the metadata benchmark also tests 256). No steady-state
allocation.

## Transfer priority

Add an archive advance mode that can acknowledge/drain existing owners **without
issuing another optional quantum**. The original advance behavior remains available
for foreground work and diagnostic tests. A completed D2H quantum may submit its
already-reserved write; publication still waits for disk completion. Pending writes
use at most staging depth minus two tickets (minimum valid depth must be checked),
reserving two for newly arriving reads. Foreground reads are polled first, then
optional ownership is acknowledged with new write work disabled while reads wait.
A running quantum cannot be preempted or freed early. No claim of independent DMA
engines or absence of same-queue compute contention.

## Scheduler and hard-pressure transitions

A separate optional backend maintenance hook returns pending/progressed/reclaimed
status, independent of request slots. It runs at safe chunk boundaries, outside the
scheduler lock just like foreground I/O. Recheck deferred old-slot releases after
relocking, before admitting any newly reused slot. Idle pending work wakes after at
most the existing 100 microsecond poll delay; ready inference remains eligible.

Stop forbids new preservation, aborts a packed chunk as already specified, and
drains **both** request and maintenance owners even when no request slots remain.
Only then may cache, disk, imported memory and model teardown proceed.

If a source lease prevents urgent allocation, request cancellation of optional
capture and keep the request waiting until acknowledgment. A decode row in this
state keeps its step/tokens; it is not reported as PoolExhausted, sampled, or reset.
Independent rows may continue. Reclamation/drain increments the admission epoch,
waking memory-blocked prompts and rows. Do not repeatedly retry a blocked row
without an epoch change. Suppress new preservation while urgent reclaim is pending;
resume only after allocation succeeds or the requesting workload releases capacity.
Once the optional source drains, ordinary swap/eviction/capacity failures retain
their existing semantics. Never extend optional-cache drop rules to live swap.

Disk failure/corruption produces a cache miss after full drain and cannot publish
backing. A failed, canceled or skipped source generation is excluded from further
optional captures until that snapshot slot is reused for a new generation. This
bounded per-slot suppression prevents an idle failure/retry loop; existing ready
backing can still be discarded normally. Urgent cancellation is latched per source,
so a different row's successful allocation cannot undo it before the next poll. Source leases release exactly once. Unexpected device pending ownership
or device loss remains fail-stop; known optional commands are accounted explicitly.

## Independent executable acceptance mechanism (defined before native code)

`tests/reference/generate_tiering_pressure_fixture.py` supplies a Python declarative
object-set/event model with stable logical IDs, separate snapshot slots, disk ready
sets and ancestor ownership. Determine LRU by sorting independently of native scans.
Fixed seeds retain full event inputs, decisions, resident/pending counts,
source generation/lease and ready backing after every transition. Generated before
native implementation: 2,048 scalar capacity/selection cases and 1,560 prefix-set
events, reusing the pinned C.1 independent prefix-set oracle with both generator
hashes embedded. Native tests drive actual cache acquire/discard/release and compare
all states and decisions. Byte-only/mixed candidate eligibility is covered by the
scalar matrix plus the actual mixed-ancestor device fake; the event traces do not
simulate DMA or substitute for the separate archive/scheduler interleaving gates. Ordinary native
tests consume the checked-in JSON, without Python or third-party dependencies.

Directed plus randomized traces must cover:

- no writes below either pressure threshold; byte-only/slot-only pressure;
- clean re-eviction without rewriting; a clean or oversized oldest candidate;
- mixed ancestors, shared pages counted once, source generation reuse;
- success, cancellation and failed writes, no early publication/free;
- read arrival during capture, two reserved tickets, saturated queues;
- urgent memory pressure while leased, retries only after release;
- one/two-slot budgets and zero-host configurations.

Test the actual archive and cache interfaces as well as policy decisions. Extend
scheduler fake-device tests for delayed acknowledgment, read priority, cancellation,
source-only idle work, stop without request jobs, source-blocked decode retry, and
leave/reuse during unlocked maintenance. Use negative controls that allow early
reuse and permit an optional write to overtake a read. Run both modes repeatedly.

CPU/Python/format, GPU/spill, host GPU and independent full-model oracle gates are
mandatory. Model state/logits must remain exact at 257/80k through production owners.
Serving: repeated cold/reuse/churn with baseline write-through, disk-off, host-only,
and tuned llama-server; revisit compatible competitors. Match weights, precision,
resource limits, workload, output lengths and cache state; compare every native
response byte-for-byte. Report bytes written, source-retention time, cancellations,
misses, drops, TTFT, token gaps, aggregate throughput and host/VRAM peaks with variance.
No performance/completion claim from policy tests or microbenchmarks alone. D's
proactive GPU→RAM preparation and queued-demand prefetch remain subsequent work.
