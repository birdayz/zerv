# Queued-demand protection and staging prefetch (D.2)

Status: **pre-implementation contract draft, 2026-09-29**. Native baseline
`576a197`; no D.2 API, flag or prefetch is implemented. The independent event
fixture and boundary tests below must exist before native implementation.
[Primary-source/native audit](../design/queued-demand-audit.md).

## Scope and non-goals

Opt-in scheduler lookahead protects the oldest queued `begin`'s best local prefix
and may pre-read one/two archive chunks into the existing host staging store.
This is bounded staging-window read-ahead, **not whole-image host prefetch**, and
must be logged/documented as such. It does not change model math, tokenization,
precision, tail canonicalization, snapshot format, context or HTTP semantics.

Eight staging tickets and existing host/device allocations remain the memory bound.
Optional write plus prefetch occupancy is at most six; two tickets remain available
for foreground reads. The window is `W * B` bytes (`W` one/two, `B` configured disk
chunk size), not a new allocation. One speculative queued job is enough initially;
protecting every waiter can starve active decode. No per-request heap allocation.

## Scheduler boundary and caller lifetime

An optional backend demand hook receives one `{slot, order, tokens}` or none,
plus stop/read-pending state. Select the oldest non-running, non-I/O queued begin;
use its existing nonwrapping arrival identity, not a reusable slot index alone.
Before unlocking for the callback, mark that operation running. Cancellation then
waits instead of releasing token storage. The hook must synchronously resolve all
token comparisons and retain only qualified cache/record identities on return.
It may submit bounded storage I/O, but must retain no token slice afterward.

On relock, clear the temporary running state. If canceled/closing/stopped, complete
that begin as canceled, respecting deferred slot release, and reconcile/cancel the
backend's old demand before any new-generation begin can consume it. Inspect
`to_release` again after this unlocked callback, as after the existing I/O and
maintenance callbacks. A callback pause/reuse test must prove old pages are released
before the new generation maps them.

Reconciliation precedes discretionary archive/preparation selection at every safe
scheduler boundary, including between packed units. It may look up metadata and
stage disk reads during a pack; it may never restore a snapshot, rename page IDs,
map target pages or upload GPU bytes there. Stop supplies no demand and drains
speculative storage ownership after the pack is aborted. Backend types without the
hook retain their current behavior; no generic hot-path token copying is added.

## Local protection versus immutable leases

Local protection is a separate cache policy preference, represented by a qualified
selected handle. Resolve its current ancestor path when testing candidates; do not
freeze a parent array or borrow token memory. Multiple operations may use the same
prefix normally. Protection forbids optional preparation, source capture/discard,
capacity eviction, and non-urgent demotion/drop along that path. Promotion and
alias rename for a live restore remain legal: this is **not** a source lease.

An explicit foreground-memory fallback may clear this preference and then evict;
it must count that override. It never clears an immutable source lease. A removed
or reused cache generation invalidates preference, never protects the replacement
by index accident. Clearing/replacing demand is non-yielding on the model thread.

If existing source ownership would make a demanded host restore unsafe, request
that owner's cancellation and defer only this begin until an admission/reclaim
epoch changes. Do not silently turn the host hit into a cold miss. Reuse the
`CacheReclaimPending` vocabulary, extending begin eligibility to honor its retry
epoch. Keep its tokens queued under normal caller ownership; do not borrow them
in the background owner. Independent runnable decode must continue. Source
acknowledgment/abort advances the epoch; failures preserve existing fail-stop rules.

## Disk owner state and handoff

One owner tracks `{request_slot, request_order, record, phase}`. Installing a
speculative read establishes the archive's existing reader pin immediately; that
active job prevents record reuse. The record index is never kept unpinned across
an asynchronous interval. Its request identity is checked again before handoff.

Phases:

1. **Selected:** compare local and disk longest valid prefixes, leaving at least one
   prompt token. Only prefetch when the disk prefix is longer than the local hit.
2. **Staging:** acquire up to the requested window within the combined optional
   allowance and submit disk reads. No `archiveMap`, model selection, GPU call or
   publication. Disk completion leaves the same tickets owned until handoff/drain.
3. **Handoff:** for the same queued generation, ordinary begin resets/adopts private
   target pages. Populate its destination map and permit uploads on the existing
   archive job; do not release/re-read/re-hash a different record. Hash/length checks
   remain mandatory before every upload. The normal read pipeline completes the
   remainder of the image. The speculative owner retires into foreground ownership.
4. **Cancel/drain:** generation disappeared/reused, stop, a better local hit supersedes
   it, admission failed, corrupt record, or foreground preemption. Forbid new issue,
   acknowledge pending disk owners, release tickets, then release reader pin. No
   target-slot reset or position publication from a staging-only error. Retire the
   old generation only after exact drain. A reused slot may wait for this drain,
   but may never attach the old record.

The archive advance boundary needs independent **issue** and **upload** permissions.
`allow_upload=false` keeps completed reads on their tickets without repeatedly
counting bytes/hashes and without calling the device. Cancellation ignores that
restriction to drain safely. A staged job is not a completed restore. The foreground
job uses the same verified bytes and tickets after permission changes.

If private target admission fails, cancel/drain the speculative job before falling
back; suppress restart for that same queued identity so the fallback cannot loop
forever. A foreground read can preempt speculative issue. Existing speculative
tickets drain normally; no forced ticket release while the disk worker owns them.

## Configuration and diagnostics

Proposed explicit mode `--prefix-cache-demand off|protect|prefetch` (default off).
`protect`: shared KV, parallel>1, radix cache, nonzero snapshots; applies to host and
GPU hits without disk. `prefetch` additionally requires disk. Proposed
`--prefix-cache-prefetch-chunks 1|2` defaults one and is invalid outside prefetch mode.
Resolve these at startup; no inert flags. Effective logs report mode, lookahead of
one request, chunk/ticket bounds and unchanged staging bytes.

Counters distinguish protected demands, urgent overrides, source-conflict waits,
speculative submitted/completed bytes, handoffs, cancellations, discarded staged
bytes and actual foreground restores. Never label a selected record as fetched or
a staged first window as a full host hit. Record request hold time, ticket occupancy,
TTFT, inter-token gaps and memory alongside throughput.

## Required executable gates before/after implementation

Before code: independent deterministic object/event fixture using exact token tuples,
named request generations, immutable record contents and ticket-owner sets. Cover
nested/partial/divergent prefixes, duplicate demand, eviction/mutation, source-held
host hits, cancel/leave/reuse during the unlocked hook, all completions out of order,
hard allocation fallback, foreground preemption, staged corruption, handoff and stop.
Expected state includes protected objects, record reader counts, ticket ownership,
bytes issued/uploaded/discarded, request result/position and zero owners at teardown.

Native interface runner: source/cache/record generations cannot be guessed; before
handoff upload callback count must be zero. A removed upload-permission guard and
an omitted request-generation check must each fail directed negative controls.
Compare event outputs to the independent fixture, not two native implementations.
The pinned distributed paper implementation is not an equivalent Vulkan ownership
oracle; retain independent numerical FP64/libllama state/logit checks separately.

After code: CPU suite and repeated concurrency tests in both modes, GPU/spill and
host-driver tests, exact 257/80k state and full-vocabulary continuation under
concurrent packed/decode/restore work. Component benchmark needs fixed traces,
warmup/repeats, pinned artifacts and real submitted-byte counts. Full-serving runs
must use final pinned binaries, no concurrent diagnostics, fresh/warm/reuse and
zero-idle pressure workloads; tuned Vulkan/HIP remain mandatory. Native responses
and counts must match off/on. Competitor quality equivalence is reported explicitly;
differing answers/counts cannot be called an exact matched-work speedup.

The API names and cancellation ordering must be checked against the independent
fixture before this draft is promoted to an implementation-ready contract. In
particular, every return path must account for both the temporary scheduler token
borrow and the longer-lived disk job; one cannot stand in for the other.
