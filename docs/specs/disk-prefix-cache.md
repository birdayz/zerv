# RAM-staged disk prefix archive (18d.6 integration)

Status: specified before implementation, 2026-09-28; production integration is
implemented and opt-in. Independent fixtures, real-model 80k state/logit and serving
gates pass; [evidence and measured tradeoffs](../bench/2026-09-28-disk-prefix-serving.md).
User explicitly requested full integration and a real 80k-token gate. Research: existing
[NVMe transport](../research/2026-09-28-nvme-store.md) and
[snapshot residency](../research/2026-09-28-snapshot-residency.md), plus the decisions
below. No filesystem-specific policy in core. No restart persistence.

## Resolved integration design

Use an immutable **write-through archive behind the existing hot radix cache**.
At a checkpoint, keep the usual hot checkpoint, then archive the complete recurrent
snapshot and its logical KV pages. Capture is a pending scheduler operation; the
source sequence is paused until capture finishes. Other sequences can decode.
A duplicate prefix already ready or being written is not written again. Hot cache
hits win when at least as long as the archive hit; a longer disk hit restores into
fresh private pages, then prefills at least one remaining token for logits.

This deliberately avoids adding disk page IDs into the existing shared/pinned KV
pool. Archived checkpoints own independent full images, not ancestor page pins;
releasing/renaming a hot radix segment cannot invalidate disk bytes. Disk records
outlive hot checkpoints and snapshot slots. Tradeoffs: cold checkpoint write cost
and duplicated shared-prefix bytes on disk, in exchange for bounded memory and
simple exact ownership. Not a claim that write-through is the fastest policy.
The previous standalone residency table remains a verified write-back primitive;
this integration does not pretend a live sequence is one of its resident snapshots.
An eviction-only write-back policy is not required to enable useful disk caching.

Sources inspected: current `checkpoint`, `kvcache`, model snapshot/page copy and
pool ownership implementations; scheduler begin/checkpoint/cancel/leave and timed
futex waits; SGLang v0.5.20 hiradix write-through/ack and mamba ownership at the
hashes recorded in snapshot-residency research. No third-party implementation is
imported. Existing exact model gates establish host checkpoint semantics; independent
Python POSIX/SHA256 goldens establish opaque disk data/integrity semantics below.

## Data and bounds

A record's logical bytes are: recurrent/conv snapshot (the runtime's exact
snapshotBytes), followed by KV groups in ascending group order, within each group
logical pages in ascending order, each the full existing pageBytes(group). Include
ceil(prefix_tokens/page_tokens) pages, including the partial last page. No conversion
or arithmetic. Source sequence does not advance until capture ends. Restore target
pages are exclusively owned, including the partial page. MTP is initially rejected
(the concurrent serving path already excludes it).

Storage uses fixed-size chunks (default 1 MiB), anonymous aligned host staging
imported into Vulkan once at startup, eight staging slots. Final chunk is zero
padded. Disk budget is preallocated, split into chunk extents; each record owns an
ordered list of extents and a SHA256 per padded chunk. Token prefix, byte length,
chunk map, hashes, ready flag and LRU live only in bounded process RAM. There is no
on-disk persistent header or recovery: file is unlinked at create and a new engine
cannot interpret old data. Identity is scoped to one loaded model/configuration.

Record count is configurable; full-context token slots and max-context chunk maps
are preallocated, checked, and limited to 256 MiB total metadata. File <=1 TiB,
chunk 4 KiB..64 MiB, depth 1..32 (transport limits), slots 1..64, records 1..4096.
Record reservations require enough free chunks, evicting unleased LRU ready records.
If a record cannot fit, skip caching, count the skip, and continue inference.
A record is published only after every exact write completion. Readers lease ready
records; eviction cannot reuse extents until every reader finishes. No per-token
or per-chunk heap allocation.

On read, validate SHA256 before GPU upload. Short/error CQEs or wrong hashes taint
and invalidate the record; drain outstanding I/O, discard partial target state and
start cold. A disk write error drains then drops the unpublished record; generation
continues from its intact source. GPU errors are propagated, not mislabeled disk
misses. Cancellation stops new chunks and drains all submitted I/O before releasing
staging, record leases, slot state or borrowed request tokens. Unknown DMA ownership
remains fail-stop as specified by the transport.

## Scheduling and adapter boundaries

`session.archive` owns bounded catalog/extent/transfer bookkeeping and borrows
`storage.Store`. A small device interface exports/imports an opaque chunk for a
specified sequence slot. It owns no GPU objects and does no model math.
`model` implements byte-stream addressing/copies with validated bounds and private
restore destinations. `serve` owns anonymous staging, imported Vulkan buffer and
storage lifetime, and supplies that adapter.

Backend begin/checkpoint may return PendingIo. The batcher retains the operation
and borrowed tokens, keeps its slot running/nonreusable, and polls one pending slot
round-robin at scheduler boundaries (not inside a packed prefill chunk). Each poll
performs at most one chunk GPU copy/hash plus bounded completion/submission work.
Decoding and other prompt work remain eligible. An idle scheduler uses a bounded
100-us completion poll timeout; no blocking disk wait or busy spin. Useful progress
can trigger an immediate next iteration. The original GPU adapter used bounded
synchronous fence waits; implemented amendment 18d.7b below replaces these with
submit/poll/drain. Neither version demonstrates hardware copy/compute overlap.

Stop/cancel drains pending operations, aborting a packed chunk first if needed during
shutdown. A disk restore that cannot reserve its target pages becomes a cold miss;
it must not hold a partial page reservation waiting for itself. Request cancellation
and shutdown never free a DMA-owned staging buffer. Track writes, reads, bytes,
hits, evictions, skips, disk failures and cancellation in shutdown statistics.

## Configuration

Explicit server options: scratch directory, disk MiB, record count and optional
DIO alignment. Directory and disk budget must be supplied together. Require shared
KV, parallel >1, no speculation and a supported snapshot layout. Concurrent serving
retains its existing non-speculative policy; the disk owner rejects an MTP model.
Reject unsupported disk combinations at startup, not silently disable disk caching. No filesystem ioctls,
mount changes or NOCOW policy; deployment guide remains operator-owned. Allocate
staging within explicit device/host budget. Disk tier defaults off.

## Acceptance gates (before enabling by default: never in this change)

1. Independent Python POSIX fixture + hashlib SHA256, generated before native
   implementation: exact padded bytes/digests; native byte transport checked using
   independent synchronous syscalls. Multiple records, fragmented extents, more
   records than hot snapshots, concurrent readers/writers, corruption, short/negative
   completion, cancellation/drain, budgets, eviction, allocation failure/no steady
   allocation. Ordinary tests require no external engine or model.
2. Scheduler fake: delayed disk operation coexists with decoding; canceled waiter
   keeps tokens/slot until drained; stop and close drain; no false completion/spin.
3. GPU byte-stream adapter: actual model capture/restore into a different slot/page
   allocation after poisoning freed state/KV. Exact full state bytes and full-vocab
   logits/teacher-forced continuation, short/misaligned and **80,000-token** prefixes.
   Compare against the same prefilled state without disk; independent existing model
   oracle gates remain mandatory, not replaced by archive self-roundtrip.
4. CPU/Python/format tests; GPU Debug/ReleaseFast + spill gate + host driver tests.
5. Repeatable component/serving measurement, pinned hashes/configs, warmup/repetitions,
   raw variance, disk-off/host-only and tuned llama-server. Serving: cold write cost,
   restore TTFT, inter-token gaps for another request, context and memory, cancellation
   and capacity pressure. No speedup claim without actual compatible comparisons;
   record other tracked competitors' compatibility/limitations.

### Adapter/lifetime details resolved before serving implementation

The serving owner is heap-stable (Vulkan commands retain pointers to its imported
buffer). It allocates one logical-page-map slice per model slot at startup and
resolves it before starting each transfer; model copy validation checks the map
still names the slot's pages. Capture follows hot checkpoint rebind/deduplication.
The byte-stream adapter rejects MTP and nonshared pools; private restore checks
both single-slot ownership and absence of checkpoint pins.

A GPU copy failure with a pending command must fail-stop immediately: even if the
archive subsequently drains disk CQEs, GPU ownership of staging is not resolved.
A known completed GPU error propagates; only DiskIoFailed/CorruptRecord become a
cold miss. Disk cancellation is reported only after all tickets drain. An owner
cannot be destroyed with active archive jobs. Destruction order: archive metadata,
commands, imported buffer, drained store, mmap, maps, owner allocation.

Pending scheduler operations keep their op, tokens and running status. The generic
backend poll hook is optional (existing CPU fakes need not implement it). A hook
is required to return PendingIo. Poll one slot round-robin per outer iteration;
exclude pending slots from decode gathering and swap victims. On stop, abort a
packed chunk before polling cancellations, then drain pending jobs, wake their
waiters and release closed slots before returning. Shutdown does not start new I/O.

Regression coverage: a delayed fake must demonstrate another slot decoding before
completion; cancel, leave and stop must keep borrowed tokens and slots until the
fake acknowledges drain. Real-model gate poisons all state/KV, remaps the target's
page table in reverse order and compares every stream byte and four teacher-forced
full vocabulary rows, not merely sampled tokens. Disk gate must use the same
serving owner as production, not a separate file-copy implementation.

### Serving measurement protocol

Use the checked-in `multiturn-distinct-v1.json`, four distinct long conversations,
phased turns (all first turns finish before any second turn), two resident serving
slots and 12,288 context per sequence. Compare disk-off, host tier and disk archive
with explicit budgets; run tuned llama-server FA/batch variants on the same model
and workload. One short warmup per fresh process; at least three process trials,
engine order alternating. The disk budget/metadata and hot snapshot count must be
recorded (a capacity comparison may intentionally buy capacity with disk, not RAM).
`run_multiturn.py` retains all delta arrival timestamps and now reports delta gaps,
aggregate output throughput, peak VRAM and host RSS/HWM. Delta events are not always
single tokens; do not relabel their gaps as exact token intervals for every engine.
Compare every zerv turn output hash across configurations; llama's arithmetic and
cache semantics differ, so report its quality/semantic differences rather than
requiring token identity. Other tracked competitors' hardware/quality differences
remain as recorded in the tiered/concurrent benchmark reports.

## Proposed replacement: eviction-driven spill (user discussion, not implemented)

2026-09-28: user rejects write-through as the cache policy and asks when RAM/NVMe
should be written. The implementation above remains the measured baseline, not
the proposed final policy. No implementation block has been opened for this revision.
[Primary-source paper/code review](../research/2026-09-28-kv-tier-papers.md) follows:
Mooncake supports eager and eviction-driven SSD offload; Pensieve and LMCache
motivate bounded ahead-of-time reclamation, and CachedAttention directly describes
RAM-pressure-driven SSD eviction. Eager GPU→RAM transfer is distinct from eager
SSD persistence; do not claim the literature rejects every write-through policy.

Proposed write triggers:
- At a reusable checkpoint boundary, preserve recurrent/conv state (currently a
  host snapshot by default in concurrent serving) and pin/share immutable GPU KV
  pages. This is not a full KV copy to RAM and does not write NVMe. Active decode's
  mutable recurrent state cannot replace an immutable historical snapshot.
- On GPU cache pressure, demote reclaimable cached KV pages into the bounded host
  tier. Only release the GPU copy after transfer completion. Live-request swap is
  separate, higher-priority state and must not be treated as disposable cache.
- On host-cache high-water pressure (bytes OR snapshot-slot capacity), choose an
  unleased cold checkpoint. If it has valid disk backing, discard its redundant
  host residency without writing again. Otherwise, reserve disk extents and a
  bounded spill job, lease its exact snapshot/page generation, and write it before
  reclaiming the source. Start before the hard limit; reserve headroom for pending
  writes in both host bytes and snapshot slots. Watermarks are policy parameters,
  not chosen/tuned percentages yet.
- Mixed GPU/host checkpoints must be fully accounted for: either materialize their
  missing GPU pages into reserved host capacity or gather through bounded staging
  while leasing every source segment. NVMe must receive the complete restorable
  image, not just an evicted radix suffix. The choice requires measurement/spec
  resolution before coding; pretending every candidate is already fully in RAM
  would be incorrect.
- A spill acknowledgement publishes the complete disk record, then permits source
  reclamation. Failed/canceled writes drain first and never publish partial data.
  Cache spill is best effort: a full queue/insufficient headroom may drop an
  unleased cached checkpoint instead of making inference wait for optional disk
  persistence. Pinned/live state is never discarded. Existing admission/swap
  behavior still applies if active state itself exhausts capacity.
- Hits use the fastest available complete state. NVMe restores through bounded RAM
  staging to private GPU pages; a full additional RAM-cache copy is not mandatory.
  Keep valid immutable disk backing after a hit. An extended prefix is a new
  checkpoint, not an in-place mutation/rewrite of the old disk record.

Required structural change: current checkpoint identity equals snapshot-slot index
(`kvcache.createTiered`, `fillEntry`, `restoreEntry`); host drop is synchronous
`evictHost() -> bool`. Simply moving `startWrite` into that callback is unsafe.
Separate checkpoint identity from physical snapshot slots, use explicit pending
spill/lease/completion ownership, and drive progress on the scheduler thread with
only file I/O in the storage worker. The existing residency primitive is research/
implementation material, not already integrated behavior. Recurrent snapshots may
need spilling even when the host KV byte budget is not full.

Before implementation: resolve mixed-residency source capture, radix ancestor
leases and mutation/reuse, reserve accounting and eviction progress; extend the
independent ownership oracle. Acceptance must include zero NVMe writes below
pressure, GPU→RAM and RAM→disk threshold transitions, slot-pressure-only eviction,
no rewrite of clean disk-backed entries, saturated-queue forward progress, retained
live state, corruption/cancel/drain and exact 80k state/logits. Repeat the cold/reuse
serving comparison; asynchronous transport alone does not establish a latency win.

## Asynchronous device quanta (18d.7b; specified before implementation)

Execution amendment to the baseline; **not yet pressure-driven admission**.
[Plan](../design/async-tiering.md), [resolved lifecycle research](../research/2026-09-28-async-transfers.md).
Implemented and verified ([report](../bench/2026-09-28-async-archive.md)).
Supersedes the synchronous adapter callback/fence-wait paragraphs above; stream
format, precision, immutable catalog and error policy stay.

Device interface: `start(ctx, slot, offset, bytes, importing) !void` borrows exactly
that span until `poll(ctx) !bool` reports true or a terminal error. Only one device
operation outstanding per archive. False means pending. Errors may return only
when no device access remains; uncertain DMA ownership fails-stop in the adapter.
`start` must not wait for completion. CPU reference adapters may finish immediately,
but still acknowledge through poll. No default synchronous callback in production.

An acquired write ticket is GPU-owned until D2H completion, then hashed/submitted
and disk-owned until completion. A completed read ticket is verified, then GPU-owned
until H2D completion. Pending jobs retain source/target slots, tokens and mappings
through both phases. Store.release is never called on a GPU-owned ticket even
though the store itself sees held/done. Padding remains zero, SHA256 covers full
chunks, only valid final bytes are transferred. Polls perform bounded work and do
not wait on device or disk. At most one device start per call; no steady allocations.

Cancellation/failure stops acquisition and drains both kinds of leases. No done,
error, record publication, slot release or reuse until this job has zero leases.
A read with partial upload is reset/released only after drain. An owned pending
archive command alone does not make ModelBackend fatal; lost device or unaccounted
pending commands still do. Disk owns a borrowed std.Io whose lifetime covers its
creation through destruction; elapsed monotonic GPU timeout uses the existing model
limit. On deadline, check completion first; still-pending DMA fails-stop, not an
unsafe cold fallback. No new process/thread/queue or unbounded resource collection.

Acceptance: unchanged independent POSIX/hash byte fixture; deterministic delayed
D2H and H2D cancellation, zero write-before-capture, no release-before-upload,
multiple slots making disk progress, start/completion faults and slot retry;
Debug/ReleaseFast and allocator-failure/no-steady-allocation tests. Production
257/80k exact state and continuation, cancellation and last-chunk corruption;
independent FP64/libllama gate and all GPU/spill/host gates. Repeat component and
three-process serving measurements with tuned llama-server. Nonblocking submission
is not a claim of hardware overlap or faster serving. Earlier eager persistence
cost remains until the separate pressure-policy increment passes its gates.
