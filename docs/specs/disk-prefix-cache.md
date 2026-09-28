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
can trigger an immediate next iteration. GPU copies are bounded synchronous fence
waits; this does not claim asynchronous GPU copy/compute overlap.

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
