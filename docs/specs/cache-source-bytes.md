# Canonical mixed-residency cache capture (18d.7c.2)

Specified before implementation, 2026-09-28. [Source audit](../research/2026-09-28-cache-sources.md),
[lease contract](cache-source-leases.md), [archive transport](disk-prefix-cache.md).
This increment adds source capture, not pressure admission or scheduler prefetch.

## Canonical stream

The stream remains snapshotBytes recurrent/conv bytes followed by every logical
KV page, grouped by KV buffer; ceil(tokens/page) pages per group. Within each page,
each group's attention layers have K [kv_head][dim][page] then V [kv_head][page][dim].
f16/f32 values remain raw little-endian bytes: no conversion, arithmetic or rounding.
In the final partial page, every element at a token position >= tokens % page is
canonical zero. Full pages and all recurrent/conv bytes are copied exactly. A full
last page has no tail. The opaque archive pads the final disk chunk as before.

This differs explicitly from the older paused-slot image's arbitrary unused bytes.
Exactness is equality to the canonical stream, all valid state bytes unchanged,
and exact full-vocabulary continuation. No restart-persistent format is involved.

`State.clearArchiveTail(snapshot_bytes, tokens, offset, bytes)` operates on a bounded
window of this stream after completion. A State must come from the validated layout
constructor. Reject zero/out-of-context tokens and out-of-stream/overflowing ranges
before modifying any byte. Empty in-range windows are allowed. Arbitrary byte
windows are supported; GPU transfer endpoints separately enforce 4-byte alignment.
Intersect tail ranges only with this window, skipping groups/pages outside it; do
not scan full preceding pages. For an odd f16 K tail, copy the aligned word and
zero only its invalid half after acknowledgment. Never clear valid paired halves.

## Source adapter

`Model.archiveSourceSubmit(commands, staging, staging_offset, snapshot, tokens,
map, offset, bytes) !bool` accepts a leased cache source, not a mutable request slot.
Snapshot/map metadata must remain leased through completion. Validate all bounds,
GPU pin/logical-page ownership and host checkpoint ownership before recording;
reject live-swap host pages, unpinned GPU pages and invalid snapshot IDs explicitly.
The pool owns its checkpoint sentinel check; the adapter must not duplicate it.

Host snapshot/page spans are copied directly into staging with bounded memcpy;
only device-resident spans require GPU copies. Return true after GPU submission,
false for a wholly CPU-completed quantum. Both require a subsequent owner poll
before hashing/publication. MTP and non-shared KV remain unsupported. No allocation
or full-image staging. Mixed capture never promotes/demotes its source.

Use compute→transfer before source GPU reads, transfer→compute after reads (later
appends may write unused tails), and transfer→host before host canonicalization.
One existing queue, no assertion of independent DMA-engine overlap. Snapshot host
writes were synchronously completed before checkpoint publication, with an explicit
transfer→host visibility barrier for mapped snapshot destinations. Host checkpoint
logical indices are maintained on demotion from the original GPU page (demotion
may cover only a suffix) and validated against the complete source map. Whole-buffer
map guards remain enforced: no raw pointer bypass of another pending owner. If a
start fails it leaves no device access; unresolved ownership fails-stop as in B.

## Production disk-owner hook

One additional bounded archive job, independent of request slots (1..65 jobs,
maximum 64 requests plus one cache source). Maps/positions sized consistently.
`startSource(cache)` acquires the oldest eligible source, starts a full-image write,
and retains the lease until catalog completion/drain. Duplicate/insufficient-capacity
skip releases immediately. Source quantum completion canonicalizes before hashing.
`pollSource(cancel)` drives this job, then releases on done or drained failure;
`destroy` rejects outstanding source ownership. Model reads keep request job IDs.
The existing live-slot write path remains available for its unchanged diagnostic
and baseline gates. Scheduler pressure integration is the next increment; adding
this hook alone must not claim automatic background persistence.

## Preimplementation oracle and acceptance

Independent Python byte-coordinate oracle decodes each selected stream byte into
(group, logical page, layer, K/V, head, dim, token, element byte), rather than using
the native range-clearing algorithm. Nonzero deterministic input, expected SHA256
of every window, generator provenance. Cover f16/f32, 128/256-token pages, one/
three/sixteen layers per group, odd/aligned tails, snapshot/group/page/K/V boundaries,
unaligned host windows and 1 MiB windows. Ordinary tests consume checked-in fixtures.

Model gate: retain older paused-source tests; new production source mode captures
at 257/80k, advances and reuses originating slot during capture, forces host ancestor
+ GPU suffix where available, poisons all destination bytes and reverses pages,
then compares canonical state and four complete vocabulary rows. Test cancellation
after actual submitted capture/upload, retry, disk corruption and source release.
All CPU, both GPU modes, host driver and spill gates, independent model oracle;
repeat component timings, preserving regressions. No serving improvement claim
until pressure-policy integration and tuned quality/resource-controlled benchmarks.
