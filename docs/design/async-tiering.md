# Asynchronous, pressure-driven prefix tiering

Plan requested 2026-09-28. This replaces neither the retired hang investigation
nor the measured baseline until each increment passes its gates. Source review:
[Mooncake / Pensieve / CachedAttention / LMCache](../research/2026-09-28-kv-tier-papers.md).
No paper's throughput numbers are predictions for this card.

## Target behavior

- Checkpoint creation preserves the exact recurrent/conv snapshot and immutable
  radix KV; **no NVMe write just because a checkpoint was created**.
- Anticipated GPU pressure starts bounded GPU→RAM preparation. Reclaim only after
  completion. Anticipated host pressure (bytes **or snapshot slots**) starts
  optional RAM/mixed-residency→NVMe preservation. Existing clean backing avoids
  another write. Live-request swap is never disposable cache.
- Copy submission does not immediately wait on the host. Independent requests
  remain eligible; dependent requests and buffers remain leased until completion.
  This is not a promise of hardware copy/compute overlap on the existing queue.
- Reads precede optional writes, with reserved staging/headroom and bounded work
  per scheduler turn. At hard capacity, discard unleased optional cache rather
  than stall inference indefinitely for persistence. Never release pending DMA.
- Keep the exact current stream format/precision. Full image = recurrent/conv
  state + all logical KV pages, including shared ancestors and the partial tail.
  Deduplicated disk blocks are a later, separate ownership change.

## Ordered implementation and gates

Exactly one active increment at a time; supporting adapters/tests are part of that
increment. Do not combine the policy rewrite with an unverified DMA lifetime change.

### A. GPU completion ownership (`src/gpu`)

**First increment.** Source inspection found another prerequisite: `Buffer.mapped`
currently rejects host access when *any* command on the device is pending. Merely
making archive copies nonblocking would make independent decode's host I/O fail.
Track pending uses of each buffer (including kernel descriptor bindings), add a
zero-time completion poll, and leave recorded-reference destruction guards intact.
Keep one queue and existing barriers. No runtime/backend boundary expansion.

Acceptance: independent transfer/affine goldens, multiple outstanding command
owners of the same buffer, unrelated host access, replay, poll/timeout/reset guards,
all CPU tests, both GPU modes, spills and host-driver gates. Measure the existing
matched native/C driver benchmark to detect bookkeeping overhead; no serving claim.
Specification: [gpu-driver.md](../specs/gpu-driver.md#asynchronous-completion-ownership-18d7a).

### B. Archive submit/poll/drain (`src/session`, supporting model/serve adapters)

Replace the synchronous device callback with explicit start/poll ownership. One
bounded GPU quantum at a time; retain its storage ticket across polls. D2H must
complete before hashing/submitting disk writes. Verified disk reads retain tickets
until H2D completion. Cancellation drains both owners, even if no disk I/O was
submitted. A start error must mean no borrowed span remains in use; unknown device
ownership fails closed. Other disk jobs can progress while the GPU quantum is pending.

Source sequence remains paused throughout capture in this increment: mutable
recurrent state cannot be read while next-token execution overwrites it. This step
changes execution, **not** write-through admission. Timeouts remain bounded/fail-stop
rather than creating an indefinite polling loop. One existing compute queue means
submission independence, not a proven DMA-engine overlap implementation.

Acceptance: independent POSIX/hash fixture unchanged, delayed device fake and
adversarial cancellation/failure/slot reuse (including cancellation mid-upload),
no steady allocations; production-owner 257/80k poisoned/permuted state and full
vocabulary equality, independent FP64/libllama semantics retained; repeated archive
component and HTTP comparison with baseline, disk off, host and tuned llama-server.
Keep all failures, exact commands/source hashes and raw timings in dated reports.

### C. Immutable cache sources and pressure policy (`src/session`)

Separate logical checkpoint IDs, reusable snapshot slots and disk record IDs.
Lease the selected checkpoint **and every ancestor segment**; leased generations
cannot be removed, renamed, promoted or demoted until capture finishes. Gather mixed
GPU/host sources through bounded staging, rather than allocating another full image.
Use the immutable snapshot slot, not the originating request's mutable state.

Implement source acquisition/release first against an independent residency trace
oracle, then byte-gather gates; only then remove checkpoint-triggered disk writes.
Pressure decisions count resident **and pending** bytes and slots. Reserve at least
one snapshot slot and bounded staging for incoming work where capacity permits;
small capacities explicitly use best-effort drop, not an unbounded extra allocation.
Expose measurable headroom controls only alongside their implementation. Initial
victims are cold unleased leaves; keep existing radix ownership. A leased path can
still be read if that does not rename/mutate its physical source; otherwise choose
another ready backing/cold path. Valid disk backing survives restores and clean
re-eviction. Publish backing only after all writes complete.

Acceptance before closing: oracle traces for byte-only/slot-only pressure, no disk
writes below pressure, no rewrite after clean re-eviction, source-generation reuse,
mixed ancestry, queue saturation/read priority, failure/cancel/drain/live state;
257/80k exact production state/logits. Required fresh serving runs compare cold,
reuse and churn with identical native outputs, explicit host/VRAM/disk budgets,
bytes written and source-retention times. This increment still requires a detailed
functional spec/oracle before its code; this plan is not that spec.

### D. Proactive preparation and hardware overlap

Using C's ownership protocol, implement bounded GPU→RAM preparation ahead of demand
(Pensieve/LMCache), host spill headroom and queued-demand prefetch (CachedAttention).
Freeze recurrent state in a bounded device snapshot only if its extra VRAM/copy
wins against pausing just the source sequence. Investigate another queue only with
explicit cross-queue visibility/ownership and actual measured overlap; no automatic
assumption that Vulkan compute queues use independent DMA engines. Prioritize reads
and tune transfer window size on this GPU, including bidirectional contention.

Acceptance: burst-pressure/known-queued-demand cases, exact logits/state, memory
bounds, repeated loaded serving against tuned competitors. Cost-based restore vs
recompute and disk block deduplication remain subsequent increments, not hidden
scope in this plan.

## Status

A completed ([ownership gates and matched component results](../bench/2026-09-28-async-ownership.md));
B completed ([archive correctness, components and fresh serving comparison](../bench/2026-09-28-async-archive.md)).
The original checkpoint-triggered write-through policy is still production behavior,
now with nonblocking archive GPU ownership. No serving speedup established; the 80k
restore regression is retained. Snapshot/page transfers remain synchronous.

C is active. C.1 source leases are implemented with full ancestor protection and
independent prefix-set oracle ([report](../bench/2026-09-28-cache-sources.md)).
C.2's independent canonical partial-tail fixture (60 layouts, 4,804 windows) and
mixed-source adapter pass CPU 81/81, both modes ×20, negative control, GPU/spill 3/3,
host GPU 2/2, independent model oracle 337/337 and exact 257/80k model gates.
Repeated component timings retain substantial variance and the historical restore
regression; [C.2 report](../bench/2026-09-28-cache-source-bytes.md).
C.3 is active at [specification/oracle](../specs/tiering-pressure.md):
pressure-triggered persistence is not integrated. [Partial-page finding](../research/2026-09-28-async-transfers.md#c-readiness-finding-partial-pages-are-not-immutable-full-byte-images).
D stays queued. The overall requested pressure-driven policy is **not complete**.
The user's active session goal also requires rigorously verified faster-or-on-par
serving against tuned compatible competitors; A/B/C.1 component gates do not close
that goal or erase the observed long-restore regression.
