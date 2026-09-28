# Pressure-policy integration audit (C.3)

2026-09-28; pre-code research/design, followed by the implementation notes below.
C.2 and C.3 evaluation gates are complete; C.3 retains a negative performance
result ([report](../bench/2026-09-28-tiering-pressure.md)). The source observations
below describe the pre-C.3 tree. Ordered plan:
[async-tiering.md](async-tiering.md).
Paper/revision ledger: [KV tier papers](../research/2026-09-28-kv-tier-papers.md).

## Source cross-check

Re-read pinned Mooncake `f8b50d5f1a2fa4d33d029ed1e82188de70b137f0`, locally
`third_party/mooncake/f8b50d5f1a2fa4d33d029ed1e82188de70b137f0/`:
`mooncake-store/include/master_config.h` and
`mooncake-store/src/master_service.cpp`, `BatchEvict`, lines 11510–11670.
A ready local-disk replica bypasses offload; a queued write pins one memory
replica; queue saturation keeps data unless force-eviction is explicitly enabled.
This confirms the prior paper/code distinction. Our optional prefix cache may
explicitly drop unleased entries at hard capacity; live swap must never follow
that rule. No third-party implementation/runtime dependency is introduced.

## Integration findings in the current native tree

- `ModelBackend.checkpoint` first saves the hot checkpoint, then calls
  `Disk.startWrite(slot, prefix)` and returns `PendingIo`. Removing that second
  action is necessary but insufficient: the scheduler currently polls only jobs
  attached to waiting request slots.
- `Batcher.pollIo` polls one request job outside a packed chunk. `run` exits on
  stop when request jobs are drained. A background owner needs independent poll,
  idle wake and stop/drain accounting even with **zero live request slots**.
  The existing post-poll `to_release` recheck must apply after every new unlocked
  callback too; otherwise slot reuse can reintroduce the already-fixed race.
- `Radix.take` drops an LRU leaf on snapshot exhaustion. `makeRoom` can demote
  internal segments and drop host leaves; `evictHost` is also used for **live
  request** swap admission. A persistence callback hidden in these synchronous
  eviction operations would not implement ahead-of-pressure work.
- Source leases prevent mutations, not merely frees. A held cold source can retain
  ancestors and block promotion of a requested prefix. Cancellation under urgent
  memory demand must drain before releasing those pins; readiness must then wake
  failed admissions. Treating temporarily leased capacity as a permanent shortage
  would incorrectly fail a request or leave its admission epoch unchanged.
- `Archive.startWrite` already rejects exact token duplicates (ready or writing).
  Reads leave ready backing intact. A candidate selector must skip a clean LRU
  source and consider another dirty source; repeatedly selecting the same clean
  leaf would starve preservation. Never cache an unchecked reusable record index
  as proof of backing. Exact ready-token equality or qualified identity is needed.
- `Archive.advance` currently can complete one quantum and immediately start the
  next write in the same call. Merely polling reads first does **not** guarantee
  read priority if the write retains the single device quantum across calls.
  Separate drain/acknowledgment from permission to issue optional work. Reserve
  staging tickets for reads, including reads arriving after a write starts.
- Host KV pages and snapshot slots are separate capacities. `pool.hostFree()`
  includes live swap consumption; source-pending pages stay occupied. Snapshot
  arrays are physically preallocated, so slot pressure is not allocator RSS.
  Ancestor-shared pages must be counted once, not once per source image.

## Decisions to carry into the functional specification

1. Single scheduler-owned optional capture; bounded staging; no full-image
   allocation. Source lifetime is independent of request lifetime.
2. Checkpoint insertion alone does not imply disk I/O. Trigger preservation from
   explicit free-byte/free-slot headroom and anticipated demand. Count leases as
   resident until acknowledgment, and distinguish preparation from reclamation.
3. Prefer read completion/submission over new optional write work. At hard
   pressure, cancel optional preservation, drain, then retry admission; unleased
   optional state may be dropped. Never block indefinitely or drop live state.
4. Skip ready backing without rewriting. Do not let one clean or oversized leaf
   prevent trying an eligible dirty leaf. Bound scans and avoid per-token full
   prefix comparisons when no capacity/candidate/backing state changed.
5. Stop prevents new preparation and drains background ownership before cache,
   imported buffers or storage destruction. Recoverable disk failure is a missed
   cache opportunity; unresolved GPU ownership is fail-stop.

## Pre-code gates still required

Detailed API/state-machine specification and an independent event-trace oracle:
byte-only and slot-only pressure; zero writes below pressure; clean re-eviction;
ancestor retention and generation reuse; multiple candidates with the oldest
clean/oversized; read arrival during capture; saturated staging; admission blocked
by a lease; stop with no request jobs; failure/cancel/drain; no lost live state.
Include deterministic scheduler tests for release/reuse during unlocked background
polls and epoch wakeup after canceled capture.

Then production state/logits gates at 257/80k and repeated cold/reuse/churn HTTP
runs with output equality, matched resource/quality constraints and tuned
llama-server. Report bytes written, retained-source duration, drops, avoided
recompute, TTFT/gaps, throughput and peak host/VRAM. A single-queue submit API is
not evidence of hardware overlap; proactive GPU→RAM and queued-demand prefetch
remain D, after C gates. No throughput improvement is inferred from this audit.

## C.3 implementation decisions and verification history

After C.2 closed (`68fb7bc`), the pre-code fixture generated 2,048 scalar selection
cases and 1,560 prefix-set transitions, including generation/ancestor leases and
ready backing. It imports the C.1 oracle and records both generator hashes. The
first random matrix was unnecessarily 23 MiB; bounded it to 2.4 MiB before native
code, preserving 1/2/3/8/64/256 capacity and 0/1/16/65,536 host-page coverage.
Native cache tests compare every prefix, generation, serial, reference count,
pending source, ready set and decision; static cases cover byte-only pressure.
Actual archive/scheduler tests, not the pure event model, cover transfer priority.

The selector streams candidates (no stack array/heap allocation). The earlier
text's "256-entry radix limit" was inaccurate: the generic Radix accepts up to
u16 entries, while this model has 64 physical snapshots. The serving suppression
array uses the exported model limit, not the request-slot limit.

During integration review, a retry-loop gap was identified: disk failure can leave
pressure unchanged, so a naive idle maintenance loop would reacquire the same
source forever. Added a bounded failed/canceled/skipped generation per physical
snapshot slot. This excludes another optional capture of that incarnation; ready
backing remains usable, and a new incarnation can be tried. This is an explicit
best-effort policy amendment, not Mooncake's data-preserving retry semantics.
Urgent source cancellation is separately latched so another row's successful grow
or request release cannot accidentally undo it before the maintenance poll.

Working-tree implementation removes checkpoint-triggered writes; the old paused
write is explicit in the regression checker. Initial server/checker compilation
caught an undeclared diagnostic variable and discarded error-set value; fixed,
raw failed build logs retained. Scheduler test compilation caught field/declaration
ordering and a shadowed helper name; fixed, logs retained.

CPU gate 81/81, cache/archive/batcher both modes ×20 pass. Read-priority negative
control (ignore allow_start) fails both archive modes. Removing only the public
discard guard still passed because the internal removal guard also protects the
source; removing **both** fails both modes. Both guards restored. Subsequent
GPU/model, component and serving evaluations are complete in the linked report:
96 immediate-turn native responses exact, a positive idle HTTP disk restore,
and no immediate-turn disk speedup. D's preparation/cadence/demand work remains
open. Raw data: `docs/bench/data/2026-09-28-tiering-pressure/`.
