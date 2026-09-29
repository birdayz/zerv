# D.2 queued demand: ownership audit and implementation decisions

2026-09-29, native baseline `576a197`. **Research/design, not implemented.**
D.1.2 is closed as a correctness/evaluation increment, not a speed claim.

## Primary-source basis (re-read for this increment)

Exact source revisions, URLs and SHA-256s remain in
[the paper ledger](../research/2026-09-28-kv-tier-papers/sources.json).

- CachedAttention arXiv **2403.19708v3**, local
  `third_party/kv-tier-papers/2403.19708v3/paper.txt`, lines189–198 (§3.3):
  known waiting jobs drive fetching; lookahead capacity `L = C_mem / S_kv`;
  protect soon-needed entries and evict later-needed ones first. Its unit is a
  whole conversation. Native radix/checkpoint semantics are not equivalent.
- LMCache **2510.09665v2**, local versioned `paper.txt`, lines265–269:
  admission-to-use delay permits asynchronous prefetch; target tier is configurable.
- Mooncake pinned **f8b50d5f1a2fa4d33d029ed1e82188de70b137f0**,
  `docs/source/design/hicache-design.md`, lines37–67: local matching precedes L3
  fetch, transferred state must remain a continuous prefix, best-effort/wait/timeout
  are explicit alternatives. This is implementation documentation, not evidence
  that its distributed RDMA mechanisms transfer unchanged to one Vulkan device.

Inspection commands: `rg -n -i 'prefetch|fetching buffer|schedul'` on the above
CachedAttention text; `rg -n -i 'prefetch|asynchronous'` on LMCache; `sed -n
'35,67p'` on the pinned Mooncake document. No source imported or linked.

## Observed native hazards

1. `Batcher.submit` borrows caller tokens until completion. `withdraw` can return
   immediately for a non-running slot, after which tokens may be freed. Passing
   queued token slices through an unlocked maintenance callback is unsafe unless
   the affected operation is marked running until the callback returns. Its generation
   must remain qualified through leave/reuse/stop. The callback must retain **no**
   token pointer after returning; cache/record identities replace it.
2. `Cache.acquireSource` permits only one active source lease per node; leases hold
   ancestor refs. They prevent *renaming*, not just eviction. Reusing source leases
   for arbitrary demand makes host hits cold-miss (`Radix.restore` checks pathHeld),
   and multiple queued users of one prefix conflict. Demand protection therefore
   needs a separate policy retention mechanism, not another source lease.
3. `Archive.lookup` returns a reusable index. An active read job pins its entry with
   `readers`; a naked index is not durable. A speculative job must establish the
   existing reader pin immediately, retain it through disk drain, and associate it
   with the queued operation's generation. Index equality alone is insufficient.
4. `Archive.advanceWith` currently polls disk completion, hashes it and starts a
   GPU upload in the same path. Calling it for prefetch is **not staging-only**.
   A separate upload permission is required. Cancellation must still drain tickets
   even when upload/optional-issue permissions are false.
5. Eight staging tickets are already budgeted once; six optional writes leave two
   for foreground reads. Speculative reads must not consume that last reserve.
   Prefetch and optional writes must share a six-ticket *combined* allowance;
   foreground reads can use all available tickets. Existing held tickets drain
   under preemption; no unsafe early release and no new ticket allocation.
6. D.1's source lease can already be active when queued demand appears. Canceling
   it does not synchronously release it. A host hit must wait for acknowledged
   release, not fall through to recomputation merely because the source is leased.
   The existing decode `CacheReclaimPending` protocol needs a begin counterpart.
7. Reserving every queued prefix indefinitely can deadlock live allocations. Demand
   protection is a bounded eviction preference, **not immutable DMA ownership**.
   Foreground allocation may explicitly break that preference after exhausting
   unprotected victims; it may never break a source lease.

## Selected scope / rejected shortcuts

Implement one oldest queued-begin lookahead owner first, with explicit opt-in
configuration. Protect its best local prefix (including a host-containing prefix)
against optional demotion/drop. Protection does not forbid promotion/rename and
never retains request token memory. Reconcile the selected generation at every
scheduler boundary before discretionary maintenance. Ordinary source leases retain
full safety precedence.

Disk prefetch is explicitly **bounded staging-window read-ahead**, initially one
or two transfer chunks for that one request, not full-image RAM materialization.
It performs disk reads only, no model selection, GPU upload, page-table mapping,
snapshot restore or publication. The existing foreground restore consumes those
same tickets after target pages are admitted; then it reads the remainder normally.
For window `W` and chunk bytes `B`, extra staged ownership is at most `W*B` within
the existing eight-ticket allocation, not additional host RAM. A record may be
hundreds of MiB or multiple GiB, so a first-window prefetch cannot be represented as
a complete host-tier hit or as CachedAttention's whole-conversation prefetch.

Do not add a second Vulkan queue, rewrite layouts, truncate context, lower precision,
copy caller tokens per request, introduce hidden heap allocation, or use a C++ runtime.
No default change or speedup assumption follows from the paper mechanism.

## State transitions to settle in the functional spec before implementation

- queued generation selected → local demand preference installed; optionally pin
  disk record and acquire bounded tickets → disk-owned/staged → foreground handoff
  (same job/tickets, upload now allowed) → ordinary verified restore → release.
- queued generation canceled/reused, better local prefix appears, record becomes
  corrupt, or stop → forbid new issue → drain exact old tickets → release old
  record pin → acknowledge generation retirement. A reused slot cannot attach the
  old job or publish its position. No request callback can hold borrowed tokens
  during asynchronous I/O.
- foreground admission cannot allocate target pages: cancel optional prefetch and
  drain before allowing a cold fallback or selecting a different record; do not
  turn a staging job into an unbounded memory reservation.
- source conflicts with demand: latch source cancellation, preserve source until
  drain, defer only affected begin on an epoch; independent decode continues.
- foreground I/O arrives while prefetch owns optional tickets: stop optional issue;
  drain/cancel non-consumed prefetch if needed for the reserve. Never wait for a
  whole image and never let a speculative owner prevent stop.

The API and exact retry/cancellation ordering are **not yet finalized**. No native
D.2 implementation may start merely from this audit.

## Executable acceptance mechanism required before code

Create an independent object/event model and deterministic fixtures, not a port of
native radix/job state. Exact token-tuples determine longest valid prefix, explicit
request generation names determine lifetimes, and ticket owners are sets. Enumerate
nested/divergent/partial prefixes, duplicate demand, queued cancellation, slot reuse,
source conflicts, hard allocation pressure, foreground preemption, out-of-order disk
completion, corruption, stop and handoff. Compare protected-object sets, copied/
read/uploaded bytes, record readers, ticket occupancy, returned positions and final
zero ownership against native public interfaces. A speculative trace must have
**zero upload calls before handoff**. Mutating upload permission or generation checks
must fail the fixture. Preserve pre-fix failures.

The available paper/distributed reference implementations are not an executable
oracle for native Vulkan/radix ownership transitions; document that non-equivalence.
Continue independent numerical FP64/libllama and same-weight full-state/vocabulary
checks rather than substituting plausible text. Register CPU tests through Bazel;
real host/disk model checks at 257/80k, packed/stop/lease tests, spill/device/host
GPU gates and repeatable component measurements remain mandatory. Repeat controlled
HTTP against tuned Vulkan/HIP with exact native identity and explicitly reported
reference quality equivalence; different generated streams cannot silently count
as a performance win.
