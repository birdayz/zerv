# Background cache sources: ownership audit

2026-09-28, before 18d.7c.1 code. Baseline `3c8cfbb`.

## Sources and scope

Re-read Mooncake offload-on-evict source/tests at
`f8b50d5f1a2fa4d33d029ed1e82188de70b137f0`: `mooncake-store/src/master_service.cpp`
BatchEvict, `tests/offload_on_evict_test.cpp` and
`tests/master_service_offload_scenario_test.cpp`. Pending offload preserves the
source, existing backing avoids rewrite, and insertion alone need not enqueue SSD.
Re-read LMCache arXiv 2510.09665v2 §5.2–5.3: explicit start/wait, shared reference
counts and a bounded duplication window. URLs, exact revisions, local third_party
paths and hashes: [paper ledger](2026-09-28-kv-tier-papers/sources.json).
These are not a reference implementation of our hybrid-state radix cache; an
independent finite-model oracle must test the zerv-specific ownership contract.

## Observed local semantics

`kvcache.Radix` stores full logical page lists but owns only each node's suffix;
leading pages belong to ancestors. `rename` changes every list referencing a moved
page. `take` can insert an ancestor of existing nodes and transfer segment pins.
`restore` promotes whole paths; `makeRoom` demotes internal nodes, not just leaves.
Consequently protecting only a selected leaf, or retaining only its suffix, is
insufficient. All ancestors must remain resident at the same physical locations.

Snapshot index currently equals the hot-store index, and a capacity drop can reuse
that slot immediately. Introduce a generation-qualified logical source identity,
explicit physical snapshot field and per-acquisition serial; do not reuse disk IDs
as hot indices. Disk-only prefixes remain in the independent full-image archive;
there is no need to duplicate them in the hot tree or force adoption of the older
standalone residency table.

Full GPU pages can be shared by unrelated branches; their multiple pins already
prevent host demotion (demotePlan requires exactly one pin and no sequence mask).
Host pages are singly owned by a segment and shared only through its ancestry.
Protecting paths therefore blocks every alias-changing promotion/demotion relevant
to a held source. GPU-only hits copy a partial page and read snapshots without
mutating the leased metadata; host promotion must be deferred/missed. Reparenting
an already protected node must be prevented, or release could walk the wrong path.

Resolved first increment: protect source plus ancestors; reject a second active
lease of one source; allow overlapping leases of different sources. Keep counters
in the Radix, so no external caller can accidentally forget a mutation guard.
Skip optional insertion if it would reparent a held path or all victims are held.
Callbacks remain synchronous in this metadata increment, preserving all existing
non-lease behavior. Alternative of copying an entire image into new host memory
would double up to 5.4 GB at 80k and is rejected; small metadata protection is enough
for source locations, but not for unused partial-page bytes.

## Partial-tail decision for the next byte-adapter increment

Proposed choice: canonical zero for positions >= checkpoint length. The stream
still contains whole logical pages in group order, with recurrent/conv snapshot
first. Exactness means exact valid KV/recurrent bytes, deterministic zero unused
tail and unchanged full-vocabulary continuation. This explicitly changes arbitrary
unused-tail bytes; do not claim byte identity to an uncanonicalized live image.
K is [head][dim][page], V is [head][page][dim], for each layer in a physical page.
For odd f16 prefix lengths, the last valid K half shares a 4-byte copy unit with
an invalid half: copy the aligned unit and clear only invalid half bytes after
acknowledgment. No conversion/rounding. Need an independent coordinate-based byte
fixture across f16/f32, odd tails, group splits and chunk boundaries before code.

Whole-page GPU reads overlap later token writes to unused bytes unless synchronized.
A source adapter must bracket transfer reads with compute→transfer AND
transfer→compute dependencies, plus transfer→host before canonicalization. The
current paused-source adapter only needs the latter; it must not simply be reused
with the source unpaused. Mixed host pages/snapshots need completed prior copies and
stable lease ownership. Canonicalization is a bounded CPU span operation, not extra
GPU allocation; performance must measure its cost and the transfer window. This
choice remains gated by the next byte fixture and actual production-source tests.

## Executable mechanism for c.1

`tests/reference/generate_cache_source_fixture.py` will model live prefixes as a
map and leases as sets of ancestors computed from prefix inclusion. Derive protection
counts from those sets, rather than duplicate native parent-chain mutation code.
Golden outcomes include live prefixes/generations/serials/path counts after each
operation, including rejected stale/duplicate completions. Tests also exercise the
actual Radix fake-device operations and allocation-failure paths. Exact comparison,
no numerical tolerance. No external runtime/model dependency or GPU job needed for
this metadata component. Measurement contract is in the preimplementation
[specification](../specs/cache-source-leases.md).

## C.2 adapter audit after c.1 gates (design only, not implemented)

- `Model.archiveSubmit` exports the mutable live slot, validates sequence masks
  and pauses it; it cannot directly accept Source.snapshot or host-tagged IDs.
  A separate source-submit entry point should validate snapshot bounds, page count,
  GPU pin/logical-page identity and host checkpoint ownership before recording.
  GPU page masks need not include the originating request: the lease is the owner.
- `Pool.host_owner` distinguishes cache (`checkpoint_owner`) from live swap slots,
  but demoteCommit does not set `host_logical`. Do not validate checkpoint host
  pages against that unmaintained field. Expose a checked Pool source-validation
  operation rather than duplicate the private sentinel in the model adapter.
- Mixed capture can CPU-copy already-host snapshot/page spans into the held staging
  ticket and submit only GPU spans. A wholly host quantum should use start/poll
  acknowledgment without a dummy GPU submission. Whole-buffer mapped guards are
  conservative: map before submitting this quantum; do not bypass ownership if
  other pending commands could still write that host allocation. D's asynchronous
  demotion may require finer range ownership; do not assume it is already solved.
- Canonicalization runs after device completion and before archive hashing. Track
  the quantum's offset/span through poll. A copy-start error must leave no DMA;
  terminal poll errors drain as in B. Existing stream size/order stays unchanged,
  but unused partial-tail bytes become canonical zero (explicit exactness contract).
- The disk owner needs a background job identity not tied to a live request slot.
  Proposed one extra bounded archive job (`slots + 1`, up to 65), one held Source,
  and explicit cache release after catalog drain. Current Archive option max is 64,
  Disk.positions has 64 entries and maps only slots*seq_pages; all three limits must
  change together if this design is selected. No request slot should be occupied
  merely to preserve a cached prefix. A later scheduler hook must progress and
  cancel/drain that job on stop independently of pending request I/O.
- Existing model checker goldens capture arbitrary unused tail bytes. New source
  tests must canonicalize expected goldens using the independently fixture-tested
  byte operation, continue/reset/reuse the originating slot while holding the
  source, demote an ancestor before acquisition to force mixed residency, then
  restore private permuted pages and compare all valid state and vocabulary rows.
  Existing paused-slot tests remain useful; do not silently weaken them.

Next pre-code gate: coordinate-based byte oracle across f16/f32, 128/256 pages,
odd token tails, KV group splits, aligned copy units crossing K/V/page/snapshot
boundaries, and arbitrary bounded stream windows. Native range-clearing should
intersect only tail pages (not scan every byte of every large record). A source
adapter prototype is blocked until that fixture/spec exists. No C.2 code shipped
by the c.1 change; no new performance result inferred from this audit.
