# Radix cache source leases (18d.7c.1)

Specified before implementation, 2026-09-28. Supporting increment of
[pressure tiering](../design/async-tiering.md); not a new disk policy by itself.
[Research](../research/2026-09-28-cache-sources.md).

## Interface and ownership

A `Handle { index:u32, generation:u64 }` identifies a hot checkpoint incarnation,
not a reusable physical snapshot slot or disk record. `Source` separately returns
`snapshot:u32`, immutable borrowed token/page-id slices, and a `Lease { handle,
serial:u64 }`. The current hot-store allocator assigns snapshot index = entry index;
callers must use Source.snapshot, not derive a snapshot from the logical handle.
Disk records retain their independent archive namespace and lifetime. This does not
adopt the older standalone residency.Table or allocate disk-only hot entries.

`Cache.coldSource()` returns the LRU unleased radix leaf's handle, or null.
`acquireSource(handle)` allows any live radix checkpoint, rejects a stale generation
(InvalidSource), a second outstanding lease of the same checkpoint (Busy), and
serial/counter overflow (SourceExhausted). It increments protection on the selected
node AND every ancestor, then returns borrowed slices. Different source checkpoints
can hold overlapping paths. Each source has at most one outstanding lease, so the
number of leases is bounded by configured hot slots. No post-init allocation.

`releaseSource(lease)` validates index, generation, serial and active state BEFORE
changing counters; stale, forged or duplicate release fails InvalidSource. Release
is legal only after ALL GPU and disk references to this source have drained, even
on cancellation/failure. This is the adapter's responsibility; a metadata lease
cannot observe hardware completion. The path cannot change while held, so release
walks the original ancestors without a second per-transfer path allocation.

Flat policy returns null for coldSource and UnsupportedSource for acquire/release;
there is no silent conversion or unprotected flat capture. APIs are scheduler-thread
only; these are not atomic counters or a concurrent access protocol. Deinit with
active source/path ownership fails-stop, like existing pending archive teardown.

## Protected mutations

- Removal/eviction/snapshot reuse must not select a protected node. At capacity,
  insertion may drop another unleased leaf; if none exists it skips the optional
  checkpoint rather than waiting for persistence.
- Inserting a NEW prefix which is a proper ancestor of any protected node is
  skipped before any device operation. Otherwise reparenting and transferring
  segment ownership would invalidate the release path. Duplicate insertion and
  adding descendants are safe: they do not modify held source bytes or ancestors.
- Demotion skips protected segments. A restore which requires host-page promotion
  on a protected path is a cold miss (0) and must not rename/free those host pages.
  GPU-resident restores remain valid read-only consumers. LRU touches are allowed.
- Generation increments on successful fill, never wraps. Exhausted entries cannot
  be reused. Transfer serial increments on acquisition, never wraps within a
  generation. Failed acquisition/release leaves all ownership unchanged.

The lease freezes metadata, snapshot reuse and page residency, **not unused bytes
of a partial KV page**. Only positions below Source.tokens.len are immutable. A
future byte adapter must canonicalize the unused tail with explicit Vulkan
read/write dependencies, or freeze a budgeted private page; this API is not
permission to race a whole-page copy with token append.

## Acceptance and measurement

Before native code, generate independent Python traces using a dictionary of live
token tuples and sets of active source paths (derive ancestor protection from
prefix relations, not native parent/counter arrays). Check every outcome, live
prefix, generation, serial, active lease and derived path count after every step.
Seed 70701; directed reparent, overlapping paths, capacity pressure, stale release,
slot reuse and random operations. Fixture generator/hash are checked in; normal
Bazel tests consume goldens without Python or third_party.

Native directed tests additionally use the existing fake device to test mixed
host/GPU ancestry, blocked promotion/demotion, unaffected sibling progress, borrowed
slice stability, exhaustion, allocation failure and no steady allocations. Run all
CPU tests in both modes plus Python/format and repeated source tests. Negative
control must detect omitted ancestor protection. Measure repeated lease roundtrips
at path depths 1/8/64/256, with exact metadata validation outside timing. No equivalent
llama-server API exists; this is component overhead, not a competitor win. Integrated
model/HTTP and tuned competitor gates belong to C's source adapter/policy increment.
