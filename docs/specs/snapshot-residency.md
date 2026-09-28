# Bounded snapshot residency (18d.6c, first increment)

[Research and independent oracle](../research/2026-09-28-snapshot-residency.md).
This specifies session metadata only, **not an integrated disk cache**.

## Ownership and limits

`residency.Table.init(allocator, entries, hot_slots, disk_slots)` allocates metadata
once. Entries 1..65535; hot slots 1..entries; disk slots 0..entries. No tensor bytes,
file descriptors or GPU objects are owned here. One serialized scheduler-side
caller. Slots are opaque indices; disk byte offsets/alignment are the adapter's
responsibility. Caller guarantees saved bytes stay immutable and namespace indices
refer to the same backing allocations for the table's lifetime.

Every allocated slot has exactly one owner. All operation failures leave metadata
unchanged. Lowest free entry/slot wins, except entries whose generation is exhausted
are retired. The table never chooses a victim, blocks, allocates after init, or
initiates I/O. `deinit` rejects saves, transfers and resident leases still in flight;
otherwise it frees metadata (including live, quiescent entries).

`Handle { index:u32, generation:u64 }` rejects stale/invalid entries. Generations
start at 1 and never wrap. Entry phases: free, saving, resident, writing, disk,
reading. Fields: generation, transfer serial, phase, hot/disk slot, resident lease
count. `Transfer { handle, serial, hot, disk, direction }` identifies one entire
snapshot move, not one chunk. Serial starts at 1 per generation, never wraps.

## Operations / transitions

- `reserve()`: find free entry then free hot slot; returns handle in **saving**.
  NoEntry/NoHotSlot on exhaustion. The adapter saves into `inspect(handle).hot`.
- `saved(handle, success)`: saving only. Success publishes resident; failure frees
  hot slot and entry, preserving generation to reject stale handles.
- `inspect(handle)`: a metadata copy, not permission to access pending bytes.
- `acquireResident(handle)`: resident only, returns hot slot and increments leases.
  Caller can read only; finish GPU access before `releaseResident`. Nonresident
  phases return Busy; overflow is LeaseOverflow. Release without a lease fails
  InvalidState. Leases must be balanced by the externally serialized caller.
- `spill(handle)`: resident, no leases. If valid disk backing exists, immediately
  releases hot slot and becomes disk, returns null (no I/O). Otherwise reserves a
  disk slot, becomes writing, returns a write Transfer. NoDiskSlot/SerialExhausted
  leave it resident. Saving/reading/writing/disk or leases return Busy.
- `restore(handle)`: resident returns null. Disk reserves hot slot, becomes reading,
  returns read Transfer. Other phases return Busy. NoHotSlot/SerialExhausted do not
  change disk ownership.
- `complete(transfer, success)`: validates handle, serial, direction, phase and both
  indices before changing anything. Invalid handles return InvalidHandle; any other
  stale/forged completion returns InvalidTransfer. Write success releases hot and
  becomes disk; write failure frees disk and becomes resident. Read success becomes
  resident **retaining disk backing**; read failure releases hot and becomes disk.
- `discardBacking(handle)`: resident only, releases optional disk backing, retaining
  hot snapshot and leases. Cannot discard the only copy or a pending transfer.
- `drop(handle)`: resident/disk with no leases only; releases all slots and entry.
  Pending/leased entries return Busy. Generation is preserved; no silent cancel.

Only call `complete` when ALL submitted I/O and GPU accesses to source/destination
have ended. `success=true` additionally requires exact byte counts and the adapter's
integrity check. A partial read destination is not usable. This precondition cannot
be inferred by a metadata-only table. Cancellation drains before completing false.
Each pending transfer itself pins both slots; the table does not expose resident
leases on its in-flight source. Failed save uses `saved(false)`, not `drop`.

## Gates

Before implementation, execute the independent Python ownership model to generate
immutable full-state traces. Compare every return/error and metadata field, not
just final live counts. Additional Debug/ReleaseFast tests cover nonwrapping limits,
allocation failure/leaks and allocation-free operation; worker + independent POSIX
byte fixtures cover slot reuse and restore with more records than hot slots.
Component measurement follows the research protocol. Ordinary tests never need
SGLang/torch or third_party. Full-model logits/state, KV ownership, pending scheduler
operations and actual tuned llama-server benchmarks gate subsequent integration.
