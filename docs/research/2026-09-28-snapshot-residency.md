# Snapshot residency before disk-cache integration — 2026-09-28

## Observed coupling and scoped decision

`session.checkpoint.Store` explicitly equates entry i with model snapshot i;
`kvcache.fillEntry/restoreEntry` pass i directly to Device.save/load.
`Model.snapshot_store` is a fixed GPU/host buffer, capped at 64 snapshots. Snapshot
bytes are the recurrent and convolution state, plus pending MTP h when enabled
(`runtime.snapshotBytes`). KV pages are separate and radix ancestors own shared
segments. Thus merely adding disk page IDs would still cap retained prefixes at
RAM snapshot capacity. Increasing snapshot_slots allocates more RAM, not disk.

18d.6c's first runnable increment is **snapshot residency bookkeeping**, in
`src/session/residency.zig`: checkpoint identity, RAM snapshot slots and disk
snapshot records are independent bounded namespaces. It makes no GPU, filesystem,
model-layout or eviction-policy decisions. No existing cache is switched over yet.
The table is an I/O ownership protocol, not a second disk transport or a generic
resource framework. The next adapter must drive the existing storage worker.

## Primary/reference inspection

Read on 2026-09-28:
- Project specs `concurrent.md` sections 18d.4/18d.5 and `nvme-store.md`; current
  checkpoint/kvcache code, runtime save/load, engine adapter and snapshot allocation.
- SGLang v0.5.20, https://github.com/sgl-project/sglang/releases/tag/v0.5.20,
  existing release archive `third_party/sglang/sglang-v0.5.20.tar.gz`, SHA256
  `b3fa51d654d52962c5deb754999ae18aeea4d06f13d497fdc5cf0246a1dfac9b`.
  Local sources under `third_party/sglang/sglang-0.5.20/python/sglang/srt/mem_cache/`:
  - `hiradix_cache.py`, SHA256
    `85252392a4d033cff8d48e956704302dfd0a8b14890d858343f0184f2c3aa80c`:
    write_backup, _track/_finish_write_through_ack, write_backup_storage,
    writing_check/loading_check and load_back. Separate node identity from pool
    indices; keep pending-write/load maps; synchronize ack before releasing locks.
  - `mamba_radix_cache.py`, SHA256
    `1b0f5eafa71598f688154ec07d7eb168f9d556defa935f0f783f1622b590f0e3`:
    TreeNode has independent mamba_value/mamba_host_value, KV and recurrent locks;
    recurrent state protection is node-local while KV locks cover the path.

These are source-backed design observations, **not a runnable SGLang disk-snapshot
oracle**. Its torch/CUDA/distributed allocator and write-through tree semantics do
not implement our finite ticket protocol, error behavior or immutable record IDs.
Do not port its code or claim identical semantics. The independently formulated
Python set/dictionary ownership model below is the exact protocol oracle; the
existing Python POSIX fixture is the independent byte oracle. A future full cache
adapter still needs real-model and serving comparisons.

## Resolved semantics and alternatives

- Snapshot contents are opaque immutable bytes after save. No new arithmetic,
  dtype conversion, tensor layout or numerical tolerance: all physical copies
  must preserve every byte. The metadata table stores no model-specific sizes.
- A handle carries entry index + nonwrapping generation. A transfer also carries
  nonwrapping per-entry serial and both slot indices. Delayed or duplicate
  completion must not release the slots of a newer transfer/reused entry.
- Both source and destination are exclusively owned until the I/O coordinator
  observes completion of **all** chunks and any GPU copy. Cancellation drains,
  then reports failure; a timeout is not permission to call completion early.
- Failed writes preserve the only valid resident snapshot; failed reads preserve
  the immutable disk record and discard the partial destination. Never publish
  partially restored state. A read failure can be retried or the entry dropped.
- Successful reads retain disk backing, so a later demotion needs no second write.
  Policy may explicitly discard backing of a resident entry to reclaim disk room.
  Resident read leases prevent spill/drop until the consumer finishes GPU access.
- Use separate capacities and lowest-free-slot scans. O(entries + hot + disk) on
  reservation, O(hot/disk) on moves, O(1) completion/lease checks; no per-operation
  allocation. Start bounded at 65535 entries, matching existing radix u16 IDs.
  Measure scanning before adding a freelist/bitset. No per-token operation here.
- Do not add filesystem logic, persistent recovery, system-prefix pin policy,
  disk extent allocation or server flags as incidental work.

## Executable gates specified before native code

`tools/py tests/reference/generate_residency_fixture.py` will execute an independent
set/dictionary model, fixed random seed, retain directed failures and randomized
interleavings with full state after each operation. Native Debug/ReleaseFast must
match return/error, every entry and both ownership maps **exactly**. Fixtures record
Python version and generator/payload SHA256. No reference runtime dependency in tests.

Additional native gates: allocation-failure cleanup, no allocation after init,
nonwrapping counters, capacity/lease failures, old completion after slot reuse,
concurrent pending records completed out of order, cancellation/failure drain, and
physical worker writes/reads using the existing independent positional.bin bytes
with more retained records than resident slots. Synchronous pread checks writes;
poison freed hot slots before reading records back. This is not a model oracle.

Component harness: metadata-only reserve/save/spill/restore/lease/drop cycles at
64/256/1024 entries, 1 and 8 resident slots, explicit no-I/O completions, warmup +
five repetitions. Include source/build hashes and raw results. Not comparable to
SGLang GPU copies or llama-server serving latency; no serving claim.

## Remaining integration questions (block later code, not the table)

KV disk extents and deduplicated ancestor ownership; integrity record format and
validation; tokens/metadata memory budget beyond the current full-context-per-entry
layout; replacement of radix's fixed 256-entry pressure scratch; pending begin and
checkpoint operations and scheduler completion wake; snapshot byte transfer adapter
and shutdown ordering. These must be resolved before wiring disk into serving.
