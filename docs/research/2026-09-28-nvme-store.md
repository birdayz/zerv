# RAM-staged NVMe transport research — 2026-09-28

Scoped next increment 18d.6b, after the imported-buffer gate. Disk bytes are opaque,
unchanged; no model math, quantization or snapshot layout changes in this increment.

**Final boundary decision (user correction): no filesystem-specific logic in core.**
The initial implementation below queried btrfs sectorsize and configured NOCOW;
that implementation was removed. Core now uses generic statx or explicitly supplied
alignment, never filesystem ioctls/attribute changes. Btrfs preparation is solely
[deployment guidance](../deployment/nvme-scratch.md). The first measurement attempt
was canceled during its build gate when this correction arrived; no throughput
result was produced for that superseded implementation.

Sources reviewed: pinned Zig 0.16.0 SDK `lib/std/os/linux/IoUring.zig` (`init`,
`submit`, `flush_sq`, `enter`, `copy_cqes`, `read`, `write`), `linux.zig` statx,
openat/fallocate/futex/syscall declarations, and Thread spawn/join. Toolchain archive
pin is in MODULE.bazel; local inspected SDK `.tools/zig-x86_64-linux-0.16.0`.
Linux v7.2 source/provenance is in [P2P ledger](2026-09-28-nvme-p2p.md), particularly
`fs/btrfs/direct-io.c`, `fs/iomap/direct-io.c`, io_uring and GUP. The source version
is not the exact Arch 7.2.6 build; integration tests run the installed kernel.

Resolved transport semantics:

- O_DIRECT requires aligned memory, offsets and lengths; statx DIOALIGN gives
  requirements where supported. Otherwise deployment must supply explicit memory
  and offset alignment, or initialization fails. Configuration may strengthen but
  never weaken a reported requirement. Invalid/zero reported values are unsupported.
- btrfs checksummed direct writes fall back to buffered I/O in the reviewed
  `btrfs_direct_write`: NODATASUM is required for that path. Operators may choose a
  prepared NOCOW scratch directory; core does not change any filesystem attributes.
  This sacrifices filesystem data checksums on disposable cache data, not on other
  files. The integration must independently guard cache bytes/state; no persistent
  recovery is promised. Preparation and alignment guidance remain deployment-specific.
- A file is exclusively created and immediately unlinked. Holding its fd keeps
  extents alive; the last close releases them, including after process exit. No
  destructive cleanup by pathname after startup and no restart persistence.
- io_uring is asynchronous but submit can perform filesystem work; put even that
  call on a dedicated worker, never the model scheduler. One ring owner, ordinary
  requests, no registered memory or SQPOLL. Ordinary anonymous mappings support
  the same disk/GPU buffer, independently established by the 18d.6a probe.
- A CQE's signed result is byte count or negative errno. Short results are not
  successful cache records. Release buffer/range only after consuming its CQE;
  drain all outstanding requests before ring teardown. EINTR is retryable;
  uncertain ring state must not release DMA-owned memory. Fail-stop is explicit.
- A slot has one owner at a time. SPSC atomic release/acquire state transitions
  protect descriptor/result fields; a futex epoch plus scan-before-wait avoids
  lost wakes. The worker touches no GPU object. Caller-held buffers permit future
  upload/download overlap with disk work in other slots. The first implementation
  batches disk completions rather than designing a speculative general scheduler.
- Memory is fixed at caller's `slot_bytes * depth`, metadata O(depth), disk is
  fixed by preallocation. All errors at init clean up before work starts. No
  per-operation allocations. No caching policy, extent allocator, GPU code,
  snapshot slot decoupling or server configuration belongs in this increment.

Independent executable oracle: Python 3.14 pinned through `tools/py` writes seeded
bytes with `os.pwrite`, verifies them using `os.pread`, emits fixture + manifest.
Native tests then compare the worker to synchronous Linux pread/pwrite in both
directions. Exact bytes, zero numerical tolerance. This deliberately avoids a
transport self-round-trip as the only oracle. No external serving engine exposes
this arbitrary file-buffer operation. Future cache integration still requires
full-model exactness and actual tuned llama-server comparison.

## Initial failures and superseded approach

First native test failed `DirectIoUnsupported`: reviewed btrfs v7.2 `getattr` does
not report STATX_DIOALIGN. Rejecting every filesystem without that field therefore
rejected the target filesystem. Resolve through its existing read-only
`BTRFS_IOC_FS_INFO`: `include/uapi/linux/btrfs.h` defines a 1024-byte record with
sectorsize at offset 36; `check_direct_IO` requires both offset and iterator
alignment against that sectorsize. `fs/btrfs/ioctl.c` confirms NOCOW on a zero-size
regular file also sets NODATASUM. Added those primary sources to the pinned source
ledger before changing initialization; use the actual filesystem query, not a
hard-coded 4-KiB assumption. On other filesystems still require STATX_DIOALIGN.

Alternative: doing pread/pwrite on the scheduler would stall other generations.
Mapping the full file as host cache would make RAM/page-fault cost unbounded.
Raw storage/P2P was rejected by the user's current constraints; see P2P note.
The accepted bounded-worker contract is in [spec](../specs/nvme-store.md).

Execution record: generated the independent fixture before native implementation
with `tools/py tests/reference/generate_disk_fixture.py` (Python 3.14.4, seed
1592596694). Payload SHA256
`d6a79d2a02fbf42adc6b8411a69d78ec8098980d69ab6dc65194370e23ce7bd2`.
Manifest carries generator SHA256 and reference write order; native tests verify
both fingerprints. SDK source hashes: [ledger](2026-09-28-nvme-p2p/zig-storage-sha256sums.txt).

Second initial test failure was worker futex WAIT: unlike WAKE it requires a fourth
argument (timeout); using the three-argument syscall wrapper left that argument
unspecified. Fixed to pass an explicit null timeout with `futex_4arg`. Both modes
then passed. No ownership failure was hidden or converted to buffer reuse.
