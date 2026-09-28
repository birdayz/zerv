# Bounded asynchronous scratch-file transport (18d.6b)

Scope: `src/storage`, Linux only, no GPU dependency. Next cache/scheduler integration
is separate. User-approved RAM staging, no raw disk, module, mount, or driver change.
Research: [staged storage](../research/2026-09-28-nvme-store.md),
[buffer measurement](../bench/2026-09-28-nvme-buffers.md).

## Contract

`Store.create(allocator, options, memory)` returns a stable allocated owner. Options:
trusted parent directory fd (borrowed only during creation), new filename, file byte budget,
staging slot size and optional explicit direct-I/O alignment (memory and offset). `resolveAlignment` exposes the same
reported/configured validation for preflight and hardware-independent tests.
No filesystem-specific ioctls, policy detection or inode-attribute changes. Memory is a
borrowed anonymous/aligned host span, divisible by slot size; 1..32 slots. Slot size
is 4 KiB..64 MiB and a multiple of 4 KiB. File budget is a positive multiple of 4 KiB,
at most 1 TiB. Arithmetic is checked before allocating or touching the filesystem.

Create `O_EXCL|O_NOFOLLOW|O_CLOEXEC|O_DIRECT|O_RDWR`, mode 0600. Never open/truncate an
existing object (including a symlink). Immediately unlink our new file: it is private
scratch until fd close, not persistent state. An unlink failure closes the fd and
reports `UnlinkFailed`; the just-created empty pathname may remain for the operator.
Preallocate the entire budget before accepting I/O; failure closes everything.
Use generic `statx(DIOALIGN)` where supplied. If that field is absent, require
explicit configured alignment instead of guessing a filesystem-specific value.
All alignment values must be positive powers of two; configuration may strengthen,
but never weaken, reported requirements. Reported zero alignment means unsupported
and cannot be overridden. No buffered fallback is explicitly requested by the store.
Require every slot's address/size and file size to satisfy the reported alignments.
Deployment owns filesystem preparation: O_DIRECT is requested but a filesystem may
internally fall back to buffered I/O. This is documented, not inferred/changed by the
transport. See [deployment guidance](../deployment/nvme-scratch.md) for the target
host's NOCOW directory and alignment.

A dedicated OS worker thread exclusively owns an ordinary io_uring (no SQPOLL or
registered buffers). The scheduler only changes bounded descriptors and wakes the
worker. Submission, filesystem work and blocking completion waits happen there.
No allocator calls after initialization; caller/GPU may use a held buffer while the
worker uses other slots. The borrowed span may be Vulkan-imported by the caller,
which must finish and destroy that import before freeing the span.

## Ticket and ownership states

One externally serialized caller, one worker. Tickets identify `(slot, generation)`.
All public methods except create/destroy are called by that one caller, not concurrently.
The worker never uses caller GPU objects. Release/acquire atomic transitions publish
metadata and bytes:

`free → held → queued → active → done → free`

- `acquire`: reserve one free slot or `QueueFull`; increment generation without wrapping.
- `buffer(ticket)`: mutable borrowed span only in held/done; reject busy/stale tickets.
  Previously obtained spans must not be accessed after submission until completion.
- `submit(ticket, read|write, file_offset, length)`: only held, positive aligned length
  <= slot capacity, checked within the file budget. Write source bytes must be ready
  (including any GPU completion/host visibility); ownership passes to the worker.
- File ranges of outstanding or unacknowledged requests must not overlap if either
  request writes. Overlapping reads are allowed. A done request retains its reservation
  until release. This prevents reuse before the consumer acknowledges completion.
- `poll`: null for queued/active, `Completion` for done; held is invalid. Completion
  records expected and actual bytes / negative kernel errno. Success means **exact**
  count, never merely a nonnegative result. Short reads/writes and ordinary I/O failures
  are reported after that operation no longer owns the memory; they are not retried
  or published as cache entries. Read bytes are usable only on exact success.
- `release`: held or done only; frees ticket and range. No cancel-and-free operation.
  Cancellation means retaining the ticket until completion and then discarding it.
- `destroy`: refuse if any ticket remains held/queued/active/done. After all releases,
  wake and join worker, destroy ring, close scratch fd, free metadata. Never free the
  borrowed span. The worker is not force-cancelled, even on slow storage.

The kernel ring has no more outstanding requests than staging slots. Worker batches
current queued requests and drains completions, publishing each finished slot. EINTR
is retried. Unexpected ring-control failure, invalid/duplicate completion or loss of
ownership knowledge is fail-stop (process termination), **not** unwinding/freeing
possibly DMA-owned buffers. No bounded-time guarantee on kernel I/O; the scheduler
can continue unrelated work while requests are pending.

## Independent correctness and measurement gates

Before native implementation, generate a fixed seeded byte fixture using pinned Python
`os.pwrite` + `os.pread`, record generator/fixture SHA256 and seed. Native transport reads
bytes written by synchronous positional syscalls, and its writes are read by those
syscalls, compared exactly to the fixture. The reference never calls our transport.
Ordinary tests consume checked-in bytes, no third-party libraries or downloads.

Debug + ReleaseFast: queue full, bad options/alignment/offset/length/overflow, stale
and duplicate tickets, busy buffer/release/destroy, overlapping writes vs concurrent
reads, held cancellation, >capacity repeated batches, file collision/symlink protection,
failure cleanup, exact byte oracle, no allocation leak, short/negative completions.
Integration fault tests may use the owned fd only while no I/O is pending to truncate
or close it deliberately; production callers must never do so.

Component benchmark: caller-owned 4-KiB-aligned memory, 8 MiB chunks, 1 GiB file,
depths 1 and 8, warmup and repeated trials; compare exact work with synchronous
positional syscalls (QD1 semantics for that reference). Record raw times, variance,
build/source hashes, filesystem and limits. QD8 is a concurrency comparison, not
same-depth latency. No serving/llama-server speedup claim until cache integration.
