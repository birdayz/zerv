# Bounded archive windows (D.0b)

2026-09-28, specified before code against `472992c`. Active package `src/serve`.
[Paper review](../research/2026-09-28-kv-tier-papers.md), especially LMCache §5's
bounded/coalesced transfers and Pensieve's recovery priority; [D.0a negative result](../bench/2026-09-28-tiering-progress.md).
This changes neither admission nor prefetch/preparation semantics.

## Research and resolved limits

Read `storage.Store.create`, `Archive.init/startDevice/advanceWith`, `Disk.create/
pollSourceWith`, CLI memory budgeting, and the model checker. Storage already
accepts aligned 4 KiB–64 MiB tickets and up to 32 tickets. Archive derives block
count, digest metadata, offsets and padding from `store.slot_bytes`, and requires
the file size to be a multiple of that size. Its metadata cap remains 256 MiB.
Only the serving owner hard-codes 1 MiB/eight tickets. The diagnostic checker also
assumes 1 MiB when locating pending chunks and corrupting the last physical block;
these must use actual store ticket size. Its independent capture/poison quantum
stays 1 MiB so the comparison does not inherit the new transport layout.

Expose `--prefix-cache-disk-chunk-mib 1|2|4|8`, default **1** until measurement
justifies another default. Explicit use without disk is invalid. Resolve at startup:
C = chunk_mib × 2^20 bytes, staging = 8C, optional-write pending limit = six tickets,
two physically reserved tickets for incoming reads. Thus staging is 8/16/32/64 MiB,
read reserve 2/4/8/16 MiB, outstanding optional staging at most 6/12/24/48 MiB.
Add staging to the device allocation budget exactly once. Extra mmap alignment
slack remains at most 1 MiB. Reject other sizes, overflow, file_bytes == 0, and
file_bytes % C != 0 before resource creation. Keep current <=1 MiB direct/import
alignment limit (all allowed chunks are divisible by every accepted alignment).
Report effective chunk/staging/ticket reservation at startup.

An image B consumes ceil(B/C) blocks and ceil(B/C)C disk bytes; at most C−1 bytes
of zero padding per record. Larger chunks reduce operation/digest metadata count,
but consume more RAM, can waste more disk capacity, and increase individual
copy/hash/cancel-drain latency. No promise of higher throughput or lower TTFT.
The stream of valid model bytes and all numerical behavior remain unchanged.

No new allocation in steady state. One device quantum per callback, existing
in-chunk source-only permission and read-priority gate, one command owner.
Cancellation drains the current GPU/CPU ticket and pending disk owners before
release; it does not split or recycle an 8 MiB ticket prematurely. No chunk size
can make optional persistence outrank a pending read. Clean backing remains ready
after reads. Errors retain current fail-stop/optional-cache distinctions.

## Executable acceptance mechanism before code

Generate `tests/fixtures/archive/windows.json` using independent Python POSIX I/O
and hashlib (`tests/reference/generate_archive_windows.py`). For each C above,
use B=2C+17, byte(i)=(73i+floor(i/257)+19) mod 256, zero-pad the final block, write
blocks in reverse positional order, read back, and record per-block SHA256 plus
generator hash. This is an independent transport/layout oracle, not native output.
Native archive tests compare both recorded digests and actual physical-file reads
with these hashes, verify streamed restored bytes, and cancel with a deliberately
unacknowledged device ticket. Probe two physically free read tickets and exercise
allow_start=false acknowledgment without new optional issue. Existing delayed
read/write/error/ticket-reuse tests and C.2 coordinate goldens remain mandatory.

Unit-test all four resolved window budgets and invalid values (0,3,16,u32-max),
non-multiple/zero file size. Extend model checker/runner with explicit chunk size:
257/80k exact source bytes/full-vocabulary rows, mixed host/GPU source, packed
interleaving, last-GPU-quantum cancellation, retry and final-block corruption must
pass. Since 8 MiB can finish the small source inside the pack, later source polls
must safely no-op; do not weaken the positive in-chunk-progress assertion.

Run CPU/Python/fmt, repeated interfaces, GPU/spill and host GPU, FP64/libllama gates.
Measure repeated 257-token components and loaded zero-idle HTTP at 1/2/4/8 MiB,
with unchanged KV/host/disk budgets except explicitly disclosed staging. Match
native text and token counts, warmup, fresh servers, alternating order and three
rounds. Include tuned Vulkan and runnable RDNA3/HIP llama-server, source hold times,
write/read bytes, restores, cancellations, TTFT/gaps, memory and throughput variance.
Retain negative results. No closure of the full plan without subsequent D.1/D.2.
