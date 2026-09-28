# Bounded asynchronous disk transport — 2026-09-28

## Scope and boundary

Completed component 18d.6b: `src/storage`, a bounded worker-owned io_uring transport
using caller-owned RAM staging. This is **not an integrated NVMe prefix cache**.
No GPU transfer, scheduler overlap, model execution or serving speed is measured here.

Per the user's correction, the core has **no filesystem-specific detection, ioctls
or attribute mutation**. It uses `STATX_DIOALIGN` or explicit memory/offset alignment;
configuration cannot weaken reported requirements. Filesystem preparation belongs
in [deployment guidance](../deployment/nvme-scratch.md), not the engine.
See the [contract](../specs/nvme-store.md) and [research/source ledger](../research/2026-09-28-nvme-store.md).

## Setup and reproduction

Observed host: Ryzen 3900X, Samsung 990 PRO, Linux 7.2.6-arch2-1, Zig 0.16.0,
Bazel 9.2.0. Scratch directory `third_party/nvme-probe` was already prepared NOCOW
on this host's btrfs volume. No filesystem attributes, mounts, drivers or other disks
were changed by this run. Alignment was explicitly configured to 4096 bytes.

```sh
tools/py bench/run_storage.py --scratch-dir third_party/nvme-probe \
  --direct-alignment 4096 --output docs/bench/data/2026-09-28-nvme-store-final
```

For repetition, choose a new output directory. The harness builds through
`tools/zerv_build.py` and gates on `bazel test //...`. Each subprocess uses a bounded
1 GiB exclusive temporary file, 8 MiB chunks, one warmup and five trials per path,
alternating path order. Worker queue depths follow ABBA: 1, 8, 8, 1. Staging is
8 MiB at QD1 and 64 MiB at QD8. The independent synchronous pread/pwrite path is
**always QD1**, including in the QD8 rounds. Byte checks cross the syscall/worker
boundary in both directions; every retained trial reports exact bytes. The caller
busy-polls in this component benchmark; this is not the planned serving policy.
`fsync` is outside write timing. No OS cache flush, drive preconditioning or durable
write timing is claimed; these short runs do not establish sustained SSD writes.

[Final manifest](data/2026-09-28-nvme-store-final/manifest.json) records the commands,
source/build/binary hashes, kernel, mount and memory state. The working tree was
uncommitted on base `51a6053`; use its source hashes, not that base commit alone.
[Raw trials](data/2026-09-28-nvme-store-final/raw.jsonl),
[summary](data/2026-09-28-nvme-store-final/summary.json), and subprocess logs are
retained alongside it. There are 40 measured trials, excluding warmups.

## Results

Decimal GB/s, mean ± sample standard deviation, n=10 per row:

| Worker depth in round | Path | Read | Write |
|---|---|---:|---:|
| 1 | Worker QD1 | 6.113 ± 0.588 | 5.259 ± 1.486 |
| 1 | Synchronous QD1 | 6.443 ± 0.359 | 5.804 ± 1.321 |
| 8 | Worker QD8 | 7.179 ± 0.220 | 6.738 ± 0.196 |
| 8 | Synchronous QD1 | 6.506 ± 0.510 | 6.414 ± 0.276 |

The worker loses to synchronous I/O at equivalent QD1 in these means. QD8 raises
observed throughput but is **not an equivalent-depth speedup** over the synchronous
reference. QD1 write variance is large: worker range 2.135–6.694 GB/s. Do not infer a
stable write-rate guarantee or a serving win from this component result.

A [prior generic-core run](data/2026-09-28-nvme-store-generic/manifest.json)
([summary](data/2026-09-28-nvme-store-generic/summary.json)) also passed all 40 byte
trials, before exposing the alignment resolver and relocating its test into the
executed test root. Worker reads were 5.80 ± 0.34 GB/s at QD1 and 6.66 ± 0.40 at QD8;
writes were 3.88 ± 2.04 and 5.59 ± 1.92. Retain this run-to-run variance rather than
selecting only the higher final result. The
[first attempt](data/2026-09-28-nvme-store/interrupted.json) was canceled during its
build gate on the user's filesystem-boundary correction; it produced no timing
trials. Its filesystem-specific implementation was removed, not benchmarked.

## Correctness and verification

Independent fixture: Python 3.14.4 `os.pwrite/os.pread`, seed 1592596694, sixteen
4096-byte blocks; payload SHA256
`d6a79d2a02fbf42adc6b8411a69d78ec8098980d69ab6dc65194370e23ce7bd2`.
Generate explicitly with `tools/py tests/reference/generate_disk_fixture.py`.
Native tests verify generator/payload hashes and use synchronous Linux syscalls
as the independent reader/writer for the asynchronous transport.

Executed checks on the final native tree:

- `bazel test //...`: 77/77 passed; final benchmark gate reused cached results.
- `bazel test --runs_per_test=20 //tests:storage //tests:storage_release_fast`:
  both targets passed all 20 runs each. Eight cases per mode include independent
  bytes, bounds/alignment, overlap, stale tickets, ownership/teardown, short and
  negative completions, collision/symlink protection, allocation failure and
  allocation-free steady state. Repeated four-slot waves and depth32 are covered.
- `bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills`: 3/3 passed
  from cache; the development driver runtime was rebuilt.
- `tools/py tools/zerv_build.py --test-host-gpu`: 2/2 passed from cache.
- `git diff --check`: passed; core search found no btrfs/NOCOW/ioctl handling.

Initial failures were retained in the research: wrong SHA helper import,
missing reported DIO alignment, and futex WAIT's missing explicit null timeout.
An imported-package test was not executed by the root test runner; moving alignment
coverage into `tests/storage.zig` fixed that coverage gap. None of these failures
is hidden by the final pass counts.

## Remaining integration gates

Snapshot residency/slot decoupling, disk extents and cache records, asynchronous
restore/checkpoint state, scheduler wake/cancellation, and server configuration are
not implemented. No production server flags are exposed for this standalone store.
Full-model exactness and interleaved serving measurements against disk-off,
host-only and a tuned llama-server are still required. A serving engine does not
expose this arbitrary file-buffer operation, so syscall component comparison is
not a substitute for those future integration gates.
