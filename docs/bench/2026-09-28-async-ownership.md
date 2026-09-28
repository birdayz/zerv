# Per-buffer asynchronous GPU ownership — 2026-09-28

Increment 18d.7a of the [tiering plan](../design/async-tiering.md), **not an archive
policy change or a serving speedup**. Research/spec preceded code. A global pending
command counter used to reject mapping even unrelated host buffers. That would
break independent decode preparation while an archive transfer was pending.

Implemented: buffer-local pending-use guards for direct copies and kernel bindings;
recorded lifetime references unchanged; nonblocking `Commands.poll()` using a
zero-time fence wait. Successful acknowledgment releases only that command's uses;
timeouts retain them. Tests cover two owners, both acknowledgment orders, unrelated
host writes, kernel-only references, replay, and double-poll/reset/destruction guards.
No new queue, shaders, driver interface or arithmetic.

## Verification actually run

[Gate logs](data/2026-09-28-async-ownership-gates/): CPU/Python/format 81/81 (11
executed in the first run, rest cached); GPU Debug/ReleaseFast/spill 3/3, all three
executed across the initial and completion calls; host-driver Debug/ReleaseFast 2/2,
both executed. Each GPU binary has 44 tests. Benchmark prerequisite gates passed
again, cached. Independent C Vulkan / scalar affine and transfer golden outputs
all match; no validation layer or deliberately induced hardware loss.

Two command-use failures retained: `bazel --version` is not a valid startup option
(use `bazel version`); `bazel run //bazel:zig -- fmt` with relative source paths runs
from runfiles (FileNotFound), so the corrected invocation uses absolute paths.
The first GPU tool call exhausted its **120-second tool deadline**, not a GPU test
failure. Its Bazel server test kept running. The next serialized Bazel invocation
waited for it and completed all gates; no simultaneous GPU workloads were started.
The longer GPU suite took roughly 40 minutes across two runtimes plus spill checks.
This does not reopen the retired hang investigation.

## Repeated matched component measurement

[Manifest, raw trials, hashes, full source snapshots and summary](data/2026-09-28-async-ownership-driver/).
RX 7900 XTX, host Vulkan runtime identified in manifest; pinned C orchestration
oracle from the build graph, same device/queue/memory types/bytes/shader as native.
CPU affinity 10, three alternating native/reference process rounds, three warmups,
seven trials per process (21 observations per cell). Wall-time submit+wait loops,
not GPU-only timestamps. Full independent output hashes checked after every run.

| Operation | Native median ± sample SD | C reference median ± sample SD |
|---|---:|---:|
| affine 65 words | 46.40 ± 1.84 µs | 46.28 ± 2.11 µs |
| affine 5,120 words | 45.30 ± 1.23 µs | 45.27 ± 1.86 µs |
| affine 1,048,576 words | 53.91 ± 1.75 µs | 52.54 ± 1.30 µs |
| 256-byte roundtrip | 47.20 ± 2.03 µs | 48.62 ± 24.41 µs |
| 1 MiB roundtrip | 282.00 ± 6.42 µs | 283.31 ± 13.42 µs |
| 64 MiB roundtrip | 11.657 ± 0.293 ms | 11.593 ± 0.127 ms |

No clear broad driver advantage. The large affine case loses 2.6% against C;
64 MiB roundtrip loses 0.55%. This is **not** a before/after isolation of counter
cost, and one affine binding set does not bound large model command overhead.
No equivalent llama-server raw-fence API; serving comparison belongs to integration.

```sh
bazel test //...
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu
tools/py bench/run_gpu_driver.py --output docs/bench/data/NEW_DRIVER_RUN
```

A's gates are closed. At this measurement point snapshot save/load, page demotion/
promotion and archive production calls were still synchronous. Subsequent archive
submit/poll/drain gates are in the [B report](2026-09-28-async-archive.md); snapshot/
page transfers remain synchronous. Pressure policy and actual copy/compute overlap
are not shipped.
