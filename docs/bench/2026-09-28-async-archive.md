# Nonblocking archive GPU quanta — 2026-09-28

18d.7b of the [paper-informed tiering plan](../design/async-tiering.md).
**Execution change, not pressure-driven caching.** Checkpoint-triggered full-image
writes and the source-sequence pause remain. Snapshot save/load and KV demote/
promote still use their original synchronous paths. No new queue or demonstrated
hardware copy/compute overlap.

## Implemented lifecycle

`session.archive.Device` now has start/poll hooks. One bounded device quantum owns
its held/disk-done storage ticket across scheduler turns. Capture completion comes
before hashing and disk submission; upload completion comes before ticket release.
Cancellation and failures drain both types of ownership. `model.archiveSubmit`
records the unchanged exact stream and submits without waiting; `archiveCopy`
remains the explicitly synchronous convenience API for diagnostic callers.

The production disk owner polls with zero timeout and enforces the existing 60 s
model GPU deadline using the caller's monotonic std.Io clock. An unresolved DMA
error/deadline fails-stop rather than recycling memory. ModelBackend distinguishes
its known pending archive command from an unaccounted pending model command when
classifying errors. The preceding [per-buffer ownership increment](2026-09-28-async-ownership.md)
allows independent host I/O while an archive fence remains unacknowledged.

## Correctness evidence

[Unit/gate logs](data/2026-09-28-async-archive-gates/) and
[real-model manifests/rows](data/2026-09-28-async-archive-model/).

- CPU/Python/format 81/81. Archive tests pass 20 runs **in each mode** against the
  unchanged independent POSIX/padded SHA256 fixture and real storage worker.
- Delayed fake GPU really retains the borrowed span; it copies only on explicit
  completion. Directed tests cover no disk write before capture, no staging reuse
  before upload, generation-qualified retry, another slot's disk progress,
  start/completion errors, cancellation, short I/O and corruption.
- Negative control: replace the device-poll result with unconditional completion.
  Both test modes fail (DeviceBusy, then pending-owner cleanup panic). Restored the
  real poll afterward; no GPU workload used the negative control. Exact mutation
  and failure log: `negative-early-ack.log` in the gate directory.
- One initial ReleaseFast test failure was a **test synchronization assumption**:
  four CPU polls did not guarantee the real disk worker had completed a canceled
  read, so the second staging slot was not necessarily free. The corrected test
  waits for disk drain while deliberately keeping the GPU pending; 40 repeated
  mode runs pass. Initial failure retained in `unit-first.log`.
- Six 257-token trials and one **80,000-token** trial: every restored state byte
  exact (182,059,008 / **5,399,773,184 bytes**) after poisoning all state/KV and
  reversing destination pages; all four continuation vocabulary rows exact.
- Each production trial additionally decodes two independently checked vocabulary
  rows in another slot **before acknowledging the pending archive capture**.
  It tests a rejected unrelated operation without erroneously setting fatal.
  This proves correct independent execution eligibility, not simultaneous hardware
  copy/compute execution.
- Production cancellation now occurs after an actual GPU upload submission. Retry
  is exact; last-chunk disk corruption drains the partially uploaded destination,
  leaves position/mapped pages zero and invalidates the disk record.
- GPU runtime/spill 3/3 (Debug executed; ReleaseFast/spill cached), host GPU 2/2
  (Debug executed; ReleaseFast cached). Both modes had executed in A's gates;
  no shader or GPU-driver code changed between A and B. Final CPU/Python/format
  81/81 cached after restoring the negative control; first full B run executed
  18 targets. Logs distinguish execution from cache hits.
- Independent FP64/libllama modes 0 and 512: **337/337 greedy tokens**, every
  intermediate/logit tolerance passed (`oracle.json`, `oracle.log`).
- Fresh HTTP comparison: all 120 responses complete; **72 native outputs identical**
  across off/host/disk and also identical to the earlier synchronous baseline.
  `report_table.py` checks that cross-run identity, not merely successful HTTP status.

## Model-owner timings and limitations

The first 257 trial is warmup, followed by five measured trials. All are exact.
The new write bracket includes the added two independent decode rows and their
reset/setup while capture is pending: **not directly comparable to the old isolated
write bracket**. Restore brackets are unchanged; wall time includes hashing,
CPU staging, file I/O and scheduler-style 100 µs idle polling, not raw NVMe bandwidth.

Five short measured trials: write bracket **311.668 ± 6.915 ms**, restore
**191.790 ± 4.094 ms** (sample standard deviations). Earlier synchronous restore:
183.774 ± 12.621 ms. No short restore improvement established.

One 80k sample: write bracket 10.232 s, restore 6.826 s (old synchronous-owner
sample restore 5.357 s). Retain the regression; do not claim asynchronous submission
alone is faster. Zero-time polls reported 67,802 incomplete checks across all 80k
capture/restore/cancel/corruption operations. Maximum successful callback wall time
4.778 ms, including 1 MiB hashing/CPU work and possible host scheduling delay;
this is neither a GPU fence duration nor a guaranteed latency bound.

## Component and serving measurements

[CPU/disk component manifest/raw trials](data/2026-09-28-async-archive-component/):
five trials after warmup, all exact. Write **1.12456 ± 0.05645 GB/s**, read
**1.54451 ± 0.08664 GB/s**, versus earlier 1.1312 / 1.5308 GB/s. This measures
archive work with CPU-copy reference hooks and integrity checks, not raw NVMe or
GPU bandwidth; no equivalent llama-server component exists.

[Serving manifest, raw requests and logs](data/2026-09-28-async-archive-serving/),
including `validated-summary.json`, `report_table.py` and `report-table.log`.
RX 7900 XTX, same Q4_0 GGUF (SHA256 `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`),
Bazel-built native and llama-server; binary hashes, full commands, driver/environment
and workload hashes are in the manifests. Prepared `third_party/nvme-probe` used
unchanged, direct alignment 4096. No other GPU job ran alongside these trials.

Protocol: `multiturn-distinct-v1.json`, four distinct ~8.5k-token prompts, two
serving slots, 12,288 context per slot, two phased turns. Three fresh process rounds
per configuration, alternating configurations, short warmup before measurement.
Native KV pool 192 pages, 4 GiB swap; off/disk have two hot snapshots, host has eight
(**more RAM**, not an equal-resident-capacity comparison). Disk budget 8 GiB / 16
records, 8 MiB imported staging. Llama: 8 GiB RAM cache / eight checkpoints, flash
attention with unified KV at batch 2048 or batch 512. See manifest for all defaults
and effective flags. Reported cells are means ± sample SD of three process-level
statistics, not pooled request quantiles.

| Configuration | Cold TTFT p50 (s) | Reuse TTFT p50 (s) | Wall (s) | Aggregate output tok/s | SSE gap p99 (ms) |
|---|---:|---:|---:|---:|---:|
| Native off | 24.691 ± .511 | 20.025 ± 4.413 | 65.086 ± 4.523 | 9.155 ± .612 | 145.728 ± 1.096 |
| Native host | 24.789 ± 1.598 | 3.254 ± .616 | 48.228 ± .611 | 12.318 ± .156 | 145.278 ± .777 |
| Native disk | 41.029 ± 1.834 | 9.746 ± .562 | 59.501 ± 2.918 | 9.999 ± .477 | 23.123 ± 1.811 |
| llama FA unified, b2048 | 30.884 ± 1.288 | 6.555 ± 1.160 | 52.937 ± 1.925 | 11.304 ± .704 | 1802.162 ± 218.914 |
| llama FA b512 | 31.815 ± 1.978 | 4.897 ± .177 | 51.610 ± 1.612 | 11.580 ± .817 | 598.648 ± 46.704 |

Every disk trial: 13 writes / **8,780,775,424 bytes written**, three restores, two
evictions; zero skips/failures/cancellations. Earlier synchronous disk wall was
58.939 s, versus 59.501 s now, within run variation: **no serving speedup established**.
Host caching and both tuned llama configurations still finish faster overall.
Nonblocking ownership is the achieved behavior, not a claim of faster inference.

Aggregate throughput includes prefill/checkpoint time. TTFT includes queueing and
checkpoint work, not isolated prefill. SSE delta gaps need not be individual token
gaps; checkpoint pauses change request overlap, so the low disk gap is not proof
of faster decode. Maximum observed total sysfs VRAM: native ~18.247 GB, llama
~17.818 GB (desktop/other allocations included, not isolated engine memory).
Per-process RSS/HWM are retained in the summary; RSS excludes driver-owned host
allocations and HWM includes weight loading. No reduced-memory claim follows.

The [earlier same-day competitor follow-up](2026-09-28-disk-prefix-serving.md)
includes vLLM: different W4A16 weights/FP8 KV, with an empty second-turn response in
every warm round. Not rerun here and not quality-equivalent. SGLang remains unrun;
these are limitations of the competitor set, not evidence that none is faster.

## Reproduction

```sh
bazel test //tests:archive //tests:archive_release_fast --runs_per_test=20
bazel test //...
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu
tools/py bench/run_archive_model.py --tokens 257 257 257 257 257 257 80000 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/NEW_MODEL_RUN
tools/py tools/verify_model.py \
  --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/NEW_ASYNC_ORACLE \
  --report docs/bench/data/NEW_ORACLE.json --modes 0,512 --runtime host
tools/py bench/run_archive.py --scratch-dir third_party/nvme-probe \
  --direct-alignment 4096 --output docs/bench/data/NEW_COMPONENT_RUN
```

Full serving invocation and commands are recorded in `manifest.json` (`argv` and
`engines`); summarization:

```sh
tools/py bench/summarize_archive.py \
  --serving docs/bench/data/2026-09-28-async-archive-serving \
  --component docs/bench/data/2026-09-28-async-archive-model \
  --output docs/bench/data/2026-09-28-async-archive-serving/validated-summary.json
tools/py docs/bench/data/2026-09-28-async-archive-serving/report_table.py
```

**A and B gates closed; C/D remain unimplemented.** The source-generation/slot/ancestor pressure policy is not implemented. Further
inspection found [partial-page immutability needs its own decision and fixture](../research/2026-09-28-async-transfers.md#c-readiness-finding-partial-pages-are-not-immutable-full-byte-images)
before an eviction-driven source can outlive the originating request without a
whole-page data race or an inaccurate exact-byte claim. No unrelated hang work,
filesystem policy, driver change, weight download or external inference dependency.
