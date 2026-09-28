# Canonical mixed cache-source capture — C.2 gates

Date: 2026-09-28. Completed byte-adapter increment C.2 of the
[paper-informed tiering plan](../design/async-tiering.md). **Not a serving speedup
or completion of pressure-driven tiering.** Production write-through is unchanged.
[Functional contract](../specs/cache-source-bytes.md),
[research/audit](../research/2026-09-28-cache-sources.md).

## Change under test

Capture a leased immutable snapshot and mixed host/GPU KV pages through the existing
bounded imported staging area, independent of the originating request slot. Host
spans use CPU copies; device spans use submit/poll. One extra archive job (65 maximum)
holds the source plus its ancestors until device and disk ownership drain.

The old arbitrary unused last-page bytes are **not immutable** while a request
appends. The new source stream canonicalizes them to zero after completion and
before hashing. All recurrent state and valid f16/f32 KV bytes remain unchanged.
Odd f16 K tails preserve the valid half of the aligned four-byte transfer unit.
This is not a claim that unused bytes match the old uncanonicalized stream.

Host logical indices now survive suffix demotion using their original GPU indices.
Source validation rejects invalid ordering, live-swap pages, unpinned/free pages,
and out-of-range IDs. Mapped snapshot saves now explicitly establish host-read
visibility. GPU source reads bracket transfer with compute dependencies to order
subsequent appends; one queue is still used. No hardware overlap claim.

## Executed CPU gates

Raw logs: [data](data/2026-09-28-cache-source-bytes/).

- Independent coordinate-decoding Python fixture, generated before native code:
  **60 layouts, 4,804 windows**, f16/f32, 128/256 pages, 1/3/16 layers per buffer,
  odd/aligned/full-page lengths and byte/group/chunk boundaries. Fixture SHA256
  `4e07a21b411165081fdc4426b642f9ca666ac838a45cabb73b14447e5e71c3de`.
- Both native model test modes pass the expected window hashes and canary/range
  checks. Initial linker failure (fixture SHA helper not linked) retained in
  `unit-first.log`; corrected by declaring the target's SHA dependency.
- Negative control `(tokens + 1) % page` instead of `tokens % page` fails the byte
  fixture in **both** modes (`negative-tail.log`), then correct code restored.
- `bazel test //...`: **81/81 pass** (`all-second.log`), including suffix demotion,
  complete mixed-source validation and independent job 64 with 64 request IDs.
- Optimized source model checker builds (`checker-build.log`). A build is not a
  successful model run.

## Device and model gates

- GPU Debug/ReleaseFast plus spill gate **3/3**, host-driver GPU **2/2**; logs
  `gpu.log` and `../2026-09-28-cache-source-model-first/host_gpu.log`. Bazel reused
  applicable cached results; the logs distinguish cached from executed targets.
- Pages/archive/model CPU tests repeated **20 times in each mode**, all pass
  (`repeated-cpu.log`).
- Independent FP64/libllama oracle **337/337 greedy**, all intermediate/logit bounds
  pass (`oracle.json`, `oracle.log`), using the pinned hermetic reference artifact.
- Production source owner at **257 and 80,000 tokens**: **182,059,008** and
  **5,399,773,184 bytes** restored exactly after poisoning/reversing destination
  pages; four full-vocabulary continuation rows and two independent rows exact.
  Appends, originating-slot reuse, mixed host ancestry/GPU suffix, capture cancel,
  upload cancel/retry and late disk corruption all exercised. Source released
  after drain; duplicate capture skipped without retaining a lease.
- Retained paused-slot owner passes both lengths too. No fixture tolerance relaxed.

[First source run](data/2026-09-28-cache-source-model-first/),
[repeated source run](data/2026-09-28-cache-source-model/),
[paused regression run](data/2026-09-28-cache-source-paused/) contain manifests,
source/build/model hashes and raw logs. The Q4_0 artifact SHA256 is
`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
C.3 policy and D preparation/prefetch remain separate; see the
[integration audit](../design/tiering-policy-integration.md).

## Component measurements and limitations

`summary.json` / `summary.log` in the CPU data directory are reproduced by its
`summarize.py`, which validates the model manifests before summarizing. Six fresh
257-token checker processes, first excluded as warmup; five measured trials,
mean ± sample SD:

| Operation | ms |
|---|---:|
| Canonical golden capture from live state | 161.183 ± 45.110 |
| Mixed-source disk preservation, including two interleaved model steps | 269.582 ± 50.558 |
| Disk restore | 212.932 ± 35.577 |
| Maximum callback wall duration in each trial | 1.950 ± 0.669 |

80k source single sample: capture 4.925 s, mixed-source disk write 8.041 s, restore
6.519 s, maximum callback 4.717 ms. Paused regression single samples: 257 restore
222.287 ms; 80k capture 7.257 s, write 10.411 s, restore 7.362 s. These are **not**
equivalent-work interleaved speed comparisons: source mode adds a 128-token ancestor
checkpoint/split prefill, cancellation, canonicalization and different step timing.
CPU repetition tests overlapped the background model chain after short trials;
long-case total time is not an isolated latency benchmark. GPU jobs were serialized.

Variance is material; no speedup claim. The historical 80k restore regression
against the old synchronous 5.357 s sample remains unresolved (B: 6.826 s).
`device_starts` includes CPU-only source acknowledgments, not just GPU submissions.
Callback timing includes hashing, CPU copying and host scheduling, not GPU time.
No equivalent llama-server immutable mixed-source component exists. Full-serving
comparison remains mandatory at C.3 integration; current production write-through
has not changed and the faster-or-on-par user goal is **not achieved**.

## Reproducible commands

Executed CPU/fixture commands are above. Device/model and summary commands:

```sh
bazel test //...
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py bench/run_archive_model.py --source --tokens 257 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/2026-09-28-cache-source-model-first
tools/py bench/run_archive_model.py --source --tokens 257 257 257 257 257 257 80000 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/2026-09-28-cache-source-model
tools/py bench/run_archive_model.py --tokens 257 80000 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/2026-09-28-cache-source-paused
tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/cache-source-oracle \
  --report docs/bench/data/2026-09-28-cache-source-bytes/oracle.json --modes 0,512 --runtime host
tools/py docs/bench/data/2026-09-28-cache-source-bytes/summarize.py
```

The model harness gates itself on CPU and host-driver tests and records build,
model, source hashes and commands. The operator-prepared scratch directory is used
unchanged. No driver, mount, filesystem policy or production dependency changes.
