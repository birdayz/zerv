# D.0a: immutable archive progress between prefill units

Date: **2026-09-28**. Parent `0b252be`. **D.0a evaluation gates closed with negative serving findings**, not completion of
proactive tiering or the user's faster-or-on-par goal. [Pre-code specification](../specs/tiering-progress.md),
[paper-based ordered plan](../design/async-tiering.md), [ownership audit](../design/tiering-preparation-audit.md).

## Question and change

C.3 deferred all archive maintenance for an entire packed/segmented prefill chunk.
Immediate-reuse serving yielded no disk restores and 12–29 s maximum source holds.
Can source-only progress at unit boundaries improve this without changing outputs,
unsafe publication, unbounded copying, or hiding a memory-budget increase?

The scheduler now calls an explicit optional `pollMaintenanceInChunk` hook. Without
that hook it still defers. The model backend only advances/drains an already-owned
immutable cache source there; no new source selection, discard or page-ID mutation/publication.
Ordinary maintenance and foreground restore remain outside chunks. The model guard
exempts only immutable-snapshot source capture, not mutable capture or imports.
Same barriers, ownership checks, one GPU owner, 1 MiB tickets, 8 MiB staging and two
reserved read tickets. No arithmetic/precision/stream-format change, new allocation,
queue, asynchronous demotion, demand prefetch or hardware-overlap claim.

## Failures found and fixed

The new adversarial test stops while the in-chunk callback is unlocked, with a
foreground read pending and a new `begin` queued into a reused unrelated slot.
Both modes timed out. Diagnostic markers isolated the wait to the **stop** branch,
not normal completion/reuse. The scheduler aborted its pack and drained running
I/O, but never completed the queued `begin`; that client stayed asleep forever.
Correction: complete all non-running queued operations with `Canceled` during stop;
retain running I/O until acknowledgment, and drain deferred releases before exit.
A separate minimal test covers every queued operation (reset/begin/checkpoint/
prefill/step) without any running backend owner. This is a newly exposed stop bug,
not a resumption of the retired GPU hang investigation.

An existing test concurrently incremented a plain shared `restored` counter. It
reported two while the scheduler-owned backend counter correctly reported three.
Per-client counters, summed after join, remove that test data race. Retained logs:
[`batcher-first-detail.log`](data/2026-09-28-tiering-progress/batcher-first-detail.log),
[`batcher-diagnostic.log`](data/2026-09-28-tiering-progress/batcher-diagnostic.log).
The first shell wrapper expired at 240 s while Bazel's 300 s tests still ran; the
30 s diagnostic reproduced the test failure explicitly. Subsequent tests pass.

Negative control: restoring whole-chunk deferral fails the existing packed-output
test's new positive progress assertion in **both modes** and times out the explicit
mid-callback handshake. [`negative-deferral.log`](data/2026-09-28-tiering-progress/negative-deferral.log).
The mutation was removed before GPU/model/serving execution.

## Correctness evidence

- Final CPU/Python/format: **82/82**, including the additional harness unit-test
  target. Legacy backends without the new hook retain whole-chunk deferral; both
  modes explicitly test this compatibility contract.
- Batcher/archive Debug and ReleaseFast ×20: pass.
- GPU/spill **3/3**, host-driver GPU **2/2**. Some unchanged optimized targets were
  content-cache hits; logs distinguish executed and cached targets.
- Independent FP64/libllama: **337/337** greedy rows, all intermediate/logit bounds
  pass, modes 0/512. [Report](data/2026-09-28-tiering-progress/oracle.json),
  [provenance](data/2026-09-28-tiering-progress/oracle-manifest.json).
  Existing C.1/C.2/C.3 independent fixtures remain unchanged and pass CPU gates.
- Extended model checker: two distinct 128-token prompts, solo gold rows followed
  by a two-member 16-unit pack while the mixed host/GPU source remains held.
  **Ten new GPU quanta inside each pack**, every third boundary simulates pending
  reads and must issue no optional quantum. Mutable exports and imports are
  explicitly rejected during every live chunk. Both full-vocabulary rows exact.
- At **257 and 80,000 tokens**, canonical archived bytes, poisoned/permuted restore,
  four continuation vocabulary rows, two independent decode rows, cancellation/
  retry and last-chunk corruption fallback pass. The long image is exactly
  **5,399,773,184 bytes**. [Model manifest](data/2026-09-28-progress-model/manifest.json).

This is scheduling exactness against native solo execution anchored by the separate
independent byte, ownership and numerical oracles—not a claim that native
self-comparison replaces a semantic reference.

## Reproducible commands and setup

RX 7900 XTX, observed VRAM 25,753,026,560 bytes; Bazel 9.2.0, pinned Zig 0.16.0.
Host GPU runtime ID and complete model/build/source hashes are recorded by the
harness manifests. Same Q4_0 model SHA `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
Operator-prepared `third_party/nvme-probe` is unchanged. GPU jobs serialized.
The host was not reserved: unrelated CPU activity and short harness-test builds
were present; do not call this an isolated-host study.

```sh
bazel test //...
bazel test //tests:batcher //tests:batcher_release_fast \
  //tests:archive //tests:archive_release_fast --runs_per_test=20
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu
tools/py bench/run_archive_model.py --prefill \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --tokens 257 257 257 257 257 257 80000 \
  --output docs/bench/data/2026-09-28-progress-model
tools/py tools/verify_model.py \
  --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/progress-model-oracle \
  --report docs/bench/data/2026-09-28-tiering-progress/oracle.json \
  --modes 0,512 --runtime host
```

The model check's write timer includes the two independent packed prompts. Comparing
it with the old pressure check, which did not run those prompts, is **not** a
cadence-only measurement. The new diagnostic counterfactual executes the same pack
but defers source progress to chunk end. Unit mode requires positive in-chunk
quanta; chunk mode requires zero, both require all exactness gates. Alternating
pairs (first pair warmup, five measured pairs) and unit tests protect this distinction.
Paired measurements pass. Five measured pairs after one warmup pair:
unit **455.251 ± 1.963 ms**, chunk **457.756 ± 1.934 ms**; paired unit-minus-chunk
**−2.505 ± 2.034 ms** (sample SD). Only a ~0.55% small-component difference, not a
serving win. [Paired manifest](data/2026-09-28-progress-paired/manifest.json).

Original six short unit-mode trials (first warmup) give capture **71.804 ± 2.439 ms**,
write including packed computation **451.753 ± 3.729 ms**, restore **185.210 ± 5.263 ms**.
The single 80k run gives capture **4.181 s**, write including packed computation
**7.393 s**, restore **5.810 s**. The historical synchronous restore was 5.357 s;
this single later result does not close that regression or establish variance.

## Serving experiment (complete, negative)

Fresh servers, fixed `multiturn-distinct-v1`, four conversations, two phased turns,
two slots/context 12,288, three rounds, warmup, alternating engine order, **zero idle**.
Native 192 f16 KV pages/4,096 MiB swap, two snapshots/tier off or eight snapshots/host;
disk variants add 8,192 MiB/16 records/alignment 4,096. Tuned llama-server Vulkan
batch 512 and pinned RDNA3/HIP batch 2,048 with the supported
`GGML_CUDA_DISABLE_FUSION=1` setting. Reference RAM cache 8,192 MiB/eight checkpoints.
Exact full CLI, effective flags, hashes, raw streams, resource samples and logs:
[serving manifest](data/2026-09-28-progress-serving/manifest.json).

All **96 native responses**, prompt counts and generated counts match the historical
native baseline exactly. All 48 reference responses complete, but text/count identity
with native is only 14/24 Vulkan and 4/24 HIP: no identical execution-stream or proven
answer-quality claim. Native generates 594 tokens/trial; reference counts vary.

Mean ± sample SD over three fresh-server rounds:

| Configuration | Wall s | Reuse TTFT p50 s | Aggregate tok/s | Gap p99 ms |
|---|---:|---:|---:|---:|
| Native off, 2 snapshots | 67.372 ± 4.299 | 21.917 ± 5.147 | 8.842 ± 0.586 | 145.887 ± 0.407 |
| Native host, 8 snapshots | 48.432 ± 1.331 | 6.037 ± 1.662 | 12.271 ± 0.332 | 144.906 ± 0.455 |
| Native disk, 2 snapshots/tier off | 65.956 ± 7.827 | 20.181 ± 8.957 | 9.098 ± 1.157 | 145.942 ± 0.465 |
| Native host+disk, 8 snapshots | 54.639 ± 0.708 | 9.170 ± 0.292 | 10.873 ± 0.141 | 145.454 ± 0.464 |
| Vulkan llama-server, batch 512 | 50.035 ± 0.395 | 3.522 ± 0.143 | 11.626 ± 0.091 | 662.299 ± 9.239 |
| RDNA3/HIP nofusion, batch 2048 | 50.329 ± 2.774 | 3.860 ± 1.549 | 11.648 ± 0.205 | 1885.004 ± 130.676 |

**Zero disk restores in all six native disk trials.** Disk2 writes 1/2/1 records,
1.178/1.030/1.237 GB total including canceled work, with 6/3/6 cancellations and
max source holds **9.305/13.928/10.819 s**. Host+disk writes two records per trial,
1.105/1.117/1.175 GB, one cancellation each, max holds **14.480/14.841/13.918 s**.
No disk failures. Compared with C.3's 249–315 MB/trial, faster progress substantially
**increases wasted write traffic without generating immediate-turn restores**.
Disk2 maximum holds improve versus C.3's ~28 s, but this does not establish useful
preservation. Host+disk is slower than host-only and both measured references; its
54.639 s point estimate is also worse than C.3's 51.383 ± 4.020 s, though the two
revisions were not interleaved. No serving speedup claim.

The same-session host-only wall time is competitive, but that is not evidence that
the disk plan succeeded. Per-process RSS/HWM and device VRAM samples are retained
in `validated-summary.json`; HIP measures the actual container PID, not Docker's
launcher. Process RSS does not account for all driver/GTT buffers or page cache;
explicit budgets, rather than RSS alone, describe the resource contract.

Reproduce validation/tables (strict complete-run check, native text+token identity,
paired order and progress assertions):

```sh
tools/py bench/run_archive_model.py --prefill --prefill-cadence both \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --tokens 257 257 257 257 257 257 \
  --output docs/bench/data/2026-09-28-progress-paired
tools/py docs/bench/data/2026-09-28-tiering-progress/report.py
```

[Validated summary](data/2026-09-28-progress-serving/validated-summary.json),
[full table/counters/resources](data/2026-09-28-tiering-progress/report.log).
 Prior failing vLLM W4A16/FP8 responses and NVIDIA-only HyperQwen remain
non-equivalent alternatives, not victories; SGLang is still untested.

## Remaining gates / plan

D.0a correctness/component/serving evaluation is complete with negative performance
findings. It is not D completion. Next: measured
window sizing; async GPU→RAM preparation with reservations/atomic qualified rename;
queued-demand protection/prefetch with retained request/record lifetime. Keep exactly
one active implementation increment. The full quality-matched performance goal is
still open, including the historical long-restore regression.
