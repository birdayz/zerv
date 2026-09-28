# Pressure-driven archive admission — C.3

2026-09-28. **C.3 correctness/component/serving evaluation gates closed; negative
performance result retained.** Parent revision `68fb7bc`; source/build hashes
accompany every measurement manifest. The broader faster-or-on-par serving goal
and D's proactive preparation/prefetch remain open.

## Question and implementation

Can optional immutable-source persistence replace checkpoint-triggered full-image
write-through without pausing the producing request or violating ownership under
pressure/cancellation? Paper basis: [Mooncake, Pensieve, CachedAttention, LMCache](../research/2026-09-28-kv-tier-papers.md).
The [functional contract](../specs/tiering-pressure.md) and independent executable
fixture preceded native code; [integration audit](../design/tiering-policy-integration.md).

The implementation provides byte/slot-pressure LRU selection, exact ready-backing
reuse, independent scheduler maintenance, two reserved read staging tickets and
read-first issue priority. Hard GPU pressure cancels optional capture and retries
blocked decode only after reclamation; the per-source cancellation latch survives
another row's successful allocation. Stop and deferred request-slot release drain
both foreground and background ownership. Explicit slot/MiB headroom controls use
existing budgets, not another full-image allocation. Checkpoint creation no longer
starts production disk writes; the old entry point is diagnostic only.

This stage still holds GPU source pages until capture drains. It does not implement
proactive GPU→RAM preparation or queued-demand prefetch and does not demonstrate
hardware copy/compute overlap.

## Correctness gates

Evidence: [logs](data/2026-09-28-tiering-pressure/).

- Independent scalar/object-set fixture: **2,048 selection decisions and 1,560
  prefix-set events**; generated before native implementation, SHA-256
  `df1c56a8b104b2f32eaadd68e0a8831dec1ae69030163ddc224853083d3c399e`.
  Exact resident/pending sets, tokens, handles, generations, serials, ancestor refs,
  ready backing and decisions; mixed-host metadata and invalid capacities separately.
- `bazel test //...`: **81/81**, including both CPU modes, Python and formatting
  (`all-third.log`). Cache/archive/batcher targets in both modes ×20 passed
  (`repeated-cpu.log`).
- Delayed archive tests: foreground read arriving during write, acknowledge existing
  owners without issuing optional work, two physically reserved tickets, ready
  backing survives read, duplicate write skipped, cancellation with zero issue budget.
- Scheduler tests: source-only idle/stop drain, reclaim-blocked row retries only on
  epoch change while another row runs, leave/reuse during unlocked maintenance,
  read-first priority and simultaneous foreground/background stop drain.
- Negative control ignoring `allow_start` fails both archive modes. Removing only
  public discard validation still passes because internal removal also guards the
  lease; removing both guards fails both cache modes. All mutations restored.
- `bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills`: **3/3**
  (one executed, two cached). Host-driver GPU tests: **2/2** (one executed, one cached)
  in [first successful pressure model run](data/2026-09-28-pressure-model-second/).
- That 257-token production-owner run passes **182,059,008 exact state bytes**, four
  full-vocabulary continuation rows and two independent rows, including mixed host/GPU
  source and production pressure selection. Capture 123.362 ms; write 239.152 ms;
  restore 188.705 ms. Single smoke sample, not a speed claim.

Failures retained: undeclared checker variable/error-set handling, fake test field
ordering/name shadowing, and a runtime enum-literal type in the new component.
The [first pressure-model attempt](data/2026-09-28-pressure-model-first/) failed the
CPU build before executing the GPU model; the explicitly typed enum fixed it.
The original overly large 23 MiB fixture was reduced to a 2.4 MiB bounded matrix
before native implementation; both generation logs remain.

## Selector component measurements

Two independent runs, six fresh sets per shape (first warms up, five reported),
100,000 rotating-LRU decisions per trial. Native code checks every selected handle
and preserve/discard action. No I/O, source acquisition or model execution is timed.
RX 7900 XTX is not involved; host/toolchain/CPU/source/binary hashes in manifests.

```sh
tools/py bench/run_pressure.py --output docs/bench/data/2026-09-28-pressure-component-a
tools/py bench/run_pressure.py --output docs/bench/data/2026-09-28-pressure-component-b
```

Mean ± sample SD, ns/decision, five trials each:

| Candidates | Slot pressure A | Slot pressure B | Host-only A | Host-only B |
|---:|---:|---:|---:|---:|
| 1 | 9.75 ± 0.05 | 10.11 ± 0.02 | 9.71 ± 0.02 | 10.09 ± 0.04 |
| 8 | 11.16 ± 0.28 | 10.93 ± 0.07 | 12.39 ± 0.15 | 12.57 ± 0.09 |
| 64 | 71.64 ± 1.35 | 80.34 ± 4.45 | 62.20 ± 3.62 | 60.19 ± 4.14 |
| 256 | 277.10 ± 8.94 | 281.48 ± 10.78 | 203.92 ± 2.97 | 198.42 ± 10.27 |

[Run A](data/2026-09-28-pressure-component-a/),
[run B](data/2026-09-28-pressure-component-b/). Production snapshot capacity is at
most 64; 256 exercises the generic policy. No equivalent llama-server component
API is available; the independent sorted oracle establishes semantics, not a
performance win against another engine. Full adapter token scans and byte accounting
are outside this microbenchmark and must be included in serving measurements.

## Production-owner and independent model gates

[Repeated pressure path](data/2026-09-28-pressure-model/): six fresh 257-token runs,
first excluded as warmup. Five measured trials, mean ± sample SD: capture
**138.343 ± 27.416 ms**, write **247.346 ± 15.566 ms**, restore
**198.891 ± 12.404 ms**. Every run checks exact state, four full-vocabulary rows and
two independent rows. Source write timing includes interleaved independent work;
it is not an equivalent-work speed comparison against the paused diagnostic.

At 80k: **5,399,773,184 bytes exact**, capture 4.613 s, write 7.421 s, restore
6.013 s. [Paused diagnostic](data/2026-09-28-pressure-paused/): 257/80k exact;
80k capture 6.659 s, write 10.277 s, restore 7.006 s. Long results are single
samples and do not close the historical 5.357 s synchronous-restore regression.
GPU jobs were serialized; selector component runs finished before this chain.

Fresh FP64/libllama comparisons pass all stated bounds and **337/337 greedy rows**
in decode/512-row prefill modes. [Report](data/2026-09-28-tiering-pressure/oracle.json),
[complete manifest](data/2026-09-28-tiering-pressure/oracle-manifest.json); native
capture artifacts retained at `third_party/pressure-model-oracle/`.

```sh
tools/py bench/run_archive_model.py --pressure --tokens 257 257 257 257 257 257 80000 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/2026-09-28-pressure-model
tools/py bench/run_archive_model.py --tokens 257 80000 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/2026-09-28-pressure-paused
tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/pressure-model-oracle \
  --report docs/bench/data/2026-09-28-tiering-pressure/oracle.json --modes 0,512 --runtime host
```

## Serving smoke: correctness passes, preservation is ineffective

[Smoke](data/2026-09-28-pressure-smoke/), same distinct four-conversation, two-turn
phased workload as the eager baseline, two request slots, 12,288-token context per
slot, 192 GPU KV pages, f16 KV, 4 GiB host KV, two host snapshot slots, 8 GiB scratch,
16 disk records, direct alignment 4096. Eight native responses and prompt/generated
token counts match the earlier baseline exactly. Warmup and full effective command
are in the manifest. Single smoke, **not a repeated speed comparison**.

Wall 69.969 s; cold/reuse TTFT p50 23.534/23.242 s; no GPU or disk prefix restores.
One completed write, 326,107,136 total written bytes (includes aborted work), three
cancellations, no disk failures. Source hold time 48.267 s total, **29.161 s maximum**;
305 CPU and nine GPU quanta. The smaller write volume is not a win: it yielded no
useful restores. The two-snapshot policy does not yet solve the serving goal.

Source audit: maintenance defers throughout a packed/segmented prefill chunk, not
merely while a GPU quantum is running. This run had 156 prefill chunks / 2,496
units; a per-decode-boundary bandwidth estimate overstates available issue turns.
[Follow-up candidates and lifetime audit](../design/tiering-preparation-audit.md).
No unverified guard removal or larger transfer default has been shipped.

The first report-helper attempt incorrectly treated both the startup and shutdown
`prefix checkpoints` log lines as final counters; fixed to require `taken,`.
No native correctness gate was weakened.

## Repeated serving: no disk-policy speedup established

The [eight-configuration matrix](data/2026-09-28-pressure-serving/) completed with
six successful configurations and two consistently failing RDNA3 configurations.
Its original manifest remains **failed**, unmodified. The explicit
[per-engine validator](data/2026-09-28-tiering-pressure/validate_matrix.py) checks
all six complete configurations and retains the failed engines separately with
**no performance score**. It confirms **96/96 native responses, prompt token counts
and generated token counts equal the earlier eager baseline**, without filtering
native failures. [Validated data](data/2026-09-28-pressure-serving/validated-summary.json),
[reproducible table/identity helper](data/2026-09-28-tiering-pressure/report.py).

The subsequent supported-fusion-opt-out [RDNA3 comparison](data/2026-09-28-pressure-rdna3-nofusion/)
passes all **48/48 HTTP turns** in two configurations × three rounds. Every one of
the build's 23 output artifacts was hash-verified against pinned
`15995a12d1d530645a4f34c72afdaa30fa680149` ([proof](data/2026-09-28-tiering-pressure/rdna3-verified.json));
no download or runtime dependency added. The wrapper alone is not the implementation
hash. These follow-up trials were serialized after, not interleaved with, the main
matrix; do not pretend this is one randomized paired experiment.

Same GGUF hash, f16 KV, two slots / 12,288 context each, four distinct conversations,
two phased turns, zero phase idle, temperature zero, no speculation and the same
requested generation cap. Native: 192 GPU pages, 4 GiB host KV, two or eight host
snapshots; disk adds 8 GiB scratch / 16 records / 4096 alignment / default headroom.
References: 8 GiB RAM prompt cache, eight context checkpoints per slot, batch sizes
2048 or 512 / microbatch 512. Fresh server each trial, untimed short warmup, three
rounds with reversed engine order in alternate rounds. Full argv, effective commands,
source/binary/model/workload hashes and resources are in each manifest.

Mean ± sample SD over three trials; TTFT columns are per-round medians:

| Configuration | Cold TTFT s | Reuse TTFT s | Wall s | Generated tok/s | Gap p99 ms |
|---|---:|---:|---:|---:|---:|
| Native off, 2 snapshots | 25.436 ± .716 | 22.321 ± 3.726 | 66.987 ± 3.753 | 8.887 ± .514 | 145.352 ± .419 |
| Native host, 8 snapshots | 24.845 ± .679 | 4.568 ± 2.489 | 47.965 ± .475 | 12.385 ± .122 | 145.127 ± .623 |
| Native disk, 2 snapshots/tier off | 24.596 ± .292 | 20.901 ± 4.208 | 65.108 ± 4.077 | 9.146 ± .555 | 145.531 ± .451 |
| Native disk+host, 8 snapshots | 25.730 ± 1.300 | 6.601 ± 3.624 | 51.383 ± 4.020 | 11.610 ± .950 | 144.938 ± .493 |
| Vulkan llama-server b2048 | 30.332 ± .680 | 5.735 ± 1.001 | 52.641 ± 3.089 | 10.963 ± .329 | 1966.651 ± 157.571 |
| Vulkan llama-server b512 | 30.744 ± .230 | 4.810 ± .657 | 50.552 ± 1.089 | 11.741 ± .164 | 591.605 ± 43.473 |
| RDNA3/HIP nofusion b2048 | 30.342 ± 1.037 | 2.961 ± .223 | 49.783 ± 1.320 | 12.695 ± .773 | 1700.190 ± 401.787 |
| RDNA3/HIP nofusion b512 | 31.053 ± 2.793 | 3.528 ± .319 | 50.598 ± .960 | 11.582 ± .086 | 592.903 ± 19.194 |

Native generates exactly 594 tokens per trial. References generate different text
and token counts despite identical weights and request controls: only 8/24 and
14/24 Vulkan responses, and 3/24 in each HIP configuration, match the complete
native text-hash/prompt/generated-count tuple. Consequently this is a natural
conversation workload, **not identical executed token streams or proven equal
answer quality**. Later prompts include each engine's own prior answer. Tok/s
uses actual generated tokens; neither shorter generations nor failed requests are
scored as wins. vLLM's earlier different-W4A16/FP8 empty-response failures remain
non-equivalent; HyperQwen is NVIDIA-only here; SGLang remains untested.

Peak observed total VRAM across trials: native 18.16–18.40 GB, Vulkan 17.72–17.82 GB,
HIP 18.28–18.29 GB (decimal). Raw per-trial process RSS/HWM is retained in table
logs/manifests. Native process HWM is ~15.48 million Linux kB,
Vulkan ~15.82 million kB and HIP ~16.12 million kB; these include loading and **do not
fully account for driver/GTT-backed host allocations or page cache**. Do not claim
native's small end-of-run RSS is its complete RAM footprint. Explicit allocated
host KV/snapshot/cache budgets are the meaningful resource controls here.

The immediate-turn disk profiles finish one write each, **zero NVMe restores in
all six trials**, and 2/2/4 (two-snapshot) or 1/1/1 (eight-snapshot) cancellations.
Total written bytes including canceled work: 275,775,488 / 303,038,464 / 248,512,512
and 314,572,800 / 266,338,304 / 287,309,824 respectively. Maximum source holds are
28.330 / 27.772 / 27.939 s and 17.130 / 14.548 / 12.525 s. No disk failures. Cache
hits and dropped/demoted/promoted counts are printed by the table helper.

**Interpretation:** pressure admission avoids the eager baseline's 8.78 GB/round
but has not made persistence useful under immediate reuse. Two-snapshot wall time
65.108 s is worse than historical eager async 59.501 ± 2.918 s; cross-run timings
are not a paired regression estimate, but there is no speedup evidence. Eight-slot
host+disk is noisy and slower on average than host-only and the best compatible
reference. Host-only remains competitive in overall wall/cold TTFT/gaps, but HIP
has better reuse TTFT and different generation work. This does **not** close the
full faster-or-on-par goal. D must address preparation, cadence and queued demand.

### RDNA3 compatibility failure and supported tuning follow-up

The eight-engine matrix finished **failed**, because both RDNA3 variants crashed
in all three trials (four failed first turns per trial; second turns never ran).
The raw matrix/manifest remains failed and is retained intact. Native and Vulkan
llama-server requests completed; their per-engine results need separate validation,
not rewriting the overall run as successful.

All six logs report `/src/ggml/src/ggml-cuda/mmvq.cu:1930:
GGML_ASSERT(ids || dst->ne[1] == 1) failed` in `ggml_cuda_mul_mat_vec_q`.
At pinned revision `15995a12`, this assertion is inside `if (fusion)`; dense fusion
requires a single output column. Source `ggml-cuda.cu:3579` exposes
`GGML_CUDA_DISABLE_FUSION=1`, returning before fusion selection; the graph optimizer
checks the same setting at line 5235. Source hashes/paths:
[data](data/2026-09-28-tiering-pressure/rdna3-research-hashes.txt).

Added explicitly named **external benchmark configurations** `rdna3-nofusion` and
`rdna3-b512-nofusion`, passing that flag into the restricted container. No reference
code, library, model, container or system setting is modified. This is a supported
compatibility opt-out, not a claim to have fixed the upstream fusion selector or
to have proved which particular fused operation supplied the offending shape.
The successful repeated fusion-opt-out comparison is reported above; failed runs
are not scored as speed wins. No native production behavior changes in this follow-up.

The unfused b512 smoke subsequently completed all eight turns (51.1 s, 640
generated tokens). This confirms a runnable configuration, not numerical parity
or an isolated fix to the fusion selector. The subsequent repeated results are above.

Resource-audit correction: the older multiturn harness sampled `/proc` for the
`docker run` launcher, not the container's server. Its earlier RDNA3/vLLM RSS
numbers must **not** be interpreted as server memory. The follow-up resolves the
named container's host init PID with read-only `docker inspect` and reports that
scope explicitly; missing/dead/unknown containers report unavailable instead of
launcher RSS. Native process memory is unchanged. Deterministic tests cover the
PID distinction, missing process and placing the fusion flag inside the container.
This is process peak RSS, not aggregate cgroup/page-cache accounting.

### Additional HTTP publication/restore gate

The immediate-turn stress matrix produced **zero NVMe restores** despite correct
outputs, so it cannot alone prove the new pressure policy's positive HTTP restore
path. A separate diagnostic adds an explicit six-second idle barrier between
turns to let optional publication complete. The new harness `--phase-idle-s`
defaults to zero, requires `--phased`, is recorded in the manifest and **is included
in wall time**; it does not alter the completed no-idle comparison. A deterministic
harness test checks one sleep per phase barrier, not one per client. This diagnostic
must both preserve native output/token identity and record an actual disk restore;
it is not used to claim a no-idle throughput win.

Observed: [two-snapshot diagnostic](data/2026-09-28-pressure-idle/) remains exact
but has zero disk restores; the explicit positive-restore assertion fails
([retained log](data/2026-09-28-tiering-pressure/idle-positive.log)). Idle alone does
not undo canceled-generation suppression or create usable headroom.

[eight-snapshot/host diagnostic](data/2026-09-28-pressure-idle-host/) passes:
**8/8 exact outputs and token counts, two completed writes, one actual HTTP disk
restore reading 668,991,488 bytes**, 958,398,464 total written bytes, zero disk
failures, one canceled optional write. Source hold total 24.947 s / maximum
12.427 s. Wall 55.3 s **includes six seconds of explicit idle**. The positive gate
is in [this log](data/2026-09-28-tiering-pressure/idle-host-positive.log); it proves
publication/HTTP restoration, not competitive zero-idle speed.

Final CPU/Python/format gate: **81/81** (`all-idle.log`), including new harness
barrier/resource/configuration tests. Native source is unchanged throughout the
model and serving evaluations. C.3 now closes with measured policy limitations;
D's bounded issue cadence, proactive demotion and queued demand remain unimplemented.
The full user goal is **not achieved**. See the [updated plan](../design/async-tiering.md).

Final recheck note: the final combined shell command hit its 240-second wrapper
limit while Debug GPU tests were still running (not a reported target failure).
The same Bazel output base serialized the retry; with an 1,800-second allowance it
returned **GPU/spill 3/3 and host GPU 2/2**, cached after completion. CPU **81/81**
was already green. Initial/retry logs are retained as `final-cpu.log`,
`final-gpu.log`, `final-gpu-retry.log`, `final-host-gpu.log`. GPU jobs were serialized;
the entire host was not reserved, and unrelated CPU activity was observed during
final checks. Do not interpret these experiments as a fully isolated-host study.
