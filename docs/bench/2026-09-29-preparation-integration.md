# D.1.2 integration — implementation and verification in progress

Status: **D.1.2 functionality/evaluation closed; no faster-or-on-par claim**.
The sections below preserve the chronological investigation; the final section
supersedes earlier open D.1.2 gates. Pre-code contract:
[preparation specification](../specs/tiering-preparation.md). Paper-derived order:
[async tiering plan](../design/async-tiering.md). Raw evidence uses the start-date
prefix `data/2026-09-28-preparation-*` (work crossed UTC midnight).

## Implemented

- Qualified radix preparation candidates include internal nodes. Atomic cache finish
  checks the source lease/path references, commits the already-drained pool transaction,
  renames aliases, and consumes the lease only on success.
- Separate bounded GPU command owner, zero-time fence polling, deadline fail-stop,
  cancel/drain, failed-incarnation suppression and distinct copied/committed/freed counters.
- GPU→host page copies retain full physical f16/f32 layout and explicit barriers.
  Packed-prefill maintenance can acknowledge but cannot publish/rename.
- LRU preparation below a free-GPU-page threshold, fixed host reserve, one window,
  no optional host eviction. Host-only works; optional archive capture is serialized.
- CLI opt-in `--prefix-cache-prepare-pages`, window `1|2|4`, explicit/default host
  reserve; defaults remain off. Decode pressure waits on the existing retry epoch.
- Finish/abort requests another maintenance evaluation, so idle preparation does not
  stop after one window while pressure remains.

## Evidence actually executed

Commands and logs: [integration data](data/2026-09-28-preparation-integration/).

1. Independent token/object-set generator produced 3,872 joint cache/pool cases
   **before** cache implementation (fixture SHA-256
   `571484d380411553dd31edd7f201c5a9103cdfd98c8968eddc57f0170005afe7`).
   Outcomes: 128 none, 774 commit, 1,248 cancel, 684 Busy, 132 InvalidState,
   906 injected CommitFault. Debug and ReleaseFast pass.
2. `bazel test //...`: 82/82 pass in the initial integration build (18 executed,
   remaining cached). This predates the scheduler regression fix below; do not
   substitute it for the final whole-tree gate.
3. `bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills`: 3/3 pass
   (Debug executed, other two cached). `tools/py tools/zerv_build.py --test-host-gpu`:
   2/2 pass (Debug executed, optimized cached).
4. `tools/py bench/run_archive_model.py --prepare --tokens 257 80000 --output
   docs/bench/data/2026-09-28-preparation-model-host-packed`: both pass. Repeat with
   `--scratch-dir third_party/nvme-probe --direct-alignment 4096` and output
   `.../2026-09-28-preparation-model-disk`: both pass. Each case verifies exact valid
   state, eight full-vocabulary continuation rows, two packed rows, a canceled
   transfer, diagnostic retry, a GPU hit while owned, delayed publication, and an
   unrelated invalid-slot error without false device-fatal classification. One live
   copy frees zero pages; another commits outside the pack and frees one page.
   Disk-enabled cases initialize coexistence, **not concurrent foreground read coverage**.
5. `tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic
   --work-dir third_party/preparation-model-oracle --report
   docs/bench/data/2026-09-28-preparation-integration/oracle.json --modes 0,512 --runtime host`:
   independent FP64/libllama 337/337, no numerical failures.
6. `tools/py docs/bench/data/2026-09-28-preparation-integration/check_cli.py`:
   16 CPU startup cases pass. Missing model prevents GPU initialization. Runtime
   page-capacity/reserve rejection still needs explicit device startup coverage.

## Newly exposed scheduler livelock: failure retained and fixed

Initial repeated Debug/ReleaseFast cache/system/serve tests: 3 targets passed,
Debug kv_system timed out in 1/10 after 300 seconds. Serial reproduction: 1/20
at 30 seconds. Diagnostic watchdog: 2/40 fail, flat/tier-off seed 300 and
radix/tier-on seed 302. These runs have **no preparation owner**. They are not
successful repetition gates.

Watchdog snapshots show an old swapped decoder, newer resident partial prompts,
and about 29,000 release epochs in ten seconds. Root cause: swap selection filtered
out the oldest waiter after a failed admission in the current epoch, allowing a
smaller newer waiter to consume the space just reclaimed for it. Multiple victims
were needed, but their freed space could never accumulate. This is a scheduler
ordering issue, not GPU or preparation-copy corruption.

The deterministic 10-page/9-page decoder/4+4+2 resident-prefix regression fails in
**both modes before the fix** (`swap-order-negative.log`). Selection now establishes
oldest order before checking the retry epoch, preserving older-prompt priority and
minimum victim residency. With the fix, batcher and real-pool system tests, both
modes ×20, pass (`swap-order-fixed.log`). A bounded watchdog now prints scheduler
state on future failures instead of leaving a silent five-minute timeout.

## Serving measurement and remaining gates

A three-round off/window1/window4/Vulkan512/HIP2048 comparison was launched before
the livelock was discovered. It uses the pinned distinct-conversation workload,
parallel2/context12288, 192 f16 pages, eight snapshots, 4096 MiB host store,
preparation target32, warmup and zero idle. The runner and full configuration are
in `data/2026-09-28-preparation-serving/`. That run uses the **pre-fix binary**;
concurrent CPU diagnostic/build activity also makes it exploratory, not clean
performance acceptance. Preserve all results, then repeat on the final binary
without competing diagnostics. Reference generated streams are not assumed equal.

Still open: final whole-tree/repeated gates after all edits, preparation-specific
late-error/stop/read-priority coverage, windows2/4 real-model ownership coverage,
capacity CLI cases, host+disk loaded preparation, clean repeated serving and exact
native response/token-count validation. D.2 queued-demand protection/prefetch is
not implemented. D.1.2 remains the sole active increment.

Incidental failures: the first joint-fixture generator had a bracket syntax error
(retained log); a packed-checker local was initially named the Zig keyword `packed`,
so formatting rejected it before any build/run; renamed to `pack_result`. No failed
run is counted as passing.

### Exploratory serving completed; final CPU gate

The pre-fix three-round run completed with **72/72 native response hashes,
prompt-token counts and completion-token counts identical** across baseline,
window1 and window4. Checked-in validator:
`tools/py docs/bench/data/2026-09-28-preparation-integration/summarize.py`;
[validated trials](data/2026-09-28-preparation-integration/serving-validated.json).

| configuration | wall seconds mean ± SD | tok/s mean ± SD | reuse TTFT p50 seconds mean ± SD | gap p99 ms mean ± SD |
|---|---:|---:|---:|---:|
| host-only, preparation off | 47.047 ± .334 | 12.626 ± .089 | 3.874 ± .793 | 144.365 ± .279 |
| preparation target32, window1 | 47.923 ± .629 | 12.396 ± .163 | 2.351 ± .358 | 144.112 ± .179 |
| preparation target32, window4 | 48.421 ± 1.407 | 12.275 ± .363 | 5.002 ± 2.120 | 144.490 ± .492 |
| tuned Vulkan512 | 48.721 ± .163 | 11.925 ± .040 | 3.567 ± .620 | 588.794 ± 22.865 |
| tuned HIP2048 nofusion | 48.397 ± .489 | 11.527 ± .317 | 3.172 ± .224 | 1988.727 ± 162.422 |

Interpretation: window1 improves this run's reuse median latency but **regresses
wall time against preparation-off**. Window4 is noisy and also regresses the mean.
These are not final-binary/controlled-load acceptance results. Only 16/24 Vulkan
and 3/24 HIP responses match the native hash **and** both token counts, so similar
wall time is not a matched-quality parity proof. No default change.

After the swap-order fix and maintenance-continuation adjustment,
`bazel test //...` passes **82/82** (`cpu-final.log`, nine tests executed and the
rest cached). The earlier statement requiring this final CPU rerun is satisfied;
remaining coverage and clean serving gates above remain open. No GPU or serving
job remains running after this exploratory run.

Final-tree device rerun also passes: GPU/spill **3/3**, host-driver GPU **2/2**
(`gpu-final.log`, `host-gpu-final.log`). Debug device runs executed again; optimized
and spill targets reused valid cached results. No new serving parity claim follows.

## D.1.2 remaining coverage and clean evaluation (2026-09-29)

The extended model gate now exercises suffix windows (the one-page ancestor is
explicitly suppressed for that phase), rather than merely passing a larger limit
while copying one ancestor page. All **12** cases pass: windows1/2/4 × host-only/
host+disk × 257/80,000 tokens. Raw manifests:
`data/2026-09-29-preparation-complete-{1,2,4}-{host,disk}/`.
Reproduction, serialized:

```sh
for W in 1 2 4; do
  tools/py bench/run_archive_model.py --prepare --prepare-window "$W" \
    --tokens 257 80000 --output docs/bench/data/2026-09-29-preparation-complete-$W-host
  tools/py bench/run_archive_model.py --prepare --prepare-window "$W" \
    --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
    --tokens 257 80000 --output docs/bench/data/2026-09-29-preparation-complete-$W-disk
done
```

Each validates exact restored state/eight vocabulary rows/two packed rows, actual
move count, private partial-tail copying, canceled transfer, and a late descendant
lease causing atomic rejection of a drained ancestor preparation. Host/GPU free
counts and committed counts remain unchanged by the rejected commit; stale source
release rejects. Capacity above the GPU pool, reserve above host capacity, full
host reservation and foreground-read issue inhibition are checked through the
production owner boundary. Disk cases additionally persist an independent 128-token
prefix and **actually upload it while preparation retains disjoint source pages**:
its next full-vocabulary row is exact and the preparation cannot publish through
the in-chunk/read-priority hook. Final abort count is two, not one. Earlier partial
checker outputs remain preserved under their original manifests.

### Controlled final-binary serving

No concurrent diagnostics, builds or other GPU jobs during measurement. Fresh
servers, warmup, three rounds with alternating order; same pinned model/workload
and native/reference configurations as the exploratory run. Disk cases use 8 MiB
windows, 8192 MiB capacity, sixteen records and alignment4096. Full commands,
binary hashes, resources and responses:
[clean manifest](data/2026-09-29-preparation-serving-clean/manifest.json).
Validation command:

```sh
tools/py docs/bench/data/2026-09-28-preparation-integration/summarize.py \
  --input docs/bench/data/2026-09-29-preparation-serving-clean \
  --output docs/bench/data/2026-09-28-preparation-integration/serving-clean-validated.json --clean
```

| configuration | wall s mean ± SD | tok/s mean ± SD | reuse TTFT p50 s mean ± SD | gap p99 ms mean ± SD |
|---|---:|---:|---:|---:|
| host-only, preparation off | 47.162 ± .750 | 12.597 ± .200 | 4.698 ± 2.687 | 144.625 ± .398 |
| host-only, target32/window1 | 47.701 ± .417 | 12.453 ± .109 | 2.527 ± .177 | 144.468 ± .099 |
| host+disk8, preparation off | 52.869 ± .200 | 11.235 ± .043 | 3.938 ± .047 | 145.186 ± .186 |
| host+disk8, target32/window1 | 51.768 ± .510 | 11.475 ± .113 | 6.149 ± 3.311 | 144.817 ± .182 |
| tuned Vulkan512 | 48.614 ± .335 | 11.952 ± .082 | 3.730 ± .808 | 570.353 ± 15.098 |
| tuned HIP2048 nofusion | 48.146 ± .682 | 11.797 ± 1.100 | 3.026 ± .311 | 1959.219 ± 132.706 |

**96/96 native responses and prompt/generated token counts match exactly.** Quality
checks against references are explicit but negative: only 15/24 Vulkan and 2/24 HIP
responses match both hash and counts. The independent numerical oracle establishes
native implementation correctness on its pinned numerical workload, not equality
of these generated serving streams. Thus wall-time proximity is not a quality-
matched performance proof, and reference tok/s counts describe different outputs.

All disk trials perform one real restore (671,088,640 physical bytes read each).
Disk writes off: 3,875,536,896 / 4,110,417,920 / 4,118,806,528 bytes; on:
2,868,903,936 / 3,514,826,752 / 2,701,131,776 bytes. Preparation is eligibility-driven:
host-only submits 0/84/0 pages and host+disk submits 0/0/31 pages. Therefore **do not
attribute every off/on timing difference to DMA preparation**. The limited available
sample does not establish its causal benefit; queued-demand protection is still
absent, and threshold32 often finds no eligible source. Host-only mean regresses,
disk mode remains slower than both references, and defaults stay unchanged.

The preparation functionality/correctness and clean comparative-evaluation gates
are now covered, with negative/mixed performance retained. This closes D.1.2 as an
increment, **not the session performance goal**. D.2 research/specification is the
next single increment; no implementation or performance claim for prefetch yet.
