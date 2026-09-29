# Authoritative InferenceX results: unchanged upstream processing

2026-09-29. User requires **100% of the upstream result-processing path unchanged**.
Completed locally against the existing36-point run. No inference rerun, new weights,
metric corrections, output substitutions, upload, publication or git push.

**These emitted upstream artifacts supersede custom adjusted tables as the
InferenceX results.** Prior observer-derived summaries remain historical diagnostics,
not replacements for any field below. We do not fix or silently reinterpret the
processor's output, even when a metric is unfavorable to zerv.

## Execution and integrity

- Same upstream revision: `f437f7bfd164422036b0de7e3818f8afb5bc70d7`.
- Full module entry point `infx.results.fixed_sequence` executes unchanged for each
  original client artifact, including its power-validation and audit stages.
- Full upstream `infx.results.collect_results` then collects processor outputs.
  No copied formula, local result-builder replacement, monkey patch or partial
  `build_result` shortcut. No website ingestion/submission workflow runs.
- Twelve upstream result/package/power modules fetched directly with pinned SHA256
  in `bazel/inferencex.bzl`. Tests compare the actual Bazel-resolved module bytes to
  every declared SHA. Original upstream source is never edited.
- `tools/inferencex_results.py` only supplies environment/paths and invokes the module
  through runpy; its offline audit guard blocks network/process creation. It does not
  inspect or change metrics. `bench/process_inferencex.py` copies input bytes, supplies
  deployment metadata, invokes the Bazel executable, verifies hashes and preserves
  files. Its extra validation checks do not replace upstream acceptance policies.
- All36 original client inputs and staged copies have identical SHA before/after
  processing. Processor output files are copied byte-for-byte into the collector's
  input directory; the collected36 JSON objects are checked for exact value equality
  with the emitted objects. Collector serialization itself is upstream code.
- Metadata explicitly identifies RX7900XTX, engine/backend, Q4_0 weights/f16KV,
  no speculative decoding, one physical GPU, actual input/output settings and server
  binary SHA. `IMAGE` identifies the binary hash, not a fabricated container tag.
  Complete actual fields/commands are recorded in each metadata file and manifest.

### Artifacts

- **[Unchanged collected results](data/2026-09-29-inferencex-processing/run-r2/agg_inferencex-local.json)**
- [Processing provenance, all36 inputs/outputs and commands](data/2026-09-29-inferencex-processing/run-r2/manifest.json)
- [Individual processor results, original-input copies, logs and power audits](data/2026-09-29-inferencex-processing/run-r2/)
- [Display of upstream fields only](data/2026-09-29-inferencex-processing/metrics.txt)
- [Source hashes](data/2026-09-29-inferencex-processing/sources.sha256)

No power telemetry was measured in the existing run. Every output keeps upstream's
**integer `power_valid: 0`**, `power_invalid_reasons: ["telemetry_file_missing"]`, and
power audit. `REQUIRE_POWER=false` uses upstream's optional-power mode, not an override
of its numerical results. There is no energy/cost/power-validity claim. An unrelated
`/workspace/gpu_metrics.csv` causes our adapter to refuse execution, rather than let
upstream's standard fallback accidentally consume another run's telemetry.

## Emitted interactivity

Each cell contains **round0 / round1 / round2**, directly from the processor, rounded
to two decimals for display. No custom percentile, cross-trial metric or TTFT
adjustment applied. Full precision is in the authoritative JSON above.

### `p90_intvty` — tok/s/user

| Nominal input | C | zerv | llama Vulkan | llama HIP |
|---|---|---|---|---|
| 1024 | 1 | 37.30 / 34.12 / 37.28 | 41.20 / 40.17 / 37.85 | 33.02 / 30.69 / 33.97 |
| 1024 | 4 | 29.16 / 27.88 / 28.71 | 31.96 / 30.12 / 32.30 | 27.73 / 26.10 / 27.57 |
| 8192 | 1 | 19.07 / 18.74 / 19.13 | 38.93 / 38.35 / 39.49 | 33.62 / 31.03 / 30.52 |
| 8192 | 4 | 11.38 / 11.45 / 11.66 | 15.40 / 13.80 / 15.63 | 16.53 / 13.97 / 15.83 |

### `median_intvty` — tok/s/user (fixed-sequence website default at researched revision)

| Nominal input | C | zerv | llama Vulkan | llama HIP |
|---|---|---|---|---|
| 1024 | 1 | 38.54 / 35.26 / 38.48 | 41.25 / 40.74 / 40.63 | 34.47 / 33.96 / 34.29 |
| 1024 | 4 | 30.14 / 30.76 / 30.11 | 33.36 / 30.33 / 33.40 | 29.71 / 28.66 / 29.46 |
| 8192 | 1 | 20.10 / 20.14 / 20.20 | 39.77 / 39.57 / 39.76 | 33.77 / 31.49 / 31.22 |
| 8192 | 4 | 11.84 / 11.72 / 12.03 | 15.79 / 15.50 / 15.79 | 22.09 / 14.36 / 21.61 |

**Raw upstream interactivity does not show zerv ahead of Vulkan.** Zerv's higher
aggregate output throughput remains present in `output_tput_per_gpu`; see the
field-only display linked above. Do not use the previously adjusted interactivity
advantage as a statement about these authoritative upstream results.

### Interpretation kept separate from the numbers

The previously documented role-only event issue remains: the pinned chat client
starts TTFT on any choices event, so zerv's early role event causes request-average
TPOT to include initial inference delay. We **have not corrected it** here. It limits
what the raw interactivity says about streaming speed. Upstream also converts
`std_tpot` reciprocally; its emitted value is preserved, not replaced with a locally
computed rate SD. This run is still a local fixed-sequence workload, not AgentX or
a dashboard hardware/scenario reproduction. Small samples, warm-cache behavior,
nonidentical generated text and quality limitations remain as in the original report.

## Commands and verification

Mandatory result stage after the raw benchmark, using a **new** output directory:

```sh
# All builds/dependencies remain Bazel; this does not run GPU inference.
tools/py bench/process_inferencex.py \
  --source docs/bench/data/2026-09-29-inferencex-local-resumed \
  --output docs/bench/data/2026-09-29-inferencex-processing/run-r2

# Optional display only, reading upstream fields without metric transformations:
tools/py docs/bench/data/2026-09-29-inferencex-processing/show_results.py

# Stabilize the small processor integration target first:
bazel test //tests:test_inferencex_results
# Once stable:
bazel test //...
```

Actual verification: focused target passed; **final83/83 tests passed**,14 executed,
69 cached. Tests execute the full processor/collector, verify input preservation,
source hashes and expected emitted values, preserve upstream's std conversion,
check absent-power behavior and failures for required power/missing metadata, and
reject path traversal in the local adapter. Actual processing/collection36/36 passed.
No GPU work was started or required for this stdlib-only result-processing change.

### Security and retained failures

[Review](../research/2026-09-29-inferencex-security.md#unchanged-result-processor-follow-up).
Result closure is stdlib-only plus the previously pinned benchmark_outcome module;
no new wheel or runtime dependency. Source/import/file-operation review and pinned
ClamAV scan before execution: **13 files,0 infected,exit0** (includes one additional
research-only module). Scanner DB is still nine days old; not a safety guarantee or
complete native-code audit. Logs and manifests remain local.

Two local adapter/test issues were fixed without changing upstream:
1. Initial integration test resolved a runfiles symlink into the source directory,
   so it could not find the Bazel executable. Fixed the test's runfiles-relative path;
   `focused-tests.log` retains the failure, r2/r3 pass.
2. First real-processing adapter check expected Boolean `false`; upstream deliberately
   emits integer0. Original failed processing directory `run/` and `processing.log`
   remain. Changed only our validation to require integer0. Fresh `run-r2/` processes
   all36 points successfully; no result coercion or historical file rewriting.
