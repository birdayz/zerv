# 2026-09-29 — native InferenceX: tiered versus untiered

## Observed result

All 24 points passed (192 measured requests, 120 warmups). Native output SHA,
prompt/output counts and finish signatures match across treatments and all three
rounds at each workload. Untiered wins throughput in three workloads. Tiered gains
2.18% throughput at 8192/C4, but loses median and P90 interactivity everywhere.
**No disk restores occurred. This is not evidence of an NVMe speedup.**

This user-authorized comparison runs only zerv. No fresh llama-server execution,
no dashboard submission, no new dependency installation. Historical competitor
results are not contemporaneous controls. Security review and limitations remain
in [the source/dependency review](../research/2026-09-29-inferencex-security.md).

## Setup and reproducibility

[Predeclared specification](../specs/inferencex-local.md#authorized-rerun-native-tiered-versus-untiered).
RX 7900 XTX 24 GiB; Qwen3.8-27B Q4_0, f16 prefill/KV; two server slots,
12288 context per slot, 192×128 live GPU positions, eight prefix checkpoints,
no speculative decoding, default reuse-join off. Both configurations keep GPU
prefix caching. Tiered enables 8192 MiB host swap and 8192 MiB disk archive,
16 entries, 8 MiB chunks, demand prefetch two. Untiered disables host swap and
disk tiering; it is not zero-host-memory operation. This intentionally changes
host/disk capacity, not GPU capacity; not an equal-total-memory comparison.

Seed42, nominal input1024/8192, output256, eight measured requests at C1/C4,
2*C warmups. Three alternating configuration-order rounds; fresh server per point,
length controls and warmups before measurement (warm-cache results). Actual prompt
lengths are recorded per case, not assumed equal to nominal ISL. Ignore-EOS forces
fixed work, not a natural-stop quality assessment. Observed wall window:
2026-09-29 12:25:38–13:18:12 UTC.

Code commit `967c181` was pushed before measurement. Native binary SHA256:
`d045c7399e679c821719747418da72e4470ccf360e5fedc32afa1468e91400ae`.
Model SHA256: `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
Unchanged InferenceX client and full processor revision:
`f437f7bfd164422036b0de7e3818f8afb5bc70d7`.

```sh
# Run with a 10800000ms outer execution timeout; use new output directories to repeat.
tools/py bench/run_inferencex.py \
  --output docs/bench/data/2026-09-29-stream-tiering \
  --engines zerv-tiered,zerv-untiered --inputs 1024,8192 \
  --output-tokens 256 --prompts 8 --levels 1,4 --rounds 3
tools/py bench/process_inferencex.py \
  --source docs/bench/data/2026-09-29-stream-tiering \
  --output docs/bench/data/2026-09-29-stream-tiering-processed
tools/py docs/bench/data/2026-09-29-stream-tiering-gates/inspect.py
```

[Measurement manifest](data/2026-09-29-stream-tiering/manifest.json) includes exact
commands, environment, build/model/tokenizer hashes and resource peaks.
[Processor manifest](data/2026-09-29-stream-tiering-processed/manifest.json) records
source/result hashes and configuration recipe fingerprints.
[Collected upstream results](data/2026-09-29-stream-tiering-processed/agg_inferencex-local.json)
are authoritative. [Inspection output](data/2026-09-29-stream-tiering-gates/inspection.txt)
contains all individual trial values, comparisons, memory peaks and tier counters.
[Measurement log](data/2026-09-29-stream-tiering-gates/measurement.log) and
[processing log](data/2026-09-29-stream-tiering-gates/processing.log) retain execution output.

## Upstream metrics

Each cell is the arithmetic mean ± sample SD of **three emitted trial values**,
not a pooled request statistic. No observer timing corrections, no replacement
formulas. Throughput and interactivity are tokens/s; TTFT/E2E are seconds.
Upstream P90 interactivity is reciprocal P90 TPOT, not P90 of reciprocal TPOT.

| Nominal input | C | Tiered output throughput | Untiered output throughput | Tiered change |
|---|---|---|---|---|
| 1024 | 1 | 37.913 ± 0.741 | 40.622 ± 0.860 | −6.67% |
| 1024 | 4 | 58.801 ± 0.576 | 64.991 ± 2.040 | −9.52% |
| 8192 | 1 | 21.244 ± 0.455 | 22.427 ± 0.222 | −5.28% |
| 8192 | 4 | 25.350 ± 0.328 | 24.810 ± 0.084 | +2.18% |

| Input | C | Tiered median interactivity | Untiered median interactivity | Tiered P90 interactivity | Untiered P90 interactivity |
|---|---|---|---|---|---|
| 1024 | 1 | 45.508 ± 1.162 | 47.914 ± 0.903 | 44.322 ± 0.719 | 47.638 ± 1.334 |
| 1024 | 4 | 36.258 ± 0.823 | 38.353 ± 1.917 | 33.809 ± 0.425 | 37.746 ± 1.644 |
| 8192 | 1 | 44.736 ± 1.221 | 47.196 ± 0.393 | 41.479 ± 0.636 | 46.066 ± 1.208 |
| 8192 | 4 | 19.953 ± 0.294 | 20.625 ± 0.130 | 18.364 ± 0.228 | 20.253 ± 0.076 |

| Input | C | Tiered mean TTFT | Untiered mean TTFT | Tiered mean E2E | Untiered mean E2E |
|---|---|---|---|---|---|
| 1024 | 1 | 1.129 ± 0.030 | 0.972 ± 0.018 | 6.754 ± 0.133 | 6.304 ± 0.135 |
| 1024 | 4 | 8.080 ± 0.207 | 7.304 ± 0.198 | 15.134 ± 0.107 | 13.785 ± 0.435 |
| 8192 | 1 | 6.209 ± 0.109 | 5.976 ± 0.039 | 12.054 ± 0.261 | 11.415 ± 0.113 |
| 8192 | 4 | 24.266 ± 0.296 | 25.070 ± 0.099 | 35.118 ± 0.446 | 36.639 ± 0.142 |

## Tier activity and memory

Whole-server-lifetime counters include length gates and warmups, not just timed
requests. Each tiered point writes four disk entries: 872415232 bytes at1024,
2239758336 bytes at8192. All report zero restores/read bytes, failures,
cancellations and prefetch jobs/handoffs. At8192/C1, tiered demotes six checkpoints
(320 pages), promotes none. At8192/C4, each tiered round demotes eight (449 pages),
promotes one (64 pages), and reuses58006 prompt tokens versus untiered49809.
At1024, both treatments reuse the same prompt-token counts and neither promotes.
This is consistent with a host-cache reuse advantage at8192/C4, not an isolated
causal experiment on disk, prefetch overlap or individual overheads.

Sampled device-wide VRAM peaks: tiered17.369–17.374 GiB, untiered17.357–17.358 GiB.
Server-process host VmHWM: tiered14.806–14.824 GiB, untiered14.760–14.762 GiB.
Host high-water includes model loading, is not steady-state cache residency, and
does not include every client/process or OS page-cache byte. Configured tier budgets
are upper limits, not measured residency. No power telemetry: upstream
`power_valid=0`, reason `telemetry_file_missing`, retained without suppression.

## Verification and limits

- 24/24 matrix/count/prompt/resource gates passed; all312 observer requests end
  with256 completion tokens, `length`, HTTP200 and DONE.
- All312 role timestamps equal first nonempty text timestamps. No early role-only
  opening event; observer traces validate behavior, never replace upstream timing.
- All192 measured output/count/finish signatures agree across treatments and
  rounds for each workload. This checks equal generated work, not broader quality.
- Original upstream input/result hashes and exact collector object equality
  rechecked by the retained read-only inspection script.
- Precommit CPU suite83/83 passed; focused harness/processor2/2 passed
  ([logs](data/2026-09-29-stream-tiering-gates/)). No production code changed
  during measurement/reporting; no new GPU-kernel change.
- An initial inspection attempt using `tools/py -c` failed because the launcher
  accepts a script path, not interpreter flags (`can't open file .../-c`). The
  retained script was then executed successfully; no benchmark failed or retried.

Three rounds and eight measured requests per point are a narrow fixed-output
experiment, not a general workload or significance claim. No default is changed.
The tiered long-C4 throughput gain coexists with worse interactivity; the other
three throughput regressions are retained. Disk-restore benefit, production
quality and current competitor ranking remain unproven here.
