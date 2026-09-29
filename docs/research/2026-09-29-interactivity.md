# Why the website has interactivity but the raw benchmark does not

2026-09-29. User-requested source research only. No benchmark resumed, no dependencies
installed, no third-party code executed, no upload/push. Public source fetched as
inert text over HTTPS at exact revisions; ordinary source inspection, not a new
malware scan or security guarantee. No upstream repository instructions followed.

## Pins, local files and reproducibility

- Benchmark repository: https://github.com/SemiAnalysisAI/InferenceX at existing
  `f437f7bfd164422036b0de7e3818f8afb5bc70d7`.
- Website repository: https://github.com/SemiAnalysisAI/InferenceX-app at
  `7f722808578bb4c93067fc593af34c19387625de`, HEAD observed with
  `git ls-remote https://github.com/SemiAnalysisAI/InferenceX-app.git HEAD`.
- Source roots: `third_party/research-serving/InferenceX-<revision>/` and
  `third_party/research-serving/InferenceX-app-<revision>/`.
- Complete fetched file list and hashes: [sources.sha256](2026-09-29-interactivity/sources.sha256).
- Trees obtained with `curl -fLsS https://api.github.com/repos/SemiAnalysisAI/REPO/git/trees/REVISION?recursive=1 -o LOCAL/tree.json`.
  Listed source files obtained using
  `curl -fLsS https://raw.githubusercontent.com/SemiAnalysisAI/REPO/REVISION/PATH -o LOCAL/PATH`.
  No install/build/workflow/uploader entry points executed.

The website source HEAD is an observed public revision, not proof of the production
site's exact deployed commit. The previously read live Kimi K3 page showed Agentic
and P90 Interactivity; public code explains the distinction below.

## Verified fixed-sequence pipeline

1. `inferencex-e2e/infx/bench_serving/benchmark_serving.py:418–491` calculates each
   successful request's TPOT as `(latency - ttft)/(output_tokens - 1)`, then latency
   percentiles. Raw saved fields include `median_tpot_ms` and `p90_tpot_ms`; they do
   not need a separate interactivity measurement.
2. **`inferencex-e2e/infx/results/fixed_sequence.py:203–207` is the missing step.**
   It converts latency ms to seconds, and for finite positive TPOT fields performs:

   ```python
   data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)
   ```

   Thus `p90_tpot_ms` produces `p90_intvty`, and `median_tpot_ms` produces
   `median_intvty`. This is result postprocessing, outside the raw load client.
   Our local runner deliberately ran the client, not the official result-processing/
   fleet/publication stack, explaining why `upstream.json` lacks `*_intvty`.
3. App `packages/app/src/components/inference/metric-registry.ts:701–708` selects
   `median_intvty` for the baseline interactivity axis. App
   `packages/app/src/components/inference/utils/resolveXAxisField.ts:37–69` forces
   fixed-sequence natural axes to **median**, while Agentic applies the selected
   percentile. Therefore the P90 label seen on AgentX is not the default
   fixed-sequence view at this app revision.

Definition: `1000/p90_TPOT_ms` is reciprocal slow-tail latency, **not**
`p90(1000/TPOT_ms)` (a fast-tail statistic). For example,25ms/token maps to40tok/s/user.
It is not aggregate output throughput divided by concurrency. The fixed-sequence
conversion broadly also inverts std_tpot; reciprocal standard deviation is not a
valid rate SD. Our own cross-trial SD is computed on the three derived rates and
must not be replaced with that converted std field.

## AgentX distinction—and why ITL here needs care

App `packages/db/src/etl/benchmark-mapper.ts:352–375` explicitly rederives Agentic
`p90_intvty = 1/p90_itl` (ITL in seconds), overriding artifact-supplied interactivity.
Its comments identify historical drift between `1/p(ITL)` and `p(1/ITL)`.
`packages/app/src/lib/benchmark-transform.ts:35–82` enforces the same definition for
unofficial overlays. Mapper tests around1519–1559 verify the inverse and full-response
preference; they were read, not executed locally.

**Do not interpret that AgentX ITL field as necessarily pooled gaps between individual
SSE events.** App `packages/db/src/etl/full-response-interactivity.ts:1–12,84–105,161–181`
prefers full-response per-request ITL and can reconstruct it as:

```
(request_end - first_content) / (output_sequence_length - 1)
```

It makes `*_full_response_itl` canonical when available. The benchmark repository's
`inferencex-e2e/infx/results/agentic/request_metrics.py:65–83,120–146` also derives
interactivity from the corresponding latency statistic and carries full-response
metrics. This per-request full-response formula is structurally the same as TPOT
with a correctly measured first-content timestamp. The names/source paths differ;
one must not simply substitute the fixed client's pooled `p90_itl_ms` for AgentX
full-response per-request ITL.

## Consequences for our earlier answer

- Our **raw** `1000/p90_tpot_ms` numbers exactly use the now-confirmed fixed-sequence
  postprocessor formula. No rerun is needed to derive them.
- The prior answer did not verify the website's metric before calling these
  website-style P90 interactivity. That was incomplete: distinguish fixed-sequence
  TPOT-derived fields and median default from the AgentX P90 selection.
- Our **first-text-adjusted** numbers remain custom observer-derived results, not
  literal outputs of the unmodified fixed-sequence postprocessor. Their formula
  matches the structure of AgentX full-response ITL, but that does not turn our
  synthetic fixed-sequence workload into an AgentX run or prove exact harness parity.
- The raw openai-chat role-event timing issue is real: see pinned
  `backend_request_func.py:368–377`, where any nonempty choices array starts TTFT,
  without requiring nonempty content. Changing the metric name does not fix this.
- Earlier throughput/request-latency evidence is unchanged. The claimed adjusted
  interactivity advantage must retain its custom-metric qualification; it is not a
  directly measured advantage on the website's AgentX scenario.

Research verified by reading both pinned conversion and app-selection code, including
source tests; no JavaScript/Python from the fetched repositories was run. Existing
raw benchmark data and previously computed numerical tables are unchanged.
