# Fixed-history competitive control

2026-09-29. D.2 competitive acceptance investigation, not completion of the performance
goal. Production baseline716b9de unchanged; benchmark now records fixed-history inputs,
output text and finish reasons. [Pre-run contract/reuse investigation](../design/competitive-fixed-history.md).

## Method and correctness

Same distinct public-document workload, but second-turn assistant history is canonical
rather than each engine's generated text. Existing benchmark behavior remains unchanged
for ordinary workloads. This prevents first-answer drift from changing later input.
Unit tests exercise different first answers producing identical second requests, legacy
own-answer behavior, and rejection of malformed history before server startup.
`bazel test //...`:82/82 targets pass, two executed, remainder cached.
Production inference/math is unchanged; previous independent337/337 oracle and D.2
production-state/error tests remain the numerical evidence, not newly executed here.

[Manifest](data/2026-09-29-fixed-history-serving/manifest.json) records the exact command,
model/binary/source/workload hashes, effective engine settings, memory samples and trials.
GGUF artifact is identical, f16 KV, greedy/non-speculative, parallel2/context12288,
192 native pages,4096MiB host tier,8192MiB disk,8MiB chunks, eight snapshots.
Tuned Vulkan512 and HIP2048/nofusion use8192MiB cache RAM/eight checkpoints.
These are explicit memory configurations, not equal measured residency: native host tier
and reference cache RAM are different stores. No unrestricted equal-resource claim.
C1 thenC4 on each fresh server, warmup, phased zero-idle turns, three rounds with alternating
engine order; no concurrent diagnostics. C4 may reuse C1's prefixes.

Input validator proves30/30 request-body hashes **and prompt counts** match native-off
for every engine. This is serialized-message equality, not per-request token-ID proof.
Output text/finish reasons are retained in raw.jsonl; exact output/count signatures:
- Native off versus prefetch:30/30;60 total native responses valid.
- Vulkan:18/30 (0/6 C1,18/24 C4).
- HIP:7/30 (0/6 C1,7/24 C4).

Thus canonical input fixes one comparability defect but does not remove numerical/output
divergence. No post-selection of matching cases is used to claim a whole-workload win.

## Observations (mean ± sample SD)

| Engine | C1 wall s | C1 reuse TTFT p50 ms | C4 wall s | C4 reuse TTFT p50 s |
|---|---:|---:|---:|---:|
| Native off |8.730±.043|404.5±4.8|38.049±4.102|5.320±3.319|
| Native prefetch2 |8.765±.011|407.2±2.1|36.577±.598|2.940±.615|
| Vulkan512 |10.834±.101|420.1±37.6|41.153±1.440|3.084±.553|
| HIP2048 nofusion |11.207±.124|325.6±19.0|39.812±.803|2.361±.033|

All requests succeeded. Native C4 has lower observed wall, but HIP reuse TTFT remains
lower in both levels; Vulkan C1 reuse is now similar within variation. Generation
length/content still differs; neither lower wall nor throughput closes matched-quality
acceptance. Full trial timing, token counts, stream gaps and counters:
`data/2026-09-29-fixed-history/level1.json`, `level4.json`.

Original own-answer C1 reused7764/7849 native versus7818/7839 Vulkan, leaving85/21
tokens. With canonical history the second prompt is7816 for both; native reuses7764,
Vulkan/HIP7767, leaving52/49. The large replay-work discrepancy is reduced. This supports
investigating shared-cache generated-state retention, but does not prove its causal
latency contribution: request token contents changed too. Production checkpoint policy
was not changed as an unverified workaround. HIP's remaining advantage needs profiling
with identical suffix tokens and separate snapshot-load/prefill/queue timings.

## Reproduction

```sh
tools/py docs/bench/data/2026-09-29-fixed-history/make_workload.py
# Exact serving invocation is manifest.argv; raw server commands are manifest.engines.
tools/py docs/bench/data/2026-09-29-fixed-history/validate_inputs.py \
  > docs/bench/data/2026-09-29-fixed-history/input-validation.json
tools/py docs/bench/data/2026-09-29-queued-demand/summarize_serving.py \
  docs/bench/data/2026-09-29-fixed-history-serving 1 \
  > docs/bench/data/2026-09-29-fixed-history/level1.json
# Repeat with level4 and level4.json.
```

## Remaining acceptance

Per-request token-ID equivalence, fixed generated-work numerical comparison or an
independently scored predeclared held-out quality suite, resource-cap equivalence and
reuse-path profiling remain open. Exact natural-language output equality is not the
same as semantic quality, but differing prose is not automatically equivalent either.
A broader quality contract must be established before its evaluation, not retrofitted
to these outputs. No access/authorization blocker; defaults remain unchanged.

## Full token-ID gate closed; reuse component profile

Developer-only CPU capture now uses production `parseChat`, `qwen38.render` and tokenizer
`encode(.{})`; no production endpoint or reference runtime dependency was added.
Both pinned competitor servers were started sequentially with their recorded serving
commands/env. `/apply-template` plus `/tokenize` responses are retained and compared
in full, not only hashed lengths. **All8 rendered prompts and all8 token-ID arrays are
identical on native, Vulkan and HIP**, and lengths/request hashes match every recorded
serving request. Model and executable hashes are pinned; HIP SHA is checked against the
previous verified `590c6cb6…` binary. Final capture:
[data/manifest](data/2026-09-29-prompt-equivalence-final/manifest.json).
The first successful run is also retained in `2026-09-29-prompt-equivalence`.

```sh
tools/py bench/check_prompt_equivalence.py \
  --serving docs/bench/data/2026-09-29-fixed-history-serving \
  --output docs/bench/data/2026-09-29-prompt-equivalence-final
```

Prefix inspection also confirms that the entire7771-token first prompt is a prefix of
the canonical7816-token second prompt. Cache reuse7764/native or7767/reference is a
policy/runtime fact, not the maximum token-prefix match. Inspection command:
`tools/py docs/bench/data/2026-09-29-prompt-equivalence/inspect_prefix.py`.

Native model/cache profile, first fixed-history conversation: same f16/pool/host snapshot
settings, initial prompt reseeded before each trial, one reuse warmup and five measured
trials. ModelBackend begin/restore, candidate-point prefill/checkpoint and final suffix
are timed separately. Each full vocabulary output exactly matches warmup. No production
code changed. [Manifest and raw rows](data/2026-09-29-reuse-profile/manifest.json).

| Operation | Mean ± sample SD ms |
|---|---:|
| Begin/reset/restore at7764 |17.257±3.192|
| Prefill45 tokens to7809 |187.357±8.516|
| Save checkpoint at7809 |10.582±1.139|
| Final7-token prefill |173.563±1.240|

Total measured components≈388.8ms versus previous HTTP reuse TTFT≈407.2ms. The small
final segment costs almost as much as the45-token segment. This focuses investigation
on separate small-prefill executions; snapshot transfer alone is not the dominant
measured component. The unmeasured remainder is **not proven scheduler overhead**:
this diagnostic excludes HTTP/tokenizer/queueing and uses direct model prefill rather
than the scheduler's packed-unit interface. A controlled counterfactual and independent
numerical gate are needed before a production policy/kernel change. Saving fewer
checkpoints may also hurt later reuse; do not hide that tradeoff.

```sh
tools/py bench/run_reuse_profile.py \
  --capture docs/bench/data/2026-09-29-prompt-equivalence/native.jsonl \
  --output docs/bench/data/2026-09-29-reuse-profile
```

The token-ID gap is closed for this workload. Fixed-work/independently scored quality
acceptance and optimization of the remaining reuse loss are still open. Different
competitive output streams are now demonstrably not caused by different input IDs.
