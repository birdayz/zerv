# D.2 held-out quality gate and later-turn memory-pressure ablation

2026-09-29. No production behavior/default changed. Implemented a reproducible
workload generator and strict scorer, executed a predeclared quality comparison,
then investigated and repeated the later-turn serving experiment with an existing
resource knob. The full performance goal remains open.

## Predeclared gate / independent oracle

[Contract](../specs/demand-quality-gate.md), frozen before generation or inference;
[pre-run hashes](data/2026-09-29-demand-quality/pre-run.sha256).
`bench/make_demand_quality.py` generates four novel 128-record tables, fixed seed
2026092901. Three tasks/table: tag lookup, two-record integer addition, three-record
priority ordering. Ground truth is computed directly from the generated records,
not by any model or by observing answers. All 12 cases retained, no post-selection.
Canonical history makes every current request identical across engines. Max64 output
tokens, greedy, thinking off; JSON must match value, type, length and order exactly.
No fence/prose extraction or answer repair. Prior answers do not contain the current
answer. This is a narrow structured-answer quality test, not broad model evaluation.

`bench/score_demand_quality.py` validates coverage, stop termination, input hashes,
answers, native output/count identity, observed memory ceilings and the preregistered
performance rule. Unit tests reject wrong answers/types/order, fences/trailing prose,
missing/duplicate responses, wrong input hashes, length finishes, errors and over-budget
VRAM. A complete synthetic passing run is also tested. Scoring code is unchanged from
its pre-run hash; extra aggregate validation tests were added after execution.

## Setup / reproducibility

[Serving manifest](data/2026-09-29-demand-quality-serving/manifest.json), adjacent
`raw.jsonl`, `summary.json`, server logs; [scores](data/2026-09-29-demand-quality/scores.json).
Model SHA256 `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`;
native code28583d1, exact binary/build/compiler/reference hashes in manifests.
Same RX7900XTX/driver, parallel2, context12288 per slot, f16KV, no speculation.
Native192pages, snapshots8, host4096MiB, disk8192MiB/16records/8MiB chunks,
D.2prefetch2, join0 or128. Tuned Vulkan `llama-fa-b512` and HIP `rdna3-nofusion`
(batch2048), cacheRAM8192MiB/eight checkpoints. Three alternating-order rounds,
phased C1/C4, warmup, no concurrent GPU diagnostics. Actual prompts ~3.75–3.86k tokens.
Common ceilings24GiB VRAM/32GiB server-process high-water RSS/8GiB scratch; these
are observed bounds, not newly enforced OS limits or equal cache allocations.

Peak observed VRAM / host VmHWM (GiB): native off17.354/14.825,
join17.354/14.824, Vulkan16.942/15.085, HIP17.449/15.379. All within ceilings.
VRAM is device-wide sampled every50ms; VmHWM includes model loading and mmap effects,
not just live cache storage. No system/driver changes or downloads.

```sh
tools/py bench/make_demand_quality.py
# Run the exact argv in the serving manifest with a fresh output directory.
tools/py bench/score_demand_quality.py docs/bench/data/2026-09-29-demand-quality-serving
# Expected exit1: retain this failure, do not weaken the gate.
tools/py bench/check_prompt_equivalence.py \
  --serving docs/bench/data/2026-09-29-demand-quality-serving \
  --output NEW_PROMPT_DIRECTORY
```

The full native/Vulkan/HIP prompt byte strings AND token arrays match all12 requests:
[prompt manifest](data/2026-09-29-demand-quality-prompts/manifest.json). Recorded
request hashes/counts are cross-checked by the existing prompt-capture harness.

## Quality outcome: FAIL, same failures on every engine

Every configuration scores **39/45** across rounds/C1/C4. Six failures are two unique
tasks repeated three times; all C1 tasks pass. Native join0, join128, Vulkan and HIP
produce identical output text, prompt/output counts and finish reasons on **all45
corresponding responses** (180 responses total). Native policy is not introducing
these errors. The failures are shared with both independent reference executables:

- `records-1`, turn2: expected IDs ordered by priorities831,947,979;
  all engines return IDs corresponding to831,979,947. Expected
  `["R1064","R1123","R1005"]`, actual `["R1064","R1005","R1123"]`.
- `records-2`, turn1: all return record objects instead of IDs and hit the64-token
  limit, with the same truncated text. Invalid format and non-stop termination fail.

Independent table values and all distinct failures are retained in
[analysis](data/2026-09-29-demand-quality/analysis.txt). The first is a shared model
answer error; the second is a shared format/budget failure. Neither is evidence of a
new native-only cache/numerical defect. Do not raise the token limit, change the prompt,
drop sorting, or loosen scoring to turn this run green.

The 100%-correct gate and its conditional performance acceptance therefore remain
**failed**. Identical generated work is independently observable on this suite, including
wrong answers; that permits a scoped equal-work timing observation, not relabeling
the declared quality test as passed or asserting general quality equivalence.

## Timings on that exact-work suite

Mean ± sample SD of three rounds; wall seconds, turn TTFT p50 milliseconds.

| C1 engine | Wall | Cold | Turn2 | Turn3 |
|---|---:|---:|---:|---:|
| Native join0 | 4.734±.134 | 3303.8±65.6 | 391.0±15.2 | 397.3±21.4 |
| Native join128 | 4.361±.117 | 3301.3±73.9 | 209.3±13.6 | 217.3±9.8 |
| Vulkan | 5.653±.239 | 4124.7±86.1 | 327.8±53.0 | 403.6±78.1 |
| HIP | 5.344±.081 | 3874.7±23.6 | 263.7±15.5 | 255.8±15.5 |

| C4 engine | Wall | Cold | Turn2 | Turn3 |
|---|---:|---:|---:|---:|
| Native join0 | 22.470±3.430 | 8271.6±299.9 | 2961.4±1807.0 | 3898.8±1876.7 |
| Native join128 | 17.183±1.059 | 8282.3±931.5 | 1178.8±181.9 | 1243.9±97.0 |
| Vulkan | 25.623±2.956 | 9903.4±236.1 | 3190.9±625.5 | 3647.2±382.7 |
| HIP | 20.895±.362 | 11575.6±2135.5 | 1941.5±377.3 | 1571.5±202.0 |

Native join128 wall is18.4% lower than the faster reference atC1 and17.8% lower atC4
on this narrow identical-work fixture. No corresponding broad quality/serving claim.
Full raw timings, tail gaps and all per-trial values are in the serving summary.

## Later-turn investigation and controlled repeat

[Pre-ablation analysis/plan](../design/demand-later-turn.md).
The previous three-turn outlier had zero reused tokens for a10305-token prompt;
other rounds reused10100, and join0 reused10219. Zero reported reuse alone does NOT
prove eviction: both failed host promotion and failed disk destination admission can
return a cold start. Source review identified these branches; it did not identify the
precise branch taken in that historical run. A retry change without a progress proof
could deadlock, so no speculative production fix was made.

Ablate only existing `kv-pool-pages=192` versus256, keeping join128 and all other
settings. Same own-answer three-turn fixture, warmup, C1/C4, three alternating rounds,
tuned references. [Manifest](data/2026-09-29-demand-pool-serving/manifest.json),
[all diagnostics](data/2026-09-29-demand-quality/pool-analysis.txt).

| Engine | C1 wall s | C1 turn3 TTFT ms | C4 wall s | C4 turn3 TTFT p50 ms |
|---|---:|---:|---:|---:|
| Native192 | 10.513±.049 | 463.9±4.1 | 49.991±5.136 | 3564.8±160.2 |
| Native256 | 10.976±.147 | 506.9±7.7 | 45.978±.088 | 3723.6±680.9 |
| Vulkan | 13.236±.126 | 447.4±42.4 | 56.830±2.927 | 5289.6±939.8 |
| HIP | 12.573±.113 | 292.6±12.4 | 51.865±.899 | 4004.4±518.6 |

All90 native output/count signatures exact. Own-answer competitors still differ, so
those wall times do not inherit the new structured fixture's exact-work conclusion.
192pages has2/24 later C4 requests with zero reuse, in two separate rounds, and two
admission waits. 256pages has0/24 and no admission waits. Native max VRAM17.357→
17.856GiB; max host RSS~14.824GiB unchanged. C4 wall improves8.0%, with much lower
variance in this small sample. C1 wall regresses4.4%; C1 turn3 remains far behind HIP.

This supports the pressure hypothesis and measures the cost/benefit of cache slack.
It does not prove a specific eviction/admission branch, nor a general policy fix.
All trials retained; do not silently recommend256 as a free speedup or change defaults.

## Verification / remaining work

`bazel test //...`: **82/82**, one test target executed in the final run, others cached;
[log](data/2026-09-29-demand-quality/cpu-final.log). Scorer and generator tests pass,
including aggregate positive/negative controls. Production GPU code unchanged;
serving/prompt harnesses ran their host-driver prerequisite gates (cached results
visible in logs). No new GPU-kernel verification claim. `git diff --check` passes.

D.2 remains the sole active block. Next: instrument or deterministically reproduce
failed promotion versus destination-admission fallback before changing retry policy;
then run independent state/logit and serving gates. The declared structured-answer
quality gate failed equally for all competitors and stays failed. Broader quality,
C1 later-turn latency and full-plan competitive acceptance remain open. No authorization
or access blocker; these are evidence and implementation gaps.
