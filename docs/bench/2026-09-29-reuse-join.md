# Short-reuse checkpoint suppression: measured gain and later-turn regression

2026-09-29. D.2 remains the active competitive-acceptance block. This is an opt-in
native policy, not a paper's claimed algorithm and not completion of the performance goal.

## Research → plan → implementation

The [paper-backed ordered plan](../design/async-tiering.md) and
[paper ledger](../research/2026-09-28-kv-tier-papers.md) motivate explicit checkpoint
storage/recomputation tradeoffs. The [fixed-history investigation](../design/competitive-fixed-history.md)
is the direct experimental basis. First establish identical input tokens; then isolate
restore, checkpoint save and prefill costs; test a controlled counterfactual; specify
and independently test the policy; measure real serving including later-turn costs.
The [pre-code contract](../specs/reuse-join.md) defines the implementation.

Added `--prefix-cache-reuse-join N` (default 0, maximum 512 and configured prefill
chunk): a short, nonempty cache-hit suffix may omit **new** checkpoint boundaries.
All suffix tokens still execute; existing checkpoints are unchanged. Cold requests,
long suffixes and the default retain the prior policy. This reduces repeated small
prefill executions but may increase later recomputation. Only concurrent shared-KV,
radix, non-MTP configurations with snapshots accept a nonzero value. No shader or
model arithmetic change, allocation in the token loop, or new runtime dependency.

## Component counterfactual

[Manifest](data/2026-09-29-reuse-counterfactual/manifest.json),
[summary](data/2026-09-29-reuse-counterfactual/summary.json).
Exact same restored prefix and 52 suffix tokens; reseed each trial, warmup then five
trials, alternating order. Sum of mean measured stages:

| Mode | Total stage time |
|---|---:|
| Split and save checkpoint | 387.65 ms |
| Split without save | 376.27 ms |
| Joined without save | 203.61 ms |

Every final vocabulary row is bit-identical across modes and repetitions, maximum
absolute difference zero, same argmax. Saving costs about 11 ms; the extra small
prefill execution costs about 175 ms. This excludes HTTP/scheduler overhead and is
not a serving benchmark. Reproduce with `tools/py bench/run_reuse_profile.py
--counterfactual` and the exact recorded arguments in its manifest.

## Correctness and failures retained

Evidence directory: [reuse-join](data/2026-09-29-reuse-join/).

- Independent pre-code boundary oracle: 1,528 cases, fixture SHA256
  `d8646cc72f31240e992982927b884b063120b1d3cdbd2c8216216507c9a699e7`.
  Regenerate with `tools/py tests/reference/generate_reuse_join.py`.
- Public generation interface checks complete suffix tokens, exact prefill count,
  checkpoint side effects, cold/default/below/exact threshold behavior in both modes.
- Negative control: forcing the actual session call's threshold to zero fails both
  session test modes (`negative-join.log`: expected zero saves, found one). Restored.
- Independent FP64/libllama model gate: **337/337**, modes 0 and 512; `oracle.json`,
  `oracle-mode*.json`, `model-oracle.log`. This numerical anchor is complemented by
  exact full-vocabulary counterfactual comparisons and production response checks;
  it is not a claim that every possible joined suffix was tested independently.
- CLI: nine incompatible configurations rejected before model load, including bounds,
  disabled chunking, small chunk, static KV, flat cache, no snapshots, MTP and one slot.
  Reproduce `tools/py docs/bench/data/2026-09-29-reuse-join/check_cli.py` against the
  SHA-checked measured binary; results `cli.json`.
- Repeated lifecycle run initially timed out: Debug batcher 2/20, ReleaseFast 1/20;
  both kv_system modes passed. Preserve `lifecycle-repeat.log`.
  Root cause: the test started the scheduler before queueing begin, then assumed
  a nonempty optional lookahead callback must happen. Begin arriving after an empty
  poll can execute directly, leaving the test waiting forever for an optional event.
  The lifetime test now queues before scheduler startup. A separate deterministic
  empty-poll interleaving test proves begin can succeed with zero demand callbacks.
  No production scheduler change. Both batcher modes ×40 pass (`callback-race-fixed.log`).
- Final CPU/Python/format gate: **82/82**, 4 executed and remaining cached
  (`cpu-final.log`). GPU/spill **3/3**, 1 executed; host-driver **2/2**, cached
  (`gpu-final.log`, `host-gpu-final.log`). Do not describe cached results as new executions.

## Serving: same configuration, tuned competitors

RX 7900 XTX; model SHA256
`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
Native binary SHA256 `f253e66557fddf989af5a9ae74c8c3ab251046e35b3ef355f7b2fca96c540a57`.
Parallel 2, context 12,288 per slot, 192 f16 KV pages, eight snapshots, 4096 MiB host
swap, 8192 MiB disk/16 records/8 MiB chunks, prefetch two. Compare join0 vs join128.
References: tuned Vulkan `llama-fa-b512`, RDNA3/HIP `rdna3-nofusion`, 8192 MiB cache
RAM/eight checkpoints. Stores and memory allocation are not semantically identical.
Exact commands, effective settings, build/model/workload hashes, memory peaks and
all raw responses live in these manifests and their adjacent raw files:

- [Fixed history](data/2026-09-29-reuse-join-serving/manifest.json)
- [Three-turn own-answer](data/2026-09-29-reuse-join-three-serving/manifest.json)

Both: phased warmup, C1/C4, three rounds, alternating engine order, no concurrent
GPU diagnostics. Times below are mean ± sample SD of three trials. Reproduce each
manifest's `argv` with a fresh output directory via `tools/py bench/run_multiturn.py`.
Three-turn fixture generation: `tools/py docs/bench/data/2026-09-29-reuse-join/make_three_turn.py`.
Validator derives expected response count from SHA-checked workload; its historical
`exact_matches_to_native_off` key denotes the selected native baseline (join0 here).

### Fixed-history workload

| Engine | C1 wall s | C1 reuse TTFT ms | C4 wall s | C4 reuse TTFT p50 ms |
|---|---:|---:|---:|---:|
| Native join0 | 8.833 ± .112 | 416.6 ± 17.3 | 36.810 ± 1.398 | 3174.9 ± 194.7 |
| Native join128 | 8.615 ± .086 | 222.8 ± 7.7 | 34.724 ± .895 | 2075.2 ± 595.4 |
| Vulkan | 10.734 ± .009 | 396.1 ± 3.1 | 39.903 ± .415 | 3518.1 ± 259.4 |
| HIP | 11.109 ± .005 | 312.4 ± .3 | 39.032 ± .638 | 2378.3 ± 45.0 |

All **60 native response/count signatures** exact. Competitors differ: Vulkan16/30,
HIP9/30 match native. Full-input token equality was separately established in the
[fixed-history report](2026-09-29-fixed-history.md); that does not establish output
quality equivalence. C4 native gap p99 is 145.4 vs145.5 ms; no gap improvement claim.

### Later-turn cost (actual prior answers, not canonical history)

| Engine | C1 wall s | C1 turn3 TTFT ms | C4 wall s | C4 turn3 TTFT p50 ms |
|---|---:|---:|---:|---:|
| Native join0 | 11.031 ± .134 | 447.0 ± 16.3 | 47.283 ± .807 | 4322.4 ± 420.9 |
| Native join128 | 10.944 ± .131 | 499.6 ± 10.8 | 49.059 ± 4.763 | 6969.6 ± 5752.4 |
| Vulkan | 12.865 ± .345 | 367.3 ± 87.4 | 54.897 ± 3.579 | 5866.2 ± 1688.1 |
| HIP | 12.690 ± .199 | 304.4 ± 29.4 | 53.406 ± 1.894 | 4533.4 ± 229.5 |

All **90 native response/count signatures** exact; all requests complete. Competitor
matches Vulkan16/45, HIP5/45. Different generated answers also change later prompts,
so these competitor timings are not fixed-work comparisons. C1 turn2 improves
446.5→243.6 ms, but turn3 regresses. C4 wall mean regresses **3.8%**, with substantial
variance. Do not hide this behind the faster fixed-history result. Default stays off.
The worst C4 third-turn delay is not causally isolated by these measurements.

## Decision / open acceptance

Keep the knob opt-in for explicitly short-reuse latency workloads. The component
experiment plus actual serving establishes a scoped native improvement, not a
universal win or full-plan completion. Quality-equivalent competitive acceptance,
matched resource/work constraints, and broader workloads remain open. Next inside
D.2: predeclare and execute a fixed-work numerical or independently scored held-out
quality gate; investigate high-variance later-turn cache behavior before recommending
joining for general multi-turn use. No new implementation track is opened.
