# Queued-demand D.2: intermediate verification

2026-09-29. **Implementation under verification, not accepted; no serving speed claim.**
Baseline e793211 plus working-tree changes. [Paper-based ordered plan](../design/queued-demand-implementation.md),
[contract](../specs/queued-demand.md). Raw evidence: [directory](data/2026-09-29-queued-demand/).

## Observed checks

- Independent generated fixture: 108 cache-prefix cases and 210 staging event cases.
- Additional cache tests: demand allows exact host promotion, follows newly inserted
  ancestors without acquiring source leases, and rejects stale generations after reuse.
- Scheduler: callback leave/stop retains borrowed token memory; reclaim-blocked begin
  preserves its request, waits for an epoch and permits independent decode. Debug and
  ReleaseFast targets passed 20 repeats each (`batcher-repeat.log`).
- Final CPU suite: 82/82 targets pass, four executed and remaining cached
  (`cpu-final-directed.log`). Includes format and Python tests.
- GPU/spill: 3/3 pass, Debug GPU executed, other results cached (`gpu-directed.log`).
  Host-driver gate: 2/2 cached passes (`host-gpu-directed.log`). These are not claimed
  as fresh executions of every target.
- Independent FP64/libllama oracle: modes0/512, 337/337 greedy matches, zero tensor/logit
  failures (`oracle.json`, `oracle.log`). This checks unchanged model math; it does not
  independently validate the new scheduler policy.

## Real-model integration

RX7900XTX, production host driver, existing pinned Qwen artifact; exact hashes, build,
commands and options are in each manifest. Each case passed canonical valid archive
state, four full-vocabulary continuation rows, one unrelated packed row, two canceled
request generations, one same-generation handoff, zero GPU uploads while staging,
and all eight tickets free after teardown.

| Window | Tokens | Manifest |
|---|---:|---|
|1|257|[first](data/2026-09-29-demand-model-first/manifest.json)|
|1|80000|[long](data/2026-09-29-demand-model-window1-long/manifest.json)|
|2|257,80000|[window2](data/2026-09-29-demand-model-window2/manifest.json)|

These native exact-state comparisons supplement the independent numerical oracle;
they are not independent implementations of the prefetch mechanism. They are correctness
checks, not repeated component timings or serving benchmarks.

## Negative controls and retained failures

Removing `Archive.Advance.allow_upload` initially **did not fail** the fixture:
six tight polls could finish before the disk worker. This was an inadequately exercised
assertion, not evidence the guard was unnecessary (`negative-upload.log`). The fixture
now waits for every staged completion and checks completed-byte counts before handoff.
With that stronger test, removing the guard aborts both build modes with
`read-ahead attempted GPU upload before handoff` (`negative-upload-completed.log`).
Ignoring request order in `Key.eql` fails both modes with
`expected error.InvalidReadAhead, found 0` (`negative-generation.log`). Both mutations
were restored before final tests. Teardown also panics on the leaked owner in that
intentional generation failure.

Earlier test setup corrupted only block0 while expecting no uploads in any completion
order; corrupting every block correctly enforces that expectation. An initial fake
backend method placed between fields caused a Zig compile error; relocating it fixed
compilation (`cli-callback.log`, `cli-callback-fixed.log`). A later standalone fmt command
encountered a missing Bazel convenience link after another configuration build; rebuilding
`//bazel:zig`, rerunning fmt and the final CPU suite succeeded.

## Reproduction

```sh
bazel test //...
bazel test //tests:batcher //tests:batcher_release_fast --runs_per_test=20
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu
# Use separate output directories for each invocation.
tools/py bench/run_archive_model.py --demand --prepare-window 1 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 --tokens 257 \
  --output docs/bench/data/2026-09-29-demand-model-first
tools/py bench/run_archive_model.py --demand --prepare-window 2 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 --tokens 257 80000 \
  --output docs/bench/data/2026-09-29-demand-model-window2
tools/py bench/run_archive_model.py --demand --prepare-window 1 \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 --tokens 80000 \
  --output docs/bench/data/2026-09-29-demand-model-window1-long
tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/demand-model-oracle \
  --report docs/bench/data/2026-09-29-queued-demand/oracle.json --modes 0,512 --runtime host
```

## Still open

CLI rejection coverage; production-model source-conflict and admission-error drain paths;
foreground preemption and corruption integration; nonwrapping arrival exhaustion handling;
repeatable component timing; final off/protect/prefetch HTTP runs with actual prefetch,
response/count comparisons and tuned Vulkan/HIP competition. No technical/access blocker
has been established. D.2 remains the only active increment, defaults remain off, and
faster-or-on-par serving remains unproven.

## Subsequent verification and serving evaluation

The following supersedes the earlier open-gate list where explicitly covered.
Arrival identity exhaustion now returns `ArrivalExhausted` before changing a slot or
releasing held logits. A directed test checks the last usable identity and continued
decode. Its initial expected fake-history seed was wrong (7 instead of the initial0,
without reset); corrected after the two-mode failure (`arrival-gate.log`,
`arrival-fixed.log`). CPU82/82 passes after correction.

Production-model error matrix: windows1/2 × 257/80k, each additionally checks foreground
preemption without further optional issue and corrupted digests on every block followed
by handoff, zero uploads, a cold miss, position0, no mapped pages, no record readers,
and all eight tickets free. Manifests:
[data/window1](data/2026-09-29-demand-errors-window1/manifest.json),
[data/window2](data/2026-09-29-demand-errors-window2/manifest.json).
Reproduce with the model commands above and those new output directories. These runs
report two handoffs and three cancellations because of the additional error cases.
Twelve CLI rejection cases pass against the exact serving binary, before model loading:
`tools/py docs/bench/data/2026-09-29-queued-demand/check_cli.py`; recorded in
`cli-rejections.json` with commands, errors and SHA.

### Serving question and setup

Does demand protection/read-ahead improve immediate-turn pressure with identical native
outputs? Six configurations, three rounds, alternating engine order, fresh servers,
warmup, phased four conversations/two turns, zero idle, parallel2/context12288,
192 f16 pages, eight snapshots, 4096MiB host tier, 8192MiB disk, 16 records,
8MiB disk chunks, direct alignment4096. No concurrent diagnostics during trials.
Tuned competitors are Vulkan `llama-fa-b512` and HIP `rdna3-nofusion` batch2048.
[Complete commands, binary/model/workload hashes, effective options, memory samples](data/2026-09-29-demand-serving/manifest.json).
Raw responses, summaries and server counters are adjacent. Validator:

```sh
tools/py docs/bench/data/2026-09-29-queued-demand/summarize_serving.py \
  docs/bench/data/2026-09-29-demand-serving \
  > docs/bench/data/2026-09-29-queued-demand/serving-validated.json
```

| Configuration | Wall mean ± sample SD, s | Aggregate tok/s mean | Reuse TTFT p50 mean ± SD, s | Stream gap p99 mean, ms |
|---|---:|---:|---:|---:|
| Native demand off |51.144 ±1.277|11.619|9.821 ±.432|145.358|
| Protect |48.600 ±5.568|12.326|3.676 ±.602|146.936|
| Prefetch1 |46.395 ±1.629|12.814|3.821 ±.258|145.300|
| Prefetch2 |45.910 ±.635|12.940|4.252 ±.183|147.514|
| Vulkan512 |54.110 ±.927|10.752|4.826 ±.584|703.824|
| HIP2048 nofusion |51.761 ±.933|11.600|4.238 ±.647|1778.439|

96/96 native responses match baseline output hashes and prompt/completion counts.
All requests succeeded. Only15/24 Vulkan and1/24 HIP responses match all three fields.
Therefore lower observed native wall time is **not matched-quality parity evidence**.
Reference generated work differs; no claim of achieving the session goal follows.

Every prefetch trial submitted one8MiB window. Prefetch2 handed it off once each trial;
Prefetch1 handed off in rounds0/2 and canceled/discarded8MiB in round1. In all successful
handoffs, zero bytes were observed completed while speculative. Thus this workload
exercises real early disk issue/handoff, but not a large completed staging window.
Protection, changed cache residency and capture timing also change serving behavior;
the roughly10.2% lower prefetch2 wall versus off is an observed configuration result,
not an isolated I/O-overlap speedup. Protect's large variance and one urgent override
remain visible. Defaults stay off. Full raw per-trial variance and counters are in
`serving-validated.json`; host RSS/high-water and sampled VRAM peaks are in the manifest.

Remaining D.2 gates: production source-conflict/admission-failure drain integration,
repeatable component timing and causal analysis, additional fresh/warm workloads and
quality-equivalent competitive comparison. No authorization/access blocker. D.2 remains
active; this report is not full-plan completion.

## Drain gates and repeatable components completed

The additional production model cases now cover the two previously open paths:

- Fill the other live slot's page map, fail private admission with `CacheReclaimPending`,
  drain speculative ownership, poll four more times to prove no same-identity restart,
  release pressure, and restore through the ordinary foreground fallback.
- Capture a128-token checkpoint, demote it to host, start an actual leased archive source,
  request that host prefix, observe deferred begin and source cancellation, drain its
  owner, then restore128 tokens and compare the next full vocabulary row exactly.
  The backend remains nonfatal and all staging tickets are free.

Both windows pass at257/80000 tokens:
[window1 manifest](data/2026-09-29-demand-drain-window1/manifest.json),
[window2 manifest](data/2026-09-29-demand-drain-window2/manifest.json).
The prior first257 run is retained in `2026-09-29-demand-drain-first`.
These supersede the earlier case counts: four cancellations, two speculative handoffs,
four archive continuation rows plus one additional exact host-hit row.

The existing archive component harness now has `--demand`. It uses64MiB deterministic
bytes,1MiB chunks, eight tickets, direct4096 I/O, a CPU-copy device callback, one warmup
and five measured trials, alternating off/window1/window2 order. Completed staging is
required before handoff; total includes that wait. Every restored byte is compared;
cancel/drain separately checks discarded bytes and zero remaining owners. Two independent
runs, exact commands/build/source hashes/raw trials:
[run1](data/2026-09-29-demand-component-1/manifest.json),
[run2](data/2026-09-29-demand-component-2/manifest.json).

```sh
# Run twice with distinct output directories.
tools/py bench/run_archive.py --demand --scratch-dir third_party/nvme-probe \
  --direct-alignment 4096 --output docs/bench/data/2026-09-29-demand-component-1
```

| Mode | Run1 total mean±SD ms | Run2 total mean±SD ms | Run2 staging mean ms |
|---|---:|---:|---:|
| Off |47.975±1.158|48.325±1.184|0.00014|
| Window1 |48.062±.623|49.492±1.731|.655|
| Window2 |49.669±2.537|49.647±1.151|.801|

**Negative performance result:** this completed-window schedule does not improve total
restore latency; window2 is slower in both runs. It deliberately does not hide staging
wait as useful computation. Cancellation after completed staging averaged230ns/411ns
for window1/2 in run2 (metadata drain, not an in-flight worker cancellation bound).
No equivalent llama-server staging-owner API is established; these numbers are not a
competitor or GPU win. They also do not explain the entire measured serving gain.

## Expanded cold/warm and idle serving

[Full manifest, exact invocation, hashes, options and memory samples](data/2026-09-29-demand-serving-idle/manifest.json).
Same model and resource settings as the zero-idle run, except six-second inter-turn idle,
levels1 then4 on each server, three rounds, and host-only added as a baseline. Thus level1
has a fresh conversation; level4 may reuse level1's prefixes. Idle is included in wall
and throughput. No simultaneous diagnostics. Native/server binary unchanged from the
zero-idle measurements. Validate each level with the checked-in validator's optional
level argument; outputs are `idle-level1-validated.json`, `idle-level4-validated.json`.
The first validator attempt incorrectly assumed all native log names were hashed;
short host-only names are not. Correcting filename resolution allowed both validations.

| Configuration | C1 wall mean±SD s | C4 wall mean±SD s | C4 reuse TTFT p50 mean±SD s |
|---|---:|---:|---:|
| Host-only |15.323±.072|48.194±.821|3.833±.942|
| Disk, demand off |15.334±.062|47.182±3.091|3.976±.262|
| Disk, prefetch2 |15.370±.164|45.879±1.325|3.695±.308|
| Vulkan512 |17.023±.158|48.671±1.697|4.469±.977|
| HIP2048 nofusion |17.094±.069|49.762±1.398|3.389±.542|

All90 native responses/counts match host-only. C4 competitor signatures match15/24
Vulkan and7/24 HIP; C1 matches0/6 for both. All requests succeed. Prefetch issued
2/1/1 windows per server round and handed off2/0/1, with one discarded8MiB completed
window. Counters are per server across both levels, not attributed to C4 alone.
C1 shows no prefetch benefit; competitor C1 reuse TTFT is lower (Vulkan.324s,
HIP.280s versus native prefetch.456s). Preserve this latency loss, even though overall
native wall is lower. Variance and differing generated streams prohibit broad parity
claims; quality-equivalence criterion remains exact output plus prompt/completion counts.

The requested production drain cases, bounded-staging component experiment and expanded
repeated serving evaluation have now run. Remaining performance acceptance requires
quality-equivalent competitive evidence, a broader declared quality assessment if exact
streams are not the chosen contract, and sufficient workload coverage—not relabeling
these differing outputs as equivalent. Defaults remain off; the session goal is open.
