# Integrated RAM-staged NVMe prefix archive — 2026-09-28

Implementation: `session.archive` + model logical byte stream + stable `serve.disk`
owner, `ModelBackend` and scheduler pending I/O. Explicit server flags, default off;
[deployment](../deployment/nvme-scratch.md), [contract](../specs/disk-prefix-cache.md).
This is a write-through archive of independent full images, not disk pages threaded
through the shared radix pool and not direct PCIe P2P. Production uses only our Zig
code and OS/Vulkan interfaces. No filesystem-specific detection or attribute changes.

## Real-model disk correctness

[Verified cases](data/2026-09-28-archive-disk-verified/): one 257-token warmup,
five repeats and one **80,000-token** case, all exact state and vocabulary rows.
80k image: **5,399,773,184 bytes**. Real scratch file, imported staging, storage worker,
SHA256, production disk owner/backend. Hot cache entries evicted; source pages released;
all physical model state and KV poisoned; target slot 1 has reversed physical pages.
Every byte and all four teacher-forced vocabulary rows match. Backend begin/cancel/drain,
retry (another four vocabulary rows) and corruption-to-cold fallback also pass.
This is a real model checkpoint gate, not an 80k HTTP workload or plausible-text check.

Single 80k sample: write including hot checkpoint 6.942 s, disk restore 5.356 s.
These are component times, not HTTP TTFT or a serving speedup. Short-image repeated
raw timings are retained in the manifests/logs. Independent FP64/libllama model gate
passed 337/337 greedy selections and intermediate/logit bounds in modes 0/512;
[adapter report](2026-09-28-archive-model.md) retains its evidence and failed attempts.

## Integration failure and deterministic regression

The first serving run is **invalid**; do not use its apparent speedup. Two disk
restores failed with `PagesMissing`. [Logs/raw](data/2026-09-28-disk-serving/),
[control logs](data/2026-09-28-disk-serving-gates/). It was interrupted with SIGINT
sent only to our benchmark Python process; its finally block stopped its owned server.
The negative binary is `third_party/archive-negative/zerv`; its SHA is recorded.
No user process was killed. The incomplete run has no final manifest.

Why: `pollIo` releases the batcher mutex. Another generation could leave a slot,
queue its old page release, then a new generation reuse that slot while the callback
ran. After polling, the scheduler could start the new begin before processing the
queued old release. On the next iteration it released the **new restore's** pages.
The former synchronous scheduler checked releases and selected the next operation
under one lock; inserting an unlocked callback invalidated that invariant.
Fix: recheck the release queue after polling, drain it before selecting new work.

A controlled Event barrier reproduces this exact interleaving without timing guesses.
The test failed in Debug and ReleaseFast before the fix (expected prior release 1,
saw 0), then both modes passed 20 runs each. Delayed cache tests also cover other-slot
decoding, normal completion, cancellation, leave and stop with borrowed tokens retained
until drain. The serving client had ignored SSE error objects; it now records those
as failures (tested) and rejects streams without `[DONE]`. Incomplete requests cannot
be counted as faster completions. A compile-only typo in the new regression test was
corrected before the executed negative control; both logs are retained.

Broader KV stress-test timeouts (also seen before production adapters were added)
were retained rather than hidden; their cause is not established here. The retired
hang goal was not resumed. The fresh all-tests gate after this fix passed
81/81. The completed serving/GPU verification and strengthened failure checks are
recorded below.

## Reference selection / reproducible protocol

Primary-source review before configuring the reference: llama.cpp
[b29c606e28a01b1bc8c1351026a0fa6e616bf6c4](https://github.com/ggml-org/llama.cpp/tree/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4),
local `third_party/hermetic-src/llama.cpp-b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`:
- `common/arg.cpp` SHA256 `6e71ad4f63c79fee81fe218ab6755ed97813ba84d179e38ae02924be1556d46e`:
  `--cache-ram`, `--ctx-checkpoints`, `--checkpoint-min-step`, `--cache-idle-slots`.
- `tools/server/server-context.cpp` SHA256 `2f5d65ce6ef0504b5c8cf55a74c68d3959c49784ba380ef836566b7a7d5fa12b`:
  positive cache RAM enables prompt caching; idle-slot caching otherwise disables itself.

Do not benchmark only the harness's old `--cache-ram 0` baseline. The current run
uses 8192 MiB reference RAM cache, eight context checkpoints per slot, two FA/batch
variants (`llama-fa-kvu`, `llama-fa-b512`). Same GGUF, two slots, 12,288 context per
sequence, four distinct ~8.5k conversations from versioned `multiturn-distinct-v1`;
phased second turns, one short warmup, three fresh-process trials, alternating order.
Native disk/off: two hot snapshots, 4 GiB host swap, 24,576-token KV pool; disk adds
8 GiB scratch and 16 catalog records. Native host-only uses eight hot snapshots:
**more RAM**, not an equal-memory speed comparison. Native arithmetic stays fixed.
Reported gaps are SSE delta gaps (not guaranteed one token per event on all engines).
VRAM is sampled at 50 ms; RSS/HWM are process metrics, not all driver-owned host pages.

## Measured serving results

[Complete three-round run](data/2026-09-28-disk-serving-fixed/),
[validated summary](data/2026-09-28-disk-serving-fixed/validated-summary.json).
All 120 measured requests completed without errors. **72/72 native turns have
identical output hashes across all three configurations and all repeats.** The
reference outputs differ; this is not a cross-engine quality-equivalence claim.
Values below are mean ± sample standard deviation of three fresh-process trials;
TTFT quantiles are per-trial quantiles, not pooled request quantiles.

| Configuration | Cold turn TTFT p50 (s) | Second turn TTFT p50 (s) | Whole workload (s) | Output tok/s, including prefill | SSE gap p99 (ms) |
|---|---:|---:|---:|---:|---:|
| Native disk off, 2 snapshots | 25.409 ± 0.588 | 19.937 ± 4.060 | 64.324 ± 4.123 | 9.259 ± 0.572 | 145.324 ± 0.176 |
| Native host tier, 8 snapshots | 24.249 ± 0.241 | 3.862 ± 1.579 | 47.123 ± 0.538 | 12.606 ± 0.144 | 145.237 ± 0.547 |
| Native disk archive, 2 snapshots | 40.223 ± 1.214 | 9.849 ± 0.730 | 58.939 ± 1.909 | 10.085 ± 0.333 | 24.423 ± 1.738 |
| llama FA, unified KV, batch 2048 | 29.968 ± 0.614 | 6.046 ± 0.892 | 53.403 ± 3.382 | 10.886 ± 0.777 | 1972.067 ± 211.342 |
| llama FA, batch 512 | 31.963 ± 2.202 | 6.072 ± 0.215 | 53.791 ± 2.517 | 10.959 ± 0.306 | 620.296 ± 57.009 |

Interpretation: disk capacity avoids recomputation and halves second-turn TTFT
against the small hot cache, but **full-image write-through raises cold TTFT by
58%**. Whole-workload mean wall time falls only 8.4% versus disk off, with appreciable
variance. More host snapshots win both latency and throughput on this workload;
both tuned llama configurations also finish faster than the native disk tier.
This is a verified capacity option, **not a new fastest-server claim**. Small SSE
gaps are measured, not proof of asynchronous disk/GPU overlap: capture pauses its
source, requests queue, and prefill/decode scheduling changes. TTFT includes that
waiting. The scheduler's independent delayed-I/O tests separately establish that
other slots remain eligible to decode during a pending capture/restore.

Overall request TTFT p95 (s): off 32.974 ± 0.124; host 33.624 ± 0.435;
disk 43.369 ± 0.988; llama unified 41.357 ± 1.248; llama batch512 41.447 ± 1.243.
Each native trial generated 594 tokens; llama generated 572–605. Throughput includes
all elapsed workload time, not decode-only speed. Every disk trial wrote 13 records,
8,780,775,424 padded bytes, evicted two records and had zero failures/skips. Restores
were 3/4/3; read bytes 2,040,528,896 / 2,860,515,328 / 2,040,528,896. The hot cache
can satisfy some second turns, so these are tier-policy results, not isolated read
latency. The warmup checkpoint is included in shutdown counters, not timed requests.

Memory: observed whole-device peak VRAM was 18.428–18.438 GB for the native variants,
18.005 GB for llama unified and 17.992 GB for llama batch512. Native process RSS at
end was 73–90 MB and HWM about 15.85 GB; llama RSS was 6.38–6.51 GB and HWM about
16.20 GB (decimal units, raw kB/bytes retained). HWM includes loading the weights;
process RSS **does not account for all driver-owned mapped host allocations**. Do
not read it as proof the host cache is free: eight ~150 MiB snapshots versus two,
a 4 GiB swap allocation, and the disk tier's 8 MiB staging + bounded metadata are
explicitly different resource configurations. The disk budget is 8 GiB. No cold
page-cache or raw-device throughput claim is made; filesystem settings were not
changed by these runs.

Short model-image component (182,059,008 bytes), five trials after one warmup:
write including hot checkpoint **252.633 ± 12.089 ms**, disk restore
**183.774 ± 12.621 ms**. The separate CPU/disk archive component measures five
trials at 1.1312 ± 0.0116 GB/s write and 1.5308 ± 0.0115 GB/s read, including
copies/SHA/catalog; [component report](2026-09-28-prefix-archive.md).
Neither replaces the serving timings above.

### Reproduction

Run one GPU workload at a time. The harness checks host-driver GPU tests before
starting. Models/reference/build and all production source hashes, exact commands,
environment, and per-round raw responses/server logs are in the manifest.
Use a **new** output directory when repeating:

```sh
tools/py bench/run_multiturn.py \
 --output docs/bench/data/NEW_DISK_SERVING_RUN \
 --workload bench/workloads/multiturn-distinct-v1.json \
 --parallel 2 --context-per-slot 12288 --levels 4 --rounds 3 \
 --phased --warmup --llama-cache-ram-mib 8192 --llama-checkpoints 8 \
 --engines 'zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=2,kv-swap-mib=4096,prefix-cache-tier=off;zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=8,kv-swap-mib=4096,prefix-cache-tier=host;zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=2,kv-swap-mib=4096,prefix-cache-tier=off,prefix-cache-disk-dir=third_party/nvme-probe,prefix-cache-disk-mib=8192,prefix-cache-disk-entries=16,prefix-cache-disk-alignment=4096;llama-fa-kvu;llama-fa-b512'
tools/py bench/summarize_archive.py \
 --serving docs/bench/data/NEW_DISK_SERVING_RUN \
 --component docs/bench/data/2026-09-28-archive-disk-verified \
 --output docs/bench/data/NEW_DISK_SERVING_RUN/validated-summary.json
```

## Final correctness follow-up

[Late-corruption gate](data/2026-09-28-archive-disk-late-corruption/) passes at
257 and 80,000 tokens. This strengthens the earlier corruption test: flip one bit
in the **last** disk chunk, after earlier chunks can already have uploaded model
state. Production begin drains, invalidates the record, releases all destination
pages and returns position zero. Cancellation/retry and every state byte/eight
continuation vocabulary rows remain exact. The additional 80k sample measured
8.149 s write and 5.357 s restore; retained rather than folded into the earlier
single-sample timing (capture first-touch also varied). Reproduce with:

```sh
tools/py bench/run_archive_model.py --tokens 257 80000 \
 --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
 --output docs/bench/data/NEW_LATE_CORRUPTION_GATE
```

`stop aborts a packed chunk before draining pending cache I/O` additionally passes
20 executions each in Debug and ReleaseFast, along with the complete batcher suite
([log](data/2026-09-28-disk-serving-gates/stop-pack.log)). The final model harness's
CPU/format/Python prerequisite passes **81/81** (five targets actually executed,
others cached); host GPU **2/2** cached. The preceding final production GPU gate
passed **3/3**, including spill checks (Debug executed, ReleaseFast/spills cached),
and its host-driver gate passed **2/2** (Debug executed, ReleaseFast cached).
[Gate logs](data/2026-09-28-disk-serving-gates/),
[CPU/host logs](data/2026-09-28-archive-disk-late-corruption/).
A formatting/test command earlier timed out waiting for the active Bazel invocation,
before test execution; it was retried after the GPU sequence completed. No concurrent
GPU workload was started.

## Other competitors and limitations

The same workload was attempted with `vllm-apc`, official ROCm v0.30.0 image
`sha256:2e7da1ad1c66836802072588adea75f9f4991da5f9545b4318e91d422c22ce6a`,
RedHatAI W4A16 revision `c063053e004e9783631651df95cf55d0bbf88b32`, declared FP8 KV.
[Existing provenance/quality differences](2026-09-25-vllm.md); these are **different
weights and KV precision**, not a numerical correctness oracle for our Q4_0 path.
No new image or weights downloaded; existing restricted/offline container recipe.

The first fresh configuration compiled graphs, then failed startup with
`Available KV cache memory: -0.6 GiB` / `No available memory for the cache blocks`.
[Failed run](data/2026-09-28-disk-vllm/),
[wrapper log](data/2026-09-28-disk-serving-gates/vllm-run.log). No requests executed.
This matches the cold-compile limitation previously observed for this competitor;
we do not turn startup failure into a throughput score. Its initial manifest is
left as `running` (incomplete) by the old exception path; the harness now records
startup exceptions as failed and closes HTTP connections on error. A new run after
compilation uses identical flags, not an increased memory budget.

SGLang's pinned source informed the archive semantics, but a verified compatible
ROCm/gfx1100 server is not installed/evaluated here. No SGLang performance claim,
no image download as incidental setup, and no claim of beating all competitors.
P2P, restart persistence, GPU copy/compute overlap, eviction-only write-back and
single-slot/MTP archive support remain outside this implemented contract.

The compile-warm retry **completed all 24 requests** over three processes:
[raw and manifests](data/2026-09-28-disk-vllm-warm/),
[validated summary](data/2026-09-28-disk-vllm-warm/validated-summary.json).
Cold-turn TTFT p50: **34.818 ± 1.450 s**; whole workload **81.624 ± 3.635 s**;
aggregate output **6.376 ± 0.255 tok/s**; SSE gap p99 **1964 ± 80 ms**.
These are workload/queue/cache measurements, not isolated prefill/decode kernels.
Important quality/coverage limitation: `conv2` second turn returns one completion
token and **no text delta in every round**, so its second-turn TTFT statistic
(3.978 ± 0.765 s) covers only **three of four** conversations and is not a comparable
four-answer reuse win. All three native configurations produced the same full
responses. vLLM generates 503–530 total tokens versus native 594 and different text;
we have not diagnosed the empty vLLM response or established quality equivalence.
Its observed peak VRAM is 25.567–25.579 GB with a 68,461-token KV pool (native pool
24,576 tokens). The manifest's process RSS/HWM is the **Docker launcher**, not the
container's model worker; do not compare that field to native host memory.

Reproduction: the preceding serving command with `--engines vllm-apc`, identical
workload/parallel/context/rounds/phased/warmup and a new output directory. Initial
cold-compile failure retained separately; no GPU memory-utilization increase or
system setting change. The final full CPU/Python/format gate after harness error
cleanup passes 81/81 (two executed, rest cached):
[all-final.log](data/2026-09-28-disk-serving-gates/all-final.log).

## Performance interpretation and next levers (proposal, not implementation)

User asks how prefill/decode compare and what remains to squeeze. This increment
changes cache/state movement, **not arithmetic kernels**. Today's fresh cold-turn
TTFT comparison is 24.25 s native host / 25.41 s native disk-off / 40.22 s native
disk versus 29.97–31.96 s llama and 34.82 s vLLM, for four ~8.5k prompts on two
slots. It includes queuing and checkpoint work; dividing prompt length by these
medians would not establish isolated prefill throughput.

The latest existing steady concurrency measurements remain the
[shared-pool ABBA](2026-09-26-shared-pool.md) and the all-three-engine
[multi-user comparison](2026-09-25-multiuser.md). They are older binaries, not
fresh decode measurements of this archive change: native ~47–48 / 85–88 / 140 /
156 output tok/s at 1/2/4/8 clients; vLLM ~29–36 / 52–63 / 93–108 / 144–159.
The matched all-three run has native 47.9/87.0/138.7/150.9, best llama configuration
means 38.4/64.3/92.6/131.0, best vLLM means 31.1/59.4/108.3/158.2. These are steady
short-prompt serving throughput, including some request overhead, not GPU-only rates.
Long prefill while others decode: 69.6k native 120–122 s, llama 111–112 s, vLLM
203–209 s; native preserves other users' progress rather than exclusively optimizing
one prompt's completion time. Do not splice different runs into a matched speedup.

Candidate priorities, each requiring its own research/spec and correctness gates:
1. **Exact FP32 multi-row decode:** existing [profile](2026-09-26-decode-v2.md)
   shows an ~46 ms eight-row step versus ~20 ms at one row, with the multi-row
   projections near 5.2 TFMA/s. Better weight reuse/issue scheduling is the large
   aggregate-throughput opportunity. Approaching one-row time suggests order-2x
   headroom for that batch in an ideal case, not a measured or promised serving gain.
2. **Mixed prefill/decode steps:** current alternation forces a latency tradeoff.
   [Measured stall sweep](2026-09-27-prefill-stall.md): stall 0 lowers eight-user
   gap p99 from ~145 to ~65 ms, but raises the 4.9k loaded prompt from 5.8 to 11.6 s.
   Mixed steps must preserve each decode row's arithmetic; no opportunistic precision
   switch by batch size. This targets both prompt latency and stream smoothness.
3. **Selective/archive write policy and copy/hash scheduling:** avoid paying every
   full-image write on the cold request; today's disk run writes 8.78 GB for this
   small workload and makes cold TTFT worse. Policy, ownership and overlap must be
   specified and measured first, not presumed safe background work.
4. **Prefill phase profiling before another GEMM rewrite:** our hand-scheduled
   f16 GEMM already gained 20–40% at component level; long-prompt attention/DeltaNet,
   small projections and submission may now matter more. Re-profile representative
   lengths and concurrent shapes; no justified new end-to-end percentage yet.

Single-stream plain decode is already close to the weight-bandwidth floor: the
[decode baseline](2026-09-24-decode-baseline.md) measured ~16.7 ms of weight traffic
inside ~20 ms per step. That implies much less plain single-stream headroom than
batched decode; the ~60 tok/s bandwidth-only bound is not a full-model prediction.
Speculation is a separate already-supported single-slot lever, not enabled for this
concurrent archive path. No new optimization track is started by this analysis.
