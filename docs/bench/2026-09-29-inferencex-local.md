# Local InferenceX fixed-sequence benchmark: zerv vs llama-server

**Result-processing authority update:** use the [unchanged upstream processor outputs](2026-09-29-inferencex-upstream-processing.md)
for InferenceX results. The custom adjusted interactivity and observer summaries below
are historical diagnostics, **not authoritative InferenceX metrics**. They are retained
for provenance and must not replace any upstream value. The new report includes all36
unmodified processed outputs and their power-validity limitations.

2026-09-29. **Completed locally: 36/36 points**, three engines × two input lengths ×
two client concurrency levels × three trials. **288 measured requests and 180
warmups passed** exact-length, termination, request/prompt and resource checks.
This uses the actual pinned upstream InferenceX fixed-sequence generator and load
client, not a replacement benchmark bearing its name. It is an **unofficial local
configuration**, not an InferenceX dashboard reproduction or an AgentX benchmark.
**No upload, submission, PR or git push is authorized or performed.**

## Results

Arithmetic mean ± **sample standard deviation across three trials**. Each trial
has eight measured requests, each generating exactly 256 tokens. Output throughput
and timed wall duration come unchanged from upstream. First-text latency is the
supplementary transparent observer's time to first nonempty content or reasoning,
including queueing and proxy forwarding. C is client concurrency; servers have two
slots, so C4 deliberately includes queueing. Input is the upstream *nominal random*
length, not the actual rendered chat length.

| Input | C | Engine | Output tokens/s | Timed wall s | First text ms |
|---|---|---|---|---|---|
| 1024 | 1 | zerv tiered | **38.04 ± 1.75** | 53.92 ± 2.54 | 1123.32 ± 62.86 |
| 1024 | 1 | llama Vulkan b512 | 33.41 ± 0.81 | 61.32 ± 1.51 | 1375.47 ± 73.03 |
| 1024 | 1 | llama HIP nofusion | 29.18 ± 0.67 | 70.22 ± 1.64 | 1215.09 ± 47.85 |
| 1024 | 4 | zerv tiered | **61.98 ± 0.51** | 33.04 ± 0.27 | 7727.25 ± 81.01 |
| 1024 | 4 | llama Vulkan b512 | 52.63 ± 0.14 | 38.91 ± 0.11 | 9375.86 ± 220.01 |
| 1024 | 4 | llama HIP nofusion | 47.76 ± 1.78 | 42.92 ± 1.62 | 10013.34 ± 530.55 |
| 8192 | 1 | zerv tiered | **21.44 ± 0.10** | 95.51 ± 0.43 | 6166.17 ± 31.80 |
| 8192 | 1 | llama Vulkan b512 | 18.14 ± 0.13 | 112.91 ± 0.84 | 7654.36 ± 61.82 |
| 8192 | 1 | llama HIP nofusion | 16.26 ± 0.58 | 126.09 ± 4.41 | 7822.25 ± 218.88 |
| 8192 | 4 | zerv tiered | **25.58 ± 0.31** | 80.07 ± 0.98 | 23078.74 ± 279.28 |
| 8192 | 4 | llama Vulkan b512 | 21.02 ± 1.39 | 97.73 ± 6.69 | 29043.93 ± 2730.55 |
| 8192 | 4 | llama HIP nofusion | 20.92 ± 0.51 | 97.93 ± 2.38 | 30252.75 ± 874.39 |

Zerv's mean output throughput is approximately **14–22% higher than the faster
measured reference** in these four settings. Its mean first-text latency is also
lower. This is a result for this fixed-output workload/configuration, **not a claim
of fastest general serving, numerical equivalence, or broad quality equivalence**.
Generated text need not match across backends; forcing equal token counts does not
establish equal answer quality. There was no fresh exhaustive reference retuning:
these are the previously tuned compatible Vulkan and RDNA3/HIP configurations.

### End-to-end and descriptive tails

Upstream E2E includes full request completion. Each p99 is computed on only eight
requests, then mean ± sample SD across trials: **not a robust tail-latency SLO**.
Supplementary text-event gaps are not guaranteed token gaps (SSE may combine tokens,
and template/special-token filtering can suppress events). All metrics/trial values,
including observer E2E and event gaps, are in the JSON summary.

| Engine | Input | C | Upstream E2E mean ms | Upstream E2E p99 ms | First-text p99 ms |
|---|---|---|---|---|---|
| zerv tiered | 1024 | 1 | 6739.51 ± 317.50 | 7083.06 ± 389.58 | 1388.29 ± 115.53 |
| Vulkan | 1024 | 1 | 7664.30 ± 188.89 | 8188.13 ± 527.38 | 1760.80 ± 212.90 |
| HIP | 1024 | 1 | 8776.44 ± 204.33 | 9770.72 ± 938.82 | 1771.39 ± 289.23 |
| zerv tiered | 1024 | 4 | 14408.25 ± 148.77 | 17437.53 ± 135.64 | 10724.61 ± 555.10 |
| Vulkan | 1024 | 4 | 17102.50 ± 35.83 | 20528.89 ± 116.12 | 12315.65 ± 328.74 |
| HIP | 1024 | 4 | 18825.00 ± 667.86 | 22457.72 ± 1148.41 | 13497.98 ± 1004.58 |
| zerv tiered | 8192 | 1 | 11937.97 ± 53.39 | 13622.90 ± 202.84 | 7368.30 ± 58.15 |
| Vulkan | 8192 | 1 | 14112.94 ± 104.07 | 15605.21 ± 261.39 | 9059.79 ± 162.96 |
| HIP | 8192 | 1 | 15760.29 ± 551.51 | 17617.06 ± 1069.25 | 9607.16 ± 747.17 |
| zerv tiered | 8192 | 4 | 34713.55 ± 428.43 | 44298.49 ± 575.94 | 30711.97 ± 385.98 |
| Vulkan | 8192 | 4 | 43091.45 ± 2966.98 | 53690.15 ± 3561.28 | 39644.70 ± 7316.06 |
| HIP | 8192 | 4 | 43296.04 ± 1161.19 | 53522.90 ± 1686.53 | 40432.95 ± 2760.57 |

### Important upstream timing caveat

The pinned openai-chat reader starts TTFT on **any choices event**, including the
role-only event. Native zerv emits that event before inference. Its upstream TTFT
therefore is **not first-token latency**; raw ITL/TPOT inherit event/TTFT semantics
and should not be used to claim token-latency wins. Original upstream values remain
unchanged in every `upstream.json`. The same byte-preserving loopback observer on
all engines separately timestamps nonempty reasoning/content. Its overhead is
included, not subtracted or characterized as zero. We do not claim an isolated
prefill or kernel speedup from these request metrics.

## Work, cache and resource matching

- Actual client revision: [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX/tree/f437f7bfd164422036b0de7e3818f8afb5bc70d7),
  six unchanged client modules. Bazel wrapper selects one random-generation worker
  before timing, offline local tokenizer, openai-chat `/v1/chat/completions`.
- Seed42, random range ratio1, infinite request rate, 2×C upstream warmups; eight
  measured requests, output256, three alternating engine-order rounds. Fresh server
  per point; gates and upstream warmup precede measurement. This is **warm-cache**
  serving, not a cold-prefill experiment. Warmups use the generated workload and
  cache reuse is allowed on all engines; identical warmup rules do not imply
  identical cache representation/reuse policy.
- Full native rendered prompts and token IDs match both reference `/apply-template`
  and `/tokenize` results. Measured request-body SHA multisets match captured upstream
  bodies. Every observed prompt usage matches captured IDs, including warmups.
  Actual distinct lengths are **998/1076** for nominal1024 and
  **7498/7596/8242/8244** for nominal8192 (random decode/re-encode plus chat rendering).
  Upstream nominal input/total-token throughput must not be represented as actual
  rendered-token throughput. Output throughput uses verified server counts.
- All requests finish with `length`, exact256 and DONE, nonempty text and no error.
  Additional per-point HTTP control: ordinary short-answer request stops before64;
  ignore_eos produces exactly8/32/64 tokens on all engines. This is stricter than
  upstream's allowable error fraction. Thinking behavior is unchanged on all engines;
  the short HTTP control alone disables thinking. No double chat templating.
- Same Qwen3.8-27B Q4_0 GGUF, RX7900XTX, f16KV; two slots,12288 context per slot,
  24576 live GPU positions. Native192×128-token pages; cache slots8, swap8192MiB,
  disk8192MiB/16entries,8MiB chunks, demand prefetch2; reuse-join stays0. References
  cacheRAM8192MiB/checkpoints8; Vulkan batch512/ub512, HIP batch2048/ub512, non-MTP.
  Recurrent-state/snapshot accounting and admission/eviction differ. This is matched
  configured live capacity and RAM budgets, **not byte-identical caches**.
- Common acceptance ceilings24GiB device,32GiB server host VmHWM. Device samples at
  50ms include other device use and can miss sub-sample peaks; host VmHWM is read
  from the actual server PID (including container PID mapping), not client memory.
  Maxima across accepted points: zerv **17.371GiB device /14.825GiB host**;
  Vulkan **16.952/15.086GiB**; HIP **17.473/15.382GiB**. These are observed usage,
  not reservations or complete system memory accounting.
- Native shutdown counters across12 points include gates/warmups: **48 disk writes,
  18,673,041,408 bytes written; zero disk restores/bytes read**. Host demotion/promotion
  does occur. This fixed-sequence run does **not demonstrate an NVMe restore speedup**
  or multi-turn tiering/AgentX performance. Counter lines are retained in the summary.

Hardware baseline: [machine](../hardware.md). Pinned Vulkan llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`; RDNA3 fork15995a12, existing restricted
reference container and artifact. Complete effective server/client commands,
environment overrides, build graph hashes and tokenizer hashes are in the manifests.
No driver/clock/power/package changes or weight downloads were made for this run.

Measured hashes:

| Artifact | SHA256 |
|---|---|
| GGUF | `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d` |
| zerv | `5c72086d80f28a7e129b95323fab63b7938b5ede6c4a17cf8eaff7170548e148` |
| Vulkan llama-server | `f6a09ab1a41b0bd75c0268af3cff2618c34f1f547ecf0b1a2a1158b8a8fb4b40` |
| HIP llama-server | `590c6cb61c27eae36aad6a9c2154e8c2b478d01c123d434b54648544c7229d15` |

## Artifacts and reproduction

- [Combined passed manifest](data/2026-09-29-inferencex-local-resumed/manifest.json).
- [Validated per-trial/aggregate JSON and raw artifact hashes](data/2026-09-29-inferencex-local-resumed/summary.json).
- [Original interrupted run](data/2026-09-29-inferencex-local/) and
  [interruption note](data/2026-09-29-inferencex-local/INTERRUPTED.md).
- [Seven freshly completed points](data/2026-09-29-inferencex-local-resumed/).
- [HTTP gate](data/2026-09-29-inferencex-http-gate-r2/),
  [actual upstream smoke](data/2026-09-29-inferencex-smoke/),
  [execution and verification logs](data/2026-09-29-inferencex-gates/).

Each measured point has unchanged `upstream.json`, `observer.json`, `server.log`,
`client.log`, `length-gate.json`, and reference prompt captures when applicable.
The combined manifest inherits29 original passed points and7 fresh points. Retain
both sibling directories; the summarizer supports relocated sibling directories.
The original manifest remains `running` as evidence of external interruption, **not**
a successful standalone run. Its incomplete30th point is excluded, not selectively
filtered by performance. The resumed run took approximately16minutes including gates.
The pause between rounds is a reproducibility limitation; no claim of an uninterrupted
single session. Resume checked binary/client/tokenizer identities and byte-identical
workloads. Later harness-only hardening also checks server command/env identity,
rejects gate-only/duplicate inherited cases, and has negative unit tests; it does not
change the measured binaries or rewrite historical manifests.

All builds and dependencies go through Bazel, Python through `tools/py`:

```sh
# Full run: set the OUTER tool timeout_ms=10800000 (three hours).
# This is not a CLI option. Individual client/startup/socket stall limits stay bounded.
tools/py bench/run_inferencex.py \
  --output docs/bench/data/NEW-inferencex-local \
  --inputs 1024,8192 --output-tokens 256 --prompts 8 --levels 1,4 --rounds 3

# Observed continuation command (use a new output directory on another invocation):
tools/py bench/run_inferencex.py \
  --output docs/bench/data/2026-09-29-inferencex-local-resumed \
  --resume-from docs/bench/data/2026-09-29-inferencex-local \
  --inputs 1024,8192 --output-tokens 256 --prompts 8 --levels 1,4 --rounds 3

tools/py bench/summarize_inferencex.py \
  docs/bench/data/2026-09-29-inferencex-local-resumed

# Fast iteration on harness code; full required gate at finalization:
bazel test //tests:test_serving_harness
bazel test //...
```

Native ignore_eos was a compatibility prerequisite: suppress EOG selection using the
existing sampler's allowed-ID path, not merely ignore a stop after sampling EOS.
Default behavior unchanged, strict Boolean parsing, unsupported tool combination
rejected. Independent162-case golden plus public generation/API tests cover masking,
ties, no logit mutation, ordinary EOS, stops, context, cancellation and bad EOS sets.
CPU Debug/ReleaseFast82/82 and host-driver2/2 gates passed before the main/resume runs
(cache hits distinguished in logs). No shader/kernel arithmetic changed. Observer
unit tests include real loopback byte preservation and role/text timing separation;
report tests reject missing/duplicate/failed/wrong-output matrix points and resume
identity/workload mismatches. Final focused `//tests:test_serving_harness` passed
(one target executed); final `bazel test //...` passed82/82 (one executed,81 cached).
See `harness-focused.log` and `final-tests.log` in the gates directory.

## Security review and retained failures

[Security review](../research/2026-09-29-inferencex-security.md),
[source/dependency/scan artifacts](data/2026-09-29-inferencex-security/),
[functional contract](../specs/inferencex-local.md).

Only hash-pinned reviewed client modules and36 wheel-only dependency artifacts were
fetched; no upstream setup/CI/upload scripts executed, no production dependency added.
Archive checks reject traversal/symlinks/.pth; hashes match canonical PyPI metadata and
lockfiles. ClamAV1.5.4/DB28129: **4446 files, zero infected, exit0**. Signatures were
**nine days old**; not all dependency/native code was audited line-by-line. A clean
scan **does not prove absence of malware**. Wrapper removes inherited credentials/
proxy environment, uses empty HOME, disables HF downloads/telemetry/remote code and
rejects non-loopback network/process creation via Python audit hooks. This is defense
in depth, **not an OS sandbox against malicious native extensions**.

Preserved negative results: initial ClamAV exit2 from readonly temporary storage;
initial native test build missing mock backend prefill; first HTTP-gate client import
failure from inherited Bazel runfiles environment; main run's external one-hour
cutoff. Corrected runs are separate artifacts. The standing `CLAUDE.md` rule now
requires a three-hour outer lifetime for this matrix, rather than disabling individual
stall limits. No malware-free guarantee, official dashboard equivalence, broad quality
acceptance, or general fastest-engine conclusion follows from these results.

## Follow-up: P90 interactivity (2026-09-29)

No new benchmark executed. Read-only calculation from the completed36 points:

```sh
tools/py docs/bench/data/2026-09-29-inferencex-local-resumed/interactivity.py
```

[Calculation script](data/2026-09-29-inferencex-local-resumed/interactivity.py),
[raw and adjusted numbers](data/2026-09-29-inferencex-local-resumed/interactivity.txt).
The [live dashboard](https://inferencex.semianalysis.com/inference/kimi-k3) help was
read on2026-09-29: it labels P90 Interactivity in tok/s/user and describes the rate
at which a single user receives tokens while streaming. That help does not specify
the percentile formula; this follow-up does not claim to have audited the current
website's calculation. Our explicit website-style calculation is the reciprocal
of p90 request-average TPOT: `1000 / p90_tpot_ms`, not p90 of request token rates,
and not aggregate throughput divided by concurrency.

Pinned upstream benchmark_serving.py lines433–437 calculates request TPOT as
`(latency - ttft)/(output_len - 1)` and lines490–491 uses NumPy's percentile.
The script uses the same linear percentile interpolation, derives a rate for each
trial, then reports the arithmetic mean and sample SD of those three rates.

Because of the native role-only chunk issue, raw TPOT includes prefill (and possibly
queueing) for zerv; it is not a fair streaming-speed comparison. Report **both** raw
and first-text-adjusted figures, never silently overwrite upstream data. The adjusted
estimate uses `(observer_end - first_nonempty_text_time)/(256 - 1)` per request,
then `1/p90(TPOT_seconds)`. It includes stream-finalization/forwarding overhead and
uses the first text event as a proxy for the first token; it is not an independently
instrumented device decode rate or an official InferenceX metric. Both references'
raw and adjusted rates happen to round identically to two decimals.

| Nominal input | C | zerv adjusted | llama Vulkan adjusted | llama HIP adjusted |
|---|---|---|---|---|
| 1024 | 1 | 44.73 ± 2.32 | 39.74 ± 1.72 | 32.56 ± 1.69 |
| 1024 | 4 | 35.82 ± 0.97 | 31.46 ± 1.17 | 27.13 ± 0.90 |
| 8192 | 1 | 40.79 ± 1.01 | 38.92 ± 0.57 | 31.73 ± 1.66 |
| 8192 | 4 | 18.48 ± 0.23 | 14.94 ± 1.00 | 15.44 ± 1.32 |

Units tok/s/user, mean±sample SD across three trials. Raw zerv reciprocals, in the
same order:36.23±1.83,28.58±0.65,18.98±0.21,11.50±0.14. Only eight requests per trial;
these are descriptive p90s. They exclude initial waiting after the first-text
adjustment, so pair them with the report's separate first-text latency. No fresh
inference, build configuration change, upload, or publication.

### Source audit clarification

[Follow-up source research](../research/2026-09-29-interactivity.md) now verifies the
actual conversion: pinned `results/fixed_sequence.py` derives `*_intvty` from
`1000/*_tpot_ms`, outside the raw benchmark client. The app's fixed-sequence axis
default is median; AgentX uses the selected percentile (the observed P90 chart),
with reciprocal ITL, preferring per-request full-response ITL. That full-response
formula is structurally `(end - first_content)/(OSL - 1)`, not necessarily pooled
SSE gaps. The custom adjusted table above is not an official AgentX result or a
literal fixed-sequence postprocessor output. Numerical tables remain unchanged;
the earlier “website-style P90” description needed this distinction.
