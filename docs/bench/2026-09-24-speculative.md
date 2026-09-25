# MTP speculative decoding: engine loop, lossless serving, speed (block 17b.3, 2026-09-24)

Questions: does speculative decoding with the GGUF's MTP layer leave every output
unchanged (gate 3)? Does its draft acceptance match llama.cpp's (gate 2, second half)?
Is it faster than llama-server's best configuration, including its own MTP speculation
(gate 4)? [Spec](../specs/speculative.md), [research](../research/speculative-mtp.md).
Gates 1 and 2's component half: [spec-verify](2026-09-24-spec-verify.md) and
[mtp-gate data](data/2026-09-24-mtp-gate/).

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV; Qwen3.8-27B-Q4_0 (sha256 `ede16c7b…`), the same file
  for both engines. llama-server 0.4.1-dev build 10964 (commit b29c606e28), Vulkan,
  `-fa on -b 2048 -ub 512`, MTP with `--spec-type draft-mtp --spec-draft-n-max N`.
- Harness: `bench/run_serving.py` (streaming SSE client, one engine at a time, warmup,
  manifests with binary hashes and commands). zerv binaries per run are in each
  manifest (`zerv_sha256`).
- zerv engines: `zerv-specN` = `--spec-draft N` with the adaptive verify-count policy
  (server default), `zerv-specN-fixed` = every draft verified; `zerv-f16-*` adds
  `--prefill-precision f16`.
- Workloads: `bench/workloads/decode-v1.json` (4 cases × 512 tokens: code, json, think,
  prose; greedy; seed 1234), `decode-v1-sampled.json` (the same at temperature 0.8,
  seed 1234), `serving-v2.json`, `long-v1.json` (37.8k-token prompts).

## Gate 3: lossless serving (passed)

Every zerv speculative run produced exactly the non-speculative output (output hash per
case):

| Run | Engines | Result |
| --- | --- | --- |
| [decode-v1-r2](data/2026-09-24-speculative/decode-v1-r2/) (greedy, 2 repeats) | zerv, zerv-spec1..4 (every draft verified; before the policy existed) | 4/4 cases equal |
| [decode-v1-adaptive](data/2026-09-24-speculative/decode-v1-adaptive/) | zerv-spec2..4 (adaptive) | equal to zerv in r2 |
| [decode-v1-sampled](data/2026-09-24-speculative/decode-v1-sampled/) (T = 0.8, top-k/top-p, seed) | zerv, zerv-spec2..4 | 4/4 cases equal |
| [serving-v2](data/2026-09-24-speculative/serving-v2/), [serving-v2-catchup](data/2026-09-24-speculative/serving-v2-catchup/) | zerv, zerv-spec3, zerv-f16, zerv-f16-spec3 | equal per precision |
| [long-v1-r2](data/2026-09-24-flash/long-v1-r2/) (37.8k tokens) | zerv-f16, -spec2, -spec3, llama default, llama MTP 3 | all ten outputs identical |

For comparison, llama-server in decode-v1-r2: its default run already gave 2 different
outputs in 2 repeats on think and prose. Its MTP 2–4 runs produced outputs its default
run never did (json, think, prose; code matched). zerv gave one output per case across
all engines and repeats.

A first run failed: [decode-v1-failed-catchup-bug](data/2026-09-24-speculative/decode-v1-failed-catchup-bug/).
The catch-up requested more rows than `verify_rows` allowed. Fixed before the runs above.

## Gate 2, acceptance (passed): [decode-v1-acceptance](data/2026-09-24-speculative/decode-v1-acceptance/)

```
python3 bench/run_serving.py --workload bench/workloads/decode-v1.json \
  --output docs/bench/data/2026-09-24-speculative/decode-v1-acceptance --repeats 1 \
  --engines zerv-spec3-fixed,zerv-spec3,zerv-spec4-fixed,llama-fa-ub512-mtp3,llama-fa-ub512-mtp4
```

New counters: zerv's `/metrics` (`zerv_spec_draft_tokens_total{stage=...}`) and
llama-server's per-response `timings.draft_n` / `draft_n_accepted`, both recorded per
request by the harness. Accepted / verified drafts:

| Case | zerv 3 fixed | llama MTP 3 | zerv 4 fixed | llama MTP 4 | zerv 3 adaptive |
| --- | --- | --- | --- | --- | --- |
| code | 369/426 (0.866) | 369/426 (0.866) | 391/478 (0.818) | 391/478 (0.818) | 368/418 (0.880) |
| json | 374/408 (0.917) | 375/405 (0.926) | 397/452 (0.878) | 397/452 (0.878) | 373/398 (0.937) |
| think | 352/475 (0.741) | 352/475 (0.741) | 366/577 (0.634) | 370/562 (0.658) | 350/443 (0.790) |
| prose | 280/689 (0.406) | 269/724 (0.372) | 289/881 (0.328) | 289/880 (0.328) | 274/537 (0.510) |

- Where both engines produce the same text, the counts are identical: code at 3 and 4
  drafts, think at 3 drafts, json at 4 drafts. That is strong evidence that zerv's MTP
  layer computes the same drafts as llama.cpp's. The other cells differ slightly
  because llama's output diverges from greedy (see gate 3).
- The adaptive policy verifies fewer of the doubtful drafts: acceptance of verified
  drafts rises, at nearly the same accepted count.

## Gate 4: speed (passed)

Decode tok/s, decode-v1 (greedy), medians of 2 repeats
([r2](data/2026-09-24-speculative/decode-v1-r2/), [adaptive](data/2026-09-24-speculative/decode-v1-adaptive/));
the acceptance run above (1 repeat) in brackets:

| Engine | code | json | think | prose |
| --- | --- | --- | --- | --- |
| zerv (no speculation) | 48.2 | 48.1 | 48.1 | 48.4 |
| zerv 2 drafts, adaptive | 101.5 | 103.1 | 95.2 | **74.4** |
| zerv 3 drafts, adaptive | 110.0 [111.7] | 114.0 [116.0] | 99.1 [100.9] | 71.2 [72.1] |
| zerv 3 drafts, fixed | 108.7 [112.6] | 113.1 [116.6] | 98.2 [100.5] | 68.1 [69.2] |
| zerv 4 drafts, adaptive | 108.1 | 113.7 | 92.2 | 67.8 |
| llama-server default | 41.1 | 41.3 | 41.2 | 40.7 |
| llama MTP 2 | 87.2 | 88.9 | 81.8 | 63.2 |
| llama MTP 3 | 91.9 [92.3] | 95.4 [91.6] | 82.8 [81.9] | 55.5 [54.4] |
| llama MTP 4 | 105.6 [105.1] | 109.1 [112.0] | 91.0 [90.9] | 58.3 [58.5] |

Best against best: code 110.0 vs 105.6 (1.04×), json 114.0 vs 109.1 (1.04×), think
99.1 vs 91.0 (1.09×), prose 74.4 vs 63.2 (1.18×). In the acceptance run: 112.6 vs
105.1, 116.6 vs 112.0, 100.9 vs 90.9, 72.1 vs 58.5.

Sampled (T = 0.8, [decode-v1-sampled](data/2026-09-24-speculative/decode-v1-sampled/);
zerv only, since llama's sampled MTP output is not comparable): no speculation 46.2, 2
drafts 87.5 / 92.4 / 79.8 / 68.0, 3 drafts 92.9 / 101.5 / 78.8 / 64.8 tok/s.

Serving-v2 (2 repeats, [after batched catch-up](data/2026-09-24-speculative/serving-v2-catchup/),
llama from [serving-v2](data/2026-09-24-speculative/serving-v2/)), TTFT ms / decode tok/s:

| Engine | short-nothink | decode-think | medium-prompt | long-prompt |
| --- | --- | --- | --- | --- |
| zerv f16, 3 drafts | 144 / 125.0 | 309 / 105.4 | 968 / 97.2 | 3081 / 108.2 |
| zerv fp32, 3 drafts | 144 / 125.8 | 310 / 105.1 | 1492 / 97.5 | 5401 / 107.8 |
| llama default | 207 / 41.9 | 423 / 40.0 | 1350 / 40.4 | 3548 / 43.6 |
| llama MTP 3 | 212 / 86.8 | 444 / 86.9 | 1363 / 80.7 | 3775 / 70.4 |
| llama MTP 4 | 215 / 88.7 | 444 / 100.5 | 1365 / 85.4 | 3729 / 74.1 |

Long context (37.8k tokens, [flash report section 5](2026-09-24-flash-attention.md)):
zerv f16 with 3 drafts decodes 256 tokens at 80.8 tok/s, against 76.3 for llama MTP 3.
TTFT is 46.6 s against 50.4 s.

## Also in this step

- Plain `step()` with the MTP layer now runs the pending MTP rows (it dropped them).
  Gate: `zerv-mtp-check` scenario C, bitwise; the old step fails it
  ([data](data/2026-09-24-speculative/step-mtp/)). It affects only draft quality after a
  step, so the measurements above are unaffected.

## Interpretation

- Speculation is lossless in every run, greedy and sampled, and is faster than every
  llama-server configuration on every decode-v1 case, serving-v2 case and the 38k case.
- The draft count is a real trade-off. Three drafts is best for code, json and think;
  two for prose, where acceptance is low. The adaptive policy makes 3 close to the best
  everywhere. `--spec-draft` stays opt-in (0 = off, no MTP memory): it costs 0.27 GB of
  weights, 8 KiB per context position for the MTP KV, and verify buffers.
- Not measured here: concurrency (the server serializes requests), and acceptance on
  agent (tool-calling) traffic.
