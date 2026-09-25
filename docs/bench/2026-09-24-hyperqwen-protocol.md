# Single-user decode on HyperQwen's protocol (2026-09-24)

Question (user): a Medium article claims 177 tok/s for Qwen3.8-27B on one RTX 3090. Are we that
much slower? Answer: the 177 is aggregate throughput of 8 concurrent requests. For **one user**,
HyperQwen's own README reports 120 / 111 tok/s (MTP, greedy / default sampling) on a 3090; on the
same protocol zerv reaches **91.3 / 79.5 tok/s** on the RX 7900 XTX, 34% above a tuned llama-server
with MTP on this machine. The gap to HyperQwen is the cost of a speculative step, not the base
decode (ours is faster).

## Setup

- Protocol: HyperQwen's single-user benchmark (README of github.com/syv-ai/HyperQwen @
  `1cf86656`; the numbers there are vLLM 0.27.1 measurements on an RTX 3090 at 250 W with its own
  ~20 GB requantization): the 8 prompts of `bench/prompts_real.jsonl` (sha256 `27da4fd8…`),
  1,024-token answers, one stream, the model's chat template with thinking on, model-default
  sampling (T 1.0, top-p 0.95, top-k 20) and greedy. Decode = 1000 / mean TPOT.
- Workloads (ours, converted): [bench/workloads/hyperqwen-real-v1-greedy.json](../../bench/workloads/hyperqwen-real-v1-greedy.json),
  [-default.json](../../bench/workloads/hyperqwen-real-v1-default.json), `run_serving.py
  --engines zerv-f16-spec3,zerv-f16,llama-fa-ub512-mtp3 --repeats 1`, one engine at a time.
- zerv `c88642a1…` (f16 prefill, native GEMM; 3 drafts adaptive, or plain); llama-server build
  10964 (`b29c606e`) with `--spec-type draft-mtp --spec-draft-n-max 3 -fa on -b 2048 -ub 512`.
  Model sha `ede16c7b…`. Data: [greedy](data/2026-09-24-hyperqwen-protocol/greedy/),
  [default](data/2026-09-24-hyperqwen-protocol/default/) (manifests with all hashes and commands).

## Results

Decode tok/s (1000 / mean TPOT), 8 prompts:

| Engine | Greedy | Default sampling | Tokens per step (greedy / default) |
| --- | --- | --- | --- |
| **zerv, 3 MTP drafts** | **91.3** | **79.5** | 2.62 / 2.21 |
| zerv, plain | 50.9 | 51.2 | 1 |
| llama-server, MTP 3 | 68.2 | 59.8 | — |
| HyperQwen MTP (3090, their README) | 120.0 | 111.1 | 2.90 / 2.75 |
| HyperQwen DFlash2 (3090, their README) | 131.2 | 121.8 | 3.34 / 3.12 |
| HyperQwen plain (their batch mode, C1) | 45–46 | 45–46 | 1 |

Mean TTFT: zerv 222 ms, llama-server 405–563 ms (their README: 164 ms on the 3090 setup).
zerv's outputs: 7 of 8 answers use all 1,024 tokens.

## Interpretation

Not an equal-conditions comparison: different GPU, different quantization (their ~20 GB
requantization vs our 15.3 GB Q4_0), their numbers are from their README and not re-run here. But
the per-step arithmetic is informative:

- Plain step: zerv 19.6 ms, HyperQwen 21.7 ms (1/46).
- Speculative cycle (drafting + verification): zerv 91.3 / 2.62 → **28.7 ms, +46% over a plain
  step**; HyperQwen 120 / 2.90 → 24.2 ms, +12%.
- Our cycle, from earlier component measurements: verifying 3 rows costs 22.7 ms, and each draft
  ~1.55 ms, 80% of it the Q6_K output head ([draft-vocab report](2026-09-24-draft-vocab.md)).
  HyperQwen's README credits a calibrated draft vocabulary and requantized heads for its cheap
  draft.
- Acceptance under T = 1 sampling: we accept a draft only when the sampled token equals it, so
  speculative output is byte-identical to non-speculative output (our gate 3). vLLM uses
  rejection sampling: the same output distribution, higher acceptance (2.75 vs 2.21 tokens per
  step here), not seed-identical.

## Consequences (TODO.md)

1. **Draft cost** is the largest single-user decode lever: the frequency-ranked draft vocabulary
   (queued in 17c) with a representative, non-benchmark frequency source. Est. +10–15%.
2. **Distribution-exact speculative sampling** as an opt-in knob (rejection sampling). Est. +20%
   at T = 1; changes which sample is drawn, not the distribution; needs its own research and a
   statistical gate.
3. Multi-user: their 177 / ~400 / ~1,035 tok/s aggregate numbers need batched decode, which zerv
   does not have (the next block).
