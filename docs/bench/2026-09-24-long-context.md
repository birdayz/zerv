# Long-context prefill and decode at the current context cap (2026-09-24)

Question: how do prefill and decode behave at the largest context the current code
supports? Context: the user's old Ollama setup served about 90k tokens
([previous deployment](../research/2026-09-23-previous-deployment.md)).

## Limit

- The FP32 KV cache takes 128 KiB per token. It lives in one state buffer capped
  at 3.75 GiB (32-bit addressing), so the maximum context is **29,504 tokens**
  (`layout.maxContext`, rounded down to a multiple of 32).
- **90k tokens** would need 11 GiB of FP32 KV. That does not fit next to 16 GB of
  weights on a 24 GB card.
- Candidates: an f16 KV cache (5.5 GiB at 90k) or a q8/q4 KV cache (2.9 / 1.5 GiB),
  plus splitting the state buffer. Not implemented.

## Measurement

Command: `zerv-model-profile MODEL 29504 512 29000 16 {f16,fp32}`, one run each,
one at a time. Data: [profile-f16](data/2026-09-24-long-context/profile-f16-p29000.jsonl),
[profile-fp32](data/2026-09-24-long-context/profile-fp32-p29000.jsonl).

| Mode | Prefill of 29,000 tokens (GPU) | Chunk at pos 0 | Chunk at pos 28,160 (attention share) | Decode at pos 29,007 |
| --- | --- | --- | --- | --- |
| f16 prefill | **46.2 s** (627 tok/s) | 521 ms | 1108 ms (56%) | 25.9 ms/token (38.7 tok/s) |
| fp32 prefill | **62.8 s** (462 tok/s) | 798 ms | 1413 ms (45%) | 25.9 ms/token (38.7 tok/s) |

The process wall times (100 s and 134 s) also include model load and the profile
tool's extra passes, so they are not TTFT.

- **Prefill attention** grows linearly per chunk, about 22 ms per 1000 positions of
  context. That is O(n²) in total. The FP32 score/softmax/PV path materializes the
  score matrix.
- **Decode attention** at 29k takes 6.3 ms of the 25.9 ms step. At 8k the whole
  step is 20.1 ms.

## Extrapolation to 90k (estimate, not a measurement)

This assumes a KV format that fits and the same per-position costs.

- **Prefill:** about 176 chunks.
  - f16: about 0.5 s of non-attention work per chunk plus 22 ms × position/1000 of
    attention, **about 4.3 min** in total.
  - fp32: **about 5.2 min**.
  - Attention would be about 2/3 of the total.
- **Decode:** at 90k context, about 39 ms/token (about 25 tok/s) with FP32 KV.
  - A half-size KV cache would reduce the attention part.

The biggest lever for long prompts is a fused (flash-style) prefill attention
kernel instead of the materialized-score path. It is not scheduled.

## Serving comparison with llama-server at 29k tokens

Question: how do llama-server's prefill and decode compare to zerv's at this length?

- Harness: [bench/run_long_context.py](../../bench/run_long_context.py). It uses the same
  engine commands and streaming client as `run_serving.py`, at context 29,504.
- Prompt: the serving-v2 cartography paragraph repeated 145 times, then "Summarize the
  passage above in about 200 words." That is 29,034 prompt tokens with the template.
  Thinking is off, greedy, max_tokens 128 (every request stops at the length limit).
- Each request starts with a distinct `Session NNNN.` tag. This rules out prefix or
  context-checkpoint reuse; llama reports `cached_tokens: 0` for every request. Every
  engine gets the same three prompts: one warmup, then two timed.
- Engines ran one at a time; the host was otherwise idle.
- Builds: zerv `40c1dcef…6735e1a86` (rebuilt from current source, byte-identical to
  the binary in the f16 serving report); llama-server build 10964 (`b29c606e`), binary
  `0f401589…bf77d85`.
- Command: `cd bench && python3 run_long_context.py --output
  ../docs/bench/data/2026-09-24-long-context/serving-p29k --zerv-binary
  ../third_party/serving-bench/2026-09-24-f16-serving-repeat/zerv --engines
  zerv,zerv-f16,llama-fa-ub512,llama-fa-ub512-nof16,llama-f32,llama-fp32-full`
- Data: [summary](data/2026-09-24-long-context/serving-p29k/summary.json),
  [manifest](data/2026-09-24-long-context/serving-p29k/manifest.json),
  [raw](data/2026-09-24-long-context/serving-p29k/raw.jsonl), server logs alongside.

| Engine | Prompt arithmetic / KV | TTFT s (2 runs) | Decode tok/s | VRAM loaded / peak MiB |
| --- | --- | --- | --- | --- |
| zerv | FP32 / FP32 | 62.78, 62.89 | 38.47, 38.49 | 21553 / 21651 |
| zerv-f16 | f16 WMMA, f32 accumulation / FP32 | 46.50, 46.60 | 38.25, 38.44 | 21552 / 21650 |
| llama-fa-ub512 (default) | f16 WMMA, f16 accumulation / f16 | 33.31, 33.33 | 37.01, 36.99 | 17368 / 17454 |
| llama-fa-ub512-nof16 | coopmat, f32 accumulation / f16 | 32.76, 32.83 | 37.23, 37.00 | 17368 / 17454 |
| llama-f32 | default prompt path / f32, no MMVQ | 36.05, 36.13 | 25.17, 25.17 | 19280 / 19366 |
| llama-fp32-full | FP32 / f32 | 97.00, 97.10 | 25.55, 25.57 | 19280 / 19344 |

- The idle desktop used 821 MiB of VRAM before each start.
- llama's own `prompt eval time` agrees with the client TTFT to within 0.1 s (for example 33,235 and 33,253 ms for the default).

Interpretation:

- **Prefill at 29k:**
  - llama's default is **1.40× faster** than zerv f16 and **1.89× faster** than zerv FP32.
  - At matched FP32 (llama-fp32-full), zerv is **1.55× faster**.
  - At 3,223 tokens, zerv f16 was within 4% of llama. The gap grows with length because
    zerv's prefill attention materializes the scores; llama uses flash attention. This
    agrees with the attention share measured above (45–56% of a chunk at 28k).
- **Decode at 29k:**
  - zerv (FP32 KV) runs 38.5 tok/s against 37.0 for llama's default with f16 KV.
  - With FP32 KV, llama drops to 25.2–25.6 tok/s, so zerv is 1.5× faster at matched KV precision.
- **VRAM:** zerv uses 4.2 GiB more than llama's default. Most of that is the FP32 KV
  cache (3.6 GiB for 29,504 positions, twice llama's f16); the materialized score buffer
  adds to it.
- **Outputs:** the three prompts differ in their session tag, so outputs are compared per
  request. zerv FP32, zerv f16 and llama-fp32-full produce byte-identical text for all
  three requests. llama's default, nof16 and f32-KV configurations diverge from that text
  on request 1 (and the default also on request 0), after 182–483 characters. This is
  consistent with the precision research: lower-precision paths flip greedy tokens. It is
  not a quality measurement.
- Limitations: one prompt shape (highly repetitive text), two timed runs per engine,
  single sequence.

## KV cache sizes at 90k tokens and prefix caching (proposal, not implemented)

Sizes per sequence, computed from the model shape: 16 attention layers × 2 (K, V) ×
4 KV heads × 256 = 32,768 values per token. The block formats use llama.cpp's
q8_0 (34 bytes per 32 values) and q4_0 (18 bytes per 32 values) layouts.

| KV format | Bytes per token | 29,034 tokens | 90,000 tokens |
| --- | --- | --- | --- |
| FP32 (current) | 131,072 | 3,805,544,448 | 11,796,480,000 |
| f16 | 65,536 | 1,902,772,224 | 5,898,240,000 |
| q8_0 | 34,816 | 1,010,847,744 | 3,133,440,000 |
| q4_0 | 18,432 | 535,154,688 | 1,658,880,000 |

- The DeltaNet state is a fixed size, independent of length: 48 × 48 × 128 × 128 FP32
  recurrent state (150,994,944 bytes) plus 48 × 3 × 10,240 FP32 conv history
  (5,898,240 bytes), **156,893,184 bytes** per snapshot.
- **Prefix caching** would restore the attention KV for `[0, p)` together with a
  DeltaNet snapshot taken at exactly `p`, then prefill only the rest of the prompt.
  - Attention KV can be cut at any position. DeltaNet state cannot be rewound, so reuse
    is possible only from a saved snapshot at or before the point where prompts diverge.
    llama.cpp's context checkpoints work this way (`--ctx-checkpoints`, default spacing
    8192 tokens).
  - A hit requires an exact token match from position 0. The cache key must include the
    model hash and the prefill precision.
  - Resuming at a position that is not on a 512-token chunk boundary changes the chunk
    schedule. Whether the result stays bit-identical to a cold run is open; the
    acceptance gate must define this.
- **Restore time (estimate, not measured):** PCIe 4.0 x16 at about 20 GB/s gives 0.2 s
  for a 29k FP32 entry. NVMe at 2–7 GB/s gives 1.7–6 s for a 90k FP32 entry, and 0.8–3 s
  in f16. Recomputing takes 46–63 s at 29k (measured) and 4–5 min at 90k (estimated).
- **Benchmark fairness:** our serving benchmarks disable llama's prompt cache
  (`--cache-ram 0`, and unique prefixes here). llama-server's defaults (8 GiB host-RAM
  cache plus checkpoints) and Ollama reuse prefixes across turns. zerv resets its state
  on every request ([serving spec](../specs/serving.md)). In multi-turn agent use, this
  difference outweighs any kernel speed gap.
