# Decode baseline: per-phase profile and competitor (block 17a, 2026-09-24)

Question: where does a zerv decode step spend its time at short and long context, and
how fast is llama-server's decode, including its speculative modes? This is the
baseline for block 17 ([design log](../design/speed.md)).

## Per-phase profile ([data](data/2026-09-24-decode-baseline/))

`zerv-model-profile MODEL CONTEXT CHUNK PROMPT STEPS` (FP32, GPU timestamps per phase,
one decode step after a prompt of 64 and of about 8,000 tokens). This profile predates
speculative decoding and the host embedding; the plain decode step has not changed
since (19.8 ms at position ~300 in [host memory](2026-09-24-host-memory.md)).

| Phase | ms at position 70 | share | ms at position 8,020 |
| --- | --- | --- | --- |
| ffn_in (gate + up, 64 layers) | 7.830 | 38.9% | 7.829 |
| ffn_down | 4.000 | 19.9% | 4.033 |
| lin_in (DeltaNet qkv, gate, α, β; 48 layers) | 2.928 | 14.5% | 2.910 |
| lin_out (ssm_out, Q5_K) | 1.444 | 7.2% | 1.446 |
| output (Q6_K) | 1.139 | 5.7% | 1.138 |
| attn_in (q, k, v; 16 layers) | 0.815 | 4.0% | 0.816 |
| delta (DeltaNet recurrence) | 0.580 | 2.9% | 0.585 |
| attn_out | 0.393 | 2.0% | 0.386 |
| attention (scores, pv, merge) | 0.547 | 2.7% | 1.784 |
| norms, swiglu, conv, qkprep, embed | 0.462 | 2.3% | 0.459 |
| **total GPU** | **20.14** | | **21.38** |

Weight bytes per projection phase (from the GGUF tensor table) and the effective read
rate:

| Phase | GB per token | GB/s | ms at 920 GB/s |
| --- | --- | --- | --- |
| ffn_in (Q4_0) | 6.417 | 820 | 6.98 |
| ffn_down (Q4_0 / Q4_1) | 3.253 | 813 | 3.54 |
| lin_in (Q4_0, F32) | 2.359 | 806 | 2.56 |
| lin_out (Q5_K) | 1.038 | 719 | 1.13 |
| attn_in (Q4_0) | 0.661 | 811 | 0.72 |
| attn_out (Q4_0) | 0.283 | 720 | 0.31 |
| output (Q6_K) | 1.043 | 916 | 1.13 |
| **all projections** | **15.05** | **811** (without output: 805) | **16.4** |

- 920 GB/s is the measured full-read rate of this card on a 1 GB buffer
  ([matvec push](2026-09-22-matvec-push.md)). The Q6_K output projection reaches it;
  the Q4_0 / Q4_1 / Q5_K projections run at 720–820 GB/s. Closing that gap is worth
  up to about 2 ms per token (10%) and is block 17c's matvec item. 08c found the large
  Q4_0 shapes instruction-bound, not read-bound. It rejected FMA dot accumulation
  (3–4% faster) because it changed output bits; the criterion is now the FP64 gates.
- Attention grows from 0.5 to 1.8 ms between positions 70 and 8,020 (FP32 KV, 128 KiB
  per token read). At 38k tokens decode is 34.8 tok/s against about 50 at short context
  ([flash report](2026-09-24-flash-attention.md), section 5). An f16 KV cache halves
  that read (block 17c).

## Competitor ([data](data/2026-09-24-decode-baseline/serving-decode-v1/))

decode-v1 (4 cases, greedy, 512 tokens, 2 repeats), decode tok/s. The llama-server
build and flags are in the manifest; MTP = `--spec-type draft-mtp --spec-draft-n-max N`.

| Case | zerv (no speculation) | llama | MTP 1 | MTP 2 | MTP 3 | MTP 4 |
| --- | --- | --- | --- | --- | --- | --- |
| code | 49.3 | 41.4 | 68.7 | 86.8 | 91.1 | 105.3 |
| json | 49.2 | 41.4 | 69.2 | 88.5 | 94.7 | 111.6 |
| think | 49.2 | 41.3 | 66.6 | 81.6 | 82.3 | 90.7 |
| prose | 49.3 | 41.3 | 59.7 | 62.7 | 55.6 | 58.5 |

- llama's n-gram speculation (`--spec-type ngram-mod`) stalled mid-request; the run was
  abandoned ([incident](data/2026-09-24-decode-baseline/serving-decode-v1/INCIDENT.md)).
- llama's MTP acceptance per drafted token was about 0.97 / 0.98 / 0.91 / 0.70 at one draft
  and 0.82 / 0.88 / 0.66 / 0.33 at four ([research note](../research/speculative-mtp.md)).
- The response: MTP speculation in zerv (block 17b,
  [report](2026-09-24-speculative.md)), which is lossless and faster than every llama
  configuration above.
