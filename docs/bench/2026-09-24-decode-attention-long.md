# Decode attention at long context: global-max and parallel-combine passes (block 17c, 2026-09-24)

Question: at 64k keys, plain decode spent about 11 ms of a 30.6 ms step in attention.
Which parts grow faster than the K/V bytes, and can they be removed without changing
any result? [Spec](../specs/decode-attention.md) ("Long-context passes").

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV; Qwen3.8-27B-Q4_0.
- `zerv-model-profile MODEL CONTEXT 512 PROMPT STEPS f16 f16` (f16 prefill, f16 KV; GPU
  timestamps per phase; the decode line is the median step). "before" is the binary
  saved at `third_party/prof-baseline/zerv-model-profile` (sha256 `5b5feed35fd1e002…`),
  built before these changes and before FMA matvec accumulation. The other binaries
  include FMA accumulation, which did not change the projection phases (ffn_in 7.814 →
  7.809 ms at 64k).
- Raw profiles: [data](data/2026-09-24-decode-attention-long/).

## What grew (reading of `model.comp`)

- P·V (pass 2): each of the chunks × 4 workgroups first reduced all live chunk maxima of
  its 6 heads to the global maximum. At 64k that is 1000 workgroups × 1000 maxima per
  head: quadratic in the context.
- Combine (pass 3): 24 workgroups (one per head), each summing all 1000 chunk partials
  serially per output dimension.

## Changes

1. `attn_gmax`: the global maxima computed once per (row, head) in a separate pass;
   P·V reads them (the same code, moved).
2. `attn_cblock`: the 8-chunk block sums in parallel (one workgroup per block and head);
   the combine then sums the blocks in order (the same two-level loop, split).

Both keep every operation and its order, so results are bit-identical.

## Correctness

- `tools/verify_model.py`, default oracle modes 0/1/13/512/512:17 and long oracle modes
  0/13/128/512/512:300: all 28 + 14 capture and logits files byte-identical to the build
  before each change (after gmax: against the FMA build; after cblock: against the gmax
  build).
- `zig build gpu-test` 29/29 (split-K decode attention against FP64, f32 and f16 KV);
  `zig build test` 85/85; `zerv-spec-check` 11/11 (verify ≡ decode).

## Results (decode step, GPU ms; phases: scores, P·V including gmax, combine including cblock)

| Position | before | gmax | gmax + cblock |
| --- | --- | --- | --- |
| 64,006 | 30.64 (2.93 / 3.99 / 4.34) | 30.19 (2.97 / 3.56 / 4.32) | **26.97** (2.99 / 3.59 / **0.73**) |
| ~8,200 (2 runs) | 20.92 / 20.93 (0.39 / 0.48 / 0.44) | — | **20.68 / 20.69** (0.43 / 0.48 / 0.17) |
| ~310 (2 runs) | **20.18 / 20.18** (0.21 / 0.33 / 0.03) | — | 20.25 / 20.25 (0.21 / 0.38 / 0.08) |

- At 64k: −3.67 ms per step (−12%); plain decode about 32.6 → 37.1 tok/s at that
  position (from GPU time; not a serving measurement).
- At 8k: −0.25 ms. **At short context: +0.07 ms (+0.35%)**, the cost of two more
  dispatches and barriers in each of the 16 attention layers. Recorded as a loss; a
  possible fix is folding the maxima into the scores pass (the last chunk's workgroup),
  not attempted.

## Scores on key pairs

`attn_scores` now gives each thread two adjacent keys (K pairs as one 64-bit f32 or
32-bit f16 load per dim; Q read from LDS as `vec4` and reused for both keys); a
workgroup covers two 64-key chunks and reduces each chunk's maximum separately. Each
key's arithmetic is the one-key kernel's. The context must now be even for f32 KV too
(f16 already required it).

- Byte-identical on both oracles (28 + 14 files, against the cblock build), gpu-test
  29/29, spec-check 11/11 (f32 and f16 KV).
- Decode step GPU ms (2 runs each): 64k 26.97 → 26.77 / 26.74 (scores 2.99 → 2.78);
  8k 20.68 / 20.69 → 20.71 / 20.71; position 300 20.25 / 20.25 → 20.26 / 20.28 (scores
  0.21 → 0.25: half the workgroups at short context). A small gain at long context and
  a small loss at short context; kept.

## P·V with wide V loads (after the serving run below)

P·V read V at about 590 GB/s at 64k: 256 threads, one dim each, so each wave loads 128
bytes (f16) per key. Three layouts, each bitwise identical (every dim keeps its four
key-lane accumulators and (l0+l1)+(l2+l3)):
- **pv64:** one wave, 4 dims per thread (one 8-byte / 16-byte load per key);
- **pv4w:** 4 waves, wave w accumulates key lane w for all dims with 4-dim loads; the
  lanes meet in 24 KB of LDS;
- **pv2w:** 2 waves, wave w accumulates lanes 2w and 2w+1 and adds them in registers;
  wave 1's sum reaches wave 0 through 6 KB of LDS.

Interleaved race of profile binaries ([data](data/2026-09-24-decode-attention-long/pv-race/),
hashes in `binaries.sha256`), decode step GPU ms (P·V phase ms), 2 runs each:

| Position | before (all 17c changes) | pv64 | pv4w | **pv2w** |
| --- | --- | --- | --- | --- |
| ~300 | 20.17 / 20.19 (0.33) | 20.28–20.35 (0.43) | **20.10** (0.20) | 20.18 (0.28) |
| ~8,200 | 20.92 / 20.94 (0.48) | 20.71–20.72 (0.48) | 20.71 (0.54) | **20.65** (0.43) |
| ~32,000 | — | **23.07** (1.58) | 23.35 (1.95) | **23.07** (1.66) |
| ~64,000 | 30.71 / 30.73 (3.99) | **26.42–26.59** (3.35) | 26.84–26.96 (3.84) | 26.56 / 26.58 (3.45) |

pv4w loses at long context (its LDS limits occupancy), pv64 at short context (a quarter
of the waves). **pv2w shipped:** equal to the original at short context, best at 8k,
within 0.05 ms of the best at 32k and 64k. Gates: byte-identical on both oracles
(28 + 14 files), gpu-test 29/29, spec-check 11/11 with f32 and f16 KV.

Net of all changes in this report at short context: 20.18 → 20.18 ms (the +0.07 ms of
the extra passes is recovered by P·V).

## Serving at 75.5k tokens (long-v2)

`bench/run_serving.py --workload bench/workloads/long-v2.json --engines
"zerv-f16@kv-type=f16;zerv-f16-spec3@kv-type=f16" --context 87040 --no-prompt-cache
--repeats 1` (binary `afa4c6e32d6c777f…`: gmax, cblock, scores pairs, FMA matvec;
[data](data/2026-09-24-decode-attention-long/long-v2-zerv/)). Before: the same workload
in [kv-precision](2026-09-24-kv-precision.md), same day, llama from that run.

| Engine | TTFT s (needle / decode) | needle tok/s (23) | decode-long-v2 tok/s (256) |
| --- | --- | --- | --- |
| zerv f16 KV, before | 119.9 / 119.6 | 32.0 | 30.6 |
| **zerv f16 KV, after** | 120.3 / 119.8 | **37.4** | **35.7** |
| zerv f16 KV, 3 drafts, before | 120.5 / 120.6 | 64.9 | 51.4 |
| **zerv f16 KV, 3 drafts, after** | 120.5 / 120.1 | **76.7** | **60.6** |
| llama-server default | 115.7 / 115.5 | 33.7 | 32.4 |
| llama-server MTP 3 | 127.9 / 126.7 | 61.4 | 49.9 |

- Outputs identical to the before run (both cases, both engines).
- Plain decode at 75.5k is now 10% faster than llama's (it was 6% slower); with 3
  drafts 21% faster than llama MTP 3. Prefill (TTFT) is still 3.5% slower than llama's.

## Remaining

- At 64k the attention phases are still 7.3 ms. With f16 KV, K and V are 2.1 GB each
  over 16 layers (4 heads × 256 dims × 64k × 2 bytes per layer): scores read K at about
  700 GB/s, P·V read V at about 580 GB/s, against 920 GB/s. The scores scratch
  (24 × 64k FP32 per layer, 0.1 GB in total) is minor. Serving comparison at long
  context after the FMA re-tune is next.
