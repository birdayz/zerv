# Fused prefill attention, batched MTP catch-up, faster sampling (2026-09-24)

Questions:
1. Can prefill attention run without the materialized score matrix, within the FP64
   gates, and how much faster (block 16a)?
2. What does the MTP prompt catch-up cost, and can it be batched?
3. Why is sampled decode slower than greedy?

Specs: [prefill.md, "Fused prefill attention"](../specs/prefill.md),
[speculative.md](../specs/speculative.md). Machine and model as in
[kv-buffers](2026-09-24-kv-buffers.md). Every GPU run was checked for a concurrent
process first.

## 1. Fused prefill attention (`src/model/flash.comp`)

Before, attention materialized 24 × chunk × context FP32 scores: score GEMM, row softmax,
P·V GEMM. It ran at about 10 TFLOP/s, and the region grew with context (1.8 GB at
512 × 38,400).

**Kernel (v2, shipped).** One workgroup per KV head and 8 query rows. Six 64-thread groups
handle the 6 query heads that share the KV head, so the caches serve their identical K/V
reads.
- Scores over a 128-key tile: lane l owns keys 2l and 2l + 1. Q is wave-uniform (scalar
  loads) and K is read directly, one 4-dim step ahead.
- Row max and sum use subgroup reductions; the online softmax keeps m, l and O.
- P goes to LDS; each lane accumulates 4 head dims of the 8 rows (V read directly, 4 keys
  ahead).
- Accuracy structure: 32-dim partial sums per score and a per-tile output partial
  (O = O·a + O_tile).

**Development path** (per-phase GPU time of an 8,192-token fp32 prefill,
`zerv-model-profile MODEL 8704 512 8192 4 fp32`; single runs,
[raw](data/2026-09-24-flash/profiles/)):

| Variant | prefill GPU ms | attention ms | chunk at 0 | at 3584 | at 7680 |
| --- | --- | --- | --- | --- | --- |
| materialized (before) | 14,275 | 2,030 | 11.9 | 124.7 | 237.7 |
| v1: 64 rows × 1 head, K/V staged in LDS, 4×4 micro-tiles | 13,644 | 1,480 | 11.3 | 87.8 | 172.7 |
| v2 (above) without prefetch | 14,076 | 1,867 | — | — | 223.7 |
| **v2 with prefetch** (RW 8, 6 heads, 4 keys per P·V step) | **12,978** | **722** | **4.4** | **42.4** | **86.5** |
| v2, RW 4 | 13,124 | 802 | 4.8 | 46.7 | 95.9 |
| v2, RW 12 | 13,176 | 939 | 6.4 | 55.4 | 110.4 |
| v2, RW 16, 3 heads (252 VGPRs) | 13,933 | 1,734 | 10.7 | 102.7 | 199.9 |
| v2, RW 16, 2 heads | 13,363 | 1,158 | 7.3 | 70.7 | 133.8 |
| v2, subgroup instead of workgroup barriers around P | 13,022 | 720 | 4.5 | 42.1 | 86.0 |
| v2, 8 keys per P·V step (252 VGPRs) | 13,167 | 918 | 5.4 | 53.9 | 108.8 |

- Attention is 2.8× faster over an 8k prompt (2,030 → 720 ms), about 18.6 TFLOP/s at the
  7680 chunk against about 6.7 before.
- v1 lost to LDS bandwidth: two 16-byte LDS reads per 16 FMAs.
- v1 was also less accurate: one 256-term dot-product chain and one output chain over
  all keys. `attn_pregate` mean normalized L2 error was 1.0e-6 against 4.8e-7
  materialized; `attn_output` reached 0.956 of its bound in the chunked modes. Partial
  sums fixed that (component worst error 2.2e-3 → 6.8e-4 of the bound).

**Gates (all executed):**
- **Component:** `tests/model_gpu.zig`, the FP64 first-order bound. Covers row blocks,
  tile edges, a context not a multiple of 64, NaN past the live keys and past the
  count, and untouched outputs. Worst error is 7.8e-4 of the bound; 26/26 GPU tests.
- **Model:** `tools/verify_model.py`, default oracles modes 0/1/13/512/512:17 and long
  oracle modes 0/13/128/512/512:300; all pass. The worst per-case tensor is at
  0.25–0.75 of its bound (the materialized path had 0.72 on the long oracle).
  Reports: [default](data/2026-09-24-flash/verify-final-default.json),
  [long](data/2026-09-24-flash/verify-final-long.json).
- **Decode unchanged:** mode-0 logits and captures are bitwise identical to the
  pre-change build on all three oracle cases.

**Materialized path removed.** It had no remaining advantage. Removed: the option, the
softmax kernel, the attention GEMM kernels and the prefill score matrix. The other
model modules are byte-identical. The score region keeps decode's 24 × decode rows ×
context.

**Process failure found and fixed before any release.** The first build sized the score
region to zero, but decode and verify write their scores there. They then wrote into
the split-K partials, which was harmless at 512-row chunks but not guaranteed for small
plans. Details are in `TODO.md`. Fixed, with a layout test, an init check and mode 0 in
the gate.

## 2. Batched MTP catch-up

The MTP KV depends only on u = `eh_proj([enorm(embed x); hnorm(h)])`, `attn_norm`, K, V,
and the k-norm and RoPE. The ≤ 5-row catch-up passes computed the whole layer at about
0.33 ms per prompt token.

It now runs inside the prefill command:
- the embeddings into `cat` (row stride 2H), then `copy2d` of the shifted h rows beside
  them;
- in-place enorm and hnorm;
- `eh_proj` on a new Q8_0 GEMM (`gemm.comp` FORMAT 8, exact d·q);
- `attn_norm`, K and V GEMMs, `qk_b` into KV cache 16;
- `hp` ← the last row.

**Gates:**
- The Q8_0 GEMM against FP64 (wide tile = narrow tile bitwise, split-K, untouched dead
  rows).
- The MTP FP64 gate: h' and logits within 2–5e-7, every draft the FP64 argmax
  ([data](data/2026-09-24-mtp-gate/)).
- `zerv-spec-check` 11/11.

TTFT, serving-v2 and long-v1 (the f16 speculative outputs byte-identical to the
non-speculative ones):

| Prompt | f16, 3 drafts, before | after | f16, no speculation |
| --- | --- | --- | --- |
| 836 tokens | 1,096 ms | 968 ms | 961 ms |
| 3,223 tokens | 3,574 ms | 3,081 ms | 3,072 ms |
| 37,827 tokens | 82.2 s | 70.0 s (46.2 s with fused attention) | 70.0 s (46.0 s) |

([serving-v2 before](data/2026-09-24-speculative/serving-v2/),
[after](data/2026-09-24-speculative/serving-v2-catchup/)).

## 3. Sampling

Sampled decode ran at 46 tok/s against 48 greedy. The sampler copied all 248,320 logits
into candidate structs and quickselected them for every token, about 0.9 ms.

`top_k` (≤ 256, which includes the model defaults) now takes one vectorized pass with a
running k-th-best threshold. Penalized ids are inserted afterwards with their penalized
logit. The order (logit, then id) is total, so the kept set is identical, and so is the
rest of the chain.

Gate: draw-for-draw equality with the old full-array path (`fast_top_k = false`) over
12 × 60 draws on 50k-token vocabularies with ties and positive and negative penalties.
Greedy argmax is vectorized too. Speed measurement follows in the next serving run.

## 3b. Wave32 (negative result, 2026-09-24 later)

RDNA3 dual-issues FMAs (VOPD) only in wave32, and the kernel supports 32-wide subgroups
(cross-subgroup reductions through LDS). An experiment build created the flash pipeline
with a required subgroup size (`zerv-model-profile MODEL 8704 512 8192 4 fp32`, 2 runs
each, [data](data/2026-09-24-flash/wave32/)):

| Subgroup size | prefill GPU ms | attention ms | chunk at 7680 |
| --- | --- | --- | --- |
| driver default (64) | 12,977 / 12,967 | 717 / 716 | 85.7 / 85.5 |
| required 32 | 13,180 / 13,132 | 939 / 936 | 112.0 / 111.1 |
| required 64 | 12,988 / 12,963 | 717 / 717 | 85.7 / 85.8 |

Wave32 is 31% slower; the experiment option was removed. Remaining levers for prefill
attention: cooperative-matrix (WMMA) attention in the f16 prefill mode (a precision
change, opt-in), recorded in `TODO.md`.

## 4. Long context against llama-server (37,827 tokens, cold, [raw](data/2026-09-24-flash/long-v1/))

`bench/run_serving.py --workload bench/workloads/long-v1.json --context 38400
--no-prompt-cache`, 2 repeats. All engines answered correctly.

| Engine | TTFT s | decode tok/s (23 tokens) |
| --- | --- | --- |
| zerv f16 | 46.0 / 46.0 | 36.2 |
| zerv f16, 3 drafts | 46.2 / 46.2 | 58.4 |
| zerv fp32 | 73.9 / 74.3 | 36.1 |
| llama-server default (FA, ub 512) | 46.6 / 46.6 | 36.4 / 37.3 |
| llama-server MTP 3 | 50.4 / 50.4 | 71.6 / 71.5 |

- Before fused attention: zerv f16 took 70.0 s and fp32 112.3 s on the same prompt
  ([kv-buffers](2026-09-24-kv-buffers.md)).
- Open at long context: llama's MTP decode is faster (71.5 against 58.4). zerv's verify
  attention ran one pass per row, each re-reading the whole KV cache from DRAM. Fixed
  next (below); decode speed at long context is measured with the `decode-38k` case.

## 5. Verify attention grid order, long context (37,822–37,827 tokens, cold, [raw](data/2026-09-24-flash/long-v1-r2/))

Change: the verify pass's `attn_scores` and `attn_pv` grids are now (rows, chunks, KV
heads) rather than (chunks, KV heads, rows). The rows of one chunk now run next to each
other and share that chunk's K/V through L2, instead of each row streaming the whole
cache from DRAM. The arithmetic per output is unchanged. Gates: verify_model modes
0/1/13/512/512:17 bitwise equal to the previous captures, and `zerv-spec-check` 11/11.

Same command as section 4, with engines zerv-f16, zerv-f16-spec2, zerv-f16-spec3,
llama-fa-ub512 and llama-fa-ub512-mtp3; 1 repeat. The `decode-38k` case asks for a
256-token summary of the same documents.

| Engine | needle TTFT s | needle tok/s (23) | decode-38k TTFT s | decode-38k tok/s (256) |
| --- | --- | --- | --- | --- |
| zerv f16 | 46.0 | 36.5 | 46.0 | 34.8 |
| zerv f16, 2 drafts | 46.3 | 81.2 | 46.4 | 72.9 |
| zerv f16, 3 drafts | 46.6 | 81.1 | 46.6 | **80.8** |
| llama-server default | 46.6 | 37.5 | 46.8 | 35.8 |
| llama-server MTP 3 | 50.3 | 71.5 | 50.4 | 76.3 |

- All ten outputs are equal to each other, byte for byte, in both cases (needle: "ORCHID-7431 … 58").
- The grid order took 3-draft decode at 38k from 58.4 to 81.1 tok/s (needle case,
  section 4 against this run).
- Without speculation, llama is 1.03× faster at 38k (35.8 against 34.8). zerv's KV
  cache is FP32 (128 KiB per token, 4.8 GB read per token at 38k, on top of 15.3 GB of
  weights); llama's is f16. The f16 KV knob (block 17c) targets this.
