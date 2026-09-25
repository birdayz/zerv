# `--decode-precision f16`: WMMA batched decode projections, kernel v1 (block 18e, part 2) — 2026-09-25

**Question.** The exact FP32 multi-row projection stops scaling past 4 rows
([part 1](2026-09-25-fp32-batched-projection.md)). Does the opt-in f16 arithmetic (the f16
prefill mode's WMMA, [spec](../specs/concurrent.md) "18e design") make a batched step with many
rows cheap, while staying batch-invariant within the mode?

**Answer.**
- **Batch invariance: yes.** `zerv-batch-check` in f16 decode mode is 399/399 bitwise equal to
  the same mode decoding each sequence alone, with staggered joins and leaves, row orders
  permuted every step, B = 1..8 and permuted page tables.
- **Speed: not yet.** Kernel v1 is flat in the row count, as intended: 45.4 ms at 1 row and
  50.0 ms at 8. But its base cost is too high. At 8 rows it roughly ties the FP32 path
  (46.3 ms), and below that FP32 is faster.
- The mode stays internal (`Model.Options.decode_precision`); no CLI flag until it wins.

## What exists

- `gemm_f16.comp -DSMALLN=1` → `gemm_f16n_{q4_0,q4_1,q5_k}`:
  - a 128 × 16 tile, 4 subgroups of 32 × 16;
  - the per-element WMMA chain of `gemm_f16`;
  - split-K over workgroup z with a shape-only chunk (`gemm.f16nChunk`, about 384 workgroups).
- The prefill `gemm_f16_*` modules are byte-identical to before.
- `gemm.validateF16n`, `moduleF16n` and `f16DecodeEligible`.
- In the runtime: `dprojection` builds the f16 projections of the batched decode (q, out, qkv,
  z, gate, up, down, ssm_out) with split parts in the shared part slots. `recordBatch` uses
  them, with separate gate/up/swiglu (no fused f16 SwiGLU yet); the remaining projections stay
  FP32.
- Tests:
  - gpu-test: `gemm_f16n` rows bitwise equal to `gemm_f16` (Q4_0, Q4_1, Q5_K; 1, 5, 16, 17 and
    40 rows). Every split part equals the unsplit kernel over its K range (4 cases, including
    the model's 17408 / 1792 split). 37/37 overall, 0 spill failures.
  - `tests/test_model.py` knows the new modules.
- `zerv-batch-check … f32 128 2048 f16` takes the decode precision as its fifth argument.

## Measurements

`zerv-batch-check` timing section: f32 KV, 128-token pages, all slots at position ~700–740,
median of 5 wall-clock `decodeBatch` calls; a model-level timing
([data](data/2026-09-25-f16-decode-mode/)). The FP32 row is from the
[18b.2 report](2026-09-25-batched-decode.md).

| Rows | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| FP32 decode (ms) | 20.06 | 21.17 | 25.23 | 46.29 |
| f16 v1 without split-K (ms) | 64.3 | 65.1 | 66.9 | 71.2 |
| **f16 v1 with split-K** (ms) | **45.4** | 45.8 | 46.8 | **50.0** |

Without split-K, a 5120-wide output has only 40 workgroups for 96 CUs. The split brings
workgroups to about 400 and saves 19 ms.

## Why v1 is slow (analysis, not yet measured per kernel)

- The step's f16 projections take about 43 ms against a DRAM floor near 16 ms for their bytes.
  That is roughly 35% of the bandwidth.
- The prefill tile design is wrong for 16 rows:
  - Each subgroup owns distinct weight rows. Dequantizing them into LDS behind two workgroup
    barriers per 32-k step buys no reuse.
  - Each 32-k step moves only ~2.3 KB of weights per workgroup, so barrier and LDS round
    trips dominate.

## Next (kernel v2, same arithmetic)

- Each subgroup streams its own weight rows. They are dequantized to f16 in a subgroup-private
  LDS slice with no workgroup barrier, several 16-k slices per step.
- B comes straight from an f16 copy of X in global memory, as `gemm_f16x` does. The f16 copy
  rounds exactly as the in-kernel conversion, so the results stay bitwise equal.
- The gates are unchanged: `gemm_f16n` bitwise equal to `gemm_f16`, and `zerv-batch-check`
  in f16 mode.
- Then the FP64 quality gate, and serving at 1/2/4/8/16 clients against the FP32 mode and
  llama-server.
