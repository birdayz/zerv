# Decode phase fusion: gate + up + swiglu in one dispatch (block 17c, 2026-09-24)

Question: how much of the decode step is spent waiting between dependent phases, and
can phases be merged without changing any value? Knob: `--decode-fusion on|off`
(`Options.decode_fusion`, default on). [Spec](../specs/model.md) ("Decode FFN fusion").

## Dependency cost (diagnostic)

A diagnostic build with every compute barrier removed (results invalid, timing only;
`zerv-spec-check` step timing, 2 runs each, interleaved):

| Build | decode step ms |
| --- | --- |
| with barriers | 20.04 / 20.00 |
| no barriers (diagnostic) | 18.40 / 18.32 |

About **1.7 ms (8%)** of every step is dependency cost. The weights alone need 16.7 ms
at the card's measured 920 GB/s. A DeltaNet layer has 8 dependent phases, an attention
layer about 12.

## Change

- `matvec.comp`: the per-row accumulation is a function (`lanePartial`), shared by the
  single-row module and a new `SWIGLU` variant. In the variant, workgroup i computes row
  i of gate and of up (two regions of one weight buffer) with exactly the single-row
  arithmetic and reduction tree, then lane 0 writes g, u and `silu(g) * u` (the swiglu
  kernel's expression).
- `matvec.SwigluPipeline`; the runtime fuses a layer when its gate and up tensors share
  a bank, format and shape: **63 of 64 layers** here (one layer's pair straddles a bank
  boundary and keeps the separate path).
- Removes one dispatch and one barrier per fused layer from the decode step. The verify
  pass (speculation) keeps its multi-row projections and the swiglu kernel.

## Correctness

- `zig build gpu-test` 29/29: the refactored single-row modules (their SPIR-V changed)
  pass all 48 matvec fixtures in both layouts, including the hashed exact outputs.
- `tools/verify_model.py`, default oracle modes 0/1/13/512/512:17 and long oracle modes
  0/13/128/512/512:300: all 28 + 14 capture and logits files **byte-identical** to the
  previous build (`third_party/model-native/2026-09-24-pv2w-*`).
- `zerv-spec-check` 11/11 with f32 KV, f16 KV and `separate` (fusion off).

## Speed

Interleaved race of spec-check binaries (3 runs each; [data](data/2026-09-24-decode-fusion/race/)):

| | before | fused |
| --- | --- | --- |
| decode step ms | 20.015 (20.001–20.019) | **19.667** (19.651–19.675) |
| verify+commit 3 rows | 22.527 | 22.554 |
| verify+commit 5 rows | 28.586 | 28.638 |

Serving decode-v1 (greedy, 512 tokens, 2 repeats; [data](data/2026-09-24-decode-fusion/decode-v1/)),
decode tok/s median:

| Engine | code | json | think | prose |
| --- | --- | --- | --- | --- |
| zerv, `--decode-fusion on` | **50.73** | **50.73** | **50.71** | **50.71** |
| zerv, `--decode-fusion off` | 49.86 | 49.90 | 49.89 | 49.88 |
| zerv 3 drafts, on | 120.50 | 124.81 | 107.56 | 75.53 |
| zerv 3 drafts, off | 120.50 | 124.98 | 107.69 | 75.62 |

All outputs identical across the four engines. Plain decode +1.7%; speculative decode
unchanged (its steps are verify passes).

## Next

The remaining ~1.35 ms of dependency cost: which other phases can merge (conv into the
lin_in epilogue, the attention passes, norms), and the same fusion for the verify pass.
