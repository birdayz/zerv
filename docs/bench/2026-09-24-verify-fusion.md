# Verify FFN fusion: gate + up + swiglu in one multi-row dispatch (block 17c, 2026-09-24)

Question: does the decode FFN fusion ([report](2026-09-24-decode-fusion.md)) also pay in
the speculative verify pass, where the multi-row kernel is issue-bound? Knob
`--verify-fusion on|off` (`Options.verify_fusion`, default on). [Spec](../specs/model.md)
("Verify FFN fusion").

## Change

- `matvec_rows.comp` with `SWIGLU`: a workgroup takes GROUP gate rows and the same GROUP
  up rows (WR = 2 GROUP weight rows, each X load shared by all of them), with the
  multi-row per-row arithmetic unchanged, then writes g, u and silu(g)·u for every input
  row. The existing 118 modules compile to identical bytes (`WR` expands to `GROUP`
  without `SWIGLU`).
- `matvec.SwigluRowsPipeline`; the table `matvec.swigluRowsGroups` /
  `SWIGLU_ROWS_CONFIG` (0 / None: no fused module for that count, the verify pass
  records the separate path). 48 new modules (counts 1–4 × 6 formats × 2 accumulations).
- The verify pass fuses the same 63 of 64 layers as decode.

## Correctness

- `gpu-test` 31/31. The new test covers every quantized fixture, both accumulations,
  unaligned and aligned offsets, and counts 1..5: g, u and y bitwise equal to the
  single-row fused module per input row, rows past the count untouched, and `record`
  rejecting a count without a module.
- **The test caught a wrong configuration:** GROUP 2 at 5 rows (4 weight rows × 5 input
  rows) gives wrong values on the Q5_K path (e.g. `q5_k-seeded-256`, fma, count 5, row
  4: u = 0xc06de584 against 0xc061c584). The error is in the value, not in rounding. The
  unfused kernel had already shown the same failure for G = 4 at 5 rows on Q5_K/Q6_K
  ([FMA report](2026-09-24-fma-matvec.md)). The in-model race could not see it: this
  model's gate/up are Q4_0. **Root cause:** Mesa 26.2.3's ACO spills the module's VGPRs
  into LDS without memory-ordering information, and its scheduler reorders two accesses to
  a reused slot ([RCA](2026-09-24-aco-lds-spill.md)). Count 5 has no fused module, and
  the K-quant fused modules stop at count 2: their counts 3–4 also spilled into LDS,
  although they passed. No shipped shader spills into LDS (`tools/check_shader_spills.py`).
- `zerv-spec-check` 11/11 (verify ≡ decode bitwise, committed state, follow-up steps)
  with verify fusion on and off, with `separate` accumulation, and with f16 KV
  ([data](data/2026-09-24-verify-fusion/)).

## Tuning (in-model race)

`tools/race_swiglu_rows.py --output docs/bench/data/2026-09-24-verify-fusion/race{1,2}`:
one spec-check binary per GROUP:CB table applied to counts 2..5, run interleaved (3
passes, alternating direction); race 2 includes `off`, the same binary with verify
fusion off. Mean verify+commit ms (spread ≤ ±0.13):

| variant | 1 row | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| off (race 2) | 20.637 | 21.550 | 22.639 | 24.754 | 28.616 |
| 1:2 (race 1) | 20.275 | **21.076** | 23.599 | 25.946 | 29.626 |
| 1:3 | 20.272 | 21.549 | 23.192 | 25.551 | 32.126 |
| 1:4 (race 2) | 20.309 | 21.428 | 23.063 | 27.316 | 29.526 |
| 2:1 (race 2) | 20.287 | 21.354 | 23.646 | 24.926 | 28.710 |
| 2:2 (race 1 / 2) | 20.280 / 20.369 | 21.454 / 21.523 | 22.466 / 22.457 | **24.483** / 24.479 | 29.124 / 29.135 |
| 2:3 | 20.273 | 21.082 | **22.359** | 25.382 | 36.762 |
| 2:4 (race 2) | 20.301 | 21.133 | 23.003 | 30.047 | 50.524 |
| 3:2 | 20.270 | 21.107 | 26.860 | 49.763 | 183.415 |

Count 1 is 1:4 in every variant. The shipped table is {1: 1:4, 2: 1:2, 3: 2:3, 4: 2:2,
5: none}; 2:2 ran in both races within 0.1 ms. The `separate` accumulation modules use
the same table, correctness-tested but not speed-tuned.

Final build (spec-check, one run each, ms):

| | 1 row | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| verify fusion on | 20.27 | 21.07 | 22.29 | 24.37 | 28.51 (separate path) |
| off | 20.58 | 21.44 | 22.54 | 24.74 | 28.59 |

## Serving

decode-v1, 3 drafts, 2 repeats (`python3 bench/run_serving.py --workload
bench/workloads/decode-v1.json --output docs/bench/data/2026-09-24-verify-fusion/decode-v1
--engines "zerv-spec3;zerv-spec3@verify-fusion=off" --repeats 2`), decode tok/s median:

| engine | code | json | think | prose |
| --- | ---: | ---: | ---: | ---: |
| verify fusion on (default) | **121.0** | **126.3** | **107.7** | **76.1** |
| off | 120.5 | 124.8 | 107.2 | 75.4 |

Outputs identical. +0.4 to +1.2%: verify + commit is 1.1–1.8% cheaper, and a cycle
also spends 4.7 ms drafting.
