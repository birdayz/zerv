# Speculative verification: N-row verify ≡ decode, and its cost (block 17b.1, 2026-09-24)

Question: can an N-row verify pass (N = drafts + 1 ≤ 5) produce logits **bit for bit**
equal to N decode steps, and how cheap can it be made without giving that up?

Spec: [speculative.md](../specs/speculative.md). Research:
[speculative-mtp.md](../research/speculative-mtp.md).

## Setup

- Machine: RX 7900 XTX (Sapphire NITRO+), Mesa 26.2.3 RADV, see [hardware](../hardware.md).
  No other GPU process during any timed run (checked with `pgrep` before each run).
- Model: `models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf` (sha256 `ede16c7b…`).
- Tree: commit `3c03b07` plus the uncommitted 17b.1 work; `src/matvec/matvec_rows.comp`
  sha256 `8afcf3c0…` for the final numbers.
- Tools:
  - `zerv-spec-check` (`bench/spec_check.zig`, `zig build spec-check-build
    -Doptimize=ReleaseFast`): the real-model gate plus wall-clock timing;
  - `zerv-matvec-rows-bench` (`bench/matvec_rows.zig`, `zig build
    matvec-rows-bench-build -Doptimize=ReleaseFast`): one projection role over all
    layers in one command (weights stream from DRAM as in a decode step; inputs and
    outputs device-local), multi-row against R single-row dispatches, every row checked
    bitwise, GPU timestamps, median of 9 after 300 ms warmup.

## Gate 1 (correctness): passed

`zerv-spec-check MODEL` checks 11 cases (prompt length 40–700, verify rows n = 1..5,
commit m ≤ n):

- all n·vocabulary verify logits equal the decode logits bitwise;
- every byte of the recurrent and conv state after commit(m) equals the state after m
  decode steps;
- 4 follow-up decode steps equal.

All 11 cases pass in every run: [r1](data/2026-09-24-spec-verify/spec-check-r1.jsonl)–r3
(first kernel), [r4](data/2026-09-24-spec-verify/spec-check-r4-g2.jsonl) (2 weight rows
per workgroup), [r5](data/2026-09-24-spec-verify/spec-check-r5-exact.jsonl) (exact-count
modules), [r6](data/2026-09-24-spec-verify/spec-check-r6-tuned.jsonl) (final, per-count
tuning). Decode itself is bitwise unchanged against 2026-09-23 on the three oracle cases
(verify_model mode 0; [default](data/2026-09-24-spec-verify/decode-default-mode0.json),
[long](data/2026-09-24-spec-verify/decode-long-mode0.json)).

`zig build gpu-test` (Debug and ReleaseFast, 23/23) includes the module-level gate: each
multi-row module equals the single-row module row by row, bitwise, on all 48 matvec
fixtures × 2 weight alignments × counts 1..5, and rows past the count are not written.

## Cost: verify + commit against one decode step

Wall clock from `zerv-spec-check` at position ≈ 300 (ms, median; min–max in the raw files):

| Kernel version | step | n=1 | n=2 | n=3 | n=4 | n=5 |
| --- | --- | --- | --- | --- | --- | --- |
| runtime count, 1 weight row / workgroup (r3) | 20.04 | 24.08 | 28.31 | 34.11 | 40.41 | 46.29 |
| runtime count, 2 weight rows / workgroup (r4) | 20.01 | 23.72 | 27.66 | 32.96 | 38.68 | 43.86 |
| **exact-count modules, per-count tuning (r6)** | 20.02 | **20.58** | **21.40** | **24.10** | **27.07** | **32.44** |

The final verify of 5 rows costs 1.62 decode steps; 3 rows cost 1.20.

## What made the difference (component bench, all layers of a role, µs)

The kernel is ALU- and issue-bound at 5 rows, not bandwidth-bound: per weight it does
the single-row arithmetic (separate `precise` multiply and add into two accumulators)
once per row. What was tried, in order ([raw](data/2026-09-24-spec-verify/matvec-rows/)):

| Variant (ffn_gate, Q4_0, 64 layers, 5 rows) | µs | vs single-row decode pass (3480) | bitwise |
| --- | --- | --- | --- |
| runtime count, G = 1 weight row / workgroup, 4-block chunks | 10268 | 2.95× | yes |
| runtime count, G = 2 | 8470 | 2.43× | yes |
| G = 3 / 4 (register spills at G = 4, CB = 4: 97116) | 8588 / 13114 | worse | yes |
| fma instead of multiply + add (changes decode numerics) | 7970 | 2.29× | no |
| raw-word prefetch of the next chunk (compiler waits on it anyway) | 7415 | 2.13× | no |
| wave32 required (VOPD forms, 464 dual ops) | 6183 | slower than wave64 | yes |
| **exact ROWS, no per-row branches, no per-weight-row guards, hoisted X index** | **5914** | **1.70×** | **yes** |
| same + fma + G = 4 (numerics change) | 4778 | 1.37× | no |

- The same-X diagnostic (`--same-x 1`: all rows read row 0, so X has one row's cache
  footprint) changed nothing (8932 → 8763): X cache capacity was not the limit.
- `RADV_DEBUG=shaderstats` showed the runtime-count kernel at 192 VGPRs, 8 waves per SIMD,
  2008 VALU instructions. The `if (r < count)` branches made ACO reload the X buffer
  descriptor per row and turn every `g < gn` guard into a `v_cndmask` per accumulator.
  The exact-count module has no such code.
- Default compute is wave64 on this driver. Forcing wave32 lets ACO form VOPD pairs but
  was slower (6183 against 5853 µs).
- The fma variants would need decode switched to fma too (to keep verify ≡ decode) and
  a new FP64 re-gate. They are recorded as the next lever, not adopted.

Per-count tuning (all formats, sum over the 8 benched roles, µs;
[raw](data/2026-09-24-spec-verify/matvec-rows/)):

| Count | G, CB tried (sum) | chosen |
| --- | --- | --- |
| 1 | g1c4 8570, g2c4 8654, g2c2 8743 (single-row passes: 8566) | G 1, CB 4 |
| 2 | g2c4 9512, g2c2 8852, g3c2 8850, g4c2 8817 | G 2, CB 2 |
| 3 | g2c4 10463, g2c2 10124, g1c4 12995, g3c2 10790, g2c3 10617 | G 2, CB 2 |
| 4 | g2c4 14791, g3c2 11852, g3c3 13098, g4c2 17453, g1c4 16427 | G 3, CB 2 |
| 5 | g2c4 14168, g2c3 14136, g2c2 14495, g3c2 15509, g1c4 22029 | G 2, CB 4 |

- Each bench run alternates single- and multi-row samples, but the variants ran in
  separate processes, not in an interleaved race (D6: separate runs drift ±3–4% with the
  thermal state). Choices whose sums differ by less than 4% (counts 1, 2 and 5) are
  within that noise. Their runner-ups would do as well.
- At count 4, G = 2 and G = 4 are pathological for the K-quant formats: Q5_K ssm_out
  runs 2504 and 4762 µs, against 1697 at G = 3; Q6_K output runs 2962 and 3832, against
  1240. Not understood; recorded, and avoided by the table.
- The table lives in `matvec.rows_groups` and `tools/compile_matvec.py`
  (`ROWS_CONFIG`); the Python manifest test checks it. A mismatch fails the GPU equality
  test, because rows go missing.

## Interpretation and cost

- The shipped design is **45 modules** (9 formats × counts 1..5, 1.2 MB of SPIR-V) and
  one kernel per count per rows pipeline (`RowsPipeline.init(..., max_count, ...)`).
- It shows that lossless verification can cost 1.2–1.6 decode steps for 3–5 rows. It
  says nothing yet about serving tok/s: drafting (the MTP layer) and the engine loop do
  not exist yet (17b.2, 17b.3).
- Estimate, not a result: with draft cost ≈ 1.6 ms per token and llama's measured
  acceptance (code 0.97), 4 drafts would give (32.4 + 6.4) ms per 4.7 tokens ≈ 120 tok/s,
  against llama MTP-4's measured 105.
- Open levers:
  - fma arithmetic in decode and verify, with an FP64 re-gate (1.37× instead of 1.70× at
    5 rows in the lab);
  - the count-4 K-quant anomaly;
  - the attention and delta passes at n rows (not yet profiled separately).
