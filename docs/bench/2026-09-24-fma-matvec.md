# FMA accumulation in the decode and verify matvec (block 17c, 2026-09-24)

Question: the multi-row verify matvec is ALU-bound at 3–5 rows
([spec-verify](2026-09-24-spec-verify.md)); a fused multiply-add per step (one rounding
instead of two) was measured faster in the lab but rejected in 08c for changing output
bits. With the FP64 gates as the criterion, is it correct, and is it faster?
[Spec](../specs/matvec-push.md) ("FMA accumulation").

## Change

`accum()` in `src/matvec/matvec.comp` and `matvec_rows.comp`:
`precise vec4 added=fma(w,x,sum)` instead of `precise` multiply then add. Decode of
weights, lane partition, block order, the two partial vectors and the reduction tree are
unchanged. Then the per-count verify table re-tuned (below).

## Correctness

| Gate | Result ([data](data/2026-09-24-fma-matvec/)) |
| --- | --- |
| `zig build gpu-test` | 29/29: all 48 matvec fixtures in both layouts (exact cases unchanged: products and sums there are representable), multi-row = single-row bitwise on all fixtures and real shapes |
| `verify_model`, default oracle, modes 0/1/13/512/512:17 | passed, 0 failures; decode-mode logits worst normalized L2 2.27e-6 → 2.01e-6 (short), 4.08e-6 → 1.56e-6 (long-think) |
| `verify_model`, long oracle, modes 0/13/128/512/512:300 | passed; mode 0 logits 2.05e-5 → 1.56e-5, worst/bound 0.639 → 0.448; prefill modes as before |
| `zerv-spec-check` (f32 and f16 KV) | 11/11 each, with the final table |
| `zerv-mtp-check` scenario C + FP64 MTP reference | 3/3 and passed on both sequences |
| `zerv-prefix-check` | gate passed |

One rounding per step makes decode more accurate against FP64 on every oracle case.

## Verify table re-tune

`python3 tools/tune_matvec_rows.py --output docs/bench/data/2026-09-24-fma-matvec/rows-tune`
(new tool: compiles each (GROUP, CB) variant of `matvec_rows.comp`, runs
`zerv-matvec-rows-bench` over 8 roles, 2 passes in alternating order). Two findings:

- **G = 4 at 5 rows is not bitwise equal to the single-row module for Q5_K (ssm_out)
  and Q6_K (output)**. It was never shipped; the tool now marks such variants invalid
  instead of stopping. Cause not investigated at the time; found later the same day: an
  ACO miscompile of VGPR spills in LDS ([RCA](2026-09-24-aco-lds-spill.md)).
- The sums from this tool drifted against the 17b.1 sweep by 30–56% on the mid-size
  roles, for the single-row passes too (e.g. `attn_q` single 1,053 → 1,640 µs at
  count 2), while the largest roles did not move. Separate processes are not
  comparable at that level (D6), so the decision was made by an interleaved in-model
  race instead.

Race: three `zerv-spec-check` binaries run A B C A B C A B C
([data](data/2026-09-24-fma-matvec/ab-race/), binary hashes in `binaries.sha256`):
A = multiply-add, old table {2: (2,2), 3: (2,2), 4: (3,2), 5: (2,4)}; B = FMA, old
table; C = FMA, the sweep's best {2: (3,2), 3: (3,3), 4: (3,2), 5: (3,3)}. Median of 3
(each is itself a median of 41 calls), ms:

| | A mul+add, old | B FMA, old | **C FMA, new** |
| --- | --- | --- | --- |
| decode step | 20.16 | 20.13 | 20.14 |
| verify+commit 1 row | 20.70 | 20.73 | 20.71 |
| 2 rows | **21.51** | 21.64 | 21.58 |
| 3 rows | 24.24 | 26.03 | **22.68** |
| 4 rows | 27.30 | 24.95 | **25.03** |
| 5 rows | 32.60 | 30.59 | **28.77** |

Spread within each cell ≤ 0.2 ms except one 20.92 outlier (B, 1 row).

## Decision

- **Adopted: C.** 3 rows (the default: 3 drafts, adaptive) −6.4%, 4 rows −8.3%,
  5 rows −11.7% per verify; 2 rows +0.3%; single-row decode unchanged. `ROWS_CONFIG` in
  `tools/compile_matvec.py` and `matvec.rows_groups` = {1, 3, 3, 3, 3}.

## Knob (added the same day)

`--matvec-accumulation fma|separate` (default fma) keeps the previous arithmetic
selectable: `separate` compiles the modules with `ACCUM_FMA 0` and its old verify table.
Gates: gpu-test 30/30 (both module sets on every fixture, multi-row = single-row for
both), `verify_model --matvec-accumulation separate` byte-identical to the pre-FMA
captures (`2026-09-24-kvtype-f32-*`, 28 + 14 files), spec-check 11/11 in both modes
(step 19.67 / 19.79 ms, 3-row verify 22.53 / 24.11 ms).

Serving check through the flag (decode-v1, 3 drafts, 1 repeat, same server build;
`python3 bench/run_serving.py --workload bench/workloads/decode-v1.json --output
docs/bench/data/2026-09-24-fma-matvec/decode-v1-knob --engines
"zerv-spec3;zerv-spec3@matvec-accumulation=separate" --repeats 1`):

| decode tok/s | code | json | think | prose |
| --- | ---: | ---: | ---: | ---: |
| fma (default) | 121.0 | 125.4 | 107.7 | 75.6 |
| separate | 111.7 | 115.4 | 100.6 | 71.3 |

Greedy outputs are identical between the two modes on all four cases (`output_sha256`).
With one repeat this confirms that the flag works; it does not measure variance. The
speedup numbers remain the ones in the serving table below.

## Serving (decode-v1, greedy, 512 tokens, 2 repeats; [data](data/2026-09-24-fma-matvec/decode-v1/))

`bench/run_serving.py --workload bench/workloads/decode-v1.json --engines
zerv,zerv-spec2,zerv-spec3,llama-fa-ub512-mtp4 --repeats 2`. The binary also contains the
decode attention changes of [decode-attention-long](2026-09-24-decode-attention-long.md)
(neutral at these short contexts). Decode tok/s median [min–max]; acceptance =
accepted / verified drafts:

| Engine | code | json | think | prose |
| --- | --- | --- | --- | --- |
| zerv (no speculation) | 49.8 | 49.7 | 49.5 | 49.8 |
| zerv, 2 drafts | 108.2 (0.92) | 110.3 (0.95) | 102.0 (0.84) | **78.6** (0.53) |
| **zerv, 3 drafts (default)** | **120.5** (0.87) | **124.8** (0.92) | **107.7** (0.76) | 75.6 (0.45) |
| llama-server MTP 4 (its best) | 105.0 (0.82) | 109.5 (0.88) | 90.5 (0.67) | 57.5 (0.33) |
| zerv 3 drafts before (decode-v1-adaptive) | 110.0 | 114.0 | 99.1 | 71.2 |

- Every zerv output equals the plain run's and the pre-FMA build's (greedy decode-v1
  outputs are unchanged by FMA on these prompts).
- 3 drafts are 9–10% faster than before on code, json and think; zerv is 1.15 / 1.14 /
  1.19 / 1.37× llama's best (prose with 2 drafts).
