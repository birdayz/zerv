# Exact FP32 batched projection beyond 4 rows (block 18e, part 1) — 2026-09-25, negative result

**Question.** The batched decode step at 8 rows costs 46 ms (2.3× one row) because the FP32
multi-row projection (`matvec_rows.comp`) stops scaling after ~4 rows. Can a restructured
kernel, bitwise equal per row to the single-row module, make 8 rows much cheaper?

**Answer: not with the two restructurings tried.** Both stay bitwise exact, but neither is a
material win. The best 8-row time for ffn_gate over 64 layers is 8.48 ms, against 8.95 ms for
today's module, which is 2.4× the one-row DRAM floor of 3.48 ms. The limiter is not register
capacity, X cache footprint or occupancy; it is still unidentified (below). Nothing was shipped:
the shipped source and all 154 matvec modules are byte-identical to before.

## Background: the exactness constraint

Each output of the single-row Q4_0 module is a fixed reduction over 512 independent FMA
chains. Chain p (lane l, vector `a`/`z`, component c) accumulates columns p, p + 512, p + 1024, …
in order. Then come `a + z`, `(v.x + v.y) + (v.z + v.w)`, and a 64-lane tree. Any kernel that
keeps every chain's order and the tree is bitwise equal. Which thread runs which chain, and
where X comes from, is free.

## Method

`zerv-matvec-rows-bench` on the real weights: every layer's copy of the role is resident and
streams from DRAM, and every row is checked bitwise (0 mismatches in every configuration
below). Modules are built from the variant source
[matvec_rows_split_rowgroups.comp](data/2026-09-25-fp32-batched-projection/matvec_rows_split_rowgroups.comp)
(sha256 `0a8029d8…`) with pinned glslc; per-module VGPRs come from `RADV_DEBUG=shaderstats`.
Raw lines: [sweep-r4.jsonl](data/2026-09-25-fp32-batched-projection/sweep-r4.jsonl). Times are
the median of 11 samples for ffn_gate (5120 → 17408, Q4_0, 64 layers).

## Results

| Variant (8 rows) | Idea | ffn_gate ms | VGPRs / waves per SIMD |
| --- | --- | --- | --- |
| today (GROUP 2, CB 2) | — | 8.95 | 256 / 4 |
| `SPLIT 4`, GROUP 8 | 4 threads share a lane's vec4 components: 4× fewer accumulators per thread, 4× more weight rows per X load | 10.95 | 192 / 8 |
| `SPLIT 2`, GROUP 8 / `SPLIT 4`, GROUP 16 | same, larger | 166 / 159 | spills 320 VGPRs |
| `ROWGROUPS 2` × 4 rows, GROUP 3 | one dispatch, both 4-row groups of a weight-row group adjacent (second weight read from cache) | **8.48** | 192 / 8 |
| `ROWGROUPS 4` × 2 rows, GROUP 6/8 | more X reuse, 4 weight reads | 11.8 / 16.1 | — |
| today with `--same-x` (X footprint of one row) | diagnostic: is X cache footprint the limit? | 8.54 | 256 / 4 |
| wave32 against wave64 (both variants) | dual issue needs wave32 | ±7%, no consistent gain | — |

The scaling of today's modules (shipped GROUP/CB per count): 1 row 3.48 ms (DRAM-bound,
920 GB/s), 2 rows 3.62, 4 rows 4.40, 8 rows 8.95.

At 4 rows, occupancy against X reuse:

| GROUP, CB | 1, 1 | 1, 2 | 1, 4 | 2, 1 | 2, 2 | 3, 1 | 3, 2 | 4, 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms | 8.18 | 6.78 | 7.13 | 4.90 | 5.31 | 5.15 | **4.46** | 4.82 |
| VGPRs / waves | 72 / 20 | 84 / 18 | 96 / 16 | 108 / 14 | 120 / 12 | 144 / 10 | 192 / 8 | 192 / 8 |

## Analysis

- From 4 rows on, the kernel is compute-side bound at ~5.2 TFMA/s. That rate is the same at
  4 and 8 rows (22.8e9 FMA in 4.40 ms, 45.6e9 in 8.95 ms).
- **ISA of the best 4-row loop** (per iteration): 246 FMAs, 164 other VALU, 16 `b128` X loads
  (16 KB), 12 weight loads and 58 SALU/control instructions.
  - At the measured rate a wave-iteration takes about 1150 clocks, against roughly 570 VALU
    clocks and about 280 clocks of vector-memory data path.
  - Neither is saturated: the wave waits about half the time.
- **Occupancy is not the cure:** GROUP 1 runs 20 waves per SIMD and is the slowest.
- **X cache footprint is not the cure:** `--same-x` gains 5%.
- **SPLIT:** reusing X across more weight rows via thread splitting costs more VALU than it
  saves. Its loop has 600 VALU instructions per 256 FMAs (per-row addresses, unaligned word
  selects, scalar decode), against 420.
- **Not identified:** the stall source. Load-to-use latency within an iteration (the compiler
  does not software-pipeline across iterations) is the leading hypothesis. Confirming it needs
  a hardware profile (thread trace), which is not available on this machine.

## Candidates recorded, not pursued now

- `ROWGROUPS 2 × 4` for batches of 5–8 rows: +5% on the projection, exact. It needs a module
  set per format and runtime selection.
- A software-pipelined loop (next iteration's loads issued before this iteration's FMAs), for
  example hand-scheduled as done for `gemm_f16x`.
- X staged in LDS and shared by several waves (TA traffic ÷ waves; LDS bandwidth then limits).

## Decision

Throughput beyond 4 rows per weight pass comes first from the opt-in WMMA `--decode-precision
f16` path (user decision 2026-09-24: both arithmetics as knobs). The exact FP32 path stays the
default at today's cost curve.
