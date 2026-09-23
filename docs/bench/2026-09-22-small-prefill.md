# Small-row prefill plans (block 13d) — 2026-09-22

**Question:** can short prompts avoid paying for a full 128-row GEMM tile, without
changing the arithmetic?

**Result:** yes.
- **Short-prompt TTFT:**

  | Prompt | zerv before | zerv now | llama-server FA | llama-server FP32 control |
  |---|---:|---:|---:|---:|
  | 23 tokens | 450 ms | **168 ms** | 194–207 ms | 324–335 ms |
  | 81 tokens | 487 ms | **390 ms** | 391–406 ms | — |

  zerv's TTFT at 23 tokens is now below llama-server at default precision in these runs.
  llama measured 162 ms in a quieter environment earlier, so call this parity or better,
  not a clear win.
- **Unchanged:** medium and long prompts (as predicted), decode rates, and greedy
  outputs.
- **Accuracy:** all oracle gates pass. Each tile is bit-identical to 128×128 at equal
  split-K.

## Change

`gemm.comp` is parameterized by thread layout: 256 threads as TX×(256/TX), each owning
TM×TN outputs. BK stays 32 with the same per-element two-level accumulation.

**Compiled tiles:**
- 128×128 (16×16 threads, 8×8 outputs)
- 128×32 (32×8, 4×4)
- 256×16 (64×4, 4×4)

128×64 was measured and dropped. Existing 128×128 SPIR-V is byte-identical to before.

**Runtime plans:** the runtime records one prefill command per plan. Each plan's grids,
split-K choices and partial slots are sized to its row count.

| Chunk rows | Tile |
|---|---|
| ≤16 | 256×16 |
| ≤96 | 128×32 |
| up to `--prefill-chunk` | 128×128 |

**Chunk policy (`chunkFor`):**
- Full chunks run on the last plan.
- A shorter remainder whose tail past a multiple of 128 fits a smaller plan is split:
  the 128-multiple runs on 128×128, then the tail on the smaller plan.
- `runChunk(commands, plan, chunk)` rejects chunks larger than the plan.

**Supporting changes:**
- The GPU layer's device-wide kernel bound was raised from 64 to 128
  ([driver spec](../specs/gpu-driver.md)), since three plans need ~73 pipelines.
- `model_capture` records one capture command per plan and follows `chunkFor`.

## Tile crossover measurement (decides the plan table)

> Caveat: this was measured with uninitialized (likely zero) X in short bursts, so
> absolute times are optimistic (see the [13e report](2026-09-22-gemm-throughput.md)).
> The tiles and plans chosen here were superseded in 13e by the scalar-X kernel.

The table is a weighted sum over the benchmarked shapes, each weighted by its count in
the model, in GPU ms per chunk (run1 / repeat). Raw data:
[run1](data/2026-09-22-small-prefill/gemm-tiles-run1.jsonl),
[repeat](data/2026-09-22-small-prefill/gemm-tiles-repeat.jsonl). This is an estimate of
total GEMM time, not a model run.

| Rows | 128×128 | 128×64 | 128×32 | 256×16 |
|---:|---:|---:|---:|---:|
| 1 | 415 / 448 | 286 / 288 | 124 / 125 | **85 / 86** |
| 8 | 409 / 434 | 289 / 293 | 126 / 129 | **89 / 91** |
| 16 | 415 / 427 | 291 / 292 | 127 / 130 | **96 / 98** |
| 23 | 417 / 424 | 292 / 294 | **128 / 130** | 169 / 170 |
| 32 | 419 / 423 | 294 / 296 | **130 / 130** | 179 / 182 |
| 48 | 418 / 422 | 298 / 303 | **232 / 239** | 259 / 262 |
| 64 | 422 / 422 | 313 / 317 | **237 / 241** | 329 / 334 |
| 81 | 425 / 425 | 544 / 559 | **350 / 355** | 485 / 491 |
| 96 | 430 / 435 | 546 / 562 | **354 / 358** | 499 / 503 |
| 128 | **442 / 445** | 569 / 576 | 446 / 457 | 631 / 636 |
| 151 | 769 / 776 | 908 / 867 | **598 / 607** | 836 / 838 |
| 192 | 775 / 785 | 927 / 974 | **659 / 668** | 938 / 948 |
| 256 | **819 / 828** | 1094 / 1119 | 882 / 897 | 1267 / 1278 |
| 324 | **1201 / 1214** | 1602 / 1656 | 1205 / 1228 | 1676 / 1681 |
| 512 | **1506 / 1538** | 2171 / 2200 | 1706 / 1741 | 2511 / 2516 |

## Correctness

| Gate | Evidence | Result |
|---|---|---|
| Every tile vs the independent matvec fixtures (scaled rows, zero rows, tails, split-K), and bit identity to 128×128 at equal split | `tests/model_gpu.zig`, Debug + ReleaseFast | pass |
| Every tile vs FP64, attention-style F32 strided/batched products | same | pass |
| Plan table and chunk policy (every prompt length 1–1999 covered, chunks within plan) | `tests/model.zig` | pass |
| Model oracle, all modes 0/1/13/29/60/512 on the final code, one manifest (`third_party/model-native/2026-09-22-small-prefill-gate2`) | [gate-run2](data/2026-09-22-small-prefill/gate-run2.json) | worst/bound ≤0.595 in modes 0–60; 0.836 short / 0.216 long in mode 512; logits ≤3.23e-6; greedy all |
| Earlier partial runs (same numbers): modes 0–60 before the kernel-bound change, and mode 512 alone | `gate-run1-mode*.json`, [gate-run1-512](data/2026-09-22-small-prefill/gate-run1-512.json) | identical results |
| Serving greedy equality | [session-run1](data/2026-09-22-small-prefill/session-run1.json) | JSON + SSE, both cases |

**Failed attempt:** the first mode-512 run failed before comparison with `ResourceLimit`
([stderr](data/2026-09-22-small-prefill/failed-mode512-resource-limit.txt)), because the
device-wide kernel bound of 64 was too small for three plans. It was rerun in a fresh
work dir after the bound was raised; that run has a complete manifest.

**Mode-512 short case at 0.836 of bound.** The 48 tokens run on 128×32 in the 96-row
plan. The worst sample is `attn_output` at layer 60, token 18: 7.7e-6, bound 9.2e-6.
The distribution barely moved relative to 13a
([errdist](data/2026-09-22-small-prefill/errdist-short-512.json)):
`attn_output` mean 5.28e-7 → 5.41e-7 and p90 9.27e-7 → 9.53e-7.

The same comparison exposed a **quality lever**. The same tile in the 60-row plan
splits K more and is clearly more accurate: `attn_output` mean 4.56e-7, `l_out`
3.43e-7 vs 4.41e-7. Long sequential K accumulation dominates GEMM error. This is queued
as 13f (a bounded chunk length or a third accumulation level, measured as an explicit
tradeoff).

## Per-phase profile

`zerv-model-profile MODEL 8192 512 N 16`; data in
[`data/2026-09-22-small-prefill/`](data/2026-09-22-small-prefill/).

| Prompt | Chunks (rows → plan) | GPU ms before (13a/13b) | GPU ms now |
|---|---|---:|---:|
| 23 | 23 → 128×32 | 431 | **160.5** |
| 81 | 81 → 128×32 | ≈476 (est.) | **372.5** |
| 151 | 128 → 128×128, then 23 → 128×32 | 830 (one 151-row chunk) | **661** |
| 836 | 512, 256 → 128×128; 68 → 128×32 | ≈2.85 s | 2.87 s |
| 3223 tail | 128 + 23 instead of 151 | 830 | 696 |

## Serving (median of 3; run1 / repeat)

The repeat reused the run1 binary (`fe36860c…`).
[run1](data/2026-09-22-small-prefill-serving-run1/summary.json),
[repeat](data/2026-09-22-small-prefill-serving-repeat/summary.json):

| Case (prompt tok) | Engine | TTFT ms | Decode tok/s | Total s |
|---|---|---:|---:|---:|
| short-nothink (23) | zerv (small-row plans) | 168 / 167 | 50.4 / 50.6 | 0.43 / 0.43 |
| short-nothink (23) | llama-server FA ub512 | 207 / 194 | 43.6 / 44.0 | 0.51 / 0.49 |
| short-nothink (23) | llama-server fully FP32 | 324 / 335 | 42.5 / 42.3 | 0.63 / 0.64 |
| short-nothink (23) | zerv before (13b run1 / repeat) | 450 / 443 | 50.6 / 50.5 | 0.71 / 0.70 |
| decode-think (81) | zerv (small-row plans) | 393 / 390 | 45.7 / 45.8 | 5.96 / 5.95 |
| decode-think (81) | llama-server FA ub512 | 406 / 391 | 41.4 / 41.4 | 6.57 / 6.55 |
| decode-think (81) | llama-server fully FP32 | 764 / 775 | 40.6 / 40.6 | 7.04 / 7.07 |
| decode-think (81) | zerv before (13b run1 / repeat) | 487 / 483 | 45.8 / 45.8 | 6.05 / 6.04 |
| medium-prompt (836) | zerv (small-row plans) | 2966 / 2940 | 45.7 / 45.7 | 5.75 / 5.72 |
| medium-prompt (836) | llama-server FA ub512 | 1240 / 1246 | 41.3 / 41.3 | 4.32 / 4.33 |
| medium-prompt (836) | llama-server fully FP32 | 2973 / 2988 | 40.1 / 40.1 | 6.14 / 6.16 |
| medium-prompt (836) | zerv before (13b run1 / repeat) | 2947 / 2900 | 45.7 / 45.7 | 5.72 / 5.68 |
| long-prompt (3223) | zerv (small-row plans) | 10821 / 10742 | (52.7 / 52.7)* | 10.98 / 10.90 |
| long-prompt (3223) | llama-server FA ub512 | 3441 / 3452 | (45.6 / 45.8)* | 3.62 / 3.63 |
| long-prompt (3223) | llama-server fully FP32 | 9885 / 9925 | (43.1 / 43.0)* | 10.07 / 10.11 |
| long-prompt (3223) | zerv before (13b run1 / repeat) | 11031 / 10861 | (52.8 / 52.6)* | 11.19 / 11.02 |

\*8-token generations; not valid decode rates (see the
[13b report](2026-09-22-decode-attention.md)).

Output hashes are identical to 13b for every case. Host background load was as
described in the 13b report; llama-server's 23-token TTFT was 162 ms in the quieter
13a runs.

**Where zerv stands against llama-server FA (default Q8_1 prompt path):**
- Faster or tied: TTFT at 23 and 81 tokens; decode at every measured length.
- **2.4× slower TTFT at 836 tokens and 3.1× at 3223.** Prompt GEMM throughput (14–18
  TFLOP/s FP32) is the remaining gap (13e).

## Commands

```sh
python3 tools/compile_model.py --output-dir .tools/model-shaders-13d   # copy into src/model/shaders
zig build test && zig build gpu-test -Doptimize=ReleaseFast
zig build gemm-bench-build -Doptimize=ReleaseFast -Dcpu=native
./zig-out/bin/zerv-gemm-bench MODEL 1 8 16 23 32 48 64 81 96 128 151 192 256 324 512 > docs/bench/data/2026-09-22-small-prefill/gemm-tiles-run1.jsonl
python3 tools/verify_model.py --modes 0,1,13,29,60,512 --oracle-dir third_party/model-oracle/2026-09-22-a \
  --work-dir third_party/model-native/2026-09-22-small-prefill-gate1 --report docs/bench/data/2026-09-22-small-prefill/gate-run1.json
python3 tools/verify_model.py --modes 512 ... --work-dir third_party/model-native/2026-09-22-small-prefill-gate1-mode512 \
  --report docs/bench/data/2026-09-22-small-prefill/gate-run1-512.json
python3 tools/check_session.py --output docs/bench/data/2026-09-22-small-prefill/session-run1.json
./zig-out/bin/zerv-model-profile MODEL 8192 512 {23,81,151,836} 16
python3 bench/run_serving.py --output docs/bench/data/2026-09-22-small-prefill-serving-run1
python3 bench/run_serving.py --zerv-binary third_party/serving-bench/2026-09-22-small-prefill-serving-run1/zerv \
  --output docs/bench/data/2026-09-22-small-prefill-serving-repeat
```

**Correction (2026-09-23).** This report calls llama-server's default prompt path "Q8_1" (integer-dot activations). On this card that is wrong. With `KHR_coopmat` available, ggml converts activations to f16 and uses `matmul_quant_f16_f16acc`: f16 weights and activations with f16 accumulation on WMMA. Q8_1 is used only when cooperative matrices are disabled. The timings above are unaffected. Evidence: [llama precision](2026-09-23-llama-precision.md).
