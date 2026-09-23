# FP32 prefill GEMM efficiency (block 13h)

Date: 2026-09-23. Question: the shipped scalar-X GEMM runs at 22–24 TFLOP/s, while the
wave64 FP32 FMA peak on this card measures 63–66 TFLOP/s
([13g](2026-09-23-coopmat-research.md)). What limits it, and can the kernel go faster
without changing a single output bit?

**Result:**

- **Part 1.** Bit-identical kernels, 1.29–1.41× faster on Q4_0/Q4_1 at 512 rows and
  1.24× on Q5_K. A 512-row chunk takes 1178 → 881 ms. Served TTFT is 24–25% lower at
  every length, with byte-identical outputs.
- **Part 2.** A 256×64 tile for full 512-row chunks brings the chunk to 788–796 ms
  and TTFT at 3223 tokens from 7755 (12b) to 5336 ms (−31% overall).
  - A new ≥512-token oracle case now gates 512-row chunks at model level for the
    first time.
[Spec section](specs/prefill.md#gemm-efficiency--block-13h-2026-09-23).

## Setup

- RX 7900 XTX (RADV, Mesa 26.2.3), capped at 339 W.
- `zerv-gemm-bench` runs on real weights with deterministic N(0,1) X.
- Component runs are sustained: 300 samples for exploration, 1000 × 2 rounds for
  the final comparison.
- GPU jobs ran one at a time.
- Source and SPIR-V hashes are in
  [hashes.txt](bench/data/2026-09-23-gemm-efficiency/hashes.txt). The 13e/13h shader
  manifests are alongside it.

## Diagnosis

**Ablations of the shipped Q4_0 kernel.**
[ablation-r1.jsonl](bench/data/2026-09-23-gemm-efficiency/ablation-r1.jsonl); source in
[experiments/ablation.comp](bench/data/2026-09-23-gemm-efficiency/experiments/ablation.comp).

- The results are invalid; the runs are timing only.
- Each ablation replaces one part with values derived from a runtime word, so the
  compiler cannot fold the FMAs away.
- A first set of ablations used compile-time constants. The compiler folded whole
  FMA chains (for example, "no LDS writes" reached 1147 "TFLOP/s"). That set is
  retained as
  [ablation-invalid-constfold.jsonl](bench/data/2026-09-23-gemm-efficiency/ablation-invalid-constfold.jsonl)
  and not used.

| Variant (512 rows, ffn_gate) | TFLOP/s |
| --- | --- |
| shipped | 22.3 |
| no dequant | 29.3 |
| no LDS reads | 25.2 |
| no X loads | 25.2 |
| no barriers | 24.0 |
| FMAs only (no dequant, LDS reads or X) | 56.5 |

No single part dominates, but together they cost 2.5×.

**ISA** (`RADV_DEBUG=shaders,shaderstats`):

1. **SGPR spills.** The fully unrolled X loads keep 8 rows × 32 k = 256 scalar values
   live. That caused **137 SGPR spills** into VGPR lanes: 58 `v_writelane` and
   69 `v_readlane` per K tile.
2. **Serialized loads.** SMEM X loads and LDS reads share `lgkmcnt`. SMEM can return
   out of order, so every K step waits `lgkmcnt(0)` before its FMAs, and the next X
   load cannot be overlapped.
3. **Unaligned block reads.** Q4_0 blocks (18 bytes, 2-byte aligned) are read through
   2-byte-granular `word_at`, with an execz branch per word.

## Steps

All steps were measured with 300 samples at 512 rows, and every one was checked byte
for byte against the shipped kernel at 23/100/512 rows. Raw data is in
[v2-steps-r1.jsonl](bench/data/2026-09-23-gemm-efficiency/v2-steps-r1.jsonl).

| Step | ffn_gate TFLOP/s | Identical |
| --- | --- | --- |
| shipped 13e | 22.6 | — |
| aligned block loads (5 words, no branches) | 25.5 | yes |
| + raw block prefetched one tile ahead in registers | 26.5 | yes |
| + X loaded one kb ahead in a **non-unrolled** kb loop | **29.6** | yes |
| same, kb loop unrolled | 25.9 (152 spills return) | yes |
| X through VMEM (per-lane index) | 18.4 | yes |
| X for 2 kb per iteration / kb-pair ping-pong / forced ordering | 27.6 / 28.9 / 27.6 | yes |
| no LDS: each lane decodes its own 4 rows ([gemm-direct.comp](bench/data/2026-09-23-gemm-efficiency/experiments/gemm-direct.comp)) | 21.1 (unrolled) / 17.3 | yes |
| 16-k LDS tile (occupancy) | 24.2 | yes |
| 16 X rows per wave (tile 256×64) | 33.5 at equal split | yes; shipped for 512-row chunks in part 2 |

The shipped design combines the aligned, prefetched block loads with the
non-unrolled, one-kb-ahead X pipeline. Q5_K gets the same prefetch of its five
aligned `uvec4` words. Q6_K and F32 get only the X pipeline; F32 masks the partial
final K tile on the pipelined X.

Two power readings at 100 ms intervals:

- The new Q4_0 GEMM draws ~291 W at ~2.26 GHz.
- The pure-FMA peak kernel draws ~273 W at ~2.60 GHz.

The GEMM is therefore still below both the power cap and the clock-adjusted FMA
peak (~55 TFLOP/s at 2.26 GHz).

## Final component comparison

Two rounds of 1000 samples, old (13e SPIR-V) vs new. Raw data is in
`final-{old,new}-*-r{1,2}.jsonl`.

| Tensor | Format | Rows | 13e TFLOP/s (r1 / r2) | 13h TFLOP/s (r1 / r2) | Speedup |
| --- | --- | ---: | --- | --- | --- |
| attn_qkv | q4_0 | 512 | 23.3 / 23.0 | 31.6 / 31.2 | 1.36 / 1.35 |
| attn_gate | q4_0 | 512 | 24.3 / 23.5 | 31.4 / 31.3 | 1.29 / 1.33 |
| ffn_gate | q4_0 | 512 | 22.6 / 22.4 | 31.4 / 31.3 | 1.39 / 1.40 |
| ffn_gate | q4_0 | 23 | 12.8 / 12.5 | 17.1 / 17.0 | 1.34 / 1.36 |
| blk.8 ffn_down | q4_0 | 512 | 22.0 / 21.7 | 30.5 / 30.5 | 1.39 / 1.41 |
| attn_q | q4_0 | 512 | 24.0 / 23.8 | 32.5 / 32.5 | 1.35 / 1.36 |
| attn_k | q4_0 | 512 | 17.9 / 17.8 | 23.0 / 23.2 | 1.28 / 1.30 |
| attn_output | q4_0 | 512 | 22.8 / 22.6 | 30.0 / 30.1 | 1.32 / 1.33 |
| blk.0 ffn_down | q4_1 | 512 | 22.0 / 22.0 | 30.3 / 30.3 | 1.38 / 1.38 |
| ssm_out | q5_k | 512 | 22.4 / 22.5 | 27.8 / 27.8 | 1.24 / 1.24 |
| ssm_alpha | f32 | 512 | 1.5 / 1.5 | 1.6 / 1.6 | 1.03 |

All rows (23/128/512) are in the raw files. At 23 and 128 rows the Q4 speedup is
1.24–1.36×. The Q5_K rows come from the final version, with the prefetch applied.
The table's Q5_K result (27.8) supersedes an intermediate 25.4 measured with only
the X pipeline.

## Correctness

| Gate | Result |
| --- | --- |
| Full output bytes vs 13e: Q4_0 ×7 tensors, Q4_1, Q5_K, F32; 1/23/100/512 rows; natural split and forced `--k-chunk` 0 / 1024 | identical (all 21 + 21 + 28 Q4_0 files, and all Q4_1/Q5_K/F32 files) |
| Q6_K (`output.weight`), 1/23/100/512 rows | identical. Forced split not tested: the model never splits or batches this tensor |
| `zig build gpu-test` (independent GEMM fixtures, every format; attention F32) | 15/15 in Debug and ReleaseFast |
| [verify_model](bench/data/2026-09-23-gemm-efficiency/gate-run1.json) modes 0/1/13/29/60/512 | pass. Every captured tensor and logit file is byte-identical to the 13f gate; only the timing-bearing `native-summary.json`/`stderr.txt` differ |
| Served outputs | byte-identical to the 12b runs (all four cases) |

## Model and serving

**Per-phase profile.** `zerv-model-profile MODEL 8192 512 N 16`; data in
[profile-p512.jsonl](bench/data/2026-09-23-gemm-efficiency/profile-p512.jsonl) and
[profile-p3223.jsonl](bench/data/2026-09-23-gemm-efficiency/profile-p3223.jsonl).

- **One 512-row chunk:** 1178 → **881 ms**.
  - ffn gate+up: 492 → 344 ms
  - ffn down: 263 → 186 ms
  - linear-attention input: 179 → 135 ms
  - ssm_out: 68 → 55 ms
  - attention q/k/v: 49 → 36 ms
  - attention out: 23 → 17 ms
  - DeltaNet scan: 62 → 63 ms (unchanged)
- **3223-token prefill:** 7703 → **5802 ms**.
- **Decode step:** unchanged, at 20.2 ms.

**Serving.** `bench/run_serving.py --engines zerv,llama-fa-ub512`, 3 repeats.
[run1](bench/data/2026-09-23-gemm-efficiency-serving-run1/),
[repeat](bench/data/2026-09-23-gemm-efficiency-serving-repeat/) (pinned run1 binary
`1e668010…`). Median TTFT, run1 / repeat:

| Prompt | zerv 12b | zerv 13h | llama-server FA ub512 |
| --- | --- | --- | --- |
| 23 tok | 100.8 | **76 / 76** | 162 / 163 |
| 81 tok | 241.7 | **185 / 185** | 368 / 368 |
| 836 tok | 2016 | **1521 / 1524** | 1208 / 1206 |
| 3223 tok | 7755 | **5874 / 5873** | 3412 / 3412 |

- Decode is unchanged: 54 / 49.3 / 49.1 tok/s, against llama-server's
  44 / 41.3 / 41.2.
- zerv's TTFT is now 2.1× lower than llama-server's up to 81 tokens.
- At 836 tokens zerv is 1.26× slower (was 1.67×), and at 3223 tokens 1.72× slower
  (was 2.27×). llama-server uses Q8_1-quantized activations there; its FP32 control
  took 2931 / 9868 ms in 13f.

## Part 2: wide tile (256×64) for 512-row chunks

[Spec](specs/prefill.md#part-2-wide-tile-25664).

- `gemm.comp` gains a compile-time `XW`: 16 X rows per 64-thread group, with X read
  as `vec2` pairs. The `XW=8` SPIR-V is byte-identical to part 1.
- Measured VGPRs 192 and 10–13 SGPR spills (vs 29 with `vec4` X), no VGPR spills.

**Tile sweep.** [tile-sweep-r1.jsonl](bench/data/2026-09-23-gemm-efficiency/tile-sweep-r1.jsonl);
300 samples; each tile at its own natural split-K. Narrow-over-wide time ratio
(> 1: the wide tile is faster):

| Tensor (M) | 23 | 32 | 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- | --- |
| attn_gate (6144) | 0.69 | 0.72 | 1.05 | 1.01 | 1.08 | 1.08 |
| attn_qkv (10240) | 0.72 | 0.75 | 1.09 | 1.01 | 1.14 | 1.09 |
| ffn_gate (17408) | 0.68 | 0.62 | 1.12 | 1.05 | 1.12 | 1.15 |
| blk.8 ffn_down (5120) | 0.59 | 0.64 | 1.15 | 1.16 | 0.99 | 1.17 |
| blk.0 ffn_down q4_1 (5120) | 0.65 | 0.61 | 1.12 | 1.14 | 0.99 | 1.17 |
| ssm_out q5_k (5120) | 0.72 | 0.63 | 1.01 | 1.09 | 1.01 | 1.14 |
| attn_output (5120) | 0.66 | 0.65 | 1.02 | 1.09 | 1.00 | 1.14 |
| attn_q (12288) | 0.44 | 0.37 | 0.54 | 0.65 | 1.01 | 1.09 |
| attn_k (1024) | 0.66 | 0.64 | 0.72 | 0.77 | 0.72 | 0.71 |

**Split check.** [split-sweep-r1.jsonl](bench/data/2026-09-23-gemm-efficiency/split-sweep-r1.jsonl)
forces every split count on three shapes at 512 rows. The natural rule (384-workgroup
target, counted in the tile's own tiles) picks the fastest wide configuration in each
case: ffn_down 3 splits, attn_output 3, attn_qkv 2. These single-process short runs
vary by ±8% between invocations, so only the within-sweep ordering is used.

**Rule.** The wide tile is used iff the plan has ≥ 512 rows and M ≥ 4096 (static; no
autotuning). This covers every full 512-row chunk except attn_k/v.

**Correctness.**

- `gpu-test`: every quantized fixture case runs on both tiles, and the wide outputs
  equal the narrow outputs byte for byte, split-K pass included (Debug and
  ReleaseFast). The bench also compared Q4_0/Q4_1/Q5_K/F32 at forced equal splits
  (0, 1024) and 1/23/100/512 rows: identical.
- **New oracle case.** No existing oracle case reached a 512-row chunk (the longest is
  221 tokens), so no 512-row prefill had ever been checked at model level (13a–13h).
  - `generate_model_oracle.py --case-set long` produces `long-prefill`: 546 prompt
    tokens + 16 greedy tokens. It captures l_out, attn_output and result_norm.
  - Fixture: `tests/fixtures/model/qwen38-oracle-long.json`. Data:
    `third_party/model-oracle/2026-09-23-long`, 4.3 GB.
  - llama vs FP64: worst intermediate 2.2e-5, logits 8.9e-6, no argmax
    disagreements.
  - The FP64 reference's shared projection buffer was sized for 512 tokens. It is now
    sized from the longest sequence. The first attempt failed on that explicit guard
    (log retained: `third_party/model-oracle-2026-09-23-long-failed1.log`). A second
    background launch exited with status 2 and left no log (cause not determined);
    the third completed in 56 min.
- **[Long gate](bench/data/2026-09-23-gemm-efficiency/gate-long-wide-run1.json)**
  (`verify_model --fixture …long.json --modes 0,512,64`): pass.

  | Mode | Worst tensor / bound | Worst logits (bound 3.55e-5) | Greedy |
  | --- | --- | --- | --- |
  | 512-row chunk (wide) + remainder | 0.27 | 9.4e-7 | 19/19 |
  | 64-row chunks | 0.71 | 1.5e-6 | 26/26 |
  | decode | 0.64 | 2.05e-5 | 562/562 |

  - The decode worst sits at position 438, a sensitive position where llama's own
    error is 8.6e-6. The mean decode logits error, 5.9e-7, equals llama's 6.0e-7.
- **Short gates.** [gate-wide-run1](bench/data/2026-09-23-gemm-efficiency/gate-wide-run1.json):
  pass, identical numbers. No short case uses a 512-row plan.
- **Default oracle regenerated.** The generator edits (the `--case-set` option and
  the buffer sizing) broke the fixture's source-hash provenance test. The default
  oracle was regenerated with the current scripts.
  - `third_party/model-oracle/2026-09-23-default-regen` took 31 min.
  - Every data file is byte-identical to `2026-09-22-a` (tensors, logits, FP64
    reference, state). Only `oracle.stderr` (libllama timing log), the absolute build
    path and the two source hashes differ.
  - The regenerated fixture replaces `tests/fixtures/model/qwen38-oracle.json`. The
    old one is kept as
    [qwen38-oracle-2026-09-22-a.json](bench/data/2026-09-23-gemm-efficiency/qwen38-oracle-2026-09-22-a.json).
  - The gate against the new directory
    ([gate-wide-run2-regen](bench/data/2026-09-23-gemm-efficiency/gate-wide-run2-regen.json))
    gives identical numbers.
  - The provenance test now also covers the long fixture.
- **Served outputs.** Byte-identical to part 1 in all four cases, including 836 and
  3223 tokens, which run full 512-row chunks.

**Profile.** [profile-wide-p512.jsonl](bench/data/2026-09-23-gemm-efficiency/profile-wide-p512.jsonl),
[p3223](bench/data/2026-09-23-gemm-efficiency/profile-wide-p3223.jsonl).

- One 512-row chunk: 881 → 788–796 ms (two runs).
  - ffn gate+up: 344 → 311 ms
  - ffn down: 186 → 159 ms
  - linear-attention input: 135 → 124 ms
- 3223-token prefill: 5802 → 5283 ms.

**Serving.** [run1](bench/data/2026-09-23-gemm-wide-serving-run1/),
[repeat](bench/data/2026-09-23-gemm-wide-serving-repeat/) (pinned run1 binary
`48c5da5c…`). Median TTFT in ms:

| Prompt | Part 1 | Wide run1 / repeat | llama-server FA ub512 run1 / repeat |
| --- | --- | --- | --- |
| 23 tok | 76 | 76 / 77 | 240* / 162 |
| 81 tok | 185 | 184 / 186 | 459* / 373 |
| 836 tok | 1521 | **1415 / 1411** | 1361* / 1212 |
| 3223 tok | 5874 | **5337 / 5335** | 3546* / 3420 |

- \*llama-server's run1 section was disturbed: 23-token TTFT of 240 ms against its
  usual 162, and decode 40 instead of 44 tok/s. The cause was not identified (the
  host sometimes has background load). The data is retained; the repeat is the
  valid comparison.
- zerv's decode is unchanged: 53–54 / 48–49 / 48–49 tok/s.
- **Against llama-server (repeat):** zerv is 2.1× faster to first token up to
  81 tokens. It is 1.16× slower at 836 tokens and 1.56× slower at 3223 tokens,
  where llama-server uses Q8_1 activations.

## Not taken (retained as experiments)

- **No-LDS direct kernel, X via VMEM, 16-k tile, fused kb pairs, 512-thread
  workgroups (XW=8, 8 waves).** All slower or no better. See the steps table and
  `v3-nosplit-r1.jsonl`.

## Commands

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
zig build gemm-bench-build -Doptimize=ReleaseFast -Dcpu=native
# ablation / experiment modules: glslc --target-env=vulkan1.1 -O -fshader-stage=compute \
#   -DFORMAT=2 -DBLOCK_BYTES=18 -DPAYLOAD_OFFSET=2 -DA_MCONTIG=0 [-DABL_*|-DV2_*] experiments/<file>.comp
zerv-gemm-bench MODEL --variant q4_0 --samples 300 --spv X.spv --grid 256x32 512
# bit identity
zerv-gemm-bench MODEL --variant V [--k-chunk C] [--tensor output.weight] --samples 1 --dump DIR 1 23 100 512
# final
zerv-gemm-bench MODEL --variant V --samples 1000 [--spv 13e.spv --grid 256x32] 23 128 512
python3 tools/verify_model.py --modes 0,1,13,29,60,512 --oracle-dir third_party/model-oracle/2026-09-22-a \
    --work-dir third_party/model-native/2026-09-23-gemm-efficiency-gate1 --report docs/bench/data/2026-09-23-gemm-efficiency/gate-run1.json
python3 bench/run_serving.py --output docs/bench/data/2026-09-23-gemm-efficiency-serving-run1 --engines zerv,llama-fa-ub512
python3 bench/run_serving.py --output docs/bench/data/2026-09-23-gemm-efficiency-serving-repeat --engines zerv,llama-fa-ub512 \
    --zerv-binary third_party/serving-bench/2026-09-23-gemm-efficiency-serving-run1/zerv
zerv-model-profile MODEL 8192 512 {512,3223} 16
```

**Correction (2026-09-23).** This report calls llama-server's default prompt path "Q8_1" (integer-dot activations). On this card that is wrong. With `KHR_coopmat` available, ggml converts activations to f16 and uses `matmul_quant_f16_f16acc`: f16 weights and activations with f16 accumulation on WMMA. Q8_1 is used only when cooperative matrices are disabled. The timings above are unaffected. Evidence: [llama precision](2026-09-23-llama-precision.md).
