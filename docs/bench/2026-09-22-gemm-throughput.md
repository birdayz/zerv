# Prefill GEMM throughput: scalar-X kernel (block 13e) — 2026-09-22

**Question:** why does the FP32 prefill GEMM stall at 14–18 TFLOP/s on the RX 7900 XTX,
and can it be made faster without changing any output bit?

**Result:**
- **Bottleneck:** the 128×128 kernel was bound by LDS reads.
- **Replacement:** a new "scalar-X" kernel reads the activation operand through the
  scalar path instead of LDS. It is **bit-identical** to the old kernel at equal split
  for every format.
  - At 128–512 rows it is **1.36–1.62× faster** under sustained load with realistic
    data, and 2.3–4.6× faster than 128×128 at 23 rows.
  - It replaces all 13d tiles.
- **Model prefill (GPU):**

  | Prompt tokens | 13d | Now | Speedup |
  |---:|---:|---:|---:|
  | 23 | 160.5 ms | 97 ms | 1.65× |
  | 836 | 2.87 s | 2.00 s | 1.44× |
  | 3223 | 10.60 s | 7.70 s | 1.38× |

- **Serving TTFT:**
  - 23 tokens: **102 ms vs llama-server's 162 ms**.
  - 81 tokens: **243 vs 369 ms**.
  - 836 tokens: 2.02 s vs 1.20 s; the gap to llama's Q8_1 path narrowed from 2.4× to 1.68×.
  - 3223 tokens: 7.76 s vs 3.41 s; the gap narrowed from 3.1× to 2.28×.
  - zerv is faster than llama-server's fully FP32 control at every length.
- **Oracle gates:** all pass, and the capture files are byte-identical between two runs.

## Research

All component data is in [`data/2026-09-22-gemm-throughput/`](data/2026-09-22-gemm-throughput/).
Experiment shader sources and ISA statistics are under
[`experiments/`](data/2026-09-22-gemm-throughput/experiments/), with a SHA256SUMS file.

1. **Emitted ISA** (`RADV_DEBUG=shaders,shaderstats`, a diagnostic environment variable
   only; [stats](data/2026-09-22-gemm-throughput/experiments/isa-stats-summary.txt)).
   For the old Q4_0 128×128 kernel:
   - wave64, 168 VGPRs, 32 KiB LDS, **VOPD 0**;
   - the K tile is fully unrolled into 2048 `v_fmac`s plus 256 `ds_load_b64`;
   - there are 16 separately guarded `buffer_load_b32` X loads per tile.
2. **Wave32 + VOPD** (`RADV_PERFTEST=cswave32`, diagnostic): 953 dual-issue
   instructions, but **no speedup**, and often slower
   ([wave32](data/2026-09-22-gemm-throughput/wave32-probe.jsonl) vs
   [wave64](data/2026-09-22-gemm-throughput/wave64-probe.jsonl)). The FMA issue rate is
   not the limit.
3. **Ablations** (same kernel, one part removed; results invalid, timing only;
   [source](data/2026-09-22-gemm-throughput/experiments/ablation-gemm.comp)):
   - Removing the global loads changed little: attn_q 16.2 → 16.8 and ffn_gate
     15.2 → 15.7 TFLOP/s.
   - Reading LDS once per K tile instead of per k gave **up to 40.7 TFLOP/s**
     (ffn_gate 35.6).
   - **Verdict: the kernel is LDS-read bound.**
4. **Bank-conflict swizzle** (A reads `tm + 16·i4` instead of `2·tm + i4`,
   bit-identical): only +7–10%.
5. **Scalar-X design:**
   - A 64-thread group reads its own X rows as wave-uniform `vec4`s through a
     read-only, `restrict` second binding of the activation buffer. The compiler turns
     these into `s_buffer_load` (SMEM), and VALU FMAs read them from SGPRs.
   - Only A goes through LDS: one conflict-free `vec4` per lane per k.
   - The parameter sweep over 13 configurations, all bit-identical, is in `sweep-*.jsonl`:
     - lane rows 2/4/8;
     - X rows per wave 4/8/16/32;
     - k per scalar load 1/2/4.
   - **Best:** 4 M rows per lane, 8 X rows per wave, `vec4` over k (tile 256×32),
     at 22–24.7 TFLOP/s.
   - More X rows per wave spilled SGPRs into VGPR lanes. Eight rows per lane needs
     64 KiB LDS and loses occupancy.
6. **Wave uniformity:**
   - `gl_SubgroupID` gives scalar loads, but its correctness depends on the driver
     picking wave64.
   - `tid/64` is correct for any subgroup size dividing 64, but the compiler cannot
     prove it uniform. X then goes through VMEM, at ~17 instead of ~23 TFLOP/s.
   - **`subgroupBroadcastFirst(tid/64)`** is correct for subgroup sizes 32 and 64, and
     scalar. It stays bit-identical and fast even under forced wave32 (21 TFLOP/s).
   - The host now queries `VkPhysicalDeviceSubgroupProperties` (bindings regenerated,
     C-ABI fixture 45 structs / 65 constants) and rejects devices without compute
     ballot or with subgroups wider than 64.
7. **Power and data dependence.** Throughput depends on the data:
   - Zero X runs up to 2× faster than random X in 9-sample bursts, and ~10% faster
     sustained.
   - Under sustained load the card sits at **310–340 W against its 339 W cap**, with
     the shader clock at ~3.1 GHz for zero X and ~2.8 GHz for random X.
   - Short 9-sample runs also include clock ramp-up.
   - **Every earlier GEMM component number in 13a/13d used uninitialized, likely zero,
     X and short runs, so it overstates absolute throughput.** Relative comparisons and
     all model-level profiles, which use real activations, are unaffected.
   - The bench now uses deterministic N(0,1) X by default and `--samples` for
     sustained runs.
8. **Q5_K:** the generic one-row-per-thread decoder ran at only 9 TFLOP/s: ~20 unaligned
   2-byte-granular word loads per K tile. Q5_K rows are 16-byte aligned (176-byte
   blocks). Five aligned `uvec4` loads with the same decode arithmetic reach 21 TFLOP/s,
   bit-identical.

## Final component comparison

Sustained runs of 1000 samples with random X, two rounds, old 128×128 vs shipped
scalar-X. The old kernel was recompiled from the preserved source; its Q4_0 SPIR-V is
byte-identical to the 13d shipped module (`d0696310…`).

| Tensor | Format | Rows | 128×128 TFLOP/s (r1 / r2) | scalar-X TFLOP/s (r1 / r2) | Speedup (r1 / r2) |
|---|---|---:|---:|---:|---:|
| blk.0.attn_qkv.weight | q4_0 | 23 | 2.7 / 2.6 | 10.9 / 10.9 | 3.98 / 4.13 |
| blk.0.attn_qkv.weight | q4_0 | 128 | 13.6 / 13.3 | 21.6 / 21.6 | 1.59 / 1.62 |
| blk.0.attn_qkv.weight | q4_0 | 512 | 15.7 / 15.2 | 23.0 / 23.0 | 1.47 / 1.51 |
| blk.0.attn_gate.weight | q4_0 | 512 | 16.8 / 16.4 | 23.6 / 23.4 | 1.41 / 1.43 |
| blk.0.ffn_gate.weight | q4_0 | 23 | 2.9 / 2.8 | 12.6 / 12.6 | 4.34 / 4.45 |
| blk.0.ffn_gate.weight | q4_0 | 128 | 14.4 / 14.2 | 21.1 / 21.0 | 1.46 / 1.48 |
| blk.0.ffn_gate.weight | q4_0 | 512 | 16.5 / 16.1 | 22.5 / 22.4 | 1.36 / 1.39 |
| blk.8.ffn_down.weight | q4_0 | 512 | 13.8 / 13.6 | 21.8 / 21.7 | 1.58 / 1.59 |
| blk.3.attn_q.weight | q4_0 | 512 | 17.0 / 16.8 | 23.8 / 23.8 | 1.40 / 1.42 |
| blk.3.attn_k.weight | q4_0 | 128 | 9.0 / 9.0 | 10.6 / 10.3 | 1.17 / 1.15 |
| blk.3.attn_k.weight | q4_0 | 512 | 12.2 / 12.1 | 17.8 / 17.5 | 1.46 / 1.45 |
| blk.3.attn_output.weight | q4_0 | 512 | 15.9 / 15.8 | 22.6 / 22.5 | 1.42 / 1.43 |
| blk.0.ffn_down.weight | q4_1 | 23 | 2.8 / 2.8 | 12.9 / 13.0 | 4.58 / 4.61 |
| blk.0.ffn_down.weight | q4_1 | 512 | 13.6 / 13.6 | 22.0 / 22.0 | 1.62 / 1.62 |
| blk.0.ssm_out.weight | q5_k | 23 | 2.4 / 2.4 | 9.0 / 9.1 | 3.76 / 3.75 |
| blk.0.ssm_out.weight | q5_k | 512 | 15.0 / 15.0 | 22.8 / 22.7 | 1.51 / 1.52 |
| blk.0.ssm_alpha.weight | f32_k | 512 | 2.2 / 2.2 | 1.4 / 1.5 | **0.62 / 0.68** |

Full table: `final-old128-*.jsonl` and `final-sx-*.jsonl`.

**Regression retained:** `ssm_alpha`/`ssm_beta` (F32, M=48) are slower. A 256-row tile
is mostly padding at M=48. The cost is ≈6 ms per 512-row chunk (~0.5%); their phase
(`lin_in`) still went from 249 to 179 ms.

## Correctness

| Gate | Evidence | Result |
|---|---|---|
| Scalar-X vs 128×128 at equal split, every format, 23/100/512 rows | `sx-all-formats-check2.jsonl` (bench `--check`, before the old kernels were removed) | bit-identical everywhere |
| Independent matvec fixtures (scaled rows, zero rows, tails, split-K, +inf padding past K to catch the k guard) | `tests/model_gpu.zig`, Debug + ReleaseFast | pass |
| Attention-style F32 strided/batched products (runtime M/K, 16-byte rows) vs FP64 | same | pass |
| Alignment and extent validation (X rows, Q5_K rows) | `tests/model_gpu.zig`, `tests/model.zig` | pass |
| Cleanup: shipped `gemm_*.spv` vs verified scalar-X modules | byte comparison | identical, all 6 formats |
| Model oracle, modes 0/1/13/29/60/512 | [gate-run2](data/2026-09-22-gemm-throughput/gate-run2.json), run1 kept | worst/bound ≤0.595; mode-512 short 0.477 (13d: 0.836); greedy all. Run1 and run2 capture files byte-identical |
| Serving greedy equality | [session-run1](data/2026-09-22-gemm-throughput/session-run1.json) | JSON + SSE, both cases |

## Per-phase profile

`zerv-model-profile MODEL 8192 512 N 16`; data in
[`profile-p*.jsonl`](data/2026-09-22-gemm-throughput/).

**Prefill GPU time:**

| Prompt tokens | 13d | Now |
|---:|---:|---:|
| 23 | 160.5 ms | 97.0 ms |
| 81 | 372.5 ms | 240.1 ms |
| 151 | 661 ms | 397 ms (one chunk) |
| 836 | 2867 ms | 1995 ms |
| 3223 | 10600 ms | 7703 ms |

**One 512-row chunk at p0=0:** 1624 ms before 13d, **1178 ms now**. Per phase:

| Phase | Before | Now |
|---|---:|---:|
| ffn gate+up | 702 ms | 492 ms |
| ffn down | 379 ms | 263 ms |
| linear-attention input projections | 249 ms | 179 ms |
| ssm_out | 98 ms | 68 ms |
| attention q/k/v | 70 ms | 49 ms |
| attention out | 32 ms | 23 ms |
| DeltaNet scan | 56 ms | 62 ms (now 5%) |

## Serving (median of 3; run1 / repeat)

The repeat reused the run1 binary (`b6a38b8f…`).
[run1](data/2026-09-22-gemm-throughput-serving-run1/summary.json),
[repeat](data/2026-09-22-gemm-throughput-serving-repeat/summary.json):

| Case (prompt tok) | Engine | TTFT ms | Decode tok/s | Total s |
|---|---|---:|---:|---:|
| short-nothink (23) | zerv (scalar-X GEMM) | 102 / 102 | 49.9 / 50.0 | 0.36 / 0.36 |
| short-nothink (23) | llama-server FA ub512 | 162 / 163 | 43.8 / 44.0 | 0.46 / 0.46 |
| short-nothink (23) | llama-server fully FP32 | 249 / 250 | 43.3 / 43.1 | 0.55 / 0.55 |
| short-nothink (23) | zerv before (13d run1 / repeat) | 168 / 167 | 50.4 / 50.6 | 0.43 / 0.43 |
| decode-think (81) | zerv (scalar-X GEMM) | 243 / 243 | 45.6 / 45.6 | 5.83 / 5.83 |
| decode-think (81) | llama-server FA ub512 | 369 / 372 | 41.5 / 41.4 | 6.51 / 6.52 |
| decode-think (81) | llama-server fully FP32 | 682 / 684 | 40.6 / 40.6 | 6.96 / 6.96 |
| decode-think (81) | zerv before (13d run1 / repeat) | 393 / 390 | 45.7 / 45.8 | 5.96 / 5.95 |
| medium-prompt (836) | zerv (scalar-X GEMM) | 2017 / 2016 | 45.5 / 45.4 | 4.80 / 4.81 |
| medium-prompt (836) | llama-server FA ub512 | 1202 / 1208 | 41.4 / 41.3 | 4.27 / 4.28 |
| medium-prompt (836) | llama-server fully FP32 | 2931 / 2953 | 40.1 / 40.1 | 6.09 / 6.12 |
| medium-prompt (836) | zerv before (13d run1 / repeat) | 2966 / 2940 | 45.7 / 45.7 | 5.75 / 5.72 |
| long-prompt (3223) | zerv (scalar-X GEMM) | 7759 / 7753 | (52.1 / 52.1)* | 7.91 / 7.91 |
| long-prompt (3223) | llama-server FA ub512 | 3405 / 3422 | (45.8 / 45.7)* | 3.58 / 3.60 |
| long-prompt (3223) | llama-server fully FP32 | 9868 / 9955 | (43.2 / 43.1)* | 10.06 / 10.14 |
| long-prompt (3223) | zerv before (13d run1 / repeat) | 10821 / 10742 | (52.7 / 52.7)* | 10.98 / 10.90 |

\*8-token generations; not valid decode rates.

All output hashes are identical to 13d. The host was quieter than during 13b/13d
(llama-server back at 162 ms at 23 tokens). Peak VRAM for zerv was 18.81 GB.

**Where zerv stands against llama-server FA (default Q8_1 prompt path):**
- **Faster:**
  - TTFT at 23 and 81 tokens, by 1.6× and 1.5×;
  - total request time for the short and 81-token cases;
  - decode at every measured length.
- **Slower:** TTFT at 836 tokens by 1.68× and at 3223 by 2.28×. llama quantizes prompt
  activations to Q8_1 and uses integer dot products; zerv stays FP32.

## Commands

```sh
RADV_DEBUG=shaders,shaderstats ./zig-out/bin/zerv-gemm-bench MODEL 512 2> isa.txt   # ISA/stat dumps
RADV_PERFTEST=cswave32 ./zig-out/bin/zerv-gemm-bench MODEL 128 512                  # wave32 probe
glslc ... -DABLATE_NOGLOBAL=1|-DABLATE_LDSONCE=1 experiments/ablation-gemm.comp       # ablations (via --spv)
./zig-out/bin/zerv-gemm-bench MODEL --samples 1000 --spv .tools/gemm-old/old128_q4_0.spv --variant q4_0 --grid 128x128 23 128 512
./zig-out/bin/zerv-gemm-bench MODEL --samples 1000 --variant q4_0 23 128 512
python3 tools/verify_model.py --modes 0,1,13,29,60,512 --oracle-dir third_party/model-oracle/2026-09-22-a \
  --work-dir third_party/model-native/2026-09-22-scalarx-gate2 --report docs/bench/data/2026-09-22-gemm-throughput/gate-run2.json
python3 tools/check_session.py --output docs/bench/data/2026-09-22-gemm-throughput/session-run1.json
./zig-out/bin/zerv-model-profile MODEL 8192 512 {23,81,151,836,3223} 16
python3 bench/run_serving.py --output docs/bench/data/2026-09-22-gemm-throughput-serving-run1
python3 bench/run_serving.py --zerv-binary third_party/serving-bench/2026-09-22-gemm-throughput-serving-run1/zerv \
  --output docs/bench/data/2026-09-22-gemm-throughput-serving-repeat
```

**Correction (2026-09-23).** This report calls llama-server's default prompt path "Q8_1" (integer-dot activations). On this card that is wrong. With `KHR_coopmat` available, ggml converts activations to f16 and uses `matmul_quant_f16_f16acc`: f16 weights and activations with f16 accumulation on WMMA. Q8_1 is used only when cooperative matrices are disabled. The timings above are unaffected. Evidence: [llama precision](2026-09-23-llama-precision.md).
