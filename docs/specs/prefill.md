# Batched FP32 prefill — block 13a

Status: **implemented and verified** (2026-09-22). All gates below pass; evidence and
measurements: [prefill report](../bench/2026-09-22-prefill.md).

Goal: process prompt tokens in chunks instead of one decode step per token, without
lowering precision. Semantics are unchanged from [the model spec](model.md); only
evaluation order and batching change.

## Why (measured)

[Serving report](../bench/2026-09-22-serving.md): time to first token is 3–32× worse than
llama-server because prefill replays the ~23 ms decode step per prompt token, re-reading
all ~15.8 GB of weights for every token. A chunk of B tokens reads each weight once and
does B dot products, making projections compute-bound. Estimated projection work:
2 × 27e9 × B FLOP per chunk; at 10–20 TFLOP/s FP32 that is ≈190–370 tok/s versus ~40 now.

Competitor precision (source-backed, pinned ggml `ggml_vk_mul_mat_q_f16`, and confirmed at
runtime on 2026-09-23 in [llama precision](../bench/2026-09-23-llama-precision.md)): on this
card llama-server's prompt matmuls use FP16 cooperative matrices. Activations are
converted to f16 and weights dequantized to f16, with **f16 accumulation**
(`matmul_quant_f16_f16acc`). The Q8_1 integer-dot kernels run only when cooperative
matrices are disabled. (An earlier version of this text said Q8_1.) zerv's default prefill stays FP32; lower-precision modes, if
ever added, must be explicit options with measured quality.

## Design

- **Chunk**: up to `prefill_chunk` tokens (`--prefill-chunk`, default 512; 0 disables
  batching) at consecutive positions
  `p0..p0+n-1`. Host writes n, p0, token ids and FP64-computed RoPE cos/sin per row
  into the io buffer; one pre-recorded command per chunk; kernels read n and p0 from io
  (no re-recording). Rows ≥ n are never written.
- **Projections**: tiled GEMM `Y[t][m] = Σ_k X[t][k]·W[m][k]`, 128×128 output tiles
  (256 threads × 8×8), split-K into 256-multiples with a fixed-order reduction when the
  tile grid is below ~384 workgroups, K-tiles of 32 with weights decoded exactly into shared memory (same exact
  decoders as matvec), FP32 FMA accumulation per K-tile then added into a running total
  (two-level accumulation to bound rounding growth for K = 17408).
- **Attention**: scores `S = Q·Kᵀ` over keys `0..p0+n-1` with the same GEMM (F32 operand,
  m-contiguous K cache), causal row softmax (scale 1/16, masked entries exactly 0),
  `O = P·V` with the same GEMM, then `O·sigmoid(gate)`. Scores buffer: 24 × chunk × context.
  KV rows are written at their positions before the score GEMM.
- **GatedDeltaNet**: one workgroup per value head scans the chunk sequentially with the
  128×128 state column held in registers (same per-token arithmetic as the decode kernel);
  conv: one thread per channel scans the chunk with a 3-value window, then SiLU and per-head
  L2 norm.
- **Logits**: only for the chunk's last row (final norm of that row, then the existing
  output matvec). Decode continues with the existing step at position p0+n.

## Correctness gates (declared before implementation)

1. GEMM component: every explicit block-08 matvec fixture case with X built from the case
   input scaled by powers of two (exact scaling) plus zero rows and a row count not a
   multiple of the tile; per element `|y − ideal·2^e| ≤ (2e-6 + 4e-6·sumabs)·2^e`
   (twice the matvec relative term, for the longer sequential K accumulation); exact cases
   exact. F32 m-contiguous (attention) variant against an FP64 CPU product with the same
   bound. Tail tiles, count from io, and batch/grouping strides tested.
2. Model: native teacher-forced prefill over each oracle case with chunk sizes 1, 13 and the
   default must meet the **same** per-tensor bounds as decode (4× llama's FP32 error, ≥2e-6)
   for every captured intermediate at every token; logits at chunk-final positions within the
   logit bound; greedy agreement rule unchanged.
3. Serving pattern: prefill prompt then decode: continuation logits within bounds and greedy
   output through the server equal to libllama (`tools/check_session.py`).
4. Determinism: repeated prefill runs bit-identical; chunking changes results only within
   the bounds above.

## Measurement

Prefill throughput vs prompt length (128–3.2K), TTFT through the server, GEMM component
TFLOP/s per shape (GPU timestamps), against llama-server default and a fully FP32
llama-server control (`GGML_VK_DISABLE_MMVQ`, `_INTEGER_DOT_PRODUCT`, `_COOPMAT`, `_F16`,
F32 KV). Two runs; failures retained.

## Small-row plans — block 13d (specified 2026-09-22, before implementation runs; tiles superseded by 13e)

**Problem (measured):** a 128×128 tile costs as much for 1 row as for 128 rows. A
23-token chunk spends 422 of 431 ms GPU in GEMMs
([profile](../bench/data/2026-09-22-profile-baseline/p23.jsonl)).

**Design:**
- **Tiles:** `gemm.comp` is parameterized by its thread layout
  (TX threads along M × 256/TX along rows, TM×TN outputs per thread). Compiled tiles:
  128×128 (16×16 threads, 8×8), 128×64 (16×16, 8×4), 128×32 (32×8, 4×4) and
  256×16 (64×4, 4×4). BK stays 32 with the same two-level accumulation, so each output
  element's summation order depends only on the K range. For the same split-K choice,
  every tile gives bit-identical results.
- **Plans:** the runtime records one prefill command per *plan*. A plan is a tile plus
  the most chunk rows it serves, chosen from a fixed table: 16 → 256×16, 32 → 128×32,
  64 → 128×64, and the rest up to `--prefill-chunk` → 128×128.
- **Plan geometry:** each plan's dispatch grids, split-K choices (same 384-workgroup
  target) and partial-slot sizes use the plan's row count.
- **Chunk selection:** a chunk of n rows runs on the first plan with `rows ≥ n`. The
  table is subject to the crossover measurement below; any change is recorded before
  the gates run.
- **Safety:** `runChunk(commands, plan, chunk)` rejects chunks larger than the plan.

**Gates:**
1. **GEMM:** every tile passes the block-13a fixture gate (same bound). At equal split,
   every tile's output is bit-identical to 128×128 (fixture cases, with and without
   split-K). Every tile also passes the attention-style F32 test.
2. **Model:** `tools/verify_model.py --modes 0,1,13,29,60,512` meets the same bounds.
   The capture tool records one capture command per plan and follows `chunkFor`.
3. **Serving:** `tools/check_session.py` greedy equality.
4. **Measurement:** GEMM bench for all tiles at rows 1–512 (the crossovers); per-phase
   profile of 23-, 81- and 151-row chunks; serving benchmark, two runs.

**Plan decision (measured before any model gate, 2026-09-22):**
- **Measurement:** the tile sweep ([GEMM data](../bench/data/2026-09-22-small-prefill/),
  two runs) found 256×16 best for ≤16 rows, 128×32 best for 23–96 rows (and for
  151/192 rows), and 128×128 best at exact multiples of 128. **128×64 was never best
  and is dropped.**
- **Plan table:**

  | Rows | Tile |
  |---|---|
  | ≤16 | 256×16 |
  | ≤96 | 128×32 |
  | up to `--prefill-chunk` | 128×128 |

- **`chunkFor` policy:**
  - Full chunks of `--prefill-chunk` rows run on the last plan.
  - A shorter remainder r whose tail `r mod 128` fits a smaller plan runs as
    `r − tail` rows on 128×128, then the tail.
  - Anything else runs on the first plan that covers it.
- **Gate modes:** these exercise every plan. Mode 13 uses 256×16. Modes 29 and 60 use
  128×32 (with 256×16 tails). Mode 512 uses 128×32 for the 48-token case and
  128×128 + 128×32 for the 221-token case.

## Scalar-X GEMM — block 13e (specified 2026-09-22, after research, before productization)

**Research** ([report](../bench/2026-09-22-gemm-throughput.md)):
- The 128×128 kernel is **LDS-read bound**. Removing the per-k LDS reads more than
  doubles throughput; removing global loads changes little.
- It is **power-limited** under sustained load (~310–340 W vs a 339 W cap; the clock
  drops with random data).
- Wave32/VOPD does not help.

**Design** — tile 256 (M) × 32 (rows), 256 threads:
- **Waves:** a "wave" is a 64-thread group, `w = subgroupBroadcastFirst(tid/64)`. This
  is uniform for any subgroup size dividing 64, so it is correct for wave32 and wave64.
  The host rejects devices with `maxSubgroupSize > 64`.
- **Lanes:** lane `tid % 64` owns 4 M rows.
- **A path:** A tiles are decoded into LDS by the existing loaders (one row per thread,
  32 values). Each lane reads one conflict-free `vec4` per k.
- **X path:** wave w owns 8 X rows. X comes through a second, read-only binding of the
  activation buffer as wave-uniform `vec4` loads (4 consecutive k), which the compiler
  turns into scalar loads. There is no X in LDS.
- **Guards:**
  - rows ≥ n and k ≥ the K end read as 0 (uniform selects);
  - X addresses need `x_base`, `x_rs`, `x_bs` ≡ 0 (mod 4);
  - reads may extend to the containing 32-k block, which validation covers.
- **Accumulation:** per element, the same order as every existing tile (part over one
  32-wide K tile in k order, then total += part). At equal split-K the outputs are
  **bit-identical** to the current kernels. Measured: every benchmarked Q4_0 shape at
  100 and 512 rows, including under forced wave32.
- **Plans:** this tile beat the best 13d plan at every measured row count, so it
  **replaces all 13d tiles**. The 128×128, 128×32 and 256×16 kernels and the non-scalar-X
  code path were removed. The shipped `gemm_*.spv` are byte-identical to the verified
  scalar-X modules.
  - Plans now carry only a row count: 32, 64, 128, 256 and `--prefill-chunk`.
  - Each plan's split-K choice is sized for its row tiles.
  - `chunkFor` runs full chunks, then the remainder on the smallest covering plan.
- **Context:** must be a multiple of 32, so score rows are whole 32-k blocks and
  16-byte aligned. Other values are rejected at `Model.init`.

**Gates:**
1. **Component:** the GEMM fixture gate for every format, plus bit identity with the
   128×128 kernel at equal split (fixtures, and the attention-style F32 test).
2. **Model:** `verify_model` modes 0/1/13/29/60/512 within the same bounds.
3. **Serving:** greedy equality.
4. **Measurement:** sustained GEMM bench with random X (all formats); per-phase
   profile; serving benchmark, two runs.

## GEMM efficiency — block 13h (2026-09-23)

**Research** ([report](../bench/2026-09-23-gemm-efficiency.md)):

- The measured wave64 `v_fma_f32` peak is 63–66 TFLOP/s
  ([13g](../bench/2026-09-23-coopmat-research.md)). The 13e kernel runs at 22–24,
  about 35% of that.
- **Ablations** (constant-folding-safe; results invalid, timing only) found no single
  dominant limiter. Removing any one of dequant, LDS reads or X loads gains
  12–30%; removing all three gains 2.4×.
- **ISA** showed three costs:
  1. The fully unrolled X loads (8 rows × 32 k = 256 scalar registers) spilled
     137 SGPRs into VGPR lanes (`v_writelane`/`v_readlane`).
  2. SMEM X loads and LDS reads share `lgkmcnt`. Since SMEM can return out of order,
     the compiler waits `lgkmcnt(0)` before the FMAs of each K step, which serializes
     them.
  3. Q4_0 blocks (18 bytes, 2-byte aligned) were read through 2-byte-granular
     `word_at`, with branches.

**Design** (same bindings, push constants, tile 256×32, threads, LDS layout and
arithmetic):

- **X pipeline.** X is read one kb (4 k) ahead in a **non-unrolled** kb loop. Only the
  current and next 8×vec4 are live, so there are no SGPR spills.
- **Q4_0/Q4_1.** Each thread (one block) reads the block through the aligned words
  that contain it. Q4_0 takes 5 words; Q4_1 takes 5, or 6 when the block starts at
  2 mod 4.
  - `validate()` already bounds the tensor end rounded up to 4 bytes, so the reads
    stay in range.
  - The words are loaded **one tile ahead** into registers, and decoded after the
    barrier into LDS.
- **Q5_K.** The same one-tile-ahead register prefetch is applied to its five
  aligned 16-byte words.
- **Q6_K, F32.** Only the X pipeline changes. For F32, the partial final K tile
  (runtime K) is masked on the pipelined X.

**Accumulation is unchanged.** Per element: `part = fma chain over each 32-k tile in
k order`, then `total += part`, with the same decoded A values. **Outputs are
bit-identical** to 13e at equal split. The next section lists the evidence.

**Gates:**

1. **Component.** `zerv-gemm-bench --dump` compares full outputs byte for byte with
   the 13e kernels:
   - Q4_0 (7 tensors), Q4_1, Q5_K and F32;
   - 1/23/100/512 rows;
   - the natural split, and forced splits (`--k-chunk` 0 and 1024).
2. **Hardware tests.** GEMM fixtures in `bazel test //tests:gpu //tests:gpu_release_fast`
   (Debug and ReleaseFast).
3. **Model.** `verify_model` modes 0/1/13/29/60/512 must pass, and every captured
   tensor and logit file must be byte-identical to the 13f gate.
4. **Serving.** Outputs must be identical to the previous run; then two serving runs
   against llama-server.

### Part 2: wide tile (256×64)

**Measured** (tile sweep in the [13h report](../bench/2026-09-23-gemm-efficiency.md);
each tile with its own split-K rule):

- 16 X rows per 64-thread group (tile 256×64) is 1.08–1.17× faster than 256×32 at
  512 rows for every projection with M ≥ 5120.
- It is slower for M = 1024 (0.71×), because too few workgroups run.
- It is slower at ≤ 32 rows (0.37–0.72×), because half the tile is padding.
- It is mixed at 64–256 rows.

**Design:**

- `gemm.comp` gains a compile-time `XW`: 8 is the narrow tile (unchanged; the
  SPIR-V is byte-identical) and 16 is the wide tile.
- The wide tile reads X as `vec2` pairs through a third view of the same binding,
  in a non-unrolled pair loop. With 16 rows, `vec4` X would spill SGPRs.
- Per element, the k order and the two-level accumulation are unchanged. At equal
  split-K the wide tile's outputs are **bit-identical** to the narrow tile's.
- Modules: `gemm_{q4_0,q4_1,q5_k,q6_k}_w`. F32 (the small-M ssm projections and
  attention) always uses the narrow tile.
- Selection: `gemm.tileFor(M, plan_rows)` returns wide iff `plan_rows ≥ 512` and
  `M ≥ 4096`. This is static; there is no runtime autotuning, so outputs do not
  depend on timing.
- `splitChunk(M, rows, K, target, tile)` counts the tile's own row tiles with the
  same 384-workgroup target. At 512 rows this changes the split-K choices: for
  example ffn_down goes from 2 to 3 splits and attn_qkv from 1 to 2. That changes
  rounding for 512-row chunks, which must then pass the oracle gates.
- The runtime keys pipelines by (bank, format, tile); at most 40 GEMM pipelines.

**Gates:**

1. `gpu-test`: every quantized fixture case runs on both tiles. The wide outputs
   must equal the narrow outputs byte for byte (including the split-K pass), and
   the narrow outputs must meet the independent bounds.
2. Model: the existing oracle cases (≤ 221 tokens) never run a 512-row chunk. A new
   oracle case, `long-prefill` (546 prompt tokens + 16 generated;
   `tests/fixtures/model/qwen38-oracle-long.json`), runs one full 512-row chunk
   (wide tile) plus a remainder. `verify_model --fixture … --modes 0,512` must
   pass within the usual bounds.
3. Serving: two runs against llama-server. The 836- and 3223-token cases run full
   512-row chunks.

## DeltaNet scan latency — block 13j (specified 2026-09-23, before implementation)

**Measured.** One 512-row chunk spends 62 ms in `delta_b` (48 layers, about 2.5 µs
per row).

- One workgroup per value head (48 workgroups of 128 threads) scans the rows
  sequentially.
- Per row, the critical path holds:
  - 2 loop barriers, plus 9 more inside `wg_sum` (the gated RMS norm);
  - dependent global loads of q/k/v;
  - per-thread `sigmoid`/`softplus`/`exp` of the gate scalars;
  - two 128-long dependent dot-product chains (`kv`, `o`).
- Only the chains are inherent to the recurrence.

**Design.** Each expression below is unchanged; only the scheduling moves.

1. **Gated RMS norm → `gnorm_b`.** `y = ((o·inv)·norm_w)·silu(z)`, with
   `inv = 1/sqrt(wg_sum(o²)/128 + eps)`, does not feed the recurrence. It moves to a
   new kernel with one 128-thread workgroup per (head, row), groups = (48, plan rows).
   Workgroups for rows ≥ count return uniformly. It uses the same `wg_sum` tree, so
   the result is bit-identical. `delta_b` writes `o` as before; `gnorm_b` runs after
   a compute barrier.
2. **Gate scalars.** `beta = sigmoid(beta_raw)`, `sp = softplus(alpha + dt)`,
   `g = a·sp` and `decay = exp(g)`, with the same expressions.
   - They are computed in a parallel prologue per segment of at most 512 rows
     (thread j takes rows j, j+128, …).
   - `beta_out`/`softplus_out`/`g_out` are written to act as before.
   - `beta` and `decay` are kept in LDS arrays; a barrier separates the prologue
     from the scan.
3. **q/k double buffer.** `qs`/`ks` get two LDS buffers. Row r+1's q_j, k_j and v_j
   are loaded into registers right after row r's barrier and written to the other
   buffer at the start of row r+1.
   - This needs one barrier per row. The buffer written at row r was last read at
     row r−2, and every thread has passed row r−1's barrier, so it has finished
     row r−2.
   - At a segment boundary, an extra barrier comes before the prologue overwrites
     the gate arrays.

The recurrence (`kv += (s·decay)·k`, `delta = (v − kv)·beta`,
`s = s·decay + k·delta`, `o += s·q`) is textually unchanged, with the same unrolled
loops.

**Gates:**

1. Model captures (every tensor and logit file) byte-identical to the 13h gates, for
   both the default and the long oracle, all modes.
   - The fma-contraction hazard from 13c applies: if restructuring changes the
     backend's fusion, results may differ.
   - In that case the change is analysed and must still pass the oracle bounds.
2. Serving outputs identical.
3. Profile: the `delta` phase time, then two serving runs.

## Explicit f16 prefill mode — block 14 (specified 2026-09-24, before implementation; implemented, gate 2 open — [evidence](../bench/2026-09-24-f16-prefill.md))

**Why.** llama-server's default prompt path on this card is f16 WMMA
([precision ladder](../bench/2026-09-23-llama-precision.md)). At matched FP32, zerv is
already 1.9–3.8× faster. An on-par comparison with llama's *fast* path needs zerv to
offer the same arithmetic class, as an explicit, measured option.

**Arithmetic (matches llama.cpp's `matmul_quant_f16` with f32 accumulation).** For
prefill projections in the f16 mode:

- `Y[t][m] = Σ_k f16(W[m][k]) · f16(X[t][k])`, accumulated in the device's f16×f16→f32
  WMMA (16×16×16), with the accumulation running across K in k order, one 16-k slice
  per WMMA.
- `f16(·)` is the driver's f32→f16 conversion. It must round to nearest-even on the
  device; a hardware test checks this.
- Weights are rounded once from their exact FP32 dequantized values:
  - Q4_0: `f16(d·(q−8))`;
  - Q4_1: `f16(fma(d, q, m))`;
  - Q5_K: `f16(fma(d·sc, q, −dmin·mn))`.
- The WMMA f32 accumulation is not IEEE-exact
  ([13g](../bench/2026-09-23-coopmat-research.md)). Results are deterministic, but not
  bit-reproducible on the CPU.

A prototype variant that keeps the weight integers exact (f16 `q−8`) and applies a
per-block f32 scale is more accurate (2.1e-4 vs 2.7e-4 normalized error against the
FP32 GEMM on ffn_gate). It is also 17% slower (49 vs 59.5 TFLOP/s), so it is not the
f16 mode. It remains documented as an alternative.

**Scope.**

- Uses f16: Q4_0, Q4_1 and Q5_K projections with `M ≥ 4096` and `M % 128 == 0`, in
  plans whose row count is a multiple of 128 (128/256/512).
- Stays FP32: everything else. That covers attn_k/v (M = 1024), the F32
  ssm_alpha/beta, attention, DeltaNet, norms, the KV cache, the output head (a matvec
  on the last row) and all of decode.
- Split-K is not used in the f16 GEMM (k_chunk = 0). The prototype matches llama
  without it on every in-scope shape.

**Kernel.** `gemm_f16.comp` (new module per format):

- tile 128 (M) × 128 (rows), 256 threads, 4 waves (wave64) of 64 × 64;
- per 32-k step, A is dequantized to f16 in LDS and X is converted f32→f16 into LDS;
- 2 × (2×4) WMMAs per wave;
- rows ≥ n read row n−1, and their results land in the plan's unused rows (plan rows
  are a multiple of the tile);
- stores are `coopMatStore`, column-major with stride `y_rs`.

**Device and interface.**

- `--prefill-precision fp32|f16` (default fp32), `Options.prefill_precision`.
- The f16 mode requires `Device.open(.{ .cooperative_matrix = true })`, subgroup
  size 64 and the f16×f16→f32 16×16×16 configuration. If these are missing, `init`
  fails with `UnsupportedDevice`; there is no silent fallback.

**Gates.**

1. **Component.** Every quantized matvec fixture case, plus random rows (row counts
   not a multiple of 128), against a CPU reference with the *same* f16-rounded inputs
   (exact FP64 sum): `|y − ref| ≤ 2⁻¹⁶·Σ|w·x| + 1e-6`, which is generous for the WMMA
   accumulation. The f32→f16 conversion must be RNE (hardware test).
2. **Model quality**, measured with `tools/prefill_quality.py` on the default and long
   oracles: zerv f16's mean and worst logit / `l_out-63` errors ≤ **llama nof16's**
   (the matched-arithmetic reference), and no argmax flips against FP64. llama
   default's 13–24 flips are not the bar.
3. **Serving.** Two runs at 23/81/836/3223 tokens against llama default and llama
   nof16.

## f16 GEMM, wave32 kernel with f16 X — block 16b (specified 2026-09-24, before implementation; implemented for Q4_0, gates 1–4 passed — [evidence](../bench/2026-09-24-gemm-f16x.md))

Evidence: [GEMM f16 lab](../bench/2026-09-24-gemm-f16-lab.md). The lab kernel k64 is
bitwise-equal to `gemm_f16.comp` on every Q4_0 shape and 1.25–1.36× faster in an
interleaved race.

**Arithmetic.** Unchanged from block 14, bit for bit:

- Every output keeps the same sequence of 16-k WMMA steps in k order. Tile shape, wave
  size and data path cannot change a bit.
- Q4_0 weights: `f16(d·(q−8))`, computed as the f16 bit pattern `0x6400|q`
  (= 1024 + q, exact), minus 1032 (exact), times d in one RNE f16 multiply. The exact
  product is rounded once, which equals rounding the exact FP32 product. f16
  subnormal results depend on the device's FP16 denormal mode: this device preserves
  them, and the component test covers scales that produce subnormals.
- X: the GEMM reads an **f16 copy of its input**, written by the producing kernel as
  `float16_t(y)` of the FP32 value it stores (the same RNE `v_cvt_f16_f32` the block-14
  kernel applies after loading X). This is bitwise-neutral end to end.

**Kernel** `gemm_f16x.comp` (Q4_0 only in this step):

- Wave32, with required subgroup size 32 and full subgroups.
- 8 waves in a 2 (M) × 4 (rows) layout of 64 × 64 wave tiles; tile 128 (M) × 256 (rows).
- BK = 64: two Q4_0 blocks per row per LDS stage, two LDS stages, one barrier per stage.
- A is dequantized once per workgroup into LDS (row stride 72 halves).
- B fragments are loaded straight from the f16 X in global memory: two 16-byte loads
  per fragment. Lane l holds row (l mod 16), k values 16q … 16q + 15. This layout was
  verified bitwise in the lab.
- Rows ≥ n read row n − 1, and their results land in unused plan rows, as in block 14.
- Stores: `coopMatStore` column-major with stride `y_rs`.

**Selection** (`gemm.f16Kernel`):

| Case | Kernel |
| --- | --- |
| f16 mode, Q4_0, M ≥ 4096, M % 128 == 0, K % 64 == 0, plan rows % 256 == 0 | `gemm_f16x` |
| Other f16-eligible projections (Q4_1, Q5_K, the 128-row plan) | `gemm_f16` (block 14, f32 X) |

- The rule is a function of (format, M, K, rows) for this device class. It becomes a
  device-keyed table once a second tile configuration is measured.

**f16 X copies.**

- Region `x16` of the activation arena: `rows × ffn` halves, reused by one input at a
  time (A.h, A.gated, A.sw). Its base is 16-byte aligned; the row stride is the
  consumer's K in halves.
- Producers: the prefill `norm` (A.h, attention and FFN inputs), `gate` (A.gated) and
  `swiglu` (A.sw) kernels. The f16 mode compiles `_h` variants of them with 16-bit
  storage. They write the copy only when a consumer in the next GEMM phase uses
  `gemm_f16x`; otherwise the FP32 variants run.
- Decode and the FP32 mode are unchanged.

**Validation** (`gemm.validateF16x`), all before recording:

- no split-K, `flags == 0`, `a_group == 1`, format rules as `validate`;
- `x_base % 8 == 0` and `x_rs % 8 == 0` (halves);
- `x_base + (rows − 1)·x_rs + K ≤` act halves; `y` stores cover whole 256-row tiles.

**Device.** The f16 mode additionally needs `VK_EXT_subgroup_size_control` with 32 in
the compute range and `computeFullSubgroups`. `Model.init` fails with
`UnsupportedDevice` otherwise; there is no silent fallback.

**Gates.**

1. **Component** (`tests/gpu_gemm_f16.zig`):
   - `gemm_f16x` is **bitwise-equal to `gemm_f16`** on the valid rows (with f32 X
     holding the same values; rows ≥ n are unread plan rows, which `gemm_f16` may skip
     by whole 128-row tiles) on random Q4_0 weights, including scales that produce f16-subnormal
     weights and a row tail (n < 256), with K ∈ {64, 512, 5120};
   - the block-14 FP64 bound holds;
   - rows beyond the plan are untouched.
2. **Producers:** each `_h` kernel's f16 output equals `f16(y)` of its FP32 output
   bitwise, and its FP32 output is bitwise-unchanged.
3. **Model:** f16-mode logits and captured intermediates are **bitwise-identical**
   before and after this change on the default and long oracles.
   `tools/verify_model.py --precision f16` still passes.
4. **Serving:** served f16-mode outputs are byte-identical before and after. TTFT is
   measured interleaved against the previous build and llama-server (3223-token and
   12k prompts, warm-up ≥ 10 s, junction temperature recorded).

## Native `gemm_f16x` machine code (specified 2026-09-24, before integration; implemented, gates 1–4 passed the same day — [evidence](../bench/2026-09-24-gemm-f16x-isa.md))

**What.** For the `gemm_f16x` (Q4_0) kernel, the model can create the pipeline from our own
RDNA3 machine code (a RADV pipeline binary) instead of compiling the SPIR-V. Interface,
bindings, push constants, grid, validation and producers are unchanged; only the machine
code differs.

**Arithmetic.** Bit-identical to the SPIR-V kernel by construction (same WMMA sequence per
output in k order, same f16 dequantization ops) and by test (component gate below).

**Artifacts** (`src/model/native/`): `gemm_f16x_q4_0.bin` (RADV 26.2.3 pipeline binary),
`.key` (its binary key), `.global` (the driver global key it is valid for),
`gemm_f16x_q4_0.s` (the generated assembly), `manifest.json` (hashes of all of them, of the
generator, of the SPIR-V the placeholder was compiled from, tool and driver versions).
Regenerated only by `tools/build_native_gemm.py` (needs the GPU, RADV and clang); ordinary
builds embed the files.

**Knob** `--gemm-code spirv|native` (`Model.Options.gemm_code`), f16 mode only:

| Value | Behaviour |
| --- | --- |
| `native` (default since the gates passed) | device opened with `pipeline_binaries`; `gemm_f16x` from the binary when the global key matches, else from SPIR-V |
| `spirv` | the previous behaviour: no pipeline-binary extension, SPIR-V only |

On a key mismatch (other Mesa build, other GPU, driver debug options) the SPIR-V kernel runs
and the server says so at startup (`gemm_f16x: native|spirv (reason)`). `Model.gemm_native`
reports what was created. Other kernels and the FP32 mode are unaffected.

**Gates.**

1. Component: `bench/isa_lab/sweep.py` PASS (all shapes, tails, alignments, subnormals).
2. Driver: `tests/gpu_*.zig` — a kernel from the binary equals the SPIR-V kernel bitwise on
   a small GEMM; a wrong expected global key falls back (`native` false, SPIR-V results);
   malformed binary options are rejected before driver calls.
3. Model: f16-mode intermediates and logits bitwise identical with `native` and `spirv`
   (`tools/verify_model.py --precision f16` hashes unchanged).
4. Serving: served f16-mode outputs byte-identical; TTFT interleaved against `spirv` and
   llama-server (3223-token and 12k prompts), junction temperature recorded.

## f16 GEMM, 32-row tile for short plans — block 18c.2 (2026-09-25; implemented, gates passed — [evidence](../bench/2026-09-25-multiuser.md))

**Problem (measured).** Short prompts run on the 128-row plan, where every f16 projection
uses the block-14 `gemm_f16` 128 × 128 tile. A 5120-wide output (ffn_down, lin_out,
attn_out) then has M / 128 = 40 workgroups for 48 WGPs: one 4-wave workgroup per WGP, no
latency hiding. ffn_down takes 61 ms of the 182 ms that a 128-row chunk costs.

**Arithmetic.** Unchanged from block 14, bit for bit. `gemm_f16.comp -DSMALLM=1`
(`gemm_f16m_{q4_0,q4_1,q5_k}`) has a tile of 32 (M) × 128 (rows), with 4 subgroups of
32 × 32 side by side along the rows. Each output element is the same WMMA chain in the same
k order. The A tile is dequantized by threads 0–63 (32 rows, two halves each). X staging,
the row tail (rows ≥ n read row n − 1) and the stores are block 14's.

**Selection** (`gemm.f16KernelFor(v, M, K, plan rows, small_tile)`), resolved when the
plans are recorded:

| Case | Kernel |
| --- | --- |
| `small_tile`, plan rows ≤ 128 (`f16m_max_rows`), f16-eligible, M / 128 × rows / 128 < 64 (`f16m_grid`) | `gemm_f16m` |
| otherwise | `gemm.f16Kernel` (block 14 / 16b) |

Measured bounds, from interleaved `race_profile.py` runs on the real model:
- **40 workgroups gain.** ffn_down 61.3 → 51.8 ms, lin_out 17.6 → 15.3 ms, attn_out
  5.5 → 4.2 ms.
- **80 or more lose.** With a threshold of 192, ffn_in went 46.5 → 70.7 ms, lin_in
  22.2 → 28.7 ms and attn_in 5.8 → 8.4 ms.
- **At 256 rows the tile loses to `gemm_f16x`.** The whole prefill went 865 → 1008 ms.

The 32 × 128 tile stages 4× more X per output element, so it pays only where the device
would otherwise idle.

**Knob.** `Model.Options.f16_small_tile` (default true); CLI `--f16-small-tile on|off`.
`off` is the previous kernel set.

**Validation** (`gemm.validateF16m`): `validateF16`'s checks (eligible shape, whole
128-row tiles, no split-K, extents); grid M / 32 × rows / 128.

**Gates.**
1. **Component** (`tests/gpu_gemm_f16.zig`): `gemm_f16m` is bitwise equal to `gemm_f16`
   on the valid rows for Q4_0, Q4_1 and Q5_K, with 1 and 2 row tiles, row tails 1..256
   and K 512..5120. Words around the output are untouched. The selection rule and the
   validation are tested at their boundaries.
2. **Model:** the long oracle in the f16 mode (`verify_model.py --precision f16
   --gemm-code native`):
   - mode 128, `--f16-small-tile on` against `off`: logits, captured tensors and serving
     logits byte-identical;
   - modes 512 and 256: byte-identical to the captures before this change;
   - FP32 oracles (default and long): byte-identical to before.
3. **Serving:** `run_concurrent.py --reference` gives outputs identical to the recorded
   solo reference.

## Fused prefill attention — block 16a (2026-09-24; implemented, gates passed — [evidence](../bench/2026-09-24-flash-attention.md))

Replaces the materialized path (score GEMM, row softmax, P·V GEMM through a
24 × chunk × context FP32 score region) with one kernel, `flash.comp`. FP32 in both
prefill precisions; the f16 mode changes only projections.

- **Semantics** (unchanged): row r of a chunk at position p0 + r attends to keys
  0..p0 + r of its KV head (head / 6), scale 1/16, softmax, then P·V; output `pregate`.
- **Kernel.** Workgroup = one KV head and 8 query rows; six 64-thread groups, one per
  query head of that KV head. Per 128-key tile:
  - Scores: lane l owns keys 2l and 2l + 1. Q rows are wave-uniform and read as scalars.
    K is read from the cache directly (coalesced), 4 dims ahead of the FMAs. Each score
    sums 32-dim partial fma chains.
  - Causal and live-key mask (keys ≥ p0 + count are never read: undefined contents),
    row maxima and sums with subgroup reductions; with 32-wide subgroups the two halves
    of a group combine through LDS.
  - Online softmax: m' = max(m, tile max), a = exp(m − m'), l' = l·a + Σ exp(s − m').
  - P to LDS per group; lane l accumulates head dims 4l..4l + 3 of the 8 rows over the
    tile's keys (V rows read directly, 4 keys ahead), then O = O·a + O_tile.
  - Output O / l.
- **Host rules.** Compute subgroups of 32 or 64 with arithmetic, aligned to 64-thread
  groups (`Subgroup.computeArithmeticFrom(32)`, as the scalar-X GEMM assumes). The kernel
  exists once per KV buffer, like the other KV kernels.
- **VRAM.** No score region: 24 × chunk × context × 4 bytes (384 MiB at 512 × 8192,
  1.4 GiB at 512 × 28k) leave the activation arena, which also removes the arena's 4 GiB
  limit on context × chunk.
- **Gates.**
  1. Component (`tests/model_gpu.zig`): against FP64 with a first-order bound (dot
     products, exponentials, n-term accumulation, rescaling), p0/count across row-block
     and tile edges, a context not a multiple of 64, NaN in cache entries past the live
     keys and in query rows past the count, untouched outputs past the count.
  2. Model: `tools/verify_model.py` FP64 gates, default oracles modes 1/13/512/512:17,
     long oracle modes 13/128/512/512:300.
  3. Long-prompt serving with the answer checked (`bench/workloads/long-v1.json`).
