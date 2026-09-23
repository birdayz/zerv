# What precision is llama-server really using? (block 14 research, 2026-09-23)

Question: what arithmetic runs in llama-server's prompt path on this card? This
decides what an apples-to-apples comparison with zerv means.

## Source (pinned llama.cpp b29c606e28 = the installed build 10964)

These files were fetched into
`third_party/llama.cpp/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-vulkan/`;
each git blob hash matches `research-tree.json`:

| File | SHA-256 |
| --- | --- |
| `ggml-vulkan.cpp` | c84f6746… |
| `mul_mm.comp` | 2f850b66… |
| `mul_mm_funcs.glsl` | 6e08fa7e… |
| `mul_mmq.comp` | fb3c2a42… |
| `mul_mmq_funcs.glsl` | 5be74d2c… |
| `mul_mmq_shmem_types.glsl` | 06df962d… |
| `quantize_q8_1.comp` | 9f16a578… |

**`ggml_vk_mul_mat_q_f16`, around line 9495.** The device has `KHR_coopmat` but not
coopmat2. For a quantized src0 and an F32 src1, the code sets `y_non_contig`
("coopmat1: force f32->f16 conversion so the f16 B-type quant pipeline is used").
That disables `quantize_y`, the Q8_1 integer-dot path. Activations are converted to
**f16**.

**`ggml_vk_get_mul_mat_mat_f16acc`, line 9133.** For quant types with coopmat1, f16
accumulation is used iff `fp16 && coopmat_acc_f16_support && prec == DEFAULT`.

- The driver reports the f16×f16→f16 configuration
  ([properties](bench/data/2026-09-23-coopmat-research/properties.json)), and fp16 is
  available.
- `llama-graph.cpp` sets `GGML_PREC_F32` only for attention KQ, flash attention, and
  some other architectures' FFN/output projections, never for Qwen3.5.

So the weight matmuls accumulate in **f16**.

## Runtime confirmation

Setting `GGML_VK_PIPELINE_STATS=matmul` prints the name of every matmul pipeline
ggml compiles; ggml compiles them on demand, so this lists exactly the ones used.

- **Default config** (FA on, ub512; [log](bench/data/2026-09-23-llama-precision-ladder/pipelines-default-probe.log)):
  `matmul_quant_f16_f16acc_aligned` for every quantized weight, and
  `matmul_f32_f32_aligned` for the F32 tensors.
- **Toggle runs** ([logs](bench/data/2026-09-23-llama-precision-ladder/)):
  - `GGML_VK_DISABLE_F16` → `matmul_quant_f16` (f16 in, f32 accumulation);
  - `GGML_VK_DISABLE_COOPMAT` → `matmul_q4_0_q8_1`, `matmul_q4_1_q8_1`,
    `matmul_q5_k_q8_1` (Q8_1 activations, integer dot);
  - both COOPMAT and INTEGER_DOT_PRODUCT disabled → `matmul_quant_f32_f16acc`.

The pipeline names match the code path, but what each pipeline does numerically is
inferred from the ggml source, not measured. The 13g hardware measurements apply to
the WMMA accumulation.

## Speed ladder (same GGUF, FA on, ub512, 3 repeats; median TTFT ms / decode tok/s)

Runs: [ladder](bench/data/2026-09-23-llama-precision-ladder/summary.json), plus the
default and zerv rows from
[delta-scan run1](bench/data/2026-09-23-delta-scan-serving-run1/summary.json).

| Config | Matmul arithmetic | 23 | 81 | 836 | 3223 |
| --- | --- | --- | --- | --- | --- |
| llama default | f16 X, f16-dequantized W, **f16 accumulate**, WMMA | 162 | 367 | 1202 | 3396 |
| llama `DISABLE_F16` | f16 X, f16 W, f32 accumulate, WMMA | 161 | 364 | 1182 | 3316 |
| llama `DISABLE_COOPMAT` | Q8_1 X, integer dot (exact int32 blocks) | 190 | 431 | 1428 | 4452 |
| llama no coopmat, no int dot | f32 X, f16 accumulate, no WMMA | 190 | 440 | 1576 | 4988 |
| llama `fp32-full` | FP32 everywhere | 249 | 685 | 2938 | 9888 |
| **zerv** | FP32 (sequential fma, two-level accumulation) | **75** | **181** | **1371** | **5201** |

Decode rates: llama 40–46 tok/s in every config; zerv 49–54.

## Consequences

- **Correction.** Earlier reports (13a–13j) called llama-server's default prompt path
  "Q8_1 int8 activations". That is wrong on this card: it is f16 WMMA with f16
  accumulation. Q8_1 is used only when cooperative matrices are disabled.
- **Matched FP32** (zerv default vs `llama-fp32-full`): zerv is 3.3× / 3.8× / 2.1× /
  1.9× faster at 23 / 81 / 836 / 3223 tokens.
- **Matched fast path.** Being on par with llama's default requires an explicit
  zerv f16 prefill mode, and it must be *at least as accurate*.
  - llama's default rounds X to f16, rounds W = d·(q−8) to f16, and accumulates in
    f16 over all of K.
  - zerv can keep W exact (the integers q−8 are exact in f16, and the block scale is
    applied in f32), round only X, and accumulate each ≤32-k block in f32 WMMA before
    f32 scaling.
  - Quality must be measured against the FP64 oracle for both engines' prefill paths.

## Quality ladder: batched prefill against the FP64 oracle (same tokens)

**Method.** `tools/prefill_quality.py` runs the new pinned-libllama capture
`tests/reference/llama_batch_capture.c`:

- settings: n_batch 2048, n_ubatch 512, FA on, KV as the server config;
- the fixture's exact token sequences, teacher-forced;
- logits at every position, and `l_out-63` (final hidden state) for every token.

zerv is scored from its `verify_model` mode-512 captures with the same code. Its logits
exist only at chunk-final and decode positions; its hidden states exist for every
token. Metrics are normalized L2 against FP64; argmax is compared with FP64 wherever the
margin exceeds `compare_model.py`'s near-tie rule, and no compared position was a near
tie. Raw data: [prefill-quality](bench/data/2026-09-23-prefill-quality/).

The cases are short-nothink (48 tokens), long-think (221) and long-prefill (562, two
ubatches):

| Engine / config | Mean logit NL2 | Worst logit NL2 | Mean `l_out-63` NL2 | Argmax ≠ FP64 |
| --- | --- | --- | --- | --- |
| llama default (f16 accumulate) | 3.2e-2 / 2.9e-2 / 2.6e-2 | 0.95 | 3.4–4.2e-2 | 0 / **13** / **24** |
| llama nocoopmat (Q8_1) | 2.5e-2 / 2.0e-2 / 1.9e-2 | 0.99 | 2.6–3.2e-2 | 0 / 6 / 10 |
| llama nof16 (f16 in, f32 accumulate) | 9.1e-4 / 7.5e-4 / 6.1e-4 | 1.4e-2 | 0.8–1.1e-3 | 0 / 0 / 0 |
| llama fp32-full, batched | 1.7e-4 / 1.5e-4 / 1.2e-4 | 2.1e-3 | 1.7–2.3e-4 | 0 / 0 / 0 |
| **zerv FP32** (512-row chunks) | 5.1e-7 / 5.9e-7 / 4.3e-7 | 6.3e-6 | 7.3–9.7e-7 | 0 / 0 / 0 |

**Reading:**

- **llama default.** The prompt path is about 5·10⁴× less accurate than zerv's FP32
  path, and it changes the greedy token at 4–6% of teacher-forced positions.
- **Where the error comes from.** Most of it is the f16 *accumulation*: nof16 is 30–40×
  better at the same speed (3316 vs 3392 ms at 3223 tokens).
- **llama's batched FP32 path is about 250× less accurate than zerv**, and also far
  worse than llama's own n_batch = 1 path (≈5e-7 logits,
  [fixture](../../tests/fixtures/model/qwen38-oracle.json)). The cause has not been
  investigated. Candidates: its batched GatedDeltaNet formulation and flash attention.
- **Apples to apples.** The fair fast-path target is **llama nof16**, llama's most
  accurate fast config (1182 ms at 836 tokens, 3316 ms at 3223). A zerv f16 mode must
  meet its speed with error at most its 6–9e-4. zerv's design keeps weights exact
  and accumulates in f32, so it should land far below that.
