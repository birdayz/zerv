# KV cache precision (block 17c, 2026-09-24)

Question: can the attention KV cache be stored in f16 (and later q8_0) as an explicit
option, what does it cost in quality, and what does it buy in context and decode speed?
Status: research for the spec in [model.md](../specs/model.md), "KV precision";
implemented and measured in [the report](../bench/2026-09-24-kv-precision.md).

## What the cache holds today (source: `src/model/model.comp`, `flash.comp`, `layout.zig`)

- 16 trunk attention layers (+ the MTP layer with speculation), 4 KV heads × 256 dims.
  Per token: K and V, 2 × 1024 FP32 = 8 KiB per layer, 128 KiB for the trunk.
- Layout per cache: K `[kv head][dim][context]` (positions contiguous, for the decode
  score kernel's one-key-per-thread reads), V `[kv head][context][dim]`.
- Writers: `qkprep` (decode, verify, MTP draft rows: RMS-normed and roped K, raw V) and
  `qk_b` (prefill chunks and the batched MTP catch-up).
- Readers: `attn_scores` (K), `attn_pv` (V), `flash` (prefill; K as `vec2` pairs of keys,
  V as `vec4` of dims).
- Nothing else touches the cache: snapshots hold the recurrent state only, and the KV
  kernels bind the KV buffer alone at binding 2.

## Value range (measured on the oracle cases, zerv FP32 captures, 2026-09-24)

From `Kcur_roped` and `Vcur` of every token of the default oracle's two cases (48 and
221 tokens, 16 layers; `third_party/model-native/2026-09-24-hostmem-default/*-0`):

| Tensor | max \|x\| | smallest non-zero \|x\| | share below f16's normal range (6.1e-5) |
| --- | --- | --- | --- |
| K (normed, roped) | 23.04 | 3.2e-7 | 4–6e-5 |
| V | 113.0 | 3.0e-7 | 5–6e-5 |

- f16's range (65504) covers both with a factor of more than 500. K is bounded by its
  RMS norm (the per-dim weight times at most √256 = 16).
- f16 has an 11-bit significand: storage rounding error ≤ 2⁻¹¹ relative (4.9e-4), about
  0.03 absolute at the largest V values. Subnormal values are rare and tiny either way.

## llama.cpp (the competitor; pinned b29c606e)

- `--cache-type-k/-v` (`-ctk/-ctv`): f32, f16 (default), bf16, q8_0, q4_0, q4_1,
  iq4_nl, q5_0, q5_1. llama-server's default is f16 for both.
- Quantized V needs flash attention. Our previous Ollama deployment ran q4_0 KV with FA
  ([previous deployment](2026-09-23-previous-deployment.md)).
- Its Vulkan flash attention (`flash_attn.comp`, `flash_attn_base.glsl` in
  `third_party/llama.cpp/b29c606e…/ggml/src/ggml-vulkan/vulkan-shaders/`) converts f16 K/V
  to f32 (or to f16 for the coopmat paths) on load and accumulates in f32 for the scalar
  path.
- So its default serving numbers already include f16 KV. zerv's FP32 KV is the more
  exact configuration; at 38k tokens llama's plain decode is 1.03× faster
  ([flash report](../bench/2026-09-24-flash-attention.md), section 5).

## f16 conversion on this device

- The conversion (`float16_t(x)` in GLSL, SPIR-V OpFConvert) has an implementation-defined
  rounding mode unless decorated. Block 14 relies on the same conversion and checks with a
  hardware test that RADV rounds to nearest even (`tests/gpu_gemm_f16.zig`).
- 16-bit storage (`storageBuffer16BitAccess`, `GL_EXT_shader_16bit_storage`) makes each
  half individually addressable, so the prefill writer (many positions at once, K
  positions contiguous) needs no read-modify-write of shared words. RDNA3/RADV supports
  it; zerv enables it today only for cooperative matrices, so the KV option must request
  it on its own.
- Loads of `float16_t` / `f16vec2` / `f16vec4` convert exactly to FP32; all attention
  arithmetic stays FP32 (same kernels, same summation order).

## Expected effects (estimates, to be measured)

- VRAM: KV 128 → 64 KiB per token (+8 → 4 KiB for the MTP cache). With the default
  knobs, `--context max` would go from about 44.9k to about 80k tokens (context-dependent
  bytes per token: KV plus decode/verify scores and split-K partials).
- Decode at 38k: the score and PV kernels read 4.8 → 2.4 GB of KV per token. The 38k
  plain-decode step takes 28.7 ms (34.8 tok/s) against 19.8 ms at short context. Reading
  4.8 GB takes at least 5.2 ms at 920 GB/s, so halving saves at least 2.6 ms, and up to
  about 4.5 ms if the attention time scales with bytes. That is 34.8 → about 38–41 tok/s
  (estimate).
- Prefill: flash reads K/V from L2/DRAM per tile; halving bytes helps the long-prompt
  attention part.
- Quality: storage rounding only. Scores q·k change by about 2⁻¹¹ relative per term,
  averaged over 256 dims.

## Alternatives considered

- **bf16:** same bytes, 3 fewer significand bits than f16; no benefit given the range
  above.
- **q8_0 (32-value blocks, f16 scale + int8):** 34 bytes per 32 values (≈ 0.53 of f16).
  Per-block scales on K `[dim][context]` would run along positions, which breaks the
  per-token write and makes appending a token a block update. A different K layout would
  be needed. It is a later step, after f16 is measured.
- **q4_0:** as q8_0, with a larger quality loss; the old Ollama setup used it for 96–128k.

## Measurement plan (becomes the spec's gates)

1. Component: the cache holds exactly `f16_RNE(x)` of the FP32 K and V (hardware test
   through `qkprep` and `qk_b`), and the f16 attention kernels equal FP64 attention over
   the same f16 values within the decode-attention bound.
2. Oracle quality, matched: `tools/prefill_quality.py` style errors against FP64 for zerv
   with f16 KV, and for llama with FP32 arithmetic and f16 KV (the matched reference);
   zerv must not be worse.
3. Long context: KL(FP32-KV ‖ f16-KV) of next-token distributions on real text after a
   37k-token prefix (256 teacher-forced decode steps), for zerv and for llama (the same
   tokens, `tests/reference/llama_batch_capture.c` in decode mode). Report mean, p99 and
   max KL and top-1 agreement; zerv's KL must not exceed llama's by more than noise.
4. Speed and capacity: 38k decode (plain and speculative) and TTFT, `--context max`.
