# KV cache precision: `--kv-type f16` (block 17c, 2026-09-24)

Question: can the attention KV cache be stored in f16 as an explicit option, at what
quality cost, and what does it buy in decode speed and context?
[Spec](../specs/model.md) ("KV precision"), [research](../research/kv-precision.md).

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV; Qwen3.8-27B-Q4_0 (sha256 `ede16c7b…`).
- llama.cpp for comparisons: the pinned libllama/ggml (b29c606e28) through
  `tests/reference/llama_batch_capture.c`, and llama-server of the same build for
  serving.
- Implementation: `KV16` variants of the five KV kernels (`qkprep`, `qk_b`,
  `attn_scores`, `attn_pv`, `flash`). They store `f16(x)` and load halves converted
  exactly to FP32; all arithmetic is unchanged. Offsets are in elements
  (`layout.KvType`). The device enables `storageBuffer16BitAccess` alone
  (`gpu.Device.Options.storage16`). Every f32 module is byte-identical to the previous
  build (`cmp` after `tools/compile_model.py`).

## Gate 1: f32 unchanged (passed)

- `tools/verify_model.py`, default oracle, modes 0/1/13/512/512:17: all 28 capture and
  logits files byte-identical to the previous build. Long oracle, modes
  0/13/128/512/512:300: all 14 files byte-identical.
- `zerv-spec-check` 11/11, `zerv-prefix-check` gate passed, `zerv-mtp-check` scenario C
  3/3 and the FP64 MTP reference passed on both sequences
  ([data](data/2026-09-24-kv-precision/)).

## Gate 2: components (passed; `zig build gpu-test`, 29/29)

- The f16 writers store exactly RNE-f16 of the FP32 values, bit for bit, for `qkprep`
  (decode and verify rows) and `qk_b` (prefill rows). The V inputs carry ties to even,
  subnormal results, ties at the subnormal scale, overflow to ±inf and signed zeros. So
  RADV's conversion is round-to-nearest-even with subnormals, equal to Zig's
  `@floatCast`.
- Split-K decode attention and fused prefill attention with f16 KV, against FP64 over
  the same f16 values: within the f32 tests' bounds. Fused worst error / bound: 8.7e-4
  (f32: 7.8e-4).
- `zerv-spec-check f16`: 11/11 (verify ≡ decode, bitwise, with f16 KV: speculation stays
  lossless). `zerv-prefix-check fp32 device f16`: restore ≡ split, gate passed.

## Gate 3: oracle quality against llama, matched ([data](data/2026-09-24-kv-precision/))

`tools/prefill_quality.py`: normalized L2 against FP64 over every position; argmax
compared where the FP64 margin is not a near tie. llama `fp32-kvf16` = FP32 matmul
environment with f16 KV (the matched reference); `fp32-full` = the same with f32 KV.
Logits (mean / max) and `l_out-63` (mean / max):

| Case | llama fp32, f32 KV | llama fp32, f16 KV | zerv f32 KV (decode) | zerv f16 KV (decode) | zerv f16 KV (prefill 512, l_out) |
| --- | --- | --- | --- | --- | --- |
| short-nothink (48) | 1.66e-4 / 7.31e-4; 2.26e-4 / 8.92e-4 | 1.67e-4 / 7.92e-4; 2.28e-4 / 8.58e-4 | 4.5e-7 / 2.3e-6; 6.3e-7 / 2.1e-6 | 1.64e-4 / 7.38e-4; 2.25e-4 / 8.95e-4 | 2.27e-4 / 8.95e-4 |
| long-think (221) | 1.47e-4 / 1.23e-3; 2.18e-4 / 1.59e-3 | 1.47e-4 / 1.20e-3; 2.18e-4 / 1.57e-3 | 4.7e-7 / 4.1e-6; 6.9e-7 / 5.2e-6 | 1.47e-4 / 1.27e-3; 2.18e-4 / 1.64e-3 | 2.15e-4 / 1.47e-3 |
| long-prefill (562) | 1.17e-4 / 2.13e-3; 1.71e-4 / 2.94e-3 | 1.18e-4 / 2.03e-3; 1.73e-4 / 2.74e-3 | 5.9e-7 / 2.1e-5; 8.4e-7 / 2.6e-5 | 1.17e-4 / 2.14e-3; 1.71e-4 / 2.87e-3 | 1.76e-4 / 3.21e-3 |

- Argmax: no flips anywhere (48/48, 221/221, 562/562 and all prefill positions).
- **Means: zerv f16 equals llama's within 1% everywhere.** zerv f32 KV is about 300×
  more accurate than llama even with llama's f32 KV: llama's "fp32" configuration keeps
  about 1.5e-4 of error from paths that the environment switches do not cover.
- **Worst token: zerv f16 exceeds llama `fp32-kvf16` by 4–6% in decode**, and by 17% on
  one prefill `l_out` maximum (long-prefill). llama's own worst-token values move ±8%
  between its f32 and f16 KV runs, in both directions. The statistic is dominated by
  which rounding hits the most sensitive token. **As declared, gate 3's max clause is
  not met.** The gate is not changed after the fact.

## Gate 4: long-context KL (passed; [data](data/2026-09-24-kv-precision/long-kl/))

`python3 tools/kv_quality.py --output docs/bench/data/2026-09-24-kv-precision/long-kl`:
long-v1's document text (37,758 raw tokens, no template), 36,000-token prefix, then 256
teacher-forced decode steps (llama in its single-token path too). KL over the full
vocabulary for 257 next-token distributions:

| Pair | KL mean | median | p99 | max | top-1 agree |
| --- | --- | --- | --- | --- | --- |
| zerv f32 KV ‖ zerv f16 KV | **3.4e-7** | 3.1e-8 | 5.7e-6 | 1.8e-5 | 100% |
| llama f32 KV ‖ llama f16 KV | 3.2e-4 | 3.4e-5 | 3.2e-3 | 3.8e-3 | 100% |
| llama f16 ‖ llama f16 (repeat) | 0 | 0 | 0 | 0 | 100% |
| zerv f32 ‖ llama f32 (scale reference) | 9.0e-4 | 8.6e-5 | 1.3e-2 | 2.2e-2 | 99.2% |

- Mean NLL of the true next tokens: zerv f32 0.749294, zerv f16 0.749309; llama f32
  0.746608, llama f16 0.746066.
- zerv's f16 KV changes the distribution about 900× less than llama's does. llama's f16
  KV run also changes arithmetic (its f16 flash-attention path), not only storage. llama is
  deterministic here (the repeat is identical).
- A second run (after the flash fix below) gave byte-identical logits for both zerv
  configurations ([long-kl-r2](data/2026-09-24-kv-precision/long-kl-r2/)).

## Performance

- **Found and fixed:** the first f16 flash kernel made prefill attention 2.1× slower
  (8k tokens: 1,531 against 720 ms; 36k-token prefill 83.4 against 67.7 s). Cause: it
  converted K at the load, so the one-step-ahead K prefetch waited on its load at once.
  Reading 32-bit words without the prefetch change gave no gain (1,528 ms). Keeping raw
  words in the prefetch registers and converting at use gives 709 ms, faster than f32.
  `zerv-model-profile MODEL 8704 512 8192 4 fp32 f16`.
- Decode kernels at position 8,194: scores 0.68 → 0.39 ms, P·V 0.71 → 0.48 ms per step.
- `zerv-kv-quality` at 36k (FP32 prefill): prefill 68.1 → 67.3 s; decode 28.1 → 24.9 ms
  per step.

## Gate 5: serving

long-v1, 37.8k tokens, `--no-prompt-cache`, context 38,400, 1 repeat
([data](data/2026-09-24-kv-precision/kv-long-v1/)); llama from the same day's
[run](data/2026-09-24-flash/long-v1-r2/):

| Engine | needle TTFT s | decode-38k TTFT s | decode-38k tok/s (256) |
| --- | --- | --- | --- |
| zerv f16 prefill, f32 KV | 46.0 | 46.0 | 34.8 |
| zerv f16 prefill, **f16 KV** | 45.1 | 45.2 | **39.5** |
| zerv f16 prefill, f16 KV, 3 drafts | 45.5 | 45.6 | **84.6** |
| llama-server default (f16 KV) | 46.6 | 46.8 | 35.8 |
| llama-server MTP 3 | 50.3 | 50.4 | 76.3 |

All six zerv outputs are byte-identical to the f32-KV outputs (needle answered
correctly; the 256-token summary identical).

long-v2, 75.5k tokens (`bench/make_long_workload.py --chars 215000`, a new needle
HERON-2958 / 314; context 87,040; `--no-prompt-cache`; 1 repeat; zerv binary
`2f46606918ecd264…`, before the decode attention changes of
[decode-attention-long](2026-09-24-decode-attention-long.md);
[data](data/2026-09-24-kv-precision/kv-long-v2/)):

| Engine | needle TTFT s | decode-long-v2 TTFT s | needle tok/s (23) | decode tok/s (256) | VRAM loaded MiB |
| --- | --- | --- | --- | --- | --- |
| zerv f16 prefill, f16 KV | 119.9 | 119.6 | 32.0 | 30.6 | 20,463 |
| zerv f16 prefill, f16 KV, 3 drafts | 120.5 | 120.6 | **64.9** | **51.4** | 21,231 |
| llama-server default (f16 KV) | **115.7** | **115.5** | 33.7 | 32.4 | 20,186 |
| llama-server MTP 3 | 127.9 | 126.7 | 61.4 | 49.9 | 21,387 |

- All four answer the needle correctly; zerv's two outputs are identical to each other in
  both cases; llama's summary differs between its default and MTP runs.
- **Losses at 75.5k:** llama's prefill is 3.5% faster, and its plain decode 6% faster.
  zerv with 3 drafts is still the fastest decode configuration (51.4 against 49.9).
  Both zerv losses are open items in `TODO.md` (prefill attention kernel; decode attention,
  where the combine pass was fixed afterwards: 64k step 30.6 → 27.0 ms).

`--context max` with f16 KV ([data](data/2026-09-24-context-max/kv-f16/results.json);
f32 in [context max](2026-09-24-context-max.md)). Every configuration loaded and served:

| Configuration | f32 KV | f16 KV |
| --- | --- | --- |
| default (1 GiB reserve) | 44,896 | **88,576** |
| host snapshots | 53,760 | 106,112 |
| no prefix cache, no speculation | 60,128 | 119,840 |
| default, reserve 0 | 52,000 | 102,592 |
| no prefix cache, no speculation, reserve 0 | 67,808 | 135,104 |

## Decision

- `--kv-type f16` is an explicit option. It doubles the context, makes decode 13% faster
  at 38k (plain decode now ahead of llama-server there), costs nothing in prefill, and
  its measured quality loss is far below llama's default.
- **The default stays f32.** Gate 3 as declared (worst token no worse than llama's
  matched configuration) is not met, by 4–6% (17% on one prefill maximum), within what
  looks like noise but is not proven to be. For long contexts, f16 is the recommended
  setting (`--kv-type f16 --context max`).
