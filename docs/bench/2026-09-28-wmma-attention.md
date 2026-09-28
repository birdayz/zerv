# WMMA prefill attention (block 16c), 2026-09-28

Question: prefill attention was 56% of an 88k-token prefill at 20 TFLOP/s
([research](../research/2026-09-28-prefill-attention.md)). How much does a cooperative-matrix
(WMMA) kernel in the f16 prefill mode's arithmetic class gain, at what quality cost?
Spec: [prefill.md "WMMA prefill attention"](../specs/prefill.md).

## Setup

- RX 7900 XTX, Mesa RADV (host 26.2.3 and the test runtime), Qwen3.8-27B-Q4_0.
- zerv `third_party/multiuser/zerv-wmma1` (sha256 `730c8d5e…`); the FP32-attention
  baseline is the 2026-09-27 sweep binary `zerv-sweep1`.
- f16 prefill, f16 KV, one sequence.
- Component benchmark: `zerv-attn-bench SPV fp32|wmma6|wmma3 ROWS P0 [REPS] [BATCH] [WARM_S]`
  (new, `bench/attn_bench.zig`: one attention layer, random data, sustained dispatches after
  a warm-up).

## Kernel iterations (component, 512 rows at position 16,384; TFLOP/s = 4·rows·keys·256·24 / time)

| Version | ms | TFLOP/s | Note |
| --- | --- | --- | --- |
| FP32 `flash.comp` (f16 KV) | 9.94 | 21.1 | the shipped kernel |
| v0: transposed LDS staging with 16-bit stores | — | 24 (in-model) | 128 workgroups; scalar stores |
| v1: 16-byte copies in the cache layout (row-major B) | 5.32 | 39.4 | 6 heads/workgroup; 3 heads: 35.4 |
| **v2: 8×8 register-transposed staging (column-major B)** | **4.94** | **42.4** | shipped |

v2 ablations (results wrong, timing only), ms:
- no global loads 4.41;
- no softmax 4.56;
- no S WMMAs 3.26;
- no P·V WMMAs 3.17;
- no barriers 4.25.

The WMMAs take ~3.4 of 4.9 ms, about 2× their ideal issue time. ACO rebuilds the Q fragments
with 16-bit moves (87 `v_mov_b16` per tile loop) and reads each B fragment with four
`ds_load_b64` (its NIR caps LDS loads at 8 bytes). The native GEMM had the same limits;
hand-scheduled ISA is the next step (as for `gemm_f16x`).

Negative results kept:
- 3 heads per workgroup (twice the workgroups) is slower than 6: GQA sharing beats
  occupancy.
- `packHalf2x16` for Q raised the worst error from 1.5% to 4.6% of the bound. It truncates
  on this driver; `packFloat2x16(f16vec2)` rounds to nearest-even.

## Correctness and quality

- **Component gate:** worst error 1.5% of the bound (test runtime), 2.2% (host driver).
  `bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills` and
  `tools/zerv_build.py --test-host-gpu` pass.
- **Model quality** (`tools/py tools/kv_quality.py --output docs/bench/data/2026-09-28-wmma-attention/quality
  --runs zerv-f16,zerv-f16-p16,zerv-f16-wmma,llama-f32,llama-f16`; 36,000-token prefix, 257
  next-token distributions; [data](data/2026-09-28-wmma-attention/quality/)):

| Pair | KL mean | median | p99 | max | top-1 agree |
| --- | --- | --- | --- | --- | --- |
| zerv f16 prefill ‖ **+ WMMA attention** | **6.8e-7** | 4.7e-8 | 1.3e-5 | 1.9e-5 | 100% |
| zerv FP32 prefill ‖ f16 prefill (projections) | 9.5e-7 | 1.1e-7 | 9.5e-6 | 2.2e-5 | 100% |
| zerv FP32 prefill ‖ f16 prefill + WMMA attention | 1.5e-6 | 1.3e-7 | 1.7e-5 | 4.0e-5 | 100% |
| llama f32 KV ‖ llama f16 KV (its own f16 attention) | 3.2e-4 | 3.4e-5 | 3.2e-3 | 3.8e-3 | 100% |

Mean NLL of the true next tokens: 0.749310 (FP32 prefill), 0.749317 (f16), 0.749260 (f16 +
WMMA); llama 0.746608 / 0.746066. The WMMA attention's change is ~465× smaller than
llama's f32→f16 change.

## Serving: single-user cold prefill, 1k–88k tokens

```
tools/py bench/run_long_context.py --output docs/bench/data/2026-09-28-wmma-attention/sweep \
  --zerv-binary third_party/multiuser/zerv-wmma1 --engines "zerv-f16@kv-type=f16,prefill-attention=wmma" \
  --paragraphs 5,20,80,160,320,440 --repeats 2 --max-tokens 16 --context 94208
```

Baseline and llama-server from [the 2026-09-27 sweep](data/2026-09-27-prefill-sweep/); two
timed requests each, spread < 0.5%.

| prompt tokens | zerv FP32 attention | **zerv WMMA attention** | llama-server | WMMA / FP32 | llama / zerv WMMA |
| --- | --- | --- | --- | --- | --- |
| 1,034 | 0.87 s | **0.86 s** | 1.29 s | 0.985 | 1.49× |
| 4,034 | 3.10 s | **2.99 s** | 4.10 s | 0.963 | 1.37× |
| 16,034 | 14.05 s | **12.60 s** | 17.02 s | 0.897 | 1.35× |
| 32,034 | 33.07 s | **27.55 s** | 38.65 s | 0.833 | 1.40× |
| 64,034 | 86.33 s | **65.48 s** | 93.49 s | 0.759 | 1.43× |
| 88,034 | 139.83 s | **100.65 s** | 150.11 s | 0.720 | 1.49× |

llama-server's own attention (`GGML_VK_PERF_LOGGER`, [logs](data/2026-09-28-llama-attention/)):
- 7.2 s of its 32k prefill;
- 59.6 s of its 88k prefill.

## Interpretation and next steps

- zerv's f16 prefill with WMMA attention is 1.35–1.49× faster than llama-server at every
  length. At 88k that was 1.07×.
- `--prefill-attention wmma` stays opt-in until the packed (parallel) variant exists.
- **Next, by size:**
  1. Hand-scheduled ISA for the attention loop (b128 LDS fragment loads, no fragment
     rebuilds, pipelined K/V staging; the `gemm_f16x` method). The loop is ~2× off its
     WMMA ceiling.
  2. The packed variant for `--parallel` > 1.
  3. Short-prompt costs: the DeltaNet scan (10.6% at 4k), Q5_K `lin_out` and Q4_1
     `ffn_down` on the native GEMM, conv.
