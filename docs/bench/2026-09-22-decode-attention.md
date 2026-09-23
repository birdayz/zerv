# Split-K decode attention (block 13b) — 2026-09-22

**Question:** does splitting decode attention over 64-key chunks
([spec](../specs/decode-attention.md)) remove the long-context decode slowdown without
losing accuracy?

**Result:** yes.
- **Long-context decode:** a decode step at position ~3.2K takes **21.9 ms GPU instead
  of 45.5 ms**.
- **Serving decode at 836 prompt tokens:** 34.8 → **45.7 tok/s**, now 11% faster than
  llama-server FA (41.3). At 81 tokens: 42.5 → 45.8 tok/s (llama 41.4). At 23 tokens:
  48.6 → 50.6 tok/s (llama 44.0).
- **Accuracy:** all oracle gates pass. On real tensors the new summation orders are
  *more* accurate than the old fused kernel (mean error −5% at ≤221 keys, −36% at
  1000–3000 keys).
- **Unaffected:** prefill and TTFT (small environmental drift, see below). Greedy
  outputs are identical to before on all four serving cases.

## Change

The fused one-workgroup-per-head attention kernel is replaced by three passes:
1. **scores:** GQA-shared, 64 keys per workgroup, plus per-chunk maxima.
2. **P·V partials:** global max from the chunk maxima, so every `exp(s − m)` matches
   the old kernel's arithmetic.
3. **Combine:** fixed-order.

The old `attn` shader was removed; all 20 other model shaders compile byte-identically.
The score kernel's K loads are unrolled by 16 (same per-head d order). The model outputs
were verified bit-identical to the non-unrolled version on every capture file
(`third_party/model-native/2026-09-22-attn13b-gate{2,3}`).

## Correctness

| Gate | Evidence | Result |
|---|---|---|
| Component: 3 passes vs FP64, p ∈ {0,1,62,63,64,65,127,1000,4095}, context 4096, stale keys = 1e30, NaN scratch | `tests/model_gpu.zig` (Debug + ReleaseFast) | worst 0.0012 of the first-order bound; p=0 exact; gates exact |
| Model oracle, modes 0/1/13/512 | [gate-run3](data/2026-09-22-decode-attention/gate-run3.json) (final), run1/run2 kept | decode: worst/bound 0.494 short, 0.281 long (was 0.755/0.260); logits 3.23e-6/2.38e-6 (was 3.53e-6/2.50e-6); greedy 48/48, 221/221 |
| Serving greedy equality | [session-run2](data/2026-09-22-decode-attention/session-run2.json) | JSON + SSE, both cases |
| Determinism | capture-vs-plain / reset replay in every mode | bit-identical |

**Accuracy investigation (quality, not just the gate).** The first version (gate-run1,
plain sequential chunk sums) raised the long-case worst `attn_gated` sample from 7.9e-6
to 1.3e-5 (bound 3.0e-5), with means +1.8%. Two analyses followed:
- **Model-level error distributions** (`tools/error_distribution.py`,
  [long](data/2026-09-22-decode-attention/errdist-long-think.json),
  [short](data/2026-09-22-decode-attention/errdist-short-nothink.json)) showed a shift
  in the max only. Upstream tensors moved similarly through propagation.
- **An FP32 emulation of the summation orders on the real oracle tensors**
  ([order study](../../bench/attention_order_study.py),
  [data](data/2026-09-22-decode-attention/order-study.json)) isolated the kernel's own
  arithmetic:

| Variant | Mean error, real K/V ≤221 keys | Mean error, 1000–3000 keys |
|---|---:|---:|
| Old fused kernel | 7.60e-7 | 9.64e-7 |
| Split, sequential chunk sums | 7.51e-7 | 6.44e-7 |
| **Split, tree chunk sum + 4 interleaved accumulators + blocked-8 combine (adopted)** | **7.24e-7** | **6.18e-7** |

With the adopted orders, model-level mean error is below the pre-13b kernel for every
compared tensor, e.g. long-case logits 6.85e-7 → 6.80e-7 and short-case 7.50e-7 →
7.08e-7.

## Performance

**Per-phase GPU profile** (`zerv-model-profile`):
- Pre-13b data at position 3247 comes from [baseline](data/2026-09-22-profile-baseline/p3223.jsonl).
- The post-unroll decode profile at ~3.2K was re-measured after the 13d build, which has
  an identical decode path: [profile-final-p3223](data/2026-09-22-decode-attention/profile-final-p3223.jsonl).
- The intermediate unroll-1 profiles are
  [profile-unroll1-p23](data/2026-09-22-decode-attention/profile-unroll1-p23.jsonl) and
  [profile-unroll1-p3223](data/2026-09-22-decode-attention/profile-unroll1-p3223.jsonl).

| Decode step | Before (fused) | Split, unroll 1 | Split, unroll 16 (final) |
|---|---:|---:|---:|
| Position ~40: step GPU ms | 22.21 | 22.40 | 21.44 |
| Position ~40: attention phases ms | 1.32 | 1.49 | 0.45 |
| Position ~3.2K: step GPU ms | 45.48 | 22.66 | 21.91 |
| Position ~3.2K: attention phases ms | 25.25 | 1.69 | 0.89 |

The unroll factors 8, 16 and 32 were measured in the same session. Scores took 0.265,
0.200 and 0.190 ms at position ~40, and 0.355, 0.359 and 0.372 ms at ~3.2K; 16 was
chosen.

**Serving** (median of 3; run1 / repeat). The repeat reused the run1 binary
(`5f8b8f78…`, `--zerv-binary`).
[run1](data/2026-09-22-decode-attention-serving-run1/summary.json),
[repeat](data/2026-09-22-decode-attention-serving-repeat/summary.json):

| Case (prompt tok) | Engine | TTFT ms | Decode tok/s | Total s |
|---|---|---:|---:|---:|
| short-nothink (23) | zerv (split-K attention) | 450 / 443 | 50.6 / 50.5 | 0.71 / 0.70 |
| short-nothink (23) | llama-server FA ub512 | 183 / 232 | 44.0 / 43.3 | 0.48 / 0.53 |
| short-nothink (23) | llama-server fully FP32 | 308 / 316 | 42.8 / 42.6 | 0.61 / 0.62 |
| short-nothink (23) | zerv before (13a run1) | 437 | 48.6 | 0.71 |
| decode-think (81) | zerv (split-K attention) | 487 / 483 | 45.8 / 45.8 | 6.05 / 6.04 |
| decode-think (81) | llama-server FA ub512 | 371 / 423 | 41.4 / 41.4 | 6.54 / 6.58 |
| decode-think (81) | llama-server fully FP32 | 750 / 759 | 40.5 / 40.6 | 7.04 / 7.05 |
| decode-think (81) | zerv before (13a run1) | 485 | 42.5 | 6.49 |
| medium-prompt (836) | zerv (split-K attention) | 2947 / 2900 | 45.7 / 45.7 | 5.72 / 5.68 |
| medium-prompt (836) | llama-server FA ub512 | 1241 / 1247 | 41.3 / 41.3 | 4.31 / 4.32 |
| medium-prompt (836) | llama-server fully FP32 | 2986 / 2988 | 40.0 / 40.1 | 6.16 / 6.15 |
| medium-prompt (836) | zerv before (13a run1) | 2906 | 34.8 | 6.56 |
| long-prompt (3223) | zerv (split-K attention) | 11031 / 10861 | (52.8 / 52.6)* | 11.19 / 11.02 |
| long-prompt (3223) | llama-server FA ub512 | 3453 / 3453 | (45.7 / 45.7)* | 3.63 / 3.63 |
| long-prompt (3223) | llama-server fully FP32 | 9933 / 9945 | (43.0 / 43.1)* | 10.12 / 10.13 |
| long-prompt (3223) | zerv before (13a run1) | 10887 | (25.2)* | 11.21 |

\*The long-prompt case generates only 8 tokens. The harness's rate,
(tokens−1)/(last−first delta), is inflated when leading tokens emit no delta, so these
are not valid decode rates. Use the per-phase profile instead: 21.9 ms/step at ~3.2K,
i.e. ≈45.7 tok/s GPU-bound, about **parity** with llama-server at that length rather
than a win.

**Environment drift:** TTFT rose for every engine compared with the 13a runs 1–2 hours
earlier. llama-server at 23 tokens went from 162 to 183/232 ms, and its FP32 control
from 250 to 308/316 ms; zerv went from 437 to 450/443 ms. During the repeat a Plex Media
Scanner was running on the host (load average ~1.5, 34 sessions); the GPU was idle
outside our jobs. Decode rates, which are GPU-bound, were stable. The benchmark does not
control host background load; TTFT comparisons across runs carry this uncertainty.

## Failures and retained attempts

- **The first repeat attempt did not run.** The harness rebuilds zerv from source, and
  13d edits had started, so the build failed.
  [log](data/2026-09-22-decode-attention/failed-repeat-build.log). `bench/run_serving.py`
  gained `--zerv-binary` so repeats reuse the exact benchmarked binary; the repeat above
  used it.
- Gate run1 (plain split) and gate run2 (adopted orders, unroll 1) are kept beside
  run3.

## Commands

```sh
python3 tools/compile_model.py --output-dir .tools/model-shaders-13b   # then copy into src/model/shaders
zig build gpu-test [-Doptimize=ReleaseFast]
python3 tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-22-a \
  --work-dir third_party/model-native/2026-09-22-attn13b-gate3 --report docs/bench/data/2026-09-22-decode-attention/gate-run3.json
python3 tools/error_distribution.py --oracle-dir third_party/model-oracle/2026-09-22-a --case long-think \
  --run before=.../2026-09-22-prefill-gate1/long-think-0 --run split=.../2026-09-22-attn13b-gate1/long-think-0 \
  --run final=.../2026-09-22-attn13b-gate2/long-think-0 --output docs/bench/data/2026-09-22-decode-attention/errdist-long-think.json
python3 bench/attention_order_study.py --oracle-dir third_party/model-oracle/2026-09-22-a --output docs/bench/data/2026-09-22-decode-attention/order-study.json
python3 tools/check_session.py --output docs/bench/data/2026-09-22-decode-attention/session-run2.json
zig build model-profile-build -Doptimize=ReleaseFast -Dcpu=native && zig-out/bin/zerv-model-profile MODEL 8192 512 3223 32
python3 bench/run_serving.py --output docs/bench/data/2026-09-22-decode-attention-serving-run1
python3 bench/run_serving.py --zerv-binary third_party/serving-bench/2026-09-22-decode-attention-serving-run1/zerv \
  --output docs/bench/data/2026-09-22-decode-attention-serving-repeat
```
