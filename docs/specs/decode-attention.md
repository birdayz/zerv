# Split-K decode attention — block 13b

Status: **implemented and verified** 2026-09-22. [Evidence](../bench/2026-09-22-decode-attention.md).

## Why (measured)

The [per-phase profile](../bench/2026-09-22-prefill.md#where-prefill-time-goes-per-phase-gpu-profile)
of one decode step at position 3247 is 45.5 ms GPU. Of that, **25.3 ms is the fused
attention kernel** (16 layers), while the whole K/V read at that length is ≈0.4 GB
(≈0.5 ms at full bandwidth). Serving decode falls from 48.6 tok/s at short context to
25.3 tok/s at 3.2K, while llama-server stays at 45.7.

Why the current kernel is slow: one workgroup per query head gives 24 workgroups on
96 CUs. Each thread's P·V is a serial chain over all keys, and each KV head's cache is
read by 6 workgroups.

## Semantics (unchanged)

For query head h (KV head h/6), position p, and keys j = 0..p:
- `s_j = (q_h · k_j)/16`
- `m = max_j s_j`
- `e_j = exp(s_j − m)`
- `o = (Σ_j e_j v_j) / Σ_j e_j`

Then `pregate = o`, `gates = sigmoid(gate_h)` and `gated = o · gates`. All values are
FP32. The model spec leaves reduction orders free; correctness is judged by the gates
([model spec](model.md)).

## Design

Keys are split into chunks of C = 64 (`chunks = ceil(context/64)`; the live chunk count
is `ceil((p+1)/64)`, read from io). Three kernels replace `attn`; grids are sized for
the full context, and chunks at or past p+1 exit uniformly.

1. **scores** (grid chunks × 4 KV heads, 64 threads, one key per thread). Each
   workgroup stages the 6 query heads of its KV head in shared memory. For each of the
   6 heads it computes `s_j = Σ_d q[d]·k_j[d]` in d order, then `·(1/16)`, and writes
   `scores[h][j]`. It also writes the chunk maximum `amax[h][c]`. Each K column is read
   once for all 6 heads.
2. **pv** (grid chunks × 4 KV heads, 256 threads, one output dimension per thread):
   - `m_h` = max of `amax[h][0..live)`. Max is order-independent, so `m_h` is exactly
     the global max.
   - `e_j = exp(s_j − m_h)` for the chunk's keys, and 0 past the live keys.
   - `apart[h][c][t] = Σ_{j∈c} e_j·v_j[t]`, using four interleaved accumulators (key j
     goes to lane j mod 4, each lane in key order), combined as (l0+l1)+(l2+l3).
   - `asum[h][c]` is a halving-tree sum of the 64 (zero-padded) e_j.
   - Each V row is read once for all 6 heads, and rows past the live keys are never read.
3. **combine** (24 workgroups, 256 threads):
   - `acc = Σ_c apart[h][c][t]` and `total = Σ_c asum[h][c]`. Each is summed in chunk
     order in blocks of 8 (block sums, then a running total).
   - `o = acc/total`, followed by the gate products.

**Summation orders chosen by measurement:** the orders in pass 2 and pass 3 were
chosen with an FP32 emulation study on the real oracle tensors
(`bench/attention_order_study.py`). It covers real K/V up to 221 keys and
concatenated real K/V at 1000/3000 keys:

| Variant | Mean error, real ≤221 keys | Mean error, 1000–3000 keys |
|---|---:|---:|
| Old fused kernel | 7.60e-7 | 9.64e-7 |
| Plain split | 7.51e-7 | 6.44e-7 |
| Chosen orders | 7.24e-7 | 6.18e-7 |

The result is deterministic, uses no atomics, and has no rescaling step (the global
max is known before exponentiation). Scratch: `amax`/`asum` hold 24·chunks words each
and `apart` holds 24·chunks·256 words (3.1 MiB at context 8192) in the activation arena.
Prefill attention is unchanged.

Alternatives considered:
- **Flash-decoding with per-split local maxima** (llama.cpp
  `flash_attn_split_k_reduce.comp` at b29c606e, retained under `third_party/llama.cpp/`).
  It needs `exp(m_c − M)` rescaling, which adds a rounding step per chunk. The
  two-pass global max avoids it, at the cost of one extra dispatch.
- **Keeping one workgroup per head with more threads:** still 24 workgroups.

## Correctness gates (declared before implementation)

1. **Component (GPU test, driver-free parts on CPU):** random FP32 q, gate, K and V caches
   at context 4096, positions p ∈ {0, 1, 62, 63, 64, 65, 127, 1000, 4095}, compared with
   an FP64 CPU reference of the formula above.
   - Per element, `|o − o₆₄| ≤ Σ_j p_j·(|v_j[t]| + |o₆₄[t]|)·(2·S + (2·|x_j| + C + chunks + 16)·u) + 1e-30`.
     Here `S = max_j (1/16)·γ₂₅₆·Σ_d |q_d k_j[d]|`, `x_j = s_j − m`, `u = 2⁻²⁴` and
     `γ_n = n·u/(1−n·u)`. This is the first-order forward-error bound of the score dot
     product, the exponent (Vulkan requires `exp` within 3 + 2|x| ulp), normalization
     and the chunked sums.
   - p = 0 must give `o = v_0` exactly.
   - Gate outputs: `|g − σ₆₄(x)| ≤ (4 + 2|x|)·2⁻²³·g`, and `gated` equals the FP32 product
     of the returned `o` and `g` bit for bit.
   - Refinement: the bound above (the `|o₆₄|`, `|x_j|` and gate terms) was refined from
     the first draft before any test run, after checking the Vulkan precision
     requirement for `exp`.
   - Stale cache entries beyond p must not affect the result: the test fills them with
     huge values.
2. **Model:** `tools/verify_model.py --modes 0,1,13,512` with all intermediates, meeting
   the same bounds as block 10/13a. The long case decodes positions 0..220, so 1–4 live
   chunks are exercised.
3. **Serving:** `tools/check_session.py` greedy equality with libllama (JSON and SSE).
4. **Determinism:** the model tool's repeated-run and capture-vs-plain bit identity.

## Measurement

- Per-phase profile at positions ≈40 and ≈3247 (`zerv-model-profile`).
- Serving benchmark (two runs) against llama-server FA ub512 and the FP32 control.
- The expected short-context decode must not regress; the short-context attention
  phase is 1.32 ms today.

## Score accumulation accuracy — block 13f (specified 2026-09-22, before implementation; implemented and verified, [evidence](../bench/2026-09-22-accumulation-accuracy.md))

**Evidence** (research data: [data](../bench/data/2026-09-22-gemm-accuracy/)):
- **Decode vs prefill at model level.** Per-layer error distributions of the 13c gate
  captures show decode *less* accurate than prefill from the first attention layer on.
  At layer 3, `attn_pregate` mean normalized L2 is 8.8e-7 decode vs 3.6e-7 prefill,
  with decode inputs (Qcur, Kcur_roped, Vcur) equally or more accurate.
- **Projections are not the cause.** Per operation, the decode matvec is the most
  accurate projection scheme on real weights and activations (mean relative error
  ~2.1e-9 vs ~5–6e-9 for the prefill GEMM; `bench/accumulation_study.py`).
- **The score dot product is.** An FP32 emulation of decode attention on the real
  short-case Q/K/V of all 16 attention layers (`bench/score_accumulation_study.py`)
  compares two score accumulations:

  | Score accumulation | Mean output error | p90 |
  |---|---:|---:|
  | Single 256-term chain (as implemented) | 7.66e-7 | 1.56e-6 |
  | Two-level: 32-term parts, then a running total, as the prefill GEMM | **2.19e-7** | **4.18e-7** |

  Two-level is better in 5386 of 6144 cases.

**Change:** the decode scores kernel accumulates `q·k` in eight 32-d parts added in
order to a running total (the prefill GEMM's structure). Loads stay batched. Everything
else is unchanged.

**Gates:**
1. **Component:** the GPU test's FP64 bound still passes, and its measured worst ratio
   is recorded.
2. **Model:** `verify_model` modes 0/1/13/29/60/512 within bounds. Decode-mode error
   distributions (`tools/error_distribution.py` vs the 13c captures) must improve for
   `attn_pregate` and downstream tensors.
3. **Serving:** greedy equality.
4. **Speed:** no decode regression (per-phase profile at ~40 and ~3.2K positions).
