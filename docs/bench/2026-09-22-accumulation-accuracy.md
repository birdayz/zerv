# Accumulation accuracy (block 13f) — 2026-09-22

**Question:** where does zerv lose FP32 accuracy relative to the FP64 oracle, and can
it be recovered without costing speed?

**Result:**
- **The premise of the block was wrong.** It targeted the prefill GEMM's long
  accumulation chains, but model-level captures show prefill more accurate than decode.
- **The real weak point** is the decode attention score dot product: a single
  256-term fma chain.
- **Fix:** accumulate it in two levels, as the GEMM does. Decode attention error drops
  ~2.3×, and decode logits error now matches llama.cpp's own FP32 error on average:
  short case 7.08e-7 → **4.45e-7** (llama 4.75e-7), long case 6.80e-7 → **4.71e-7**
  (llama 4.63e-7).
- **Cost:** none measurable (decode 20.09 vs 20.10 ms).
- **Gates:** all oracle gates pass; greedy equality holds.

## Research

Data: [`data/2026-09-22-gemm-accuracy/`](data/2026-09-22-gemm-accuracy/).

1. **Model level:** error distributions of decode vs prefill captures from the 13c gate
   ([errdist-modes-short](data/2026-09-22-gemm-accuracy/errdist-modes-short.json)).
   Prefill (chunks 13 and 60) is more accurate than decode on every tensor, e.g.:

   | Tensor | Decode mean | Prefill mean | llama |
   |---|---:|---:|---:|
   | `l_out` | 4.09e-7 | 3.24e-7 | 2.94e-7 |
   | logits | 7.08e-7 | 3.76e-7 | 4.75e-7 |

2. **Per-operation projection accuracy**
   ([`bench/accumulation_study.py`](../../bench/accumulation_study.py),
   [data](data/2026-09-22-gemm-accuracy/accumulation-study.json)). This is an FP32
   emulation of each scheme on real Q4_0 weights (dequantized exactly) and real FP64
   activations, as relative error against the FP64 dot product:

   | Scheme | ffn_gate (K=5120) | ffn_down (K=17408) |
   |---|---:|---:|
   | Prefill GEMM (32-term fma parts + running total) | 5.1e-9 | 6.2e-9 |
   | GEMM with a third level (candidate) | 2.9e-9 | 2.8e-9 |
   | Decode matvec (rounded products, per-lane chains, 64-lane tree) | 2.2e-9 | 2.0e-9 |
   | Matvec with fma (candidate) | 2.0e-9 | 1.95e-9 |

   The matvec is the most accurate, so projections do not explain decode's model-level
   deficit.
3. **Per-layer localization** (layers 0–3, decode vs prefill): errors are equal or
   better in decode until the first attention layer. There, `attn_pregate-3` is
   8.8e-7 in decode vs 3.6e-7 in prefill, with decode's Q/K/V inputs more accurate.
   Decode stays worse afterwards.
4. **Score accumulation**
   ([`bench/score_accumulation_study.py`](../../bench/score_accumulation_study.py),
   [data](data/2026-09-22-gemm-accuracy/score-study.json)). An FP32 emulation of the
   13b decode attention on the real short-case Q/K/V of all 16 attention layers:

   | Score accumulation | Mean output error | p90 |
   |---|---:|---:|
   | One 256-term fma chain (as implemented) | 7.66e-7 | 1.56e-6 |
   | Two-level (32-term parts) | **2.19e-7** | **4.18e-7** |

   Two-level is better in 5386/6144 cases.

The first emulation run produced O(1) errors, which was a bug in the study, not the
kernel: the imported `fma` helper takes the accumulator first. It was fixed before any
conclusion was drawn.

## Change

`attn_scores` (the decode split-K pass 1) accumulates each 32-d part as an fma chain in
d order and adds the parts in order to the total. K loads are batched by 32, replacing
the batches of 16. Only this shader changed.

## Correctness and quality

| Gate | Evidence | Result |
|---|---|---|
| Component: 3-pass decode attention vs FP64 (context 4096, positions across chunk edges) | `tests/model_gpu.zig` | pass |
| Model oracle, modes 0/1/13/29/60/512 | [gate-run1](data/2026-09-22-gemm-accuracy/gate-run1.json) | all pass. Decode worst/bound: short 0.494 → 0.283; long 0.281 → 0.400 (one sample) |
| Serving greedy equality | [session-run1](data/2026-09-22-gemm-accuracy/session-run1.json) | JSON + SSE, both cases |

Decode-mode error distributions before (13c) and after
([short](data/2026-09-22-gemm-accuracy/errdist-decode-short-nothink.json),
[long](data/2026-09-22-gemm-accuracy/errdist-decode-long-think.json)), as mean / p90:

| Tensor | Short before | Short after | Long before | Long after | llama (short) |
|---|---:|---:|---:|---:|---:|
| `attn_pregate` | 9.97e-7 / 1.44e-6 | **4.27e-7 / 6.19e-7** | — | — | 4.01e-7 / 5.95e-7 |
| `attn_gated` | 1.14e-6 / 1.74e-6 | **5.41e-7 / 8.42e-7** | 1.17e-6 / 1.62e-6 | **5.89e-7 / 8.28e-7** | 5.31e-7 / 8.64e-7 |
| `l_out` | 4.09e-7 / 8.25e-7 | **2.79e-7 / 5.83e-7** | 4.50e-7 / 9.22e-7 | **3.15e-7 / 6.02e-7** | 2.94e-7 / 5.73e-7 |
| logits | 7.08e-7 / 1.27e-6 | **4.45e-7 / 8.90e-7** | 6.80e-7 / 1.12e-6 | **4.71e-7 / 8.30e-7** | 4.75e-7 / 9.17e-7 |

**Retained caveat: the worst single samples.** Mean and p90 roughly halve, but a few
extreme samples grew:
- long-case `attn_gated` max 8.5e-6 → 1.2e-5 (llama's max 7.6e-6);
- worst decode logit row 2.4e-6 → 4.1e-6;
- worst serving-continuation logit row in mode 512: 4.3e-6 → 6.3e-6, against a bound
  of 9.4e-6.

These are isolated (token, layer) samples. They are consistent with near-tied softmax
cases where any rounding change moves the output, but they were not investigated
individually.

## Speed

Per-phase profile ([p23](data/2026-09-22-gemm-accuracy/profile-p23.jsonl),
[p3223](data/2026-09-22-gemm-accuracy/profile-p3223.jsonl)):

| Position | Decode step before | After | Scores phase before | After |
|---|---:|---:|---:|---:|
| ~40 | 20.10 ms | 20.09 ms | 0.201 ms | 0.187 ms |
| ~3.2K | 20.49 ms | 20.49 ms | 0.361 ms | 0.370 ms |

## Serving (median of 3; run1 / repeat)

The repeat reused the run1 binary (`1566998e…`).
[run1](data/2026-09-22-gemm-accuracy-serving-run1/summary.json),
[repeat](data/2026-09-22-gemm-accuracy-serving-repeat/summary.json):

| Case (prompt tok) | zerv TTFT ms | zerv decode tok/s | llama-server FA TTFT | llama-server decode |
|---|---:|---:|---:|---:|
| short-nothink (23) | 101 / 101 | 53.9 / 54.4 | 162 / 162 | 44.1 / 44.0 |
| decode-think (81) | 243 / 242 | 49.2 / 49.4 | 369 / 371 | 41.4 / 41.4 |
| medium-prompt (836) | 2021 / 2017 | 49.0 / 49.1 | 1206 / 1210 | 41.4 / 41.3 |
| long-prompt (3223) | 7772 / 7758 | (8-token run) | 3407 / 3415 | (8-token run) |

Speed is unchanged from 13c. All four served outputs are identical to 13c, including
the 256-token `decode-think` generation.

## Candidates not taken

- **GEMM third accumulation level:** ~2× more accurate per GEMM in emulation. Prefill is
  already the more accurate path; it would cost ~32 more accumulator registers per
  thread in the scalar-X kernel. Deferred.
- **Matvec fma accumulation:** ~5% more accurate per operation. The matvec kernel is
  under the user's earlier "no further matvec work for now" instruction, and the gain
  is small.
