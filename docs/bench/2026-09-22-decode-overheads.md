# Decode step overheads (block 13c) — 2026-09-22

**Question:** where does the decode step spend time beyond streaming weights, and can
it be removed without changing any output bit?

**Result:** yes, for the two slow kernels. The RMS norm and the DeltaNet decode scan
were latency-bound on patterns the compiler could not optimize. Rewriting both,
bit-identically, takes the decode step from **21.42 to 20.10 ms GPU** at short context
(−6.2%) and from 21.91 to 20.49 ms at ~3.2K:

| Kernel | Before | After | Calls per step |
|---|---:|---:|---:|
| Norm | 1.36 ms | 0.23 ms | 129 |
| DeltaNet | 1.00 ms | 0.58 ms | 48 |

All 34 oracle capture files are **byte-identical** to the 13e run.

## Research

**Link cost (dispatch + barrier):** on this card a dependent link costs ~1.7–1.9 µs in
the model and 0.44 µs as an empty kernel in isolation
([kernel chain](data/2026-09-22-decode-overheads/)). The ~610 links per step are not
the problem.

**Norm: ~10.4 µs per call, and 11.4 µs in isolation.** The bench is
`zerv-kernel-chain`: 129 dependent norms, device-local buffers.
- **ISA:** per-element bounds guards (`if (i < width)`) became **136 divergent branches
  with 130 scalar descriptor reloads** (`s_load_b128/b256`) and 206 waits. The kernel
  serialized on descriptor fetches, not data.
- **Fix:** norm rows are always the hidden size (5120 = 20 × 256), so the loop is
  compile-time (`HIDDEN` in `model.comp`, checked against `config.hidden`):
  straight-line code with 4 descriptor loads. Loads are grouped before stores (the
  buffer aliases), and the norm weights are prefetched.
- **Result:** 4.5 µs in isolation; 1.7 µs per call in the model (inputs hot in L2).

**DeltaNet: 21 µs per layer against ~7 µs of state traffic.** Each thread read its
128-value state column twice, with the second loop interleaving loads and stores to the
same buffer. Holding the column in registers (as the prefill scan already does): one
batch of loads, the same arithmetic, one batch of stores. Result: 12 µs per layer.

**The declared bit-identity gate failed first** (gate-run1: all capture files differed;
the tolerance gates passed, decode worst/bound 0.836 vs 0.494). Isolation:
- **Localization:**
  - Old norm + new DeltaNet was bit-identical, so the DeltaNet change is exact.
  - The first differing capture was `attn_post_norm` with identical inputs.
- **Differential test:** the new `zerv-kernel-chain compare OLD NEW ROWS` mode showed
  12 of 1000 random rows with a one-ulp different inverse RMS.
- **Candidates excluded:**
  - The run-time vs compile-time divisor of the mean (the width is back in the push
    constants; it did not change the result).
  - Identical ISA tails.
- **Cause:**
  - Once unrolled, the sum of squares `ss += v*v` starting from a literal 0 folds its
    first term to `v0*v0`. The compiler then fused the other product of
    `v0² + v1²`: `fma(v0,v0,v1²)` instead of the old loop's `fma(v1,v1,round(v0²))`.
  - An explicit `fma()` did not prevent this.
  - `precise` made it worse (split into multiply + add; 231/1000 rows differ).
- **Fix:** seed the accumulator with a run-time +0 (`uintBitsToFloat(flags &
  0x80000000)`; bit 31 is never set, documented in the runtime). Every step is then a
  single-product `fma` in element order, exactly the old `v_fmac` chain. Result:
  **0 of 1000 rows differ**, then all 34 capture files identical (gate-run2).

This is a general hazard, now documented in the kernel: unrolling an accumulation that
starts from a literal zero can change which product is fused.

## Correctness

| Gate | Evidence | Result |
|---|---|---|
| Bit identity vs 13e, all capture files of modes 0/1/13/29/60/512 | [gate-run2](data/2026-09-22-decode-overheads/gate-run2.json), `third_party/model-native/2026-09-22-decode13c-gate2` vs `...-scalarx-gate2` | 34/34 identical (gate-run1, which failed, is kept) |
| Norm old vs new on 1000 random rows (residual add, rows from io) | `zerv-kernel-chain compare` | 0 differ |
| Serving greedy equality | [session-run2](data/2026-09-22-decode-overheads/session-run2.json) | JSON + SSE, both cases |
| CPU/GPU/Python tests | required checks | pass |

## Per-phase profile (decode, GPU ms)

`zerv-model-profile MODEL 8192 512 N 64`; data in
[`data/2026-09-22-decode-overheads/`](data/2026-09-22-decode-overheads/):

| Position | Step before | Step after | Norm before → after | DeltaNet before → after |
|---:|---:|---:|---|---|
| ~40 | 21.42 | 20.10 | 1.362 → 0.226 | 1.003 → 0.578 |
| ~850 | 21.59 | 20.18 | 1.359 → 0.228 | 1.010 → 0.592 |
| ~3240 | 21.91 | 20.49 | 1.371 → 0.229 | 1.006 → 0.586 |

Prefill is unchanged (23 tokens 97.0 vs 99.1 ms; 3223 tokens 7703 vs 7708 ms).

**What remains in a 20.1 ms step:**
- Weight-streaming matvecs take ~18.5 ms at 720–915 GB/s; a raw read probe reaches
  ~920 GB/s.
- Everything else is ~1.6 ms.
- Under decode load the card also sits at its 339 W power cap (shader clock ~2.5 GHz,
  memory clock at its top level).

## Serving (median of 3; run1 / repeat)

The repeat reused the run1 binary (`8882b737…`).
[run1](data/2026-09-22-decode-overheads-serving-run1/summary.json),
[repeat](data/2026-09-22-decode-overheads-serving-repeat/summary.json):

| Case (prompt tok) | Engine | TTFT ms | Decode tok/s | Total s |
|---|---|---:|---:|---:|
| short-nothink (23) | zerv (13c) | 101 / 101 | 53.8 / 53.8 | 0.34 / 0.34 |
| short-nothink (23) | llama-server FA ub512 | 161 / 162 | 44.1 / 44.1 | 0.46 / 0.46 |
| short-nothink (23) | llama-server fully FP32 | 249 / 250 | 43.2 / 43.1 | 0.55 / 0.55 |
| short-nothink (23) | zerv before (13e run1 / repeat) | 102 / 102 | 49.9 / 50.0 | 0.36 / 0.36 |
| decode-think (81) | zerv (13c) | 242 / 242 | 49.1 / 49.2 | 5.43 / 5.42 |
| decode-think (81) | llama-server FA ub512 | 370 / 371 | 41.4 / 41.4 | 6.52 / 6.53 |
| decode-think (81) | llama-server fully FP32 | 681 / 684 | 40.6 / 40.5 | 6.97 / 6.98 |
| decode-think (81) | zerv before (13e run1 / repeat) | 243 / 243 | 45.6 / 45.6 | 5.83 / 5.83 |
| medium-prompt (836) | zerv (13c) | 2020 / 2016 | 49.0 / 49.0 | 4.61 / 4.61 |
| medium-prompt (836) | llama-server FA ub512 | 1204 / 1211 | 41.4 / 41.3 | 4.27 / 4.29 |
| medium-prompt (836) | llama-server fully FP32 | 2934 / 2954 | 40.1 / 40.1 | 6.10 / 6.12 |
| medium-prompt (836) | zerv before (13e run1 / repeat) | 2017 / 2016 | 45.5 / 45.4 | 4.80 / 4.81 |
| long-prompt (3223) | zerv (13c) | 7764 / 7753 | (56.5 / 56.3)* | 7.91 / 7.90 |
| long-prompt (3223) | llama-server FA ub512 | 3407 / 3427 | (45.8 / 45.8)* | 3.58 / 3.60 |
| long-prompt (3223) | llama-server fully FP32 | 9876 / 9968 | (43.0 / 42.9)* | 10.06 / 10.16 |
| long-prompt (3223) | zerv before (13e run1 / repeat) | 7759 / 7753 | (52.1 / 52.1)* | 7.91 / 7.91 |

\*8-token generations; not valid decode rates.

- **Decode:** +7.6–7.7% at every length. zerv now decodes **18–22% faster than
  llama-server FA**.
- **Total request time:** the 256-token `decode-think` request takes 5.43 s vs 6.52 s.
- **Unchanged:** TTFT and all output hashes (identical to 13e).

## Failed run retained

**The first serving attempt ran concurrently with the session check** — my error: two
GPU tool calls were issued in one step. Two servers of ~18.8 GB each oversubscribed
the 24 GB of VRAM, so the driver evicted buffers to system memory. A prefill command
reading weights over PCIe then exceeded the compute-queue lockup timeout:
- the kernel reset both compute queues ("ring comp_1.3.x timeout … reset succeeded");
- the session server returned HTTP 500 (`DeviceLost`);
- the benchmark aborted.

Logs and kernel messages are in
[failed-concurrent-run/](data/2026-09-22-decode-overheads/failed-concurrent-run/). The
partial benchmark output directory was deleted before this note (invalid run); its log
is kept.

Run alone, the session check passed and the benchmark completed. Rule reaffirmed:
GPU jobs strictly one at a time.
