# DeltaNet kernels: removing a 128-VGPR scratch spill (block 17c, 2026-09-24)

Question: the spill gate ([ACO RCA](2026-09-24-aco-lds-spill.md)) warned that the decode
DeltaNet kernel `K_DELTA` spills 128 VGPRs and the prefill `K_DELTAB` 130, both to scratch
(32–33 KB per wave). Why, and what does it cost? Knob `--delta-state-out on|off`
(`Options.delta_state_out`, default on). Raw data: [data/2026-09-24-delta-spill/](data/2026-09-24-delta-spill/).

## Cause (compiler IR)

- `RADV_DEBUG=shaderstats`: register demand before scheduling is 384 (`K_DELTA`) and 386
  (`K_DELTAB`), i.e. three sets of 128 values, where the design needs one (the state
  column) plus the 128 k values that both inner loops read.
- The final NIR shows only the intended two sets: loop 1 loads `ks` (64 two-float loads)
  and multiplies `col[i]*decay` (no FMA contraction); loop 2 reuses both and loads only
  `qs`.
- ACO's liveness dump (`ACO_DEBUG=liveinfo`) shows the third set. The state column is
  loaded from `state[base + i*128]`, byte offset `i*512`; buffer loads have a 12-bit
  immediate offset, so for i ≥ 8 the compiler builds one VGPR address per element (120
  addresses, e.g. `%29 = v_add_u32 0x1000, %20`). It keeps them live across the whole row
  loop to reuse them for the final `state[base + i*128] = col[i]` stores. ACO spills 122 of
  them to scratch right after the loads and rematerializes them after the loop (8 scratch
  loads in total): pure waste.

## Fix

The final store goes to a separate push constant `state_out` (the same value as `ssm`,
set by the host at every call site). The compiler cannot prove the two equal, so it
recomputes the store addresses after the loop. Same arithmetic, same stores.

| kernel | demand before scheduling | VGPRs spilled | scratch per wave |
| --- | ---: | ---: | ---: |
| `K_DELTA` before | 384 | 128 | 32,768 |
| `K_DELTA` with `state_out` | 263 | 7 | 1,792 |
| `K_DELTAB` before | 386 | 130 | 33,280 |
| `K_DELTAB` with `state_out` | 265 | 9 | 2,304 |

The remaining 7–9 are genuine: the state column plus the reused k values need about 263
registers.

Modules: `delta` and `delta_b` are now built with `-DSTATE_OUT` (`tools/compile_model.py`).
`delta_legacy` and `delta_b_legacy` are the previous modules byte for byte, selected by
`--delta-state-out off`. No other model module changed.

## Correctness

- `tools/verify_model.py`, default oracle (modes 0/1/13/512/512:17) and long oracle
  (0/13/128/512/512:300): all 28 + 14 capture and logits files **byte-identical** to the
  previous build (`2026-09-24-fmaknob-default`, `2026-09-24-swiglu-long`). This covers
  decode (`delta`) and prefill (`delta_b`).
- `zerv-spec-check` 11/11: f32 KV with `state-out` and `legacy` (2 passes each), f16 KV.
- `zerv-mtp-check` short-nothink: all 19 dump files identical to the previous run;
  scenario C 3/3.
- `zerv-prefix-check`: gate passed (`all_bit_identical` is informational and was false
  before as well).
- Serving outputs identical with the knob on and off (below).

## Speed

`zerv-spec-check` wall clock, same binary, knob alternated, 2 passes (ms):

| | step | verify+commit 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 19.658 / 19.677 | 20.297 / 20.299 | 21.145 / 21.136 | 22.410 / 22.440 | 24.626 / 24.624 | 28.739 / 28.729 |
| state-out | **19.389 / 19.399** | **19.811 / 19.805** | **20.644 / 20.633** | **22.138 / 22.120** | **24.234 / 24.176** | **28.434 / 28.402** |

`zerv-model-profile MODEL 8192 512 3223 4 f16 f16` (lab build, 2 passes): prefill delta
phase 262.7 / 261.8 → **248.8 / 248.4 ms** (−5%), prefill GPU 2,771 / 2,760 → 2,752 /
2,749 ms.

Serving (`bench/run_serving.py`, 2 repeats), outputs identical across all engines:

| decode-v1, decode tok/s | code | json | think | prose |
| --- | ---: | ---: | ---: | ---: |
| plain, state-out | **51.4** | **51.4** | **51.4** | **51.4** |
| plain, legacy | 50.6 | 50.6 | 50.6 | 50.6 |
| 3 drafts, state-out | **123.3** | **127.8** | **110.0** | **77.4** |
| 3 drafts, legacy | 121.9 | 126.4 | 108.6 | 76.1 |

| serving-v2 `--no-prompt-cache` (fp32 prefill), TTFT ms | short (23) | think (81) | medium (836) | long (3,223) |
| --- | ---: | ---: | ---: | ---: |
| state-out | 74.9 | 181.2 | **1,360.8** | **5,048.2** |
| legacy | 75.1 | 182.1 | 1,373.9 | 5,070.9 |

Plain decode +1.6%, speculative decode +1.1 to +1.7%, TTFT −0.4 to −1.0%.

## Observations and negative results

- **A scratch-heavy kernel slows other kernels in later submissions.** In a lab build
  where both delta kernels used little scratch, the decode step's ffn_in phase ran 7.59
  instead of 7.79 ms (−2.6%, 2 passes), although its own shader was unchanged. When only
  the decode kernel was fixed and the prefill `delta_b` still spilled 33 KB per wave, the
  decode step did not gain at all. RADV sizes the compute scratch ring per queue for the
  largest scratch user it has seen (`radv_queue.c`, `COMPUTE_TMPRING_SIZE`), so one
  spilling kernel affects every later submission on that queue. The hardware mechanism is
  not established. A DRAM write-back explanation was considered and ruled out: it would
  follow the step's own kernel, not the prefill kernel. Consequence: no shipped kernel
  should need much scratch, even one that runs rarely.
- **Negative: re-reading k from LDS (`memoryBarrierShared()` between the two loops).**
  Demand 384 → 258, 2 spills. It was faster for 1–2 rows (decode step −0.13 ms in the lab)
  but slower for 3–5-row verify passes (+0.1 to +0.5 ms) and **2.5× slower for prefill**
  (delta 261 → 646 ms): 128 extra LDS reads per row. `state_out` beats it at every row
  count; the variant was removed.
