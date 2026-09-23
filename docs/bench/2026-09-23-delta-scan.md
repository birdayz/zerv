# DeltaNet prefill scan latency (block 13j)

Date: 2026-09-23. Question: the batched GatedDeltaNet scan (`delta_b`) took 62 ms of a
~790 ms 512-row chunk. How much of it is scheduling rather than the inherent
recurrence? [Spec section](specs/prefill.md#deltanet-scan-latency--block-13j-specified-2026-09-23-before-implementation).

## Change (the arithmetic is unchanged)

- **Gated RMS norm.** It moved to the new kernel `gnorm_b`: workgroup (head, row),
  same expressions and `wg_sum` tree. That removes 9 barriers per row from the
  sequential scan.
- **Gate scalars** (`beta`, `softplus`, `g`, `decay`). These are computed in a
  parallel prologue per ≤512-row segment and kept in LDS. They are written to act
  exactly as before.
- **q/k** are double-buffered in LDS, with the next row's q/k/v prefetched into
  registers. That leaves one barrier per row instead of two.

## Results

Profiles: `zerv-model-profile MODEL 8192 512 N 16`; two runs at 512 rows
([r1](bench/data/2026-09-23-delta-scan/profile-p512-r1.jsonl),
[r2](bench/data/2026-09-23-delta-scan/profile-p512-r2.jsonl)) and
[p3223](bench/data/2026-09-23-delta-scan/profile-p3223.jsonl).

| | 13h | 13j |
| --- | --- | --- |
| `delta` phase, 512-row chunk | 62 ms | **45.4 / 45.9 ms** (includes `gnorm_b`) |
| 512-row chunk total | 788–796 ms | 777 / 780 ms |
| 3223-token prefill | 5283 ms | 5192 ms |
| decode step | 20.2 ms | 20.15 ms (the decode kernel is unchanged) |

**Serving** ([run1](bench/data/2026-09-23-delta-scan-serving-run1/),
[repeat](bench/data/2026-09-23-delta-scan-serving-repeat/) on the pinned run1 binary
`66bfccfc…`). Median TTFT in ms, run1 / repeat:

| Prompt | zerv 13h | zerv 13j | llama-server FA ub512 |
| --- | --- | --- | --- |
| 23 tok | 76 / 77 | 75 / 76 | 162 / 162 |
| 81 tok | 184 / 186 | 181 / 182 | 367 / 368 |
| 836 tok | 1415 / 1411 | **1371 / 1385** | 1202 / 1200 |
| 3223 tok | 5337 / 5335 | **5201 / 5237** | 3396 / 3392 |

- Decode throughput is unchanged: 54.3 / 49.4 / 49.0–49.2 tok/s, against llama's
  44.0 / 41.3 / 41.2.
- Host RSS while serving is 59 MB.

## Correctness

| Gate | Result |
| --- | --- |
| Default oracle, modes 0/1/13/29/60/512 ([gate-run1](bench/data/2026-09-23-delta-scan/gate-run1.json)) | pass; all 70 captured tensor/logit files byte-identical to 13h |
| Long oracle (562 tokens), modes 0/512/64 ([gate-long-run1](bench/data/2026-09-23-delta-scan/gate-long-run1.json)) | pass; all 17 files byte-identical to 13h |
| Served outputs (4 cases × 2 runs) | byte-identical to 13h |
| `zig fmt`; `zig build test` Debug + ReleaseFast (68); `gpu-test` Debug + ReleaseFast (15); Python unittest | pass |

## Negative experiment (retained)

Source: [model-fence.comp](bench/data/2026-09-23-delta-scan/experiments/model-fence.comp).

- **VGPR spills.** `RADV_DEBUG=shaderstats` shows that both DeltaNet kernels spill
  VGPRs. The decode `delta` kernel spills 126 (32 KB scratch); the new `delta_b`
  spills 130. The compiler interleaves the two 128-term loops and keeps too much
  live.
- **Fence experiment.** A `memoryBarrierShared()` between the loops removes the
  spills (0 and 4), but it is much slower:
  - prefill `delta` rises to 116 ms (from 45);
  - decode `delta` goes from 0.58 to 0.63 ms.

  It was reverted. The spills are cheaper than the serialization the fence forces.
  Register pressure in these kernels remains an open lever.

## Remaining gap

At 3223 tokens zerv takes 5.2 s against llama-server's 3.4 s. llama-server's default
prompt path is f16 WMMA with f16 accumulation; zerv is FP32. At matched FP32, llama
takes 9.9 s ([llama precision](2026-09-23-llama-precision.md)).
