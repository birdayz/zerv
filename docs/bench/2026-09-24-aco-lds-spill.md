# RCA: wrong K-quant multi-row matvec results from ACO's LDS spilling (2026-09-24)

**Status:** root cause found; workaround and gate in place; upstream report not filed yet.
**Bottom line:** our shader is correct. Mesa 26.2.3's ACO compiler places VGPR spills in
LDS for single-wave workgroups without memory-ordering information. Its pre-RA scheduler,
which runs after spilling, then moved a spill store above a reload from the same, reused
slot, and one accumulator read another value.

## Symptom

- `matvec_rows.comp` with 4 weight rows per workgroup × 5 input rows gives wrong values on
  the Q5_K and Q6_K paths: unfused GROUP 4, or fused SWIGLU GROUP 2 (4 weight rows). Q4_0,
  Q4_1, F32 and Q8_0 are correct in the same configuration.
- Exactly one of the 20 (weight row, input row) outputs per workgroup is wrong, and the
  error is in the value, not in rounding (`q5_k-seeded-256`, count 5, input row 4:
  0xc06de584 against 0xc061c584).
- Seen twice: the FMA table sweep ([FMA report](2026-09-24-fma-matvec.md), "G = 4 at 5 rows
  is not bitwise equal", recorded as "cause not investigated"), and the verify-fusion GPU
  test ([verify fusion](2026-09-24-verify-fusion.md)).

## Environment

RX 7900 XTX (gfx1100), Mesa 26.2.3 RADV/ACO (source `third_party/mesa/mesa-26.2.3`,
tarball sha256 `1628058a8d2c0615975de5a15ab7bbb9638c50000b5bed9456ff423ea034a81f`),
glslc as pinned in `tools/compile_matvec.py`. Raw data:
[data/2026-09-24-aco-lds-spill/](data/2026-09-24-aco-lds-spill/). Full shader dumps
(several MB) in `third_party/kquant-rows-bug/`.

## Reproducer

```sh
L=third_party/kquant-rows-bug/r5g4; mkdir -p $L
# every format of matvec_rows.comp with ROWS=5 GROUP=4 CB=2 (FMA arithmetic), e.g.:
glslc --target-env=vulkan1.1 -O -fshader-stage=compute -DFORMAT=13 -DBLOCK_BYTES=176 \
  -DPAYLOAD_OFFSET=48 -DLANES=64 -DALIGNED_WORDS=0 -DROWS=5 -DGROUP=4 -DCB=2 \
  src/matvec/matvec_rows.comp -o $L/q5_k.spv   # likewise q6_k, q4_0, q4_1, q8_0, f32 (LANES=256)
zig build matvec-rows-bench-build -Doptimize=ReleaseFast -Dcpu=native
./zig-out/bin/zerv-matvec-rows-bench MODEL --spv-dir $L --group 4 --samples 3 5
```

Real model weights, every row checked bitwise against the shipped single-row module:

| modules (weight rows × input rows) | Q5_K `ssm_out` wrong values | Q6_K `output` wrong values | others |
| --- | ---: | ---: | --- |
| 4 × 5 | 61,440 (= 48 layers × 5120 / 4) | 62,080 (= 248,320 / 4) | 0 |
| 4 × 4 (control) | 0 | 0 | 0 |
| 3 × 5 (control, shipped unfused) | 0 | 0 | 0 |

One wrong value per workgroup, on every workgroup.

## Isolation (DFS)

1. **Register statistics** (`RADV_DEBUG=shaderstats`): the failing modules use 256 VGPRs
   and spill 210 (Q5_K) / 201 (Q6_K) VGPRs. LDS grows from the shader's 5,120 bytes to
   16,384, and scratch is 0: the spill slots are in LDS. On the K-quant path, 4 × 5 needs
   about 288 VGPRs (40 vec4 accumulators plus 32 decoded vec4). The 4 × 4 K-quant modules
   spill 15–17 VGPRs into LDS and are correct; 3 × 5 does not spill.
2. **ACO spills to LDS only for single-wave workgroups** (`aco_spill.cpp`, `spill()`:
   compute stage, `workgroup_size <= wave_size`, no stack pointer). Our 64-lane matvec
   workgroups on wave64 qualify.
3. **Spills in scratch instead of LDS are correct:**
   - forced wave32 (`--wave 32`; the workgroup becomes two waves): same 210 spills in
     scratch (5,632 bytes), 0 wrong values;
   - wave64 with a dummy 45 KB shared array (`matvec_rows_pad.comp`), so no LDS is left
     for spill slots: 211 spills in scratch (11,008 bytes), 0 wrong values.
4. **ACO pass bisection** (`ACO_DEBUG`, shader cache disabled; [aco-debug/](data/2026-09-24-aco-lds-spill/aco-debug/)):

   | `ACO_DEBUG` | wrong values (Q5_K / Q6_K) | spills / LDS / scratch |
   | --- | --- | --- |
   | (none) | 61,440 / 62,080 | 210 / 16,384 / 0 |
   | `validateir,validatera,validate-livevars` | 61,440 / 62,080, validators silent | same |
   | `nosched-ilp` (post-RA scheduler off) | 61,440 / 62,080 | same |
   | `force-waitcnt`, `force-waitdeps` | 61,440 / 62,080 | same |
   | `novn` | 61,440 / 62,080 | same |
   | `noopt` | 0 / 0 | 174 / 16,384 / 256 (different code) |
   | **`nosched`** (all scheduling off) | **0 / 0** | **210 / 16,384 / 0 (identical spilling)** |

   `nosched` differs from `nosched-ilp` only by the pre-RA scheduler (VOPD scheduling is
   wave32-only). With identical LDS spilling, the pre-RA scheduler decides correctness.
5. **The moved instruction.** Spilling runs before scheduling, so both builds have the same
   reload temps and slots. `reload_diff.py` counts, for each of the 225 LDS reloads, the
   same-slot spill stores before it in its block ([output](data/2026-09-24-aco-lds-spill/reload_diff.out)):
   exactly one differs. Block 31 (the main loop body), slot 9 (LDS offset 7424):

   ```
   nosched (correct)                                   sched (default, wrong)
   +1293  %3637 = ds_read_addtid_b32 offset0:7424      +1277  ds_write_addtid_b32 %1273 offset0:7424
   +1294  %1978 = v_fmac_f32 ..., %3637                +1294  %3637 = ds_read_addtid_b32 offset0:7424
   +1306  ds_write_addtid_b32 %1273 offset0:7424       +1295  %1978 = v_fmac_f32 ..., %3637
   ```

   The spiller reused slot 9: first it holds an accumulator (reloaded as `%3637` and
   FMA-accumulated into `%1978`), then the decoded weight `%1273`. The scheduler hoisted the
   store of `%1273` above the reload, so the accumulator became a weight value on every
   loop iteration: one corrupted accumulator component per workgroup, as observed.

## Root cause (Mesa 26.2.3 source)

- `aco_spill.cpp` lines 1397 and 1412 (`ds_write_addtid_b32`) and 1460 and 1479
  (`ds_read_addtid_b32`) build the LDS spill/reload instructions **without a
  `memory_sync_info`**, so they default to `storage_none`. `aco_ir.h:53` describes it as
  "no synchronization and can be reordered around aliasing stores". The scratch path
  (lines 1401, 1415, 1466, 1472) tags its spills
  `memory_sync_info(storage_vgpr_spill, semantic_private)`.
- `aco_interface.cpp`: `spill()` (line 104) runs before `schedule_program()` (line 125),
  so the scheduler sees real spill memory instructions.
- `aco_scheduler.cpp`: `add_to_hazard_query` records only nonzero storage classes as
  aliasing (lines 646–655), and `perform_hazard_query` forbids reordering two memory
  instructions only when their storage classes intersect (line 757). Two `storage_none`
  LDS spill accesses never intersect, so a spill store may pass a reload from the same
  slot.
- **Proposed upstream fix** (not verified with a patched Mesa build): give the LDS
  spill/reload instructions `memory_sync_info(storage_vgpr_spill, semantic_private)` like
  the scratch path (or a dedicated class), so the scheduler keeps same-slot accesses in
  order. The validators (`validateir`, `validatera`, `validate-livevars`) do not check the
  ordering of spill-slot memory accesses, which is why they stay silent.

## Workaround and gate

- **No shipped shader may spill VGPRs into LDS.** `tools/check_shader_spills.py` runs a
  command with the shader cache disabled and `RADV_DEBUG=shaderstats`. It fails on VGPR
  spills with scratch 0 (all slots in LDS), warns on scratch spills (correct but slow),
  and reports SGPR spills. Limitation: the statistics cannot show a mixed LDS + scratch
  spill in a single-wave workgroup.
- Results ([spill-gate/](data/2026-09-24-aco-lds-spill/spill-gate/)): `gpu-test` 2,458
  pipelines, 0 fail, 0 warn. `zerv-spec-check` (f32 and f16 KV), `zerv-mtp-check` and
  `zerv-model-profile` with f16 prefill: 0 fail, 2 warn, both 128-lane (two-wave)
  kernels spilling to scratch (below). **Negative control:** the 4 × 5 build fails the
  gate (2 pipelines).
- Tables: the fused verify modules for Q5_K/Q6_K stop at count 2 (their 4-weight-row
  modules at counts 3–4 spilled 15–53 VGPRs into LDS, although they passed the tests), and
  count 5 has no fused module. Those cases take the unfused path, whose shipped table does
  not spill.
- Tuning tools: `tools/tune_matvec_rows.py` marks spilling candidates invalid.
  `tools/race_swiglu_rows.py` runs every candidate table through the full GPU fixture
  suite (all formats) under the spill gate before racing it.

Side findings:
- The decode DeltaNet kernel `K_DELTA` (128 VGPRs spilled, 32 KB scratch) and the prefill
  `K_DELTAB` (130, 33 KB) spill to scratch: correct, but a performance item (decode
  `delta` 0.58 ms per step; prefill delta 9.5% of prefill GPU time).
- The 17b.1 "count-4 K-quant anomaly" (G = 4 at 4 rows pathologically slow) is these
  spills. G = 2 at 4 rows does not reproduce with the current FMA kernel: no spills, 0
  wrong values, Q5_K `ssm_out` 2.33 ms against 4.69 ms for 4 single-row passes.

## 5 whys

**A. Why were the results wrong?**
1. Why did one output per workgroup come out wrong? One accumulator was reloaded from an
   LDS spill slot after another value had been stored there.
2. Why was the other value stored first? ACO's pre-RA scheduler hoisted that spill store
   above the reload, and the spiller had reused the slot.
3. Why could the scheduler do that? Mesa 26.2.3 emits LDS spill instructions without
   memory-sync info (`storage_none`), which the scheduler treats as never aliasing.
   Scratch spills are tagged and safe.
4. Why did our shader spill into LDS at all? A 64-lane workgroup on wave64 is a single
   wave (the LDS-spill case), and 4 weight rows × 5 input rows on the K-quant path needs
   about 288 VGPRs against 256.
5. Why did we build and ship-test a spilling configuration? Candidates were explored for
   speed only. Register pressure and spills were never a selection criterion: a spilling
   variant was assumed to be only slow (17b.1 reported "register spills" purely as a
   timing).

**B. Why didn't we catch it the first time?**
1. Why wasn't it fixed when first seen (FMA table sweep)? It was detected: the bench's
   bitwise check flagged G = 4 at 5 rows on Q5_K/Q6_K. But it was handled as a tuning dead
   end ("marked invalid, cause not investigated"). No bug entry was opened in TODO.md, so
   it lived only in a report's findings list.
2. Why was an unexplained wrong result allowed to become a footnote? There was no rule
   that an unexplained wrong result stops the work until it is root-caused, and the active
   goal (performance) made "avoid the bad variant" look sufficient. The variant was
   excluded, but the mechanism was still live in other modules: the 4 × 4 K-quant modules
   were spilling into LDS the same way.
3. Why did the second occurrence reach a selected table? The fused-table race judged
   candidates in-model (spec-check timing, bitwise gate). This model's gate/up tensors are
   Q4_0, so the race exercised only the Q4_0 fused modules while the table applied to
   every format. The model-level gates cover the formats this model uses, not the modules
   we ship.
4. Why was full-format validation not part of selection? Tuning (race tools) and the
   fixture suite (`gpu-test`, all formats) were separate steps, and `gpu-test` ran only
   after the winning table was installed. It caught the bug there, one step later than it
   should have.
5. Why was there no check between "our math is right" and "the GPU computes our math"?
   The verification strategy compares our arithmetic with references and treated the
   shader toolchain (glslc + RADV/ACO) as correct. Nothing checked compiler-output
   properties such as spills, and ACO's own validators do not cover spill-slot ordering.

## Corrective actions

| # | Action | Status |
| --- | --- | --- |
| 1 | Bug entry at the top of TODO.md; work rule "always DFS into bugs" (an unexplained wrong result is top priority; avoiding the trigger is not a fix) | done |
| 2 | Root cause to the Mesa source line, with a reproducer and IR evidence | done (this report) |
| 3 | Gate `tools/check_shader_spills.py` (no VGPR spills in LDS), with a negative control, added to the required checks when shaders or tuning tables change | done |
| 4 | Tuning tools reject spilling candidates and validate every candidate on all formats before timing it | done |
| 5 | Shipped fused tables avoid spilling (K-quant fused ≤ count 2, no count-5 fused module) | done |
| 6 | Report upstream (Mesa issue with this reproducer and the proposed fix); optionally verify the fix with a locally built Mesa selected via `VK_ICD_FILENAMES` (no system change) | open, needs the user (network access, Mesa build) |
| 7 | Remove the scratch spills of `K_DELTA` / `K_DELTAB` (performance) | done: 128/130 → 7/9 spilled VGPRs ([report](2026-09-24-delta-spill.md)) |
