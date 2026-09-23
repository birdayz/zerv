# Matvec DFS: scalar decode versus packed traversal

2026-09-22, block08b only; serving and normalization remain paused. Contract:
[optimization spec](../specs/matvec-optimization.md). Prior exact layouts and
[reference precision findings](gpu-matvec.md) still apply. The independently
extracted 48-case fixture is unchanged, as are the complete real-model numerical
bounds. No activation quantization, scale factoring out of the dot, FP16 dot,
optional subgroup/storage capability, repacking or foreign production math.

## Evidence and causal limits

[All 33 attempts / 32 passing configurations](../bench/data/2026-09-22-matvec-dfs/experiments.json),
[median microseconds CSV](../bench/data/2026-09-22-matvec-dfs/experiments.csv).
Each passing candidate checked all48 fixtures, replay/guards/changed input, and
all11 full dense shapes against retained independent CPU goldens. Each has seven
submit/fence trials; these are exploratory single-round results, not the final
repeated paired comparison. 576 candidate-source/module/binary/output hashes were
rechecked when assembling the ledger. Original large case/golden input hashes
were verified by every tuning invocation. Detailed final evidence belongs in the
[dated benchmark report](../bench/2026-09-22-matvec-optimization.md).

Original Q6 disassembly:
[compiler dump](../bench/data/2026-09-22-matvec-dfs/baseline-isa.txt).
It reports64 invocations,256 shared bytes and wave64. The scalar decoder repeatedly
loads/extracts packed fields, converts a binary16 using integer/branch logic, and
accumulates one coefficient per iteration. The LDS reduction remains, but **there
is no emitted `s_barrier`** in this one-wave baseline: blaming expensive hardware
workgroup barriers for the original loss would be unsupported. No hardware-counter
profiler is installed; disassembly and controlled ablations are the evidence, not
an occupancy/stall/bandwidth-counter measurement. Timings include host submit/fence
costs and do not isolate GPU timestamp time.

Exploratory Q6 full-projection medians, microseconds:

| Step | Same-tuner median | What changed |
|---|---:|---|
| Original scalar | 5736.158 | 64 threads,1 row; diagnostic X cap applied equally |
| Hardware half | 4638.568 | Only `half_at` replaced with core unpack |
| Byte-block64×1 | 2041.828 | Reuse low/high planes, independent sums, shift/unpack half |
| Packed4,64×1 | 1226.6 | Four adjacent payload bytes/activation values per lane |
| Packed metadata64×1 | 1225.5 | Read packed subscale words explicitly |

The hardware-half ablation independently establishes a material conversion cost.
Block traversal combines load/decode reuse, shorter loops and independent sums;
do not attribute that entire gain to any one instruction. Packed4 materially
improves the same64×1 scheduling versus the byte-block path. Metadata packing by
itself does not establish a significant Q6 gain. Source loops for the selected
implementation are compile-time unrolled to keep it maintainable.

Hardware half primary-source provenance is in
[the pinned Khronos ledger](2026-09-22/matvec-optimization-sources.json).
The registry reference URL returned403; the official pinned GitHub mirror was
retained instead. `unpackHalf2x16` returns FP32 from each half without requiring
Float16 arithmetic/storage. The six full-finite-half fixture fields (380928 exact
outputs) gate its behavior, including half subnormals.

## Scheduling/precision searches and retained losses

- Byte-block lanes32/64/128/256/512, rows1/4/8: increasing parallelism helps the
  tiny48-row F32 projection, but quantized rows do not improve monotonically.
- Packed4 lanes16/32/64/128/256 and rows1/2/4:64×1 is the strongest overall quant
  choice here. Row packing is not a general win;32×1 versus64×1 Q6 is close, while
  Q4/Q5 favor64. Only the selected one-row geometry goes into production.
- FMA for dot accumulation passes the original gates but produces small/mixed
  changes. It is not selected; no dead runtime FMA switch was added.
- Packed8 makes Q6 worse (~1396–1480µs) and Q5 worse (~111–156µs), despite more
  metadata reuse. No register-pressure explanation is claimed without counters.
- Q5 still trails the installed FP32 reference in initial measurements. Reference
  shader algebra/precision and its different host graph boundary complicate exact
  cost attribution; the matched FP32 input setting remains mandatory.
- Q4 default reference MMVQ quantizes activations and remains a separate timing/
  quality control. Default-MMVQ speed is not a precision-matched denominator.

## Grid and source-rebuild corrections

The old65537-row fixture did **not** exercise 2D on this device: its queried X
limit exceeds248320. The new capped, balanced grid genuinely exercises2D and
launches at mostY-1 surplus groups. The row guard is uniform across a workgroup
and is before every barrier. A CPU test also covers the maximum supported row
count. F32's256-thread selection checks local invocation/X/shared limits; lower
limits and short rows select64. Tests restrict each host-side limit in turn and
run independent F32 fixtures, without claiming the physical device changed.

`bench/rebuild_matvec_baseline.py` verifies the original checked-in snapshot,
recompiles its shaders byte-identically, and rebuilds/tests the original native
program in both modes. It does not use the surviving old executable. The tuner
uses the same current diagnostic executable and capped grid for candidate and
baseline ablations. Final production comparisons use the original rebuilt
program's original grid as part of the intended implementation difference.

Diagnostic host support is retained under
`docs/bench/data/2026-09-22-matvec-dfs/source/diagnostic`; remaining pre-integration
native/build sources are the original `gpu-matvec-repeat/source` snapshot.
`matvec_workload-v1.zig` reconstructs the exact pre-ownership-fix variant by
reversing only the recorded transfer-flag patch; v2 is the captured source.
Rebuilt binary hashes may change with source/build paths. Every tuning binary
itself is retained and hashed. Final paired runs snapshot the full current tree.

Failures preserved under the DFS evidence directory:
- Initial tuner copied the executable without its executable mode; permission
  failure before fixtures/timings. Fixed using `copy2`, fresh output directory.
- Initial integration GPU test tried to embed a shader outside the test package
  root. CPU42 passed; GPU compilation failed. Fixed by bounded mapped-file loading
  in the explicit hardware test, not by broadening production package visibility.
- Pipeline replacement now transfers error-cleanup ownership explicitly; a failed
  recording must not destroy the kernel both through a local copy and its Plan.

### Final emitted code cross-check

[Selected Q6 dump](../bench/data/2026-09-22-matvec-dfs/optimized-q6_k-isa.txt),
[Q5 dump](../bench/data/2026-09-22-matvec-dfs/optimized-q5_k-isa.txt),
[static observations](../bench/data/2026-09-22-matvec-dfs/isa-observations.json),
and [exact diagnostic commands/binary pins](../bench/data/2026-09-22-matvec-dfs/profile-manifest.json).
These are untimed zero-iteration runs with process-local `RADV_DEBUG=shaders`;
benchmark timings do not use that setting.

The selected Q6 remains wave64/shared256 and has nine static128-bit buffer loads
versus none in the scalar baseline. Its unrolled loop processes32 coefficients
per active lane per block iteration, so its larger *static* instruction count is
not a claim of larger dynamic work. Maximum named VGPR index rises19→47; that is
an assembly observation, not an occupancy measurement. Neither Q6 dump emits
`s_barrier` or scratch memory instructions.

The compiler folds half unpack followed by scale multiplication into
`v_fma_mix_f32(..., neg(0))`, eight such instructions for Q6 and sixteen for Q5,
rather than emitting standalone half conversions. This uses the existing exact
half operand and produces the required FP32 scale product; it does **not**
quantize x, accumulate in FP16, or fuse the dot accumulator. The NIR retains
`exact` FP32 products and the exhaustive half fixtures pass. Consequently the
GLSL built-in alone should not be described as a one-instruction standalone
conversion in the final ISA.

## Late DFS branch: alignment and the remaining Q5 gap

The original native workload deliberately sets weight offset2 (F32 offset4),
then places x after rounded-up weights plus16 bytes. In full-shape cases this also
means x is only4-byte aligned, versus the external reference's aligned tensor
allocations. This is an API stress test, not the likely eventual aligned GGUF
weight-bank layout. The original benchmark is preserved, including its Q5 loss.

Two further correctness-gated runs kept FP32 inputs and identical weight bytes:
- `matvec-aligned-generic`: offset0, generic existing shader; Q5 **91.860µs**.
- `matvec-aligned-specialized`: same placement, aligned Q4_1/Q5 word reads;
  Q5 **80.501µs**. Only the latter comparison isolates the shader specialization;
  the former placement change affects both W and x alignment.

For20/176-byte Q4_1/Q5 blocks, a4-byte-aligned weight view guarantees every packed
four-byte field stays aligned across rows/blocks. An offline variant can omit
cross-word joins and conditional loads. Plan selects it only for those formats
and offsets divisible by4.18/210-byte Q4_0/Q6 blocks still alternate alignment,
so do not use that shortcut. The six prior modules remain **byte-identical**;
two new modules are added. All48 independent fixtures now run in both address
layouts, with guard/replay/zero-input checks; the original tests/tolerances remain.

Final benchmarking labels `native` (original stress placement) and
`native-aligned` separately, alongside `native-baseline`, `reference-f32` and
`reference-default`. Never report an alignment-changing comparison as the isolated
speedup over the scalar baseline. The aligned path is selected from validated
views, with no public tuning option, repacking, per-dispatch branch or new GPU
capability. Reference graph-host work is still not identical to native submission.
