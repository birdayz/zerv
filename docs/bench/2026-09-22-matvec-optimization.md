# Matvec DFS optimization — 2026-09-22

**Result:** the full Q6 vocabulary projection improves from **5.61 ms to 1.22 ms
(~4.6×)** at the unchanged stress alignment. Every measured shape improves over
our scalar baseline. The optimized component is competitive with this installed
FP32-input Vulkan reference on most shapes, but not uniformly faster. A separate
aligned Q5 path removes much of its remaining gap; small reference differences
are not robust kernel-only wins. **No native model/serving speed claim.** Serving
and normalization remain paused.

## Why it was slow; what changed

[Research/ISA and retained experiments](../research/matvec-optimization.md),
[pre-implementation/selection contract](../specs/matvec-optimization.md).

1. The scalar shader repeatedly decoded one coefficient, reloaded/extracted its
   packed fields, and manually converted its half scale. An isolated core-half
   ablation reduced Q6 from 5736 to 4639µs; this was a material conversion cost.
2. Cooperating lanes now consume whole packed words, reuse quant planes/scales,
   load adjacent FP32 activation vectors and accumulate independent strict sums.
   Same 64×1 byte-block versus packed-4 medians were 2042 versus 1227µs for Q6.
   This combines load reuse, shorter loops and more independent work; no hardware
   counters are available to apportion all savings to one instruction.
3. The 48-row F32 projection needs more column parallelism: 256 invocations when
   limits/K permit, with a verified 64-thread core fallback. Quantized rows keep 64.
4. Plan resolves aligned Q4_1/Q5 variants from the validated weight offset. Their
   20/176-byte blocks stay aligned and do not need cross-word joins. The generic
   two-byte path remains supported/tested. Q4_0/Q6 cannot use that shortcut.
5. Balanced capped 2D geometry avoids large padded grids and now really exercises
   the 65537-row fixture on this device. The guard is uniform before all barriers.

The one-wave baseline already had **no emitted hardware `s_barrier`**. The selected
Q6 also has none: an expensive workgroup-barrier story would be wrong. Selected
ISA shows vector 128-bit loads; the compiler folds exact half-to-FP32 scale products
into mixed-input FP32 multiplication with zero addend. That is not FP16 dot
accumulation or input quantization. Products/decoded weights retain their original
strict equations. FMA accumulation, larger row tiles and 8-byte payloads were tried
but not selected. All 32 successful tuning configurations passed the original
48 fixtures and 11 complete real-model shapes; all regressions/failures are retained.

## Setup and boundaries

RX 7900XTX gfx1100, RADV 26.2.3, Ryzen9 3900X, CPU 10 affinity, unchanged powersave
CPU governor/natural GPU clocks. No driver/package/clock/power changes. Zig 0.16.0
ReleaseFast, native CPU target, raw Vulkan/libc only in the native executable.
Offline pinned glslc/spirv-val generate eight core-Shader-only modules. No optional
FP16 storage/arithmetic, subgroup feature, foreign math or runtime shader compiler.

Same SHA-pinned Qwen3.8-27B-Q4_0 GGUF and complete 11 dense shape/type pairs as
[the original report](2026-09-22-gpu-matvec.md). Same weights and FP32 x values.
Reference: installed ggml 0.24.0 dirty 456172ec, llama build 10964/b29c606e; exact
libraries, driver/toolchain, model, adapters, compiled binaries and source trees
are pinned in each manifest. This is not a claim against the strongest possible
server/backend/build on this card.

- `native-baseline`: original scalar code rebuilt from the checked-in historical
  snapshot; its shaders recompiled byte-identically, both native test modes passed.
- `native`: optimized, **same original stress placement** (weights offset 2, F32
  offset 4; x follows rounded weights + 16 and is only 4-byte aligned for full shapes).
- `native-aligned`: separately labelled offset 0 placement; full-shape x is naturally
  16-byte aligned. This selects direct-word Q4_1/Q5 modules. Never substitute this
  layout when calculating the isolated baseline-to-optimized stress speedup.
- `reference-f32`: prebuilt one-node ggml graph, `GGML_VK_DISABLE_MMVQ=1`, FP32 x.
- `reference-default`: retained precision control; quantizes activations to Q8_1
  where MMVQ is selected. Not the precision-matched denominator.

All resident allocation/upload/pipeline compile, independent CPU dots, readback,
hashing and output checks are outside timing. Native reuses recorded commands and
synchronously submits/waits. Reference synchronously computes a prebuilt graph but
still manages graph/command state per call: **host work differs**. These are
submit-to-completion CPU timings, not isolated GPU timestamps/hardware counters.
Three warmups plus initial execution, 7 trials ×3 alternating rounds, repeated in
a fresh run. 10 iterations/trial Q6, 256 F32, 64 otherwise. No concurrent benchmark
jobs; no claim of an exclusive desktop or locked clocks. Raw variance is retained.

## Complete results

All values below are **median microseconds** across 21 trial means per engine/run.
Smaller is better. σ is the sample standard deviation of those means, not a
request latency percentile. Full min/max/σ and raw logs accompany both summaries.

### Final repeat

| Format K×M | Scalar baseline | Optimized stress | Aligned native | Ref FP32 | Ref default† | Stress speedup | Stress σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| q6_k 5120×248320 | 5611.892 | 1220.216 | 1221.566 | 1253.047 | 1252.976 | 4.60× | 5.375 |
| q4_0 5120×6144 | 203.947 | 81.647 | 81.729 | 84.734 | 85.535 | 2.50× | 3.423 |
| q4_0 5120×10240 | 263.261 | 111.312 | 111.598 | 106.861 | 106.534 | 2.37× | 1.740 |
| q4_1 17408×5120 | 504.006 | 140.860 | 136.317 | 169.335 | 125.686 | 3.58× | 7.621 |
| q4_0 5120×17408 | 339.103 | 137.405 | 138.480 | 142.382 | 139.268 | 2.47× | 6.102 |
| f32 5120×48 | 75.232 | 50.580 | 50.331 | 67.154 | 67.128 | 1.49× | 0.982 |
| q5_k 6144×5120 | 310.818 | 99.203 | 83.579 | 86.232 | 86.472 | 3.13× | 2.713 |
| q4_0 5120×1024 | 112.714 | 57.612 | 55.949 | 69.876 | 72.312 | 1.96× | 5.539 |
| q4_0 6144×5120 | 207.961 | 84.419 | 86.982 | 83.496 | 86.008 | 2.46× | 3.770 |
| q4_0 5120×12288 | 283.553 | 122.334 | 123.675 | 118.386 | 118.635 | 2.32× | 2.128 |
| q4_0 17408×5120 | 370.764 | 144.989 | 140.101 | 150.828 | 136.448 | 2.56× | 8.352 |

† Default reference precision differs; retained as a quality/timing control.

### Independent final run1

| Format K×M | Scalar baseline | Optimized stress | Aligned native | Ref FP32 | Stress speedup |
|---|---:|---:|---:|---:|---:|
| q6_k 5120×248320 | 5686.703 | 1226.223 | 1233.193 | 1276.158 | 4.64× |
| q4_0 5120×6144 | 208.523 | 82.119 | 83.073 | 86.262 | 2.54× |
| q4_0 5120×10240 | 261.575 | 111.982 | 112.416 | 105.222 | 2.34× |
| q4_1 17408×5120 | 504.049 | 142.127 | 139.403 | 175.764 | 3.55× |
| q4_0 5120×17408 | 338.021 | 142.055 | 140.839 | 141.584 | 2.38× |
| f32 5120×48 | 76.218 | 50.135 | 51.090 | 68.036 | 1.52× |
| q5_k 6144×5120 | 309.454 | 99.741 | 86.198 | 89.440 | 3.10× |
| q4_0 5120×1024 | 112.144 | 56.284 | 56.399 | 69.448 | 1.99× |
| q4_0 6144×5120 | 211.920 | 87.223 | 90.019 | 86.590 | 2.43× |
| q4_0 5120×12288 | 289.216 | 123.711 | 123.824 | 119.566 | 2.34× |
| q4_0 17408×5120 | 367.210 | 143.134 | 148.819 | 151.245 | 2.57× |

### Interpretation and losses

- Q6 gains ~4.6× versus scalar in both final runs; the earlier pre-alignment paired
  runs also measured ~4.61×. Its 1.220 ms stress median is~2.6% below 1.253 ms reference
  in the final repeat. This includes the unequal host boundary, not a GPU-only win.
- F32 gains ~1.5× versus scalar and is~25% below reference at this component boundary.
- Q4_1 gains ~3.6× versus scalar. FP32 native remains faster than FP32 reference, but
  default activation-quantized reference is faster than either native placement.
- Q5 stress still loses: 99.2 versus 86.2µs (~15%). Aligned native is 83.6µs in repeat
  and 86.2µs in run1, versus 86.2/89.4 reference respectively. The small favorable
  median differences overlap observed variation; call this **near parity**, not a
  decisive kernel-only victory. The stress-to-aligned improvement is material,
  but changes placement as well as selecting the aligned shader.
- Q4_0 5120×10240, 6144×5120 and 5120×12288 still lose to reference in the final repeat.
  Some small differences change sign or overlap dispersion in other runs. Do not
  collapse this table into an unsupported “faster on every shape” claim.
- Larger row tiles and 8-byte decoding regressed; FMA accumulation had small/mixed
  effects. No production knob for rejected variants remains. The original loss,
  all exploratory results, intermediate paired runs and noisy trials are preserved.

## Verification, provenance and replay

- 42 CPU tests and 8 real-GPU tests pass Debug and ReleaseFast; 48 Python tests and
  fmt pass. Each of 48 independent fixtures runs at both address layouts, with
  guards/replay/changed x. F32 local-size fallback is forced through each limiting
  host property and checked independently. The65537-row fixture is explicitly 2D.
- Exact isolated-column/cancellation/zero cases and 380928 finite-half outputs per
  layout are unchanged. General acceptance remains
  `abs(actual-ideal) <= 2e-6 + 2e-6*sumabs`; no tolerance relaxation.
- Every full shape/output passed in both final runs. Maximum native error/sumabs
  was **1.1363624e-8**, nonfinite outputs 0. Default reference normalizedL2 reaches
  **0.00407152**, which is why its timings are not FP32 precision-matched evidence.
- Fresh final replay verifies 62 retained research sources, regenerates the exact
  independent fixture and all 8 SPIR-V modules byte-identically, then runs the
  native/Python/fmt gates. [Replay logs](../research/2026-09-22/matvec-final-replay/).
- Both final source snapshots (132 files each) and 542 root artifacts/run were
  rehashed: **1348 checks**, also verifying the live current source matches both
  snapshots. [Audit](data/2026-09-22-matvec-dfs/final-hash-verification.json).
  The earlier paired stage passed 1146 source/artifact checks; tuning 576 checks.
- Baseline rebuild [manifest/log](data/2026-09-22-matvec-dfs/baseline-rebuild-manifest.json)
  retains compiler/source identities. Surviving old binaries were not used to
  reconstruct it. Research/oracle material is not a production dependency.

Commands actually executed (use fresh suffixes to repeat):

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
python3 bench/rebuild_matvec_baseline.py \
  --output third_party/matvec-dfs/rebuilt-baseline-final
python3 bench/run_gpu_matvec.py --cpu 10 --aligned \
  --baseline-run third_party/matvec-dfs/rebuilt-baseline-final \
  --output docs/bench/data/2026-09-22-matvec-final-run1
python3 bench/run_gpu_matvec.py --cpu 10 --aligned \
  --baseline-run third_party/matvec-dfs/rebuilt-baseline-final \
  --output docs/bench/data/2026-09-22-matvec-final-repeat
python3 tools/replay_matvec.py --output-dir .tools/matvec-final-replay
```

Rebuild restores source into a fresh ignored directory, validates hashes, compiles
old shaders, and runs both test modes. Ordinary tests use owned fixtures/modules
without external libraries. Explicit replay needs the existing pinned tools/model/
reference libraries; no package install, driver change or weight download occurs.
For tuning, the [experiment ledger](data/2026-09-22-matvec-dfs/experiments.json)
points to each candidate's source, exact command/defines and individual manifest.
`bench/tune_matvec.py` rebuilds the diagnostic executable, verifies 48 fixtures,
then checks/times all 11 existing SHA-pinned full cases. Original large corpus:
`third_party/gpu-matvec-bench/2026-09-22-gpu-matvec-repeat/`; the original benchmark
runner reconstructs it from the pinned GGUF if needed. Large experiment modules/
binaries/outputs are under `third_party/matvec-dfs/`, with hashes in the manifests.

Raw data:
- [Final run1](data/2026-09-22-matvec-final-run1/), [final repeat](data/2026-09-22-matvec-final-repeat/).
- [Earlier paired run1](data/2026-09-22-matvec-optimized-run1/), [repeat](data/2026-09-22-matvec-optimized-repeat/): identical six generic/small modules before aligned specialization.
- [33 tuning attempts /32 passed](data/2026-09-22-matvec-dfs/experiments.csv), [source/ISA/failures](data/2026-09-22-matvec-dfs/).
- [Aligned Q5 compiler dump](data/2026-09-22-matvec-dfs/aligned-q5_k-isa.txt) has 14 static buffer loads versus 19 generic, not a hardware traffic counter; [profile command/pins](data/2026-09-22-matvec-dfs/aligned-profile-manifest.json).

Initial harness execution-mode failure and the test-package embed compilation
failure are retained with fixes in the research report. No native numerical
failure was discarded. Block08b's investigation/rebuild/test/repeat/replay loop is
closed; remaining Q4 and two-byte Q5 losses are explicit. **Do not resume block09
or serving without authorization.**
