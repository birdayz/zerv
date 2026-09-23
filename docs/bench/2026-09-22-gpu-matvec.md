# 2026-09-22 — native resident GPU matvec vs installed ggml Vulkan

## Result and scope

**Correct native packed-weight projections, slower than the reference on every
measured shape. No native model execution or serving claim.** All eleven complete
base-model dense shape/type pairs pass independent CPU and FP32-input reference
GPU checks, including the entire Q6_K [5120,248320] output projection (not a slice).
The first 64-thread/shared-reduction implementation is a diagnostic baseline, not
our performance objective achieved. Keep its losses for subsequent optimization.

[Research](../research/gpu-matvec.md), [contract](../specs/gpu-matvec.md),
[fixture/native validation](../research/2026-09-22-gpu-matvec-validation.md).
Raw runs: [first](data/2026-09-22-gpu-matvec-run1/manifest.json),
[repeat](data/2026-09-22-gpu-matvec-repeat/manifest.json). Each includes all stdout,
stderr, 110 numerical records, full source snapshots, library/build identities,
workload/input hashes and [per-trial summary](data/2026-09-22-gpu-matvec-repeat/summary.json).
Large cases/output binaries retained in `third_party/gpu-matvec-bench/<run-id>/`,
with every file hash in each manifest. Rebuild from source with the runner below.

## Setup and fairness

Same SHA-verified Qwen3.8 Q4_0 GGUF, Ryzen3900X / RX7900XTX / RADV26.2.3,
Zig0.16 ReleaseFast native target; CPU10, unchanged powersave governor. Three
warmups after one initial execution, seven trials, three alternating engine rounds;
21 observations per engine/workload/run. 256 calls/trial for F32 [5120,48], 10 for
full Q6 vocabulary, 64 for other shapes. One fixed signed dyadic vector per width
(index LCG32/xor, recorded formula); weights and inputs resident and reused.
Pipeline/graph creation, allocation, uploads, independent CPU calculation,
readback/comparison and file I/O are outside timings. Wall time includes submit and
synchronous completion; these are **not timestamped kernel-only timings**.

Native links Vulkan/libc/loader only; no foreign math or inference library. The
external C adapter links pinned installed ggml Vulkan/base (dirty build identified
by exact binary hashes). Its graph is prebuilt/allocated, but Vulkan's public
graph-compute API still handles graph contexts and command recording internally;
no reusable graph-plan API is implemented there. Native replays a recorded command
buffer. This is the closest exposed equivalent math/ownership boundary, **not
identical host work**. Driver/library/device details and allocations are retained.
Native uses one combined resident allocation with disjoint W/x/y views, host upload
staging and small output readback; requirement-byte totals include host allocations,
not exclusively VRAM. Reference allocation/scratch policy differs and is not hidden.

Matched precision: native FP32 inputs versus `GGML_VK_DISABLE_MMVQ=1` reference.
Also measure the unmodified reference default, which quantizes activations to Q8_1
for these Q4/Q5 shapes on AMD but not Q6_K/F32. Default is a **separate precision
control**, not a fair equal-quality denominator to manufacture a speed ratio.
All other GGML_VK_* overrides are removed for each reference process. Driver/global
environment and hardware snapshots are recorded; no clocks, power, packages or
system settings changed. Desktop activity remains, no exclusive GPU isolation.

## Measured microseconds per synchronous call

N = native; R = reference with FP32 input; D = default reference. Shapes [K,M].
Every entry is the median of 21 trials, derived from raw JSON.

| Type / shape | First N | First R | Repeat N | Repeat R | Repeat D | Repeat N/R |
|---|---:|---:|---:|---:|---:|---:|
| q6_k [5120,248320] | 5650.225 | 1251.435 | 5624.886 | 1254.846 | 1256.290 | 4.483× |
| q4_0 [5120,6144] | 204.101 | 85.274 | 206.842 | 84.755 | 85.411 | 2.440× |
| q4_0 [5120,10240] | 263.710 | 112.700 | 255.971 | 111.560 | 111.697 | 2.294× |
| q4_1 [17408,5120] | 505.910 | 169.697 | 503.806 | 170.291 | 131.689 | 2.958× |
| q4_0 [5120,17408] | 336.265 | 137.316 | 336.174 | 139.344 | 135.154 | 2.413× |
| f32 [5120,48] | 75.989 | 68.080 | 75.049 | 67.284 | 67.243 | 1.115× |
| q5_k [6144,5120] | 328.210 | 86.661 | 307.923 | 85.223 | 85.943 | 3.613× |
| q4_0 [5120,1024] | 112.146 | 67.833 | 112.173 | 70.049 | 70.711 | 1.601× |
| q4_0 [6144,5120] | 225.453 | 86.167 | 207.413 | 85.004 | 84.973 | 2.440× |
| q4_0 [5120,12288] | 290.726 | 119.563 | 281.403 | 117.480 | 117.762 | 2.395× |
| q4_0 [17408,5120] | 361.391 | 150.584 | 360.777 | 149.917 | 135.573 | 2.407× |

### Variance, retained rather than discarded

Repeat min..max µs and sample standard deviation µs (all 21 trials):

| Type / shape | Native min..max (SD) | FP32 reference min..max (SD) |
|---|---:|---:|
| q6_k [5120,248320] | 5411.976..6516.863 (260.097) | 1238.692..1274.585 (8.878) |
| q4_0 [5120,6144] | 185.945..275.366 (26.278) | 84.079..88.398 (1.004) |
| q4_0 [5120,10240] | 228.151..357.008 (40.157) | 108.294..113.854 (1.314) |
| q4_1 [17408,5120] | 493.975..676.799 (61.542) | 166.260..175.854 (3.136) |
| q4_0 [5120,17408] | 317.749..477.815 (54.173) | 131.874..142.703 (3.453) |
| f32 [5120,48] | 74.599..76.256 (0.442) | 66.889..67.663 (0.233) |
| q5_k [6144,5120] | 267.432..443.223 (56.365) | 83.779..88.242 (1.601) |
| q4_0 [5120,1024] | 108.899..113.391 (1.381) | 67.783..74.800 (1.386) |
| q4_0 [6144,5120] | 191.923..274.546 (27.033) | 83.327..86.992 (1.014) |
| q4_0 [5120,12288] | 250.629..417.247 (48.978) | 116.471..132.837 (3.390) |
| q4_0 [17408,5120] | 345.447..519.857 (57.955) | 141.151..151.180 (2.106) |

No stable speedup is claimed. Native Q6 is ~4.5× slower, Q5 ~3.6×, and the large
Q4_1 projection ~3.0× on repeat. Byte-wise extraction/exact half conversion and
shared-memory reduction are optimization candidates, not yet profiler-proven
causes. No loss was removed from either run. Even the small F32 projection loses.

## Numerical and replay evidence

48 independent synthetic/actual-row cases were generated and replayed identically
**before native implementation**. Includes six exhaustive finite-half fields
(380928 outputs), column/row/block placement, exact cancellation and zero vectors,
seeded wide reductions, 65537-row dispatch tail, and actual-model shape samples.
Native hardware tests also replay after download, change input without rerecording,
check nonoverlap/output sentinels and resource lifetimes. Full actual-shape
benchmark results are checked against independent external decoded-weight CPU
long-double dot and sumabs; the FP32 GPU reference must pass too.

Max native error/sumabs on full shapes was `1.7049784054129182e-8` on both runs
(denominator floor1e-6), below the predeclared per-row `2e-6 + 2e-6*sumabs` bound.
Nonfinite outputs: zero. Per-case max absolute/relative, normalized L2 and sumabs
errors are in manifests; exact fixture cases are checked without numerical epsilon.
Default reference activation quantization reached normalized L2 error
`0.004071521400469705` versus the decoded-weight CPU ideal. This is operator
precision evidence for these vectors, not model quality/perplexity validation.

All source snapshots and artifact hashes rechecked after both runs:
469 files in the first run (125 source +344 artifacts), 470 in repeat (126+344).
The one source-count difference is the added Python fixture/benchmark-gate test;
measured native/C implementations and shader bytes are unchanged. See each run's
`snapshot-verification.json`. Native runtime dependency audit passed.

## Re-run from source

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
# Fresh output paths; verifies/restores only small pinned research sources when opted in.
python3 tools/replay_matvec.py --output-dir .tools/NEW-matvec-replay --restore-sources
python3 bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-run1
python3 bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-repeat
```

No old executable is required. Pinned system libraries, shader tools, local Zig
and the existing verified model are prerequisites; mismatches fail explicitly,
no automatic package install/weight download. The replay tool compares independently
regenerated fixtures and SPIR-V bytes, then runs fmt, Debug/ReleaseFast CPU+GPU and
Python tests. Verified replays: `.tools/gpu-matvec-replay` and final expanded-test replay
`.tools/gpu-matvec-replay-final`; manifests/logs copied under the corresponding
`docs/research/2026-09-22/gpu-matvec-replay*/` directories.

There is no arbitrary resident-matvec llama-server endpoint. A component GPU library
comparison is not a server benchmark; actual tuned llama-server remains mandatory
at model-session and HTTP milestones. Next block: GPU normalization/gates/reductions,
including resolution of official `output_gate_type` semantics before implementation.
