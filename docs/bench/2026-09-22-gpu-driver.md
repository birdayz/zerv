# 2026-09-22 — native Vulkan driver, real transfers and compute dispatch

Controlled block07 complete. [Spec](../specs/gpu-driver.md),
[research](../research/vulkan-driver.md),
[correctness, rerun commands and failures](../research/2026-09-22-gpu-driver-validation.md).
**Native driver infrastructure, not quantized GPU model operators or serving.**

## Verified scope

Device/compute queue selection, bounded allocation and coherent host mapping,
trusted SPIR-V pipeline/descriptors, explicit copy/barrier/dispatch recording,
reusable command buffers/fences, pending-state/resource ownership and cleanup.
No per-submit allocations or command rebuilding. Default CPU tests stay driver-free;
GPU executables link only system Vulkan/libc/ELF-loader directly, not ggml/llama/
shaderc or an inference runtime. LLVM+LLD are pinned development tools.

Independent C-header ABI fixtures cover **43 structures / 61 constants / every
field offset**. External C driver orchestration and independently scalar-checked
GPU outputs preceded native implementation. Full exact output hashes for affine
sizes1,63,64,65,5120,65537,1048576 words plus 64-word untouched tails, and transfer
roundtrips256B/1MiB/64MiB. Shader/ABI/golden regeneration repeated identically.

Latest gates: **39 CPU tests + 3 explicit real-GPU tests pass Debug/ReleaseFast**;
**43 Python tests and format check pass**. Includes resource/budget rollback,
128-buffer/64-kernel/32-command limits, retained-reference limits, cross-device
rejection, invalid states/ranges/layouts/groups/modules, host access while pending,
finite fence polling, replay after readback, all previous quant/tokenizer gates.
No validation layer is installed; actual device loss/OOM was not induced.

## Important invalidated results

The first two output-correct runs selected host memory type10, flags0xce, in **both**
C and native harnesses. Allocation metadata inspection caught that these AMD optional
memory flags require deviceCoherentMemory, which was not enabled. The driver's
acceptance and matching outputs were not proof of valid API use. This violates
VUID-vkAllocateMemory-deviceCoherentMemory-02790; both manifests are explicitly
**invalidated**, not benchmark evidence. Raw outputs/timings/source snapshots remain:
[run1](data/2026-09-22-gpu-driver-run1/manifest.json),
[repeat](data/2026-09-22-gpu-driver-repeat/manifest.json).
A separate earlier sysfs-connector preflight failed before any timing:
[record](data/2026-09-22-gpu-driver/manifest.json).

The corrected policy accepts only base memory flags0x0f, rejecting protected/lazy/
optional types. Independent fixtures were regenerated before the native correction;
all numerical outputs and shader bytes stayed equal. Added explicit cross-replay
barriers and corresponding tests. All results below use this corrected implementation.

## Matched work and environment

Native Zig and independently authored external C use the same selected device/queue,
physical memory types, allocation requirement sizes, shader module, buffer lengths,
barriers and operation counts. Both use pre-recorded commands, preallocated mapped
host/device buffers and synchronous submit+10s-bounded fence wait. Pipeline creation,
allocation, host fill, hashing, output checks and affine readback are **outside timing**.
Full final outputs/tails gate acceptance; there is no readback after each timed affine
call. Native adds ownership/state/limit checks. Setup differs: native owns separate
pool/fence per command object, C owns one shared pool/fence; not a setup-time comparison.

3 warmups, 7 trials × 3 alternating rounds =21 observations/engine/workload. Affine
65/5120/1048576 words:1000/1000/100 calls per trial; transfer256B/1MiB/64MiB:
1000/100/10 roundtrips. Each timed call includes host submission + GPU completion,
**not GPU timestamp/kernel-only duration**. No arbitrary SPIR-V/transfer endpoint in
llama-server exists; direct raw-driver C is the equivalent component comparison.
Actual tuned llama-server remains required at model-session and serving milestones.

Ryzen9 3900X, RX7900XTX RADV/Mesa26.2.3, Linux7.2.6, CPU10, existing powersave;
Zig0.16 ReleaseFast/native/LLVM/LLD, GCC16.2.1 O3/native C. Native and reference both
select compute family1, host memory type5 (0x0e, coherent+cached system memory),
device memory type0 (0x01, device-local). No optional feature enabled. Largest case
owns four64MiB allocations:128MiB host +128MiB device, within explicit512MiB budget.
All sizes/type indices recorded and matched independently. Allocation rounding for
partial shapes is retained (e.g.260 logical bytes→272 allocated bytes per buffer).

No exclusive workstation isolation or clock changes. Corrected run temperatures
49→53°C, repeat52→53°C; pre-run GPU busy snapshots3–4%, post-run78–84% may include
our just-completed work. VRAM returned to the same1,111,130,112-byte baseline after
both runs. These are snapshots, not proof of zero competing activity or peak usage.
The first corrected run had substantially greater variance; retain it rather than
selecting only the quiet repeat.

## Results

Median microseconds per submit+completion; lower is better.

| Corrected run | Workload | Native | External C |
|---|---|---:|---:|
| First | affine65 words | 66.653 | 64.821 |
| First | affine5120 words | 66.438 | 64.349 |
| First | affine1048576 words | 89.002 | 82.442 |
| First | roundtrip256B | 86.902 | 90.388 |
| First | roundtrip1MiB | 431.031 | 467.566 |
| First | roundtrip64MiB | 12339.751 | 12276.338 |
| Repeat | affine65 words | 50.465 | 50.361 |
| Repeat | affine5120 words | 50.246 | 50.247 |
| Repeat | affine1048576 words | 62.585 | 61.930 |
| Repeat | roundtrip256B | 62.853 | 62.983 |
| Repeat | roundtrip1MiB | 351.388 | 332.245 |
| Repeat | roundtrip64MiB | 11700.834 | 11667.604 |

Repeat dispersion (µs):

| Workload | Native min–max / σ | C min–max / σ |
|---|---:|---:|
| affine65 | 48.109–51.659 / 0.824 | 50.135–50.794 / 0.201 |
| affine5120 | 48.806–50.961 / 0.698 | 50.017–50.790 / 0.228 |
| affine1048576 | 59.373–72.475 / 2.900 | 61.464–63.065 / 0.562 |
| roundtrip256B | 61.623–73.781 / 2.798 | 62.590–63.609 / 0.274 |
| roundtrip1MiB | 284.893–455.395 / 58.558 | 287.539–454.504 / 59.528 |
| roundtrip64MiB | 11575.927–12034.640 / 131.257 | 11585.403–11998.045 / 118.296 |

Most repeat medians are close; native loses **5.8%** on1MiB roundtrip (sign reverses
from the first run),1.1% on large affine and0.3% on64MiB. Small differences overlap
noise; no driver speedup claim. ~50µs submit+wait for tiny dispatches motivates
batching recorded operator sequences rather than synchronizing each scalar operation.
This is an observation for later graph work, not measured native inference performance.

## Replay and retained evidence

```sh
python3 bench/run_gpu_driver.py --cpu 10 --output docs/bench/data/NEW-gpu-driver
```

Fresh paths, no old executable needed. Pinned research headers/tools are explicit
prerequisites, restored/verified as documented in [development](../development.md).
Full build/test logs, commands, host/driver/compiler/module/dependency identities,
source snapshots, full output checks, allocation metadata, telemetry and per-trial
records are retained. Both corrected binaries and all source snapshots verified;
[record](data/2026-09-22-gpu-driver-core-repeat/snapshot-verification.json).

- [Corrected first manifest](data/2026-09-22-gpu-driver-core/manifest.json),
  [summary](data/2026-09-22-gpu-driver-core/summary.json).
- [Corrected repeat manifest](data/2026-09-22-gpu-driver-core-repeat/manifest.json),
  [summary](data/2026-09-22-gpu-driver-core-repeat/summary.json).

Next sole block: actual-shape quantized GPU matrix-vector operators and independent
numerical/reference-GPU comparisons. Native Qwen execution and Chat Completions are
still not implemented.
