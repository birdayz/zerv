# Block 08 matvec validation log

## Independent readiness gate, before native implementation

2026-09-22: `tests/reference/gpu_matvec.c` built with system cc, strict FP,
installed-identical pinned headers, external ggml Vulkan/base libraries only.
`generate_gpu_matvec.py` verifies library/header/model hashes before extraction.
One-node GPU graph and independent CPU long-double dot/sumabs from external exact
weight decoding; Python scalar independently checks every explicit-case row.
GPU F32 input forced with `GGML_VK_DISABLE_MMVQ=1`, all other GGML_VK_* overrides
removed. RX7900XTX identity checked. Not a clean-source ggml build claim (dirty
installed version remains pinned by binary hash).

Initial 37-case corpus succeeded in `.tools/gpu-matvec-fixture1`, fixture hash
`d20743819c0d83fc6f5b22af35967654ceba074001a7037ab87333bd73d4d941`.
Added exact alternating-sign cancellation/zero-input cases for all five formats,
and 65537 output rows intended to exercise 2D dispatch. Correction from block08b:
the original device X limit was larger, so that fixture actually stayed1D; the
new capped/balanced grid and explicit test now exercise2D. No threshold adjustment.
Two full independent extractions (`.tools/gpu-matvec-fixture2`, `fixture3`) produce
identical final 48-case fixture bytes:
`724ceb2946f9341c95f4d3381b2d804fc759f453a7cbe7698c13a1534304f670`.
Six global-half cases each cover 63488 rows = 380928 exact outputs. All exact cases
pass; all other cases satisfy the predeclared per-row bound. Eleven actual-model
shape/type pairs each contribute first/middle/last rows, not whole matrices yet.
Raw case files, CPU/GPU outputs, logs and compile commands retained in work dirs.
No native shader existed at this gate.

Verified commands (fresh paths required; installed pinned references/model needed):

```sh
python3 tests/reference/generate_gpu_matvec.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir .tools/gpu-matvec-fixture2 --output .tools/gpu-matvec-fixture2.json
python3 tests/reference/generate_gpu_matvec.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir .tools/gpu-matvec-fixture3 --output .tools/gpu-matvec-fixture3.json
cmp .tools/gpu-matvec-fixture2.json .tools/gpu-matvec-fixture3.json
cmp tests/fixtures/gpu/matvec.json .tools/gpu-matvec-fixture2.json
```

Native implementation/tests/full-shape benchmarks remain the active work, not
claimed complete by the fixture result. No full-model or HTTP execution yet.

## Native implementation, checks and measured gate

Owned `src/matvec` GLSL + Zig interface now implement all five types with explicit
byte-view geometry, strict FP32 decoded weights/products/tree sums and no optional
GPU arithmetic/storage features. Five Vulkan1.1 modules compiled/spirv-val validated
and replayed byte-identically twice before hardware checks. SPIR-V inspection shows
only Shader capability and NoContraction annotations. Native tests pass all 48
fixtures, input hashes, exact half-domain output hashes, guards, replay after
readback, changed input and disjoint same-buffer views; resource/alias/device and
partial allocation rollback failures are checked too. CPU tests exhaust finite and
nonfinite half validation in each global field/block and cover F32 finite/subnormal
policy, shapes/overflow, addresses/dispatch bounds and malformed binary case inputs.

Initial CPU compile failed because a test array length used a non-comptime local
`width`; added `comptime` to that test's format block-width expression. No native
numerical mismatch or threshold change occurred. Shader numeric goldens were never
updated from native output.

Latest required gates actually executed: **42 CPU +7 GPU tests** Debug/ReleaseFast,
**48 Python**, fmt check. The earlier first replay had 41 CPU +6 GPU tests; the final
additional tests cover binary-case rejection and partial workload-allocation unwind.
Native/benchmark implementation and numerical expectations did not change.

`tools/replay_matvec.py --output-dir .tools/gpu-matvec-replay` succeeded, verifying
all 62 retained source hashes, a fresh independent fixture with the same hash and
all five shader modules, then running both suites. [Replay manifest](2026-09-22/gpu-matvec-replay/manifest.json)
and [full commands/output](2026-09-22/gpu-matvec-replay/commands.log). Missing-source
network restore is opt-in; this replay verified existing files, not a fresh network
restore. Native builds/tests use committed files without `third_party`/ggml.

Two full-shape real-model benchmark runs passed (110 checks each). Full vocabulary
and all other dense shapes satisfy the CPU numerical gate. Native is slower on
**every measured shape**, repeat full Q6 ~5.625ms vs reference FP32 ~1.255ms.
Default reference activation quantization is separately measured, never silently
substituted for the FP32 baseline. [Complete results/variance/replay](../bench/2026-09-22-gpu-matvec.md).
All 939 source/artifact snapshot hashes verified. No validation layer installed;
no claim of layer-clean or hardware fault-injection coverage. No native model/HTTP
execution. Block08 verification/measurement cycle closed; only block09 active next.

Final replay after the two added negative tests also passed: `.tools/gpu-matvec-replay-final`.
[Final manifest](2026-09-22/gpu-matvec-replay-final/manifest.json) and
[commands/results](2026-09-22/gpu-matvec-replay-final/commands.log) retain
42 CPU +7 GPU tests in both modes, 48 Python and identical independent outputs.
