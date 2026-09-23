# GPU matrix-vector block 08

Specification written before implementation, 2026-09-22. Now implemented with
[independent tests and repeated measurements](../bench/2026-09-22-gpu-matvec.md). [Research](../research/gpu-matvec.md).
This is not model inference or HTTP serving.

## Operation and boundary

`y[r] = sum(c=0..K-1, decode(W[r,c]) * x[c])`, resident row-major weights,
contiguous FP32 x/y, no transpose/bias/activation/batch. Types F32, Q4_0, Q4_1,
Q5_K, Q6_K; reject others. Packed layouts and strict FP32 weight decoding follow
quantization.md, q4_1.md, q5_k.md, q6_k.md. No runtime ggml/shader compiler dependency.
FP32 accumulation; no input quantization or silent FP16 intermediate.

Cohesive `src/matvec` depends on existing gpu infrastructure, not model/container
packages. Small Plan owns one fixed-type pipeline and prevalidated push constants;
borrows three GPU buffers through that pipeline. Buffers/Plan have stable addresses,
externally serialized, and outlive recorded commands. Destroy commands before Plan,
Plan before buffers. Record into caller's Commands; no submission or allocation in
record/dispatch, no hidden global/cache. Reusable submission remains caller-owned.
Finite waits/cancellation/device-loss semantics stay those of gpu.Commands.

K 1..32768, M 1..1048576; quant K must be a multiple of its block element count. All sizes/offsets
checked in u64, then fit uint32 shader addressing and physical storage-buffer limits.
Buffers' logical lengths must be multiples of4; weight offset even (F32 multiple4),
x/y offsets multiple4. Allow 2-byte packed rows and nonzero offsets. Reject any
output/read-view overlap on the same buffer; disjoint views may share a buffer.
Reject wrong-device, dead resources, insufficient extents and unsupported local-size
limits before recording. Exactly one workgroup per row: 64 invocations for packed
weights; F32 uses256 for K>=1024 when device local-size/invocation/shared limits
permit, otherwise64. Balanced 2D flattening caps X at65535 and guards surplus rows
uniformly before any barriers. Output outside its view must be unchanged.
[DFS selection contract](matvec-optimization.md) and archived scalar baseline
retain the precision gates; core half unpack/packed-word reuse need no optional
GPU capabilities. Q4_1/Q5_K views aligned to4 bytes select direct-word variants;
two-byte-aligned views retain the generic cross-word path. Selection is at Plan
construction, not a public precision/tuning option. No activation quantization or
FMA/reassociation of decoded weights.

Host weight-validation helper: exact byte length, all F32 weights finite, all global
half fields finite (Q6 trailing field!). No allocation/writes. Validation is required
before upload, not by reading GPU weights back at Plan creation. Caller must preserve
validated immutable weight bytes. x may change between executions via appropriately
synchronized GPU/host work. Numerical domain: finite normal/zero FP32 inputs/weights
and intermediates without overflow/FP32 underflow. Half subnormals are supported
because their FP32 representation is normal. Nonfinites and FP32-subnormal F32
weights reject in upload validation; no per-dispatch GPU data scan. Broader FP32
subnormal arithmetic is explicitly unsupported until float-controls are enabled.

## Independent oracle / fixture format (before native implementation)

External C executable only, links pinned installed ggml-base and ggml-vulkan.
Input: eight little-endian u32 words: magic 0x38564d5a, schema1, ggml type, K, M,
packed bytes, input bytes, reserved0; then exactly packed weights and K FP32 values.
Bounds above, no trailing/truncated data, verify actual ggml type traits. Construct
one `ggml_mul_mat` node with no-alloc context, GPU-allocate tensors, upload once,
use synchronous graph compute, copy output outside timing. CPU reference decodes
one row with ggml's independent decoder, accumulates products and absolute products
in long double. Output binary: M FP64 ideal dots, M FP64 sumabs, M GPU FP32 values.
Metadata/logs retain backend, version, active MMVQ setting and timings. Generator
requires pinned libraries/headers, refuses overwrites, records all source/input/output
hashes. It compares generated weight decoding with separate Python scalar formulas.

Fixtures: deterministic cases with parameters or explicit packed bytes/x, independent
CPU expected dots/sumabs and GPU outputs (or canonical exact output fingerprints),
shapes, seeds, payload hashes, generator and oracle provenance. Large half-domain
packed inputs are reconstructed from recorded base block + field + ascending finite
half bits, not committed as redundant megabytes. Native tests need no external engine.

- Isolated columns/packed low/high/sign/scale placement; distinct dyadic x per column.
- Every finite half in each global field, single nonzero x, exact decoded result.
- Seeded signed inputs, zeros, alternating signs/cancellation, multiple blocks,
  widths32/256/5120/6144/17408 as applicable, odd output row counts, 2-byte starts.
- Actual-model small rows from all dense shape/type pairs; full real dense shapes
  gate benchmarks, including the complete 248320-row output projection.

Exact cases require numerical FP32 equality (canonical +0 for either signed zero).
Other fixed cases require finite results and, for each row,
`abs(actual - ideal) <= 2e-6 + 2e-6 * sum(abs(decoded_weight * input))`.
This is a fixed-corpus acceptance threshold, not a worst-case theorem for all input
vectors. It accounts for reduction/reassociation but does not hide layout errors:
exact isolated/half-domain cases gate those independently. Report max absolute,
relative (denominator max(abs(ideal),1e-6)), error/sumabs (floor1e-6), normalized L2,
and nonfinite count. Check reference itself before native implementation. If the
reference fails, preserve evidence and investigate before revising scope/thresholds.
Default MMVQ results are a separate precision/quality control, not the FP32 golden.

Native tests additionally check all geometry/overflow/alias/type/finite-field errors,
resource lifetime/cleanup, output sentinels, reused commands after readback, changed
input, shared nonoverlapping views, and unchanged memory on rejected setup. Run
CPU and explicit GPU suites in Debug/ReleaseFast, shader validation, Python tests.

## Component performance gate

Preallocated resident packed weights, same FP32 x/y, same model tensor rows. Host
setup, pipeline compile, upload, CPU golden calculation, output copy/hash/comparison
outside timed loops. Native synchronous recorded submit/fence vs reference prebuilt
one-node synchronous graph (reference internally manages graph/command state each
call; report non-equivalent host overhead). Also retain reference default-MMVQ
measurements with precision metrics. Three warmups, seven trials, three alternating
rounds, repeat fresh run. Representative complete actual dense shapes/types, plus
small-output latency; full numerical outputs checked before accepting timings.
Capture CPU affinity/governor, GPU memory/temperature/clocks without modifying them,
versions/environment, source/binary/SPIR-V/artifact hashes, raw logs/variance/losses.
Harness must rebuild from retained source, not rely on a stale executable. Block
closes only with recorded repeated correctness-gated results. Actual tuned
llama-server remains mandatory at model-session/HTTP milestones.
