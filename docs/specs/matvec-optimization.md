# Matvec DFS optimization — block08b, before implementation

Serving and block09 are paused by user request. Optimize only matvec with the
same packed weights, FP32 inputs and predeclared numerical/exact fixture gates.
No activation quantization, tolerance relaxation, runtime foreign math, drivers/
clocks/power changes. Preserve the original implementation and reproducible rebuild.

## Experimental sequence and acceptance

1. Preserve baseline sources/binary and inspect process-local RADV shader dumps;
   compiler disassembly is not a GPU hardware-counter profile. First isolate exact
   hardware half unpack versus manual conversion without changing work assignment.
2. Implement independently derived block-oriented traversal: reuse each packed
   word and half/subscale for several coefficients, independent FP32 partial sums;
   tune lanes per row, rows per group, and block distribution. Avoid copying any
   reference implementation. Keep the generic strict scalar kernel as a fallback.
3. Test fused multiply-add only for dot accumulation (decoded weights remain strict);
   evaluate with unchanged exact cases and original mixed-error bounds. If not
   selected, do not leave a dead production tuning option.
4. Sweep measured configurations, not guesses. Every candidate must pass all 48
   independent fixtures and full actual-shape output gates before acceptance. Retain
   failed candidates, hashes, parameters, diagnostics and losses. Shared reduction
   must have uniform barriers even for partial row groups. Bound workgroup limits,
   byte-offset arithmetic and uint32 word accesses including two-byte alignment.
5. Choose per-format/shape paths at Plan construction; no dynamic per-weight format
   dispatch, per-token allocations, repacking, readbacks or descriptor mutations.
   Preserve public ownership, byte-view/alias contracts and finite-weight policy.
   New dispatch tiling may cap X to65535 to explicitly exercise two-dimensional
   dispatch even when the actual device allows a larger X count.

Core `unpackHalf2x16` needs no Float16 arithmetic/storage capability. All finite half
patterns must still match exactly. No subgroup-specific fast path without queried
support and a validated fallback; initial candidates retain core Shader capability.
Experimental compiler options belong in the test/benchmark tool, not runtime knobs.

## Measurement/replay

A diagnostic executable may replace the test workload pipeline before recording,
with externally compiled owned shader bytes and a validated rows-per-group value.
It must use identical resident buffers, offsets, commands and synchronous fences
for baseline/candidate; record compilation/setup outside timing. No correctness
claim for a candidate based solely on timing. Existing independent CPU/ggml GPU
outputs stay immutable. Final measurements: fresh paired baseline/candidate and
FP32-reference runs, all eleven complete dense shapes, 3 warmups /7 trials /
3 alternating rounds, repeated. Keep default-reference precision control separately.
Record implementation/SPIR-V/binary hashes, source snapshots, input/output hashes,
error metrics, variance, CPU affinity and driver environment. Rebuild baseline from
retained checked source snapshots, not only a surviving executable.

Existing spec's one-64-thread-group-per-row is baseline scheduling, not mathematical
semantics. Update it to the verified selected scheduling after experiments. CPU and
GPU Debug/ReleaseFast, Python, fmt, shader validation and replay gate completion.

## Selected integration contract (after 30 passing tuning configurations)

Use four-byte packed loads with two-byte alignment handling, core half unpack,
eight independent strict FP32 accumulators and compile-time-unrolled K-group
traversal. Keep 64 invocations and one row per group for quantized weights; wider
row tiles, 8-byte payloads and FMA were tested but are not selected. No new GPU
feature or subgroup requirement. F32 uses 256 invocations for K>=1024 when local
size/invocation/shared-memory limits permit, otherwise the same strict shader
compiled for64. This small/core path is also independently fixture-tested. The
original scalar implementation remains a source-rebuildable counterfactual under
the pre-DFS benchmark snapshot, not a dead runtime tuning switch.

Cap group X to65535, compute Y=ceil(M/cap), then X=ceil(M/Y), bounded by queried
limits. This balances the final grid (at most Y-1 surplus groups), rather than
launching thousands of empty rows. A row guard is workgroup-uniform and precedes
all shared reduction barriers. K=1,M=65537 now really executes a 2D grid on this
device; the original device-wide X limit had allowed that fixture to stay 1D.
Read containing uint32 words only inside the logical multiple-of-four buffer;
extract exactly the four requested payload bytes, including two-byte-aligned
fields. No expanded/repacked weights, input quantization or tolerance changes.

## Alignment DFS extension (before aligned production path)

The original benchmark intentionally uses weight offset2 (F32 offset4), with
non-16-byte-aligned activations for the full shapes; the external reference uses
its own aligned tensor allocations. Preserve this stress comparison. Add a
separately labelled natural-alignment mode (weight offset0) and compare both,
never silently replace the old workload. Aligned diagnostics passed all48 exact/
reduction fixtures plus all11 full shapes. The generic aligned Q5 case was91.9µs;
a compile-time aligned word-load variant was80.5µs versus ~99µs in the original
stress run. This is exploratory evidence pending paired repeats.

Q4_1 has20-byte blocks and Q5_K has176-byte blocks, both multiples of4. If their
validated weight-view offset is a multiple of4, every four-byte payload field is
aligned and `word_at` may directly load one uint32. Select an offline-compiled
aligned module only for these two formats/offsets, before recording. Preserve the
generic two-byte path and its original benchmark. Q4_0/Q6_K blocks are18/210 bytes,
so even aligned starts still alternate alignment; do not apply that shortcut.
No precision change, repacking, specialization driver feature or runtime knob.
Run every independent GPU fixture in both alignment modes, including guards,
replay and changed input, and repeat all full-shape reference comparisons.
