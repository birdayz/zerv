# Explicit SIMD decode experiment — 2026-09-22

Research before changing the validated CPU decoder. Baseline measured with the
repeatable harness: Q4 median 6.293 ms vs external 4.762 ms; Q8 6.277 ms vs 4.606 ms
per 10,485,760 values. These are CPU component results, not inference speeds.
Baseline code is retained at
`docs/bench/data/2026-09-22-quant-cpu/source/quant.zig` with its manifest hash.

## Hypothesis and fully scoped change

The initial implementation expresses each lane as scalar loads/conversions/stores.
Make independent lane operations explicit vectors so the compiler can lower them
to SIMD on the observed AVX2/F16C-capable Ryzen 3900X. Keep identical format,
checked lengths, whole-row finite-scale validation, aliasing/lifetime rules, and
strict floating-point semantics. No changes to the goldens/tolerances or workload.

Inspected Zig **0.16.0 language reference, Vectors and Relationship with Arrays**,
retained in `.tools/zig-x86_64-linux-0.16.0/doc/langref.html` (compiler archive hash
in development docs). It specifies elementwise vector arithmetic and shifts;
array/vector value coercion and fixed-length slice `.*` conversion are supported.
Vectors larger than native width can lower to multiple SIMD instructions.
**Array/vector pointer casts are not valid** because vector byte layout is not
specified. Therefore load/store via array values, never vector pointer casts or
stronger-than-input alignment assumptions.

- Q4: load 16 bytes as a vector value; low/high nibble extraction separately;
  widen to signed lanes before subtracting 8; convert to 16 FP32 lanes and multiply
  by broadcast scale; store low-half then high-half as two 16-value array writes.
- Q8: load 32 bytes, bitcast unsigned byte lanes to signed two's-complement lanes,
  convert/multiply 32 lanes, store 32-value array. Signed lane bitcast is not a
  pointer/layout cast.
- Every lane computes the already-researched scalar product with no reduction or
  reassociation. The same bit-exact scalar/reference goldens remain the oracle.
- CPU feature selection belongs to the compiler target, not hard-coded x86
  assembly. Other targets may scalarize; no portability/performance claim is made.

## Acceptance before retaining

Run full independent goldens in Debug/ReleaseFast, malformed/unaligned/atomic-error
cases, and the same interleaved repeated component benchmark with unchanged input /
output hashes. Retain only if correctness is unchanged and measured performance
improves; preserve losses/noise. Inspect emitted SIMD instructions to verify the
hypothesis rather than assuming source vectors imply efficient machine code.
The function remains a CPU diagnostic component, not a GPU compute path.
