# CPU Q4_0 / Q8_0 / Q4_1 / Q5_K / Q6_K decoding

Status: implemented and independently verified, including an explicit-SIMD CPU
variant. [Completed operation research](../research/quant-blocks.md) and
[SIMD research](../research/quant-vector.md) preceded their respective code changes.
This is diagnostic infrastructure, not a model loader or a GPU kernel.

## API

`decode(comptime format, encoded: []const u8, output: []f32) -> error!void`, where
format is `q4_0`, `q8_0`, `q4_1`, `q5_k` or `q6_k`. K formats have their own
[Q5_K](q5_k.md) and [Q6_K](q6_k.md) packed-subscale contracts. Q4_1 is specified and verified in its
[extension contract](q4_1.md). Format dispatch is resolved before the row loop. All input blocks form one contiguous row; output holds
exactly `blockElements(format)` values per block: 32 for Q4_0/Q8_0/Q4_1,
256 for Q5_K/Q6_K. `blockBytes(format)` returns 18/34/20/176/210 respectively.
Q6_K's global half is trailing at byte 208, not at the payload prefix.

- No allocations, foreign-library calls, or runtime dependency beyond Zig stdlib.
- Packed scale is little-endian binary16, including valid negative/subnormal/zero
  encodings. Q4 nibble order and signed Q8 conversion follow the researched format.
- Input may be byte-unaligned. Output is normal aligned f32 storage.
- Caller owns buffers; source and output storage must not overlap.
- Empty packed input plus empty output succeeds.
- Non-multiple block length: `InvalidBlockLength`.
- Non-exact output length: `InvalidOutputLength`.
- Any scale with all-one exponent: `NonFiniteScale`.
- All validation precedes all writes. On any error the output remains unchanged.
- Strict FP semantics: preserve signed zero; results must match the oracle's
  binary32 bits exactly. No approximation/fast-math mode is permitted for this API.

## Fixture-generator contract

`tests/reference/generate_quant_goldens.py` is an explicit external development
program, never invoked as an engine dependency. It accepts a library path and
output path, refuses overwriting an existing output, and requires little-endian
IEEE float storage. No network or package installation occurs. It records the
actual oracle version/commit/binary hash, including dirty provenance.

Before writing fixtures it compares every oracle output with independent Python
binary16/scalar arithmetic. Any mismatch aborts. Output JSON has a schema version,
generator SHA-256, fixed pattern definition, counts, finite-domain output hashes,
and explicit multi-block input/output hex cases for both formats. The generator
writes only when every comparison succeeds.

Finite-domain order: ascending binary16 bit patterns, skip exponent 31. Q4: one
block per scale with payload `j | ((15-j) << 4)` for j=0..15. Q8: eight blocks per
scale; concatenated signed coefficients come from byte values 0..255. Fingerprint
canonical little-endian FP32 bytes. Expected counts are 63,488 scale cases,
2,031,616 Q4 values, and 16,252,928 Q8 values.

## Acceptance

Native tests consume committed goldens without ggml installed. They verify explicit
fixture bytes and both exhaustive output hashes, including case/value counts.
They additionally cover all 2,048 non-finite binary16 encodings, length errors,
empty rows, unaligned input, and failure atomicity when a later block is invalid.
Both Debug and ReleaseFast must pass; optimization cannot change the exact results.

This finite test suite exhausts all scale/coefficient products, **not** all possible
rows, tensor layouts, or model behavior. No inference/performance claim follows
from passing it. Row placement cases and explicit API invariants complement it.
