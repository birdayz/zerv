# Q4_0 / Q8_0 scalar decoding research

**Scoped research completed 2026-09-22 before native implementation.** This is the
first diagnostic primitive, not a claim that all target model tensor types are
supported or that a scalar decoder is the production GPU fast path.

## Sources inspected

- ggml source commit `456172ec733a135778adcd32d00e576a58232e45`:
  [block layout](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-common.h),
  [decoding operations](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-quants.c).
- Installed external oracle `/usr/lib/libggml-base.so.0.24.0`, Arch `ggml 0.24.0-1`.
  Its API reports version `0.24.0`, commit **`456172ec-dirty`**; do not describe it
  as a clean upstream build. Exact binary SHA-256:
  `7d9065538f5df6342613b4fa92e661d5ad8fd811c2dbe16ff0e4b62a77777073`.
- Both exported row decoders were confirmed present. A scale=1 Q4 row was invoked
  independently and produced the expected low-half/high-half ordering.

This installed library is an **external fixture-generation tool only**. The Zig
module/test runner must not import C headers, link this library/libstdc++, or need
it when consuming committed goldens. No upstream implementation is copied in.

## Complete operation for the bounded scope

Each block reconstructs 32 numbers. Serialized bytes are little-endian.

- Q4_0: 18 bytes = 2-byte IEEE binary16 scale + 16 packed bytes. For payload byte
  `b[j]`, output `j` is `scale × ((b[j] & 15) - 8)`; output `16+j` is
  `scale × ((b[j] >> 4) - 8)`. Nibbles are **not interleaved** in output.
- Q8_0: 34 bytes = 2-byte binary16 scale + 32 signed two's-complement int8 values.
  Output `j` is `scale × signed(payload[j])`.
- Every finite binary16 value converts exactly to binary32. Multiplication by any
  relevant integer needs at most 19 significant bits and is exactly representable
  as finite binary32. Thus **bit-exact FP32 results, including signed zero**, are
  the acceptance rule, with strict floating-point semantics. This rule is specific
  to these scalar products; it does not extend to dot-product reductions.
- Both positive and negative scales are valid; zeros and binary16 subnormals are
  valid. Exponent bits all ones identify non-finite scales; reject them in our
  bounded API as malformed weight data rather than passing NaNs/Infs downstream.
- Validate whole-row input length, exact output length, and all scales before
  writing any output. Empty input/output is valid. Reject incomplete blocks and
  length mismatches. Byte input may be unaligned; use byte reads, not pointer casts.
- Source and destination must not overlap; caller owns both buffers. No allocation,
  FFI, device work, or asynchronous lifetime in this scalar operation.

No unresolved semantic questions remain for this operation. K-quants, quantizing
weights, tensor strides, matrix multiplication, GPU dispatch, and file parsing are
outside this scope and need their own research/specs.

## Reference mechanism

The fixture generator is a separate Python process. It opens the explicitly named
external library, records its actual SHA/version/commit, and binds only:

```text
void dequantize_row_q4_0(const void *blocks, float *out, int64_t count)
void dequantize_row_q8_0(const void *blocks, float *out, int64_t count)
```

Validate host little-endianness, 4-byte C float, alignment, byte sizes, and counts
before calls. Buffers remain alive until synchronous return. No model weights or
network access are required. Cross-check oracle bytes against Python binary16
unpacking and scalar multiplication, then emit stable JSON goldens, not code.

Coverage: all **63,488 finite binary16 encodings**, times every Q4 signed value
and every Q8 signed value. Q4 uses each low nibble `j` and high nibble `15-j`;
Q8 uses eight blocks to cover all 256 int8 encodings. Compare SHA-256 of the
concatenated canonical little-endian FP32 outputs in ascending scale-bit order.
This covers every scalar product, not every possible multi-block input sequence.
Additional explicit multi-block byte/output fixtures cover lane/block ordering;
invalid-input tests exercise our rejection policy independently of ggml assertions.

The generated manifest must include pattern/version, case/value counts, oracle
identity, generator hash, and output fingerprints. Goldens are produced before
native code; the native tests consume them without loading the oracle. Golden
regeneration must be explicit and refuse accidental overwrite.

## Reference extraction verified before native code

Executed successfully on 2026-09-22:

```sh
python3 tests/reference/generate_quant_goldens.py \
  --library /usr/lib/libggml-base.so.0.24.0 \
  --output tests/fixtures/quantization.json
```

Every external-oracle output matched independent Python scalar arithmetic exactly.
Q4: 63,488 scales / 2,031,616 values, output SHA-256
`449e72de0b7ea1c1361fdfe61711b91a874f2d82879d668a51737a091714dee1`.
Q8: 63,488 scales / 16,252,928 values, output SHA-256
`af57e84914fd977b6e0b876298a736878546522f73d4321b788c99fe62253e96`.
Explicit 11-block examples also passed. The golden contains binary/generator
provenance; future native code must match it, not generate its own expected output.
