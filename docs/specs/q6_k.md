# Q6_K CPU decoding — controlled block 06

Written before fixtures/native code. Pinned block_q6_K and dequantize_row_q6_K at
ggml 456172ec733a135778adcd32d00e576a58232e45 are retained under third_party and
inspected; [layout findings](../research/mixed-quant-blocks.md). The model's one
Q6_K tensor is output.weight, [5120,248320], 1,042,944,000 packed bytes.

## API and mathematics

quant.Format.q6_k uses 210 bytes / 256 values. Bytes 0..127 carry low nibbles,
128..191 two-bit high planes, 192..207 signed int8 subgroup scales, **208..209**
the LE binary16 global scale. Unlike the other formats, the half is at the end.
Validate that trailing half in every block before any output write. Preserve the
existing allocation-free/unaligned/nonoverlap/ownership/error-atomic contracts.
All finite halves and all signed int8 scale values are valid, including zeros.

For half h=0..1, group g=0..3, lane l=0..31: low byte index h*64+(g%2)*32+l;
low nibble for g<2, high nibble for g>=2. High bits are
(qh[h*32+l]>>(2*g))&3. Coefficient = (low_nibble | (high_bits<<4)) - 32.
Signed subgroup index is h*8+g*2+l/16. Output index h*128+g*32+l.
Strict FP32 `(global_scale * signed_subscale) * signed_coefficient`.
Half×[-128,127]×[-32,31] fits binary32 precision (largest non-power-of-two
significand growth is 127×31); preserve operation/sign ordering including signed
zero. Exact bits, no tolerance/fast-math. No GPU/matvec/inference in this block.

## Independent fixture plan

External pinned dequantize_row_q6_K plus independent scalar byte unpacking,
signed conversion and FP32 packing must agree for every generated result. Refuse
existing output; record generator/helper/library/model hashes; repeat generation
byte-identically. Native tests consume only embedded JSON, never the library.

- Every finite global half with 4 coefficient-phase blocks. Fixed subgroup scales
  [-128,-127,-64,-33,-32,-2,-1,0,1,2,31,32,63,64,126,127]; each subgroup's coefficient
  for phase p/lane l is p*16+(l%16)-32. Thus all 64 coefficients at each selected
  signed subgroup scale/global half: 253,952 blocks / **65,011,712 values**.
- Separate signed-scale fingerprint: each of 256 byte encodings at each of 16
  subgroup positions, for all 4 coefficient phases, global half=0x3555. This adds
  16,384 blocks / 4,194,304 values. It does not exhaust all three scale domains jointly.
- Explicit every low/high payload bit in isolation, 11 edge global halves,
  256 seeded finite blocks, and actual output-weight samples including row/tensor
  boundaries. All lane groups, halves, signs, coefficient endpoints exercised.
- Native empty/bad lengths/256-value dimensions, unaligned input and every 2,048
  nonfinite trailing half encodings at both block positions, output atomicity.
  The first two bytes are quant payload, not a scale; test nonfinite-looking
  payload bytes explicitly. Previous format/Unicode/tokenizer gates remain.

## Measurement and bounded workload

Extend matched model-quant benchmark with q6_k explicitly; preserve prior format
contracts. Use actual output.weight bytes at [5120,1], [5120,64], [5120,4096]. The
last is a **4096-row slice, not all 248320 vocabulary rows**, bounding diagnostic
output at 80 MiB rather than ~4.74 GiB. Full model file hash remains verified.
Native/independent reference full output hashes match on every measured slice.

Same 10,000/256/4 calls per trial, 3 warmups, 7 trials, 3 alternating rounds, CPU10;
all arrays/mappings preallocated and hashes/I/O outside timed loops. Public ggml
type-traits decoder resolved once, no Python per-call cost; native extra finite
validation remains. Repeat and record raw results/variance/losses and source/binary
identities. This CPU diagnostic result is not GPU execution or llama-server speed.
