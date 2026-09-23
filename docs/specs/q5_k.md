# Q5_K CPU decoding — controlled block 05

Written before goldens/native implementation. Primary source research is pinned to
`456172ec733a135778adcd32d00e576a58232e45`: block_q5_K, get_scale_min_k4 and
dequantize_row_q5_K in retained oracle-ggml-{common.h,quants.c}. [Equations/layout](../research/mixed-quant-blocks.md).
48 actual ssm_out tensors use Q5_K, each [6144,5120], 21,626,880 bytes.

## Contract

Extend quant.Format with q5_k: **176 bytes / 256 values** per block. Same bounded,
allocation-free diagnostic API and error atomicity; do not assume 32 elements.
Both half fields at offsets 0/2 must be finite across every block before any write.
All length/empty/unaligned/nonoverlap/ownership rules from quantization.md apply.
No GPU, matrix products, inference or Q6_K implementation in this block.

Bytes 4..15 pack eight unsigned 6-bit scales and eight unsigned 6-bit minima.
For group g<4: scale=s[g]&63, min=s[g+4]&63. For g>=4:
scale=(s[g+4]&15)|((s[g-4]>>6)<<4),
min=(s[g+4]>>4)|((s[g]>>6)<<4).
Bytes 16..47 are high bitplanes qh; 48..175 pack low nibbles qs. Group g/lane l
coefficient = nibble(qs[(g/2)*32+l], low if even/high if odd) |
(((qh[l]>>g)&1)<<4). Output group order 0..7, lane order 0..31.
Strict FP32 result `(d * group_scale) * coefficient - dmin * group_min`.
Half×6-bit×5-bit significands fit FP32 exactly; subtraction may round. Preserve
signed zero/subnormals. Bit-exact external/scalar equality, no approximate math.

## Independent fixtures / gates

Generator uses the pinned external dequantize_row_q5_K ABI and checks **every**
output against independent scalar unpacking/arithmetic/FP32 packing before writing.
It records library/generator/model hashes and refuses existing output. Real tensor
positions come from the independent container oracle. Repeated generation must be
byte-identical; ordinary native tests load only embedded goldens.

- Exhaust every finite half encoding in each field against partner fields
  [0,0x3555,0xfbff]. The fixed semantic pattern has group scales
  [0,1,15,16,31,32,62,63], minima in reverse, and q[g][lane]=(lane+5*g)%32.
  Fingerprint order: field 0 then 1, ascending finite bits, partners in listed order.
  380,928 blocks / **97,517,568 values**. This is not all pairs of global fields.
- Separate packed-scale fingerprint: every byte value at each of 12 scale-byte
  positions, leaving other bytes at the fixed pattern; globals 1 and 0.5.
  3,072 blocks / 786,432 values; exercises shared high/nibble fields independently.
- Explicit all eleven global-edge pairs (as Q4_1), all 256 single high-plane/lane
  bits, 256 seeded packed payloads with finite globals, and eight real blocks from
  each of all 48 tensors. Include coefficient zero/full, sign/cancellation, ordering.
- Native: same fingerprints and explicit bytes, all 2,048 nonfinites in each field
  and each of two blocks, malformed block/output sizes, empty rows, byte-unaligned
  inputs and atomic failure. Earlier quant/tokenizer goldens remain untouched.

## Matched measurements

Extend the existing real-model benchmark to select q5_k explicitly. Actual
blk.0.ssm_out.weight at [6144,1], [6144,64], [6144,5120]; same 10,000/256/4 calls,
three warmups, seven trials, three alternating rounds, CPU 10. Native compile-time
format selection occurs before loops. External adapter resolves public
`ggml_get_type_traits(Q5_K)->to_float` before timing; no wrapper probing or Python
per-call timing. Independent full output hashes gate every trial. Both buffers
preallocated, no I/O/hashes/allocations inside timing; native finite validation
remains enabled, extra relative to the external valid-input assumption.

Repeat runs, record all variance/losses, commands/library/header/compiler/model and
source snapshots. Preserve Q4_1 benchmark support/default unchanged. No diagnostic
CPU speed ratio is a llama-server inference/serving claim.
