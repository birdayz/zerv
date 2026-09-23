# Remaining actual-artifact quant blocks — 2026-09-22

Research for the remaining native decoder/kernel slices. **Q4_1 now implemented
and verified**; [results](../bench/2026-09-22-q4_1.md). Q5_K is also [implemented/verified](../bench/2026-09-22-q5_k.md), as is [Q6_K](../bench/2026-09-22-q6_k.md). The native
container and CPU decoding now cover
Q4_0/Q8_0/Q4_1/Q5_K/Q6_K. Required base-model additions are Q4_1, Q5_K and Q6_K (plus ordinary
F32 values). No additional model download needed.

Inspected pinned ggml `456172ec733a135778adcd32d00e576a58232e45` source retained as
`third_party/research/2026-09-22/oracle-ggml-{common.h,quants.c}`, source hashes in
ledger; installed binary oracle remains ggml 0.24.0 / dirty commit with pinned hash.
Reference functions `dequantize_row_q4_1`, `dequantize_row_q5_K`,
`dequantize_row_q6_K` are independent external oracles, not implementation code.

- **Q4_1**: 20 bytes/32 values. Binary16 scale at byte 0, binary16 minimum at byte
  2, sixteen packed nibble bytes at 4. First 16 values use low nibbles, next 16 use
  high nibbles. Values are `scale*unsigned_nibble + minimum`.
- **Q5_K**: 176 bytes/256 values. Binary16 super-scale/super-minimum at bytes 0/2;
  12 packed scale/min bytes at 4; 32 high-bit bytes at 16; 128 nibble bytes at 48.
  Eight groups of 32 each use unsigned six-bit scale/min and five-bit coefficients.
  For group g<4, scale=`s[g]&63`, min=`s[g+4]&63`; for g>=4,
  scale=`(s[g+4]&15)|((s[g-4]>>6)<<4)`,
  min=`(s[g+4]>>4)|((s[g]>>6)<<4)`.
  Coefficient at group g/lane l is nibble from `qs[(g/2)*32+l]` (low if even, high
  if odd), OR bit `((qh[l]>>g)&1)<<4`.
  Value = `(super_scale*group_scale)*coefficient - super_minimum*group_min`.
- **Q6_K**: 210 bytes/256 values. 128 low-nibble bytes, 64 two-bit-high bytes,
  16 signed int8 subscales, trailing binary16 super-scale at 208.
  Each 128-value half comprises four groups of 32. For group g=0..3/lane l,
  low byte index=`half*64+(g%2)*32+l`, nibble low for g<2, high for g>=2;
  high bits=`(qh[half*32+l]>>(2*g))&3`; coefficient is reconstructed unsigned
  six-bit value minus 32. Signed subscale index=`half*8+g*2+l/16`.
  Value = `(super_scale*signed_subscale)*signed_coefficient`.

All half values must be finite before any native diagnostic output write, including
both scale fields for Q4_1/Q5_K; signed zero/subnormals are valid. Maintain strict
FP32 ordering and exact output bits. Maximum finite significand growth is bounded:
Q5 half mantissa × six-bit scale × five-bit coefficient fits FP32 precision; Q6
half mantissa × signed eight-bit scale × six-bit coefficient likewise. Still
validate against actual oracle bits, rather than relying solely on this bound or
assuming compiler FMA behavior. Exhaust global scales with fixed lane patterns,
cover sign/minimum edge combinations, high-bit planes, every subgroup, byte-unaligned
input, malformed lengths and failure atomicity. Include extracted real-model blocks.

Before native changes: complete the functional spec and executable fixture generator,
require external-vs-independent-scalar agreement, then extend tests/benchmarks with
blockElements rather than assuming every format has 32 values. Do not weaken old
Q4_0/Q8_0 exhaustive gates. GPU fused dot products need separate accuracy/performance
gates; passing diagnostic dequantization does not validate GPU execution.
