# Cooperative-matrix prefill GEMM: feasibility research (block 13g, closed: negative for the FP32 default)

Question: can the RDNA3 WMMA units, reached through `VK_KHR_cooperative_matrix`, run
the prefill projections faster than the FP32 scalar-X kernel while staying
FP32-equivalent by default?

The scalar-X kernel runs at about 24 TFLOP/s effective
([13e](../bench/2026-09-22-gemm-throughput.md)). That is near the card's
single-issue FP32 peak, so a further gain needs other hardware. Status: research.
No engine code yet.

## Observed facts (this machine, 2026-09-23)

- `vulkaninfo` (Mesa 26.2.3, API 1.4.354):
  - `VK_KHR_cooperative_matrix` rev 2, with `cooperativeMatrix = true` and
    `cooperativeMatrixRobustBufferAccess = false`;
  - `cooperativeMatrixSupportedStages`: one stage;
  - `VK_KHR_shader_float16_int8` present;
  - integer-dot 8-bit accelerated = true;
  - **no `VK_KHR_shader_bfloat16`**.
- llama-server's startup log agrees: `fp16: dot2 | bf16: 0 | ... | matrix cores:
  KHR_coopmat`. So only fp16 (and int8) cooperative matrices are usable. A
  3 × bf16 exact split is not available.
- **Gate 1 answered.** `tools/coopmat_properties.py` is a ctypes probe of
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`, a development tool only; raw
  output is in
  [properties.json](../bench/data/2026-09-23-coopmat-research/properties.json). The
  driver reports 14 configurations, all 16×16×16 with subgroup scope:
  - `f16×f16 → f32`;
  - `f16×f16 → f16`;
  - every signed/unsigned 8-bit combination → 32-bit, with and without
    saturation.

  There is no bf16, fp8 or larger-K shape. vulkaninfo does not print this list.
- Toolchain: `glslc` (shaderc 2026.3, glslang 1.4.357) compiles a
  `GL_KHR_cooperative_matrix` compute shader for `--target-env=vulkan1.1`, 1.2 and
  1.3, and `spirv-val` accepts the result. The probe
  (`.tools/coopmat-probe/probe.comp`) covers:
  - f16 A/B and an f32 accumulator;
  - `coopMatMulAdd`;
  - an element-wise `acc * scale` between accumulators;
  - a `coopMatLoad` with stride 0.
- The GPU layer (`src/gpu/device.zig`) currently creates a Vulkan 1.1 device with
  no extensions and no feature structs. Coopmat needs:
  - the extension;
  - `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` in the device `pNext`;
  - `shaderFloat16`, and 16-bit storage if f16 is read from buffers;
  - possibly subgroup-size control (open).

## Numerics proposal (to be verified, not yet a fact)

Weights are already exact small integers times FP32-representable scales:

| Format | Weight | Scale structure |
| --- | --- | --- |
| Q4_0 | `d·(q−8)` | `q−8 ∈ [−8,7]` |
| Q4_1 | `d·q + m` | |
| Q5_K | `d·sc·q − dmin·mn` | 6-bit `sc`/`mn` over sub-blocks of 32 |
| Q6_K | `d·sc·(q−32)` | int8 `sc` over sub-blocks of 16 |

The integer parts are exact in fp16 (|q| ≤ 32). Products fp16 × fp16 are exact in
f32 (11 + 11 bits).

**Activation split.** Scale each activation row by a power of two, so that
`max|x|` lands in [2^14, 2^15). Then write `x = s·(h1 + h2 + h3)` with each `h`
rounded to fp16.

- Elements within 2^−6 of the row maximum are represented exactly (3 × 11 bits
  cover 24).
- Smaller elements lose bits only below fp16's subnormal limit, an absolute error
  ≤ 2^−39·max|x|. That is far below one FP32 rounding of the accumulated dot
  product.
- A 2-part split (22 bits) costs one rounding at ~2^−23 relative. That would be a
  measured-quality candidate, not the default.

**Block scales.** Keep one accumulator per scale block. Then add
`scale ⊙ block_acc` into the main accumulator:

- the `d` matrix is loaded with stride 0, broadcast along tokens;
- the Q4_1/Q5_K min terms use per-(token, block) activation sums, broadcast along
  weight rows.

This needs no knowledge of the implementation-defined element layout. One open
question: is a stride-0 `coopMatLoad` defined behavior? Check SPV_KHR_cooperative_matrix.

**Cost model.** Take the fp16 WMMA peak as ~4× single-issue FP32. That is
123 TFLOP/s at 2.5 GHz, before the power cap.

- 3 passes → ≤ ~41 TFLOP/s effective.
- 2 passes → ≤ ~61 TFLOP/s effective.

The per-block rescale adds VALU and LDS work every 32 elements of K (Q6_K: every
16). The ceiling gain over 24 TFLOP/s is therefore modest for the exact variant,
about 1.3–1.7×. It must be measured.

**Integer alternative.** Use `s8×s8 → s32`, which is exact.

- Weights are exact int8.
- Activations become per-row fixed point (power-of-two scale) with 3–4 signed
  8-bit limbs, one WMMA pass per limb.
- Per-block int32 sums are exact and deterministic. Limbs and block scales are
  combined in FP32.
- Error: an absolute ≤ 2^−(8·limbs)·max|x| per element, so it is fixed-point
  rather than floating. This avoids depending on the WMMA float accumulation
  semantics, at the cost of one more pass than fp16.
- RDNA3's int8 and fp16 WMMA rates must be measured, not assumed.

## Results (2026-09-23)

Report: [2026-09-23-coopmat-research](../bench/2026-09-23-coopmat-research.md).
Raw data: [`data/2026-09-23-coopmat-research/`](../bench/data/2026-09-23-coopmat-research/).

1. **Device path works.** `Device.open(.{ .cooperative_matrix = true })` enables
   `VK_KHR_cooperative_matrix`, `shaderFloat16` and `storageBuffer16BitAccess`.
   `tests/gpu_coopmat.zig` checks the f16 A (row-major) × B (column-major) → f32
   layout and a stride-0 column-major load broadcasting a per-row scale; both are
   exact on non-negative data. RADV compiles `coopMatMulAdd` to a real
   `v_wmma_f32_16x16x16_f16` (wave64, a 4-VGPR accumulator).
2. **The f16 WMMA's f32 accumulation is not IEEE** (`bench/coopmat_numerics.py`,
   numerics-2.json). Every f16 × f16 product is exactly representable in f32, yet:
   - Single positive products: 8192/8192 exact.
   - Single negative products: only 4994/8192 exact. The rest are one ulp too
     negative.
   - 16-term random sums agree with round-to-nearest of the exact value in only
     ~37% of cases; others differ by several ulps (up to 3840 ulps relative to the
     result under cancellation).
   - Integer weights × N(0,1) f16 activations: WMMA mean absolute error 2.7e-7,
     versus 7.4e-9 for sequential f32 fma on the same inputs (37×).

   So the plan to "split x into 3 f16 parts and keep FP32 accuracy" fails at the
   accumulation, not at the representation.
3. **The s8 × s8 → s32 WMMA is exact** (16384/16384 random cases with int32 C).
4. **Throughput** (register-resident chains, 6144 wave64 workgroups, sustained
   ~60–120 ms dispatches, two rounds):

   | Operation | Rate |
   | --- | --- |
   | f16 WMMA | 128–137 TFLOP/s |
   | s8 WMMA | 128–138 TOPS |
   | scalar FP32 `v_fma_f32` (wave64) | **63–66 TFLOP/s** |

   The shipped FP32 GEMM runs at 22–24 TFLOP/s ([13e](../bench/2026-09-22-gemm-throughput.md)),
   about 35% of the measured VALU FP32 peak. Earlier notes calling it "near peak" were
   wrong: RDNA3 issues wave64 `v_fma_f32` at twice the single-SIMD32 rate.
5. **Accuracy of exact-integer limb schemes on real data** (`bench/limb_study.py`;
   FP64 reference; mean relative error normalized by Σ|w·x|):

   | Scheme | ffn_gate K=5120 | ffn_down K=17408 | Effective ceiling |
   | --- | --- | --- | --- |
   | shipped FP32 GEMM | 5.1e-9 | 6.2e-9 | 63–66 TFLOP/s (VALU) |
   | 4 int8 limbs (32-bit fixed point) | 4.6e-9 | 6.1e-9 | ~34 TOPS |
   | 3 int8 limbs (24-bit) | 1.5e-8 (3×) | 3.3e-8 (5×) | ~45 TOPS |
   | 2 int8 limbs (16-bit) | 3.6e-6 (700×) | 8.3e-6 (1300×) | ~68 TOPS |
   | f16 x, IEEE f32 accumulation (optimistic model of the f16 WMMA) | 4.2e-6 | 3.0e-6 | ~135 TFLOP/s |

   For these rows, the activation peak/rms is 12–30 (median), up to 113.

## Conclusion

No cooperative-matrix scheme beats the FP32 VALU path at FP32-grade accuracy on this
card:

- The only FP32-grade WMMA scheme (4 exact int8 limbs) has a ceiling of about
  34 TOPS. That is below the measured 63–66 TFLOP/s FP32 FMA peak.
- The fast variants are 3–1300× less accurate than FP32 (3 limbs, 2 limbs, f16).

The larger lever for prefill is the FP32 GEMM itself, at ~35% of the VALU peak.

A 3-limb (~45 TOPS ceiling) or f16 WMMA path remains a possible **explicit,
lower-precision** option, with quality measured end to end. It is not a default.

## Open questions (historical; the conclusion above answers the decision)

1. ~~Coopmat property list~~ (answered above). Still open: is there a
   subgroup-size requirement (wave32 vs wave64) on RADV/gfx11?
2. The ISA produced for the probe (`v_wmma_f32_16x16x16_f16`?). The cost of the
   element-wise accumulator ops and of the stride-0 load.
3. WMMA f32 accumulation semantics. Is each product exact? Is the 16-term sum
   rounded per step, or fused/wide? What order? Characterize with adversarial
   inputs against a sequential-fma reference.
4. Throughput of 1/2/3-pass variants with real quantized shapes (M = 5120…17408,
   K = 5120/17408, N = 512 tokens), including the block rescale.
5. Accuracy of each variant through the existing FP64-oracle gates
   (`tools/verify_model.py`), and the served-output impact.
