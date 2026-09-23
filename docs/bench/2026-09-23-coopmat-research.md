# Cooperative-matrix (WMMA) prefill: numerics, throughput and accuracy (block 13g)

Date: 2026-09-23. Question: can RDNA3's WMMA units, reached through
`VK_KHR_cooperative_matrix`, run the prefill projections faster than the shipped FP32
GEMM (22–24 TFLOP/s) while staying FP32-grade by default?

Answer: **no**. This is a research-only block; no engine code changed. The details
are in the [research note](research/coopmat-prefill.md).

## Setup

- RX 7900 XTX, Mesa 26.2.3 RADV, Vulkan 1.4.354; glslc from shaderc 2026.3.
- The card is power-capped at ~339 W.
- GPU jobs ran one at a time.
- Tool hashes are in [hashes.txt](bench/data/2026-09-23-coopmat-research/hashes.txt);
  the toolchain is in [toolchain.txt](bench/data/2026-09-23-coopmat-research/toolchain.txt).

New research tooling (development only; not linked into the engine):

- `tools/coopmat_properties.py` is a ctypes probe of
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.
- `zerv-coopmat-probe` (`bench/coopmat_probe.zig`) has two modes:
  - one 16×16×16 `D = A·B + C` per workgroup, with f16/f32 or s8/s32 element types;
  - a `peak` mode that times register-resident chains (`bench/coopmat/peak.comp`).
- `bench/coopmat_numerics.py` compares every output element with exact rational
  arithmetic.
- `bench/limb_study.py` emulates exact-integer limb schemes in FP32 on real weights
  and activations against the FP64 reference.

GPU-layer change: the `Device.open` option `cooperative_matrix` enables the
extension, `shaderFloat16` and `storageBuffer16BitAccess`, after checking that all
three are supported. Otherwise it fails with `UnsupportedFeature`.

- The Vulkan ABI inventory adds `vkEnumerateDeviceExtensionProperties`,
  `vkGetPhysicalDeviceFeatures2` and three feature structs. The bindings and the
  C-ABI fixture were regenerated: 50 structs, 71 constants, byte-compared.
- The binding generator now handles extension enumerants without `extnumber`.
- `tests/gpu_coopmat.zig` checks the layout and the stride-0 row broadcast.

## Results

**Supported configurations** ([properties.json](bench/data/2026-09-23-coopmat-research/properties.json)):
14 configurations, all 16×16×16 with subgroup scope:

- f16×f16→f32;
- f16×f16→f16;
- all s8/u8 combinations → 32-bit, with and without saturation.

There is no bf16. llama-server's startup log agrees (`bf16: 0 | matrix cores:
KHR_coopmat`).

**Numerics of the f16 WMMA with f32 accumulation.**
[numerics-3.json](bench/data/2026-09-23-coopmat-research/numerics-3.json) reproduces
numerics-2 exactly with the final probe source.

| Experiment | Equal to the correctly rounded exact value |
| --- | --- |
| single product, positive (exact in f32) | 8192 / 8192 |
| single product, negative (exact in f32) | 4994 / 8192 (the rest are 1 ulp too negative) |
| 16 random terms | 3406 / 8192; max error 3840 ulps of the result |
| integer weights × N(0,1) f16, 16 terms | 10914 / 16384, mean abs error 2.7e-7 (sequential f32 fma: 16358 / 16384, 7.4e-9) |
| 1 + 2^-e, two products | exact through e = 23 |

**s8 WMMA:** 16384 / 16384 exact ([i8/](bench/data/2026-09-23-coopmat-research/i8/)).

**Throughput** ([peak.jsonl](bench/data/2026-09-23-coopmat-research/peak.jsonl)). The
setup is 6144 workgroups × 20000 iterations, median of 9, two rounds; each cell lists
2 / 4 / 8 independent chains:

| Operation | Round 1 | Round 2 |
| --- | --- | --- |
| f16 WMMA (TFLOP/s) | 116.6 / 127.2 / 136.6 | 128.0 / 129.4 / 134.8 |
| s8 WMMA (TOPS) | 129.3 / 131.7 / 137.8 | 128.5 / 131.1 / 136.9 |
| FP32 `v_fma_f32` wave64 (TFLOP/s) | 63.5 / 63.1 / 66.2 | 63.2 / 62.7 / 65.8 |

The ISA was checked with `RADV_DEBUG=shaders`:

- The VALU variant compiles to 256 `v_fma_f32` (VOP3, wave64) per loop body.
- The probe's multiply-add compiles to `v_wmma_f32_16x16x16_f16`.

**Accuracy on real data** ([limb-study.json](bench/data/2026-09-23-coopmat-research/limb-study.json)).
The data is 64 Q4_0 weight rows × 48 tokens of the `short-nothink` oracle
activations, with the FP64 reference. Errors are relative to Σ|w·x|:

| Scheme | ffn_gate (K=5120) mean / p99 | ffn_down (K=17408) mean / p99 |
| --- | --- | --- |
| shipped FP32 GEMM | 5.1e-9 / 2.6e-8 | 6.2e-9 / 2.8e-8 |
| 4 int8 limbs | 4.6e-9 / 2.5e-8 | 6.1e-9 / 3.0e-8 |
| 3 int8 limbs | 1.5e-8 / 5.8e-8 | 3.3e-8 / 2.6e-7 |
| 2 int8 limbs | 3.6e-6 / 1.5e-5 | 8.3e-6 / 7.3e-5 |
| f16 x, IEEE f32 accumulation (optimistic) | 4.2e-6 / 1.3e-5 | 3.0e-6 / 1.1e-5 |

## Interpretation

- **The f16 WMMA is not an FP32-grade accumulator on this hardware.** Even in its
  optimistic IEEE model, rounding x to f16 costs ~1000× the FP32 GEMM's error. This
  rules out the "3 × f16 split" plan.
- **Exact-integer WMMA is FP32-grade only with 4 limbs.** Its ceiling (137/4 ≈ 34 TOPS)
  is below the measured scalar FP32 peak (63–66 TFLOP/s), before any overhead for
  dequantization or scaling.
- **The shipped GEMM is at ~35% of the measured FP32 VALU peak.** The 13e
  description ("near peak") was wrong, and the TODO has been corrected. Raising VALU
  efficiency is the larger lever for long-prompt TTFT.
- **Explicit lower-precision modes** (3 limbs, ~45 TOPS ceiling, 3–5× the FP32 error;
  or f16 WMMA) remain possible options for later. They must be explicit and have
  quality measured end to end.

## Commands

```sh
python3 tools/coopmat_properties.py --output docs/bench/data/2026-09-23-coopmat-research/properties.json
zig build coopmat-probe-build -Doptimize=ReleaseFast -Dcpu=native
python3 bench/coopmat_numerics.py --work third_party/coopmat-numerics-3 --output docs/bench/data/2026-09-23-coopmat-research/numerics-3.json
for m in 0 1 2; do for c in 2 4 8; do
  glslc --target-env=vulkan1.1 -O -fshader-stage=compute -DMODE=$m -DCHAINS=$c bench/coopmat/peak.comp -o .tools/coopmat-peak/m$m-c$c.spv
done; done
# two rounds of: zerv-coopmat-probe peak .tools/coopmat-peak/mM-cC.spv 6144 20000 C 9 | cat >> peak.jsonl
python3 bench/limb_study.py models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf --oracle-dir third_party/model-oracle/2026-09-22-a \
    --rows 64 --output docs/bench/data/2026-09-23-coopmat-research/limb-study.json
```

## Process notes

- A first `peak` sweep redirected the whole loop into one file. The probe writes
  stdout positionally, so each run overwrote the last and only one line survived. It
  was rerun with per-run appends; the lost lines are not used.
- The first version of `tests/gpu_coopmat.zig` expected exact results on signed data
  and failed. That failure was the first observation of the non-IEEE accumulation. The
  test now uses non-negative data, where the path is exact, and the numerics live in
  the research tool.
- Four probe SPIR-V files were briefly written to `/tmp` and deleted unused. They
  were regenerated under `.tools/coopmat-peak/`.
