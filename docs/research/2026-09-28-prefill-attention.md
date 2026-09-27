# Prefill attention on RDNA3: where we stand and the optimal kernel (research, 2026-09-28)

Question: prefill slows from 1,301 tok/s at 4k to 630 tok/s at 88k. What dominates, how far
is it from the hardware, what do competitors do, and what is the best kernel design for
Qwen3.8's attention (24 query heads, 4 KV heads, head dim 256, 16 full-attention layers) on
the RX 7900 XTX?

Status: research and measurements. Nothing implemented yet.

## 1. Measurements (this machine, 2026-09-27/28)

**Single-user cold prefill sweep** ([data](../bench/data/2026-09-27-prefill-sweep/)):
- zerv f16 prefill, f16 KV, binary `third_party/multiuser/zerv-sweep1`;
- llama-server `llama-fa-ub512`, the in-graph build.
- Two timed requests per length; the spread is < 0.5%.
- The vLLM run died loading weights during a host memory shortage (another process);
  it is not in the table.

| prompt tokens | zerv TTFT s | llama TTFT s | zerv tok/s | llama tok/s | llama / zerv |
| --- | --- | --- | --- | --- | --- |
| 1,034 | 0.87 | 1.29 | 1,184 | 804 | 1.47 |
| 4,034 | 3.10 | 4.10 | 1,301 | 983 | 1.32 |
| 16,034 | 14.05 | 17.02 | 1,141 | 942 | 1.21 |
| 32,034 | 33.07 | 38.65 | 969 | 829 | 1.17 |
| 64,034 | 86.33 | 93.49 | 742 | 685 | 1.08 |
| 88,034 | 139.83 | 150.11 | 630 | 586 | 1.07 |

**zerv per-phase GPU time** (`zerv-model-profile MODEL 90112 512 N 4 f16@native f16`,
[profiles](../bench/data/2026-09-27-prefill-sweep/profile/)):

| phase | 4k | 32k | 88k |
| --- | --- | --- | --- |
| attention | 0.17 s (5.6%) | 10.24 s (31.3%) | **78.4 s (56.4%)** |
| ffn_in | 0.90 (29.9%) | 7.10 (21.7%) | 19.2 (13.8%) |
| ffn_down | 0.56 (18.4%) | 4.37 (13.4%) | 11.7 (8.4%) |
| lin_in | 0.41 (13.5%) | 3.23 (9.9%) | 8.7 (6.3%) |
| delta (DeltaNet scan) | 0.32 (10.6%) | 2.53 (7.7%) | 6.8 (4.9%) |
| lin_out (Q5_K, SPIR-V f16) | 0.26 (8.5%) | 2.01 (6.1%) | 5.4 (3.9%) |
| attn_in | 0.12 (4.1%) | 0.99 (3.0%) | 2.7 (2.0%) |
| conv | 0.12 (3.9%) | 0.94 (2.9%) | 2.5 (1.8%) |
| total GPU | 3.0 s | 32.7 s | 139.1 s |

**Attention efficiency:**
- The last 88k chunk (512 rows at position 87,552) spends 905 ms in attention for 17.7
  TFLOP (QKᵀ and PV, 16 layers × 24 heads × 256 dims): **19.5 TFLOP/s**.
- Over the whole 88k prompt attention is ~1.52 PFLOP.
- **Ceilings** measured on this card (docs/bench/2026-09-23-coopmat-research.md): FP32
  `v_fma` 63–66 TFLOP/s; f16 WMMA 128–137 TFLOP/s. Our WMMA GEMM loop reaches ~89% of the
  WMMA issue rate (docs/bench/2026-09-24-gemm-f16x-isa.md).
- So attention runs at **31% of the FP32 vector peak and 15% of the WMMA peak.**

**The GEMMs are not the problem.** The shipped native Q4_0 GEMM runs at 77–102 TFLOP/s,
dequantization included, near published RDNA3 f16 GEMM results (91–99 TFLOP/s). The
49–63 TFLOP/s in docs/design/speed.md predates the native kernel.

**Competitors' attention** (estimates, to be measured directly):
- **llama-server:** its short-prompt rate (983 tok/s at 4k) applied to 88k gives ~90 s
  of non-attention work, which leaves ~60 s of attention (~25 TFLOP/s).
- **vLLM (ROCm attention backend on gfx1100):** took 203–209 s for 69.6k tokens in the
  interference runs (docs/bench/2026-09-26-shared-pool.md).
- **Nobody on this card runs prefill attention near the hardware.**

## 2. Why our FP32 kernel is slow (`src/model/flash.comp`)

- **Workgroup:** 8 query rows × 6 query heads of one KV head (6 × 64 threads). Per
  128-key tile, lane l owns keys 2l and 2l+1.
- **S = QKᵀ:**
  - per head dimension a lane loads one K pair from global memory and does 8 rows × 2
    keys = 16 FMAs;
  - Q arrives as wave-uniform scalars;
  - each of the 6 head groups loads the same K again (L0/L1 hits, but 6× the memory
    instructions).
  - ~16 FMA per vector memory instruction: **issue-bound on memory instructions**, not
    on math.
- **P·V:** per key a lane loads one V vec4 and two P vec4 from LDS for 32 FMAs.
- **Register tile too small:** 8 rows × 2 keys (S) and 8 rows × 4 dims (O). A GEMM-style
  tile (e.g. 8 × 8 per thread) with K and V staged once per workgroup in LDS raises FMAs
  per load 4–8×.

## 3. What llama.cpp does on this card (source read: `third_party/hermetic-src/llama.cpp-b29c606e…/ggml/src/ggml-vulkan`)

- RDNA3 has cooperative matrices without NV coopmat2, so llama selects `FA_COOPMAT1`
  (`get_fa_tuning_params_coopmat1`, `flash_attn_cm1.comp`).
- **Tiles:** Br = 16 query rows, Bc = 64 keys (16 × 4 subgroups), workgroup = 4
  subgroups.
- **Numerics:** Q stored to LDS as f16. Sᵀ = K·Q via `coopMatMulAdd` with f32
  accumulators (WMMA). Softmax scalar through LDS (`sfsh`). P = exp(...) stored as f16.
  O via WMMA (`PVMat`, f32 or f16 accumulation depending on the op precision).
- **Several workgroup barriers per key tile** (S store, softmax, P store, PV).
- One query head per workgroup: K/V are not shared across the 6 query heads of a KV head.
- This is the arithmetic class of our f16 prefill mode: f16 operands, WMMA
  accumulation (not IEEE FP32 on this card, see §5), f16 P.

## 4. Design of an optimal RDNA3 prefill attention (WMMA, f16 mode)

Constraints:
- wave32 WMMA 16×16×16 (f16 in, f32 accumulate);
- ≤ 256 VGPRs per wave (with occupancy 1–2 waves/SIMD, as the GEMM);
- LDS ≤ 64 KiB per workgroup;
- head dim 256; 6 query heads per KV head;
- K cache `[kvh][d][ctx]`, V cache `[kvh][ctx][d]`;
- causal; paged (tables per 128-token page).

Proposed structure (FlashAttention-2 style, adapted):

1. **GQA packing.**
   - A workgroup handles one KV head and R query rows of all 6 of its query heads:
     M = 6R query rows share each K/V tile staged in LDS once (6× less K/V traffic than
     llama's per-head workgroups).
   - R = 16 → 96 rows, 6 or 12 waves.
2. **Transposed products avoid LDS round trips for P.**
   - Compute Sᵀ = K·Qᵀ (keys × queries) with K as operand A and Qᵀ as operand B.
   - The f32 accumulator layout of a 16×16 tile gives lane l (mod 16) one query column
     with 8 of its 16 keys (the other 8 in lane l ± 16).
   - Pᵀ as operand B of Oᵀ = Vᵀ·Pᵀ needs 16 keys per lane. Those two halves exchange with
     one cross-lane permute (`v_permlanex16` / DPP) after the f16 conversion.
   - No LDS store and barrier for P (llama goes through LDS). Row max and sum run per
     query column: an in-lane reduction over 8 values plus one cross-half exchange.
3. **Register budget for head dim 256** (the hard part):
   - Qᵀ for 16 queries × 256 dims is 16 B fragments = 128 VGPRs;
   - Oᵀ (256 dims × 16 queries, f32) is 16 accumulators = 128 VGPRs.
   - Together they exceed one wave's budget. Options:
     - **(a) Split the head dimension across two waves.** Each owns 128 dims of O (64
       VGPRs) and computes partial Sᵀ over its 128 dims of Q (64 VGPRs); the two partial
       Sᵀ tiles (16 × Bc, small) are summed through LDS. One exchange per tile; both waves
       then run the same softmax on identical sums (same arithmetic, bitwise the same P).
     - **(b) Keep Q in LDS** and reload fragments per key tile (LDS bandwidth: 128 VGPR of
       loads per 16×Bc tile).
     - **(c) Recompute:** Q from global memory each tile (L1-resident; wastes
       memory-instruction issue).
   - (a) looks best; (b) is the fallback. Decide by measurement.
4. **K/V staging.**
   - Bc = 32 keys: K 32×256 f16 (16 KiB) + V 32×256 f16 (16 KiB); double-buffered 64 KiB.
   - K arrives d-major (`[d][ctx]`), which is operand A's natural "row = key" order only
     after a transpose.
   - Options:
     - transpose while staging (b16 writes, or registers + permutes);
     - a key-major K cache layout (changes decode attention too; a cross-cutting change,
       measured separately);
     - Sᵀ as K operand B instead.
   - To settle in the lab by ISA inspection and timing.
5. **Causal tiles:**
   - key tiles fully below the diagonal skip the mask;
   - tiles past the last query are never loaded;
   - the diagonal tile masks per element.
6. **Softmax in f32** with the online rescale per tile (FlashAttention-2). The rescale of
   Oᵀ is 64–128 VALU multiplies per tile, amortised over 2 × 16 WMMAs per wave.
7. **Occupancy:**
   - with ~200–256 VGPRs, 1–2 waves per SIMD, as in the native GEMM;
   - latency is hidden by software pipelining (next tile's K/V loads issued before this
     tile's WMMAs), proven in `gemm_f16x` v4.

**Expected ceiling** (to be verified by a WMMA-only probe of this loop, as done for the
GEMM):
- per 32-key tile and 16 queries a wave does 16 (S, split over dims) + 16 (O) WMMAs
  against ~100–150 VALU ops (softmax, conversions, permutes, rescale);
- ACO counts a WMMA as 32 VALU cycles, so VALU work adds ~25–30%;
- estimate 60–80% of the WMMA peak: **80–100 TFLOP/s**, 4–5× the current kernel.
- Over an 88k prompt that is **78 s → ~16–19 s**, TTFT ~140 s → ~78–80 s (1.9× llama).

## 5. Numerics and the precision contract

- **WMMA accumulation on this card is not IEEE FP32:** up to thousands of ulps on
  cancelling sums (docs/bench/2026-09-23-coopmat-research.md). Q and P also round to f16.
  So a WMMA attention is a **precision change** of the f16 prefill mode, not an exact
  speedup.
- Competitors use the same class: llama.cpp cm1 (f16 Q, f16 P, WMMA) and vLLM (f16/bf16).
- **Gate** (as for block 14's f16 prefill):
  - component error against FP64 over positions across tiles, pages and the causal edge;
  - end-to-end quality against the FP64 oracle on the long and default fixtures, in the
    same metrics as the f16 prefill quality gate;
  - a new explicit knob (`--prefill-attention fp32|wmma`), default unchanged until the
    gate and the benchmarks say otherwise.
- **The exact path (FP32, default prefill mode) also has headroom:**
  - a register-tiled FP32 kernel (K/V staged once per workgroup, 8×8-style thread tiles)
    should reach ~40–45 TFLOP/s (2–2.3×), bounded like the FP32 GEMMs at 31–41 TFLOP/s;
  - it keeps the FP32 contract (a different summation order: FP64 component gate as for
    block 16a).

## 6. Other prefill costs (after attention)

- **DeltaNet scan** `delta`: 10.6% at 4k (a sequential scan, 40 ms per chunk).
- **`lin_out`:** 8.5%; Q5_K on the SPIR-V f16 GEMM, not on the native kernel.
- **Q4_1 layers of `ffn_down`.**
- **conv:** ~14× off its memory roofline (docs/design/speed.md).

At 4k these four are ~27% of prefill, and they matter most at short and medium prompts.

## 7. Plan (proposal)

1. **Direct competitor numbers:**
   - llama-server attention time at 32k and 88k with `GGML_VK_PERF_LOGGER`;
   - vLLM once the host has memory again.
2. **WMMA attention lab kernel** (GLSL coopmat first, as `gemm_f16x` began):
   - GQA-packed, transposed products, split head dim;
   - a WMMA-only loop probe for the ceiling;
   - bitwise self-consistency across tile and page edges.
   - Then hand-scheduled ISA if the compiler leaves > 20% (the `gemm_f16x` path).
3. **Precision gate and knob**; end-to-end sweep 1k–88k against llama-server.
4. **FP32 register-tiled attention** for the exact mode.
5. **Short-prompt costs:** delta scan, Q5_K/Q4_1 on the native GEMM, conv.
