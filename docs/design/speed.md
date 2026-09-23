# Speed program: faster than llama-server (design and decision log)

Started 2026-09-24 at the user's request: "hardcore optimizations; faster than
llama-server; TTFT is the key metric, but also tok/s; prefill first, grind until no
further optimization is found; never sacrifice correctness; no hardcoding — real
trade-offs are config knobs; write designs down so decisions are not forgotten."

This file is the design record. Measured facts link their evidence; **estimates are
labelled as estimates** and are not results. Each implemented step gets its own
dated report in `docs/bench/`.

## Rules (decided)

- **Correctness gates every step.** A change is either bit-identical on the model
  oracles, or it re-passes the FP64 model gates (default and long oracle, every
  prefill mode), its component gate, and the served-output comparison. A step that
  fails a gate is not merged; the failure is recorded.
- **Precision is explicit.** No mode changes arithmetic silently. Lower-precision
  paths exist only as explicit, measured options (`--prefill-precision` today).
- **Config knobs only for real trade-offs** (speed vs VRAM, context, precision,
  latency). Kernel and tile choices are not knobs; they are resolved before execution
  from a checked-in, device-keyed tuning table (data, not constants in code), with safe
  defaults for unknown devices. No dead knobs: a knob exists only when both sides of the
  trade-off are implemented and measured.
- **Compare against the best tuned llama-server**:
  - its default configuration, the precision-matched one, and a sweep of `-ub`/`-b`
    (512–2048) with flash attention on, for prefill;
  - its MTP speculation, if zerv uses speculation, for decode.
  - Cold and warm (prefix-cached) TTFT are both reported. Component timings never
    replace serving runs.

## Where the time goes (measured)

Per-phase GPU time for one 512-row prefill chunk, from `zerv-model-profile`
([data](../bench/data/2026-09-24-long-context/)). GEMM FLOPs per chunk: 24.9 TFLOP.
Measured peaks on this card ([coopmat research](../bench/2026-09-23-coopmat-research.md)):
FP32 `v_fma` 63–66 TFLOP/s, f16 WMMA 128–137 TFLOP/s.

| Phase (ms) | FP32 @0 | FP32 @28k | f16 @0 | f16 @28k |
| --- | --- | --- | --- | --- |
| ffn_in (gate+up GEMM) | 320 | 320 | 187 | 176 |
| ffn_down | 163 | 161 | 112 | 107 |
| lin_in (DeltaNet projections) | 126 | 124 | 77 | 72 |
| lin_out | 51 | 49 | 32 | 30 |
| attn_in + attn_out | 48 | 47 | 32 | 30 |
| attention (scores + softmax + pv) | 12 | **637** | 11 | **624** |
| delta (DeltaNet scan) | 47 | 45 | 43 | 41 |
| conv | 16 | 16 | 15 | 14 |
| norm, swiglu, gates, other | ~15 | ~15 | ~14 | ~14 |
| **total** | **798** | **1413** | **521** | **1108** |

The first four rows plus attn_in/attn_out are GEMMs (Q4_0/Q4_1/Q5_K weights, on-the-fly
dequantization). Derived:

- **GEMM efficiency.** FP32 runs at 31–37 TFLOP/s (50–57% of the VALU peak); f16 at
  49–63 TFLOP/s (37–48% of the WMMA peak). GEMMs are 84–89% of a chunk at short
  positions.
- **Attention.** The materialized-score path runs at ~10 TFLOP/s. At 28k positions it
  is 45–56% of a chunk; over a 29k prompt it is ~18 s of the f16 46 s.
- **Non-GEMM work.** About 80 ms per chunk (15%). `conv` is about 14× off its memory
  roofline (21 MB per layer); `delta` is a sequential scan.
- **Decode.** 15.3 GB is read per token (all matrices), so 960 GB/s allows about
  62 tok/s. At short context we reach 54 (about 87%). At 29k, FP32 KV adds 3.8 GB of
  attention reads per token.
- **The GGUF contains an MTP layer** (`blk.64.nextn.*`, `nextn_predict_layers = 1`),
  which enables lossless speculative decoding.

## Plan (block 16, in order)

Estimated effects assume the profile above. They are **estimates**, re-derived from
measurements after each step.

### 16a · Fused (flash-style) prefill attention — first

- **What.** One kernel per (KV head, block of query rows) that streams K/V tiles
  through LDS, keeps the running max, sum and output in registers (online softmax), and
  never writes the score matrix.
  - GQA: the 6 query heads sharing a KV head reuse each loaded K/V tile.
  - Causal masking by tile; FP32 arithmetic.
- **Why first.** It is the largest measured loss at long prompts (the only phase where
  llama is ahead by more than a few percent). It also removes the score buffer that caps
  the activation arena at about 79k context (see `TODO.md`, long context).
- **Estimate.** Attention at ≥35 TFLOP/s instead of ~10: 29k f16 TTFT 46 s → about
  33 s; 3223-token TTFT −5%.
- **Correctness.**
  - The summation order changes (tiles plus online rescaling). Component gate against
    FP64, over positions across tile, chunk and plan edges, with the causal edge and GQA
    mapping.
  - Model gate: the FP64 default and long oracles in every prefill mode.
  - Research first: FlashAttention-2, and llama.cpp's Vulkan `flash_attn.comp` (scalar
    and coopmat variants, pinned in `third_party/`) as a reference implementation and a
    competitor measurement.

### GEMM ceilings on this card (external evidence, 2026-09-24)

1. **FP32 SGEMM** on an RX 7900 XTX, 4096³, by S. Verdier, ["Deep Dive into Matrix
   Optimization on AMD GPUs"](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html)
   (read 2026-09-24):
   - rocBLAS: 30.5 TFLOP/s.
   - Best **compiler-generated** kernel (HIP/LLVM, LDS layout and double buffering):
     **33.5 TFLOP/s**.
   - The later steps edited the ISA by hand: 37.8 → 41.3 → **49.0 TFLOP/s** (80% of a
     61.4 TFLOP/s peak). The author also reports VALU utilization capped at about 75%
     with `v_dual_fmac` (wave32).
   - Vulkan loads SPIR-V only; the driver compiler (ACO) generates the ISA, so
     hand-written ISA is not available to us.
   - Our FP32 Q4 kernels already run at 31–37 TFLOP/s, with dequantization included.
2. **f16 WMMA GEMM** on an RX 7900 XTX, square sizes, by amarbaro,
   ["A Mojo fp16 GEMM that beats hipBLASLt on a consumer RDNA3 card"](https://forum.modular.com/t/a-mojo-fp16-gemm-that-beats-hipblaslt-on-a-consumer-rdna3-card-rx-7900-xtx-with-the-receipts/3471)
   (posted 2026-09-06, read 2026-09-24):
   - At 1536–4096: **91–99 TFLOP/s**; hipBLASLt 69–97 TFLOP/s.
   - Kernel shape: a 4×2 wave layout over a 128×128 block, two LDS buffers with one
     barrier per K step, two-deep global prefetch, XOR-swizzled A, transposed B, 188
     VGPRs and no spills. Small sizes are dispatched with 64×64 tiles.
   - It also reports that clock warm-up (10 s instead of 1 s) moved 4096³ from 66 to
     91 TFLOP/s on the same binary.
   - Ours (Q4 dequantization included): 49–63 TFLOP/s. llama-server's default on the
     same shapes: 47–59 ([f16 prefill report](../bench/2026-09-24-f16-prefill.md)).

**Consequences (estimates):**

- f16: headroom about 1.4–1.6× on GEMMs (target 80–90 TFLOP/s with dequantization on
  our shapes). That is roughly −25–30% per chunk.
- FP32: realistic 40–44 TFLOP/s (about 1.15–1.25×). Our own ablation bounds the
  current structure at 45.6 TFLOP/s without LDS reads
  ([round 2](../bench/2026-09-23-gemm-round2.md)). Beyond that needs ISA control we
  do not have.
- **Measurement rule added:** clocks and power must be in steady state. Warm up for at
  least 10 s before timed GEMM, profile or serving runs, and record the clocks.

### 16b · GEMM round 3

- **Weight repacking at load time** into the exact order the kernel consumes (per
  format and tile). The layout transform happens once, while uploading, and VRAM stays
  the same. This removes address arithmetic and permits wide loads.
  - LDS reads were the largest measured cost of the FP32 kernel (+27% when removed,
    [round 2](../bench/2026-09-23-gemm-round2.md)).
  - This also applies to the f16 WMMA kernel.
- **LDS double-buffering and K-tile software pipelining**, so SMEM/LDS latency is
  hidden (round 2 showed the compiler sinking loads).
- **A device-keyed tile table** from an offline autotuner (`zerv-gemm-bench` sweep),
  checked in as data.
- **Estimate.** f16 GEMMs from 37–48% to about 60–70% of peak; FP32 from 50–57% to
  about 65–70% of the VALU peak. That is −20 to −25% per chunk.
- **Correctness.** Bit-identical where the summation order is unchanged; otherwise the
  model gates.

### 16c · Fusions

- SwiGLU into the ffn_in epilogue (or the ffn_down prologue).
- Residual add + RMSNorm into GEMM epilogues and prologues.
- The causal conv (kernel 4) into the lin_in epilogue.
- Gates into the next GEMM's prologue.
- Each fusion removes a full activation round trip. Estimate: −20 to −30 ms per chunk.

### 16d · Chunkwise-parallel Gated DeltaNet

- Replace the per-token scan within a chunk by the chunkwise (WY/UT) form: intra-chunk
  matrix products plus an inter-chunk state pass. That is tensor-friendly work instead
  of a serial loop.
- Estimate: `delta` 43 → about 12 ms per chunk.
- **Numerically different** (triangular solve, reordered sums), so model gates plus an
  FP64 component test. Reference: flash-linear-attention's chunked gated delta rule,
  and llama.cpp's `delta-net-base.cpp` (pinned).

### 16e · Remaining non-GEMM kernels and chunk size

- `conv` to near its roofline.
- Re-tune `--prefill-chunk` (512 today) once 16a removes the score buffer; the chunk
  size stays a knob (it trades activation VRAM for GEMM efficiency).

### Combined estimate (not a result)

f16 mode:

- 3223 tokens: 3.54 s → about 2.3–2.6 s. llama default: 3.40 s.
- 29k: 46.5 s → about 24–27 s. llama default: 33.3 s.

FP32 default at 3223: 5.2 s → about 3.5–3.9 s, about llama default's speed with far
higher precision. Short prompts are already about 2× faster than llama.

### Decode (block 17, after prefill)

- **MTP speculative decoding** with the GGUF's nextn layer: draft 1 token, verify with
  the main model in one 2-row step. It is lossless: greedy is exact, and sampling uses
  rejection sampling. Typical acceptance would give roughly 1.4–1.8× tok/s (estimate).
  The comparison must include llama-server with its MTP enabled.
- **KV precision knob** (`f16` next to FP32): halves attention reads at long context
  and doubles capacity. Its quality must be measured against FP32 and llama's f16/q4_0.
- **Matvec**: the last ~10% to the bandwidth roofline.

## Zig and comptime: where it helps (assessment)

- **Where the time is.** More than 99% of TTFT is GPU kernel time. Host work per
  512-row chunk is about 1 ms, already pre-recorded commands with no per-token
  allocation. The speed levers are GPU algorithms (above), not host code.
- **Comptime is useful for:**
  - resolving and validating kernel geometry, push-constant layouts, plan tables and
    tile tables at compile time;
  - generating specialization-constant tables;
  - making every kernel variant a checked, typed entry rather than a runtime string.
  - Host SIMD (`@Vector`) is already used where the host matters (tokenizer, NFC).
- **Option, not planned:** Zig's SPIR-V backend, to write kernels in Zig with comptime
  specialization. The driver compiler (ACO) still does the codegen, so it is not a
  speed lever by itself. Revisit only if kernel-variant maintenance becomes the
  bottleneck.

## Decision log

- **2026-09-24 D1.** Correctness rules above are unchanged by the speed program.
- **2026-09-24 D2.** Order: 16a flash attention → 16b GEMM → 16c fusions → 16d
  chunkwise DeltaNet → 16e rest; then decode (block 17). Block 15 (prefix cache) is
  parked with its open gates listed in `TODO.md`.
- **2026-09-24 D3.** Kernel and tile choices come from a checked-in device-keyed
  table, not from knobs or constants in code.
- **Open, for the user (Q1).** Which mode carries the headline claim:
  - (a) **FP32 default vs llama default.** The hardest target: the FP32 VALU peak is
    half the f16 WMMA peak. The estimate says about parity at 3k tokens.
  - (b) **f16-input / f32-accumulate** (`--prefill-precision f16`) vs llama default.
    It is more accurate than llama's default on the FP64 oracle (llama flips 13–24
    greedy tokens; zerv f16 flips none), but the worst-token gate against llama nof16
    is still unmet (block 14).

  Both are reported either way.
- **2026-09-24 D4 (user: "yeah do whatever it takes").** GEMM first (16b before 16a),
  starting with the f16 WMMA path, since that is where the measured headroom is. Q3 is
  answered by this.
  - Hand-written ISA is outside the Vulkan boundary: RADV compiles SPIR-V with ACO. The
    only route to our own ISA would be a userspace submission layer on the amdgpu
    kernel interface (our own driver), or ROCm/HIP (C++, not allowed).
  - Not pursued unless a measured ceiling proves it necessary. It would need an
    explicit approval to expand the boundary.
  - Instead, the SPIR-V is shaped and the ACO ISA inspected.
- **2026-09-24 note: routes to our own ISA** (user asked "vulkan can't load
  assembly?"):
  - **Standard Vulkan: no.** Shader input is SPIR-V; RADV/ACO compiles it.
    `VK_KHR_pipeline_executable_properties` lets us read the ISA produced (inspection,
    not injection).
  - **Hack: patched driver binaries.** The installed driver (Mesa 26.2.3 RADV,
    `vulkaninfo`) supports `VK_EXT_shader_object` (binary shader code) and
    `VK_KHR_pipeline_binary`. Both export the driver's own compiled binary for later
    re-import (caching). Patching the ISA inside such a blob and importing it is
    outside the spec (undefined behavior). It is tied to one exact Mesa build, and the
    format and validation can change with any update.
    - If ever used: a research-only experiment first; in production only behind an exact
      driver-UUID match, with the SPIR-V build as the verified path for any other driver.
    - Needs the user's explicit approval.
  - **Other routes:** our own userspace driver on the amdgpu ioctl interface (very
    large); a patched Mesa (changes system packages, needs approval); ROCm/HIP (C++,
    not allowed).
  - **Plan:** shape SPIR-V and inspect the ISA first. The best known f16 GEMM on this
    card (91–99 TFLOP/s) is compiler-generated, so the target does not need ISA
    injection. Revisit only at a measured compiler wall.
- **2026-09-24 D5 (user).** zerv will support multiple GPU backends, **selectable by
  configuration**. If a measured compiler wall justifies it, a different backend (for
  example a native amdgpu submission path that can load our own ISA) is acceptable, as
  long as it is swappable per config and gated like everything else. Until such a wall
  is measured, the work stays on Vulkan/SPIR-V.
- **(Answered by D4.) Q3.** Swap 16a and 16b (GEMM before flash attention). Estimated savings
  per prompt length L, f16 mode:
  - flash attention: about 0.71 × 22 µs × L² / 1024;
  - GEMM at 1.5×: about 0.28 ms × L.

  They break even near L ≈ 17k. Below that, GEMM is the bigger lever: short prompts,
  bruh cold starts at about 12k, and warm steps.
- **Open (Q2).** Speculative decoding (MTP) counts for decode tok/s if llama-server runs
  with its MTP as well.
- **2026-09-24 16b status (measured).**
  - The [lab](../bench/2026-09-24-gemm-f16-lab.md) produced k64: wave32, 128×256, BK
    64, direct f16 X. It is bitwise-equal to the block-14 kernel and 1.25–1.36× faster
    on Q4_0 in an interleaved race.
  - [Integrated for Q4_0](../bench/2026-09-24-gemm-f16x.md), bitwise-identical end to
    end:
    - f16 prefill GPU time −18% at 3223 tokens;
    - TTFT 3.04 s against llama default's 3.39 s (1.12× faster, from 1.04× slower);
    - 12k: 12.9 s against 12.46 s.
  - Findings that constrain further GEMM work on this driver, source-backed in Mesa
    26.2.3:
    - LDS loads are capped at 8 bytes;
    - LDS shaders run in CU mode;
    - SPIR-V cannot emit `s_setprio`.
    - Measured on the card: the WMMA operands' replicated half-wave is read by the
      hardware.
  - The card sustains GEMMs at the 110 °C junction limit, so kernels are compared only
    interleaved (`bench/race_gemm_f16.py`).
  - Per clock the kernel reaches about 63% of the WMMA peak. The measured losses are
    the A-side LDS fragment loads and the per-step barrier, whose known remedies need
    scheduling control that SPIR-V does not give. This is the first measured
    compiler-level wall (D5), noted but not yet acted on.
  - The remaining 16b steps: Q4_1 and Q5_K ports, the 128-row plan, tail fill for
    M = 5120 shapes, then FP32.
- **2026-09-24 D6 (process).** Kernel variants are compared only in interleaved races
  on this card; separate runs drift ±3–4% with the thermal state.
