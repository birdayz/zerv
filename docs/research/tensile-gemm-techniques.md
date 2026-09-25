# Tensile GEMM techniques, mapped to our native kernel (research note, 2026-09-24)

Question (user): can AMD's Tensile (the assembly GEMM generator behind rocBLAS/hipBLASLt) be used
for zerv, and what can it teach our hand-scheduled `gemm_f16x`
([report](../bench/2026-09-24-gemm-f16x-isa.md))?

**Answer: not as a component** — it computes dense f16/bf16 GEMMs on matrices in memory (our
speed comes from fusing the Q4_0 dequantization into the GEMM), targets ROCm's launch ABI (kernel
argument segment, HSA code objects) rather than RADV's (descriptor set, inline push constants),
uses its own k-order and tiling (outputs would not be bit-identical to our reference), and
AGENTS.md reserves other engines for oracles and competitors. **As reading material, yes:** its
parameter space is AMD's own catalogue of GEMM scheduling techniques for their GPUs, including
RDNA3 (gfx1100 = navi31 in `Common.py`).

## Source

- `github.com/ROCm/Tensile` @ `e8a8999e0e7374aaae546a6d7cb703d9e06b0ebf` (2025-06-13; the
  standalone repository is no longer developed, Tensile moved into `ROCm/rocm-libraries`), shallow
  clone in `third_party/research-gemm/Tensile/` (88 MB), read 2026-09-24. Research material only,
  not a dependency.
- Read: `Tensile/Common.py` (the documented parameter space, lines ~500–1420) and
  `Tensile/KernelWriterAssembly.py` (17,313 lines; `s_setprio` use, WMMA store paths).
- Not done: running Tensile or rocBLAS here (ROCm is not installed; installing it is a system change
  that needs the user's approval).

## Techniques and what they mean for us

Our constraint: every output must receive its WMMAs in ascending k (bit-identity with the SPIR-V
kernel, the `--gemm-code` contract). Anything that reorders or splits the summation is excluded.

| Tensile parameter | What it does (Tensile's own description) | Our kernel |
| --- | --- | --- |
| `ExpandPointerSwap` | copy of the unroll loop with LDS buffer offsets as immediates | **done** (loop unrolled by two stages, all LDS offsets immediate) |
| `PrefetchLocalRead` (PLR) | LDS→VGPR reads N MAC iterations ahead | **done** (A fragment i reloaded 12 WMMAs ahead, i-major order) |
| `PrefetchGlobalRead` 1 / 2 | double-buffer global→VGPR→LDS; PGR2 prefetches again while writing LDS | PGR1-like (raw A one stage ahead). **Candidate:** two stages ahead (needs 5 more VGPRs or a smaller dequant temp set) |
| `ScheduleIterAlg` 3, `GlobalReadPerMfma`, `LocalWritePerMfma` | spread global reads and LDS writes between matrix instructions at a controlled density ("a full VMEM FIFO blocks other issue") | **partly** (B loads after their last use, dequant 5 ops per WMMA gap). **Candidate:** tune the density (race variants) |
| `s_setprio 1` around MAC blocks (`AggressivePerfMode`), `s_setprio 3` for stores | priority for the wave issuing matrix work; stores prioritized so another WG's loop hides them | **done** (per-WMMA `s_setprio`, +0.3–1.3%). Store priority: we run one WG per CU, so there is no other WG's loop on the same CU to hide stores behind |
| `StoreRemapVectorWidth` | put MI outputs into LDS, read them back along the contiguous M dimension, write wide (documented reason: "MI output … not continuous, store performance … poor") | **done independently** (our finding 3, epilogue through LDS, the largest single gain: 1.117×). Confirms the diagnosis |
| `OptNoLoadLoop` 1/2 | interleave stores with the final MACs of the last iteration | **candidate** (TODO: epilogue during the last stage, ~1%) |
| `SourceSwap` | swap the MI operand order so the output lands in store-friendly lanes | **candidate, risky:** computing Yᵀ = X·Wᵀ would put 16 consecutive m values in 16 lanes (coalesced b32 stores, no LDS remap), but whether a transposed WMMA rounds bit-identically must be tested |
| `PersistentKernel`, `PrefetchAcrossPersistent` | one WG per CU loops over tiles; the next tile's global loads are issued in the current tile's last iteration | **candidate:** overlaps each tile's epilogue and the next tile's prologue (row-count read, first A/B fetch); per-tile arithmetic unchanged. Est. 1–3% |
| `WorkGroupMapping` | remap WG ids so the WGs resident at one time cover a compact box of tiles (L2 reuse) | **candidate:** today tiles run x (M) fastest, so the A tile of (x, y) and (x, y+1) are 136 WGs apart at ffn_gate. A box order reuses A (weights, the larger stream) while it is in L2/MALL. Less memory traffic, more clock at the power cap. Cheap to try (prologue remap) |
| `.align32 8, 0xbf800001` before MAC blocks | align the hot loop for instruction fetch | **tried 2026-09-24, no effect:** `.Lloop` on a 64-byte boundary (`gen_f16x.py --abl align`), bit-identical, races vs v4 0.998 / 1.001 (ffn_gate, 3,328 rows) and 1.006 / 0.995 (ffn_down, 512 rows): noise. Not adopted |
| `StaggerU` | start each tile's summation at a different k offset (DRAM channel/TLB conflicts) | **excluded:** changes the summation order ("the difference is the order that the summation elements are added") |
| `GlobalSplitU`, `StreamK` | split K between WGs, then reduce | **excluded:** changes rounding (also for the deterministic fix-up) |
| `1LDSBuffer`, `MaxOccupancy` | trade LDS for occupancy | not useful: we are VGPR-bound at one WG per CU |
| `NonTemporalA/B` | streaming cache hints | unlikely: both weights (other y tiles) and X (other x tiles) are reused |

## Consequences

- Nothing to adopt wholesale. The two biggest effects we found (fused dequant + custom pipeline,
  and the store remap) are either outside Tensile's scope or already done.
- New candidates for the generator, cheapest first: loop alignment (tried: no effect);
  `WorkGroupMapping`-style tile order (needs the grid size, which the placeholder ABI does not
  pass: M is not among the inlined push constants; and the weight tiles likely already hit the
  96 MB MALL on the second row of tiles, so a small gain at most); `OptNoLoadLoop`-style epilogue interleave; persistent kernel with cross-tile prefetch;
  density tuning of load/store scheduling; `SourceSwap` (bit-identity test first). Each is
  bit-identical by construction except `SourceSwap`, and each is judged by the usual interleaved
  race plus the bitwise sweep. Recorded in TODO.md under the native kernel follow-ups.
