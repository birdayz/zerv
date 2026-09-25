# Multi-row decode projection scaling beyond 5 rows (block 18a, 2026-09-24)

Modules: `src/matvec/matvec_rows.comp`, Q4_0 (FORMAT 2, BLOCK_BYTES 18, PAYLOAD_OFFSET 2,
LANES 64, ACCUM_FMA 1), ROWS = R, GROUP / CB: shipped table for R <= 5 (1/4, 3/2, 3/2, 3/3),
CB 2 with GROUP 1 or 2 beyond; glslc/spirv-val pinned as tools/compile_matvec.py. Built into
third_party/rows-scale/R{R}_G{G}/q4_0.spv (hashes in modules.sha256).

Command per module (under the spill gate):

    python3 tools/check_shader_spills.py --log R{R}_G{G}.spill.txt -- \
      zig-out/bin/zerv-matvec-rows-bench MODEL --spv-dir third_party/rows-scale/R{R}_G{G} \
      --group G --samples 21 --warmup-ms 1500 --roles ffn_gate.weight,attn_qkv.weight R

Every layer's copy of the role is resident (streams from DRAM as in a decode step). All rows
bitwise equal to the single-row module (0 mismatches everywhere). R16_G2 spills VGPRs to
scratch (spill-gate.txt): excluded.
