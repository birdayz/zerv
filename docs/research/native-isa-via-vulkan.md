# Own RDNA3 machine code through Vulkan (research note, 2026-09-24)

Question (user): the shader compiler limits our prefill GEMMs. Can we run LLVM-compiled
or hand-written RDNA3 kernels while keeping Vulkan, and what could it gain?

Status: **approved for experiments** (user decision D7, 2026-09-24, in
[design/speed.md](../design/speed.md)). Route 2 is built and measured: a hand-scheduled
`gemm_f16x` is bit-identical and 1.20–1.40× faster at component level
([report](../bench/2026-09-24-gemm-f16x-isa.md)). It still depends on a private driver data format
(pinned by the global key, SPIR-V fallback).

## Measured headroom (corrects an earlier chat estimate)

- f16 prefill GEMMs (`gemm_f16x`, Q4_0): 76–89 TFLOP/s in the lab, about 85 TFLOP/s in
  the model (ffn_in at 3,223 tokens: 73.5 TFLOP in 0.867 s). That is **63–66% of the WMMA
  peak per clock**; sustained measured peak 131 TFLOP/s at 2.94 GHz
  ([lab](../bench/2026-09-24-gemm-f16-lab.md)). An earlier chat estimate used 31–37
  TFLOP/s, an FP32-mode figure, and overstated the headroom as 1.5–2×.
- The per-clock losses measured by ablation: A-side LDS fragment loads (removing them:
  79% of peak) and the per-step barrier (72%). The known remedies (`s_setprio` around
  WMMA, LDS loads issued one WMMA ahead, split global prefetch; used by rocm_wmma_gemm)
  need instruction scheduling that ACO does not let us control.
- Estimate, not measured: the GEMMs at 80–90% of peak per clock is 1.2–1.4×. With the
  GEMMs at about 75–80% of f16 prefill GPU time, TTFT would be 12–22% lower. The card
  runs these GEMMs at its 339 W cap, so part of any gain may be lost to clocks.

## Routes (Mesa 26.2.3 source, `third_party/mesa/mesa-26.2.3`)

1. **RADV's LLVM backend (`RADV_DEBUG=llvm`): not usable.** The driver links LLVM 22,
   but `radv_cooperative_matrix_enabled` (`radv_physical_device.c:186`) returns false
   with `use_llvm`, so WMMA kernels cannot run on it.
2. **Own code through `VK_KHR_pipeline_binary`: feasible.**
   - The extension is always exposed (`radv_physical_device.c:761`).
   - A compute pipeline created with `VkPipelineBinaryInfoKHR` skips compilation
     (`radv_compute_pipeline_create`, `radv_pipeline_compute.c:285`) and deserializes the
     binary (`radv_compute_pipeline_import_binary`, line 241 → `radv_shader_deserialize`,
     `radv_pipeline_cache.c:74` → `radv_shader_create_uncached`).
   - The data is RADV's serialized shader: `radv_shader_binary_legacy` (the shader config
     with register counts, LDS and scratch; the `radv_shader_info` struct; code size) and
     then the machine code. `radv_create_pipeline_binary_from_data` copies it with no
     check over the code.
   - Mechanism: compile a placeholder SPIR-V with the kernel's exact interface (bindings,
     push constants, workgroup size, wave32, LDS), fetch its binary, replace the code and
     the register/LDS fields, and create the pipeline from it. Buffers, barriers, command
     buffers and the runtime stay as they are.
   - Code source: the installed clang 22.1.8 assembles gfx1100 code (checked:
     `v_wmma_f32_16x16x16_f16`, `s_setprio`) and compiles LLVM IR with the `amdgpu_cs`
     calling convention Mesa uses (`-target amdgcn-mesa-mesa3d -mcpu=gfx1100`). So both
     LLVM-compiled and hand-written kernels are possible, with clang as a build-time
     tool like glslc; nothing is linked.
   - Risks:
     - The layout is a C struct copied per Mesa version. Pin it: record the global key
       from `vkGetPipelineKeyKHR` (it includes the driver's cache UUID), and on any
       mismatch fall back to the SPIR-V kernels.
     - The kernel must follow RADV's shader ABI: which user SGPRs carry descriptor
       pointers and push constants, and where workgroup and local IDs arrive. Read it
       from the placeholder's disassembly.
     - Correctness gates as today: bitwise equality to `gemm_f16`, plus the spill gate.
3. **Full native submission path (KFD or amdgpu ioctls, no Vulkan): possible, larger.**
   The same kernels, plus our own memory, queues, code objects and synchronization. Only
   worth it if route 2 hits a wall.

## Proposed first measurement (if approved)

1. Round trip: `gemm_f16x`'s own binary through `vkGetPipelineBinaryDataKHR` and back.
   It must stay bitwise-identical and equally fast.
2. Replace the code with ACO's disassembly reassembled by clang: identical code, which
   proves the ABI and config handling.
3. Hand-schedule the hot loop (`s_setprio` around WMMA, LDS fragment loads one WMMA
   ahead) and race it against the shipped kernel (interleaved race, bitwise gate).
4. Adopt only if ≥ 1.2× on the Q4_0 shapes, as a per-config kernel source (SPIR-V
   default until all gates pass), recorded as a D5 decision.
