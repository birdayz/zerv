# Target machine

**Observed 2026-09-22.** See [raw capture](research/2026-09-22/hardware.txt).
Free memory is a moving snapshot, not a capacity guarantee.

| Item | Observation |
| --- | --- |
| OS | Arch Linux, x86_64; kernel `7.2.6-arch2-1` |
| CPU | Ryzen 9 3900X, 12 cores / 24 threads, AVX2/FMA |
| System RAM | About 62 GiB usable; about 47 GiB available during initial inspection |
| Swap | 4 GiB total, about 3.6 GiB already used during initial inspection |
| GPU | Radeon RX 7900 XTX, RADV NAVI31, PCI ID `1002:744c` |
| GPU target | `gfx1100` per AMD compatibility matrix; RDNA 3, not CDNA |
| Physical VRAM reported by sysfs | **25,753,026,560 bytes = 23.984375 GiB** |
| Existing VRAM use | About 1 GiB during initial inspection; reserve space for other users |
| PCIe link | 16.0 GT/s, x16 current and maximum (PCIe 4.0 x16) |
| Available project filesystem | About 1.7 TiB free on NVMe-backed filesystem |
| GPU access | `/dev/dri/renderD128` and `/dev/kfd` present |

## Installed compute stack

- Mesa/RADV `26.2.3`; Vulkan device API `1.4.354`, loader `1.4.357`.
- `vulkan-radeon`, `vulkan-tools`, `ggml-vulkan 0.24.0-1`,
  `llama-cpp 0.4.1-1` installed already. These are **reference tools**, not project deps.
- `llama-server --version`: build 10964, commit `b29c606e28`.
- `llama-server --list-devices` identifies `Vulkan0` as the target card.
- Reference backend reports FP16 dot2, no BF16/FP4 capability, integer dot product,
  64 KiB shared memory, default subgroup size 64, `KHR_coopmat`.
- `vulkaninfo` reports `VK_KHR_cooperative_matrix`, `VK_EXT_memory_budget`,
  `VK_KHR_shader_float16_int8`, `VK_KHR_shader_integer_dot_product`,
  `shaderFloat16`, `shaderInt8`, and subgroup-size control.

These are capability/enumeration observations, **not proof any model or all its
operators work**. Query cooperative matrix shapes and subgroup limits explicitly
before selecting kernels. Do not equate advertised cooperative matrices with a
particular fast instruction path; inspect generated code and measure.

## Tooling gaps and choices

- No `zig`, `cmake`, `ninja`, `hipcc`, `rocminfo`, `rocm-smi`, or `amd-smi`
  executable found during initial inspection. Clang, C compiler, Python, curl,
  and Vulkan tools are available.
- No ROCm/HIP packages found in the installed-package query. `/dev/kfd` existing
  does not mean the HIP userspace toolchain is installed.
- Zig download index lists **0.16.0** as the latest stable release on the inspection
  date. Follow-up: pinned **0.16.0** downloaded into gitignored `.tools/`, archive
  SHA-256 verified, and native component tests run in Debug/ReleaseFast. No system
  installation. See [development.md](development.md) for commands/provenance.
- Proposed first native GPU path: raw Vulkan API + our own compute shaders.
  Existing RADV makes this the shortest path to real device measurements with no
  C++ wrapper. This is a bring-up choice, not proof Vulkan is fastest.
- AMD's current matrix lists the 7900 XTX as `gfx1100`; its distribution matrix
  does not establish support for this Arch/kernel combination. Evaluate any future
  driver/toolchain change separately. No `HSA_OVERRIDE_GFX_VERSION` workaround by default.

## Benchmark hygiene on this workstation

Record background graphics/compute activity, temperatures, clocks, power policy,
GPU driver/compiler versions, actual memory budgets, and CPU affinity/thread count.
Thermal throttling, swap pressure, shader compilation, and desktop use can swamp
small improvements. Do not alter clocks, power limits, or drivers without an
explicit experiment. Recheck memory before loading a large artifact.

**Observed 2026-09-24 (per-wave probes of the native GEMM, [report](bench/2026-09-24-gemm-f16x-isa.md)):**

- An 8-wave workgroup in CU mode puts waves 0, 2, 4, 6 on one SIMD and 1, 3, 5, 7 on
  another (SIMD ids {1,3} or {0,2}); two workgroups share a WGP.
- The SIMD arbiter is **strictly age-ordered** for free-running waves: with no barrier,
  wave 0 finishes its whole loop first, then wave 2, and so on. One wave alone saturates the
  WMMA pipe (1–4 active waves per SIMD: 109 / 217 / 339 / 444 µs, linear).
- ACO models a gfx11 f16 WMMA 16×16×16 as 32 VALU cycles; ablations agree that every other
  VALU instruction on the SIMD adds its cycle to the matrix pipe's time.
- The first scalar load of the host-visible `io` buffer takes **~15 µs in the first round of
  every dispatch** (all waves read one host address at once), 0.4 µs later and 0.5 µs from
  device-local memory.
- Kernel timing: one dispatch per timed sample reads 10–15% lower and ranks variants
  differently from back-to-back dispatches (the clock ramps between fences). Compare
  kernels with batched dispatches (`batch=` in the ISA lab).

**Observed 2026-09-24 (sustained f16 GEMMs):**

- The junction ("hotspot") temperature sits at **110 °C**, the limit, while board power
  is 310–339 W, the cap is 339 W, and vddgfx is 966–1007 mV.
- Shader clocks settle at 2.6–2.85 GHz, and separate runs of the same kernel differ by
  up to about 4%.
- A pure register-resident WMMA loop runs at about 2.94 GHz and about 280 W.
- Sustained compute on this card is therefore **thermally as well as power limited**.
  Compare kernels only interleaved ([bench/race_gemm_f16.py](../bench/race_gemm_f16.py)),
  and record the junction temperature (the GEMM harnesses do).
- Idle readings: edge 53 °C, junction 60 °C, mem 66 °C.
- Whether the cooler is performing normally is an open question. Reference 7900 XTX
  boards with a vapor-chamber defect reach 110 °C early. Fan and cooling settings were
  not changed.
- **Identity (lspci, DMI):**
  - GPU: Sapphire **NITRO+ RX 7900 XTX Vapor-X** (subsystem `1da2:e471`), a large
    AIB cooler rather than the reference design. Power cap 339 W (default), range
    305–350 W.
  - Board: Gigabyte **X570 AORUS ELITE**.
  - GPU fan control is firmware-automatic (`pwm1_enable` = 2) with zero-RPM at idle.
    The overdrive fan-curve interface is not exposed with the current
    `ppfeaturemask`.
- **Thermal logger** ([tools/thermal_log.py](../tools/thermal_log.py), read-only).
  - It reads every hwmon sensor plus amdgpu `gpu_metrics` v1.3: edge, hotspot, memory
    and VR temperatures, fan RPM, power, clock, gfx voltage, and the firmware's
    throttle reasons.
  - Layouts are from Linux v7.2 `kgd_pp_interface.h`, `amdgpu_smu.h` and
    `smu_v13_0_0_ppt.c`, pinned under `third_party/linux-amdgpu/` (sha256 `1a9f13c2…`,
    `8c8ad6de…`, `3fdd5a86…`).
  - A throttle bit means the firmware's throttling percentage for that limiter is
    nonzero. One idle sample in 20–30 showed `TEMP_HOTSPOT` at 48 °C, so single
    samples are noise; judge shares under load.
- **Idle snapshot** (2026-09-24, CPU busy with other jobs,
  [data](bench/data/2026-09-24-thermal/idle.summary.json)):
  - GPU: edge 42 °C, hotspot 48 °C, memory 52 °C, VRs 44–45 °C, fan 0 RPM.
  - CPU Tctl 70 °C; NVMe 45–49 °C.
  - Gigabyte WMI sensors 36 / 40 / 70 / 40 / 47 / 50 °C. They are unlabeled; temp3
    tracks the CPU.
  - Case-fan RPMs are not visible. The board's ITE Super I/O needs the `it87` module,
    which is not loaded, and loading it is a system change.
- **Load test pending.** The edge-to-hotspot gap, fan RPM and case-sensor rise under a
  sustained GEMM will show which of these limits the card: airflow (case fans help),
  the fan curve, or die contact.

Hardware-dependent defaults must be derived at runtime. Do not put `card1`,
`Vulkan0`, 24 GiB, or 64-thread subgroups into generic engine assumptions.

Sources: local capture; AMD compatibility matrix, Zig index, and pinned llama.cpp
build documentation listed in the [source ledger](research/2026-09-22/sources.json).
