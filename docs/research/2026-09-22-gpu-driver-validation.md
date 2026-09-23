# Native Vulkan bring-up and verification — 2026-09-22

Block07, not model execution. [Research](vulkan-driver.md),
[contract](../specs/gpu-driver.md). No driver, package, clock or power changes.

## Independent gates and reproducibility

- Pinned official C header compiles into an external ABI extractor: **43 structures,
  61 constants**, all fields/sizes/alignments verified natively. Current ABI JSON SHA
  `842f367f0b28ad7d0ff1eb0f587b2b123f1c1f7d92aa1db268b8b5ae281c3e66`.
  Regenerated twice identically. Native bindings regenerated/formatted and compared
  byte-identically too. Default CPU tests do not link Vulkan or read third_party.
- Independent C orchestration + Python scalar expected hashes executed actual GPU
  work before native device code. 7 affine sizes (1,63,64,65,5120,65537,1048576 words),
  all output words and 64-word untouched tails; 3 transfer roundtrips (256B/1MiB/64MiB).
  Repeated full fixtures and compiled shader are byte-identical. Shader SHA
  `6708045a6c97ed0afdf465f4e785cb4908a69d3a9c39faf7d83baa6f202e0570`.
- Allocation diagnostics later added to the C harness, then regenerated twice;
  **all expected input/output hashes and the shader stayed identical**, not changed
  to accommodate native results. A subsequent disabled-memory-feature correction
  and explicit cross-replay barriers were independently regenerated twice, again
  with every expected output unchanged. Current dispatch fixture SHA
  `c0483629fb7e9c43a8fb8908c1ab504af91245a4fe6155f3fec9d9fe820e0368`.
  [Initial external log](../bench/data/2026-09-22-gpu-driver-bringup/oracle-first.log),
  [allocation-aware log](../bench/data/2026-09-22-gpu-driver-bringup/oracle-allocations.log).

## Native observations

Real driver tests pass in Debug and ReleaseFast on RADV7900XTX, selected compute
queue family1 at runtime. All 10 full-output cases match independent reference
hashes, including rerecorded/replayed command behavior and partial workgroups.
This is our Zig driver orchestration and our diagnostic shader; no foreign inference
library, shaderc runtime or subprocess is used by the native executable.

Negative/resource gates include zero/out-of-range sizes, coherent memory selection,
insufficient-budget rollback, busy child/pending command destruction/reset/re-submit,
zero-time fence polling, finite-timeout policy, source/destination overlap/alignment,
layout/push/module/group checks, cross-device rejection, 128-buffer/64-kernel/32-command
caps, 64-buffer/32-kernel retained-reference caps, and resource-counter return to zero.
Default CPU tests verify raw VkResult preservation/device-loss state without hardware.
Actual device loss/OOM is not deliberately induced; do not claim hardware fault
injection or full failure-branch coverage. Polling can complete immediately on a fast
GPU; timeout-state checks execute when zero-time polling actually returns timeout.
No Vulkan validation layer is installed, so no validation-layer-clean claim.

## Failures and fixes retained

1. ABI inventory initially tried treating `void` and `VK_DEFINE_HANDLE` as structure
   dependencies; both failed before any fixture. Primitive/handle categories fixed.
2. A transcribed loader SHA missed two characters, causing preflight refusal before
   compilation/execution. Corrected against the actually measured library hash.
3. Registry comments contained `[_DYNAMIC]`, which was initially mistaken for an array
   dimension. Generator now excludes comment text. The registry also contains a
   Vulkan SC alternative `pName` member; generated Zig initially had duplicate pName.
   Filtered api="vulkansc" nodes, regenerated the **C** ABI fixture twice and verified
   all sizes/alignments/field offsets/constants were unchanged. Only helper/C-extractor
   provenance changed. No ABI expectations were patched from native output.
4. Zig ABI introspection needed explicit comptime iteration/branch quota. Taking
   addresses of all extern functions through introspection also introduced Debug-info
   linker references; skip function declarations before @field. CPU tests remain
   driver-free rather than papering over the failure by linking them to Vulkan.
5. Linking only libvulkan with Zig's non-libc startup crashed at vkCreateInstance.
   The system loader/ICD requires the OS libc startup/TLS contract. GPU targets now
   explicitly use **system libc + Vulkan**, within the OS/raw GPU API boundary.
   No C/C++ engine or math code was added. Default CPU targets remain unchanged.
6. Zig0.16's default linker rejected system GCC16 crt1.o `.sframe` relocations:
   `unhandled relocation type R_X86_64_PC64`. LLD without LLVM also crashed the build.
   GPU targets explicitly select the bundled LLVM **and** LLD, after which real
   dispatch passed. No compiler/CRT/system package replacement or section stripping.
7. A negative dispatch test assumed UINT32_MAX exceeded maxComputeWorkGroupCount.x.
   This card actually reports x=UINT32_MAX, y/z=65535. Corrected the test to use
   queried limits plus1 on representable axes. The rejected assertion occurred
   while recording; the huge dispatch was never submitted. Numerical fixtures
   and implementation limits were not loosened.
8. First benchmark preflight glob included card1-DP-1 connector directories, causing
   `IsADirectoryError .../card1-DP-1/device/device`. Restrict to exact card+digits.
   No timed trials ran; [failed record](../bench/data/2026-09-22-gpu-driver/manifest.json).

9. **Allocation audit found a substantive shared oracle/native policy bug.** Both
   picked host type10 with propertyFlags0xce (DEVICE_COHERENT/UNCACHED_AMD), but no
   deviceCoherentMemory feature was enabled. The driver accepted allocations and
   all outputs matched; nevertheless this violates
   **VUID-vkAllocateMemory-deviceCoherentMemory-02790**. Initial run1/repeat manifests
   are explicitly **invalidated**, with raw timings/source snapshots preserved.
   Filter every non-base flag (allowed0x0f), test unsupported flags on both host and
   device paths, check real chosen flags, rerun independent C goldens before the
   native correction, then rerun tests/measurements. Added the core HOST_CACHED ABI
   constant from the official header; all prior ABI expected values stayed equal.
   This is why shared-driver/output agreement alone is not full API validation.
10. Also made cross-replay dependencies explicit: transfer→transfer before each
    roundtrip, transfer→compute before dispatch (including replay after readback),
    and compute→transfer in the state test before overwriting a shader output.
    Added execute→readback→execute coverage. No numerical expectations changed.

GPU executables' direct ELF dependencies are libvulkan.so.1, libc.so.6 and the
system ELF loader; no ggml/llama/shaderc dependency. Driver implementation dependencies
remain infrastructure behind the raw system API. Inspect actual manifests rather
than carrying the earlier all-static CPU executable claim over to GPU executables.

## Explicit rerun commands

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
zig build gpu-test --summary all
zig build gpu-test -Doptimize=ReleaseFast --summary all
python3 tests/reference/generate_vulkan_abi.py \
  --output third_party/NEW-vulkan-abi.json --work third_party/NEW-vulkan-abi-work
cmp tests/fixtures/gpu/abi.json third_party/NEW-vulkan-abi.json
python3 tools/generate_vulkan_bindings.py --output third_party/NEW-vulkan-bindings.zig
zig fmt third_party/NEW-vulkan-bindings.zig
cmp src/gpu/vk.zig third_party/NEW-vulkan-bindings.zig
python3 tests/reference/generate_vulkan_goldens.py \
  --output third_party/NEW-vulkan-dispatch.json \
  --shader-output third_party/NEW-affine.spv --work third_party/NEW-vulkan-dispatch-work
cmp tests/fixtures/gpu/dispatch.json third_party/NEW-vulkan-dispatch.json
cmp tests/fixtures/gpu/affine.spv third_party/NEW-affine.spv
python3 bench/run_gpu_driver.py --cpu 10 --output docs/bench/data/NEW-gpu-driver
```

New paths required. Ordinary CPU/explicit GPU tests need only repo files and the
system GPU runtime; regeneration additionally needs the pinned public research
header/XML, glslc/spirv-val and external C compiler/OpenSSL development interface.
No old executable is required by the benchmark runner.
