# Native Vulkan infrastructure — research 2026-09-22

Block 07, scoped to Linux x86_64 device/memory/transfer/compute dispatch. No model
math, quant GPU kernel, framework, or foreign inference implementation in this block.

## Primary sources and executable oracle

Vulkan-Headers v1.4.354 resolves to **01393c3df0e5285b54ee6527466513f9e614be94**;
Vulkan-Docs v1.4.354 to **ea5259d68356334a2928d5d6c327ccaea2f2af08**. Retained core
header, XML API registry and relevant spec chapters under
`third_party/vulkan/1.4.354/`; [per-file URLs/hashes](2026-09-22/vulkan-sources.json).
Inspected fundamentals, device/queue creation, buffer allocation/binding/mapping,
command lifecycle, synchronization/host memory domains, compute dispatch,
descriptors and compute pipeline contracts. Existing pinned llama-vulkan.cpp is
research only: its coherent staging buffers, compute/transfer barriers and host
readback confirm the applicable API sequence, not an implementation to copy.

An external C harness using the pinned public **C system-API header** is permitted
as an ABI/driver oracle. Extract sizeof/alignof/every field offset before native
bindings, and exercise independently authored transfers + a deterministic unsigned
integer affine shader against independent CPU expected values before native code.
A raw Vulkan benchmark is the equivalent driver-component comparison; llama-server
has no same-operation endpoint. This cannot establish an inference speedup.

## Observed platform / toolchain

RX7900XTX RADV NAVI31, Vulkan device1.4.354, loader1.4.357, Mesa26.2.3, no Vulkan
validation layer installed. `vulkaninfo --summary` and full capture were run again;
full capture retained under third_party/research/2026-09-22/vulkaninfo-full.txt.
There are compute+transfer queues (graphics family 0:1, dedicated compute family1:4),
maxStorageBufferRange=4294967295, maxComputeWorkGroupInvocations=1024,
nonCoherentAtomSize=64. Query at runtime; do not encode target-specific indices.
Host visible+coherent system-memory and device-local memory types exist. VRAM heap
23.984375 GiB; reported available budget ~22.95 GiB at this capture, not a guarantee.
No packages/drivers/clocks changed. Missing validation layer is a verification
limitation, not grounds to silently install one.

Installed tools: glslc2026.3 (SPIRV SDK1.4.357), spirv-val2026.3
`vulkan-sdk-1.4.357.0-0-g9a49b0883`. SHA256:

- glslc: 4a4743cde357af0949cdbc07668802297327e993e548f80f4e2ee67ba9b6c74d
- spirv-val: 02fae2475ba0f3cb4987aca3c9eb938e8543cf269244bfcb856d8e0c252c753b
- /usr/lib/libvulkan.so.1: 7d9f8ced1fae1d02ee7953a5a22e4efae74545e83333eedc41c53a50f51ebab6
- /usr/lib/libvulkan_radeon.so: ddc11778b3e01b73d55028595a6dfc51afd8e3cb5f901fd175a74cdc9ba79248

Compile our own diagnostic GLSL with `glslc --target-env=vulkan1.1 -O`, validate
SPIR-V before accepting it, record source/compiler/module identities. Compiler is
a development tool, never a shaderc runtime dependency. Checked-in compiled fixture
allows ordinary CPU and explicit GPU tests without a shader compiler or headers.

## Resolved semantics / bounded first design

- Use Vulkan1.1 core APIs (no extensions/features required by the integer diagnostic).
  Direct C ABI calls to the installed **system Vulkan loader** are within the raw
  GPU infrastructure boundary. No C/C++ inference/math runtime. The loader itself
  depends on system libc and requires its startup/TLS; GPU targets explicitly link
  libc and use bundled LLVM+LLD (the default Zig linker rejects system crt .sframe).
  Don't write a private ELF loader. Zig DynLib without
  libc selects a limited ElfDynLib path, so prefer explicit system Vulkan linkage
  only for GPU executables. Ordinary native component tests remain driver-free.
- Own opaque handles and extern ABI declarations in Zig. API declarations are
  mechanically derivable from the pinned public registry, not copied inference
  code. ABI fixtures independently compile the official header. Never import that
  header or third_party at native build/runtime.
- Explicit device selection; default first suitable discrete compute device, not
  hardcoded AMD/index. Fixed bounded enumeration, one externally serialized queue;
  reject overflow/no suitable device. Limits/memory properties queried at init.
- Each buffer owns an allocation; bind at offset0 with the queried compatible type
  bits and actual allocation requirement. Charge requirement.size (not logical
  size) against explicit byte/allocation budgets before vkAllocateMemory. First
  host path requires HOST_VISIBLE|HOST_COHERENT, prefers non-device-local memory;
  explicitly reject absent coherent memory, rather than forgetting flush/invalidate.
  Noncoherent support/suballocation are separate measured extensions, not hidden
  assumptions. **Only base flags 0x0f are accepted.** Primary spec
  VUID-vkAllocateMemory-deviceCoherentMemory-02790 forbids allocating a DEVICE_COHERENT
  AMD type unless deviceCoherentMemory is enabled; advertised types are not permission
  to use disabled features. Likewise reject protected/lazy/unknown extension flags.
  An allocation-metadata audit found the initial C/native policies selected type10
  (0xce); their first timings were invalidated despite matching outputs. Corrected
  fixtures use core-only types, with unchanged output hashes and explicit cross-replay
  barriers. Device buffers prefer DEVICE_LOCAL. No per-dispatch allocations.
- Coherent host writes before QueueSubmit are available to device automatically.
  Coherent is **not synchronization**: compute/transfer writes still need explicit
  stage/access dependencies, and GPU→host needs a HOST_READ destination dependency
  plus fence completion before the CPU reads. Read synchronization spec sections
  synchronization-host-access-types, buffer-memory-barrier and Host Write Ordering.
- Copy offsets/count nonzero and 4-byte aligned, checked bounded ranges, nonoverlap,
  same device. Explicit transfer→compute, compute→transfer, transfer→host and
  compute→compute dependencies. One queue means no queue-family ownership transfer.
- Reusable command pool/buffer/fence: initial→recording→executable→pending;
  finite wait success returns executable, timeout stays pending. No one-time or
  simultaneous-use flags. Pending commands cannot reset/destroy/re-submit.
  Recorded commands retain referenced resources until reset/destruction; pipelines
  retain descriptor buffers. Bounded references prevent dangling handles.
- Shader modules are trusted, validated development artifacts, not arbitrary user
  uploads. Pipeline layout/storage-buffer bindings and push constants are checked
  against device limits; dispatch group counts checked. This package cannot prove
  arbitrary shader memory safety; caller's shader/shape contract remains mandatory.
- Device loss/driver failures are errors, not successful completion. No implicit
  device-wide wait in hot code, no external engine delegation. Device teardown
  requires children gone; owning objects must stay at stable addresses.

[Functional specification](../specs/gpu-driver.md) owns implementation acceptance.
