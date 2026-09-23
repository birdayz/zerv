# Native GPU driver package — block 07

**Pre-implementation contract.** [Completed scoped research](../research/vulkan-driver.md).
Exactly one active package, src/gpu. No GPU model operators in this block. Raw
Vulkan1.1/Linux x86_64 only initially; unsupported targets/capabilities fail explicitly.

## Interface and ownership

- Device owns instance/logical device/one compute queue and queried physical limits/
  memory properties. Explicit device index optional; default suitable discrete GPU.
  Fixed enumeration bounds (16 devices/32 queue families), explicit memory budget,
  bounded child counts (128 buffers, 128 kernels, 32 command objects). (Briefly raised
  to 128 kernels for three per-tile prefill plans in block 13d; reverted to 64 in 13e,
  which used one GEMM kernel per format, ~45 kernels in total. Raised to 128 again in
  block 16b: a kernel binds one weight bank, and the f16 mode's two WMMA kernels plus
  the three f16-copy producers bring it to 71 kernels on 4 banks; FP32 needs 55.) Subgroup properties
  (`vkGetPhysicalDeviceProperties2` + `VkPhysicalDeviceSubgroupProperties`, Vulkan 1.1
  core) are queried at open and exposed as `Device.subgroup`.
  The optional `Options.cooperative_matrix` (block 13g) enables
  `VK_KHR_cooperative_matrix`, together with the `cooperativeMatrix`, `shaderFloat16`
  and `storageBuffer16BitAccess` features. It first checks the extension list and
  `vkGetPhysicalDeviceFeatures2`, and fails with `UnsupportedFeature` if any is
  missing. The default is off. Today only research tools and `tests/gpu_coopmat.zig`
  use it; the model does not ([research](../research/coopmat-prefill.md)). No other
  extension or feature is enabled.
  No hidden singleton; all operations externally serialized, stable owner addresses.
- Buffer owns Vulkan buffer+memory, logical bytes, actual allocation bytes and optional
  persistently mapped coherent host span. Device-local or host-coherent locations.
  Allocation requires positive bounded logical size, compatible memory type, and
  available aggregate requirement-size budget. Only base DEVICE_LOCAL/HOST_VISIBLE/
  HOST_COHERENT/HOST_CACHED memory flags are supported: reject every memory type
  with protected/lazy/extension bits, since no optional feature is enabled. In
  particular honor VUID-vkAllocateMemory-deviceCoherentMemory-02790. Own no overlapping suballocations.
  Destruction fails while referenced. Host spans borrowed; don't access while any
  command on that device is pending. Callers must not retain/use spans during submit.
- Kernel owns a trusted SPIR-V compute pipeline, layout, descriptor pool/set and
  fixed storage-buffer bindings (1..8) plus bounded push-constant byte count (<=128,
  multiple of4 and device maximum). Descriptors use whole logical buffers, within
  maxStorageBufferRange. Referenced buffers outlive the kernel. No mutable descriptor
  changes or specialization features yet. Entrypoint main; shader contract is caller's.
- Commands own one reusable pool/primary command buffer/fence, fixed retained resource
  arrays (64 buffers/32 kernels). Record copy, barrier, dispatch with explicit bounds.
  No per-submit allocation, descriptor updates or rerecording. State guards reject
  illegal begin/end/submit/wait/reset/destroy transitions. Timeout leaves pending.
  Successful wait permits replay; reset releases recorded references. Objects cannot
  be copied after dependents take their address. Device destruction rejects children.

Native public package hides driver details from future model code. ABI declarations
are internal except testing; ordinary CPU tests do not link the driver. Explicit
GPU test/benchmark targets link only the system Vulkan loader, not ggml/shaderc.

## Correctness and errors

Copy nonzero multiple-of4 byte ranges; all arithmetic/ranges checked before recording,
no overlaps and same device. Buffer usage includes transfer source/destination and
storage. Dispatch groups must fit queried maxComputeWorkGroupCount; push-constant
length must match kernel contract. Kernel creation validates SPIR-V header/alignment/
size and layout limits before driver calls; arbitrary user shaders unsupported.

Typed barrier scopes: transfer (transfer stage/read+write access), compute (compute
shader stage/read+write), host (host stage/read+write). Whole-memory barriers initially;
conservative correctness over performance guessing. Native command dependencies must
match the independently validated API sequence. Coherent mapped memory does not remove
barriers or fence waits. No host access while pending. All create paths unwind acquired
handles on failure. Resource limits/busy/state/unsupported/driver errors distinguished;
VkResult retained in diagnostics, never swallowed. No retry loops after device loss.

## Independent fixtures before implementation

1. Compile official pinned C header, extract size/alignment/each used field offset and
   used numeric constants; generate JSON. Native ABI tests compare all entries, no
   production C import. Pin generator/header/XML and repeat byte-identically.
2. Compile/validate an independently CPU-checkable diagnostic shader: for word i<n,
   output[i] = (input[i]*1664525 + 1013904223) XOR i, modulo2^32. Local size64;
   guarded partial last group. Tail sentinels unchanged. Test lengths1,63,64,65,
   5120,65537,1048576 with seeded/adversarial uint32 values, exact bits only.
3. External C API harness executes upload→device→download transfers and actual affine
   dispatch before native code. Independent scalar expected values verify every
   output. Store tool/driver/device/module/source IDs and raw logs. Same API driver
   is infrastructure, not an independent hardware implementation; independent ABI/
   scalar arithmetic and separately authored orchestration establish the useful gate.

Native tests: resource/extent arithmetic, memory selection, ABI all-fields, malformed
module/push/bind/group inputs, failure cleanup via deliberately insufficient budget,
state transitions, referenced-child rejection, repeated create/destroy, real transfers,
partial workgroups, sentinels, rerecord/replay, output equality and no leaked children.
Run Debug/ReleaseFast CPU+explicit GPU tests. No validation layer is installed; record
that limitation, don't claim a clean validation-layer run or install it incidentally.

## Measured acceptance / replay

Reusable recorded commands, preallocated buffers, pipeline compilation/host fill/hash/
readback checks outside timed loops. Direct synchronous submit+fence-wait for both
native and external C; wall time includes API submission and GPU completion, not GPU
kernel-only time. Transfer roundtrip sizes256B/1MiB/64MiB, affine dispatch sizes
65/5120/1048576 words. Identical warmups (3), trials (7), three alternating rounds;
full output/sentinel hashes before accepting trials. CPU10, record device/driver/tool
hashes, effective device/queue/memory types and bytes, no clock/power changes, retain
variance/losses. Fresh run directories/source snapshots; checked-in runner must rebuild
and revalidate without an old executable. Repeat and record before closing07.

There is no same-operation llama-server endpoint or exposed arbitrary SPIR-V dispatch;
compare direct raw-driver C work and document this distinction, not a llama-server
speedup. Next08's quantized matvec must compare equivalent reference GPU operators;
model/HTTP milestones still require actual tuned llama-server comparisons.
