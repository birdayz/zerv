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
  the three f16-copy producers bring it to 71 kernels on 4 banks; FP32 needs 55.
  Command objects raised to 256 in block 18c.2: with several slots every prefill plan is
  also recorded as 16 layer segments (80 commands for 5 plans).
  Raised in block 17b to 192 kernels and 64 command objects: speculative verification
  compiles one multi-row module per row count (5 kernels per rows pipeline), and the MTP
  draft path records a draft command per (first-pass rows, drafts), 20 at 5 rows. A
  Raised to 256 kernels in block 18e part 4 (2026-09-26): the batched decode's `gemm_rows`
  modules, one per row count 1..8 per weight bank.
  command may retain 64 distinct kernels (was 32): with the KV cache split over several
  buffers, the attention kernels exist once per KV buffer.) Subgroup properties
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
  changes. Entrypoint main; shader contract is caller's.
- **Specialization constants** (added 2026-09-24 for `--kv-page-tokens`,
  [concurrent.md](concurrent.md)): `Kernel.Options.constants` holds up to 8 u32 values.
  `constants[i]` is `constant_id = i`, given at pipeline creation as a
  `VkSpecializationInfo`. The driver compiles them as literals, so there is no per-dispatch
  cost. An ID the module does not declare has no effect (Vulkan). More than 8 is
  `InvalidLayout`. The empty default passes no specialization info, exactly as before.
- Commands own one reusable pool/primary command buffer/fence, fixed retained resource
  arrays (64 buffers/32 kernels). Record copy, barrier, dispatch with explicit bounds.
  No per-submit allocation, descriptor updates or rerecording. State guards reject
  illegal begin/end/submit/wait/reset/destroy transitions. Timeout leaves pending.
  Successful wait permits replay; reset releases recorded references. Objects cannot
  be copied after dependents take their address. Device destruction rejects children.

- **Pipeline binaries** (added 2026-09-24, before implementation; evidence
  [gemm_f16x ISA report](../bench/2026-09-24-gemm-f16x-isa.md), decision D7). Optional
  `Options.pipeline_binaries`: when the device lists `VK_KHR_pipeline_binary` and its
  dependency chain at Vulkan 1.1 (`VK_KHR_maintenance5`, `VK_KHR_dynamic_rendering`,
  `VK_KHR_depth_stencil_resolve`, `VK_KHR_create_renderpass2`) and the `pipelineBinaries`
  feature, `open` enables them (plus the `maintenance5` feature) and records the driver's
  global key (`vkGetPipelineKeyKHR` with no create info; RADV: a hash of the driver build,
  the GPU's compiler info and the compiler options) in `Device.pipeline_key` (32 bytes).
  If anything is missing, `pipeline_key` stays null and `open` succeeds: native kernels are
  an optimization with a fallback, not a requirement. The three KHR commands are fetched
  with `vkGetDeviceProcAddr`; a null pointer counts as unsupported.
  `Kernel.Options.binary` (data ≤ 1 MiB, binary key 1..32 bytes, expected global key):
  when the device's key equals the expected key, the pipeline is created from the binary
  (`vkCreatePipelineBinariesKHR` from key and data, then `vkCreateComputePipelines` with
  `VkPipelineBinaryInfoKHR`; the binary handle is destroyed right after) and
  `Kernel.native` is true; otherwise the kernel is created from its SPIR-V exactly as
  without the option. A driver error on the binary path is returned, never retried with
  SPIR-V. The binary's contents (machine code, register/LDS configuration, ABI) are the
  caller's trusted contract, like the SPIR-V: the driver copies them without checks.

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

## Imported coherent host buffers — NVMe prerequisite (2026-09-28)

Opt-in `Options.host_import` enables `VK_EXT_external_memory_host`; ordinary allocation
and serving defaults are unchanged. The implementation queries (not assumes)
`minImportedHostPointerAlignment` and external-buffer import support for our exact usage
(transfer source/destination plus storage). Reject unsupported and dedicated-only imports.
Only ordinary `HOST_ALLOCATION` payloads are supported, not foreign mapped device memory.

`Buffer.initImported(device, memory)` borrows a caller-owned, stable host range. It must be
positive and its address and length multiples of the queried alignment; its backing pages
must remain valid, accessible and unmodified by concurrent CPU/disk operations while GPU
commands use them. No overlapping or duplicate imports. The caller supplies adequate
padding for buffer memory requirements. Intersect the pointer's memory-type bits with the
buffer's bits and select a HOST_VISIBLE | HOST_COHERENT type. Validate requirements and
aggregate device allocation budget before importing. Failure rolls back all Vulkan objects
and leaves accounting unchanged. Success counts imported bytes towards the same budget.

`mapped` borrows the original pointer, not a second mapping. GPU writes must be made visible
to host before CPU/disk reads, and CPU/disk writes must finish before submission. The existing
pending-command and retained-reference checks apply. `deinit` destroys buffer then Vulkan
memory, never unmaps/frees the caller's range. The caller frees it afterwards. Independent
ABI oracle: pinned Vulkan-Headers C probe (`vulkan_fixtures_update`), same as existing GPU
bindings. Device correctness: distinct-byte imported-host → device → different imported-host
round trip, invalid alignment/length, option disabled, aggregate budget exhaustion with
unchanged accounting, destruction while retained. Run both modes and the production driver.

Primary semantics: Vulkan-Docs v1.4.357, `VkImportMemoryHostPointerInfoEXT`,
`vkGetMemoryHostPointerPropertiesEXT` (VUIDs 01747–01755), allocation-size VUID 01745,
and extension issues 2–6. Source provenance: research/nvme note. This remains within the
existing OS/GPU-driver boundary; no new runtime library.

## Asynchronous completion ownership (18d.7a)

2026-09-28 pre-implementation amendment; supersedes the earlier device-wide host
mapping prohibition, not the barriers or stable-owner contract.
[Resolved research](../research/2026-09-28-async-transfers.md),
[ordered integration plan](../design/async-tiering.md).

`Commands.poll() Error!bool` is a zero-time fence wait: false preserves pending
state and every resource lease; true acknowledges completion exactly once and
returns the command to executable. Other errors propagate using the existing wait
contract (device loss is terminal; uncertain ownership is not completion).
Polling an unsubmitted/already acknowledged command is InvalidState. `run` remains
submit + bounded wait for synchronous callers.

`Buffer.pending_uses` counts outstanding submitted references, including kernel
bindings. Copy/dispatch duplicates may contribute multiple balanced uses. Counters
are bounded by command/binding limits, externally serialized, allocation-free.
Increment only after successful submission, decrement on acknowledged completion
or terminal lost-device cleanup; never on timeout. `mapped` rejects pending uses
of **that buffer**, not unrelated device commands. Returned spans cannot be used
across a conflicting submission. Recorded references still prevent destruction
until reset/deinit even after completion. Device.pending still counts commands.

Acceptance before advancing: native-independent byte/affine goldens, two pending
owners of one allocation and unrelated host access, both acknowledgment orders,
replay, kernel-only bindings, double-poll and pending reset/destroy rejection;
`bazel test //...`, both GPU modes/spill gate and host GPU gate. Run the existing
matched native/C driver benchmark (fixed sizes, warmup, seven trials) to quantify
counter overhead; this is a component change, not a serving speedup.
