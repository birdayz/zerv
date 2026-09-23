# Serving robustness fixes after a two-server GPU incident (2026-09-24)

## Incident (observed)

- A leftover zerv test server was still loaded on port 18080 (f16 prefill, 8
  prefix-cache slots, about 21 GB of VRAM). The user then started a second zerv on the
  same port.
- The second bind succeeded. zerv listened with Zig 0.16's
  `listen(.{ .reuse_address = true })`, which sets SO_REUSEPORT as well as SO_REUSEADDR
  (`lib/std/Io/Threaded.zig`, `netListenIpPosix`). So both processes listened on
  127.0.0.1:18080 (`ss -ltnp`) and split the connections between them.
- VRAM use reached 25.3 GB on the 24 GB card. The kernel logged `ring comp_1.2.0 timeout`
  for the user's process at 16:21:25 and reset the compute queue. Vulkan reported
  VK_ERROR_DEVICE_LOST, and the request failed with `generation failed: DeviceLost`.
- After that, the process kept answering `/health` with 200 while every request failed.
- Stopping either server printed `zerv: stopped`, then panicked with `snapshot store in
  use`. `Model.cleanup` freed the prefix-cache snapshot buffer before the copy command
  that still retained it (and the state arena). The process exited with SIGABRT.

## Fixes

1. **Cleanup order** (`src/model/runtime.zig`): the snapshot copy command is released
   before the snapshot store.
2. **Exclusive listener bound before loading** (`src/serve/listen.zig`, `src/main.zig`):
   SO_REUSEADDR only, never SO_REUSEPORT. A busy port fails before the model is loaded.
3. **Free-VRAM check before any device allocation** (`Model.init`,
   `gpu.Device.memoryBudget`):
   - Needed bytes are weights + activation arena + state arena + snapshots + 256 MiB of
     headroom. They are compared with `VK_EXT_memory_budget`'s heapBudget − heapUsage.
   - The Vulkan bindings gained `vkGetPhysicalDeviceMemoryProperties2`,
     `VkPhysicalDeviceMemoryProperties2` and `VkPhysicalDeviceMemoryBudgetPropertiesEXT`.
   - These were generated from the pinned registry with `tools/generate_vulkan_bindings.py`,
     and the C-ABI fixture was regenerated with `tests/reference/generate_vulkan_abi.py`.
     Two regenerations were byte-identical (`third_party/vulkan-budget-2026-09-24/`).
     The inventory is now 53 structs and 74 constants.
4. **Unusable engine** (`src/serve/http.zig`, `src/serve/engine.zig`, `src/main.zig`):
   - After a failed generation, a lost device or a pending command marks the server
     failed.
   - Health checks then return 503 `failed`, chat requests 503 `engine_failed`, serving
     shuts down, and the process exits with status 3.

Contract: [serving spec, Startup and Engine failure](../specs/serving.md).

## Verification

- `zig build test` (Debug and ReleaseFast): 81/81. New tests:
  - `listener: exclusive port, immediate rebind after connections close`. A second
    listener is refused, both with our socket and with std's `reuse_address` socket. The
    port rebinds immediately while a server-closed connection is in TIME_WAIT.
  - `HTTP server: an unusable engine fails health checks and requests, and ends run`.
    With a fake engine: 500 for the failing request, then 503 `failed` and
    503 `engine_failed`, `run` returns `EngineFailed`, and connections are refused.
- `zig build gpu-test` (Debug and ReleaseFast): 18/18, including `memory budget: the
  device-local heap reports this process's own allocations`. With a 1 GiB buffer, usage
  grows by 1 GiB and free space shrinks by 1 GiB.
- The Python suite passes (the ABI fixture test now expects 53/74). `zig fmt --check`
  is clean.
- **Real binary** `fd3f6770…` (built from the current source):
  - [`tools/check_shutdown.py`](data/2026-09-24-serving-fixes/shutdown.json): SIGINT
    mid-stream drains the stream and exits 0 with the prefix cache enabled (8 slots).
    The panic is gone.
  - [`tools/check_exclusive.py`](data/2026-09-24-serving-fixes/exclusive/report.json),
    server A at context 29,504, fp32:
    - B on the same port: exit 1 after **1.1 ms** with "127.0.0.1:18100 is already in
      use", before loading. VRAM unchanged.
    - C on another port: exit 1 after 3.0 s with "the model needs 22180 MiB (including
      256 MiB headroom), 1809 MiB are free", before any device allocation. VRAM
      unchanged.
    - A stayed healthy, answered a greedy request, and exited 0 on SIGINT.
  - Idle free VRAM reported at startup: 23,738 MiB. That is 24 GiB less the desktop's
    about 0.82 GB, so RADV's budget includes other processes. A actually used about
    21,930 MiB (sysfs), against the 22,180 MiB estimate including headroom.

## Limitations

- A real device loss was not reproduced on purpose. The server-side behaviour is tested
  with a fake engine. The native engine's `usable` is `!device.lost and device.pending
  == 0`, and `device.lost` is set when Vulkan returns VK_ERROR_DEVICE_LOST, which is what
  the incident returned.
- The VRAM check is a snapshot at startup. Two servers starting at the same moment, or
  another process allocating later, are not prevented.
