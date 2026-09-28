# Asynchronous transfers: ownership prerequisite

2026-09-28; source inspection before implementation. [Plan](../design/async-tiering.md).
The paper findings and exact artifacts are in [the tier review](2026-09-28-kv-tier-papers.md).
Re-read LMCache v2 §5.2–5.3 (start/wait hooks, reference-counted copies, bounded
window) and Mooncake `BatchEvict` at the pinned revision: pending offload keeps a
source reference; a completed disk replica allows memory eviction without rewriting.
These motivate ownership and policy, not Vulkan synchronization semantics.

## A: resolved GPU semantics

Primary Vulkan-Docs **v1.4.354**, commit
`ea5259d68356334a2928d5d6c327ccaea2f2af08`,
[URL](https://github.com/KhronosGroup/Vulkan-Docs/blob/ea5259d68356334a2928d5d6c327ccaea2f2af08/chapters/synchronization.adoc).
Already retained at `third_party/vulkan/1.4.354/spec/synchronization.adoc`; original
hashes/URLs in [ledger](2026-09-22/vulkan-sources.json). Re-read lines 2800–2885
(`vkWaitForFences`, fence-wait execution dependency) and host access sections.
A zero timeout **does not wait**; VK_TIMEOUT means incomplete, not lost ownership.
VK_SUCCESS establishes the host dependency for that fence. It is not a global
completion event for other commands. Coherent memory still requires synchronization
of conflicting accesses, not a blanket prohibition on unrelated host allocations.

Our `Buffer.mapped` currently rejects any device-wide pending command, including
unrelated archive DMA. Actual model decode/prefill maps separate host I/O buffers,
so nonblocking archive submission would currently fail with ResourceInUse. Change
the guard to **pending uses of that allocation**. Whole-buffer tracking remains
conservative (not per-range). Memory allocations do not alias, including the explicit
no-overlapping-import contract. Existing raw imported spans still require their
owner's range/ticket protocol; a map guard cannot revoke an already borrowed pointer.

Recorded command references are separate from pending uses. Copies retain buffers;
dispatches retain kernels, whose immutable bindings retain buffers. On successful
submit increment pending uses of every recorded buffer and every retained kernel
binding; duplicates are allowed and balanced (bounded by 256 commands × (64 direct
buffers + 64 kernels × 8 bindings), far below u32). On successful wait or terminal
device loss decrement exactly those uses once. Timeouts preserve them. Replay adds
them again. Reset/deinit still rejects pending commands and releases recorded refs
only when legal. No allocator or per-submit resource discovery, and no changed
numerics, layouts, queue family, transfer barriers or shader.

Alternative: deduplicate kernel bindings into the direct buffer array. Rejected for
this increment: it changes the record-time 64-buffer capacity contract of model
commands and is not needed for correctness. Measure the bounded counter loops rather
than assume their cost is zero. Hardware driver failure is not deliberately induced.

Executable correctness mechanism already available before code: checked-in
independent C Vulkan + Python scalar fixtures in `tests/fixtures/gpu`, exercised by
`tests/gpu.zig`; independently authored C benchmark in `bench/run_gpu_driver.py`.
Add explicit multiple-owner/independent-buffer lifecycle tests using those same
bytes. A fence can complete before the first poll, so deterministic ownership tests
assert before acknowledgment, never require a hardware timing race. Fake/delayed
completion belongs to the archive increment, not a fake GPU driver.

Observed setup: `bazel version` reports 9.2.0; Zig 0.16.0 is the pinned build input.
`/sys/class/drm/card*/device` reports AMD 1002:744c, idle (0% busy), 556,429,312 VRAM
bytes used at inspection. No background inference process found, system changes,
new packages or model downloads. An initial `bazel --version` failed (startup option);
`bazel version` is the working command.

## B: resolved archive lifecycle (research while A is being verified)

The existing `storage.Store` already exposes the necessary held/done spans: acquire
returns a generation-qualified held ticket; buffer() permits held/done only; submit
requires held; release rejects queued/active. Disk completion can be polled repeatedly.
The archive must add its own GPU lease; storage cannot know a held/done span is in DMA.

Use one archive-wide pending device quantum (`ticket, slot, chunk`), independent
of the disk-pending array. Job.pending counts both types. For D2H: acquire/zero →
start GPU → poll complete → hash entire padded span → disk submit → exact completion
→ release. For H2D: disk submit → exact completion/hash check → start GPU → poll
complete → release. Transitions move the same ticket, without extra copies or
allocations. Completion is never inferred from submission, another fence or a
cancellation request. Source slot stays paused; maps/positions remain stable.

At most one start and one device poll per advance, with one disk completion and
one new disk request; a false device poll does not prevent other slots' disk progress.
A callback start/poll error may unwind only if device access has ended. Production
panics on uncertain ownership. Cancellation records failure, stops new chunks and
still polls/drains the pending device and every disk ticket. Read corruption marks
the record bad for all readers; each reader drains its own uploads before resetting
its private model destination. Wrong/short disk results remain recoverable misses,
not GPU faults. The immutable record remains valid after mere cancellation.

Split the model's archiveCopy into submit plus its existing synchronous convenience
wrapper, with identical validation/addressing/barriers. Disk borrows the caller's
std.Io for monotonic deadline checks: poll(0), then fail-stop if still incomplete
past the existing model GPU timeout. No timed wait, busy wait or new timeout knob.
The single compute-capable queue remains; source-slot dependencies remain explicit.

ModelBackend's existing failure check considers any pending command fatal. With B,
the owned archive command is legitimately pending: subtract just that known command
when evaluating another operation's failure, while still treating device loss or
any unaccounted pending model command as fatal. No broad suppression of GPU errors.

Executable reference remains the independently generated POSIX/padded SHA256
fixture and C/FP64/libllama model gates; the scheduler/ownership protocol is ours,
not a CUDA/Mooncake reimplementation. Add delayed-copy fakes which deliberately
retain the span until explicit acknowledgment and inject failure at both start
and completion, never merely sleep hoping a race happens. Reuse the real storage
worker for byte/integrity/cancel tests. Exact production 80k images and independent
model logits are complementary gates, not interchangeable self-comparisons.

## C readiness finding: partial pages are not immutable full byte images

Further source inspection (not an implementation): `kvcache.fillEntry` pins the
source request's last partial page; it does not freeze the bytes after the checkpoint
position. Subsequent tokens can write that tail while the checkpoint still correctly
names its earlier tokens. The current archive avoids this by pausing the whole
source sequence. A background cache-source spill cannot simply claim a pinned page
is an immutable *whole-page image*. Only the meaningful prefix is immutable.

`layout.State` documents K as `[kv_head][dim][page]` and V as
`[kv_head][page][dim]`, one K/V piece per attention layer in each physical page.
A future source adapter must either freeze a private partial page at checkpoint
creation (budgeted ownership/copy) or canonically zero/exclude invalid tail elements
with explicit copy/write dependencies. Canonicalizing f16 tails also requires care
with 4-byte Vulkan copy alignment and packed key-pair reads. This decision needs
its own independent layout fixture and exact-prefix continuation test with the
original source advancing during spill. It must not be guessed inside a renamed
`evictHost` callback. No C implementation or weakened exactness gate in A/B.
