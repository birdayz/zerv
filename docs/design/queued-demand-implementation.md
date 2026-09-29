# D.2 implementation and acceptance plan

2026-09-29. This is the single active increment; the full performance goal remains open.
Research: [primary-source audit](queued-demand-audit.md), [source ledger](../research/2026-09-28-kv-tier-papers/sources.json).
Contract: [queued demand](../specs/queued-demand.md). Parent ordered plan: [async tiering](async-tiering.md).

## Paper-derived choices

CachedAttention 2403.19708v3 §3.3 motivates protecting soon-needed queued prefixes.
LMCache 2510.09665v2 motivates using admission delay for asynchronous reads.
Mooncake's pinned HiCache design motivates comparing local matches before fetching.
These sources do not establish numerical correctness or performance on this machine.
Native scope is one oldest queued request and one/two disk staging chunks, not
whole-conversation RAM prefetch or a distributed cache implementation.

## Ordered work and gates

1. **Independent fixture, then ownership boundaries.** Token-prefix sets specify
   protected ancestry; ticket-owner events specify staging, cancellation and handoff.
   Implement policy protection separately from immutable source leases, and separate
   disk issue from GPU upload. Fixed eight tickets; at least two reserved for demand.
2. **Scheduler/backend integration.** Borrow queued tokens only during a running,
   synchronous callback. Use slot plus arrival order for handoff. Drain cancellation
   before reuse; private-map validation precedes ownership transfer. Source conflicts
   wait on reclaim epochs instead of silently recomputing. Keep defaults off.
3. **Adversarial verification.** Debug/ReleaseFast fixture and directed tests: host
   promotion, ancestry changes, stale generations, callback leave/stop, retry epochs,
   foreground preemption, failed admission, corruption, and teardown. Remove upload
   and generation guards in negative controls and require tests to fail.
4. **Production-device/model verification.** CPU suite, GPU/spill and host-driver
   gates; windows1/2 × 257/80k archive states, full-vocabulary continuations, packed
   work, cancellation and handoff. Independent FP64/libllama oracle remains mandatory;
   native off/on equality is an additional integration check, not an independent oracle.
5. **Measure final binaries.** Repeated fixed-trace component measurements with actual
   submitted/completed bytes. Then serialized three-round serving comparisons against
   off/protect/prefetch and tuned Vulkan/HIP. Include workloads that actually prefetch,
   TTFT, token gaps, throughput, memory, quality and variance. Zero prefetch is a negative
   exercise result, not proof of prefetch performance. Preserve regressions and failures.
6. **Close only with evidence.** Record manifests and exact commands under docs/bench,
   update specifications and controlled queue, and commit verified scoped changes.
   Faster-or-on-par serving is not accepted from a component win or unequal token streams.

## Observed progress (not acceptance)

- Independent fixture generated before APIs: 108 cache and 210 staging cases,
  SHA256 `db8e161e83684e0b8f3b5bca20ea9be7d2eddb3e8c12b7c90865fe2e4bdbe64a`.
- Boundaries and opt-in scheduler/backend/CLI implementation exist; CPU suite passed
  82/82 targets (17 executed, remainder cached) in `cli-callback-fixed.log`.
- First production-model window1/257 run passed exact canonical state, four full
  vocabulary continuation rows, one independent packed row, two canceled generations,
  and one handoff. Manifest: `../bench/data/2026-09-29-demand-model-first/manifest.json`.
- Remaining gates above are still open. No D.2 serving/performance claim.

## Retained failures

Raw logs live in `../bench/data/2026-09-29-queued-demand/`. Initial fixture corruption
changed only block0, but out-of-order completion could legitimately upload another
valid block first. The zero-upload expectation required corrupting every block;
that fixture setup was corrected, not the runtime ordering. A test helper initially
placed a method between Zig struct fields; compilation rejected it. Moving the
method after all fields fixed the build (`cli-callback.log`, `cli-callback-fixed.log`).

Subsequent checks and remaining gates are recorded in the
[D.2 intermediate verification report](../bench/2026-09-29-queued-demand.md).
Both windows now pass 257/80k model checks and the independent model oracle passes
337/337 greedy comparisons. An upload-guard negative control exposed insufficient
waiting for completed disk reads; strengthening that fixture makes both guard
mutations fail in both build modes. No serving benchmark has yet closed this increment.

A first three-round serving evaluation now exists in that report: native prefetch2
45.910±.635s versus off51.144±1.277s, Vulkan54.110±.927s and HIP51.761±.933s.
All96 native outputs/counts match; competitor streams differ. Actual issue/handoff
occurred, but successful handoffs observed no completed speculative bytes. Do not
attribute the entire configuration difference to I/O overlap or claim quality parity.
Arrival exhaustion, twelve CLI rejections and production corruption/preemption tests
are now covered. Remaining gates are recorded in the report.

The production source-conflict/admission-failure matrix, two repeatable component runs
and expanded idle C1/C4 serving are now executed and recorded in the report. Completed-
window component timing is neutral/slower, not a win. Expanded serving preserves90
native signatures but not competitor signatures; C1 reuse TTFT also loses to competitors.
Quality-equivalent competitive acceptance remains the active gate.
