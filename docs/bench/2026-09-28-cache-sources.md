# Radix source ownership — 2026-09-28

18d.7c.1, prerequisite of the [full tiering plan](../design/async-tiering.md).
[Research](../research/2026-09-28-cache-sources.md) and
[contract](../specs/cache-source-leases.md) preceded native code. **Not yet a
pressure-driven serving policy or a performance-goal completion.**

## Implemented

Cache source handles are generation-qualified, physical snapshot IDs explicit, and
acquisitions serial-qualified. A source protects its complete ancestor path,
including mixed host/GPU segments. Eviction, snapshot reuse, promotion/demotion
and ancestor insertion cannot invalidate the source. Other branches can progress;
GPU-only read hits remain eligible. Failed/duplicate/stale releases change nothing.
Counters and serials do not wrap; flat policy rejects capture explicitly.

This protects metadata/residency, not unused partial-page bytes. Production still
uses checkpoint-triggered writes until the separate byte-adapter and policy gates.
No new GPU or I/O code, process, dependency, filesystem change or download.

## Verification actually executed

[Raw evidence](data/2026-09-28-cache-sources/):

- Independent Python prefix-set oracle generated **2,345 transitions**, then native
  Debug/ReleaseFast compared every outcome, live prefix, source generation/serial,
  active flag, derived ancestor protection count, candidate and tree invariants.
  The oracle derives ancestors from token-prefix inclusion, not native parent links.
- Directed mixed-residency case: held host ancestor/GPU leaf, blocked promotion,
  blocked ancestor insertion, stable borrowed pages/tokens/snapshot, unrelated
  branch insertion/demotion/eviction, successful exact restore after release.
- Allocation-failure rollback, allocation-free steady operations, serial/generation/
  counter exhaustion, invalid/duplicate release and unsupported flat policy.
- Both test modes repeated **20 times** after restoring the negative control; all
  pass. Full CPU/Python/format **81/81**, first full run executed 17 targets.
- Negative control: increment only the leaf's protection, not its ancestors. Both
  modes fail the fixture; retained in `negative-no-ancestors.log`. Restored before
  any measurements. No negative-control GPU execution.
- Initial compile failure: Zig does not widen `?u16` to `?usize`; changed the four
  new parent-walk conversions to explicit optional handling. First failure retained
  in `unit-first.log`; corrected run and repetitions pass.

## Metadata component measurement

Two fresh runs, each one warmup and five trials per depth; 100,000 acquire/release
pairs per trial. Fixed chain setup outside timing; every source counter/generation/
serial and checksum checked outside timing. Source/build/binary hashes, CPU and
commands in each run's manifest. No CPU affinity or isolation; retain variance.
This does not measure tensor copies, eviction cost, or HTTP latency. No equivalent
llama-server source-lease API exists; no competitor speed claim.

| Ancestor path depth | Run 1 ns/pair, mean ± SD | Run 2 ns/pair, mean ± SD | Ownership metadata bytes |
|---|---:|---:|---:|
| 1 | 5.154 ± .055 | 4.769 ± .038 | 24 |
| 8 | 30.970 ± .303 | 31.953 ± .499 | 192 |
| 64 | 295.636 ± 2.133 | 305.892 ± 6.665 | 1,536 |
| 256 | 1,352.297 ± 107.784 | 1,223.707 ± 82.684 | 6,144 |

The deepest path costs ~1.2–1.4 µs/pair here, not a proof that full serving is fast.
The user's faster-or-on-par goal remains open; the prior archive regression is not
closed by this microbenchmark.

```sh
tools/py tests/reference/generate_cache_source_fixture.py
bazel test //tests:kvcache //tests:kvcache_release_fast --runs_per_test=20
bazel test //...
tools/py bench/run_cache_sources.py --output docs/bench/data/NEW_SOURCE_RUN
```

Next: canonical partial-tail byte fixture, mixed-source GPU/host adapter and actual
pressure-policy integration, followed by model/HTTP/competitor gates. Proactive
preparation and demand prefetch remain after that, not silently marked complete.
