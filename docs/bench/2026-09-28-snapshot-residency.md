# Snapshot residency bookkeeping — 2026-09-28

## Completed scope, not an integrated NVMe cache

First increment of 18d.6c: `src/session/residency.zig`. An immutable snapshot's
identity is independent of resident and disk slot indices. Bounded metadata tracks
save, spill, restore, read leases, retained disk backing, generations and transfer
serials. A delayed completion cannot release a newer transfer's slots. Failed
writes retain the resident source; failed reads never publish the partial target.
Both source and destination stay owned until completion; cancellation drains first.

The existing production `checkpoint.Store`/`kvcache` path **has not been switched
to this table**. This is a tested building block plus a physical disk component
test, not disk-backed serving, persistence across restarts, or a retention policy.
No filesystem-specific code, server flags or GPU kernels were added.

[Specification](../specs/snapshot-residency.md),
[research and primary/reference provenance](../research/2026-09-28-snapshot-residency.md).

## Correctness evidence

Before native implementation, executed:

```sh
tools/py tests/reference/generate_residency_fixture.py
```

Independent Python 3.14.4 dictionary/set ownership model (not native self-comparison):
**4,860 full-state transitions**, three capacities/seeds, directed and randomized
valid/error operations. Native tests compare each return/error, every entry, and
both ownership maps exactly. Seeds, generator and payload hashes live in the
[fixture manifest](../../tests/fixtures/residency/manifest.json); source-backed
SGLang observations are explicitly not claimed as equivalent protocol semantics.

Physical component test: twelve 4096-byte snapshots retained through only **two
resident slots**, using the real storage worker. Data is from the existing
independent Python POSIX fixture. Writes are checked by synchronous `pread` before
publication, completions consumed in reverse order, freed resident bytes poisoned,
and records restored in reverse order and compared byte-for-byte. Includes an
actual drained/canceled write followed by retry with stale completion rejected,
short read after deliberate idle-file truncation, and negative BADF write completion
after deliberate idle-fd close. The failed write's source remains byte-exact and
resident. These are **small opaque snapshots**, not Qwen state tensors or logits.

Executed checks:

```sh
bazel test //...
bazel test --runs_per_test=20 //tests:residency //tests:residency_release_fast
```

- 79/79 CPU/Python/format tests passed (26 executed, others cached on the explicit
  final run). Both residency modes passed all 20 repetitions each, four cases per
  mode including the SHA helper. Coverage includes init-allocation failure cleanup,
  allocation-free operations, resource limits, generation/serial exhaustion and
  lease overflow. No assertion-only checks disappear in ReleaseFast.
- `bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills`: 3/3
  passed (Debug executed; ReleaseFast and spill gate cached).
  `tools/py tools/zerv_build.py --test-host-gpu`: 2/2 passed (Debug executed,
  ReleaseFast cached). Logs are retained alongside the component data.
- Staged source/docs whitespace check passed, excluding raw `docs/bench/data/**/*.log`
  and the raw P2P `host.txt` capture. The unrestricted staged check flags tool-emitted
  trailing spaces in those files; they are retained verbatim rather than editing evidence.
- Initial `bazel test //...` failed compilation of the new **benchmark**, comparing
  `?u32` disk IDs to a `usize` loop index. Added explicit checked u32 casts; no engine
  semantics changed. The failed build and final test logs are retained with the data.

## Repeatable component measurement

Observed Ryzen 3900X, Linux 7.2.6-arch2-1, Zig 0.16.0, Bazel 9.2.0, native-CPU
ReleaseFast. Sources/build/binary hashes and CPU description are recorded in each
manifest. Working tree based on `51a6053`; use source hashes, not that base alone.

```sh
tools/py bench/run_residency.py --output docs/bench/data/2026-09-28-snapshot-residency
tools/py bench/run_residency.py --output docs/bench/data/2026-09-28-snapshot-residency-repeat
```

Choose new directories for repetition. Harness gates on all CPU/Python/format tests
and builds through `tools/zerv_build.py`. Each run: 64/256/1024 entries, 1/8 resident
slots, disk slots equal to entries; one warmup plus five measured trials, alternating
capacity/depth order. Each trial performs 64 complete cycles over every record:
reserve/save/spill/complete; restore/complete/lease/release/backed demotion; drop.
The harness checks every record's identity/location and all ownership release.

**No actual I/O/GPU work is timed here.** Transfer completion is immediate simulated
success. Allocation and state validation are outside timing. These short, hot,
inlinable loops characterize metadata cost, not scheduler latency, disk latency,
or complete cache lookup. No equivalent external implementation of this bounded
protocol exists; do not compare these times with SGLang transfers or llama-server.

Nanoseconds per record, mean ± sample SD, n=5 per run:

| Entries | Resident slots | Save/spill run 1 | Save/spill run 2 | Restore/lease/demote run 1 | Restore/lease/demote run 2 |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 44.5 ± 9.3 | 36.5 ± 1.5 | 11.9 ± 2.2 | 9.8 ± 0.1 |
| 64 | 8 | 46.7 ± 6.9 | 39.6 ± 2.7 | 12.3 ± 2.3 | 9.8 ± 0.5 |
| 256 | 1 | 148.7 ± 19.2 | 119.0 ± 9.5 | 13.2 ± 1.9 | 10.3 ± 0.6 |
| 256 | 8 | 153.5 ± 16.1 | 122.8 ± 8.8 | 12.6 ± 1.7 | 9.9 ± 1.0 |
| 1024 | 1 | 476.3 ± 46.2 | 434.4 ± 71.2 | 12.3 ± 1.4 | 11.2 ± 1.8 |
| 1024 | 8 | 491.6 ± 60.0 | 443.8 ± 74.6 | 12.4 ± 1.6 | 11.3 ± 1.9 |

Drop means are 2.8–3.6 ns/record; all trial values and variances are in the summaries.
Metadata spans are 3,080–3,136 bytes at 64 entries and 49,160–49,216 at 1024,
excluding the small table object and allocator overhead. Reservation scans grow
with capacity as expected; completion and retained-backing demotion do not scan.
Keep both runs and their variance; no end-to-end speedup or worst-case bound is
inferred from the lower second-run means.

- Run 1: [manifest](data/2026-09-28-snapshot-residency/manifest.json),
  [raw](data/2026-09-28-snapshot-residency/raw.jsonl),
  [summary](data/2026-09-28-snapshot-residency/summary.json).
- Run 2: [manifest](data/2026-09-28-snapshot-residency-repeat/manifest.json),
  [raw](data/2026-09-28-snapshot-residency-repeat/raw.jsonl),
  [summary](data/2026-09-28-snapshot-residency-repeat/summary.json).

## Next gates

18d.6c remains active. Resolve bounded disk record layout/integrity and chunked
snapshot transfer coordination, then use this table in checkpoint residency. KV
extents/ancestor ownership and pending scheduler operations remain separate staged
integration work. Current radix pressure scratch and per-entry full-context token
storage must not be assumed to scale just because this metadata table can. Real
model state/logits, cancellation under serving load, and tuned llama-server serving
comparisons remain mandatory before exposing a production disk tier.
