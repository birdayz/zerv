# Immutable prefix archive component — 2026-09-28

Observed: `session.archive` implements bounded immutable token-prefix records,
chunk extent ownership, SHA256, asynchronous capture/restore, reader leases, LRU,
cancellation/drain and invalidation. Production serving was not integrated in this
component increment; the later [integration report](2026-09-28-disk-prefix-serving.md)
records the production owner, scheduler and real-model/HTTP gates.
Spec: [disk prefix cache](../specs/disk-prefix-cache.md). Independent oracle:
`tests/reference/generate_archive_fixture.py` uses POSIX I/O and hashlib over the
existing positional transport fixture; provenance is in `tests/fixtures/archive/oracle.json`.

Commands:
```
tools/py tests/reference/generate_archive_fixture.py
bazel test //tests:archive //tests:archive_release_fast
tools/py bench/run_archive.py --scratch-dir third_party/nvme-probe --direct-alignment 4096 --output docs/bench/data/2026-09-28-prefix-archive-retry
```

Both native modes pass. The benchmark gates on `bazel test //...`: 81/81 pass
(80 cached, one executed in the retry). The first harness invocation was interrupted
by the outer 120-second tool timeout during `kv_system`; its partial log is retained
in [initial run](data/2026-09-28-prefix-archive/tests.log). No measurements or manifest
were produced by that interrupted wrapper. The retry used a 600-second outer limit.
This is not a diagnosis or reopening of the retired hang investigation.

[Complete retry evidence](data/2026-09-28-prefix-archive-retry/): manifests,
binary/source hashes, commands, logs and raw five trials. One warmup; four 64 MiB
records per trial, 1 MiB chunks, depth eight, real preallocated 256 MiB direct-I/O
scratch on the existing operator-prepared directory. No filesystem attributes changed.
CPU copies, hash and archive bookkeeping included; fsync outside write timing;
read-target poisoning and full exact byte comparison outside read timing.

| Direction | Mean GB/s | Sample SD | Min–max |
|---|---:|---:|---:|
| Write | 1.1312 | 0.0116 | 1.1158–1.1464 |
| Read | 1.5308 | 0.0115 | 1.5234–1.5508 |

The complete archive path is materially slower than the earlier raw transport
(~6–7 GB/s); hash/copy/catalog work and the poll cadence are included here. No
attribution from a profiler yet. No GPU, overlap, 80k-model or serving speed claim.
llama-server has no equivalent opaque immutable archive API; serving comparisons
remain required at integration. Independent bytes/integrity checks—not another
server—are this component's correctness oracle.
