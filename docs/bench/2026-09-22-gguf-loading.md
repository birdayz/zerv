# 2026-09-22 — native GGUF parse/index comparison

Question: how expensive is zero-copy validated container/index construction for
the actual Qwen3.8 artifact versus the installed independent GGUF reader?

Correctness gate: full model SHA verified; all metadata/descriptor fields and
payload samples agree exactly. All small independent fixtures and native Debug /
ReleaseFast tests passed before timings. See [validation](../research/2026-09-22-gguf-validation.md).

## Reproduce

From the project root (use a fresh output directory on repeat):

```sh
python3 bench/run_gguf.py --library /usr/lib/libggml-base.so.0.24.0 \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --output docs/bench/data/2026-09-22-gguf-qwen38 --cpu 2
python3 bench/run_gguf.py --library /usr/lib/libggml-base.so.0.24.0 \
  --model tests/fixtures/gguf/default.gguf \
  --output docs/bench/data/2026-09-22-gguf-fixture --cpu 2
```

[Specification](../specs/gguf-benchmark.md). Ryzen 9 3900X, Linux 7.2.6, Zig 0.16.0
ReleaseFast native target, logical CPU 2. Three alternating-order rounds, 3 warmups
per worker, 7 trials × 10 parse/free cycles per round (21 observations per engine).
Warm mapped header/page cache; complete file hashing and output validation excluded
from timing. No GPU work. No clock/power changes or exclusive machine reservation.
Full compiler/model/reference/binary/source hashes, host, commands and raw data:
[Qwen run](data/2026-09-22-gguf-qwen38/manifest.json),
[small-fixture run](data/2026-09-22-gguf-fixture/manifest.json).

## Observed per-parse time

| Workload | Native median (min–max) | Reference median (min–max) | Native/reference |
| --- | ---: | ---: | ---: |
| Complete Qwen artifact, 51 KV / 866 tensors | 12.899 ms (12.387–14.523) | 27.521 ms (26.673–31.860) | 0.469 |
| 2,592-byte fixture, 25 KV / 8 tensors | 23.352 µs (21.002–31.169) | 8.613 µs (8.429–14.053) | 2.711 |

Sample standard deviations: Qwen native 0.478 ms, reference 1.682 ms; fixture
native 2.464 µs, reference 1.944 µs. Values come from checked-in summary/raw trials.

**Interpretation:** native is about 2.13× faster on this startup component's real
artifact, but 2.71× slower for a tiny file. This is not an equivalent ownership
comparison: native borrows strings/arrays and validates UTF-8 and complete payload
bounds, while reference copies metadata. Native page_allocator mapping overhead
is significant on tiny files. Reference ctypes calls/count getters are included;
reference file mapping is private copy-on-write (no writes), native read-only.
Both include parser allocations and destruction. Neither decodes weights or uploads
them. No full-loading, GPU, token-generation, or serving speedup is claimed.

Later source changes adding chat exports and stricter oversized index rejection
do not retroactively alter these recorded source hashes. Re-run for new performance
claims. Keep wins and losses visible instead of tuning only the tiny synthetic case.

## Repeat with self-contained source snapshot

Repeated the same Qwen command with output directory
`docs/bench/data/2026-09-22-gguf-qwen38-repeat`. The harness now saves all native,
build, test, oracle, fixture and benchmark source files under `source/`, not just
hashes, so an uncommitted source revision can be rebuilt independently. Raw first
runs remain intact. [Repeat manifest](data/2026-09-22-gguf-qwen38-repeat/manifest.json).

Repeat medians: native **12.370 ms** (12.278–13.294, sample stddev 0.448 ms), reference
**26.959 ms** (26.232–28.781, stddev 0.651 ms), 21 trials each; ratio **0.459**.
Same exact independent inventory/output gate passed. The real-artifact result
reproduced; tiny-file loss and ownership caveats above still apply.
