# NFC component benchmark

Scope: close the performance-validation loop of [native NFC](normalization.md),
not tokenization or inference. Primary-source algorithm/oracle research is in
[normalization research](../research/normalization.md). The timer/compiler-barrier
mechanism is the already-inspected Zig 0.16.0 `std.Io.Clock.awake` and
`std.mem.doNotOptimizeAway`, as in [the quant harness](quant-benchmark.md).

## Correctness and workload

Use the three fixed `tests/fixtures/nfc9/manifest.json` workloads: ASCII,
multilingual decomposed accents/Hangul, and a long nonstarter run with repeatedly
reversed combining classes. Before timing, require native and pinned HF Tokenizers
0.22.2 NFC results to equal each recorded output byte-for-byte. Require fixture,
generator and table provenance; rerun all native tests in Debug/ReleaseFast.

For each workload and engine: three untimed warmup calls, seven timed trials.
Iterations per trial: ASCII 1,000; multilingual 100; marks 10. Three independent
process rounds alternate native/HF order; every round covers all three cases.
Single explicit permitted CPU affinity, warm buffers, unchanged clock/power policy.
Report per-call nanoseconds, median/min/max/sample standard deviation and 21 trials
per workload/engine. Keep all raw observations. Do not time setup, hashing, JSON or
logging. Consume native output through a compiler memory barrier inside timing;
recheck complete results after each trial outside timing.

Native preallocates output and exact scratch and calls the public validated API
(no heap work inside timing). The HF API receives a prepared Python string and
returns an allocated Python string; Python/Rust boundary, internal allocations and
string ownership costs remain timed. Native starts with UTF-8 bytes and validates
on each call. Disclose this ownership/encoding asymmetry rather than claiming
identical API overhead. Record input/output byte lengths, hashes, scratch bytes
and native output capacity. Validation and instrumentation are not serving work.

## Reproducibility and errors

The Python harness requires a new output directory, checks CPU selection and exact
reference version, builds/tests ReleaseFast, and rejects missing/duplicate/extra
trials, schema/count/hash mismatches or nonpositive timings. Record commands and
stdout/stderr, host/CPU, compiler version/hash, native executable hash, Python/package
versions and oracle extension hash. Snapshot all native/test/harness source and
embedded data/license files, including `.bin` and `.txt`; hash every snapshot.
Ordinary builds/tests have no HF, network or `third_party/` dependency. Unit-test
the result validator independently of the oracle environment.

## llama-server comparison boundary

The installed llama-server's Qwen35 tokenizer does **not** apply NFC: actual
`/tokenize` probes gave `[68,52033]` for decomposed `e + U+0301`, whereas official
HF returns `[933]`. Both return `[933]` for composed `é`. See
[recorded probes](../research/2026-09-22/tokenizer-probes.json) and
[reference identity](../bench/2026-09-22-reference-bringup.md).

There is no standalone equivalent NFC operation in that server to time. Comparing
NFC-only time to `/tokenize` request time would omit BPE and HTTP on one side.
Therefore use HF for this isolated block and explicitly leave the **actual
llama-server comparison open until native tokenization exists**. At that milestone,
check exact IDs on semantically equivalent NFC inputs before repeated `/tokenize`
timings; retain unnormalized-input discrepancies as compatibility results, never
hide them to manufacture a win. Full model/serving comparisons remain separate gates.
