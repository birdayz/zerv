# Tests in parallel, and the bugs the load exposed — 2026-09-26

**Question.** The unit tests ran as one binary, one test at a time (~96 s on a 24-thread
machine; Zig 0.16's runner is serial within a binary,
[ziglang/zig#15953](https://github.com/ziglang/zig/issues/15953), open). How fast can the
required checks be, without weakening any test?

**Answer.**

| | before | after |
|---|---:|---:|
| `zig build test`, cold (fresh local cache) | ~96 s (compile 37 s + run 59 s) | **9.4 s** |
| `zig build test`, warm (no change) | ~60 s (run) | **~4.1 s** |
| `zig build test` after a `src/` edit | ~96 s | **~4–7 s** |
| all four required checks (fmt, Debug, ReleaseFast, Python) | serial, several minutes | **`zig build check`: 22.6 s cold, ~19–22 s after a `src/` edit** |

Same tests and goldens (the random cases of the parallelized differential tests are drawn
per round now; the properties checked are unchanged). The warm run uses 63.7 s of CPU in
4.0 s of wall time: it is at the machine's throughput, so further gains need less work,
not more parallelism.

Setup: AMD Ryzen 9 3900X (12 cores, 24 threads), Zig 0.16.0 (`.tools/`), times from
`zig build … --summary all` and bash `time`; other processes were running (a desktop
recorder, bruh sessions, a secret scanner), which moved single runs by up to ~2.5 s.
Commands and design: [development.md, "Tests in parallel"](../development.md).

## What changed, measured

1. **One binary per test file** (`build.zig` `unit_tests`; `tests/root.zig` became
   `tests/quant.zig`). Cold: 96 s → 34 s wall (per-file compile and run measured with
   `zig test --test-no-exec` in parallel).
2. **Comptime SHA-256 removed** from `src/text/nfc.zig` and `src/tokenizer/classes.zig`
   (`table_sha256` → `tableSha256()`, run time; only tests use it). The NFC test compiled in
   25 s, `tokenizer_split` in 7 s; both ~1 s now.
3. **Concurrent rounds** (`tests/parallel.zig`) in the sampler differential tests (session
   binary 32 s → 4.2 s, then split into `session`, `sampler`, `sampler_nucleus`: 0.1 / 2.0 /
   2.1 s), the prefix-cache randomized tests (configurations), the tokenizer encode cases.
4. **Golden SHA-256 in a ReleaseFast object.** Measured SHA-256 throughput: Debug 134 MB/s,
   ReleaseFast 1,752 MB/s (64 MiB, SHA-NI CPU). With `parallel.hashChunks` for the decode:
   NFC fingerprints 4.28 → 0.42 s, Q5_K 3.84 → 0.42 s, Q6_K 3.05 → 0.39 s; digests unchanged
   (the goldens still pass). A new test checks the object's digests against std's.
5. **No `std.http.Client` in the server test.** It compiles the TLS stack: the one test cost
   47.9 s of ReleaseFast compile (every other `serve` test 15–20 s, measured one test per
   compile with `--test-filter`). A 60-line keep-alive HTTP/1.1 test client replaces it and
   also asserts that all requests used one connection. `serve` ReleaseFast 52 s → 12 s.
6. **Stripped optimized test binaries.** Empty ReleaseFast test: 14.4 s with debug info,
   6.4 s stripped. `-Dtest-symbols` keeps them.

7. **Cached test results.** `enableTestRunnerMode` passes the build runner's random seed
   (new every invocation) to each test binary, so no test run was ever a cache hit. A fixed
   seed (`-Dtest-seed`, default 0): no-change `zig build test` 4.1 s → **0.1 s**, `zig build
   check` → 3.8 s (the Python tests always run). After a `src/` edit everything still rebuilds
   and reruns (~8 s): `zerv` is one module.

8. **Package modules** (user request: "split src/ into … modules", then judge zig build
   alone against Bazel). `src/` is ten modules with declared dependencies plus the `zerv`
   umbrella; each unit test imports only its packages. Rebuilt after one edit (each run
   also reverted the previous probe, so it includes that group): `session` edit → 7 test
   binaries (session, sampler, sampler_nucleus, prefix, serve, batcher, tools), 6.0 s;
   `quant` edit → the 4 quant binaries (+ the reverted session group), 6.5 s; `gpu` edit →
   gpu_abi, matvec, model, serve, batcher, tools (+ quant group), 4.0 s. Before: all 19.

Negative: `zig build test -fincremental` (0.16) produced a `test-gguf` binary that aborted
("unwind info unavailable"), and was no faster (6.1 s against 7.3 s). A custom concurrent
test runner cannot help under `zig build`: the build runner drives the test binary one test
index at a time over its `--listen` protocol.

## Bugs found by running under load (DFS)

Running the batcher tests under full-machine load (24 copies at once, 240–480 runs per
variant) turned up four timing assumptions and two defects:

| | where | kind | evidence |
|---|---|---|---|
| A canceled prompt with one-unit chunks ran **all** its remaining chunks (the cancel check ran only for a chunk in flight; between chunks the operation was re-queued). Violates concurrent.md ("waits for the running unit"); wastes GPU time on a dead request. | `src/serve/batcher.zig` `endPack` (committed, 18c) | **code bug**, fixed | waited 49 ms in 20/20 runs with `segments=1` (4 ms with 4); now a load-independent assertion: at most 3 chunks ran (was 11); the reverted fix fails it deterministically |
| The test's bound (50 ms) sat 1 ms above the bug's latency | `tests/batcher.zig` | test | as above |
| "pending prompts pack into one chunk" **hung** when the short prompt was queued first (it ran alone, then the long one alone; the wait for a two-member chunk never ended) | `tests/batcher.zig` (committed) | test race, **hang** | `HEAD`: 5/240 hangs under load; now a third slot's blocked reset holds the scheduler until both prompts are queued: 0/480 |
| `stop` returned before pending memory releases (a slot left just before `stop` kept its pages) | `src/serve/batcher.zig` `run` | code, in the **uncommitted 18d.2 shared-pool work of another session** | pool test "every page came back" failed under load; releases now run before stopping |
| The shared-pool test's contention depended on thread timing (`admission_waits = 0` in 9/20 runs) | `tests/batcher.zig`, same uncommitted work | test | the scheduler now starts after six resets are queued (6 × 3 + 7 pages > 10) |

Stress result with all fixes: 480/480 runs of the batcher binary pass under full load
(`HEAD`: 230/240 pass, 5 hangs, 5 failures). Harness: `/tmp` scripts running 24 copies of
the test binary with `xargs -P 24` and a timeout; instrumentation (progress marks, a
watchdog task) located the hang; the watchdog variant perturbed timing enough to hide it
(0/480), the marks-only variant did not (6/480, all after "resets done").

## Limitations

- Timings are single runs on a shared desktop; repeated warm runs varied 4.1–4.3 s (one
  6.7 s outlier with a background scanner at 100% CPU).
- The GPU tests (`zig build gpu-test`) are unchanged: one serial binary (shared device).
- `zig build check`'s floor is the ReleaseFast LLVM compile of the larger test binaries
  (~12 s); any `src/` change recompiles all of them.
