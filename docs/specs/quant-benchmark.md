# Quant decoding component benchmark

Scope: a repeatable **CPU diagnostic component** benchmark for the validated Q4_0
and Q8_0 decoders, not a GPU or model-serving benchmark. The scalar product's
research/oracle gate is already satisfied in [quantization.md](quantization.md).

## Researched mechanism / contract before implementation

Zig 0.16.0 `std.Io.Clock.awake` maps to Linux `CLOCK_MONOTONIC` and exposes integer
nanoseconds. Python `time.perf_counter_ns()` supplies a monotonic elapsed timer.
Only synchronous CPU decode calls occur inside timing; neither timing path needs
GPU synchronization. Use `std.mem.doNotOptimizeAway(output)` after each native
call: its pointer path is a volatile compiler barrier with memory clobber. Consume
input/output hashes outside timing so repeated work is not optimized away. Relevant
pinned stdlib paths: `lib/std/Io.zig`, `lib/std/mem.zig`, `lib/init/src/main.zig` in
[the pinned compiler](../development.md). Its process-init/stdout APIs were inspected.

The external oracle uses the same researched exported decoder ABI as the fixture
generator. Buffers/counts are aligned/validated by the harness before timing;
ctypes call overhead remains inside reference timing and is disclosed. Native
includes whole-row scale validation; external C asserts length but does not reject
non-finite scales. Inputs here are finite; report that asymmetry rather than hiding
it. No C++/oracle dependency may enter the native benchmark binary.

## Fixed workload v1

- Formats: Q4_0 and Q8_0; **5120 × 2048 = 10,485,760 decoded values** per call.
- 327,680 contiguous blocks. Binary16 scale bits `0x3000 + (block % 0x1000)`.
  Payload byte j of block b is `(b*37 + j*19) mod 256`.
- 40 MiB output, 5.625 MiB Q4 input or 10.625 MiB Q8 input. This is a synthetic
  working set with a model-relevant hidden dimension, not checkpoint tensors.
- Preallocate and touch buffers; 3 full warmup calls; 5 timed trials per format,
  16 full decode calls per trial. No allocation, formatting, hash computation, or
  input generation within timing. Reuse the same buffers (warm working set).
- Single CPU, explicit affinity chosen by runner; untouched clock/power policy.
  Three comparison rounds alternate native-first/reference-first order. Each
  native process runs both formats; disclose this coarser ordering.
- ReleaseFast with native CPU target. Record compiler flags and binary hashes.

## Harness outputs and acceptance

Native benchmark emits JSONL with format/trial/iterations/value count/elapsed ns
and full input/output SHA-256 values. The external runner independently constructs
the same inputs and invokes the oracle, records the same fields, and requires
hash equality for every native result before accepting any timing. Missing/extra
trials, invalid times/counts, bad hashes, or process failure abort the benchmark.

The implemented runner accepts explicit Zig/library paths, CPU index, and a fresh
output directory. It reruns Debug/ReleaseFast correctness tests and rebuilds the
native benchmark from current sources with `ReleaseFast -Dcpu=native` before timing,
so an arbitrary stale binary cannot silently stand in for current code. It records
timestamps, OS/CPU/affinity, source/binary/toolchain hashes,
oracle identity (including dirty status), benchmark config, raw JSONL and native
stderr. It refuses overwrite. Store data under `docs/bench/data/<run-id>/` and write
a dated Markdown report under `docs/bench/` with median/min/max and variation, exact
commands, correctness gate, limitations, and next action.

This initial comparison is with the installed ggml row decoder, **not** a claim to
have beaten the best CPU decoder. Always retain `llama-server` plus other qualifying
servers as mandatory references for future **serving** benchmarks. Component
results cannot substitute for that end-to-end requirement.
