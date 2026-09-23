# 2026-09-22 — Q4/Q8 CPU decode: correctness, baseline loss, SIMD, repeat

## Question and scope

Can our first native quant decoder reproduce the external reference exactly, and
how does explicit lane vectorization change its CPU component performance?

**This is not Qwen inference, a GPU benchmark, or a serving speed claim.** The
installed external row decoder is a useful component baseline, not proven the best
available implementation. We still do not serve a model. Future serving benchmarks
must always include tuned `llama-server` and other qualifying high-performance servers.

## Correctness gate

- Reference-golden generator invoked the installed ggml decoder and independently
  checked every result with Python binary16/scalar arithmetic before native code.
- 63,488 finite binary16 scale encodings, every Q4/Q8 coefficient: **18,284,544
  decoded values** match native FP32 bits, including signed zero/subnormals.
- Explicit multi-block/unaligned fixtures and all 2,048 non-finite encodings per
  format pass; invalid later blocks leave the full output unchanged.
- Each benchmark run reran native Debug and ReleaseFast suites (5/5 tests each).
  Harness validation unit tests (4/4) cover malformed/duplicate/missing results,
  wrong counts, bad hashes, invalid times, and result-schema errors.
- Every measured trial's complete output hash and generated-input hash match the
  separately executed external decoder. Timing results fail closed on mismatch.

## Setup

- Ryzen 9 3900X, CPU affinity **0**, one thread; no GPU work. Arch Linux/kernel and
  CPU details, governor and source/toolchain identities are captured in each manifest.
- Zig 0.16.0; native benchmark `ReleaseFast -Dcpu=native`. Binaries are static ELF;
  no dynamic `NEEDED` dependencies, no C/C++ imports or oracle links in native code.
- External `/usr/lib/libggml-base.so.0.24.0`, version 0.24.0, reported
  **`456172ec-dirty`**, exact binary hash in each manifest. Do not relabel it clean.
- Fixed synthetic workload v1: 5120 × 2048 FP32 outputs (40 MiB), 5.625 MiB Q4 or
  10.625 MiB Q8 input. Scale/payload generation is specified in
  [the benchmark contract](../specs/quant-benchmark.md).
- Buffers allocated/initialized before timing; 3 warmups; 5 trials × 16 decode
  calls per format/process; 3 comparison rounds alternating native-first and
  reference-first. **15 trial averages per implementation/format/run**.
- Same reused/warm buffers; no cache flushing. Monotonic CPU elapsed time; compiler
  memory barriers and output hashing prevent dead-code elimination. Hashing,
  allocation, input preparation, process launch and log serialization are not timed.
- The native API performs whole-row non-finite validation on every call. The
  external C decoder does not provide that rejection/failure-atomicity contract;
  ctypes invocation overhead is included in its timing. Report this asymmetry.

## Results

Milliseconds per full decode call; lower is better. Overhead is the native/reference
median ratio minus one **within the same run**. Standard deviation is across trial
averages, not individual calls. Raw min/max values are in each summary.

| Run | Format | Native median ms | Reference median ms | Native overhead | Native / reference stddev ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Scalar start | q4_0 | 6.293 | 4.762 | +32.1% | 1.115 / 1.267 |
| Scalar start | q8_0 | 6.277 | 4.606 | +36.3% | 1.252 / 1.029 |
| Explicit SIMD | q4_0 | 3.691 | 3.504 | +5.3% | 0.148 / 0.122 |
| Explicit SIMD | q8_0 | 3.871 | 3.631 | +6.6% | 0.166 / 0.173 |
| SIMD repeat | q4_0 | 3.771 | 3.640 | +3.6% | 0.150 / 0.186 |
| SIMD repeat | q8_0 | 4.101 | 3.661 | +12.0% | 0.155 / 0.205 |

## Interpretation and limitations

The initial scalar-expression decoder lost clearly. Explicit array-value SIMD
loads/conversions/stores preserved all golden bits and narrowed the same-run gap,
but **we still do not beat the component reference**. Repeat results retain a Q8
gap and show why repeated measurements matter. No quality/validation checks were
removed to manufacture a win.

The reference itself became faster in later runs, so it would be misleading to
attribute the entire absolute before/after difference to our source change.
This is a shared workstation: affinity is controlled, but background load, core
frequency and temperature are not isolated/locked. The manifests record the
observed governor; no clock/power policy was changed. Trial averages are correlated
within a process. Small gaps need a stronger controlled experiment before a claim.

Inspected disassembly contains vector widening, `vcvtph2ps`, `vcvtdq2ps`, and
`vmulps`: explicit vectors do lower to SIMD on this target. See
[data/2026-09-22-quant-simd/disassembly.txt](data/2026-09-22-quant-simd/disassembly.txt).
We have not profiled validation-versus-conversion bandwidth costs separately, and
we should not prematurely optimize a diagnostic CPU path over the required native
GPU/model bring-up. Next: component profiles to separate those costs, artifact
inventory/research, and independently validated GPU primitives.

## Exact reproduction

From project root, choose a fresh output directory for each run:

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 bench/run_quant.py \
  --zig .tools/zig-x86_64-linux-0.16.0/zig \
  --reference /usr/lib/libggml-base.so.0.24.0 \
  --cpu 0 \
  --output docs/bench/data/YYYY-MM-DD-quant-unique-run
```

The harness reruns correctness, rebuilds the current native source, validates the
reference comparison, and records raw data/config/hashes. Repeating requires a new
output path, not an overwrite. To reproduce the **initial scalar source** rather
than current SIMD, use the retained
[data/2026-09-22-quant-cpu/source/quant.zig](data/2026-09-22-quant-cpu/source/quant.zig)
in an isolated working copy; its hash is recorded in the baseline manifest. Do not
replace current working code or compare binaries without matching source hashes.

## Evidence

- **Scalar start:** [manifest](data/2026-09-22-quant-cpu/manifest.json), [raw trials](data/2026-09-22-quant-cpu/trials.jsonl), [summary](data/2026-09-22-quant-cpu/summary.json); build/test/native stderr and raw reference output alongside.
- **Explicit SIMD:** [manifest](data/2026-09-22-quant-simd/manifest.json), [raw trials](data/2026-09-22-quant-simd/trials.jsonl), [summary](data/2026-09-22-quant-simd/summary.json); build/test/native stderr and raw reference output alongside.
- **SIMD repeat:** [manifest](data/2026-09-22-quant-simd-repeat/manifest.json), [raw trials](data/2026-09-22-quant-simd-repeat/trials.jsonl), [summary](data/2026-09-22-quant-simd-repeat/summary.json); build/test/native stderr and raw reference output alongside.

All three manifests report `passed`; no model/serving benchmark was run.
