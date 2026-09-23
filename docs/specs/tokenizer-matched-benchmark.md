# Matched direct tokenizer measurement (03b)

Written before adapter/harness implementation, 2026-09-22. Scope is tokenizer
measurement then evidence-driven optimization, not resuming native serving.
The previous direct-native/HTTP table is not comparable speed evidence.

## Research and shared boundary

Pinned llama.cpp b29c606e28a01b1bc8c1351026a0fa6e616bf6c4, retained
`third_party/llama.cpp/<revision>/include/llama.h` and `src/llama-vocab.cpp`.
Vocab source SHA256 b9588d7116c11573b378c43bf3c85f87249ad5eb9626324abded4abc7c91e6dc.
Header SHA256 fedb52ea9291c9900e637ed6ffec339919dd27c3dadc2c8cf216c552e0488dda;
installed header identical. Origin: https://github.com/ggml-org/llama.cpp/tree/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4.
Installed libllama SHA256 c352cb4b1f5456dffbc4483ba1e0be7a547b21a0f7e63462ab8fb333f51245e1
is exactly the library recorded for actual llama-server. Check loader dependencies
and identities on every run; no recompiling a weaker reference.

`llama_tokenize` takes explicit-length UTF-8 and fills caller storage. Internally
it creates a std::string/vector and copies IDs to that buffer. No HF offset work.
`llama_detokenize` copies raw token pieces; flags remove_special=false,
unparse_special=true. Qwen has no add_space_prefix/clean_spaces. Independently
verify raw sequence equality, including incomplete UTF-8, padding and additions.
Both APIs return negative required capacity on overflow; failures abort timing.

- Encode: same UTF-8 bytes in, owned ID vector out, consume then free per request.
  Native public encode allocates output and scratch. Test-only C wrapper allocates
  a byte-length upper-bound ID buffer, calls llama_tokenize(add_special=false,
  parse_special=true), consumes IDs and frees the allocation. Allocator choice
  differs (native smp_allocator vs libc/C++ allocator), but both include complete
  request allocation and destruction. Neither borrows precomputed expected IDs.
  Output capacities can differ; no reference capacity-probe/double tokenization.
- Decode: identical ID array to preallocated raw-byte buffer, no UTF-8 replacement,
  all additions rendered, no BOS/EOS removal. Both include the whole sequence call.
- Vocab loaded once per process, excluded from timing; no HTTP, Python per-call,
  JSON, hashes, input generation or output validation in timed loops on either side.
- Three existing fixed workloads (ASCII, multilingual, rendered chat) must already
  be NFC according to the pinned HF normalizer. Native still executes its full NFC
  and validation pipeline; llama has no NFC. Results compare their **common NFC
  input domain**, not arbitrary Unicode normalization support. Do not disable native
  work or normalize only one timed engine's input. Additional changed-NFC inputs
  are correctness cases, not a dishonest equivalent-input timing comparison.
- Three warmups, seven trials, 100 encode or 1,000 decode calls per trial; three
  alternating engine-order rounds, CPU 2, ReleaseFast/native target. Warm tables,
  no cross-request native result cache. No exclusive-core/clock-isolation claim.

## Correctness and artifacts

Native full actual-GGUF checker plus Debug/ReleaseFast tests precede timings.
The C adapter validates 1,240 fixture encode cases after independent official NFC
preparation, 70 raw decode cases, and all timed workloads against independent
fixtures. Timed workload input bytes are unchanged for both engines. Outputs are
validated outside each trial, actual IDs/bytes emitted and independently hashed by
the harness; reject duplicate/missing trials, bad types/times/counts/hashes.

Test-only binary corpus: `ZTBC`, version u32=1, encode/decode/workload u32 counts,
then records: length-prefixed name bytes, text bytes, u32 ID count + LE IDs,
length-prefixed raw bytes. All integers LE; fields <=1 MiB, records <=100,000;
reject truncation/trailing data/invalid names. Data derived only from pinned HF/raw
fixtures, never native outputs. Adapter is explicit development C linked to the
external engine; no native build/runtime dependency.

Harness refuses an existing output directory, records failure as well as success,
commands, compiler/header/library/model/binary hashes, host/CPU/affinity/governor,
corpus hash, actual validation/trials, median/min/max/stdev, and complete source/data
snapshots. Two baseline runs before optimization. Preserve baseline source/binary
identity; optimization requires the same workloads and policies and repeated runs.
Existing actual llama-server correctness gate remains evidence, but new component
ratios must be labeled **vs libllama**, not end-to-end server speedups.

## Optimization gate

Identify costs before replacing work. Research/spec any algorithm change, keep
exact official merge ordering (including forward ranks, ties, stale edges), added
matching, Unicode/NFC, bounds and ownership. Run all independent fixtures and
resource/cleanup tests; add adversarial cases for new paths. No expected output
changes to excuse a mismatch. Report regressions and rejected experiments, not
only wins. No quant/GPU/model implementation in this task.

## Observed baseline and first optimization decision

Two matched baseline runs completed before native changes. Instrumentation is
reproducible with `bench/profile_tokenizer.py`: it copies project sources into a
fresh ignored research directory, inserts stage clocks into that copy only, runs
the full checker, then collects per-stage totals and piece-length counts. No
profiling hooks/global clocks enter production. Per-piece clocks perturb timings;
use profiles to identify costs, not to claim uninstrumented absolute stage time.

All three existing timed workloads have pieces <=16 bytes. ASCII pieces are 50%
one byte, 25% 2–4, 25% 5–8. BPE and splitting dominate ASCII/chat; normalization
also dominates multilingual. Current BPE constructs linked symbols and a priority
queue for even one byte. First scoped experiment: bounded <=16-byte rank scan.

Short-piece algorithm (before code): initialize byte IDs and cached adjacent rules;
select the least rank across the active ordered sequence, strict-less tie handling
selects the leftmost. Replace that pair with its result, compact IDs/rules, recompute
only the two newly adjacent pairs, and rescan **all ranks**, including newly eligible
lower ranks. No vocabulary-whole-piece shortcut. At most 16 symbols makes quadratic
scan/compaction bounded; longer pieces retain the existing heap. One-byte pieces
append directly. Private stack scratch, no new ownership/cache/lifetime policy.

Acceptance: unchanged full official/raw fixtures, new independent ASCII-pair and
seeded length-boundary cases before implementation, explicit forward-rank/tie tests,
allocation failures/limits, Debug/ReleaseFast and matched repeated measurements.
Keep the heap path's long-word cases; no changing fixture expectations on failure.

## Fairness controls and broader workloads (before optimization)

The reference-call audit is [recorded](../research/tokenizer-comparison-fairness.md).
Every subsequent full run also includes `llama-preallocated`: the same public C
API, but outer output allocation/free removed from timing. This is explicitly a
reference-favorable control, not matched owned-output semantics. Initial control
run reduced encode medians by 0.8–1.6%, not enough to explain the component gap.

New independent generator `tests/reference/generate_tokenizer_optimization.py`
produced 17,793 cases **before any native optimization**: all 16,384 ASCII pairs,
lengths around the short/heap boundary, ties/repetitions, composed/decomposed Unicode,
and seeded mixed text. Repeated generation is byte-identical; SHA256
ad1d1aaf5251b489394bad678d4667309ad399eca069a5d0a0e5d659d6e1e6dc.
The expanded runner validates these plus the original 1,240 cases and 70 raw cases.

It adds three fixed independent timed workloads: `long_word` (4,096 bytes/512 IDs),
`unicode_long` (2,304 bytes/448 IDs), and `unique_code` (4,738 bytes/2,464 IDs).
The first two force long heap pieces; code includes unique seeded identifiers of
length 8–32. Exact generator/fixture hashes and input hashes pin workloads. The
native/library cases must agree before timing; NFC-changing inputs are still
correctness-only. Expanded baseline is measured before replacing the BPE path.

## Paired-baseline control and next candidate

The first short-scan timing run drifted slower even in unchanged reference and
raw-decode code. Do not interpret cross-run subtraction as an optimization result.
The runner can now interleave a retained pre-optimization binary, but only after
checking its passed manifest, exact corpus identity, binary SHA and every saved
source hash. Each baseline process still checks the full current corpus.

CPU 2 was observed busy outside the benchmark (about 16% during a one-second
/proc/stat sample; other host workloads active), with the existing `powersave`
governor. No processes were stopped and no system policies changed. A paired
repeat uses CPU 10 for **all** engines; its sibling and that core were idle in the
sample. This is not exclusive isolation. Report CPU and dispersion per run.

Second candidate, after validating the short scan: initial byte-pair lookup.
Every initial BPE edge is a pair of byte roots; probing the general numeric-pair
hash table for each one is unnecessary. Build a fixed 256×256 array of `{rank,
result}` during validated vocabulary initialization. Fill absent with max-u32 rank;
for each already validated merge with one-byte operands, store its rule at
`left_byte*256+right_byte`. This is **all byte pairs**, not a workload result cache.
It adds exactly 524,288 owned bytes and one allocation per tokenizer, freed on
all init failures/deinit. No new per-request allocation or semantic limit changes.
Both short and heap initial-edge scans use this array; newly formed edges still
use the general rule map. Byte IDs need not equal byte values. Never invent a
merge from the vocabulary or skip rank-order resolution. Full independent tests,
allocation-failure tests, long-word cases and repeated paired comparisons gate it.

## Baseline reconstruction / rerun contract

`bench/rebuild_tokenizer_baseline.py` rebuilds from a passed run's source snapshot,
never from its ignored old binary. It checks compiler/model/corpus/source hashes,
rejects path escapes or an existing destination, copies to a fresh ignored directory,
runs archived Debug/ReleaseFast tests, and validates the new executable against the
archived full actual-GGUF corpus. A new manifest records its actual binary hash;
absolute build paths can prevent byte-identical binaries. The restored directory
can be supplied directly to `--baseline-run`. Failures remain recorded; no old run
or current source is overwritten. Reconstruction and a full paired rerun were
executed successfully; [replay evidence](../bench/data/2026-09-22-tokenizer-replay/manifest.json).
