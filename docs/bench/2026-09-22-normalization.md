# 2026-09-22 — native Unicode-9 NFC component

## Scope and correctness

Finish the `src/text` building block before native tokenizer work. This is **not
BPE, model execution, or a llama-server speed comparison**. Native UTF-8 output is
compared byte-for-byte to the official tokenizer's pinned HF 0.22.2 NFC backend.
That backend uses Unicode **9**, not the host Python's Unicode 16 properties;
[research](../research/normalization.md), [contract](../specs/normalization.md),
[benchmark contract](../specs/normalization-benchmark.md).

Executed before measuring:

- Native Debug and ReleaseFast: **17/17 tests** each, including all 93,610 normative
  NFC relations, two exhaustive 1,112,064-scalar fingerprints (plain and combining
  context), long/equal-class runs, invalid UTF-8, bounds and error atomicity.
- Table, fixture and generator SHA provenance checks; independent regeneration
  into a new directory produced byte-identical tables, goldens and manifest.
- Full corpus output equality before timing and after each trial in both engines.
- Python harness/provenance suite: **14/14 tests**; missing, duplicate and corrupt
  results reject rather than entering the summary.
- `readelf -d zig-out/bin/zerv-nfc-bench`: **no dynamic section**. Native code uses
  neither HF nor foreign implementation code. Data comes from normative UCD.

## Reproducibility / setup

Ryzen 9 3900X, Linux 7.2.6, pinned CPU 2; Zig 0.16.0 ReleaseFast native target;
Python 3.14.7 with tokenizers 0.22.2 in the isolated oracle venv. No GPU work,
driver/power/clock changes, or serving process in this experiment. The host was not
isolated from all other work; affinity is placement, not exclusive CPU ownership.

Raw runs include exact commands, full stdout/stderr, compiler/native/oracle binary
hashes, source/data/license snapshots and package versions:

- [First run](data/2026-09-22-nfc/manifest.json), [summary](data/2026-09-22-nfc/summary.json).
- [Repeat run](data/2026-09-22-nfc-repeat/manifest.json), [summary](data/2026-09-22-nfc-repeat/summary.json).

Commands actually executed:

```sh
.tools/tokenizer-oracle-venv/bin/python tests/reference/generate_nfc.py \
  --ucd third_party/unicode/9.0.0 \
  --data third_party/nfc9-verification-01/nfc9.bin \
  --fixtures third_party/nfc9-verification-01/fixtures
cmp src/text/data/nfc9.bin third_party/nfc9-verification-01/nfc9.bin
diff -r tests/fixtures/nfc9 third_party/nfc9-verification-01/fixtures
.tools/tokenizer-oracle-venv/bin/python bench/run_nfc.py \
  --cpu 2 --output docs/bench/data/2026-09-22-nfc
.tools/tokenizer-oracle-venv/bin/python bench/run_nfc.py \
  --cpu 2 --output docs/bench/data/2026-09-22-nfc-repeat
```

Use new output paths to repeat. Each run rebuilds/tests, uses three comparison
rounds alternating native/HF first, three warmups per case/process and seven trials
per round: **21 trials per workload/engine**. Iterations per trial are 1,000 ASCII,
100 multilingual and 10 marks. Buffers are warm; setup, hashing and JSON are outside
timing. Native uses a compiler memory barrier to retain output work.

| Workload | Input bytes | Output bytes / capacity | Native scratch bytes |
|---|---:|---:|---:|
| ASCII identity | 12,800 | 12,800 | 0 |
| Multilingual accents/Hangul/CJK/emoji | 14,592 | 12,288 | 53,248 |
| Repeated reversed combining-class run | 16,385 | 16,385 | 65,544 |

The native table is 67,248 bytes. Scratch is caller-owned u21 elements occupying
four bytes each here; the counts above were emitted by the executable, not inferred
from an estimate. API code also uses bounded stack locals (including 256 counting
positions). Native timings include UTF-8 validation and exact scratch sizing on
every call, but no heap work. HF receives already-decoded Python strings, preserves
its internal normalization alignment information and allocates returned strings;
Python/Rust boundary and ownership costs remain timed. These are equivalent text
outputs, **not identical allocation/offset-tracking interfaces**.

## Results

Microseconds per complete input normalization; ratios are native/HF (lower favors
native). Values derived from the checked-in raw JSON, not test runtimes.

| Run | Workload | Native median µs | HF median µs | Native/HF |
|---|---|---:|---:|---:|
| First | ASCII | 4.765 | 718.389 | 0.00663 |
| Repeat | ASCII | 5.200 | 715.708 | 0.00727 |
| First | Multilingual | 511.352 | 562.520 | 0.90904 |
| Repeat | Multilingual | 528.394 | 572.583 | 0.92283 |
| First | Marks | 818.152 | 770.066 | **1.06244 (loss)** |
| Repeat | Marks | 704.243 | 863.954 | 0.81514 |

Repeat-run dispersion (µs; 21 trials each):

| Workload | Native min–max | Native sample σ | HF min–max | HF sample σ |
|---|---:|---:|---:|---:|
| ASCII | 3.919–7.482 | 1.014 | 690.576–766.381 | 18.394 |
| Multilingual | 478.409–570.123 | 28.031 | 506.297–628.939 | 35.386 |
| Marks | 654.716–884.223 | 65.416 | 644.290–898.717 | 79.392 |

The ASCII identity path avoids generic normalization/alignment allocation work;
the large API-level difference is not evidence for equally large tokenizer or
serving improvement. Multilingual medians favor native in both runs, but observed
ranges overlap. **The marks comparison changes sign across repeats:** retain the
initial 6.2% loss; no stable speed advantage is established there. No optimization
claim is made from the favorable repeat alone. These results close the measurement
loop, not the ultimate fastest-server objective.

## Why this is not benchmarked as an NFC win over llama-server

The pinned actual server's `/tokenize` does not perform NFC. Existing real probes:

| Input | Official HF IDs | llama-server IDs |
|---|---|---|
| composed `é` | `[933]` | `[933]` |
| decomposed `e + U+0301` | `[933]` | `[68,52033]` |

[All probes](../research/2026-09-22/tokenizer-probes.json);
[server/model identity and actual request record](2026-09-22-reference-bringup.md).
Those are previous compatibility observations, **not new server timings in this
run**. There is no equivalent NFC endpoint to benchmark. Timing NFC-only against
HTTP plus token splitting/BPE would be misleading.

The controlled queue therefore keeps actual llama-server `/tokenize` comparison
as a hard gate for the complete native tokenizer. Match exact token IDs on equivalent
NFC inputs before timing, and keep unnormalized/special-token discrepancies visible.
Model-session and full HTTP serving later require their own matched, tuned
llama-server comparisons. Native serving is still absent.
