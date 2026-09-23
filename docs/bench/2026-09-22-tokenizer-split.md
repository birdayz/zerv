# 2026-09-22 — native Qwen text splitting vs HF

## Building-block boundary

This closes **only the text splitter** in `src/tokenizer`, after the Unicode-9 NFC
block was closed. No BPE, added-token handling, byte decoding, GPU or serving code
was started alongside it. [Research](../research/pretokenization.md),
[spec and benchmark contract](../specs/tokenizer-split.md), [queue](../../TODO.md).

The native fixed-profile iterator is allocation-free and returns borrowed UTF-8
slices. Unicode-16 L/M/N/whitespace data is derived from normative UCD; it differs
from the Unicode-9 properties required by the separate NFC backend. Neither a
foreign regex library nor copied reference implementation is in native code.

## Independent correctness gates executed

- Before native implementation, generated **47,919 HF Split cases**: exhaustive
  length-0–5 strings over eight branch-sensitive characters, Unicode CaseFolding
  entries in contraction contexts, seeded mixed/property-boundary text, official
  chat prompts and long runs. Exact byte ends and full input coverage required.
- Every one of **1,112,064 scalars** independently checked against HF regex
  L/M/N/whitespace; zero UCD discrepancies. Native reproduces the independent
  property fingerprint. An additional exhaustive research probe verified the
  contraction character class, including LONG S U+017F.
- Debug and ReleaseFast: **20/20 native tests** pass, including all previous blocks,
  strict UTF-8, limits, borrowed pointer boundaries and independent/exhausted iterator
  state. **19/19 Python tests** pass, including corrupt/incomplete benchmark rejection.
- Independent generator rerun into `third_party/split-verification-01/` produced
  byte-identical table, fixtures and manifest. Table/generator/golden SHA checks pass.
- All benchmark outputs match reference ends before and after every trial.
- Native ELF: `readelf -d zig-out/bin/zerv-split-bench` reports **no dynamic section**.

## Reproduction

Ryzen 9 3900X, Linux 7.2.6, CPU affinity 2, unchanged power/clock policy; Zig 0.16.0
ReleaseFast native target; isolated Python 3.14.7, HF Tokenizers 0.22.2. The CPU is
pinned, not exclusively reserved. No GPU or model-serving process in these runs.

Commands executed (use new output paths on rerun):

```sh
.tools/tokenizer-oracle-venv/bin/python tests/reference/generate_split_goldens.py \
  --ucd third_party/unicode/16.0.0 \
  --tokenizer third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json \
  --chat tests/fixtures/chat-template.json \
  --data third_party/split-verification-01/classes16.bin \
  --fixtures third_party/split-verification-01/fixtures
cmp src/tokenizer/data/classes16.bin third_party/split-verification-01/classes16.bin
diff -r tests/fixtures/tokenizer-split third_party/split-verification-01/fixtures
.tools/tokenizer-oracle-venv/bin/python bench/run_split.py \
  --cpu 2 --output docs/bench/data/2026-09-22-tokenizer-split
.tools/tokenizer-oracle-venv/bin/python bench/run_split.py \
  --cpu 2 --output docs/bench/data/2026-09-22-tokenizer-split-repeat
```

Each run rebuilds/tests first, then three independent process rounds alternate
native/HF first. Three warmups, seven trials per round, 100 iterations per workload
per trial: **21 observations per engine/workload**. Hashing, setup and JSON are
outside timing; native traverses every returned piece with a compiler barrier.

Native includes full input UTF-8 validation and returns borrowed views. HF receives
a prepared Python string and constructs a list of strings/offsets, including its
Python/Rust boundary and allocation costs. Equivalent splitting, **not identical
ownership/offset-tracking work**. Do not generalize these ratios to full tokenization.

Data: 1,981 property intervals, **15,856 immutable table bytes**; no runtime heap
scratch in the splitter. Native state is bounded and input remains caller-owned.

| Workload | Input bytes | Pieces |
|---|---:|---:|
| ASCII text/contractions/numbers/CRLF | 17,408 | 6,144 |
| Multilingual/marks/emoji/LONG S/numbers | 13,568 | 2,304 |
| Long whitespace before a word | 16,385 | 2 |

## Measured results

Microseconds per full input; lower native/HF ratio favors native.

| Run | Workload | Native median µs | HF median µs | Native/HF |
|---|---|---:|---:|---:|
| First | ASCII | 329.239 | 4,421.517 | 0.07446 |
| Repeat | ASCII | 307.882 | 4,279.057 | 0.07195 |
| First | Multilingual | 154.413 | 2,116.641 | 0.07295 |
| Repeat | Multilingual | 145.359 | 2,051.486 | 0.07086 |
| First | Whitespace | 201.274 | 954.327 | 0.21091 |
| Repeat | Whitespace | 193.596 | 917.996 | 0.21089 |

Repeat dispersion (µs, 21 trials):

| Workload | Native min–max | Native sample σ | HF min–max | HF sample σ |
|---|---:|---:|---:|---:|
| ASCII | 306.013–329.777 | 10.530 | 4,230.257–4,480.567 | 75.684 |
| Multilingual | 144.301–154.435 | 4.280 | 2,031.572–2,981.471 | 230.503 |
| Whitespace | 193.198–206.411 | 6.219 | 909.170–1,133.285 | 74.510 |

The specialized borrowed-slice API is faster than the general allocating HF API
on these inputs in both runs. This is a component observation, **not a fastest
splitter or serving claim**. Allocation/offset tracking differences and the HF
multilingual outlier are retained, not hidden by averaging runs together.

## Raw evidence and llama-server gate

- [First manifest](data/2026-09-22-tokenizer-split/manifest.json),
  [summary](data/2026-09-22-tokenizer-split/summary.json).
- [Repeat manifest](data/2026-09-22-tokenizer-split-repeat/manifest.json),
  [summary](data/2026-09-22-tokenizer-split-repeat/summary.json).

Each contains command logs, all trial rows, exact compiler/native/oracle hashes,
package versions and 53-file source/data/license snapshots. Full snapshot hashes
are checked; no research source is a native build dependency.

Actual llama-server exposes full `/tokenize`, not an isolated splitter endpoint.
Comparing this subset to that endpoint would omit vocabulary lookup/BPE/HTTP on the
native side. The **next tokenizer block must compare exact IDs and repeated timings
against actual llama-server** on semantically equivalent inputs, separately recording
its missing NFC and special-token differences. Native model execution and the
Chat Completions endpoint remain unimplemented.
