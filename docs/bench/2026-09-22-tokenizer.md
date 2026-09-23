# 2026-09-22 — native complete tokenization and actual llama-server comparison

## Scope and correctness

Controlled block 03 now has native owned byte-BPE tables, bounded Qwen encoding,
raw-piece/sequence decoding, and a strict actual-GGUF adapter. Research/spec and
independent fixtures preceded native code. [Contract](../specs/tokenizer.md),
[research and corrections](../research/tokenizer-bpe.md), [queue](../../TODO.md).
**No model execution or native Chat Completions endpoint is implied.**

Executed gates:

- **248,320 raw pieces** agree exactly with the independent llama C API oracle;
  independently checked against byte-alphabet reversal, literal additions and empty
  UNUSED padding. Header/library/adapter/compiler identities are in the
  [oracle record](../research/2026-09-22/tokenizer-raw-oracle.json).
- **1,240 official HF encode cases**, **70 raw decode cases**, full model table
  fixture and actual 16 GB GGUF adapter agree exactly. Includes Unicode/NFC edges,
  all additions, 201 decode-only tokens, 27 direct forward-rank cases, long repeated
  words, seeded mixed text and real official chat-rendered prompts.
- Debug/ReleaseFast: **24/24 native tests**; **25/25 Python tests**, including malformed
  tables, bounds, ownership, injected allocation failures, invalid-ID decode
  atomicity and benchmark result validation. Independent C extraction and fixture
  regeneration are byte-identical. No expected output comes from native code.
- Native `zerv-tokenizer-bench` is a static ELF with **no dynamic section**. The C
  adapter/libllama is external test tooling only, never a native build/runtime dep.

## Actual llama-server gate

Started the pinned **real** llama-server build10964/b29c606e on loopback 18081 with
the exact model, CPU 2 affinity, t/tb=1, ngl=0, context=2048, one slot, b/ub=256,
FP16 KV, flash attention on, no speculation/context shifting/web UI/model warmup,
and the pinned official template. Tokenization is CPU work; no generation was
requested. This is not a tuned inference-serving tournament.

[Exact launch command](data/2026-09-22-tokenizer-server/command.txt),
[binary/config manifest](data/2026-09-22-tokenizer-server/manifest.json),
[effective properties](data/2026-09-22-tokenizer-server/props.json),
[full server log](data/2026-09-22-tokenizer-server/server.log).
The server was stopped after both runs; port 18081 no longer listens.

For all 1,240 cases, actual `/tokenize` used add_special=false, parse_special=true
and with_pieces=true. The returned raw pieces reconstructed **every original input**.
**860 inputs matched official/native IDs directly. 380 differed on NFC-changing
inputs.** The repeat harness strengthened the gate by submitting each changed input
after official NFC: **all 380 then matched exactly**, zero residual differences.
No normalization incompatibility is hidden or counted as a native failure.
[Complete strengthened comparison](data/2026-09-22-tokenizer-repeat/server-comparison.json).
The first run only classified NFC-changing inputs; its original record is retained.

All three timed workloads are NFC-equivalent and produce exactly the same IDs on
native, HF and actual server. HTTP timing uses with_pieces=false and a reused
connection; JSON request bytes are prepared before timing. HTTP includes request/
response handling and client JSON decode; **native and HF component times do not
include HTTP**. Therefore the HTTP numbers are not a zerv serving speedup ratio.

## Reproduction and workloads

Ryzen 9 3900X, Linux 7.2.6; Zig 0.16.0 ReleaseFast/native; isolated Python 3.14.7,
HF Tokenizers 0.22.2. Each harness run tests/builds, validates all native cases and
actual server cases, and independently rehashes the entire model before timing.
No system package/driver/clock/power changes. CPU affinity is not exclusive host
isolation; client and reference server share CPU 2.

Run the command in the server record, then (use fresh output directories):

```sh
.tools/tokenizer-oracle-venv/bin/python bench/run_tokenizer.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --tokenizer third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json \
  --server http://127.0.0.1:18081 \
  --server-record docs/bench/data/2026-09-22-tokenizer-server \
  --cpu 2 --output docs/bench/data/2026-09-22-tokenizer-repeat
```

The actual first output directory was `2026-09-22-tokenizer`; repeat was as above.
Server-record files are explicit inputs, not invented automatic launch settings.
Capture `GET /props` and binary hashes when reproducing with another server instance.
The native checker can be run independently after `zig build tokenizer-bench-build
-Doptimize=ReleaseFast`:

```sh
zig-out/bin/zerv-tokenizer-bench \
  models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf tests/fixtures/tokenizer/manifest.json
```

Three engine-order rounds alternate native/HF/HTTP vs HTTP/HF/native. Per workload
and operation: three warmups, seven trials per round (**21 observations**). Each
trial has 100 encode calls, 1,000 decode calls, or 20 HTTP calls. Outputs are checked
outside timed loops; native uses compiler barriers. Native includes temporary
request allocations and output free in encoding; vocab is loaded once per process.
Native decode writes raw bytes to preallocated buffers; HF allocates UTF-8 text,
including replacement behavior and Python objects. HF word cache is warmed by
warmups; native has no word-result cache. No identical-API-overhead claim.

| Workload | UTF-8 input/output bytes | Token count |
|---|---:|---:|
| ASCII contractions/numbers/newlines | 2,112 | 832 |
| Multilingual/emoji/numbers | 1,664 | 800 |
| Official rendered chat | 300 | 53 |

## Results — historical, not an apples-to-apples llama speed comparison

**Correction:** the direct-native versus HTTP comparison below does not measure
relative tokenizer performance. It omits transport/JSON/server work on the native
side. HF timings also have different offset/ownership costs. Retained as raw
historical observations, not speedup evidence. A matched direct libllama comparison
is now the active task; see [its contract](../specs/tokenizer-matched-benchmark.md).


Microseconds per complete call, medians. Separate APIs are labeled explicitly.

| Run | Workload | Native encode | HF encode | Actual llama-server `/tokenize` round-trip |
|---|---|---:|---:|---:|
| First | ASCII | 87.501 | 781.648 | 734.460 |
| Repeat | ASCII | 110.153 | 861.934 | 820.555 |
| First | Multilingual | 110.567 | 471.990 | 612.874 |
| Repeat | Multilingual | 137.609 | 511.496 | 673.967 |
| First | Chat | 12.228 | 71.775 | 252.950 |
| Repeat | Chat | 16.236 | 80.802 | 275.336 |

| Run | Workload | Native raw decode µs | HF text decode µs |
|---|---|---:|---:|
| First | ASCII | 3.205 | 84.908 |
| Repeat | ASCII | 3.933 | 89.372 |
| First | Multilingual | 3.046 | 88.730 |
| Repeat | Multilingual | 3.759 | 95.601 |
| First | Chat | 0.214 | 7.351 |
| Repeat | Chat | 0.298 | 7.860 |

Repeat encode dispersion (µs; full decode dispersion also retained in raw summary):

| Workload/API | Min–max | Sample σ |
|---|---:|---:|
| ASCII/native | 91.395–123.448 | 12.322 |
| ASCII/HF | 812.399–1,128.086 | 96.378 |
| ASCII/HTTP | 767.536–904.989 | 34.580 |
| Multilingual/native | 112.137–172.473 | 18.741 |
| Multilingual/HF | 479.015–573.224 | 30.933 |
| Multilingual/HTTP | 625.580–724.627 | 33.173 |
| Chat/native | 12.356–19.997 | 2.849 |
| Chat/HF | 74.239–95.764 | 6.829 |
| Chat/HTTP | 254.395–314.102 | 18.545 |

The repeat is slower across APIs; retain that drift and dispersion rather than
selecting only the favorable first run. Native component medians beat the allocating
HF API on these workloads in both runs, with ownership/offset/cache caveats above.
Actual llama-server compatibility and HTTP measurements satisfy this block's real
server gate, **not native serving or fastest-server claims**. Vocabulary initialization
is recorded separately in command logs (one initial standalone observation was
156.947 ms); it is excluded from steady request timing and is not a tuned load result.

## Raw evidence / remaining work

- [First manifest](data/2026-09-22-tokenizer/manifest.json),
  [summary](data/2026-09-22-tokenizer/summary.json).
- [Repeat manifest](data/2026-09-22-tokenizer-repeat/manifest.json),
  [summary](data/2026-09-22-tokenizer-repeat/summary.json).

Each run includes all raw trials, commands, validation output, model SHA, binary/
reference/compiler hashes, and **64-file complete source/data/license snapshots**;
all snapshot hashes verified. The repeat adds normalized server retry checking and
its unit test; native code/binary is unchanged between runs. Research's initial 66
forward-dependency count included downstream effects; the executable fixture records
27 direct forward references. Neither issue changes the native correctness result.

The serving/quant queue is paused while matched tokenizer measurement and optimization
are active. The broader queue includes Q4_1 decoding and the remaining quant
formats, GPU primitives, model/session execution and HTTP. Generated raw bytes still
need the later sequence/stream UTF-8/JSON boundary. **No native model or server is
implemented yet.**
