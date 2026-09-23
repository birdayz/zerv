# Tokenizer comparison fairness audit — 2026-09-22

The direct-native/HTTP timing table was not an apples-to-apples comparison. The
replacement uses the **same installed libllama** as actual llama-server, called
in-process in a standalone external adapter. This is component performance, never
native HTTP/server performance. [Measurement contract](../specs/tokenizer-matched-benchmark.md).

## Checked reference usage

Inspected exact source revision b29c606e28a01b1bc8c1351026a0fa6e616bf6c4:

1. `tools/server/server-context.cpp`, `post_tokenize`: reads add_special=false by
   default, parse_special=true by default, calls `tokenize_mixed` for content.
2. `tools/server/server-common.cpp`, `tokenize_mixed`: for string input, calls
   `common_tokenize(vocab, s, add_special, parse_special)`.
3. `common/common.cpp`, `common_tokenize`: allocates a vector with
   `text.length() + 2*add_special` capacity/length, calls `llama_tokenize` once,
   resizes to the returned length; only retries if capacity was insufficient.
4. Our C adapter uses that same public call and flags, allocates the same byte-count
   output capacity (minimum one for empty input), and **does not probe capacity or
   tokenize twice in the timed call**. It avoids vector zero-fill and server JSON
   conversion, so it does not deliberately burden llama's wrapper.
5. `llama_detokenize(...remove_special=false, unparse_special=true)` is the public
   bulk raw-byte API. We do not make one external call per token or decode through
   Python. Both sides write into a preallocated whole-sequence raw-byte buffer.
6. Actual loader resolution is checked against the pinned libllama hash, not merely
   against the compiler link argument. It matches the prior real server manifest.
   Both vocabularies remain loaded throughout a benchmark process. `vocab_only`
   avoids weights/context; it does not select another tokenization algorithm.

Sources retained in ignored third_party; exact URLs/revisions/SHA256 are in the
[source ledger](2026-09-22/sources.json). An attempted guessed `server-routes.cpp`
URL returned 404; the pinned directory listing located the actual handler in
`server-context.cpp`. No missing-source assumption was used as evidence.

## Conservative allocation control

The matched encode comparison includes request-owned output allocation/free on
both sides (native smp_allocator vs libc/C++ allocators). To test whether our C
wrapper manufactures the difference, the harness additionally times the public
llama API with **outer output allocated once and reused**, checked against the
same independent IDs. This control favors llama relative to native owned output;
it is labeled `llama-preallocated`, not claimed as identical ownership. No wrapper
allocation speedup is assumed; the actual measured control is retained.

Exact independent outputs gate every variant/trial. Warmups, iterations, CPU,
engine-order alternation, buffers and workload bytes are explicit. Native NFC,
UTF-8 validation, bounds and scratch allocation remain enabled. llama has no NFC;
all timed inputs are independently required to already be NFC. The 380 decomposed
correctness cases are normalized separately for comparison and are **not** timed
as if both APIs offered the same general-Unicode semantics.

## Build and workload limits

Installed Arch llama-cpp 0.4.1-1 / build10964 is the reference actually used, not an
invented scalar/debug implementation. [Package metadata](2026-09-22/tokenizer-fairness/package.txt),
[server loader](2026-09-22/tokenizer-fairness/server-loader.txt), and
[cached package BUILDINFO](2026-09-22/tokenizer-fairness/build-info.txt) are captured.
BUILDINFO records LTO and split-debug packaging, but does **not** establish every
compiler flag or prove that this is the fastest possible native-tuned build.
Native zerv uses ReleaseFast/native. Report this asymmetry; claims are limited to
the installed reference, not all llama builds or a fastest-tokenizer tournament.

The three initial fixed workloads are too narrow to establish general workload
superiority. They contain repeated text and short split pieces. No cross-request
result cache is being added to exploit them. Before adopting short-piece tuning,
keep independent long-piece correctness, and add long/mixed text benchmark coverage
so a short-piece gain cannot conceal a long-piece regression. CPU affinity is not
exclusive isolation; preserve dispersion and repeated runs, not just best trials.
