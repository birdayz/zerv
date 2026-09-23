# Native byte-BPE tokenizer / Qwen3.8 profile

Scope: controlled block 03, `src/tokenizer`. [Research](../research/tokenizer-bpe.md)
resolved Unicode versions, byte alphabet, ordered merges, added-token flags and
complete official/GGUF inventory equality. This block composes the verified NFC
and splitter; it does not implement inference or an HTTP serving endpoint.

## Semantics

- Strict UTF-8 input. Extract added tokens with leftmost-longest matching before
  NFC; no strip or word constraints. Normalize ordinary spans with Unicode-9 NFC,
  then apply the verified Qwen Unicode-16 splitter, then byte BPE to each piece.
- Initialize one symbol per raw UTF-8 byte via all 256 byte IDs. Select the lowest
  merge-list rank, breaking ties by original left position. Revalidate queued edges
  after merges. Forward rank dependencies are legal. Never return an arbitrary
  whole-vocabulary match instead of following merges (201 real decode-only tokens).
- All declared added tokens are recognized. No disable-special mode, dropout,
  unknown fallback, offsets, truncation, automatic BOS/EOS, or byte prefix/suffix.
  Reject unsupported configuration rather than silently selecting another profile.
- `piece(id)` returns borrowed **raw bytes** with all added spellings rendered.
  Valid UNUSED IDs return empty bytes, matching the independent raw-piece oracle.
  Out-of-range/missing IDs return `InvalidTokenId`, deliberately stricter than HF's
  silent omission. `decode(ids, writer)` validates all IDs before writing and
  concatenates pieces without UTF-8 repair. Writer failures may leave a prefix.
  Never replace partial UTF-8 per token: `[127,102]` must yield raw bytes for `é`.
  Sequence/stream UTF-8 replacement and JSON encoding belong to the later output
  boundary; this API does not mislabel arbitrary bytes as a Unicode string.

## API / ownership / resource limits

`Tokenizer.init(allocator, entries, merges, limits)` copies raw token bytes and
builds owned dense ID descriptors, byte IDs, added-token lookup and numeric merge
index. A fixed 256×256 initial byte-pair rule table adds 524,288 owned bytes and
one allocation; it is built only from validated merges and freed on every init
failure/deinit. It is not a prompt/result cache. Entries have ID, kind (normal/added/unused), and raw bytes; merge records have
left/right/result IDs, with array order defining rank. IDs may have holes for tests;
missing IDs cannot decode or appear in merges. Normal entries must be nonempty,
all 256 byte roots unique/present, and every merge's normal result must be exactly
its operands concatenated. Reject duplicate IDs, duplicate byte roots/pairs, invalid
references/kinds/added UTF-8/empty additions and inconsistent merged bytes. UNUSED
pieces must be empty. Forward references are resolved against the complete table.

Default initialization limits: vocabulary ID space 1,048,576; 2,097,152 merges;
64 MiB total piece bytes; 16 KiB per piece; at most 1,024 additions. Hard-cap hash-map
counts below Zig's overflow boundary and use checked arithmetic. Failures clean up
all allocations; `deinit` frees all owned data. Caller input may be freed after init.
After initialization tables are read-only and concurrent encodes use independent
workspaces. No global mutable state.

`encode(allocator, input, limits)` returns owned `[]u32`, freed by that allocator.
Default limits: 1 MiB input, 4 MiB total normalized bytes, 1,048,576 output tokens.
Temporary normalization/symbol/queue buffers are bounded and reused between pieces
within a request. Pieces of at most 16 bytes use a bounded stack rank scan with
leftmost ties and full reconsideration of newly eligible lower ranks. Longer pieces
retain the heap: reserve capacity before merging, fewer than 3n candidate insertions
for n input bytes. Both paths use the initial byte-pair table, then the general
merge index for newly formed edges. No heap work per merge; no allocation in piece lookup
or generated-token decoding. Encode errors return no partial owned output. Public
errors include invalid UTF-8, input/normalized/token limits, overflow and allocation
failure. Limits are caller-configurable; no resource knob without enforcement.

`fromGGUF` is the strict Qwen3.8 adapter: gpt2/qwen35, exact target vocabulary/merge
counts, validated token types and all 33 expected additions at expected IDs. Convert
normal glyph strings through the byte alphabet, preserve added literals, make
UNUSED pieces empty. Build operands/results from the complete string→ID index,
never rank-order creation. Do not infer official special flags from GGUF control
flags; this initial raw API renders all additions and exposes no skip-special mode.
GGUF mapping/container need not outlive the owned tokenizer. Other model profiles
are unsupported by this adapter, not silently treated as Qwen3.8.

## Independent fixture/raw-byte oracle — before native implementation

Pinned official HF Tokenizers 0.22.2 supplies IDs and decoded text. Pinned installed
llama build10964/b29c606e supplies **raw pieces** through `llama_token_to_piece` with
special=true. Installed `/usr/include/llama.h` was byte-compared to the retained
pinned header, identical; library SHA is recorded in the oracle manifest.

Use a small **external test-only C adapter**, compiled with the installed C header
and linked to libllama. It calls default model parameters, sets vocab_only=true and
n_gpu_layers=0, loads the actual GGUF without a context/weights, and dumps every
ID's raw piece. This avoids guessed ctypes struct layout. It is never linked to or
invoked by native builds/production. Frame dump as u32 vocabulary count, then u32
length + raw bytes per ID (little endian). Check errors, count/capacity and writes;
refuse existing output files. Record adapter/header/compiler/library hashes.

Before native code, generator must compare every raw piece against independent
ByteLevel alphabet reversal (regular), literal additions and empty padding, and
check HF decoded text against replacement-decoding whole byte sequences. Generate
exact HF encode IDs for Unicode/version edges, each addition in context, all 201
decode-only entries, forward-rank cases, overlap/stale/tie cases, seeded mixed text,
long words and real official chat prompts. No native outputs create expectations.

Use a full data-only fixture to avoid pruning merge semantics: magic ZBPE, u32
vocabulary/merge counts; dense entries {u32 byte length,u8 kind (1 normal,2 added,
3 unused),3 zero reserved bytes,raw bytes}; then {left,right,result:u32} records in
original rank order. Pin source/output hashes and retain the artifact data license.
A JSON fixture pins exact IDs and raw decoded hex, full-piece SHA and oracle
identity. Native tests use only these checked-in files, never HF/llama/third_party.

Gates: all fixtures exact in Debug/ReleaseFast; all raw pieces match the full oracle
fingerprint; malformed/limit and allocation-failure cleanup tests; fixture
regeneration identical; real GGUF adapter matches the full fixture's vocabulary,
merges, encoding and decoding (not just a small synthetic vocabulary).

## Measurement / actual server gate

The [matched direct libllama contract](tokenizer-matched-benchmark.md) supersedes
the original mixed-boundary timings for relative component performance. Retain
actual HTTP comparisons as integration evidence, not a direct-call speed ratio.

Build an independent native checker/benchmark, not a fake serving endpoint. Load
vocabulary once; time complete encode (including normalization and request scratch)
and raw decoding separately on fixed workloads. Exact IDs/bytes gate all timings.
Compare to pinned HF with cache/ownership differences disclosed. Use pinned CPU,
three warmups, seven trials, three alternating rounds; 100 encode calls or 1,000
decode calls per component trial, 20 reused-connection HTTP calls per server trial.
Retain raw results, variance,
commands/binary hashes and complete source/data/license snapshots. Reference timing
includes its Python/Rust/API costs; native includes allocation/free of request output.

Also start actual pinned llama-server on loopback, exact GGUF, no speculative work;
compare `/tokenize` with add_special=false, parse_special=true. Include composed/NFC
texts where IDs are equivalent, plus separately reported decomposed/special-policy
mismatches. Time repeated real HTTP calls with a reused connection, report round-trip
separately from direct component time (no claim that omitted native HTTP is faster
serving). Retain full launch/effective config and stop the reference after use.
This actual-server gate cannot be replaced by a library-only comparison. Full
model/Chat Completions comparisons remain later queue items.
