# Qwen vocabulary, BPE and decoding — 2026-09-22

**Active block 03: native implementation passes independent fixtures; actual-server
comparison and measurement loop in progress.** NFC
and splitting are closed separately. Follow [TODO.md](../../TODO.md); do not start
quantization/GPU/HTTP implementation alongside this package.

## Complete artifact vocabulary comparison executed

The external ggml GGUF reader, not our native parser, extracted every vocabulary
entry, type and merge from the actual artifact. Compared to the pinned official
Qwen tokenizer JSON:

- **248,044 regular entries:** identical strings at identical IDs, all GGUF type 1.
- **247,587 merges:** identical strings in identical rank order. Every pair and
  concatenated result exists in the regular vocabulary; pair keys are unique.
- **33 added entries:** identical spellings/IDs, 248044–248076, all flags
  normalized/lstrip/rstrip/single_word=false. GGUF has 27 control + 6 user-defined.
- **243 padding entries:** 248077–248319, all GGUF UNUSED type 5, absent from official
  HF vocabulary. Total GGUF vocabulary 248,320, all spellings unique.
- Every regular character belongs to the reversible ByteLevel alphabet, all 256
  byte symbols exist. Decoded regular-piece storage totals **1,843,045 bytes**;
  maximum regular piece length is **128 bytes**.

[Full result](2026-09-22/tokenizer-vocabulary-comparison.json) includes framed hashes,
oracle binary identity, source/probe hashes and all added-token details. This
compares tokenizer metadata, **not another hash of the 16 GB tensor payload**.
The artifact's existing full SHA verification remains the weight identity gate.

```sh
python3 docs/research/2026-09-22/compare-tokenizer-vocabulary.py \
  models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json \
  docs/research/2026-09-22/tokenizer-vocabulary-comparison.json
```

Use a new output path. Initial exploratory comparison **failed due to a probe
assumption**, not an artifact mismatch: the official JSON stores legacy merge
strings (`"Ġ Ġ"`), not pair arrays. Joining each string as an iterable introduced
extra spaces. The retained script explicitly validates the actual representation;
all merges then compare exactly. Do not infer a tokenizer incompatibility from
that failed probe.

## BPE ordering and reachability

Primary source: HF v0.22.2 `models/bpe/word.rs`. Rank is merge-list position; choose
lowest rank first, breaking ties by left position. Merged neighbors change candidate
edges; stale queue entries must not execute. No dropout, unknown token, suffix,
byte fallback, or ignore-merges mode in this official profile.

**201 regular tokens have no byte-seeded merge derivation.** They remain valid
output/decode tokens, not UNUSED padding. All 201 were independently decoded and
re-encoded through HF and did not re-encode to their original single ID:
[raw results](2026-09-22/tokenizer-decode-only-probes.json). Example `俱乐部` decodes
from 104328 but encodes to `[98112,96404,95938]`. A shortcut returning any whole-word
vocabulary match would therefore be incorrect with ignore_merges=false.

An exploratory rank-order reachability scan left **66 edges** without inputs after
refusing to add forward-dependent results. That counted downstream effects as well:
the later fixture generator measures **27 direct forward-rank dependencies**.
The earlier description of all 66 as direct dependencies was inaccurate. Forward
references are legal: a lower-ranked merge can become eligible after a higher-ranked
one creates its input. Recursive byte-seeded reachability confirms zero unreachable
merge edges and only the 201 no-merge regular entries:
[reachability result](2026-09-22/tokenizer-merge-reachability.json). Initialization
must resolve operands from the complete vocabulary, not require earlier ranks to
have produced every operand. A queue must permit newly eligible lower ranks.

Proposed own implementation: byte-to-ID lookup, linked live symbols, numeric
pair→(rank,result ID) table, rank/left-position priority queue with stale-edge
validation. No copied HF heap code. Reserve bounded scratch before merging; at
most n−1 successful merges and fewer than 3n inserted candidate edges for n bytes.
The subsequent [functional spec](../specs/tokenizer.md), executable raw oracle and
1,240 HF encode cases preceded implementation of this design.

## Added tokens and byte decoding

Primary sources retained under `third_party/tokenizers/v0.22.2/tokenizers/`:
`tokenizer/added_vocabulary.rs`, `pre_tokenizers/byte_level.rs`, and BPE `word.rs`.
HF extracts non-normalized added tokens **before** normalizing ordinary spans,
using leftmost-longest matching. All 33 Qwen additions are non-normalized and have
no strip/word restrictions. The initial native profile can recognize all of them;
a special-disable mode must not be approximated from GGUF control flags.

In particular, the six FIM/repository tokens 248060–248065 are **non-special in the
public official tokenizer**, but GGUF marks them control. HF skip-special decoding
retains `<|fim_prefix|>`. GGUF types alone cannot implement official skip-special
semantics. [All 33 added-token context probes and decode cases](2026-09-22/tokenizer-added-decode-probes.json).

ByteLevel maps bytes 33–126, 161–172 and 174–255 to same-valued Unicode scalars;
the remaining bytes, ascending, map to U+0100 onward. Reverse this data mapping for
regular pieces; added spellings are literal ASCII. ByteLevel's postprocessor here
has trim_offsets=false and adds **zero tokens**; it is not a BOS/EOS inserter.

Preserve **raw bytes across token boundaries**. HF IDs `[127]` and `[102]` each
decode alone to U+FFFD, but `[127,102]` decodes to `é`. Independently lossy-decoding
pieces and concatenating strings would corrupt valid output. A native raw-piece
API should allocate nothing and defer UTF-8 assembly/replacement to a specified
sequence/stream decoder. Full sequence text output and raw pieces require distinct
oracle comparisons.

HF silently omits unknown IDs, including padded IDs; the executed llama raw-piece
oracle confirms all 243 UNUSED pieces are empty. The adopted contract returns empty
raw pieces for valid UNUSED IDs and `InvalidTokenId` for unknown/missing IDs. It
renders all additions without a skip-special option. Raw bytes are deliberately
separate from later sequence/stream UTF-8/JSON handling.

## Oracle/spec gate and subsequent implementation

- Finish added-token/byte-piece oracle extraction, including all regular/added/
  padding pieces, invalid IDs, partial UTF-8 and skip-special policy. The pinned
  llama C API offers `llama_token_to_piece(..., special=true)` without UTF-8 loss;
  a vocab-only reference load can avoid weight execution. Establish/check its ABI
  rather than guessing a ctypes model-parameter layout.
- Write the functional spec: owned/borrowed vocabulary data, error cleanup, bounds,
  per-request scratch, output ownership, raw-piece vs text decoding and concurrency.
- Generate independent HF exact-ID fixtures before BPE implementation. Include the
  201 decode-only tokens, forward-rank dependencies, equal-rank ties, stale edges,
  all added tokens, NFC version edges and real chat-rendered prompts. If using a
  reduced vocabulary fixture, preserve IDs and validate identical HF behavior;
  independently exercise the complete real GGUF inventory as an integration gate.
- Implement only after those gates, then test and measure the complete pipeline.
  Actual llama-server `/tokenize` accepts add_special=false, parse_special=true,
  with_pieces=true (invalid UTF-8 pieces are byte arrays). Compare **exact IDs first**
  on semantically equivalent inputs; preserve NFC/special-policy differences.
  Separate HTTP round-trip from direct component time, and document API/allocation
  differences. Do not compare this block to server generation throughput.

The listed before-code gates were subsequently completed: [spec](../specs/tokenizer.md),
[test-only C adapter](../../tests/reference/tokenizer_pieces.c),
[HF fixture generator](../../tests/reference/generate_tokenizer_goldens.py), and
[raw-oracle identity/validation record](2026-09-22/tokenizer-raw-oracle.json).
The installed header is byte-identical to the pinned header, avoiding guessed ABI.
All 248,320 raw pieces agree with independent alphabet reversal/literals/empty
padding; all official single-piece replacement-decoded strings agree with HF.
Repeated C extraction and fixture regeneration are byte-identical.

Native owned tables, bounded encoding, raw-piece/sequence decoding and the strict
real-GGUF adapter now pass 1,240 exact-ID cases, 70 raw decode cases, and the complete
piece fingerprint. The complete suite has 24 native tests (including allocation
failure cleanup, copied ownership, malformed tables, limits, borrowed output and
error-atomic invalid-ID decoding). Actual-server comparisons and repeated component
measurements subsequently passed; the [initial report](../bench/2026-09-22-tokenizer.md)
is corrected to reject direct-vs-HTTP speed comparisons. A [matched-call audit](tokenizer-comparison-fairness.md)
and native tokenizer optimization are now the active scope. Model execution/serving remain absent.

Two development compile failures were corrected before verification: Zig 0.16 uses
`std.enums.fromInt`, not the removed `std.meta.intToEnum`, and a benchmark `else for`
statement initially had an extraneous trailing semicolon. No failing binary was
used to generate expected outputs.
