# Qwen pretoken splitting — 2026-09-22

Scope: **one block in `src/tokenizer`**, ordinary text splitting only. No BPE,
added-token extraction, normalization, byte alphabet conversion or model execution.
NFC is a separately verified preceding block. [Controlled queue](../../TODO.md).

## Primary sources and executed resolution

- Official tokenizer JSON at Qwen/Qwen3.8-27B revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` specifies regex `Split`, `Isolated`,
  invert=false, followed by ByteLevel without its own regex/prefix space.
- HF Tokenizers v0.22.2 `src/pre_tokenizers/split.rs` and `src/utils/onig.rs`:
  Split delegates to the Oniguruma backend; retain matches and unmatched spans
  separately in isolated mode. UTF-8 matching, no locale configuration supplied.
- Unicode 16.0.0 `UnicodeData.txt`, `PropList.txt`, `CaseFolding.txt` define the
  candidate properties. Sources are retained in revision-qualified `third_party/`
  and recorded in the source ledger. No source implementation is imported/copied.

[Executed exhaustive probe](2026-09-22/pretoken-properties.json), reproducible with
[the research script](2026-09-22/probe-pretoken.py): every one of 1,112,064 scalars
was checked through HF's real regex Split for L, M, N, whitespace and contraction
letter case-fold membership. **Zero discrepancies** against Unicode-16 data:
141,028 letters, 2,501 marks, 1,911 numbers, 25 whitespace scalars. These properties
are **not Unicode 9**, despite that version being required by HF's NFC backend.

For contraction letters s/t/r/e/v/m/l/d, the regex admits 17 scalars: the eight ASCII
lowercase/uppercase pairs plus **LONG S U+017F** for s. Default simple case-fold
membership agrees exhaustively; no Turkic locale is requested. Fixture generation
must also probe every Unicode CaseFolding entry in contraction contexts, including
multi-character folds, not infer whole-regex behavior from membership alone.

```sh
.tools/tokenizer-oracle-venv/bin/python docs/research/2026-09-22/probe-pretoken.py \
  third_party/unicode/16.0.0 docs/research/2026-09-22/pretoken-properties.json
```

The output path must not already exist. [Raw branch probes](2026-09-22/pretoken-probes.json)
confirm LONG S, CR/LF, whitespace lookahead and non-ASCII prefixes. Examples:
`'ſt` → `'ſ`, `t`; two spaces + tab + X → two spaces, tab + X;
newline + space + tab → newline, space + tab. U+001C is **not** whitespace, whereas
U+0085 is; either can prefix a following word in this pattern's second branch.

## Exact pattern and state transitions

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Apply alternatives in the given order at the current scalar, with greedy ranges:

1. Apostrophe contractions; ASCII fold plus U+017F in the s position. Consume only
   the matched contraction, not following letters.
2. A maximal L/M run, optionally preceded by exactly one scalar that is not CR,
   LF, L or N. A leading mark is itself valid; greedily consuming it as an optional
   prefix cannot change the resulting maximal run. A number is not a prefix.
3. Exactly one N scalar (all N categories, not just decimal ASCII digits).
4. A maximal non-whitespace/non-L/M/N run, optionally preceded by one **ASCII
   space**, then any immediately trailing CR/LF. The prefix is not arbitrary space.
5. From a contiguous whitespace run, consume through its **last CR/LF**, leaving
   whitespace after that break untouched. This implements greedy `\s*` followed
   by greedy CR/LF with backtracking, without a general regex engine.
6. If no CR/LF was present, consume the whole whitespace run at end of input;
   otherwise consume all but its last scalar when at least two are available.
   The omitted scalar satisfies the negative nonspace lookahead. For a single
   whitespace before nonspace, this alternative fails.
7. Consume that single remaining whitespace when no earlier alternative matched.

These alternatives cover every valid scalar and never produce empty pieces.
Examples such as ` !a` must consume ` !` then `a`, while `!a` is one word-prefixed
piece; stripping punctuation or whitespace would be wrong. Output boundaries are
UTF-8 byte offsets, not Python scalar offsets or ByteLevel-alphabet string offsets.

## Design alternatives and independent gate

A foreign regex library exceeds the permitted runtime boundary. A general regex VM
would add unnecessary state/compilation overhead to this fixed profile. Implement
an allocation-free specialized iterator, with a compact immutable range classifier
built from UCD (including First/Last ranges). ASCII property lookup is precomputed;
non-ASCII uses a range search. This is original implementation of the documented
pattern, not copied reference code. Separate the classifier and matcher within the
same tokenizer package; a future BPE layer consumes borrowed byte slices.

Before native code, generate reference ends from **HF Split with the official
pattern**, not from our proposed branch algorithm. Corpus: exhaustive short strings
over letters/numbers/space/tab/CR/LF/punctuation/marks; all Unicode folding entries
in contraction contexts; seeded multilingual/property-boundary text; real official
chat-rendered prompts; adversarial long whitespace/mark runs. Independently verify
every scalar's property membership again during generation. Require byte-exact
piece ends and full input coverage; no numeric tolerance.

Native gates also include invalid UTF-8, empty input, repeated exhaustion, input
byte limits and borrowed-buffer boundaries. Benchmark equivalent HF regex Split
without NFC/ByteLevel/BPE, disclosing HF allocated strings/offsets vs native borrowed
slices. Actual llama-server has no standalone regex-split endpoint; do not time this
subset against full `/tokenize` as a speed claim. The complete-tokenizer block must
perform that actual server comparison on correctness-equivalent inputs.

## Executed native loop

The spec and 47,919 independent HF split fixtures preceded native implementation.
The resulting iterator passes all exact boundaries and the exhaustive property
fingerprint in Debug and ReleaseFast; the complete suites now pass 20 native and
19 Python tests. Independent fixture regeneration is byte-identical. Two
[component benchmark runs](../bench/2026-09-22-tokenizer-split.md) retain raw timings,
API ownership caveats and full source/data snapshots. No general regex engine,
BPE or serving functionality is implied by this completed splitter block.
