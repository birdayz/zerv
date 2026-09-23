# Qwen3.8 tokenizer and template investigation — 2026-09-22

Status: NFC and text splitting are independently verified; complete native BPE
encoding/decoding remains queued. Sources:

- Official revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
  `tokenizer.json` and `tokenizer_config.json`, retained under
  `third_party/Qwen/Qwen3.8-27B/<revision>/`.
- llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, `src/llama-vocab.cpp`,
  `src/unicode.cpp`, `src/unicode-data.cpp`, `include/llama.h`, under
  `third_party/llama.cpp/<revision>/`. External oracle only, never production code.

## Observed source semantics and unresolved compatibility

The official tokenizer has **NFC normalization**, regex Split (Isolated), then
ByteLevel with no prefix space and no regex. BPE has no dropout/unknown token,
empty continuing/end suffixes, no byte_fallback and no ignore_merges. ByteLevel
encodes arbitrary UTF-8 bytes to GPT-2's reversible byte alphabet; raw bytes and
byte-encoded Unicode token strings must not be confused. The regex is:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Numbers split individually; mark categories join words. Branch priority, whitespace
backtracking and Unicode categories matter. The selected llama QWEN35 path uses
this pattern, a custom splitter and rank-prioritized BPE, breaking equal-rank ties
by left position. Its BPE path does not appear to perform NFC. Test decomposed
accents against both oracles before deciding native compatibility policy. A matching
ASCII corpus alone would not resolve that difference. Added tokens also include
**non-special** entries, so `special=false` does not mean bypass all added tokens.

The official template explicitly errors on developer roles. It inserts reasoning
effort instructions (default xhigh; medium inserts none; low inserts short-thinking
instruction), strips message edges, rejects misplaced system messages and missing
user queries. For text-only assistant history it preserves a `<think>` block by
default, and closes messages with im_end. Disabling thinking emits an empty think
block in the generation prefix. The artifact publisher advertises changes to
roles/tools; byte-exact comparison remains required. Native template support must
reject unsupported paths, not silently render an approximation.

## Executable oracle setup / planned gates

Created isolated `.tools/tokenizer-oracle-venv` with Python 3.14.7. No system package
was changed; these packages are development oracles only. Pinned direct packages:
`tokenizers==0.22.2`, `Jinja2==3.1.6`, `regex==2026.9.10`. Transitive versions and
binary hashes must be recorded with generated fixtures. Initial attempted regex
version `2026.9.18` was unavailable; pip aborted without installing it. The selected
`2026.9.10` came from the actual available-version output.

Use Hugging Face Tokenizers to read the official tokenizer JSON and Jinja2's
sandbox to render the exact official and embedded templates (with raise_exception
and deterministic Unicode-preserving tojson). Compare against the installed
llama tokenizer using its public C API or independent reference server endpoints.
Include NFC/decomposed accents, combining-only text, number categories, CRLF,
whitespace, control bytes, emoji, CJK, added-token spelling and special-token modes.
Record exact prompt bytes, token IDs and decoded bytes. Do not use the reference
server in production. A benchmark must time the same tokenization/template modes.

No tokenizer policy/spec is finalized until these experiments resolve differences.

## Executed gates and decisions

Official HF Tokenizers 0.22.2 vs actual llama-server `/tokenize` probes confirmed:
`e\u0301` → official `[933]`, llama `[68,52033]`; decomposed Hangul also differs.
NFC inputs, Hindi marks, CJK, number categories, emoji, whitespace and selected
added-token spellings agreed. [Raw probes](2026-09-22/tokenizer-probes.json).
Thus NFC is a required explicit official-tokenizer behavior, not an assumption of
GGUF equivalence. Native NFC is now independently verified; complete BPE/added-token
encoding and decoding remain the active [research block](tokenizer-bpe.md) in
[TODO.md](../../TODO.md).
The native splitter now passes its [separate research/fixture gate](pretokenization.md).

Before native template code, `generate_chat_goldens.py` rendered **100 cases**
through both source templates using independent Jinja2. Three differ: no user,
only tool-response-like user, and developer prefix. The supported ordinary text
cases agree. Adopted official text-only semantics and reject publisher extensions;
[contract](../specs/chat-template.md). The independent fixture gate preceded native
implementation. Native Debug/ReleaseFast now match all cases, with limit/UTF-8/
validation-atomicity/writer-error tests and [component measurements](../bench/2026-09-22-chat-template.md).

Further normalization investigation resolved a second mismatch: HF's NFC backend
uses **Unicode 9.0.0** property data, not Python's Unicode 16.0.0. Single-scalar
agreement hid 120 ordering differences in combining-mark contexts. Native official
compatibility must pin Unicode 9 data explicitly; see [the source trace, algorithm
and required gates](normalization.md). Native normalization now passes those gates
with [two recorded component benchmark runs](../bench/2026-09-22-normalization.md).

## Initial splitter observations and subsequent resolution

Before the one-active-block queue was established, HF pretokenizer probes found
that case-insensitive `'s` also consumes LONG S U+017F: `"'ſt"` splits into `"'ſ"`
and `"t"` (raw strings after reversing ByteLevel's alphabet). An ASCII-only case
fold is therefore insufficient. U+001C is not regex whitespace. New Unicode-16
Todhri U+105C0 and Tulu-Tigalari U+11380 are recognized as letters, unlike the
Unicode-9 NFC backend's property version. This does **not** establish exhaustive
classifier agreement. The subsequent [exhaustive probe and native gate](pretokenization.md)
resolved it with zero discrepancies against Unicode-16 classes.

Retained Unicode-16 `PropList.txt` and HF v0.22.2 BPE `word.rs` are in the source
ledger. The latter confirms rank-first, left-position tie ordering and stale-edge
checks after neighboring merges. These observations do not authorize copying its
implementation or starting BPE alongside the active package.
