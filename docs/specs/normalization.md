# Native Unicode-9 NFC

Research completed in [normalization.md](../research/normalization.md), including
93,610 independent normative NFC checks before native implementation. This explicit
version matches the official tokenizer's pinned backend; it is not Unicode 16 NFC.

`text.nfc.scratchSize(utf8)` validates UTF-8 and returns the required count of u21
scratch elements (twice fully decomposed codepoints); pure ASCII requires zero.
`normalize(input, output, scratch)` returns the number of UTF-8 output bytes.
Caller owns disjoint input/output/scratch buffers. No allocation, global mutable
state, I/O, foreign code or per-character heap work. Invalid UTF-8, insufficient
scratch/output and size overflow are typed errors. Output remains untouched on
all errors; scratch may be modified. Empty input succeeds. Callers impose request
byte limits before this component; operations are bounded by input and buffers.

Algorithm: canonical decomposition (pre-expanded immutable mappings), arithmetic
Hangul decomposition, stable CCC counting-sort only within unordered nonstarter
runs, then unblocked primary composition including Hangul. No compatibility
normalization, stripping, locale processing, surrogate acceptance or stream-safe
CGJ insertion. Table misses preserve the scalar with CCC zero. Encode only after
checking the exact resulting output size. ASCII uses a validated identity fast path.

Own generation tool consumes normative Unicode-9 data and emits data-only packed
little-endian tables: header counts; decomp rows {cp,len,4 scalars} (24 bytes), CCC
rows {cp,class} (8 bytes), composition rows {pair=(a<<21)|b:u64,result:u32} (12 bytes).
Rows sorted by lookup key; reject duplicate pairs, cycles or expanded length >4.
Native table bytes are trusted checked-in data with source/generator/output hashes
and Unicode data license retained. Never import Rust/Python implementation code.

Before native code, generator validates all Unicode-9 NormalizationTest NFC relations
through independent HF NFC and writes binary golden records (u32 input length,u32
output length,input UTF-8,output UTF-8), plus all-scalar and mark-context output
fingerprints. Reject existing output paths. Native Debug/ReleaseFast must reproduce
all records and fingerprints, including codepoints not in Unicode-9 data. Test
invalid UTF-8, exact/insufficient buffers, ASCII/empty, composition blocking and
long reversed/equal CCC runs. Fixture generator may use isolated tokenizers; builds
and native tests must not. A [separate component benchmark](normalization-benchmark.md)
times normalization with exact reference hashes; [two executed runs](../bench/2026-09-22-normalization.md)
complete this block's measurement loop, not tokenizer or serving validation.
