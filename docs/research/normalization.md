# Qwen-compatible NFC research — 2026-09-22

**Native Unicode-9 NFC is implemented and independently verified.** A consequential
version difference was resolved before implementation, rather than assuming any
library called NFC is identical. [Contract](../specs/normalization.md);
[measured results, including an unstable marks comparison](../bench/2026-09-22-normalization.md).

## Executed probes and source trace

HF Tokenizers 0.22.2 NFC and Python 3.14.7 / Unicode 16.0.0 NFC agree on every
individual Unicode scalar (1,112,064 cases). However, inserting each scalar into
`[ + U+0301 + scalar + U+0323` found **120 different outputs**. Example U+07FD:
Python reorders the NKO combining mark, HF does not. This is not explained by the
llama no-normalization behavior: these probes call the HF normalizer directly.
[Raw counts and differences](2026-09-22/nfc-oracle-scalar-probe.json).

Source trace:

- `tokenizers` v0.22.2 `tokenizers/src/normalizers/unicode.rs`: NFC delegates to
  `NormalizedString.nfc`.
- `tokenizers/Cargo.toml` requires `unicode-normalization-alignments = "0.1"`.
- crates.io reports a single version **0.1.12**, archive SHA-256
  `43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de`.
  Downloaded/extracted under `third_party/unicode-normalization-alignments/0.1.12/`.
- Its `src/tables.rs` declares **UNICODE_VERSION = (9,0,0)**, and generator declares
  the Unicode 9.0.0 UCD URL. This explains the later-assigned mark differences.
- PyPI's current latest tokenizers is **0.23.2**; its inspected Cargo.toml still uses
  the same dependency. Merely upgrading the package does not resolve the source
  compatibility issue. No speculative package upgrade performed.
- An attempted root Cargo.lock download at the tokenizers tag returned 404; the
  successful source files are under `tokenizers/`, with ledger hashes.

**Decision for native Qwen compatibility:** use explicit **Unicode 9.0.0 NFC**,
matching the pinned official tokenizer backend, not a silent Unicode 16 substitution.
Preserve unassigned/private-use/noncharacter scalars as that version specifies.
The component must expose its Unicode version. This is not a claim that Unicode 9
is the newest standard. Generate data from normative UCD, not by copying a Rust
normalizer or its code into production. The independent HF oracle remains external.

## Algorithm and independent gates

Normative study: UAX #15 revision 55 and Unicode 16 core chapter 3 sections 3.11
(D108–D120) and 3.12 (Hangul arithmetic). Algorithmic definitions are stable across
these versions; version-specific property data is not. Retained under
`third_party/unicode/16.0.0/`, with URLs/hashes in the ledger. Unicode 16 data/test
files were initially fetched for investigation, **not accepted as Qwen tables**.

NFC = recursive **canonical-only** decomposition, stable canonical combining class
ordering, then primary composition. Compatibility-tagged decompositions are not
used. Only swap adjacent classes a>b>0; never reorder across a starter (ccc=0),
and preserve equal-class order. Use bounded stable counting sort of nonstarter
runs instead of quadratic insertion on attacker-controlled long combining runs.

Compose against the last starter only when unblocked: no intervening starter or
class >= the candidate's class. Successful composition replaces the starter and
removes the candidate; it does not advance the previous unconsumed combining class.
Full_Composition_Exclusion (not just CompositionExclusions) excludes singleton,
nonstarter and explicit exclusions. Indic zero-class + zero-class pairs also compose;
a check restricted to nonzero-class second characters would be incorrect.

Hangul constants: SBase=AC00, LBase=1100, VBase=1161, TBase=11A7; LCount=19,
VCount=21, TCount=28, NCount=588, SCount=11172. For SIndex=s-SBase, decompose into
LBase+SIndex/588, VBase+(SIndex%588)/28, and (if SIndex%28 != 0) TBase+SIndex%28.
Reverse L+V to AC00+(L-1100)*588+(V-1161)*28; append valid nonzero T offset to an
LV syllable. Bounds must precede unsigned subtraction.

Before code: fetch/pin Unicode **9.0.0** UCD and NormalizationTest; generate only
CCC, expanded canonical decompositions and nonexcluded composition pairs. Preserve
data license. Validate every normative NFC relation through independent HF NFC;
compare every scalar and the mark-context probe. Current Python Unicode 16 is a
secondary oracle only on cases where its data version agrees; record all divergence,
never regenerate HF expectations from native output. Native tests must include
complete NFC conformance cases, composition blocking, leading marks, Hangul edges,
invalid UTF-8, resource limits, output failure atomicity and adversarial long runs.
The required repeatable native-vs-HF normalization benchmark is now executed and
[recorded](../bench/2026-09-22-normalization.md); it is not a llama-server speed claim.

## Unicode-9 oracle preflight passed

Downloaded pinned Unicode 9.0.0 `UnicodeData.txt`, `DerivedNormalizationProps.txt`
and `NormalizationTest.txt` under `third_party/unicode/9.0.0/`. The independent HF
NFC normalizer passed all **93,610 NFC relations across 18,722 normative rows**:
NFC(c1)=c2, NFC(c2)=c2, NFC(c3)=c2, NFC(c4)=c4, NFC(c5)=c4. Zero mismatches;
[source hashes/counts](2026-09-22/nfc9-reference-preflight.json). This confirms the
version selection before native table/normalizer code, not a native correctness claim.

## Native verification loop completed

The own implementation uses binary-search immutable UCD-derived tables, arithmetic
Hangul, stable counting sort of unordered nonstarter runs, and unblocked composition.
The checked-in table is 67,248 bytes, with data license and a compile-time SHA-256
fingerprint tested against the independently generated manifest. Caller-owned
buffers, no allocation or foreign runtime dependencies; output is error-atomic.

Debug/ReleaseFast both pass 17/17 native tests, including 93,610 NFC relations and
all 1,112,064 scalars in each of two fingerprint contexts. Independent regeneration
into `third_party/nfc9-verification-01/` was byte-identical. Two complete benchmark
runs validate outputs before/after timing and retain source/data snapshots. ASCII
identity and multilingual medians favor native; the adversarial marks comparison
changes sign across repeats, so no stable advantage is claimed. Actual llama-server
has no equivalent NFC operation; the full tokenizer milestone retains the required
server comparison. This closes only the normalization building block.
