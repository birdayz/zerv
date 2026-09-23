# Qwen text pretoken splitter

Scope: `tokenizer.qwen_split`, a single allocation-free package building block.
[Completed scoped research](../research/pretokenization.md) defines branch order,
Unicode version and the independent oracle. It is **not a complete tokenizer**.

## Interface and ownership

`Iterator.init(input: []const u8, limits: Limits) !Iterator` validates the complete
UTF-8 input and byte limit once. Default `max_input_bytes` is 1 MiB; callers may
set another bound, including zero. Over-limit input returns `InputTooLarge` before
UTF-8 validation; malformed input returns `InvalidUtf8`. No I/O or allocation.
Input must remain alive and unchanged for the iterator lifetime. Separate iterators
are independent; immutable data is shared and requires no synchronization.

`next() ?[]const u8` returns the next nonempty borrowed contiguous byte slice in
left-to-right order. Concatenating pieces yields the original input exactly. Empty
input and exhausted iterators return null; repeated exhaustion is harmless. Work
is linear in input length with bounded property lookup per visited scalar, including
long whitespace runs; there is no regex recursion or quadratic backtracking.

Export `properties(cp: u21)` for the shared tokenizer classifier: immutable boolean
letter/mark/number/space flags. For non-scalars it returns no flags; input iteration
never admits them. Property version is explicitly Unicode 16.0.0, unlike NFC 9.
Expose the immutable table's SHA-256 for provenance checking.

Recognize the exact official fixed pattern documented in the research; do not
strip, normalize, add prefix space, split added tokens specially, translate byte
alphabets or perform BPE. The later encoder supplies normalized ordinary spans
*after* added-token extraction. This component itself accepts any valid UTF-8,
including unassigned/private-use/noncharacters/NUL. No locale or mutable modes.

## Data and oracle contract before implementation

Generate data from normative Unicode-16 UnicodeData/PropList, not reference code:
magic `U16C`, u32 row count, then sorted {start:u32, flags:u32} little-endian rows.
Flags occupy low bits L=1/M=2/N=4/White_Space=8. Each row starts a maximal interval;
row zero starts at zero; omitted categories have zero flags. Include a zero-flags
sentinel at 0x110000. Preserve the Unicode data license. ASCII lookup can be
precomputed from this same data at compile time.

The generator checks **every scalar** against HF 0.22.2 Regex Split class membership
and records the independent one-byte-per-scalar property SHA-256. Full pattern
expectations come solely from HF Split using the official tokenizer JSON. Generator
must verify isolated/non-inverted settings and reject existing output paths.
Fixture record format: input byte count:u32, piece count:u32, input UTF-8 bytes,
then one u32 exclusive byte-end per piece. Require nonempty, contiguous HF pieces
whose concatenation equals input; do not use HF's character offsets as byte ends.

Include all strings of length 0–5 over `a`, `1`, space, tab, CR, LF, `!`, U+0301;
all CaseFolding input scalars in contraction contexts; a fixed seed with multilingual
and property-boundary samples; official chat fixture outputs; long whitespace and
combining sequences. Manifest pins sources, official JSON, generator and output
hashes, oracle extension/version, corpus count and fixed performance workloads.
No reference library, model artifact or `third_party/` dependency in native tests.

Acceptance: exact independent split ends and exhaustive class fingerprint in Debug
and ReleaseFast, strict malformed UTF-8/limit tests, iterator lifetime/exhaustion
behavior, provenance hashes, and byte-identical regeneration. No heap allocations
in the implementation; tests may allocate scratch to inspect fixture outputs.

## Component benchmark contract

Use fixed ASCII, multilingual and adversarial whitespace workloads from the
independent manifest. Compare native iterator construction plus complete traversal
to HF `Split.pre_tokenize_str` with the official regex only. No NFC/ByteLevel/BPE.
Native consumes each returned slice with a compiler barrier; HF materializes a list
of strings and offsets. Disclose that ownership/offset-tracking asymmetry. Full ends
and their SHA-256 are checked before and after each trial, outside timing.

Pinned CPU; three warmups, seven trials per process, three alternating engine-order
rounds, 100 iterations per workload/trial. Timers: Zig `std.Io.Clock.awake` and Python
`perf_counter_ns`, previously researched for component harnesses. Report per-call
median/min/max/sample deviation and all raw trials. Run native correctness suites
first; record host, compiler/native/reference binary hashes, package versions,
commands, fixture and complete source/data/license snapshots. Reject malformed or
incomplete trial sets; unit-test that validator. Rerun with a fresh output directory.

No equivalent isolated operation is exposed by llama-server: standalone splitter
vs full HTTP tokenization is not a valid speed comparison. Actual `/tokenize` timing
and exact-ID gates remain mandatory at the complete tokenizer milestone in
[TODO.md](../../TODO.md), followed by separate model/serving comparisons.
