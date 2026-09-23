# Native GGUF loading research — 2026-09-22

## Sources and scoped understanding before implementation

Pinned ggml `456172ec733a135778adcd32d00e576a58232e45`:
`docs/gguf.md`, `include/gguf.h`, `include/ggml.h`, `src/gguf.cpp`, `src/ggml.c`.
Stable local root: `third_party/ggml/456172ec733a135778adcd32d00e576a58232e45/`.
URL prefix: `https://raw.githubusercontent.com/ggml-org/ggml/456172ec733a135778adcd32d00e576a58232e45/`.
Block sizes also checked in the retained `oracle-ggml-common.h`. Reference binary
remains ggml 0.24.0 / `456172ec-dirty`; binary hash is authoritative (see quant goldens).

GGUF v3 is byte-sequential: literal `GGUF`, u32 version, u64 tensor count, u64 KV
count; KV pairs (length-prefixed key, u32 type, typed value); tensor descriptors
(length-prefixed name, u32 rank, u64 dimensions, u32 tensor type, u64 relative data
offset); padding to alignment; tensor bytes. Strings use u64 byte lengths, no
terminating NUL. Numeric values are little-endian for our supported profile. Rank
is 1–4 for this loader; dimension 0 is the contiguous/quantized row width. Tensor
byte size is `(ne[0]/block_elements)*block_bytes*product(ne[1..])`, with checked
arithmetic. Do not infer precision from the filename or offsets from neighboring
names. Data offsets are relative to the aligned data section, not file start.

Metadata type IDs 0..12 cover unsigned/signed integers of 8/16/32/64 bits, f32/f64,
1-byte bool, string and homogeneous arrays. Array encoding includes u32 element
type and u64 count. Numeric array data is packed; strings remain length-prefixed.
Booleans must be 0 or 1. Unknown metadata keys can be retained without interpreting
model semantics; unknown type IDs cannot be safely skipped. Float bit patterns
are container data; model-specific finite-value constraints belong in model validation.

## Explicit spec/reference differences

- GGUF prose permits nested arrays, but the pinned C reader rejects them. Initial
  native scope explicitly rejects nested arrays, not silently misparses them.
- Format defaults to little endian; other byte order is explicitly unsupported.
- Reference tensor names must be **<64 bytes**, though prose says at most 64.
  Native profile follows <64. Empty/zero-dimensional/zero-sized tensors are not
  useful for this model and are explicitly rejected in our inference profile.
- Reference requires descriptor-order contiguous padded offsets; native initially
  supports that profile. Gaps/reordering/overlap are rejected explicitly.
- `general.alignment` is U32, positive power of two (default 32), including small
  powers in the reference. Bound the accepted value by configured limits.
- With no tensor, the reference does not pad the end of metadata. With tensors,
  header and every tensor allocation are padded. Native requires complete payloads
  including final padding; no header-only file is considered a loaded model.
- Reference metadata-only mode does **not** establish complete tensor payloads.
  Our loader must check actual file length and each tensor range independently.

## Actual artifact selected and preliminary inventory

Explicit download: `unsloth/Qwen3.8-27B-GGUF` revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, `Qwen3.8-27B-Q4_0.gguf`,
16,056,478,688 bytes, SHA-256
`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
Local path `models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf`; write `.part`, verify full
SHA before renaming. Disk/memory capacity checked; no system package changes.

An exploratory **header-only reference read during download** found 866 tensors,
51 metadata pairs, and these actual types: 456 F32, 352 Q4_0, 48 Q5_K, 8 Q4_1,
1 Q6_K and 1 Q8_0. In particular output.weight is Q6_K and recurrent output matrices
are Q5_K. This confirms the filename alone is insufficient. This preliminary read
is not evidence of a complete download or native payload loading.

Supported layout sizes for the initial loader (decoding is a separate capability):
F32 1/4, F16 1/2, BF16 1/2, Q4_0 32/18, Q4_1 32/20, Q8_0 32/34,
Q5_K 256/176, Q6_K 256/210 (elements/bytes). Other tensor types fail explicitly.

## Ownership, resource bounds and mapping

Parser takes immutable bytes and returns borrowed metadata/tensor views plus owned
bounded indexes. No tensor copy/dequantization or GPU allocation. Validate string/
array/count/metadata budgets, checked sizes, unique keys/names, alignment and all
payload bounds before exposing a successfully loaded container. Numeric and string
array views iterate without allocating an object per token/merge.

Separate file storage uses read-only private `std.posix.mmap` on the target Linux
machine; inspected Zig 0.16.0 `lib/std/posix.zig`, `Io/File.zig`, `Io/Dir.zig`.
Open regular file, stat/bound length, map without prefaulting, close descriptor;
unmap only after borrowed views/indexes are released. No libc/C++ import required.
The file must remain unmodified/untruncated for the mapping lifetime: MAP_PRIVATE
is not a snapshot, and external truncation can cause SIGBUS. Reject non-regular /
empty/oversized files; do not silently allocate/copy the entire model as fallback.

## Executable reference-validation mechanism

A separate Python `ctypes` runner calls the installed gguf **writer and reader**.
The inspected `gguf_init_params` ABI is `{ bool no_alloc; void *ctx; }`; pass
`{true,nullptr}` to avoid allocating tensor bodies. Bind metadata getters only
after checking tags, and tensor dimensions through the returned four i64 extents.
Keep contexts/buffers alive until synchronous calls complete and always free them.

Generate small files with the reference writer (all metadata scalar types, numeric
and string arrays, empty arrays, custom/default alignment, all supported tensor
types), then reopen with its reader. Golden JSON records metadata type and SHA-256
of canonical serialized value bytes, tensors' padded shapes/type/relative offset/
byte size, data start/alignment and sampled payload hashes. Samples concatenate
first up-to-64 bytes, midpoint up-to-64, and last up-to-64 bytes of each tensor;
overlap for small tensors is intentional. On the actual artifact compare every
descriptor and metadata value hash plus every tensor's sample against native views.
Full file SHA separately anchors the weight identity.

Native unit tests use data-only small fixtures, not the reference library. Add
truncation at every byte of a valid fixture, overflow/count/length/alignment/tag /
shape/duplicate/offset mutations, budget exhaustion and allocation-failure tests.
Test mapping lifetime/failure cleanup separately. A valid container is not yet a
validated Qwen execution plan. Benchmark parse/index creation separately from file
hashing/mapping, and compare reference metadata-only parsing honestly (different
payload validation/copying contracts). No model-serving claim follows from loading.

### Oracle bring-up finding: alignment setter is not context configuration

The first custom-64-alignment fixture failed reference reread:
`tensor 'tensor.type_1' has offset 32, expected 64`. Inspection confirmed
`gguf_set_val_u32("general.alignment",64)` changes metadata but not the writer
context's internal alignment. This is a reference-tool API pitfall, not a reason
to weaken the native offset check. Failed reference-written files are retained in
`third_party/gguf-writer-alignment-failure/`.

Resolution: construct the custom-alignment reference context by **reading a minimal
metadata-only v3 buffer** with that alignment first, then add metadata/tensors and
write with the reference writer. The reader initializes its alignment field. This
uses inspected public APIs, not private C++ layout access. Reopen/check every output
before allowing it to become a golden.

### Reference fixture gate passed before native parser implementation

Executed `python3 tests/reference/gguf_oracle.py --library
/usr/lib/libggml-base.so.0.24.0 fixtures --output tests/fixtures/gguf` successfully:

- `default.gguf`: 2,592 bytes, alignment 32, data offset 1,440, 25 KV / 8 tensors.
- `aligned64.gguf`: 2,752 bytes, alignment 64, data offset 1,472, 26 KV / 8 tensors.
- `metadata.gguf`: 992 bytes, reader data offset 975, 25 KV / 0 tensors.

The writer pads even the metadata-only file, whereas reader offset excludes its
trailing padding. Native parsing therefore requires bytes through 975 in that
case and tolerates the extra 17 bytes. Golden manifests contain actual reader
metadata hashes, dimensions/types/offsets/sizes and file-backed payload samples.
