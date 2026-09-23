# Native GGUF loader specification

Research: [gguf-loading.md](../research/gguf-loading.md). Implementation readiness
requires the specified external writer/reader fixtures to be generated successfully
before native parser code. This boundary is container loading, not model execution.

## Public boundaries

- `artifact.gguf.Container.parse(allocator, immutable_bytes, limits)` validates a
  supported GGUF and owns only metadata/tensor indexes. `deinit` frees those indexes.
  Values/names/tensor data borrow input bytes; caller must preserve immutable input
  until all views and the container are discarded. Parser does no I/O or GPU work.
- `artifact.MappedFile.open(io, path, max_bytes)` owns a read-only private Linux
  mapping of a regular file. `deinit` unmaps exactly once. No whole-file read/copy or
  implicit reference/backend dependency. File must remain unmodified while mapped.
- Metadata retains exact tags and encoded bytes, with checked scalar/string/array
  accessors. Array iteration is linear and bounded; no per-element allocation.
- Tensor exposes name, original rank, four extents (unused extents 1), format,
  relative offset, byte size and immutable payload slice. Format IDs match GGUF,
  not a compiler-inferred enum ordinal. `findMetadata`/`findTensor` use owned indexes.
- A standalone native `zerv-inspect` reports JSON descriptors/hashes from a mapped
  file for comparison. It must fail nonzero on any malformed/unsupported input.
  It does not advertise health/readiness or pretend to generate model output.

## Supported profile / rejection

GGUF v3 little endian, nonempty ASCII hierarchical keys (lowercase/digits/underscore
segments), unique metadata keys and tensor names. Strings are UTF-8; keys/names
contain no NUL. Metadata IDs 0..12, all scalar types and homogeneous non-nested
arrays. Unknown metadata keys are retained. Unknown/nested value types are rejected.

Tensor types: F32=0, F16=1, Q4_0=2, Q4_1=3, Q8_0=8, Q5_K=13, Q6_K=14, BF16=30.
These are **layout** capabilities, not a promise of kernels/decoders for every type.
Rank 1..4, positive dimensions fitting i64, quantized dimension 0 a multiple of
block width. Validate all count/product/rounding/offset arithmetic without wrapping.

Default alignment 32; override must be U32 and a positive power of two within the
limit. Header/tensor offsets must match the supported contiguous padded layout;
no gaps, overlap, reordering, or truncated final payload/padding. No tensors means
no required metadata padding. Trailing file data is tolerated but not tensor data.

Defaults: at most 16,384 tensors, 65,536 metadata pairs, 64 MiB parsed header,
16 MiB per string, 16,777,216 array entries and 1 MiB alignment. Key ≤65,535 bytes,
tensor name 1..63 bytes. Limits are explicit caller-supplied values, checked before
allocation/work. File mapping default cap is specified by the application, not
hidden in generic parser code.

Errors distinguish truncated input, unsupported version/value/tensor type/profile,
invalid key/string/bool/rank/dimensions/alignment, duplicate names/keys, arithmetic
overflow, resource limits, invalid tensor offsets and missing payload. On every
failure all parser allocations are released; bytes are never modified. No caller
may use a partial container after an error.

## Reference acceptance and repeatability

`tests/reference/gguf_oracle.py` is external tooling only. It creates small files
using the pinned reference writer and reads them back with its reader, preserving
version/dirty commit/library and generator hashes. Ordinary Zig tests consume
committed binary/JSON goldens without this script/library. For large artifacts,
its inspect mode and native inspection must agree on all metadata/tensor fields
and payload sample hashes. Full artifact SHA/size must match the selected pin.

Both Debug/ReleaseFast: golden equality, all-prefix truncation tests, structured
malformations, custom/default alignment, every metadata type/empty array,
allocation failure cleanup, zero-copy/lifetime invariants, wrong typed accessor
errors and array iteration. Repeatable component benchmark must record workload /
source hashes, warm/cache state, validation/ownership asymmetries and raw trials.
No load/generation compatibility claim is allowed from only filename/header checks.
