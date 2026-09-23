# Native GGUF validation — 2026-09-22

The selected **complete** artifact was downloaded to `.part`, then verified against
publisher SHA-256 before rename:

- `models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf`
- `unsloth/Qwen3.8-27B-GGUF` revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`
- 16,056,478,688 bytes
- SHA-256 `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`

The [repeatable loader harness](../../bench/run_gguf.py) independently hashed the
complete file again and compared native/reference inspection **field for field**:
51 metadata value hashes, all 866 names/types/shapes/offsets/sizes and every tensor's
first/middle/last sample hash. Exact agreement. Data starts at 10,996,704 (alignment
32); total tensor bytes 16,045,481,984. Native independently bounds the entire tensor
payload/final padding, not just reference metadata-only parsing. Evidence:
[full inventory](../bench/data/2026-09-22-gguf-qwen38/container.json) and
[manifest](../bench/data/2026-09-22-gguf-qwen38/manifest.json).

| Wire type | Tensors | Payload bytes |
| --- | ---: | ---: |
| F32 | 456 | 105,058,304 |
| Q4_0 | 352 | 13,358,039,040 |
| Q4_1 | 8 | 445,644,800 |
| Q5_K | 48 | 1,038,090,240 |
| Q6_K | 1 | 1,042,944,000 |
| Q8_0 | 1 | 55,705,600 |

The Q8_0 matrix is in the MTP block. `qwen35.block_count=65` and
`qwen35.nextn_predict_layers=1`: base count is **64**, not 65. `blk.64.*` contains
15 tensors totaling 265,197,568 bytes; initial execution must skip this optional
block. Output projection is Q6_K; 48 recurrent output projections are Q5_K; eight
FFN down projections use Q4_1. Remaining decoders/kernels are still required.
[Readable scalar metadata](2026-09-22/qwen38-gguf-metadata.json) also records actual
tokenizer IDs, array counts and the embedded template. Those fields were first
extracted while the download was in progress; complete-file comparison now anchors
their source identity, rather than treating partial weights as validated.

## Native gates executed

Debug and ReleaseFast both passed all 12 tests at loader completion (14 after chat
template tests were added). Loader coverage includes all three external fixtures,
every truncated prefix, typed getters and arrays, UTF-8/bool/tags, duplicates,
shape/rank/alignment/offset/overflow/limits, every allocation failure, and mmap
ownership/error behavior. Custom-index limits are additionally bounded below the
u32 hash-map capacity overflow threshold. The inspector was separately checked
against every fixture's reference JSON, not only against our own parser objects.

Container loading does not numerically validate tensors, allocate GPU storage,
implement tokenization/model graph, or expose a serving endpoint. The loader's
[component benchmark](../bench/2026-09-22-gguf-loading.md) is not inference speed.
