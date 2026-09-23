# Qwen3.8-27B research

**Inspected 2026-09-22; selected artifact downloaded, SHA-verified and natively
container-loaded. External Vulkan generation succeeded; native execution is absent.** This is a real official
model, not a presumed typo for Qwen3.5. Resolve identifiers from metadata rather
than inferring architecture from a marketing version number.

## Identity and source of truth

- Official repository: `Qwen/Qwen3.8-27B`.
- Inspected revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Model card license: Apache-2.0; native vision-language dense model.
- `architectures = ["Qwen3_5ForConditionalGeneration"]`, `model_type = "qwen3_5"`.
- API reports 27,781,427,952 BF16 parameters across the published artifact; this is
  not the exact parameter count of a text-only GGUF or the bytes read per decode.
- [Config snapshot](research/2026-09-22/model-config.json) and
  [generation config](research/2026-09-22/generation-config.json) are retained.

## Language architecture

| Property | Value from official card/config |
| --- | --- |
| Hidden size / FFN intermediate size | 5,120 / 17,408 |
| Vocabulary | 248,320 padded; input and output embeddings are not tied |
| Decoder layers | 64: **48 Gated DeltaNet + 16 full gated attention** |
| Repeating layout | Three linear/recurrent layers then one full attention layer |
| Full attention | 24 query heads, 4 KV heads, head dimension 256 |
| RoPE | 64 rotary dimensions; partial factor 0.25; theta 10,000,000 |
| MRoPE | Interleaved; sections `[11, 11, 10]` |
| Linear attention | 16 Q/K heads and 48 V heads, head dimensions 128 |
| Convolution | Kernel width 4 |
| Recurrent state dtype | `mamba_ssm_dtype = float32` |
| Norm epsilon / FFN activation | 1e-6 / SiLU |
| Native context ceiling | 262,144 tokens; not an affordable default on this card |
| MTP | One extra hidden layer declared; disabled for initial bring-up |

Consequences: this is **not** ordinary Llama attention and not an MoE. DeltaNet
recurrence, its convolution history, grouped heads, gated normalization, partial
RoPE, and attention output gating all need independent fixtures. There is no
sparse-expert shortcut for weight bandwidth. Prefix reuse, branching, and
speculative rollback must restore recurrent state as well as attention KV.

Model config includes `output_gate_type = "swish"`. Inspected Transformers
`modeling_qwen3_5.py` and llama.cpp `qwen35.cpp` apply **sigmoid** to the full
attention output gate; llama.cpp uses SiLU for the linear gated normalization.
This is an **unresolved semantic mapping question**, not proof of an upstream bug.
Before implementing the graph, trace the exact checkpoint's reference execution
and prove which operation this field controls (or whether it is unused). Do not
change sigmoid to SiLU on the basis of the field name alone. This blocks declaring
model-execution research complete. [Block09 findings](research/gpu-primitives.md)
now trace the unconditional source-level gate choices, converted norm/A parameters
and GDN L2 epsilon placement; executable intermediate validation is still pending.

## Tokenization, templates, and generation

- The official tokenizer config carries the template. Implement its supported
  semantics exactly; do not substitute a generic ChatML string formatter.
- Thinking defaults on; `reasoning_effort` supports `xhigh` (default), `medium`,
  and `low`. Historical thinking is preserved by default (`preserve_thinking`).
- Official template enforces role/order/content constraints and includes tool and
  multimodal paths. The text-only milestone must reject unsupported branches,
  not silently drop them. Decide/document developer-role behavior explicitly.
- Generation config specifies EOS IDs **248046 and 248044**, whereas text config
  lists 248044. Define termination from the applicable generation/tokenizer
  metadata and reference behavior; don't hard-code a single EOS from one file.
- Recommended thinking sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0,
  presence penalty 0, repetition penalty 1.0.
- Recommended non-thinking sampling: temperature 0.7, top-p 0.80, top-k 20,
  min-p 0, presence penalty 1.5, repetition penalty 1.0.

These are upstream recommendations, not a license to change benchmark sampling.
Deterministic greedy fixtures are a separate correctness mode. Tokenizer files,
merge/pretokenization rules, byte handling, and rendered prompt goldens still need
full extraction and study before writing our tokenizer/template implementation.

## Candidate artifacts (only selected Q4_0 downloaded/verified)

Publisher: `unsloth/Qwen3.8-27B-GGUF`, revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`. Full candidate metadata in
[quant-artifacts.json](research/2026-09-22/quant-artifacts.json).

| File | Bytes | GiB |
| --- | ---: | ---: |
| `Qwen3.8-27B-Q4_0.gguf` | 16,056,478,688 | 14.954 |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | 15.334 |
| `Qwen3.8-27B-UD-Q5_K_M.gguf` | 19,771,509,664 | 18.414 |
| `Qwen3.8-27B-UD-Q6_K.gguf` | 21,983,677,344 | 20.474 |
| `Qwen3.8-27B-Q8_0.gguf` | 29,047,086,048 | 27.052 |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1,369,590,656 | 1.276 |
| `mmproj-F16.gguf` | 927,607,488 | 0.864 |

Start artifact evaluation around Q4; Q5 is a quality/capacity experiment, not an
assumed fit for every context. The selected Q4_0 artifact is now verified; see [actual tensor inventory and
MTP accounting](research/2026-09-22-gguf-validation.md). Quantization suffixes are not tensor-type manifests:
inspect every tensor's actual type, shape, alignment, and bytes. Dynamic/mixed
quantization needs multiple decoders. Publisher quality claims are unverified here.
Their card advertises template changes (developer-role and tool behavior), so
compare embedded templates with the official template rather than assuming parity.
Separate weight equivalence from prompt-rendering equivalence.

GGUF is an interchange format, not a dependency on ggml. Our reader/decoders must
be written from the format specification with independent compatibility tests.
Choose supported quant types only after artifact inspection. A repacked native
layout may follow, with conversion correctness checks and cache invalidation by
artifact hash/layout version. Do not download BF16 + all quants just to initialize
the project.

All source URLs/revisions are in the [ledger](research/2026-09-22/sources.json).
