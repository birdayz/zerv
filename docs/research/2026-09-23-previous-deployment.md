# Ollama long-context configuration reference (2026-09-23)

Purpose: preserve previously observed technical settings as a reference for later
long-context and KV-precision work. This is not a benchmark result. Personal account,
recovery-storage and shell-history details have been removed.

## Configuration and public source

The observed configuration used:

- `OLLAMA_LLM_LIBRARY=vulkan`
- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_KV_CACHE_TYPE=q4_0`
- `OLLAMA_KEEP_ALIVE=1h`
- a per-model context configuration, used at about 90–128k context.

The Ollama registry manifest for `library/qwen3.8:27b`, fetched 2026-09-23 as
metadata only ([copy](2026-09-23-ollama-qwen38-27b-manifest.json)), lists:

- the model layer: 16,810,714,464 bytes, `from: qwen3.8:27b-q4_K_M`;
- a 931 MB vision projector.

## Findings and limitations

- The model was most likely **Q4_K_M** (Ollama's default `27b` tag). The tag may have
  been re-pointed; this was not verified against the original deployment artifact.
- The KV cache was **4-bit (q4_0)** with flash attention. That is what fit about
  90–128k context in 24 GB next to about 17 GB of weights.
- zerv currently serves the Q4_0 artifact with an FP32 KV cache, capped at 8192
  context by default.
- Long-context capacity at this level would need an explicit lower-precision KV option
  (F16/Q8/Q4) with measured quality. Not scheduled yet.
