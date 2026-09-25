# KV buffers: context beyond the single 4 GiB state buffer (2026-09-24)

Question: can the KV cache leave the single ≤ 4 GiB state buffer without changing a bit, and
how much context does the card then hold?

Spec: [model.md, "KV buffers"](../specs/model.md). User request: "okay fix it".

## Why there was a cap

- Vulkan exposes at most `maxStorageBufferRange` = 4 GiB − 1 per storage-buffer binding
  (a 32-bit field). RADV on this card also caps one allocation and one buffer at 0xfffffffc
  (`vulkaninfo`: `maxMemoryAllocationSize`, `maxBufferSize`).
- zerv kept every attention layer's KV plus the recurrent state in one buffer (with a
  3.75 GiB margin), so context stopped at 29,523 tokens (27,786 with the MTP layer),
  whatever VRAM was free.

## Change

- The KV caches (16 trunk layers, then the MTP layer's) live in their own KV buffers, as
  many consecutive caches per buffer as fit 3.75 GiB (`layout.State`).
- The four kernels that touch the cache (`qkprep`, `qk_b`, `attn_scores`, `attn_pv`) and
  the prefill attention GEMM exist once per KV buffer, bound where the state arena was.
  **No shader changed**; the model modules are byte-identical.
- A command may now retain 64 distinct kernels (was 32).
- `Options.kv_capacity` forces smaller buffers (test hook). `zerv-model-capture` and
  `tools/verify_model.py --kv-capacity-mib` expose it.
- New one-cache limit: 491,520 tokens. The activation arena (prefill scores) and free
  VRAM bind first.
- The server now explains `InvalidContext` (multiple of 32) and `ContextTooLarge`.

## Verification (all executed)

| Gate | Result |
| --- | --- |
| `verify_model.py` default oracles, modes 0 and 512, default buffers ([report](data/2026-09-24-kv-buffers/verify-default.json)) | pass (same worst/bound ratios as before) |
| same, forced 24 MiB KV buffers: **6 KV buffers**, 55–83 kernels ([report](data/2026-09-24-kv-buffers/verify-split-default.json)) | pass |
| long oracle (562 tokens), modes 0 and 512, default and forced split | pass |
| **bitwise**: `logits.bin` and `tensors.bin` of every case and mode against the pre-change tool (`baseline-capture`, sha `91e8be93…`) on the same inputs, and default against the 6-buffer split | **identical, 24 of 24 files** |
| `zig build test` (Debug and ReleaseFast) 83/83; Python unittest; `zig fmt --check` | pass |

Scripts: [gate.sh](data/2026-09-24-kv-buffers/gate.sh) (interrupted after three runs by
a session interruption), [gate2.sh](data/2026-09-24-kv-buffers/gate2.sh) (the rerun of the
rest).

## Beyond the old cap (end to end)

The server was run with `--context 38400 --spec-draft 3 --prefix-cache-slots 2`: it needed
23,440 MiB of 23,789 MiB free, and the card showed 24,061 of 24,560 MiB in use. The prompt
was 37,827 tokens: a "needle" sentence, then 100,000 characters of this repository's docs,
then a question about the needle ([manifest](data/2026-09-24-kv-buffers/needle-manifest.json),
[client](data/2026-09-24-kv-buffers/needle_run.py)).

| Prefill | Speculation | TTFT (cold) | Answer |
| --- | --- | --- | --- |
| fp32 | 3 drafts | 112.3 s | correct ("ORCHID-7431 … 58") |
| f16 | 3 drafts | 82.2 s | correct |
| f16 | off | 70.0 s | correct |

- One run each; no llama-server comparison at this length yet. For scale, llama default
  took 33.3 s at 29k in an earlier report.
- **Findings for the next steps:**
  - prefill attention materializes the score matrix, O(L²): the dominant cost at this
    length (block 16a);
  - the MTP prompt catch-up (≤ 5-row passes) costs about 12 s here, 0.3 ms per prompt
    token (batched catch-up needed).

## Maximum context now (computed from the layout code, 23,759 MiB free)

With the runtime's own layout functions and its `needed` formula
([maxctx.zig](data/2026-09-24-kv-buffers/maxctx.zig)); FP32 KV, 512-row prefill chunks
unless stated:

| Configuration | Max context | Before |
| --- | --- | --- |
| no speculation, 8 snapshot slots | 38,656 | 29,504 |
| no speculation, 2 slots | 43,872 | 29,504 |
| 3 drafts, 8 slots | 35,168 | 27,776 |
| 3 drafts, 2 slots | 40,128 | 27,776 |
| 3 drafts, 2 slots, 256-row chunks | 47,488 | 27,776 |
| 3 drafts, 0 slots, 256-row chunks | 49,376 | 27,776 |

Free VRAM binds now. The next gains are lossless (the prefill score matrix, snapshots and
the token embedding off the card) and one trade-off knob (KV precision).
