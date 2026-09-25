# `--context max`: the largest context per memory configuration (2026-09-24)

Question: how much context fits on this card with each combination of memory knobs, and
does the server work at that context? [Spec](../specs/model.md) (`--context max`).

## Setup

- RX 7900 XTX (24 GiB), Mesa 26.2.3 RADV; 769 MiB of VRAM in use by the desktop before
  each start; the driver reported 23,789 MiB free.
- zerv binary sha256 `862df2ab5be37fde…` (run2; full hash in the results file).
- `python3 bench/context_max.py --output docs/bench/data/2026-09-24-context-max/run2`:
  starts `zerv --context max` per configuration, records the chosen context, zerv's
  VRAM accounting and the VRAM in use (sysfs), and serves one short request.
- FP32 KV cache (128 KiB per token for the 16 attention layers, plus 8 KiB with MTP).

## Results ([raw](data/2026-09-24-context-max/run2/results.json))

| Configuration | Flags beyond `--context max` | Context | VRAM in use MiB | Served |
| --- | --- | --- | --- | --- |
| default (3 drafts, 8 device snapshots, host embedding) | | 44,896 | 23,286 | yes |
| no speculation | `--spec-draft 0` | 50,592 | 23,280 | yes |
| host snapshots | `--prefix-cache-memory host` | 53,760 | 23,283 | yes |
| no prefix cache | `--prefix-cache-slots 0` | 53,760 | 23,283 | yes |
| host snapshots, 256-row chunks | `--prefix-cache-memory host --prefill-chunk 256` | 55,456 | 23,282 | yes |
| no prefix cache, no speculation | `--prefix-cache-slots 0 --spec-draft 0` | 60,128 | 23,280 | yes |
| device embedding | `--embedding-memory device` | 39,808 | 23,283 | yes |
| default, no reserve | `--vram-reserve-mib 0` | 52,000 | 24,244 | yes |
| no prefix cache, no speculation, no reserve | `... --vram-reserve-mib 0` | 67,808 | 24,244 | yes |

- The chosen contexts use all the room they are given. VRAM in use is 769 MiB (desktop)
  plus zerv's computed need minus the unallocated 256 MiB headroom, within 10 MiB.
- With the default reserve, 1.2 GiB stays free (24,560 MiB card); with reserve 0, about
  300 MiB. Filling the card is the user's choice. Another process that grows past what
  is free oversubscribes VRAM, which has ended in a lost device here before (`TODO.md`,
  block 15 incident).
- The trade-offs, per GiB of VRAM: about 7,100 tokens of context (with MTP) or 7,650
  (without). Measured costs of the knobs:
  - `--spec-draft 0`: loses the 2.1–2.4× decode speedup ([report](2026-09-24-speculative.md));
  - `--prefix-cache-memory host`: 10–19 ms TTFT; `--prefix-cache-slots 0`: no prompt
    reuse between requests ([host memory](2026-09-24-host-memory.md));
  - `--prefill-chunk 256`: prefill speed not measured in this run.
- An f16 KV cache would halve the 128 KiB per token (block 17c, next).

## Limitations

- Only loading and a short request were run at these contexts. The longest prompt served
  so far is 37.8k tokens ([flash report](2026-09-24-flash-attention.md)); a
  prompt near 60k tokens has not been run.
- The free-VRAM number is a snapshot at startup.
