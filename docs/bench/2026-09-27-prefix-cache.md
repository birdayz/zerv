# Prefix cache on the shared pool (18d.4): flat vs radix, dedup, swap sharing (2026-09-27)

Question: with `--parallel` > 1, can zerv reuse prompt prefixes (multi-turn, shared system
prompts) exactly, with the cache policy behind an interface? Do the radix policy's
deduplication and sharing-preserving swaps work under pressure? Spec:
docs/specs/concurrent.md "18d.4 design".

## Setup

RX 7900 XTX, Qwen3.8-27B-Q4_0, f16 KV, `--parallel 8`, host snapshots, 24 checkpoints.
- Binaries: `third_party/multiuser/zerv-ckpt3` (sha256 `654ff3bb…`, the 18d.4 commit) and
  `zerv-pool2` (`99686b2c…`, the pool refactor with the admission fix).
- Workload `bench/workloads/multiturn-v1.json`: a 9.6k-token system prompt, 8 conversations
  of 4 turns, `bench/run_multiturn.py`. Reference: the run with no prefix cache
  ([zerv-noprefix](data/2026-09-27-multiturn/zerv-noprefix/)).

## Correctness

| Gate | Result |
| --- | --- |
| `tests/kvcache.zig` (fake device: page contents, snapshots) | flat and radix restore exactly, return every page; radix dedup; 300 random ops keep the tree invariants |
| `tests/pages.zig` (pure page accounting, 20,000 random ops incl. aborts) | every sequence sees its own content after each op; negative control fails |
| `tests/kv_system.zig` (real batcher + pool + policies, simulated contents, reads through the tables) | exact under pressure with every policy; 3 negative controls caught (admission bug, shared partial page, pinned pages swapped) |
| `tests/batcher.zig` | 15 tests; the new "every prompt segment is admitted" reproduced the bug below before the fix |
| `zerv-batch-check … prefix` ×4 configs | 420/420 bitwise, incl. dedup rebinds and 4 swaps of sequences sharing prefix pages (only private pages moved) ([data](data/2026-09-27-kv-pool-refactor/batch-check/)) |
| `… shared`, `… swap` ×4 configs (refactored pool) | 168/168 each |
| multiturn identity (flat + radix, 1/4/8 conversations) | 104/104; after the fixes 52/52 (radix) |
| swap-pressure identity (24-page pool, 200 ms slice, prefix cache on) | 30/30 with 1,340 swaps and 21 checkpoints dropped for memory |

## Bug found and fixed (negative result)

The first swap-pressure run with the prefix cache on (the committed 18d.4 binary too) failed
3 of 30 requests with `PagesMissing`, not with wrong output. Admission covered only the first
prefill op of a prompt. Prompts split at checkpoint points had later segments run without
pages; this was masked when page rounding covered them. Fixed by admitting every prompt op
([gate data](data/2026-09-27-kv-pool-refactor/gate-swap-slice/),
[after the fix](data/2026-09-27-kv-pool-refactor/gate-swap-slice-fixed/)).

## Results

Multi-turn TTFT (one run each; the vLLM and llama-server rows are from
[their runs](data/2026-09-27-multiturn/)):

| Users | zerv no cache | zerv flat | zerv radix | vLLM APC | llama `-kvu` |
| --- | --- | --- | --- | --- | --- |
| 1, follow-ups | 7.8–8.0 s | 0.43–0.46 s | 0.43–0.46 s | 0.63–0.88 s | 0.35–0.42 s |
| 4, all turns p50 | 25–35 s | 0.52–0.59 s | 0.57–0.78 s | 0.98–1.30 s | 0.68–30.5 s |
| 8, all turns p50 | 45–80 s | 0.64–0.91 s | 0.65–0.75 s | 1.29–1.84 s | 0.80–45 s |

- **Leaf-first eviction** fixed the 8-user first turn: 51 s (LRU dropped the system-prompt
  checkpoint) → 0.65 s.
- **Cold burst** (fresh server, 8 conversations at once, 2 rounds):
  - radix deduplicated **518 pages** per run (7 × 74 system-prompt pages, ~4 GB of pool);
  - flat deduplicated 0;
  - TTFT was equal: there was no memory pressure, and the 8 cold prefills themselves
    dominate (turn 0 up to 63 s). Prefix singleflight would address that; it is parked.
- Flat and radix differ only when a prefix is prefilled twice cold; the saved memory pays
  under pressure. **Default stays flat** until a pressure benchmark shows radix's gain.
