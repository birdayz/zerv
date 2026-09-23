# Prefix cache (block 15)

Status: specified 2026-09-24 before the policy implementation. The model primitive and its
exactness evidence came first ([evidence](../bench/2026-09-24-prefix-cache.md)).
Reference behavior: llama-server b29c606e `tools/server/server-context.cpp`
(context checkpoints at 3455–3635, restore search at 3296–3398, host-RAM prompt cache at
1547–1655; see [research](../research/prefix-cache.md)).

## Goal

A request whose prompt starts with tokens the model has already processed does not
process them again. The main case is the agent loop: each step resends the whole
conversation (bruh: about 12k tokens of tools and system prompt) plus a little new text.

## Model primitive (`model.Model`)

- `Options.snapshots = N` allocates N device slots of `snapshot_bytes` = 156,893,184
  bytes each. A slot holds the recurrent state: every DeltaNet state (48 × 48 × 128 × 128
  FP32) and every convolution history (48 × 10,240 × 3 FP32).
- `saveSnapshot(slot)` copies that state, as of `position`, into the slot.
- `loadSnapshot(slot, p)` copies it back and sets `position = p`. The caller guarantees
  that the attention KV below `p` is unchanged since the save.
- The KV is not copied. It stays in the state arena, and kernels read only keys below the
  current position.
- Both are one recorded device copy with compute↔transfer barriers, synchronous.

## Policy (`session.prefix.Cache`)

State:

- **History H**: the tokens whose effect is in the model state. The KV for positions
  `[0, |H|)` and the recurrent state after `|H|` tokens both belong to H.
- **Snapshots**: pairs `(pos, slot)`, sorted, each with `pos ≤ |H|`. A snapshot holds the
  recurrent state after `H[0..pos)`.

For a request with prompt T (n ≥ 1 tokens), let `d = LCP(H, T)`:

1. **Keep:** if `0 < |H| = d < n`, the prompt extends the history. Start at `|H|` with
   no restore.
2. **Restore:** otherwise, if there is a snapshot with `pos ≤ min(d, n − 1)`, load the
   one with the largest such `pos` and start there. At least one prompt token is always
   processed, because the logits of the last one are needed.
3. **Reset:** otherwise zero the recurrent state and start at 0.
4. Snapshots with `pos > start` are dropped, and H is truncated to `start`.
5. **Snapshot points:** positions p with `start < p < n` where `T[p]` is the message
   boundary token `<|im_start|>`. The state at p covers `T[0..p)`, which is everything
   before that message.
   - The last such p is always taken. This is the start of the generation prompt, and
     any continuation of the conversation re-renders from there.
   - Earlier ones are taken in order when no snapshot exists yet (normally the end of the
     system prompt), or when p is at least `spacing` = 4096 tokens after the previous
     retained or taken point.
6. The prompt is prefilled in segments ending at the snapshot points, with a save after
   each segment. Every token that is processed (prompt and decode) is appended to H.
7. **Slot eviction:** when no slot is free, the existing snapshot i whose removal leaves
   the smallest gap `pos[i+1] − pos[i−1]` is dropped (`pos[−1] = 0`; for the last one the
   new point stands in as `pos[i+1]`). This thins the snapshots evenly and keeps both the
   early (system prompt) one and the newest ones.
8. **Failure:** any backend error (reset, restore, save, prefill, step) invalidates the
   cache: H becomes empty, all snapshots are dropped, and the next request resets.
   - Cancellation between tokens and client disconnects leave H consistent, because a
     token is appended only after its step has succeeded.

The cache is per process, so the model, weights and prefill precision are fixed. Only
tokens are compared: the chat template, tools and sampling do not affect reuse.

## Numerics

- **Restore is exact.** A run restored at p gives logits bit-identical to a run whose
  prefill was split at p ([exactness data](../bench/data/2026-09-24-prefix-cache/)).
- **A split prefill is not always bit-identical to an unsplit one.**
  - The GEMM plan (split-K) is chosen by chunk row count, so a row that lands in a short
    chunk is summed in a different order.
  - The measured difference is ≤ 5.3e-6 absolute on the logits, with the same argmax.
  - Split prefills pass the FP64 model-oracle gates, like any other chunking.
- **Keep reuses decode-path KV** for previously generated tokens. That KV was produced by
  the matvec decode kernels, not the prefill GEMMs, which is also what llama-server does.

## Interface

- `zerv --prefix-cache-slots N` (default 8; about 150 MiB of VRAM each). `0` disables
  the cache: every request resets, as before block 15.
- Responses carry `usage.prompt_tokens_details.cached_tokens` = the start position (the
  OpenAI and llama-server field).
- Metrics:
  - `zerv_prompt_cached_tokens_total`;
  - `zerv_prefix_cache_requests_total{outcome="keep|restore|reset"}`.
- One log line per request: the outcome and the reused token count (no token content).

## Not in this block

A host-RAM tier (llama-server `--cache-ram`, 8 GiB by default) that saves a whole
conversation's KV and state when another conversation takes over the slot. With one
slot and alternating conversations, zerv reuses only their shared prefix (the system
prompt boundary snapshot). This will be decided from the benchmark.

## Acceptance

1. **Model primitive:** `zerv-prefix-check` on the real model shows restore bit-identical
   to split at 11 positions (including off the chunk grid), keep bit-identical, and run
   repeats bit-identical.
2. **Oracle:** split prefills (`CHUNK:SPLIT` modes) pass the FP64 gates on both the
   default and the long oracle.
3. **Policy tests** (CPU, `tests/prefix.zig`):
   - unit cases for keep, restore, reset, points, spacing, eviction, invalidation,
     cancellation and the one-token minimum;
   - a randomized differential test in which a fake backend's logits depend on the whole
     KV content and the recurrent state. Cached request sequences must produce exactly
     the output of uncached runs, and every restore must satisfy the KV guarantee.
4. **Serving:** real-model multi-turn requests with the cache on and off. The response
   texts are compared and cached_tokens is checked.
5. **Benchmark against llama-server:** a replay of a recorded bruh session (full tool
   set) and a fresh-session case. The llama-server configuration is its defaults,
   including prompt cache and checkpoints. Per-request TTFT and cached tokens are
   reported. Target: warm-request TTFT on par with or better than llama-server.
