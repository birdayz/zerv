# Prefix caching for a hybrid (attention + DeltaNet) model: llama-server's design (research, 2026-09-24)

Question: how does llama-server reuse earlier work across requests for Qwen3.8 (16
attention layers, 48 recurrent DeltaNet layers), and what must zerv match? Feeds the
[prefix-cache spec](../specs/prefix-cache.md).

## Source

llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (installed build 10964),
`tools/server/server-context.cpp` (git blob `b6835e43459e`), under
`third_party/llama.cpp/<commit>/`. The option defaults below are from `llama-server --help`
of the same build.

## Why a hybrid model needs checkpoints

The attention KV can be truncated to any prefix length. The recurrent state is a single
running summary: it cannot be rewound, even by one token. llama.cpp calls this
`COMMON_CONTEXT_SEQ_RM_TYPE_RS` / FULL, and handles it with **context checkpoints**:
copies of the parts of memory that cannot be rolled back, taken at chosen prompt
positions (3465–3473).

## What llama-server does (source-backed)

- **Slot reuse** (3250–3398):
  - The new prompt is compared with the slot's cached tokens (longest common prefix,
    `n_past`).
  - If the recurrent state is past the usable point, it searches its checkpoints from the
    newest for one whose position is at or before the common prefix, and restores it.
  - Without one, it reprocesses the whole prompt ("forcing full prompt re-processing due
    to lack of cache data").
  - At least one prompt token is always evaluated, for the logits (`[TAG_PROMPT_LOGITS]`).
  - Checkpoints past the new position are erased.
- **Keep:** when the prompt extends the cached tokens exactly, nothing needs restoring.
  The live state already ends at `n_past`.
- **Checkpoint creation** (3549–3634). A batch is broken so that a checkpoint can be
  taken before processing:
  - at the start of the **last user message** of the prompt;
  - at user-message starts at least `--checkpoint-min-step` (default 8192) after the
    previous checkpoint, or at the first one when none exists;
  - **4 + n_ubatch** and **4** tokens before the end of the prompt.

  Message starts come from the template's message delimiters (`<|im_start|>user`, and
  `<|im_start|>user\n<tool_response>` for tool results; `qwen3-coder.cpp`).
- **Limits:** at most `--ctx-checkpoints` (default 32) per slot. Checkpoints within
  min-step of an earlier one are erased first, then the oldest (2309–2372).
- **Host-RAM prompt cache** (`--cache-ram`, default 8192 MiB; 1547–1655):
  - When a task would drop more than half of a slot's cached tokens, or no slot matches
    by LCP similarity above `--slot-prompt-similarity`, the slot's full state (KV,
    recurrent state and checkpoints) is saved to host RAM.
  - The best-matching saved prompt is then loaded.
  - This covers switching between conversations.

## Consequences for the agent loop (bruh)

- **Where the next request diverges.** bruh does not send reasoning back, and the
  template renders a history assistant turn as `<think>\n\n</think>\n\n…`. The prompt of
  step k ends `<|im_start|>assistant\n<think>\n`, so step k+1 diverges one token before
  its end.
  - llama restores its "4 before the end" checkpoint.
  - zerv needs a snapshot at or before `n − 1`. The start of the generation prompt (the
    last `<|im_start|>`) works for every continuation of the conversation.
- **Thinking off, or clients that return the generated text verbatim.** The next prompt
  can extend the live state exactly. That is keep, with no restore.
- **A new session with the same system prompt and tools.** It diverges inside the first
  user message. A snapshot at the first message boundary after the system block reuses
  the whole tools prefix (about 12k tokens for bruh).

## Measured model facts (zerv)

See [exactness data](../bench/data/2026-09-24-prefix-cache/) and the
[report](../bench/2026-09-24-prefix-cache.md). Snapshot restore is bit-exact. A prefill
split at a position off the plan boundaries changes GEMM summation order (split-K plans
depend on chunk rows) by ≤ 5.3e-6 on logits, and passes the FP64 oracle gates.
