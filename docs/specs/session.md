# Generation session — `src/session` (block 11)

One sequence at a time over the resident model (`src/model`). Backend-agnostic
(`Generation(Backend, Tokenizer, Sink)`), so control flow is unit-tested with CPU
logits and verified end-to-end against the libllama greedy oracle.

## Flow and bounds

`reset → prefill(prompt) → loop { sample; EOS? ; emit; step }`.
Prompt must be non-empty and shorter than the model context; generated tokens are
capped at `min(max_tokens, context − prompt)` (`finish=length`). The final token
is never stepped. `prefill` runs the prompt through batched chunks
([prefill spec](prefill.md)); a backend with chunking disabled (`--prefill-chunk 0`)
falls back to one decode step per prompt token.

## Sampling (`sampler.zig`)

Order: presence/frequency/repetition penalties (generated tokens only, OpenAI
semantics) → top-k → top-p → min-p (on untempered probabilities, llama.cpp
order) → temperature → categorical draw with Xoshiro256++ seeded per request.
Temperature 0 without penalties is exact argmax, lowest id on ties. Parameter
ranges are validated; non-finite logits are an error. Draws are not bit-compatible
with llama.cpp's RNG: only greedy output is compared token-for-token.

## Termination and text

- EOS set resolved from the tokenizer: `<|im_end|>` (248046) and `<|endoftext|>`
  (248044); the EOS token counts as a completion token and is not emitted.
- Output text matches llama-server's pipeline for this model: the Qwen3-Coder PEG
  parser at the pinned oracle commit. Research and source lines:
  [output parsing](../research/output-parsing.md). The steps are, in order:
  1. **Render.** Each non-EOS token renders as `llama_token_to_piece(special=false)`
     does: control tokens as nothing (`Tokenizer.outputPiece`), user-defined tokens
     (`<think>`, `</think>`, `<tool_call>`, …) as their text.
  2. **UTF-8.** A streaming UTF-8 assembler replaces invalid sequences with U+FFFD
     per maximal subpart. This is verified against CPython `errors="replace"` for
     every split point.
  3. **Stops.** One stop-string matcher (≤4 strings, ≤64 bytes) runs on the **raw**
     text, which includes `</think>` and whitespace. The earliest match wins; the
     held-back bytes are released at the end.
  4. **Split** (`text.Splitter`). Thinking is on when the rendered prompt ends with
     `<think>\n`.
     - Reasoning starts after all leading isspace bytes (space, `\t`, `\n`, `\v`,
       `\f`, `\r`). It runs until the first `</think>` (consumed) or `<tool_call>`
       (kept as the start of content), and **keeps trailing whitespace**.
     - A suffix that could begin either delimiter is held back.
     - If the stream ends inside reasoning, the held partial delimiter is dropped.
     - Content starts after all isspace bytes that follow the delimiter, and runs to
       the end with trailing whitespace kept.
     - With thinking off, everything after leading isspace bytes is content, and
       delimiters stay literal.
     - In tool mode (tools given, `tool_choice` auto), content ends at the first
       `<tool_call>` and the rest goes to the call parser, which also constrains the
       token after a complete call ([tool calling](tool-calling.md)).
- A sink error (client disconnect) aborts generation immediately.
  Cancellation of the calling task (server drain deadline) is observed between
  tokens (`io.checkCancel`), while the device is idle.

## Verification

CPU: `tests/session.zig` (UTF-8, stops, sampler distributions/top-k selection,
scripted generation for reasoning split, length/context caps, stops, cancellation).
End-to-end: `tools/check_session.py` — native server greedy output for each oracle
case equals the libllama greedy continuation decoded with llama's raw pieces, with
identical prompt/completion token counts, in JSON and SSE modes.
