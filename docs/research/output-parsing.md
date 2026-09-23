# Output text pipeline: llama-server reasoning/content split (research, 2026-09-22)

Question: how does llama-server turn generated tokens into `reasoning_content` and
`content` for Qwen3.8-27B? The [serving benchmark](../bench/2026-09-22-serving.md)
saw one mismatch with zerv: llama-server keeps a trailing `\n` at the end of
`reasoning_content`, and zerv trimmed it. This note reads the reference code. It
feeds the output-text section of the [session spec](../specs/session.md).

## Sources (source-backed facts)

The reference is llama.cpp commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. The
installed `llama-server --version` reports build 10964 at commit b29c606e28, the
same commit. The files were fetched from
`https://raw.githubusercontent.com/ggml-org/llama.cpp/<commit>/<path>` into
`third_party/llama.cpp/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/<path>`. Each git
blob hash matches the commit's tree listing (`research-tree.json`).

| Path | git blob | SHA-256 |
| --- | --- | --- |
| `common/chat.cpp` | 3a204e12d758 | a226aa0cbf844490cd480bb805c458df16af8771b1ecb53a8fe608e3b2e60d3c |
| `common/parsers/qwen3-coder.cpp` | 7938a2027932 | d3d64cd8446c1b9a7f269993c91dbbb27c60ef4993e364cbb918f50a816a36db |
| `common/peg-parser.cpp` | 10735389ea19 | 594680aeee74b5b35ae7193075ec78e9fdff526061eb05d17e1d823f9b51f7c7 |
| `common/chat-peg-parser.cpp` | ffa43a318888 | a069958a4967609f67a3feea22a37dd3a8d712687eb37c56d75ea078aac540e3 |
| `tools/server/server-context.cpp` | b6835e43459e | 2f5d65ce6ef0504b5c8cf55a74c68d3959c49784ba380ef836566b7a7d5fa12b |
| `src/llama-vocab.cpp` | (earlier snapshot) | b9588d7116c11573b378c43bf3c85f87249ad5eb9626324abded4abc7c91e6dc |

Also fetched for context: `common/chat.h`, `chat-peg-parser.h`, `peg-parser.h`,
`chat-auto-parser*.{h,cpp}`, `chat-diff-analyzer.cpp`, `parsers/parsers.{h,cpp}`,
`tools/server/server-task.cpp` and `server-common.cpp`.

## Pipeline

1. **Token text.** `server-context.cpp:3800-3802,3878` renders each sampled token with
   `common_token_to_piece(..., special)`. `special` is true only for tokens in the
   format's `preserved_tokens`. `llama-vocab.cpp:3595-3601` renders CONTROL and
   UNKNOWN tokens as nothing when `special` is false. USER_DEFINED tokens render as
   their text either way. UNUSED tokens match no branch in the BPE switch, so they
   render as nothing too.
   - The Qwen3.8 GGUF token types (read from the model file) are:
     - control (3): 248044–248057, 248060–248065, 248070–248076. This includes
       `<|im_start|>`, `<|im_end|>` and `<|endoftext|>`.
     - user-defined (4): `<tool_call>` 248058, `</tool_call>` 248059,
       `<tool_response>` 248066, `</tool_response>` 248067, `<think>` 248068 and
       `</think>` 248069.
     - unused (5): the `[PAD…]` tokens, 248077 and up.
   - So `</think>` is part of the raw text, and a control token adds no text.
2. **Stop strings** (`server-context.cpp:567-595,1836-1870`) are matched on the raw
   generated text. That text includes `</think>` and all whitespace. There is no
   separate reasoning or content channel at this stage.
   - Among all matches, the earliest one wins. The text from the match onward is
     erased.
   - A suffix that could still begin a stop string is held back from streaming.
   - No stop check runs while the text ends in an incomplete UTF-8 sequence.
3. **Parse.** The template contains `<tool_call>`, `<function=` and `<parameter=`,
   so `chat.cpp:1204-1209` selects the specialized Qwen3-Coder parser.
   - None of the earlier specialized detectors match; each of their marker strings
     occurs 0 times in the embedded template.
   - `chat.cpp:1449-1460` parses `generation_prompt + text` with the LENIENT flag,
     for both partial and final parses.
   - The generation prompt is `<|im_start|>assistant\n<think>\n` with thinking on,
     and `<|im_start|>assistant\n<think>\n\n</think>\n\n` with thinking off. This is
     the diff between rendering with and without `add_generation_prompt`.
   - The grammar for the content-only case (no tools, no response format) is at
     `qwen3-coder.cpp:74-81,173`:

     ```
     literal("<|im_start|>assistant\n")
       + ( optional("<think>" + space + reasoning(until_one_of{"</think>","<tool_call>"})
                    + ("</think>" | peek("<tool_call>")))
           << content(rest) )
     ```

   - `a << b` is `a, space, b` (`peg-parser.cpp:994`).
   - `space` consumes every `std::isspace` byte: space, `\t`, `\n`, `\v`, `\f`, `\r`
     (`peg-parser.cpp:493`).
4. **Mapping** (`chat-peg-parser.cpp:300-326`): reasoning and content node texts are
   concatenated. If the reasoning is only ` `, `\n`, `\r` and `\t`, it is cleared.
5. **Streaming** (`server-task.cpp:162-177`): each partial parse of the text sent so
   far is diffed against the previous one. The deltas are that diff.

## Consequences (semantics to match)

With thinking on, the parsed input starts `<think>\n` + output:

- **Reasoning** starts after all leading isspace bytes. It runs up to the first
  `</think>` or `<tool_call>` in the text, and **trailing whitespace is kept**.
  - `</think>` is consumed.
  - `<tool_call>` is not consumed: it starts the content.
  - Both are found in text, so they also match when spelled with several ordinary
    tokens.
- **Content** starts after all isspace bytes that follow `</think>`. It runs to the
  end of the text and keeps its trailing whitespace. A later `</think>` or
  `<think>` stays literal content.
- **End of text inside reasoning.** The until parser handles this at
  `peg-parser.cpp:661-706`.
  - If the text ends with a proper prefix of a delimiter, for example `</thi` or
    `<`, that prefix is dropped (not moved to content). In lenient mode the
    closing literal returns NEED_MORE, and the reasoning node is kept.
  - An incomplete UTF-8 tail is also dropped.
  - Otherwise, all of the text is reasoning, including trailing whitespace. The
    content is empty.
- **Thinking off.** The empty think block is inside the generation prompt. The output
  is all content after its leading isspace bytes, and `</think>` or `<tool_call>`
  in the output stays literal.
- **Streaming.** A suffix that is a proper prefix of `</think>` or `<tool_call>` is
  held back from reasoning. It is released when it can no longer start a delimiter.
- **Stop strings** see the raw text. A stop at `</think>` ends the stream with the
  reasoning complete and empty content. A stop that matches across the
  reasoning/content boundary also ends it.

zerv before this change differed in three ways:

- It trimmed trailing whitespace from reasoning.
- It matched stop strings separately per channel, after `</think>` had been removed
  and before whitespace trimming.
- It rendered control tokens as their text.

It also switched channels only on the `</think>` token, not on text.

## Known, deliberately unmatched divergences

- **Invalid UTF-8 inside reasoning.** llama's until parser fails on INVALID, so the
  optional reasoning block fails. The whole output, including the prompt's
  `<think>\n`, then becomes content. zerv replaces invalid sequences with U+FFFD and
  keeps them in reasoning. This is a parser failure mode, not intended output
  semantics.
- **Incomplete UTF-8 tail in content.** llama sends the raw bytes, and the JSON
  dump replaces them. zerv emits U+FFFD. Both surface a replacement character.

## Correctness mechanism

- **Unit tests.** `tests/session.zig` covers each rule above with scripted tokens:
  leading and trailing whitespace, a `</think>` spelled as text, `<tool_call>`, a
  partial delimiter at the end, a stop across the boundary, and control-token
  rendering.
- **Differential check.** `tools/check_parity.py` runs llama-server and zerv one
  after the other, never both at once. Each case is a greedy request, in JSON and
  in SSE mode:
  - thinking on and off;
  - `max_tokens` cutting inside reasoning;
  - stop `</think>`;
  - stop `\n\n`.

  The check compares `reasoning_content`, `content`, `finish_reason` and the token
  counts. A case where the greedy tokens diverge numerically is reported as such,
  not as a formatting result.
