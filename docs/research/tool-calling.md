# Tool calling for Qwen3.8: template, llama-server request mapping and parser (research, 2026-09-24)

Question: what exactly must zerv do so that OpenAI-style tool calling (`tools`,
`tool_choice`, `tool` messages, assistant `tool_calls`) behaves like the official
template and like llama-server for this model? Feeds the
[tool-calling spec](../specs/tool-calling.md). Continues the
[output-parsing research](output-parsing.md), which covered the content-only case.

## Sources

- Official template: `third_party/Qwen/Qwen3.8-27B/1d4bf0f2…/tokenizer_config.json`
  (`chat_template`), the same text as `official-template.jinja` beside it.
- Hugging Face transformers commit `935f7ab4f432dfeccfe60b190c546463f3b1e895`
  (main on 2026-09-23), `src/transformers/utils/chat_template_utils.py`, fetched to
  `third_party/transformers/<commit>/…`, SHA-256
  `3125114cf05646e7bc526ec30a6838da15bc9aa4591e42527e1012eaca3d276d`, git blob
  `cfa2a416a510`. Lines 481–494: the chat-template environment is
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
  extensions=[AssistantTracker, loopcontrols])` with
  `tojson(x) = json.dumps(x, ensure_ascii=False, indent=None, separators=None,
  sort_keys=False)`. This, not Jinja2's HTML-escaping, key-sorting `tojson`, is the
  filter the template was written for.
- llama.cpp commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (the installed
  llama-server, build 10964), under `third_party/llama.cpp/<commit>/`. Git blobs match
  `research-tree.json`:
  - `common/chat.cpp` 3a204e12d758: message/tool parsing (`common_chat_msgs_parse_oaicompat`
    380–480, `common_chat_tools_parse_oaicompat` 587–620, `common_chat_tools_to_json_oaicompat`
    558–575), template workarounds (`func_args_not_string` 1041–1058,
    `requires_non_null_content` 1032–1039), message diffs for streaming (266–340),
    final parse (1437–1511).
  - `common/parsers/qwen3-coder.cpp` 7938a2027932: the Qwen3-Coder/Qwen3.5 tool grammar.
  - `common/chat-peg-parser.cpp` ffa43a318888: AST → message mapping (arguments JSON
    assembly, 276–450).
  - `common/json-schema.cpp` 6898840e7d1f: schema kinds and `value_types` (227–330, 368–427).
  - `common/json-schema-to-grammar.cpp` e0426098c08a: `SPACE_RULE` (229).
  - `common/jinja/value.cpp` 6999ef7d6706: llama.cpp's own `tojson` (235–262, 1468–1545).
  - `common/jinja/caps.cpp` c5962ab77685, `tools/server/server-common.cpp`
    (request mapping, 1200–1330), `tools/server/server-task.cpp` (responses, 155–240,
    405–500).
- Empirical: [llama-server reference capture](../bench/data/2026-09-24-tool-parity/llama-reference/)
  (`tools/check_tool_parity.py --engines llama-fp32-full`, workload
  [tool-parity-v1.json](../../bench/workloads/tool-parity-v1.json)): `/props`,
  `/apply-template` for 20 render cases, and 11 greedy generation cases in JSON and SSE.

## Template: tools and tool messages (source-backed)

- With a non-empty `tools` list the first block is always
  `<|im_start|>system\n` + (reasoning instructions + `\n\n` if any) + `# Tools\n\nYou
  have access to the following functions:\n\n<tools>` + for each tool `\n` +
  `tool|tojson` + `\n</tools>` + the fixed format instructions, then `\n\n` + the
  stripped system message if it is non-empty, then `<|im_end|>\n`. An empty list is
  falsy: no tools block.
- `tool` messages: stripped content wrapped as `\n<tool_response>\n…\n</tool_response>`.
  A run of consecutive tool messages opens with `<|im_start|>user` (only when the
  previous message exists and is not a tool message) and closes with `<|im_end|>\n`
  after the last one of the run (or at the end of the list). A tool message at index 0
  therefore opens no user block; this is what the template does.
- Assistant `tool_calls`: after the (think block and) stripped content, each call is
  `<tool_call>\n<function=NAME>\n` (the first one preceded by `\n\n` when the content is
  non-empty; later ones by `\n`), then for each argument
  `<parameter=KEY>\nVALUE\n</parameter>\n`, then `</function>\n</tool_call>`. String
  values are inserted raw; other values as `tojson`. `arguments` must be a mapping
  (`|items`); the loop is skipped when `arguments` is `''`.
- User messages whose stripped text is `<tool_response>…</tool_response>` do not count
  as the last query (already implemented). Tool messages never do.

## llama-server request → template mapping (source-backed and observed)

- Tools are normalized to `{"type": "function", "function": {"name", "description"
  (default ""), "parameters" (default {})}}`; other fields (e.g. `strict`) are dropped.
  `type` must be `"function"`. The rendered order is exactly this.
- Assistant `tool_calls[].function.arguments` strings are parsed as JSON before
  rendering, because the template reports `supports_object_arguments`
  (`/props` → `chat_template_caps`). Non-string arguments are used as given. An empty
  string fails to parse (llama error); the template itself would skip it.
- Null content becomes `""`; text parts are concatenated.
- `parallel_tool_calls` defaults to the template capability, which is **true** here
  (`/props`).
- `tool_choice`: `none`, `auto` (default), `required`; named-function objects are
  parsed too. With `none`, tools are still rendered but the output parser is the
  content-only one and no grammar is used.

**Oracle check (observed).** An independent renderer
([render_tools.py](../../tests/reference/render_tools.py): pinned Jinja2 3.1.6 with the
transformers environment and the mapping above) reproduces llama-server's
`/apply-template` output byte for byte on 16 of the 20 render cases. The four
differences are all float formatting inside `tojson`: llama.cpp's own Jinja writes
floats with `ostream <<` (6 significant digits, no `.0`), so `1.0` → `1` and `-0.0` →
`-0`, where Python writes `1.0` and `-0.0`. Python's `json.dumps` is the reference
the template was written for; zerv follows it and records this as a llama.cpp
divergence (it only affects float literals in tool schemas or historical arguments).

## Output: grammar and parser (source-backed)

For this template `supports_reasoning` is true (the source contains `<think>`), so the
Qwen3-Coder "missing `<tool_call>`" workaround is off and the only call start is
`<tool_call>`. With tools and `tool_choice != none` the parser is:

```
"<|im_start|>assistant\n" + (reasoning << content(until "<tool_call>") << tool_calls)
tool_calls = repeat(call (+ call* if parallel), min = required ? 1 : 0, max = 1)
call       = "<tool_call>\n" + one_of(tools: "<function=" NAME ">\n" args "</function>\n")
             + "</tool_call>" + space
args       = required parameters in any order, then optional ones, each
             "<parameter=" KEY ">\n" VALUE "\n</parameter>\n"
```

- VALUE for a parameter whose schema types are only `string`: raw text up to the first
  `\n</parameter>\n`. If the types exclude `string`: a JSON value (schema-constrained in
  the grammar). Mixed: JSON alternatives of the declared non-string kinds first, then
  the raw string.
- Types come from `value_types` over the parameter schema: `$ref` (same document),
  `anyOf`/`oneOf` (union), `type` arrays (union), `const`/`enum` (types of the values),
  `allOf` (intersection), `type` names (`number` = number|integer), objects by
  `properties`, arrays by `items`, strings by `pattern`/`minLength`/`maxLength`/known
  `format`; otherwise any type. Missing or empty `parameters` means
  `{"type": "object", "properties": {}}`.
- **Arguments string** (`chat-peg-parser.cpp`): built in generation order as compact
  JSON: `{` + `"KEY":` + value, joined with `,`, then `}` at `</function>`. Keys and
  string values use nlohmann `dump()` escaping (`\"`, `\\`, `\b \f \n \r \t`, other
  controls `\u00XX` lowercase, everything else raw UTF-8). JSON values are inserted as
  generated (e.g. `["zig allocators", "vulkan compute"]` keeps its space).
- **Grammar** (`include_grammar`, lazy for `auto`): generation is unconstrained until
  the trigger word `<tool_call>`, then constrained to the rule above. After a complete
  call only `space` (`SPACE_RULE`: `| " " | "\n"{1,2} [ \t]{0,20}`), another call (if
  parallel) or end of generation is allowed.
- **Final message.** `content` is `""` when empty; `finish_reason` is `tool_calls` when
  generation ended by EOS or a stop word with at least one parsed call, otherwise
  `stop`/`length`. A length cut inside a call returns the partial call:
  observed `{"path":"hello.py"` (closing quote added, no closing brace).
- **Streaming.** Diffs of successive partial parses. Observed chunks per call: a header
  `{index, id, type: "function", function: {name, arguments: "{"}}`, then argument
  fragments (`"city":"`, `Paris`, `"`, `}`) with `index` only. Ids are 32 random
  alphanumerics. The role chunk has `content: null`.

## Observed reference behavior (llama-fp32-full, greedy)

| Case | finish | calls (arguments) | completion tokens |
| --- | --- | --- | --- |
| bruh-date (thinking) | tool_calls | bash `{"command":"date"}` | 56 |
| weather-typed | tool_calls | `{"city":"Paris","unit":"fahrenheit","days":3,"include_hourly":true}` | 65 |
| parallel-default | tool_calls | Paris, Tokyo | 53 |
| parallel-false | tool_calls | Paris only | 27 |
| length-cut (24 tokens) | length | `{"path":"hello.py"` | 24 |
| tool-choice-none, tool-result-final | stop | none | 26, 22 |

`parallel-false` stops one token after the first call: the grammar leaves only
`space` or end, and the greedy choice is `<|im_end|>`, which is counted.

## Consequences for zerv

- Rendering must follow the template with Python `json.dumps` semantics, including
  float `repr`, key order as received, and the llama tool normalization.
- A faithful output path needs the XML parser above, the arguments-JSON assembly, and
  the incremental streaming of call headers and argument fragments.
- Grammar-constrained decoding is llama-server's mechanism, not part of the model or
  template. Two parts of it change which tokens can be generated:
  1. **Between calls**: only whitespace per `SPACE_RULE`, another `<tool_call>` (when
     parallel) or EOS. This decides where generation ends (see `parallel-false`) and
     is cheap to emulate with a small token mask.
  2. **Inside a call**: known function and parameter names, and schema-valid JSON for
     non-string values. Emulating it needs a token-level automaton including a JSON
     schema validator.
  The first is needed for matching termination and token counts; the second only
  matters when the unconstrained model would produce an invalid call.
- `tool_choice: required` and named functions cannot be honoured without the in-call
  grammar (they force a call). They must be rejected rather than ignored.
