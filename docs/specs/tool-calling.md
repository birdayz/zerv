# Tool calling — OpenAI Chat Completions v1 (block 12c)

Status: specified 2026-09-24 before implementation; implemented and verified the same
day ([evidence](../bench/2026-09-24-tool-calling.md)). Research:
[tool-calling](../research/tool-calling.md). Extends [serving](serving.md),
[chat template](chat-template.md) and [session](session.md).

## Request

| Field | Accepted | Otherwise |
| --- | --- | --- |
| `tools` | null, or an array of ≤ 128 objects `{type: "function", function: {name, description?, parameters?}}`. `name`: non-empty string ≤ 256 bytes. `description`: string (default `""`). `parameters`: object or null (default `{}`). Other keys are ignored. | 400 `invalid_request_error`, param `tools` |
| `tool_choice` | absent, null, `"auto"`, `"none"` | `"required"` and named-function objects: 400 `unsupported_parameter` (they need in-call constrained decoding, which zerv does not implement) |
| `parallel_tool_calls` | absent/null (= true), boolean | 400 |
| message role `tool` | `content` string, text parts or null (→ `""`); `tool_call_id` string or absent (not rendered) | 400 |
| assistant `tool_calls` | array of ≤ 128 `{id?: string, type: "function", function: {name: string, arguments}}`; `arguments` is a JSON-object string, `""` (no arguments), or an object | 400 `invalid_request_error`, param `messages` (e.g. arguments that are not a JSON object) |

`functions`, `function_call` and role `function` stay rejected (400
`unsupported_parameter`). Duplicate JSON keys are rejected everywhere, including inside
`arguments` strings. Tool definitions, call names and rendered argument values count
toward the template byte budget (1 MiB) and must be valid UTF-8.

**Parameter types** (used by the output parser). For each property of the tool's
`parameters` (after the default `{"type": "object", "properties": {}}` for a missing or
empty schema), the type set follows llama.cpp's `value_types`:
`$ref` (only `#/…` within `parameters`; cycles contribute nothing) → the target;
`anyOf`/`oneOf` → union; `type` array → union over each type with the rest of the
schema; `const` → the value's type; `enum` → the values' types; `type` names
`string|integer|boolean|null|array|object` → that type, `number` → number and integer;
no `type`: `properties` or non-`true` `additionalProperties` → object, `allOf` →
intersection, `items`/`prefixItems` → array, `pattern`/`minLength`/`maxLength`/format
`date|time|date-time|uuid[1-5]` → string, otherwise all types. An unknown `type` name,
a malformed `enum`, an unsupported or unresolvable `$ref` → 400. A root that is not an
object schema has no known properties.

## Prompt rendering

The official template's tools branch (research note), implemented natively:

- Tools block: each tool rendered as `tojson` of the normalized
  `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `tojson` is Python `json.dumps(ensure_ascii=False)`: separators `", "` and `": "`,
  keys in received order, escapes `\" \\ \n \r \t \b \f` and other C0 controls as
  `\u00XX` (lowercase), everything else raw (including DEL, U+2028/2029). Integers as
  received (big integers keep their digits); floats as Python `repr` (shortest
  round-trip digits; exponent form `d[.ddd]e±XX` when the decimal exponent is < −4 or
  ≥ 16, otherwise fixed with at least one fractional digit; `-0.0`).
  Divergence: llama.cpp's own Jinja prints floats with 6 significant digits and no
  `.0`; zerv follows the template's reference environment.
- Tool messages, consecutive-tool grouping and assistant `tool_calls` exactly as the
  template; argument values: strings raw, other values `tojson`.
- Gate: byte equality with the independent Jinja oracle
  ([render_tools.py](../../tests/reference/render_tools.py) →
  `tests/fixtures/chat-tools.json`), through the request parser; errors where the
  oracle errors.

## Output parsing (tools present and `tool_choice` auto)

Stop strings and the reasoning split work as before on the raw text. After the
reasoning, **content** runs up to the first `<tool_call>` (text level), keeps its
trailing whitespace, and is streamed with a held-back suffix that could still begin
`<tool_call>`. From `<tool_call>` on, the call parser consumes bytes:

```
call   = "<tool_call>\n<function=" NAME ">\n" arg* "</function>\n</tool_call>"
arg    = "<parameter=" KEY ">\n" VALUE "\n</parameter>\n"
after a call: SPACE_RULE whitespace, then another call (if parallel) or the end
```

NAME and KEY: bytes up to `>`, no newline, ≤ 256 bytes, whitespace-trimmed. VALUE: raw
bytes up to the first `\n</parameter>\n`. NAME and KEY are not checked against the
request's tools (see constraints).

**Arguments string**, as llama.cpp assembles it: `{` + `"KEY":VALUE` joined by `,` +
`}`, keys escaped like nlohmann `dump()` (the `tojson` escapes; no spaces added).
VALUE by the property's type set (unknown tool or key → all types):

- only string → JSON string of the raw text;
- otherwise the whitespace-trimmed text if it is exactly one JSON value of a declared
  non-string kind (`integer` requires an integer literal), inserted as generated; else a
  JSON string of the raw text (string declared, or fallback for a non-conforming value
  from the unconstrained model).

**Response.** `message.tool_calls = [{id, type: "function", function: {name,
arguments}}]` (omitted when there are none); `content` is `""` when empty.
`finish_reason` is `tool_calls` when generation ended by EOS or a stop string with at
least one call, otherwise `stop`/`length`. Ids are `call_` + 24 hex digits (request
counter and clock) + 2 hex digits (call index); unique within a process run.

**Streaming.** When `>` ends NAME: a header delta
`{tool_calls: [{index, id, type: "function", function: {name, arguments: "{"}}]}`
(the parser's `call_begin` event implies the leading `{`).
Then argument fragments `{tool_calls: [{index, function: {arguments}}]}`: at each KEY
`"KEY":` (preceded by `,` after the first); string-only values stream as `"`, escaped
text as it arrives (holding back a suffix that could begin `\n</parameter>\n`), and `"`
at the close; other values are emitted whole at the close; `}` at `</function>`.
Concatenated fragments equal the non-streaming arguments string.

**Incomplete calls.** At a length limit (or EOS inside a call) a call whose NAME is
complete is reported with its arguments so far: an open string value gets its closing
`"`, a buffered value is appended as generated, and no `}` is added (llama-server
reports the same partial form). A call without a complete NAME is dropped.

**Malformed calls.** If the bytes cannot continue the grammar above (the model is not
constrained inside calls), generation stops: the unfinished call's raw text, from
`<tool_call>`, is emitted as content; earlier complete calls are kept; `finish_reason`
is `tool_calls` if there are any, otherwise `stop`. `zerv_tool_call_parse_failures_total`
counts these. llama-server cannot produce this case (its grammar forbids it).

`tool_choice: none`: tools are rendered, the output uses the content-only parser.

## Constrained decoding

Only the continuation after a complete call is constrained, matching llama-server's
grammar there: the next token is sampled, with the request's normal sampling chain,
from EOS tokens; tokens whose output text is only spaces, tabs and newlines and keeps
the text since `</tool_call>` a prefix of `SPACE_RULE` (`"" | " " | "\n"{1,2}[ \t]{0,20}`);
and, when `parallel_tool_calls` is true, the `<tool_call>` token. Text-spelled
`<tool_call>` (several ordinary tokens) is not offered there. Inside calls sampling is
unconstrained: unknown function or parameter names are passed through and
non-conforming values are typed by the fallback above. This is a documented
divergence from llama-server, which also constrains names and JSON values after the
`<tool_call>` trigger.

## Acceptance

1. Rendering: byte equality with the oracle fixture (Debug and ReleaseFast).
2. Output parser: unit tests over scripted streams — every chunking gives the one-shot
   result; typing, escaping, partial and malformed calls; SSE fragments concatenate to
   the JSON arguments.
3. Constraint: a between-call token that the rule forbids is never sampled, even when
   it has the highest logit; allowed ones are sampled normally.
4. Parity: `tools/check_tool_parity.py` (llama-fp32-full vs zerv, one engine at a time)
   is `equal` for every generation case in JSON and SSE: reasoning, content, calls (name
   and parsed arguments), finish reason and token counts.
5. End to end: a real `bruh -p openai-compat --only bash` session against zerv
   completes a simple task; transcript and server log recorded.
