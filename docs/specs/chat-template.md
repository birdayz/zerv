# Qwen3.8 official text-only chat template

Update (2026-09-24, block 12c): tools, `tool` messages and assistant `tool_calls` are
now rendered as well; their contract and oracle are in [tool calling](tool-calling.md).
The text-only rules below are unchanged.

Readiness: [tokenizer/template research](../research/tokenizer.md); official pinned
`tokenizer_config.json` fully inspected. This component deliberately implements the
**official template's text-only subset**, not the publisher's altered template.
The two sources agree for ordinary system/user/assistant text; publisher extensions
(developer prefix, high effort alias, missing-user allowance, tools) are excluded.
An eventual model plan/API must select and report the official policy explicitly.
No generic Jinja interpreter or copied upstream implementation is introduced.

## Native boundary

`chat.qwen38.render(messages, options, writer)` borrows typed UTF-8 string messages
and writes exact prompt bytes to a caller-owned `std.Io.Writer`. No heap allocation.
Message: role (system, user, assistant, developer), text content, optional text
reasoning_content. Developer is explicitly rejected. Tools, multimodal content,
content arrays, template overrides and assistant tool_calls are not in this API;
the future protocol adapter must reject them rather than discard them.

Options: enable_thinking=true, reasoning_effort=xhigh (xhigh/medium/low),
preserve_thinking=true, add_generation_prompt=true; maximum 1,024 messages and
1 MiB total content/reasoning bytes. Empty input and invalid UTF-8 fail. System is
allowed only at index zero. Require at least one user whose stripped text is not
both prefixed `<tool_response>` and suffixed `</tool_response>`; its last index is
the history cutoff. All validation happens before writes. Writer I/O/capacity errors
may leave a prefix, and callers must discard that response. Output limits belong
to the supplied writer. Calls share no state and may execute concurrently.

Stripping exactly follows Jinja/Python `str.strip`: U+0009..000D, U+001C..0020,
U+0085, U+00A0, U+1680, U+2000..200A, U+2028..2029, U+202F, U+205F, U+3000.
These are data predicates, not a locale or regex dependency. Do not normalize NFC
here; tokenization is a separate component. Interior bytes are preserved.

When thinking is enabled, xhigh/low prepend the official literal instruction to
the system message (separated from nonempty system content by two LF); medium
adds none. Empty system plus empty instructions emits no system message. User
messages are ChatML with stripped content. Assistant history contains the official
think delimiters and stripped reasoning if preserve_thinking or its index is after
the last real user; otherwise only content. Every historical message ends with
`<|im_end|>\n`. Generation suffix starts `<|im_start|>assistant\n`, followed by
`<think>\n` when enabled, otherwise `<think>\n\n</think>\n\n`.

## Independent fixture gate (before native implementation)

Use isolated Jinja2 3.1.6 `ImmutableSandboxedEnvironment(trim_blocks=true,
lstrip_blocks=true)`, a raise_exception helper and the literal official template.
Render the same text-only cases through the embedded publisher template and record
both successes and differences/errors. No tools are provided, so tojson behavior
is not exercised. Record generator, template/config and package binary hashes.
Output refuses overwrite. Native tests consume static fixtures with no Python/
Jinja/runtime oracle dependency. Cases cover all options, empty/whitespace/system
messages, multi-turn cutoff/reasoning, Unicode stripping, embedded literal markers,
missing user, invalid role order and publisher-only extensions. Success requires
byte-for-byte equality in Debug and ReleaseFast, plus native input/limit/error tests.

Component benchmark: repeated fixed fixture corpus, same options and exact output
hashes, compare native rendering against the precompiled pinned Jinja template;
exclude template compilation and fixture parsing. Record independent process CPU
affinity, warmup, repeated times and hashes. This is not BPE or serving throughput.
