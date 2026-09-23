# Tool calling: parity with llama-server and an end-to-end bruh run (2026-09-24, block 12c)

Question: does zerv's tool calling (`tools`, `tool_choice`, `parallel_tool_calls`,
`tool` messages, assistant `tool_calls`) behave like llama-server for Qwen3.8-27B,
and can a real agent (bruh) use zerv?
[Spec](../specs/tool-calling.md), [research](../research/tool-calling.md).

## Setup

- Model `Qwen3.8-27B-Q4_0.gguf` (`ede16c7b…`), RX 7900 XTX (RADV), one engine at a time.
- Reference: llama-server build 10964 (`b29c606e`, binary `0f401589…`), engine
  `llama-fp32-full` from `bench/run_serving.py` (FP32 prompt and KV paths, so greedy
  tokens are comparable with zerv's FP32 default), official template file,
  `--reasoning-format deepseek`, `--jinja`.
- zerv binaries built from the current source with `zig build server -Doptimize=ReleaseFast -Dcpu=native`:
  run1 `ef66c18a…`, run2 and bruh `282ff781e8ad36c596e0c85185419e68bace4cb7c9110a3d6fb573daf90c0f01`
  (run2 adds the leading `{` to the SSE call header).
- Workload: [tool-parity-v1.json](../../bench/workloads/tool-parity-v1.json):
  - 11 greedy generation cases: bruh's captured request (thinking on and off), typed
    arguments, a multi-line string, parallel calls on and off, a final answer after a
    tool result, `tool_choice: none`, a length cut inside a call, an array argument, and
    thinking with low effort;
  - 20 render cases.

## Prompt rendering

The independent oracle is [render_tools.py](../../tests/reference/render_tools.py): pinned
Jinja2 3.1.6 with the transformers chat-template environment. Fixture:
`tests/fixtures/chat-tools.json`, 27 cases plus 830 number literals.

- **Oracle vs llama-server `/apply-template`:** 16 of 20 are byte-equal
  ([llama capture](data/2026-09-24-tool-parity/llama-reference/raw.json)). The 4 differences are all
  float formatting inside `tojson`: llama.cpp's Jinja prints `1.0` as `1` and `-0.0`
  as `-0`. zerv follows Python's `json.dumps`, the environment the template targets.
  This is a documented divergence ([research](../research/tool-calling.md)).
- **zerv vs the oracle:** all 26 renderable cases are byte-equal through the request
  parser, and the one invalid case is rejected. All 830 number literals print exactly as
  Python does. These are unit tests (`tests/tools.zig`), run in Debug and ReleaseFast.

## Generation parity

Command:

```sh
python3 tools/check_tool_parity.py --engines llama-fp32-full --output docs/bench/data/2026-09-24-tool-parity/llama-reference
python3 tools/check_tool_parity.py --engines zerv --zerv-binary third_party/tool-parity/zerv-282ff781 \
  --reference-raw docs/bench/data/2026-09-24-tool-parity/llama-reference/raw.json \
  --output docs/bench/data/2026-09-24-tool-parity/zerv-run2
```

**Result: 22 of 22 are `equal`** (11 cases × JSON/SSE), in both
[run1](data/2026-09-24-tool-parity/zerv-run1/) and
[run2](data/2026-09-24-tool-parity/zerv-run2/report.json). The comparison covers:

- reasoning_content;
- content;
- tool call names and parsed arguments;
- finish_reason;
- prompt and completion token counts.

The raw argument strings are also byte-identical in every case. For example:

- `{"path":"hello.py","content":"print(\"hello world\")\nfor i in range(1, 4):\n    print(i)\n"}`
- a length cut returns `{"path":"hello.py"`, as llama does;
- `parallel_tool_calls: false` stops after one call at 27 tokens, as llama does.

**SSE chunking (run2).** The header chunk
`{index, id, type, function: {name, arguments: "{"}}` and the string-argument fragments
match llama's sequence. One documented difference remains: non-string values (arrays,
numbers) arrive in one fragment at the value's close, where llama streams them in
pieces. The concatenated strings are identical.

**Differences that are not measured here:**

- Call ids: llama uses 32 random characters; zerv uses `call_` + the completion id + an
  index.
- The role chunk: llama sends `content: null`; zerv sends `""`.
- llama's grammar also constrains function and parameter names and JSON values inside a
  call. zerv constrains only what may follow a complete call; inside a call it parses
  without constraints. No workload case needed the in-call grammar (the unconstrained
  model already produced valid calls). `tool_choice: required` and named functions are
  rejected with 400 `unsupported_parameter`.

**Regression check (content-only output).** The splitter now emits events instead of
channel/byte pairs. `tools/check_parity.py --zerv-binary third_party/tool-parity/zerv-282ff781`
on the earlier output-parity workload is 12/12 `equal` against llama-fp32-full
([report](data/2026-09-24-tool-parity/output-parity-regression/report.json)).

## End-to-end: bruh

- bruh `/home/USER/bin/bruh`, SHA-256 `5d3a0c0f…`, built from `~/projects/bruh` at
  `cfd2a6b7` (test files modified in the worktree).
- zerv `282ff781…` with `--context 29504`, FP32 prefill.
- Command: `OPENAI_BASE_URL=http://127.0.0.1:18095/v1 OPENAI_MODEL=qwen3.8-27b bruh -p
  openai-compat --only bash --plain --no-context-files …`, run in a scratch directory.
- Data: [transcripts, sessions, server log, hashes](data/2026-09-24-bruh-e2e/).

| Task | Steps | Result | Wall time |
| --- | --- | --- | --- |
| "Create hello.py that prints 1 to 3, run it, tell me its output" | 2 | File written, run, output reported | 7.4 s |
| "The unit tests fail. Run them, fix the bug in calc.py, rerun" | 4 | Two parallel calls in step 1. Found `- 1` in `mean`, fixed it with `sed`, tests pass (verified independently afterwards) | 18.8 s |
| `--resume` of the same session: "add a test for mean with a single-element list and run the tests" | 3 | `test_mean_single` added, 3/3 tests pass (verified) | 16.4 s |

- Server metrics after the three tasks:
  - 9 requests;
  - 0 rejected;
  - 0 failed;
  - 0 malformed tool calls (`zerv_tool_call_parse_failures_total`).
- Per-step TTFT was 1.9–4.0 s for prompts of 1.2k–2.5k tokens. Decode was 46–47 tok/s.
- Each request prefills the whole conversation again, because zerv has no prefix cache
  (queued).
- The server exited cleanly on SIGINT.

## Limitations

- One greedy workload of 11 generation cases. Sampled (temperature > 0) tool calling is
  covered only by the bruh runs, which use bruh's default sampling.
- The in-call grammar is not implemented (see above). A malformed call from the
  unconstrained model is handled as the spec describes: the raw text becomes content, or
  an incomplete call is reported, generation stops, and the failure is counted. This
  path is covered by unit tests but never occurred on the real model in these runs.
