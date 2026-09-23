# Serving — OpenAI Chat Completions v1 (block 12)

**Implemented and tested subset.** Incoming interface: `POST /v1/chat/completions`
(user requirement; no alternative inference API). Code: `src/serve`, executable
`zerv` (`src/main.zig`).

## Startup

`zerv --model PATH [--host 127.0.0.1] [--port 8080] [--context 8192]
[--alias qwen3.8-27b] [--max-waiting 16] [--vram-budget-gib 23]
[--prefill-chunk 512] [--drain-timeout 30] [--prefill-precision fp32]
[--prefix-cache-slots 8]` ([prefix cache](prefix-cache.md)).

1. **Port first.** The server binds and listens (`src/serve/listen.zig`) before loading
   anything.
   - The socket sets SO_REUSEADDR only, so a restart can bind while old connections are
     in TIME_WAIT. It never sets SO_REUSEPORT: std's `reuse_address` sets both, which let
     a second zerv share the port and its connections (2026-09-24 incident, `TODO.md`).
   - Another listener on the address makes startup fail at once with
     `<host>:<port> is already in use`.
   - Connections that arrive while loading wait in the kernel backlog (128) and are
     served when the model is ready. So `/health` and `/ready` answer only when requests
     can be served.
2. **Free VRAM.** Before any device allocation, the device-local bytes the model needs
   (weights, activation arena, state arena, prefix-cache snapshots, plus 256 MiB of
   headroom for driver objects) are compared with the driver's budget
   (`VK_EXT_memory_budget`: heapBudget − heapUsage, which includes other processes on
   RADV).
   - If they don't fit, startup fails with `not enough free VRAM: the model needs N MiB
     …, M MiB are free`. This is done instead of oversubscribing VRAM, which ended in a
     compute timeout and a lost device.
   - The numbers are printed at every start. Without the extension, no check is made and
     the start line says so.
   - Two servers starting at the same moment can still both pass the check; that race
     is not covered.
3. **Load.** Loads an explicit local GGUF (no downloads), validates architecture, every
   tensor and weight scale, and builds the tokenizer and resident model.
The container and the file mapping are released once the weights are resident, and
the tokenizer and sampling defaults are copied out, before serving starts.
Loopback by default. Sampling defaults come from the artifact's
`general.sampling.*` metadata (temperature 1.0, top_p 0.95, top_k 20), as
llama-server does. Failures are fatal and explicit; no CPU or smaller-context fallback.

## Endpoints

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | JSON or SSE (`stream: true`) |
| `GET /v1/models` | the served id(s) |
| `GET /health`, `GET /ready` | `{"status":"ok"}` (connections are accepted after load); `/ready` is 503 `{"status":"draining"}` during shutdown; both are 503 `{"status":"failed"}` once the engine is unusable |
| `GET /metrics` | Prometheus text: requests/rejected/overloaded/failed, token and time counters, queue gauge |

Other paths → 404, wrong methods → 405, missing content-length → 411,
body > 4 MiB → 413. Errors use `{"error":{message,type,param,code}}`.

## Request contract

Accepted: `model` (must equal the served id, else 404 `model_not_found`),
`messages` (system/developer/user/assistant; string content, text content parts or
null assistant content; assistant `reasoning_content`), `stream`,
`stream_options.include_usage`, `max_tokens`/`max_completion_tokens` (min wins),
`temperature` [0,2], `top_p` (0,1], `top_k` ≥0, `min_p` [0,1],
`presence_penalty`/`frequency_penalty` [−2,2], `repeat_penalty` (0,10], `seed`,
`stop` (string or ≤4 strings ≤64 bytes), `reasoning_effort` (low, medium,
high→template xhigh, xhigh), `chat_template_kwargs` {enable_thinking,
preserve_thinking, reasoning_effort}. Duplicate JSON keys are rejected. Tool calling
(block 12c): `tools`, `tool_choice` auto/none, `parallel_tool_calls`, `tool` messages
and assistant `tool_calls`, as specified in [tool calling](tool-calling.md).

Explicitly rejected (400 `unsupported_parameter`): n≠1, logprobs/top_logprobs,
legacy `functions`/`function_call`/role `function`, `tool_choice` required or a named
function, non-empty logit_bias,
response_format other than text, audio/modalities/prediction, non-text content parts.
Template errors map to the official template messages (e.g. developer role →
"Unexpected message role."). Prompts ≥ context → 400 `context_length_exceeded`.
Unknown top-level keys are ignored (OpenAI clients send many optional fields).

## Response contract

Non-streaming: `chat.completion` with one choice, `message.content`,
`message.reasoning_content` (only when non-empty, as llama-server omits it),
`message.tool_calls` (when the model called tools), `finish_reason`
`stop|length|tool_calls`, `usage`. Streaming: `chat.completion.chunk` events: role chunk,
then `reasoning_content` / `content` / `tool_calls` deltas as generated, a final empty delta with
`finish_reason`, an optional usage chunk (`choices: []`), then `data: [DONE]`.
A mid-stream failure sends an `error` event and closes the connection. The
reasoning/content split, whitespace handling and stop-string semantics follow
llama-server's parser for this model ([session spec](session.md#termination-and-text)).

## Scheduling and lifecycle

One model, one sequence: generations are serialized by a mutex. At most
`--max-waiting` requests may wait (including the running one); beyond that → 503
`overloaded`. Each request renders, tokenizes and validates before admission; the
model state is reset per request (no cross-request state; no prefix cache yet).
A client disconnect during streaming aborts generation at the next token. Connections
are bounded (64) and handled concurrently by `std.Io.Group` tasks. No TLS or auth:
non-loopback deployment is unsupported without a separate access-control plan.
Prompts/completions are never logged.

**Engine failure.** After a failed generation the server asks the engine whether it can
run again (`Engine.usableFn`). The native engine says no when the device is lost or a
command is left pending after a timeout. The server then:

- answers every health check with 503 `failed`;
- answers every chat request with 503 `engine_failed`;
- shuts down as if stopped;
- exits with status 3 ("zerv: stopped: the GPU device was lost"), without tearing down
  the device, whose objects may still be pending. A supervisor can restart it.

Failed requests are logged at warning level.

**Shutdown.** The first SIGINT/SIGTERM sets a flag, which `Server.run` polls every
20 ms. A second signal exits immediately with status 130. Draining then proceeds:

1. `run` stops accepting and closes the listener, so new connections are refused.
2. Chat requests on existing connections get 503 `shutting_down`. Every response
   carries `connection: close`.
3. Admitted requests, whether queued or generating, finish normally for up to
   `--drain-timeout` seconds.
4. After that, the remaining connection tasks are canceled. These are idle
   keep-alive connections, or a generation past the deadline, which stops at its
   next token. The process then releases the model and device and exits 0.

An accept failure from a transient resource shortage (fd limits, memory, an
aborted connection) is logged and retried after 100 ms rather than ending
serving.

## Verification

`tests/serve.zig`: request parsing/rejections, exact JSON/SSE encodings, and
real-socket loopback tests with fake engines. The loopback tests cover:

- JSON, SSE, errors, 404/405 and metrics;
- a 503 `overloaded` response with `max_waiting = 1`, while the queued requests
  complete;
- a client disconnect mid-stream canceling generation and freeing the slot;
- drain: new connections refused, a late chat request answered 503
  `shutting_down`, `/ready` answered 503, the admitted request completing, and
  idle connections closed;
- the drain deadline canceling a running generation.

`tools/check_shutdown.py` sends SIGINT to the real binary during a stream and
checks that the stream completes, that new connections are refused, and that the
process exits with status 0. `tools/check_parity.py` runs the same greedy requests
through llama-server and zerv, one after the other, and requires equal
`reasoning_content`, `content`, `finish_reason` and token counts in JSON and SSE
modes.

End-to-end: `tools/check_session.py` (greedy equality with libllama through the real
server) and `bench/run_serving.py` (matched llama-server comparison).
