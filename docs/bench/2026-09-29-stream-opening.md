# Remove standalone role-only opening SSE event

2026-09-29. User-requested HTTP behavior change, **not a performance benchmark**.

Previously `streamResponse` wrote/flushed `delta.role=assistant,content=""` before
calling the engine. The pinned InferenceX client counts the first choices event as
TTFT, so this event made native raw TTFT artificially short and charged prompt
processing to TPOT. That TTFT was not evidence of generated-token responsiveness.
The earlier interpretation/adjusted diagnostics are not authoritative results.

## Change

- Deleted the standalone opening event and removed the `.role` delta variant.
- One per-stream started flag: the first nonempty reasoning/content/tool payload
  contains `role: assistant` in the **same** event. Subsequent deltas omit role.
- Empty text/argument callbacks do not emit SSE. A no-payload completion attaches
  role to the terminal finish event, after generation returns. A pre-output error
  emits only the normal error event; no fabricated first-token event.
- No changes to model math, token selection, finish/usage/DONE behavior, non-streaming
  responses, upstream benchmark client or upstream result-processing code.
- No per-token heap allocation added; serialization uses the existing writer.
  Performance impact has not been measured and no speedup is claimed.

Contract/research: [serving spec](../specs/serving.md#no-standalone-opening-role-event-2026-09-29),
[metric source trace](../research/2026-09-29-interactivity.md). The OpenAI delta permits
role alongside payload; independent captured reference responses establish retained
payload/finish meanings, not a requirement to reproduce the reference's separate
role event. Tests deliberately enforce our stricter no-standalone-role policy.

## Actual verification

- Focused `bazel test //tests:serve //tests:serve_release_fast`: **2/2 passed**, both
  executed. Exact SSE serialization and real socket tests cover content-first,
  reasoning-first, tool-first, empty output, pre-output error, empty callback
  suppression and role appearing only once. Existing socket-gating test still proves
  emitted payload reaches the client before generation continues. Existing tool,
  keep-alive, shutdown and cancellation coverage passes.
- Negative control: temporarily reintroduced the old role-only opening bytes/flush.
  The serve target **failed** as expected (extra event plus missing first-payload
  field; initial failure included a null assertion panic). Removed the injected
  regression and strengthened that field-presence assertion to fail cleanly.
- Final `bazel test //...`: **83/83 passed**,8 executed,75 cached. Includes formatting,
  all CPU Debug/ReleaseFast targets and Python tests. No GPU/shader changes or GPU
  benchmark execution; no need to reinterpret an old benchmark as testing this code.
- `git diff --check` is also clean.

[Logs](data/2026-09-29-stream-opening/): `focused.log`, `negative-control.log`,
`final-tests.log`. Old benchmark artifacts remain unchanged and describe the **old
binary/event behavior**. New performance/TTFT claims require a fresh run using the
unaltered upstream client and processor; none was started automatically here.
No upload, publication or git push.
