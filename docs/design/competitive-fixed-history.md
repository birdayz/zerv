# Competitive input equivalence and reuse-TTFT investigation

2026-09-29, baseline716b9de. D.2 competitive acceptance remains the sole active increment.

Observed C1 idle measurements: native second prompt7849 tokens/reuse7764; Vulkan
7839/reuse7818. Thus85 versus21 tokens remain, and the second input is not identical.
Native session `prefillCheckpointed` saves at message-boundary candidates during
prefill; the shared-checkpoint path does not save generated assistant-token state.
`src/session/root.zig` generation only records decoded tokens for the separate
single-slot `request.cache` path. This is a source-backed candidate explanation,
not measured proof that checkpoint policy accounts for the whole latency gap.

Before modifying production policy, add a fixed-history benchmark control:
- Optional fixture `assistant_history`, exactly turns-1 strings per conversation.
  Subsequent inputs use these canonical strings rather than the previous response.
  Ordinary fixtures continue using server responses unchanged.
- Record canonical request-body SHA256, output text, finish reasons, prompt and output
  counts. Fixed history does not force generated tokens or equal their quality.
- Reject malformed canonical history before starting servers. Unit tests cover actual
  second-request construction, unchanged legacy behavior, and malformed history.
- Keep the original distinct public-document workload, changing only canonical history
  and recording its new hash. All engines receive identical messages/parameters.
- Same GGUF SHA, f16 KV, non-speculative greedy sampling, context12288/parallel2 and
  existing pool/host/disk budgets. Tuned Vulkan and HIP remain the competitors.
- Fresh C1 then C4 reuse, three repeated rounds initially, preserving errors and all
  results. Eligibility for exact-work performance conclusions requires identical input
  hash, prompt counts, output hash and output counts, not just same max_tokens.
  Record mismatches; do not post-select easy prompts to claim broad parity.

Request hashes establish serialized message equality, not tokenizer-ID equality;
existing independent tokenizer/template fixtures support the implementation, but
per-request token-ID extraction remains a separate competitive evidence gap.
Do not call differing output streams equivalent without a predeclared independently
scored quality suite. Fixed history is a control, not the full performance acceptance.

## Per-request token-ID extraction (next verification step)

Use a CPU-only developer executable calling the same `serve.api.parseChat`,
`chat.qwen38.render` and tokenizer `encode(.{})` functions as production preparation.
Its input is an array of exact request bodies, output is rendered prompt plus every
u32 token ID. It adds no production endpoint or inference dependency. Compare all
four conversations × two fixed-history turns, not just token counts.

Pinned HIP reference source `third_party/research-serving/llama.cpp-RDNA3-7900xtx-opt`
revision15995a12, `tools/server/server-context.cpp` lines5047–5105: `/apply-template`
uses `oaicompat_chat_params_parse` (same chat request parser, without inference);
`/tokenize` accepts `add_special=false`, `parse_special=true` and returns full IDs.
Run both exact serving competitors with their recorded command/env, sequentially,
and retain API responses. Require byte-identical rendered prompts and full token arrays
against native; independently cross-check lengths against recorded serving prompt counts.
Hash model, binaries, native adapter and fixture. This is a correctness gate, no timing
claim; startup can load weights even though extraction is CPU-only. Existing scratch,
models and drivers only. Ordinary tests remain independent of external competitors.

Token capture passed all8 cases on both competitors (manifest under
`docs/bench/data/2026-09-29-prompt-equivalence`). Next profile the first fixed-history
conversation's reuse request with native production ModelBackend/radix cache settings.
Follow `prefillCheckpointed` exactly: begin/restore, each candidate segment and checkpoint,
then final suffix. Report begin, intermediate prefill, checkpoint and final prefill times
separately. One first-request seed, one reuse warmup, five measured reuse repetitions;
reseed cache for each repetition to prevent the target request becoming its own hit.
Compare every repeated final vocabulary row bitwise with the warmup. This is a model/
cache component diagnostic, excludes HTTP/tokenizer/scheduler waits, and cannot itself
explain the entire competitor gap. Do not substitute it for serving performance.
