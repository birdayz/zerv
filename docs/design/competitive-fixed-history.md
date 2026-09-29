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
