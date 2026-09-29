# Bounded short-reuse checkpoint suppression

2026-09-29, D.2 performance follow-up; originally specified before production code.
Implemented and evaluated: [report](../bench/2026-09-29-reuse-join.md). Default remains
off; later-turn regressions are measured and competitive quality acceptance stays open.
Research: `docs/design/competitive-fixed-history.md`, controlled counterfactual under
`docs/bench/data/2026-09-29-reuse-counterfactual`. Papers motivate balancing checkpoint
storage/recomputation, not this exact threshold; this is a native measured policy.

`--prefix-cache-reuse-join N`, default0/off, range0..512 and no greater than configured
prefill chunk. Nonzero requires parallel>1, shared KV, radix, snapshots>0, non-MTP.
On a successful local/disk `begin` returning `0 < start < prompt.len`, suppress all
new prefill checkpoint boundaries only if `prompt.len-start <= N`. Cold misses and
longer suffixes preserve original behavior. The same full suffix tokens are sent to
prefill; no work/tokens/precision changes or retained caller buffers. Existing cache
entries remain intact. No new intermediate snapshot is retained; later requests may
recompute more tokens. This tradeoff is explicit and the default is unchanged.

Resolve threshold at startup and log it. Session Request carries the already validated
threshold. Candidate selection exposes a pure helper used by generation, while existing
candidatePoints and store behavior remain unchanged for other callers. Threshold0,
start0, exact boundary N, N+1 and invalid start>=length are covered by independent
exhaustive token-boundary fixtures generated before code. No per-token allocation.

Correctness: policy fixture in both modes; no checkpoint side effects on skipped path;
production f16 shape-fixed prefill still computes all tokens. Counterfactual must report
all vocabulary comparisons and future-cache loss. Independent FP64/libllama337-token
math gate remains mandatory after wiring. Model counterfactual and repeated HTTP
baseline/on responses/counts must be exact on declared workloads; differences are a
failure to claim losslessness, not permission to loosen tolerance. Test cold, short
reuse, long reuse, cancellation and multi-conversation pressure through existing
interfaces. Include repeated requests/multi-turn workloads to expose recomputation.

Acceptance requires measured full-serving improvement and reports regressions against
both tuned competitors. This policy alone does not establish competitor quality parity.
