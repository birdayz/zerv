# Server architecture: pluggable core (proposal, 2026-09-26 — nothing decided)

Status: **notes for discussion.** The user asked to write this down and understand it
better before deciding any interface. Nothing here is a commitment. The shared KV pool
(concurrent.md "18d.2") was built inside today's structure, independently of this proposal.

Goal (user, 2026-09-26): an architecture for the core server where the important
strategies are pluggable (KV and state storage, weight loading, the optimizer, …) and
configurable at run time, so that an AI agent can observe the server and tune it.

## Principles under discussion

1. **Three planes.**
   - **Data plane:** kernels and recorded GPU commands, per token. No runtime dispatch;
     everything is resolved at init (AGENTS.md).
   - **Control plane:** scheduler, planner, store and admission, per request or per unit
     (a segment, a batch). Runtime interfaces are acceptable: one indirect call per
     10–50 ms unit.
   - **Management plane:** config, metrics, admin API, agent. Never on the GPU's critical
     path.
2. **Mechanism vs policy.** Mechanisms (model, pool, tiers) expose capabilities and measured
   costs. Policies (planner, eviction, admission) are decisions over snapshots.
3. **One owner per resource:**
   - the GPU: the executor thread;
   - page accounting: the pool;
   - tier bytes: their tier;
   - planes exchange snapshots and messages.

## Candidate interfaces (none decided)

| # | Interface | Role | Notes |
| --- | --- | --- | --- |
| 1 | Source | Bytes by name or range, identity (sha256), async reads | local file / HF cache / NVMe / object store |
| 2 | Format | Tensor inventory and metadata from a source | GGUF / safetensors |
| 3 | Model | Capabilities, **state description**, operations, calibrated costs | Qwen3.8 on Vulkan today; MoE+MLA on CDNA later |
| 4 | Device backend | Buffers, kernels, commands, transfers (inside Model) | Vulkan / KFD+CDNA |
| 5 | State pool | Pages + fixed state slots: alloc, refcount, page tables | one mechanism; policy elsewhere |
| 6 | Tier | Async put/get/evict/stat of pages and snapshots | GPU pool, host, NVMe, object store |
| 7 | Checkpoint store | Prefix tree (the "memo"), placement, fingerprints, tenants | over the tiers |
| 8 | Cost model | Operator → cost vector (calibration + feedback) | |
| 9 | Planner | Pure: (request, snapshots, config) → plan of operators | direct enumeration now; Cascades-style later |
| 10 | Executor | Runs plans, reports actuals | today's batcher, evolved |
| 11 | Session / model family | Template, tokenizer, sampler, tools | exists (session, chat, tokenizer) |
| 12 | Frontend | Protocol → requests → streams | OpenAI chat |
| 13 | Telemetry + control | Metrics registry, event ring, typed config with change classes, admin API | |

**The state description** is probably the key abstraction, because it lets the pool, store
and planner work for any architecture:

```
StateClass = per_token_paged { bytes_per_token, page_tokens, window? }   // KV, MLA latent, sliding window
           | per_sequence_fixed { bytes, snapshot_only_at_positions }     // DeltaNet / Mamba state
```

- **Qwen3.8:** 64 KiB/token of paged KV (f16) plus a fixed 157 MB recurrent state that can
  only be resumed where a snapshot exists.

**Paged vs contiguous KV:** kept as one mechanism, not a plugin.
- Contiguous is paging with one page per sequence (`--kv-page-tokens context`); the measured
  difference is 0.06% (docs/bench/2026-09-24-paged-kv.md).
- The allocation *policy* is what varies.

## Change classes for runtime config

| Class | Examples | Change |
| --- | --- | --- |
| live, output-neutral | stall ms, pack size, prefill order, tier sizes, eviction, checkpoint interval, admission | atomic config snapshot, picked up at the next scheduler iteration |
| drain + rebuild | `--parallel`, pool size, KV type, decode precision, split rule | refused live; drain and re-init |
| never automatic | anything that changes output arithmetic | explicit human flag; changes the checkpoint fingerprint |

An agent observes metrics and proposes config diffs. zerv validates them against the schema
and bounds, rate-limits them, audits them, and rolls back on SLO regression. A misbehaving
agent can make the server slower, never wrong.

## Tiered KV and checkpoints (discussed)

- **Unit of storage and transfer:** the page. In f16 a page is 128 tokens × 64 KiB = 8 MiB,
  a good NVMe and PCIe transfer size.
- **Unit of validity:** the checkpoint, a recurrent-state snapshot at p (page-aligned) plus
  pages [0, p).
- **Prefix tree:** pages are content-addressed, keyed by the parent hash, tokens, the
  arithmetic fingerprint and the tenant. Shared prefixes are stored once. Tiering is per
  page, with leaf-first eviction.
- **Restore vs recompute** at 100k tokens (estimates):
  - checkpoint size ~6.7 GB (f16 KV) or ~3.4 GB (8-bit);
  - host restore ~0.3 s, NVMe ~1–2 s;
  - re-prefill several minutes (29k measured at 46 s).
- **Restore vs recompute is a planner decision:** cost vectors, where prefill spends shader
  time that other users feel and a fetch spends copy-engine time that overlaps with compute.

Prior art to study before specifying (not yet verified): SGLang RadixAttention/HiCache,
vLLM automatic prefix caching and LMCache, TensorRT-LLM KV reuse, NVIDIA Dynamo KVBM,
Mooncake. Hybrid models are the awkward case for all of them: vLLM uses 1,568-token pages on
this model, observed in its log (docs/bench/2026-09-25-vllm.md).

## Optimizer ideas borrowed from Cascades (~/projects/fdb-go)

- Logical vs physical plans.
- Required properties with enforcers: data movement is the enforcer.
- The prefix tree as the memo.
- Bounded search with cost limits.
- A lexicographic comparator: feasibility, then weighted cost, then a deterministic
  tie-break.
- Determinism as a rule; EXPLAIN and a differ.
- Property tests on the cost model: transitivity, monotonicity, total preorder.
- A full rule engine only when the plan space needs it (multi-GPU, MoE).

## Open questions (for the user)

1. Model and executor comptime-generic (one model per process), or everything behind
   vtables (several models per process, e.g. drafts)?
2. One process per model behind a router, or several models per process?
3. Does the NVMe tier persist across restarts (a versioned on-disk format)?
4. Config: a file plus live admin-API diffs, or the API only?
5. Which interfaces to fix first, and how much of `runtime.zig` to split before the store.
