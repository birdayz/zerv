# Research TODOs: big Qwen hybrids and GLM-5.3-Flash on MI3xx with zerv — 2026-09-26

**Scope.** What must be researched, specified and verified before zerv can serve these
models on AMD Instinct (MI300X/MI325X `gfx942`, MI355X `gfx950`):
- the largest official open Qwen hybrids: Qwen3.5-397B-A17B, and Qwen3.8-2.4T-A95B;
- GLM-5.3-Flash.

Direction: AMD-first (user, 2026-09-26). Platform prerequisite: the
[Instinct backend note](2026-09-26-instinct-backend.md). Market context:
[bigger models](2026-09-26-bigger-models.md).

**Status.** Research plan. Nothing here is started. These are *queued* items, not the
active block (see `TODO.md`).

**"Qwen 3.9" does not exist officially** [S, Hugging Face API, 2026-09-26]. The newest
official Qwen LLMs are the 3.8 family.
- A repository `QwennAI/Qwen3.9-245B-A29B` exists. It comes from a look-alike org
  ("Qwenn", created 2026-08-23), claims 245B parameters, and ships **0.8M parameters (one
  F32 tensor file) plus a pickled `training_args.bin`**.
- It is fake. Do not download or load it: pickles execute code.
- "Big Qwen" below means Qwen3.5-397B-A17B (the practical first step) and
  Qwen3.8-2.4T-A95B (the largest official open Qwen).

**Pinned material** (`third_party/research/2026-09-26-instinct/`, ledger
[sources.json](2026-09-26-instinct/sources.json)):
- **Model metadata** (configs, chat templates, tokenizers, safetensors indexes, READMEs,
  licenses):
  - `zai-org/GLM-5.3-Flash@eb9eb208`
  - `Qwen/Qwen3.5-397B-A17B@84726181`
  - `Qwen/Qwen3.8-2.4T-A95B-FP8@d2dc3565`
- **Reference implementations:**
  - transformers `@27166ea0`: `glm5_next`, `qwen3_5_moe`, `kimi_linear`, `deepseek_v32`;
  - AMD ATOM `@68e0df5e`: `glm5_next`, `qwen3_5`, `qwen4_exp`, `kimi_k3`;
  - SGLang `@710a41ba`: `qwen3_5`, `glm5_next`.

## 1. The models [S, configs; derived memory figures D]

| | Qwen3.5-397B-A17B | Qwen3.8-2.4T-A95B | GLM-5.3-Flash |
| --- | --- | --- | --- |
| License | Apache-2.0 | Qwen license: separate license for selling API access above $50M/12 months | MIT |
| Layers | 60: 45 GatedDeltaNet + 15 gated attention (3:1) | 92: 69 GatedDeltaNet + 23 gated attention | 45: 34 KDA (Kimi Delta Attention) + 11 DeepSeek sparse attention (MLA + indexer) |
| Hidden | 4,096 | 8,192 | 4,096 |
| Full-attention heads | 32 q / 2 kv × 256, partial RoPE 0.25, output gate | 64 q / 4 kv × 256 | MLA: q_lora 1,536, kv_lora 512, qk 256 with no RoPE part (`mla_use_nope`), v 256, 64 heads |
| Linear attention | GDN: 16 qk / 64 v heads × 128, conv 4 | GDN: 16 qk / 128 v heads × 128; `output_gate_type: swish`, as in 27B (SiLU gated norm) | KDA: 64 heads × 128, short conv 4, forget gate with lower bound −5 |
| MoE | 512 experts, top-10, softmax → top-k → renormalize; 1 shared expert (1,024) behind a sigmoid gate; expert FFN 1,024 | 512 experts, top-10 + 1 shared; FFN 2,048 | 288 experts, top-8; sigmoid scores + `e_score_correction_bias` (noaux_tc), renormalized, × 2.5; 1 shared expert; first 3 layers dense (12,288); SwiGLU clamp at 10 |
| Residual | standard | standard | **mHC hyper-connections**: 4 streams; per block pre / post / comb weights from a projection; comb made doubly-stochastic by 20 Sinkhorn iterations (FP32); final collapse is the plain mean |
| MTP | 1 layer | 1 layer, "trained with multiple steps" | 1 layer (`num_nextn_predict_layers`), indexer shared with it |
| Vocabulary / tokenizer | 248,320, same family as Qwen3.8-27B | 248,320 | 154,880, GLM tokenizer (new work) |
| Context | 262k | 262k (1.01M extended) | 1M (`max_position_embeddings`) |
| Mode | thinking by default | **thinking mandatory, text-only** (README) | reasoning model |
| Checkpoints | BF16 807 GB; GPTQ-Int4 236 GB | BF16 4.9 TB; FP8 (128×128 blocks) 2.5 TB | FP8 E4M3 (OCP) 328 GB, with some modules left in BF16 |
| KV per token, FP8 | 15 KB | 46 KB | ~5.8 KB (MLA latent + pooled indexer keys) |
| Recurrent state per sequence (BF16) | 94 MB + 3 MB conv | 289 MB + 9 MB conv | 71 MB + 5 MB conv |

**Where each fits at 90% of HBM** [D]. Sequences at 19k context, the mean OpenRouter
request:

| Model | Weights format | Fits on |
| --- | --- | --- |
| Qwen3.5-397B | FP8 (~415 GB) | 2×MI355X (~260 sequences) or 4×MI300X |
| Qwen3.5-397B | 4-bit (~214–226 GB) | 1×MI355X (~85–115 sequences) or 2×MI300X |
| Qwen3.8-2.4T | FP8 (2.5 TB) | no 8-GPU node; 16 GPUs across 2 nodes |
| Qwen3.8-2.4T | MXFP4 (~1.3 TB) | 8×MI355X (~650 sequences); 8×MI300X only barely (~70) |
| GLM-5.3-Flash | FP8 (~331 GB) | 2×MI355X (~990 sequences); 2×MI300X is too tight (~15 GB left), use 4 |
| GLM-5.3-Flash | 4-bit (~170–180 GB) | 1×MI355X (~420–470 sequences) |

- Any 4-bit variant we make ourselves is a quality question, measured against the
  FP8/BF16 reference (Q-gates below).
- **MI300X cannot use OCP-FP8 checkpoints as they are.** Its E4M3 is the FNUZ variant
  (max 240, against OCP's 448), so weights need conversion (rescale or requantize) with
  a measured error bound.

## 2. Order

| # | Block | Why first | Hardware |
| --- | --- | --- | --- |
| P | Instinct platform: KFD runtime (locally on `gfx1100`), then `gfx942` smoke tests | prerequisite for everything | local, then 1×MI300X |
| Q1 | **Qwen3.8-27B on MI300X** (the model we already serve), FP8 safetensors | reuses zerv's model code and oracles; isolates backend bugs from model bugs | 1×MI300X |
| Q2 | Qwen3.5-35B-A3B → 122B-A10B → **397B-A17B**: the MoE FFN, one GPU first | adds only the MoE FFN to known layers | 1×MI300X/MI355X, then 2× |
| G1 | **GLM-5.3-Flash** | new: KDA, MLA+DSA, mHC, GLM tokenizer | 2×MI355X (FP8) |
| Q3 | Qwen3.8-2.4T-A95B | needs 8-GPU expert parallelism and 4-bit weights | 8×MI355X |

Every block follows the project loop: research → spec in `docs/specs/` → independent
oracle → implementation → Debug/ReleaseFast tests → component and serving benchmarks
against the best competitor on the same GPU (ATOM, SGLang, vLLM, recorded in
`docs/bench/`).

## 3. Research TODOs

### P · Platform (details: [Instinct backend](2026-09-26-instinct-backend.md))
- [ ] P0 Dependency-boundary approval (KFD uapi at runtime; LLVM assembler only as a tool).
- [ ] P1 `docs/specs/kfd.md`: device discovery, VM, BO ownership and lifetimes, queue
  sizing (context-save area × XCC), signals and events, bounded waits, errors and
  teardown.
- [ ] P2 Code-object format decision: standard AMDGPU ELF with kernel descriptors; our own
  loader; how metadata is carried. Assembler pinning.
- [ ] P3 Numerics probes on `gfx942`/`gfx950`:
  - MFMA FP32 accumulation order and exactness (F16/BF16/FP8/F8F6F4);
  - conversion tables for FNUZ-FP8, OCP-FP8, MXFP4/6 and their scales, checked
    exhaustively.
- [ ] P4 Component benchmarks with fixed shapes:
  - HBM bandwidth, dispatch latency, XCD placement, L2/Infinity Cache reuse;
  - SDMA and xGMI copy rates;
  - checked against rocm-bandwidth-test and hipBLASLt on the same VM.
- [ ] P5 Multi-GPU primitives:
  - peer memory mapping and `SDMA_XGMI` queues;
  - our own all-gather, reduce-scatter and all-to-all with a **fixed reduction order**, so
    results do not depend on timing (batch and parallelism invariance, as on the single
    card).

### Q1 · Qwen3.8-27B on Instinct
- [ ] Artifact: safetensors loader (sharded index; BF16 / FP8 128×128 block scales, with
  the `modules_to_not_convert` list honored) and its validation (hashes, tensor inventory
  against the index).
- [ ] Oracles:
  - the existing FP64/libllama gates cover the Q4_0 GGUF;
  - for FP8 safetensors, new fixtures from transformers on CPU at the pinned commit, per
    layer and end to end (27B in FP32 fits in this machine's 62 GB RAM only per layer, so
    fixtures are per layer plus reduced depth).
- [ ] Kernels on `gfx942`: FP8/BF16 MFMA GEMM (prefill), decode matvec at HBM rate, GDN
  chunked and recurrent, split-K attention, norms, RoPE, sampling. Each needs a bitwise or
  bounded-error contract.
- [ ] Serving parity with the RX 7900 XTX build (same API, prefix cache, `--parallel`),
  then benchmarks against ATOM, SGLang and vLLM on the same MI300X using the
  OpenRouter-shaped workload.

### Q2 · Qwen3.5 MoE (35B-A3B → 122B-A10B → 397B-A17B)
- [ ] Semantics, from transformers `qwen3_5_moe`, checked against ATOM and SGLang:
  - router: FP32 softmax → top-k → renormalize, including the **tie-breaking rule**, which
    decides determinism;
  - shared expert with its sigmoid gate;
  - expert SwiGLU;
  - MTP layer weights and use;
  - whether `mlp_only_layers` is empty.
- [ ] Tokenizer and template identity with Qwen3.8-27B. **Checked 2026-09-26** [O]:
  - 397B's `tokenizer.json` has the same vocab, merges, pre-tokenizer and normalizer as
    27B's. It lacks 7 added tokens (ids 248070–248076: audio and TTS markers).
  - Its chat template differs (sha `a4aee8af…`, 27B's is `c3cf9e34…`); still to diff and
    gate.
  - The tokenizer gates carry over with the added-token list adjusted.
- [ ] 4-bit artifacts:
  - GPTQ-Int4 format (group size, zero points, act-order) for 1×MI355X;
  - our own MXFP4 variant (`gfx950` has FP4 MFMA; `gfx942` needs dequantization to
    FP8/BF16);
  - **Q-gate:** KL divergence against BF16/FP8 on a fixed corpus plus task evals, reported
    as a quality tier, never as equal quality.
- [ ] MoE kernels:
  - token → expert grouping (stable sort; deterministic ordering);
  - grouped GEMM for prefill;
  - decode where each token reads 10 experts, which is weight-bandwidth bound: batch by
    expert, and measure at 1–64 concurrent sequences;
  - fused shared expert.
- [ ] Oracle for 397B:
  - per-layer fixtures from transformers on CPU (one layer ≈ 6.7B parameters ≈ 27 GB in
    FP32: feasible here);
  - reduced-depth end-to-end fixtures;
  - system-level check against ATOM/SGLang top-k log-probabilities on the rented GPU
    (different arithmetic: tolerances declared, not bitwise).
- [ ] 2-GPU (FP8) split: which layers go where, tensor parallelism vs expert parallelism.
  Needs P5.

### G1 · GLM-5.3-Flash (every item research first; none of it exists in zerv)
- [ ] **KDA (Kimi Delta Attention)**:
  - exact recurrent and chunked forms (transformers
    `recurrent/chunk_kimi_delta_attention`, `kimi_linear`);
  - the per-channel forget gate: a low-rank `f_a`→`f_b` projection plus `dt_bias`, then
    `lower_bound · sigmoid(exp(A_log) · g)` with lower bound −5, in FP32;
  - L2-normalized q/k, short conv, gated RMSNorm;
  - how far zerv's GatedDeltaNet kernels and oracle harness carry over (both are
    delta-rule recurrences);
  - state dtype (FP32 in the reference?) and batch invariance.
- [ ] **MLA + DeepSeek sparse attention (DSA)**:
  - latent KV (`kv_lora_rank` 512), absorbed projections for decode;
  - the indexer: 32 heads × 128; top-2,048 selection; `index_kpool` 4 with compression,
    and the tail always selected; indexer RoPE interleaving;
  - sharing with MTP;
  - **deterministic top-k** (ties) for batch invariance;
  - prefill and decode paths.
- [ ] **mHC hyper-connections**: 4 residual streams (4× activation traffic), FP32
  projections, the Sinkhorn loop (20 iterations, ε) — whether it fuses into one small
  kernel per block — the final mean collapse; numerical sensitivity (FP32 required?).
- [ ] **MoE**: sigmoid router with correction bias, `n_group` 1 / `topk_group` 1 (grouping
  is a no-op?), renormalize, × 2.5, shared expert, dense first 3 layers, SwiGLU clamp at
  10.
- [ ] **FP8 checkpoint**: exact quantization config (`fmt e4m3`, dynamic activations; block
  size?), the BF16 exceptions (attention MHA/MQA, hyper-connections, `dt_bias`, gates),
  and the FNUZ conversion for MI300X.
- [ ] **Tokenizer and template**:
  - GLM `tokenizer.json` (154,880 tokens), normalization and pretokenization;
  - chat template: reasoning and tool-call format;
  - a new tokenizer oracle (HF tokenizers) and output parser, following the Qwen
    tokenizer work (`docs/research/tokenizer*.md`).
- [ ] **Oracle**: per-layer and reduced-depth fixtures from transformers `glm5_next` at the
  pinned commit (one layer ≈ 7 GB BF16 ≈ 28 GB FP32: feasible here). System-level
  comparison against ATOM (which ships `glm5_next`) and SGLang.
- [ ] **Serving**: prefix cache for MLA latent KV + KDA state snapshots; 1M-context policy
  (memory budgeting like `--context max`); MTP in batched serving.

### Q3 · Qwen3.8-2.4T-A95B
- [ ] Semantics beyond 397B:
  - `output_gate_type: swish`, **resolved 2026-09-26** [O]:
    - Qwen3.8-27B's official config carries the same `"swish"`
      ([gpu-primitives](gpu-primitives.md)). zerv already computes that gate: the GDN
      output is RMSNorm × SiLU(z), and SiLU is swish. It passes the FP64/libllama gates.
    - SGLang passes the value to `RMSNormGated` as the activation; transformers hardcodes
      `silu`.
    - The deviating variant is Qwen3.8-Flash-Next's `sigmoid` (ATOM rejects anything else
      for it).
  - Tokenizer: `tokenizer.json` is **byte-identical** to Qwen3.8-27B's [O].
  - Chat template: the text-only branch of 27B's template, with vision removed and thinking
    mandatory (`enable_thinking=false` raises) [O]. It needs new rendering fixture cases,
    not a new renderer.
  - Multi-step MTP.
- [ ] Deployment:
  - 8×MI355X with MXFP4 weights (made by us; Q-gate against the official FP8 on per-layer
    fixtures and on served outputs of a reference engine), or 16 GPUs in FP8 over two
    nodes (scale-out networking: out of scope until demand exists);
  - expert-parallel all-to-all over xGMI (P5); recurrent state of 289 MB per sequence
    limits concurrency.
- [ ] Readiness estimate (2026-09-26, [D]; the effort weights are judgement, not
  measurement):

  | Part | Share of the total work | Done today | Basis |
  | --- | --- | --- | --- |
  | API, streaming, tokenizer, output parsing, sampling | ~10% | ~90% | tokenizer byte-identical; template is a text-only, thinking-mandatory subset of 27B's |
  | Layer semantics (GDN, gated attention, norms, partial RoPE, 1-step MTP), verified | ~10% | ~60% | verified for 27B, but dimensions are compile-time constants (`src/model/config.zig`), kernels are fixed to them, and MTP is 1-step |
  | Scheduler, prefix cache, memory at hundreds of sequences | ~10% | ~30% | 8 slots today; no prefix cache in `--parallel` |
  | MoE (router, grouped GEMM, shared expert) | ~10% | 0 | – |
  | Safetensors loading, FP8 block scales, own MXFP4 + quality gate | ~8% | ~5% | GGUF/mmap loader patterns only |
  | Instinct runtime (KFD) | ~10% | 0 | research note only |
  | `gfx950` kernels near roofline | ~25% | ~5% | designs transfer (split-K, chunked DeltaNet, flash attention); code does not |
  | 8-GPU expert/tensor parallelism, collectives | ~12% | 0 | – |
  | Oracle and fixtures | ~5% | ~30% | harness pattern exists. One 2.4T layer is ~26B parameters (~105 GB in FP32): per-layer fixtures need expert subsets or a larger host |

  **About 15–25% overall.** The parts that decide speed (Instinct kernels, MoE, multi-GPU)
  are at zero.
- [ ] Market check before building: $113k per 30 days on OpenRouter, 7 providers, flat
  $2/$6 ([bigger models](2026-09-26-bigger-models.md)). Proceed only if Q2/G1 are
  profitable.

## 4. Competitors to measure on the same hardware
- **AMD ATOM** (`ROCm/ATOM`, MIT) already ships `glm5_next`, `qwen3_5`, `qwen4_exp` and
  `kimi_k3`.
- **SGLang** ships `glm5_next`, `qwen3_5`, `kimi_linear` and `qwen4_exp`.
- **vLLM main** has `qwen3_5` but no `glm5_next` file at `2bb7605a`.
- These are the baselines for every Q/G block, run with the InferenceX-style agentic
  traces and our OpenRouter-shaped workload. They are already in the tracked set
  (`docs/performance.md`) as vLLM; ATOM and SGLang join it for Instinct work.
