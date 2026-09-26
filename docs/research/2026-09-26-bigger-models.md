# Bigger open models on rented AMD GPUs: market, fit and a gated plan — 2026-09-26

**Question.** Does the rented-AMD opportunity ([rented GPUs](2026-09-26-rented-gpu-economics.md))
extend to bigger open models? Candidates: the larger Qwen models, GLM-5.3(-Flash), Kimi K3,
MiniMax M3 and DeepSeek V4. And how could a business start while keeping the money at risk
gated on demand?

**Status.** Research only. Nothing was rented, signed up for or contacted. Public pages and
JSON APIs were read on 2026-09-26.

Labels: **[S]** sourced ([sources.json](2026-09-26-bigger-models/sources.json) and the raw
files next to it), **[D]** derived, **[A]** assumed.

Tables: `python3 tools/model_market_summary.py docs/research/2026-09-26-bigger-models` →
[tables.md](2026-09-26-bigger-models/tables.md).

## Answer

- **The market is 10–17× larger per model** [S]. On OpenRouter, the 30 days to 2026-09-25:

  | Model | Spend | Providers |
  | --- | --- | --- |
  | Kimi K3 | $4.8M | 19 |
  | GLM-5.3 | $4.4M | 40 |
  | GLM-5.3-Flash | $2.8M | 33 |
  | DeepSeek V4.1-Flash | $2.6M | 28 |
  | DeepSeek V4-Flash | $2.0M | 31 |
  | Qwen3.8-27B | $0.28M | 16 |

  The whole open Qwen `qwen3_5` family (27B, 35B-A3B, 122B, 397B, 2.4T) comes to about
  $0.74M.
- **Renting MI355X with AMD's own stack is already profitable for the high-priced models**
  [S, InferenceX profit estimator; agentic traces, InferenceX rent $2.90/GPU-h]:

  | Model | MI355X revenue per GPU-hour at 100% busy | Multiple of rent |
  | --- | --- | --- |
  | Kimi K3 | $15.78 (ATOM) | 5.4× |
  | MiniMax M3 | $13.43 | 4.6× |
  | GLM-5.2 | $4.92 | ~1.6–1.8× (thin) |
  | DeepSeek V4.1-Flash | $5.17 | ~1.6–1.8× (thin) |
  | DeepSeek V4-Pro | $4.74 | ~1.6–1.8× (thin) |

  **At current prices, making money does not require our engine.** An engine matters once
  prices compress, which is the pattern:
  - Kimi K2.5/K2.6 now sell at $2.25–4.00 output against K3's $15 [S].
  - DeepSeek V4-Flash sells at $0.28–0.35.
- **For these flagship MoEs, AMD's software gap is small; its hardware gap is not.**
  - Per GPU, MI355X's best stack delivers 0.5–1.3× the best 8-GPU B200/B300 result (Kimi
    K3 0.91× B300 and 1.33× B200) [S].
  - The big lead belongs to NVIDIA's 72-GPU racks: GB200/GB300 with disaggregated
    prefill/decode reach 1.5–4× MI355X per GPU (Kimi K3 $31.79, MiniMax M3 $51.57) [S].
  - MI355X nodes have 8 GPUs per scale-up domain, and software cannot change that. On
    flagship MoEs, a hand-rolled engine competes against AMD's own ATOM team on AMD's
    hardware, with a structural rack-scale deficit.
  - AMD's rack-scale answer is being sold: Vultr lists **MI455X** ("contact sales", 72 GPUs
    per rack over UALink) [S]. No price or benchmark exists for it yet.
- **Where our engine fits best: hybrid linear-attention models** [S, configs]. The
  delta-rule/linear-attention hybrid is now in several families:

  | Family | Hybrid layout |
  | --- | --- |
  | Qwen 3.5–3.8 | Gated DeltaNet + full attention |
  | Kimi K3 | 69 KDA (delta-rule) layers + 24 full-attention layers |
  | GLM-5.3-Flash | 34 linear-attention + 11 DeepSeek-sparse-attention layers |
  | Qwen3.8-Flash-Next | 36 linear + 12 full |

  zerv's DeltaNet kernels, state handling and batch-invariant decode apply to them. AMD's
  coverage of the Qwen hybrids is thin: InferenceX has only an MI325X row for Qwen3.5-397B
  and no MI355X row [S].
- **The size of each rental step jumps with model size** (24/7, 730 h) [S/D]. The cheap
  per-GPU prices come only as 8-GPU VMs:

  | GPUs per replica | Models | Rent per month (24/7) |
  | --- | --- | --- |
  | 1 | Qwen 27B / 35B-A3B / 122B at FP8 | $1.9k (one MI300X, DigitalOcean on-demand $2.59); $2.2k (one MI355X, spot $2.97) |
  | 2 | DeepSeek V4.1-Flash, MiniMax M3 | $4.3k (2 × MI355X spot) |
  | 4 | GLM-5.2 / DeepSeek V4-Pro (InferenceX's smallest runs) | $8.7k (4 × MI355X spot) |
  | 8 | Kimi K3 | $15.1k (8×MI355X VM, Vultr on-demand $2.59); $10.8k (8×MI300X, Vultr $1.85) |

  Kimi K3 is the prize, and its entry ticket is a whole node.
- **Licenses [S]:**
  - Apache-2.0: the Qwen 3.5/3.6/3.8 open models.
  - MIT: GLM-5.3-Flash and DeepSeek.
  - Kimi K3: selling API access is allowed below $20M revenue per 12 months.
  - MiniMax M3: requires "Built with MiniMax M3", and authorization above $20M/yr.
  - GLM-5.3: security review above $10B.
  - Qwen3.8-2.4T: extra terms above $50M.
  - **Qwen3.8-Flash-Next: its Qwen Community License requires a separate license before
    any Model-as-a-Service use.** It is excluded, which is presumably why only one provider
    serves it.

**Verdict.** Yes, it extends, but the order matters:
1. **Hybrid Qwen models on a single GPU first:** 27B, then 35B-A3B, 122B-A10B, and 397B-A17B
   (FP8 on 2 GPUs, or FP4 on 1). This is the smallest market, but it has the best
   architectural fit, the thinnest AMD coverage, Apache-2.0 licenses, and $1.4–2.2k/month
   rental units.
2. **Kimi K3 second**, once multi-GPU MoE with expert parallelism works. It has the largest
   margin, it is a delta-rule hybrid, and it needs one 8×MI355X node ($15k+/month).
3. **Commodity sparse-attention MoEs (DeepSeek V4, GLM-5.3, MiniMax M3) last.** Margins are
   thin, 13–40 providers compete, AMD's own stack is strong there, and NVIDIA's racks lead.

## 1. Market [S]

OpenRouter, 30 full days to 2026-09-25 ([raw](2026-09-26-bigger-models/openrouter-models.json)).
Prices are USD per 1M tokens across the listed endpoints.

| model | license | params | providers | input median | output min / median / max | spend per 30 d | requests/day | prompt:completion | cached |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-27b | apache-2.0 | 28B | 16 | 0.22 | 1.80 / 2.55 / 4.40 | 284k | 1.87M | 22 | 71% |
| qwen/qwen3.6-35b-a3b | apache-2.0 | 36B | 10 | 0.14 | 0.70 / 1.00 / 1.80 | 56k | 1.84M | 8 | 40% |
| qwen/qwen3.5-122b-a10b | apache-2.0 | 125B | 5 | 0.29 | 2.08 / 2.40 / 3.20 | 26k | 0.22M | 7 | 0% |
| qwen/qwen3.5-397b-a17b | apache-2.0 | 403B | 10 | 0.55 | 2.34 / 3.55 / 4.50 | 126k | 0.45M | 7 | 40% |
| qwen/qwen3.8-2.4t-a95b | other | 2,446B | 7 | 2.00 | 6.00 / 6.00 / 6.00 | 113k | 0.13M | 16 | 85% |
| qwen/qwen3.8-flash | other (no MaaS) | 180B | 1 | 0.15 | 0.47 | 61k | 1.42M | 28 | 88% |
| z-ai/glm-5.3-flash | mit | 321B | 33 | 0.15 | 0.14 / 0.50 / 1.00 | 2,822k | 49.08M | 37 | 84% |
| z-ai/glm-5.3 | other | 753B | 40 | 1.35 | 1.19 / 4.40 / 8.80 | 4,412k | 6.15M | 57 | 89% |
| moonshotai/kimi-k3 | other | ~2.8T stored (MXFP4) | 19 | 3.00 | 9.04 / 15.00 / 22.50 | 4,782k | 3.76M | 59 | 89% |
| minimax/minimax-m3 | other | 427B | 13 | 0.30 | 0.96 / 1.20 / 3.00 | 234k | 4.13M | 57 | 87% |
| deepseek/deepseek-v4.1-flash | mit | 763B stored | 28 | 0.22 | 0.29 / 0.87 / 1.50 | 2,598k (16 d scaled) | 35.58M | 63 | 89% |
| deepseek/deepseek-v4-flash-0731 | mit | 304B stored | 31 | 0.14 | 0.13 / 0.35 / 1.32 | 1,983k | 71.18M | 26 | 81% |
| deepseek/deepseek-v4-pro-0813 | mit | 1,650B stored | 22 | 1.09 | 0.75 / 3.73 / 4.95 | 972k | 4.20M | 40 | 86% |

- **Parameter counts.** "params" is the Hugging Face safetensors element count. For packed
  FP4/INT checkpoints ("stored") it counts storage elements, not logical parameters.
- **What buyers paid per request** (spend ÷ requests, 30 days) [D]:
  - Kimi K3 42 m$, GLM-5.3 24 m$, Qwen3.8-2.4T 29 m$, Qwen3.5-397B 9.4 m$, DeepSeek
    V4-Pro 7.7 m$, Qwen3.8-27B 5.1 m$.
  - GLM-5.3-Flash, MiniMax M3 and DeepSeek V4.1-Flash: 1.9–2.4 m$.
  - DeepSeek V4-Flash: 0.5–0.9 m$.
- **Traffic shape.** The flagship traffic is agentic: 40–60 prompt tokens per completion
  token, 81–89% of them cached.

## 2. Serving on MI355X today [S]

InferenceX profit estimator ([raw](2026-09-26-bigger-models/inferencex-profit-estimator.json)):
- agentic-trace benchmark at each model's default interactivity target;
- the model's *default* OpenRouter price (for GLM-5.3 that is the bottom of the range:
  $1.19 against a $4.40 median);
- revenue per GPU-hour at 100% busy, against InferenceX's rent rate.

| InferenceX model | target tok/s/user | price in / cached / out | MI355X best | MI355X revenue / rent | best 8-GPU NVIDIA | best rack-scale NVIDIA | MI355X GPUs per replica |
|---|---|---|---|---|---|---|---|
| Kimi-K3 | 45 | 3 / 0.3 / 15 | 15.78 (ATOM) | 5.4× | 17.36 (B300 vLLM) | 31.79 (GB300 Dynamo) | 8 |
| MiniMax-M3 | 83 | 0.3 / 0.06 / 1.2 | 13.43 (ATOM) | 4.6× | 22.44 (B300 TRT) | 51.57 (GB200 Dynamo) | 2–8 |
| GLM-5.2 | 100 | 0.379 / 0.07 / 1.19 | 4.92 (ATOM) | 1.7× | 6.39 (B200 Dynamo) | 12.72 (GB300 Dynamo) | 4, 8 |
| DeepSeek-V4.1-Flash | 125 | 0.3 / 0.006 / 1.2 | 5.17 (ATOM) | 1.8× | 6.00 (B300 SGLang) | 7.63 (GB200 SGLang) | 2, 4 |
| DeepSeek-V4-Pro | 24 | 0.264 / 0.009 / 0.792 | 4.74 (SGLang) | 1.6× | 9.91 (B300 vLLM) | – | 4, 8 |
| Qwen-3.5-397B-A17B | 45 | 0.55 / 0.22 / 3.5 | no MI355X row (MI325X SGLang 4.50) | – | no row | no row | – |

**Example: one 8×MI355X Kimi K3 node** [D]:
- At 60% busy, revenue is 0.6 × $15.78 × 8 = $76/h, against $20.7/h rent (Vultr on-demand,
  8×MI355X at $2.59). That leaves about **$40k/month gross margin**, before people, platform
  fees and reserve capacity.
- The node would need about **1.2%** of Kimi K3's OpenRouter spend.
- It rests on today's $15 price and on ATOM's throughput. Our engine is not required.

## 3. Fit with our engine

| Family | Attention | Our reuse | AMD stack status [S] | Min MI355X |
| --- | --- | --- | --- | --- |
| Qwen 3.5–3.8 dense and MoE (27B … 397B) | Gated DeltaNet + full attention (1 in 4) | high: same layers, plus a MoE FFN | thin (397B: MI325X only) | 1 (FP8 up to 122B; 397B in FP4) |
| Kimi K3 | KDA delta-rule (69) + MLA full (24), latent MoE 896 experts, 16 active + 2 shared | medium: delta-rule recurrence, MoE and MLA new | ATOM and vLLM, TP8 | 8 |
| GLM-5.3-Flash | linear attention (34) + DeepSeek sparse attention (11) | medium | not in InferenceX | 2 (FP8) |
| GLM-5.3, DeepSeek V4.x, MiniMax M3 | sparse-attention indexers, MLA/MQA, large MoE | low | strong: ATOM, MoRI, SGLang, Mooncake | 2–8 |

- **Multi-GPU MoE is a new subsystem.** It needs expert-parallel all-to-all over xGMI, and
  FP4 (MXFP4) matrix kernels. Under AGENTS.md, AMD's collective library (RCCL) is off-limits
  (C++), so we would write our own collectives. This comes on top of the Instinct backend
  prerequisite in [rented GPUs](2026-09-26-rented-gpu-economics.md#5-platform-prerequisite-open-needs-research-and-your-approval).

## 4. Gated plan (proposal; spending is the operator's decision)

**Principles:**
- Set one hard risk budget up front, kept in its own account or entity. Spend nothing from
  personal savings beyond it without a gate passed on evidence.
- Rent by the hour or spot for development, and make no 12-month reservations until revenue
  covers them.
- Every gate produces a public, reproducible artifact: benchmark reports in `docs/bench/`
  style and code, if released. That makes the same work evidence for customers and for a
  prospective employer such as AMD.
- Entity, VAT and liability questions belong with a tax adviser, not here.

| Stage | Spend (order of magnitude) | Work | Gate to pass before the next stage |
| --- | --- | --- | --- |
| 0 | $0 | On the RX 7900 XTX: prefix cache and long context in `--parallel`, FP8/16-bit state. Research the CDNA backend (KFD submission, MFMA ISA, code objects) and write the boundary proposal | proposal approved; features pass the correctness gates |
| 1 | ≤ ~$500; AMD Developer Cloud offers $100 of credits or complimentary MI300X hours after sign-up | Measure vLLM, SGLang and ATOM on MI300X/MI355X for Qwen3.8-27B (and 35B-A3B / 122B) on InferenceX-style and OpenRouter-shaped workloads. First Zig KFD dispatch and one MFMA GEMM | our kernel ≥ the best stack on a real shape, and ≥ 1.5× headroom to roofline confirmed |
| 2 | ~$2–5k over months, spot or hourly, 1 GPU | CDNA backend for the Qwen hybrids on one GPU: correctness gates plus full serving | ≥ 1.5× lower cost per request than the best measured stack at equal interactivity, reproducible and published |
| 3 | capped at ~3 months of 1–2 GPUs (~$5–15k) | Go live for one Qwen model. This needs an aggregator admission or direct customers, which is a contact decision | utilization above break-even and positive gross margin for 2 consecutive months; otherwise stop |
| 4 | from revenue only | Multi-GPU MoE, then Kimi K3 on one 8×MI355X node ($15k+/month) | node margin positive at the *then* Kimi price, not today's |

**Notes on the gates:**
- Stages 0–2 have value even if the business never starts: they are the portfolio that makes
  the AMD outcome plausible.
- From Stage 3 onward, rent stops being the dominant cost. The real questions are admission,
  uptime and support.

**Direction (user, 2026-09-26): AMD-first ("going all in on AMD").**
- This is not yet a work-queue change. Implementation needs an explicit `TODO.md` queue
  change, and approval of the Instinct backend's dependency boundary
  ([rented GPUs, section 5](2026-09-26-rented-gpu-economics.md#5-platform-prerequisite-open-needs-research-and-your-approval)).
- The case for MI355X is the lowest engine efficiency needed of any rented GPU, but it is
  narrower than first reported:
  - after the Vultr price correction, a rented B200 needs 65% against MI355X's 46%;
  - see the risks in the rented-GPU report.

## 5. Open questions

- Current dense and hybrid Qwen numbers on MI300X/MI355X for the best stacks. This is
  Stage 1, and paid.
- Whether MI355X stays near $2.6–3/h. Vultr on-demand is 8-GPU only, and DigitalOcean's is
  spot. Rental prices rose through 2026 ([rented GPUs](2026-09-26-rented-gpu-economics.md)).
- How fast Kimi K3's price follows K2.x down. It sets whether Stage 4 is ever reached.
- Admission criteria of aggregators for a small provider. Not asked (no contact).
- Kimi K3's exact logical parameter count, and whether it fits one MI355X node's 2.3 TB with
  headroom. InferenceX runs it at TP8 on one node [S]; the checkpoint stores ~2.78 TB.
