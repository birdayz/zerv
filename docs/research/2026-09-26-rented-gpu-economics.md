# Could a faster engine on rented GPUs outcompete? Qwen3.8-27B — 2026-09-26

**Question.** Suppose zerv became the fastest engine per GPU, hand-tuned to the hardware
limit. Could a business that rents datacenter GPUs (MI355X, MI300X, H100/H200/B200) sell
Qwen3.8-27B tokens more cheaply than providers that own theirs? Renting costs more than
owning; buying MI355X-class hardware is too much upfront for now.

**Status: research and estimates.**
- No inference ran for this report, and no service was contacted or signed up for.
- Prices and benchmarks come from public pages and public JSON APIs read on 2026-09-26.
- Follows [selling tokens from one RX 7900 XTX](2026-09-26-token-economics.md).

Labels: **[S]** sourced ([sources.json](2026-09-26-rented-gpu-economics/sources.json) and the
raw files next to it), **[D]** derived by the model below, **[A]** assumed.

Reproduce all tables: `python3 tools/rented_gpu_economics.py
docs/research/2026-09-26-rented-gpu-economics/inputs.json
docs/research/2026-09-26-rented-gpu-economics/inferencex-llama70b.json`, which writes
[tables.md](2026-09-26-rented-gpu-economics/tables.md). Test: `tests/test_rented_gpu_economics.py`.

## Answer

**Corrected later the same day.** Vultr lists on-demand, hourly-billed prices, but only as
8-GPU VMs: MI355X $2.59, MI300X $1.85, MI325X $2.00, B200 $3.50 and H100 $2.30 per GPU-hour
[S]. They are now the "public rent" for those GPUs. The first version used
$2.97 (MI355X spot) and $6.79 (B200), and concluded that renting could only compete on AMD.
With the Vultr prices, rented B200 is competitive too. MI355X still needs the lowest engine
efficiency.

1. **At today's Qwen3.8-27B prices, yes, and you do not need the fastest engine.**
   - Buyers paid **5.3 m$ per request** on OpenRouter on 2026-09-25 [S].
   - Serving that mean request on a rented GPU at full load costs **0.51–1.2 m$** with the
     engines InferenceX measured in Oct 2025, at public rental prices [D].
   - **Renting is not the obstacle; traffic is.** The whole OpenRouter demand for this model
     (2.5 M requests/day, ~$284k over the last 30 days [S]) fits on **~15–55 GPUs** with
     those engines, or **~7–17** with a strong one (table D) [D]. Sixteen providers already
     split it.
2. **At commodity prices, renting works only with a much better engine.** Commodity here
   means what a ~30B dense model sells for 17 months after release: Qwen3-32B at
   $0.08 / $0.28, or 0.82 m$ per request [S].
   - With the Oct-2025 engines, rented H100, MI300X and MI325X never break even. Rented B200,
     MI355X and H200 need 61–95% utilization.
   - With an engine at 70% of the roofline [A]:
     - rented MI300X, MI325X, MI355X or B200 break even at **20–28%** utilization and earn
       **$2.0–3.8k per GPU per month** at 60%;
     - rented H100 or H200 need 46–54% (table C).
3. **Against owners who run today's engines, renting can win on MI355X, MI300X/MI325X or B200,
   but only well above today's efficiency.** In a competitive market those owners set the
   price floor.
   - At 50 tok/s per user (the market median), the cheapest owner is a B200 at
     **0.25 m$/request** [D].
   - To match it, a renter needs this share of the roofline:

     | Rented GPU | Share of roofline needed | Speedup over the Oct-2025 engines |
     | --- | --- | --- |
     | MI355X | **46%** | 2.3× |
     | MI300X / MI325X | 55–56% | ~4× |
     | B200 | 65% | 2.0× |
     | H100 / H200 | 103–123% | not achievable |

   - At 30 tok/s per user (bigger batches), an *owned* MI355X becomes the floor. A renter then
     needs 83% of roofline on MI355X, and ≥ 100% on everything else.
4. **Any engine lead is temporary, and owners with equal software always win.**
   - In 8 months, InferenceX's single-node DeepSeek-R1 throughput per GPU rose 2–9× at
     ≥50 tok/s per user [S], most of it on AMD:

     | GPU, precision | Oct 2025 (tok/s per GPU) | Later (tok/s per GPU) | Gain |
     | --- | --- | --- | --- |
     | MI355X, FP8 | 220 | 1,994 (ATOM + MTP, Jun 2026) | 9× |
     | B200, FP4 | 2,020 | 6,216 (May 2026) | 3.1× |
     | H200, FP8 | 512 | 1,082 (Sep 2026) | 2.1× |

   - Public rent is **1.7–2.8× the owner's hourly cost** [S/D]. Once owners run software as
     good as ours, they undercut us by that factor.

**Verdict.**
- **Near term:** a rent-first business is viable because new-model prices carry large
  margins. What limits it is winning traffic, not the engine or the rent.
- **Later:** the engine's job is to keep the business viable as prices fall to commodity
  levels.
  - **Best candidate: MI355X.** It needs the least efficiency (46% of roofline at 50 tok/s
    per user), rents for $2.59/GPU-h on-demand (8-GPU VM) against $3.50 for a B200 with the
    same 8 TB/s, and has 288 GB of HBM.
  - **B200 is a real alternative** at 65%, against TensorRT-LLM-class competition.
  - Either way the engine has to keep up with ATOM, SGLang and TensorRT-LLM, which close gaps
    quickly.
- **H100/H200 cannot compete on cost** against owners.
- **Prerequisite (section 5):** zerv runs on Vulkan on RDNA3 today. An Instinct GPU needs a
  new backend, and possibly an expansion of the dependency boundary that needs your approval.

## 1. Inputs

### GPU rental prices, USD per GPU-hour, read 2026-09-26 [S]

| GPU | Mem / BW / FP8 dense | Owning (InferenceX, hyperscaler volume) | InferenceX rent | Public prices | Rent used / owning |
| --- | --- | --- | --- | --- | --- |
| H100 SXM | 80 GB / 3.35 TB/s / 1,979 TF | 1.17 | 2.00 | **Vultr 2.30 (8-GPU VM)**; Vast median 3.14; RunPod 3.49; Nebius 3.85 (4.50 from Oct 1); Crusoe 3.90; Lambda 3.99–4.29; DigitalOcean 4.41, 3.26 reserved 12 months | 2.0× |
| H200 | 141 GB / 4.8 / 1,979 | 1.22 | 2.90 | **DigitalOcean 3.40 reserved**, 4.47 on-demand; Crusoe 4.29; Nebius 4.50 (5.40); Vast median 4.74 | 2.8× |
| B200 | 180 GB / 8.0 / 4,500 | 1.73 | 3.70 | **Vultr 3.50 (8-GPU VM)**; Lambda 6.79–6.99; Nebius 7.15 (8.50); Vast median 9.38 | 2.0× |
| MI300X | 192 GB / 5.3 / 2,615 | 0.95 | 1.30 | **Vultr 1.85 (8-GPU VM)**; DigitalOcean 1.91 reserved, 2.59 on-demand (1 or 8 GPUs); Hot Aisle 2.99 VM (was 1.99), 3.39 bare metal; Crusoe 3.45 | 1.9× |
| MI325X | 256 GB / 6.0 / 2,615 | 1.10 | 1.60 | **Vultr 2.00 (8-GPU VM)**; DigitalOcean 2.88 reserved, 3.80 on-demand | 1.8× |
| MI355X | 288 GB / 8.0 / 5,033 | 1.50 | 2.90 | **Vultr 2.59 (8-GPU VM)**; DigitalOcean spot 2.97 (interruptible); Crusoe: contact sales | 1.7× |

- **Sources.** Owning and InferenceX rent come from InferenceX's options API, built on the
  SemiAnalysis AI Cloud TCO model. InferenceX gives MI355X an owning cost of $1.50/h against
  $1.73/h for B200.
- **"Rent used"** (bold) is the cheapest public price per GPU: on-demand, reserved or spot.
  Vultr's cheap on-demand prices come only as 8-GPU VMs, for example $20.72/h for 8×MI355X.
  A single GPU costs more: DigitalOcean charges $2.59/h for one MI300X and $4.41/h for one
  H100.
- **The rental market is tight** [S]:
  - Hot Aisle raised new MI300X rentals from $1.99 to $2.99 on 2026-07-14, citing 100% capacity.
  - Nebius raises H100/H200/B200 prices on 2026-10-01.
  - DigitalOcean introduced new pricing on 2026-08-01.
- **Small starts are possible.** Single-GPU rentals exist for MI300X (DigitalOcean, Hot
  Aisle, AMD Developer Cloud) and MI325X (DigitalOcean). MI355X is available as an 8-GPU VM
  (Vultr) or spot (DigitalOcean).

### Request shape and prices [S]

**Mean OpenRouter request for this model** (2026-09-25): 5,584 uncached input tokens,
12,891 cached input tokens and 976 output tokens (see
[activity json](2026-09-26-token-economics/openrouter-qwen3.8-27b-activity.json)).

Revenue per request under three price scenarios:

| Scenario | Prices (input / cached / output, $ per 1M) | Revenue per request |
| --- | --- | --- |
| Paid today | What buyers actually paid: $13,431 over 2,534,379 requests | **5.30 m$** |
| Median list | $0.22 / $0.085 / $2.55 | 4.81 m$ |
| Commodity | $0.08 / $0.008 [A] / $0.28 (Qwen3-32B, DeepInfra) | **0.82 m$** |

### Today's engines [S]

- **Qwen3.8-27B has no InferenceX data.** InferenceX lists it as "experimental" but has no
  rows and no benchmark config.
- **The closest measured dense model** is Llama-3.3-70B FP8 at 8k input / 1k output
  (snapshot 2025-10-29; the model has since been deprecated there).
- **Best total tok/s per GPU at each interactivity level**
  ([raw](2026-09-26-rented-gpu-economics/inferencex-llama70b.json)):

  | GPU | ≥ 30 tok/s per user | ≥ 50 tok/s per user |
  | --- | --- | --- |
  | H100 | 1,607 (vLLM, TP4) | 1,071 (vLLM, TP8) |
  | H200 | 3,019 (TRT-LLM, TP4) | 2,254 (TRT-LLM, TP4) |
  | B200 | 6,062 (TRT-LLM, TP2) | 4,529 (TRT-LLM, TP4) |
  | MI300X | 1,598 (vLLM, TP4) | 1,041 (vLLM, TP8) |
  | MI325X | 1,574 (vLLM, TP8) | 1,154 (vLLM, TP8) |
  | MI355X | 4,665 (vLLM, TP1) | 2,280 (vLLM, TP4) |

- **These are dated.** The DeepSeek-R1 history quoted in the Answer shows how much engines
  have gained since ([raw](2026-09-26-rented-gpu-economics/inferencex-dsr1-history-best.json)).
  "Today's engine" below therefore means Oct-2025 engines. That **overstates our advantage**,
  most of all on AMD.

## 2. Model [D]

Per GPU, for the mean request (`tools/rented_gpu_economics.py` documents the formulas):

- **Prefill is compute-bound.** 48.7 GFLOP per token for the linear layers, plus
  attention over the cached prefix and the new tokens: 16 attention layers ×
  393,216 FLOP per context position. Both come from the
  [model config](2026-09-22/model-config.json).
- **Decode is memory-bound.** Each step reads 26.4 GB of FP8 weights once. Per sequence, it
  also reads FP8 KV (32 KiB per token × ~19k tokens) and reads and writes 75 MiB of 16-bit
  DeltaNet state. The batch is limited by bandwidth time, compute and HBM capacity.
- **Prefill steals time between decode steps.** A user sees step time / (1 − prefill
  share). The model picks the batch that maximizes requests/s at the interactivity target.
- **One efficiency η** scales compute and bandwidth together. For today's engines, η is
  fitted per GPU: the same model, applied to Llama-3.3-70B at InferenceX's workload and
  tensor-parallel size, is made to reproduce the measured throughput.
  - The fitted values at 50 tok/s per user are: H100 0.26, H200 0.41, B200 0.33, MI300X 0.17,
    MI325X 0.17, MI355X 0.22.
  - At 30 tok/s per user: 0.35, 0.41, 0.43, 0.23, 0.17, 0.49.
- **"Our engine"** is η = 0.5 / 0.7 / 0.85 [A].
- **Calibration on our own card** [D from existing measurements]:
  - zerv's decode streams weights at 720–915 GB/s of ~960 (~80%);
  - its prefill runs at ~64 TFLOP/s against the measured 135 TFLOP/s f16 WMMA peak (~47%).
  - η = 0.7 is therefore ambitious but not outlandish, on hardware we have already tuned.
- **Prefill share.** For this traffic, prefill takes ~30% of GPU time at η = 0.7. Prefill
  efficiency matters as much as decode.

**Limitations:**
- Tensor parallelism is modeled with no communication cost, which is optimistic for TP > 1.
  This makes the fitted η for TP4/TP8 rows too low and the roofline too high.
- No MTP or other speculation: every engine would gain from it.
- FP8 everywhere; FP4 on B200 and MI355X is an upside, with a quality trade-off.
- DeltaNet recurrence FLOPs are ignored (~0.2 GFLOP per token).
- No memory is reserved for idle cached conversations.
- One scalar η for compute and bandwidth.
- The results are estimates for comparison, not a simulator.

## 3. Results (≥ 50 tok/s per user; the ≥ 30 tables are in [tables.md](2026-09-26-rented-gpu-economics/tables.md))

**Cost per request at 100% busy, milli-USD** (× 1.025 = USD per 1M output tokens, with all
cost charged to output):

| GPU | Today's engine, owned | Today's, rented (InferenceX rate) | Today's, rented (public) | Ours η 0.5, rented | Ours η 0.7, rented | Ours η 0.85, rented | Roofline, rented |
|---|---|---|---|---|---|---|---|
| H100 SXM | 0.60 | 1.03 | 1.18 | 0.54 | 0.38 | 0.30 | 0.26 |
| H200 SXM | 0.28 | 0.67 | 0.79 | 0.64 | 0.44 | 0.36 | 0.31 |
| B200 | 0.25 | 0.53 | 0.51 | 0.33 | 0.23 | 0.19 | 0.16 |
| MI300X | 0.50 | 0.68 | 0.97 | 0.29 | 0.20 | 0.16 | 0.14 |
| MI325X | 0.54 | 0.79 | 0.99 | 0.29 | 0.20 | 0.16 | 0.14 |
| MI355X | 0.33 | 0.64 | 0.57 | 0.23 | 0.17 | 0.14 | 0.11 |

**Efficiency a renter needs to match the cheapest owner running today's engines**
(owned B200, 0.25 m$ per request):

| GPU (rented) | η needed at public rent (> 1 = impossible) | η needed at InferenceX rent | Speedup over today's engine on that GPU |
|---|---|---|---|
| H100 SXM | 1.03 | 0.89 | 4.7× |
| H200 SXM | 1.23 | 1.05 | 3.1× |
| B200 | 0.65 | 0.68 | 2.0× |
| MI300X | 0.55 | 0.39 | 3.9× |
| MI325X | 0.56 | 0.45 | 4.0× |
| MI355X | 0.46 | 0.51 | 2.3× |

**Break-even utilization and monthly margin per rented GPU** (public rent, USD, 730 h;
selected rows, all in tables.md):

| GPU | Engine | Break-even, paid today | Break-even, commodity | Margin at 60%, paid today | Margin at 60%, commodity |
|---|---|---|---|---|---|
| B200 | today's | 10% | 61% | 13,520 | −58 |
| B200 | ours η 0.7 | 4% | 28% | 32,259 | 2,852 |
| H100 SXM | ours η 0.7 | 7% | 46% | 12,543 | 530 |
| MI300X | today's | 18% | never | 3,093 | −660 |
| MI300X | ours η 0.7 | 4% | 24% | 20,095 | 1,980 |
| MI355X | today's | 11% | 69% | 8,698 | −246 |
| MI355X | ours η 0.7 | 3% | 20% | 34,522 | 3,764 |
| MI355X | ours η 0.85 | 3% | 16% | 42,561 | 5,013 |

- **How to read the "paid today" margins.** They only hold while buyers keep paying today's
  prices and the GPU can be filled. One rented GPU at 60% with our engine would take 4–9% of
  the entire OpenRouter traffic for this model (table below).

**GPUs needed for all of this model's OpenRouter traffic** (2,534,379 requests/day, 100%
busy) [D]:

| GPU | Today's engine | Ours η 0.7 |
|---|---|---|
| H100 SXM | 54.3 | 17.2 |
| H200 SXM | 24.4 | 13.8 |
| B200 | 15.2 | 7.0 |
| MI300X | 55.2 | 11.4 |
| MI325X | 52.1 | 10.6 |
| MI355X | 23.2 | 6.7 |

**The price of speed** (ours η 0.7, rented, one GPU per replica, no speculation): cost per
request, m$.

| GPU | ≥ 30 | ≥ 50 | ≥ 100 | ≥ 150 tok/s per user |
|---|---|---|---|---|
| B200 | 0.26 | 0.30 | 0.44 | 0.79 |
| MI300X | 0.24 | 0.30 | 0.70 | infeasible |
| MI355X | 0.19 | 0.21 | 0.31 | 0.58 |

- **What a fast tier earns elsewhere.** Fireworks charges 1.5× for "Fast", and Groq sells
  this model at 450 tok/s for $0.80 / $4.00 ([token economics](2026-09-26-token-economics.md)).
  At a 1.5× premium, a 100 tok/s tier on MI355X or B200 pays for itself: cost rises
  ~1.5× from 50 tok/s.
- **Speculation helps most here.** zerv's MTP gives 2.1–2.4× single-stream on the RX 7900
  XTX ([report](../bench/2026-09-24-speculative.md)).

## 4. What "the fastest engine" would have to be

- **Efficiency.** ≥ 46% of roofline on MI355X (55% on MI300X, 65% on B200) at 50 tok/s per
  user, on long (~19k-token), cache-heavy agentic requests. That means prefill *and* decode near their
  limits: prefill takes ~30% of GPU time here.
- **Features the traffic requires**, none of which zerv has in parallel mode today:
  - a **prefix cache** (71% of the market's prompt tokens are cache hits);
  - **FP8 KV** and 16-bit recurrent state, with 100+ concurrent sequences at 19k+ context;
  - 262k context;
  - MTP inside batched serving;
  - multi-GPU or multi-replica serving and scheduling.
- **Speed of iteration.** The target moves 2–9× per year on AMD. Model churn is also fast
  (Qwen3.5 → 3.6 → 3.8 in about six months), although these three share the `qwen3_5`
  architecture.
  - A per-architecture, hand-tuned engine pays off only if one kernel set covers a family
    of models. vLLM and SGLang support new models on release day.

## 5. Platform prerequisite (open, needs research and your approval)

- **zerv today** is raw Vulkan on RADV for RDNA3 (`docs/hardware.md`). Its kernels are
  WMMA-specific and tuned to that card.
- **Instinct GPUs** (CDNA3/4 with MFMA, HBM) need new kernels. They also need a way to
  submit work.
  - Whether any Vulkan driver exposes MI300X or MI355X compute is **not verified**.
  - AMD's ROCm/HIP runtime is C++, which AGENTS.md forbids as a dependency. The remaining
    route is the kernel's KFD interface through its C ABI, with our own code objects.
  - The boundary for that route has to be recorded, and you have to approve it before any
    work starts (AGENTS.md).
- **NVIDIA** would need the CUDA driver API (a C ABI) and PTX. The same boundary decision
  applies.

## 5a. AMD access programs (read 2026-09-26, nothing signed up for) [S]

From [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html),
[AMD cloud access](https://www.amd.com/en/developer/resources/cloud-access.html),
[getting started](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html)
and [AI Developer Program](https://developer.amd.com/ai-developer-program):

| Program | For | Hardware | Cost |
| --- | --- | --- | --- |
| AMD Developer Cloud | independent developers, open-source contributors (ML, low-level GPU programming) | MI300X: 1× (192 GB, 20 vCPU, 240 GB RAM, 720 GB boot, 5 TB scratch) or 8× | free credit with approval; otherwise pay-as-you-go through DigitalOcean ($2.59/GPU-h for 1× on-demand) or Vultr ($1.85/GPU-h, 8× only) |
| Instinct Evaluation Program | start-ups, enterprises, ISVs evaluating commercial deployment | MI300X, MI325X via partner clouds | free with approval; answer within 2 weeks |
| Radeon Test Drive | workstation developers (local AI inference) | Radeon AI PRO R9700 (RDNA4), PRO W7900 (48 GB RDNA3, same `gfx1100` as our card), W7800; via Colfax (NA), Comino (EU) | free with approval, up to 14 days |
| AI & HPC Cluster | academic and non-profit research | MI325X, MI300X, MI250, MI210 | free with approval, up to 1 year |

- **Credit.** Members of the AI Developer Program (free to join) get **$100** of Developer
  Cloud credit, expiring 30 days after deposit.
  - The cloud-access FAQ still describes an older offer: 25 hours (~$50) expiring after 10
    days. The two AMD pages disagree.
  - Credit is granted at AMD's discretion from a use-case description, needs a valid card,
    and covers only MI300X time, not storage.
  - Powered-off VMs are billed until destroyed. VMs are destroyed when credit runs out with
    no payment method on file.
- **Images.** Bare Ubuntu (install any ROCm, or none) or ROCm "Quick Start" images with vLLM,
  SGLang, PyTorch, Megatron and JAX containers. A bare VM is what a direct-KFD Zig backend
  would need; whether `/dev/kfd` is exposed in these VMs is not stated (to verify).
- **Terms.** The user may not *"rent, lease, … provide access to … AMD Developer Cloud to a
  third party or use AMD Developer Cloud to provide a service to a third-party"*. Development
  and benchmarking only, **no paid serving**. The user owns the work. The terms also contain
  export-control and patent-related restrictions.
- **Billing.** DigitalOcean GPU Droplets bill per second with a 5-minute minimum.
- **Price after the credit.** AMD publishes no Developer Cloud-specific rate: its pages,
  the vLLM forum announcement and lablab.ai only say "pay-as-you-go". The rate the
  DigitalOcean-hosted console (devcloud.amd.com) charges is visible only after login. It
  is assumed to equal DigitalOcean's list price, $2.59/GPU-h for MI300X on-demand
  (unverified).
- **Next generation.** Vultr lists **MI455X** "contact sales", with *"UALink over Ethernet
  connects all 72 GPUs inside one rack"*. AMD's rack-scale answer to GB200/GB300 NVL72 is
  therefore being offered.

## 6. Risks and open questions

- **Demand.** The whole OpenRouter market for this model is a few GPUs. A business needs
  several models, or direct customers. Admission to OpenRouter was not checked (no contact).
- **Price decay.** Qwen3.5/3.6-27B still sell at $1.56–3.25 output. Qwen3-32B has fallen to
  $0.28–0.57 after 17 months ([token economics](2026-09-26-token-economics.md)). When and
  how far the Qwen3.8-27B price falls sets the payback window.
- **Rental supply.**
  - Prices are rising (Hot Aisle, Nebius).
  - The cheap on-demand prices (Vultr) come only as 8-GPU VMs, $15–20k/month for 24/7 use;
    spot capacity can be reclaimed.
  - Uptime is bounded by the provider's, while OpenRouter demands 95%.
- **Stale competitor data.** A current dense-model data point (vLLM, SGLang, ATOM on MI300X
  or MI355X with Qwen3.8-27B) would sharpen every "today's" number. Getting it needs a paid
  rental; that is your decision.
- **Model validity.** The roofline model has not been checked against a measured Qwen3.8-27B
  run on any datacenter GPU.

Follow-up: [bigger open models on rented AMD GPUs](2026-09-26-bigger-models.md).
