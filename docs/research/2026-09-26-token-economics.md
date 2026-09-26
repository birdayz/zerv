# Unit economics: selling Qwen3.8-27B tokens from one RX 7900 XTX — 2026-09-26

**Question.** What could this machine earn by selling Qwen3.8-27B (Q4_0) inference through
`POST /v1/chat/completions` at market rates, what does it cost, and is it better than renting
the GPU out? Result as a sensitivity range, with break-even assumptions.

**Status: research and estimates.** No inference ran for this report. Performance numbers are
existing measurements in `docs/bench/`. Prices come from public pages and public JSON APIs
read on 2026-09-26, without accounts, sign-ups or contact.

Labels: **[M]** measured on this machine, **[S]** sourced (URL and date in
[sources.json](2026-09-26-token-economics/sources.json) or the raw files next to it),
**[D]** derived from M/S by the formula shown, **[A]** assumed.

Reproduce every table: `python3 tools/token_economics.py
docs/research/2026-09-26-token-economics/inputs.json` (standard library only). The inputs
file labels every number. Output: [tables.md](2026-09-26-token-economics/tables.md). Test:
`tests/test_token_economics.py`.

## Answer

- **At full load, selling tokens pays well at today's prices.**
  - One card earns **$0.60–1.90 per hour** [D] at the 4-bit floor to high list prices
    (T1), against **€0.16 per hour** for power at €0.35/kWh [D] (460 W wall).
  - Break-even utilization, dedicated box with a used GPU [A]: **7%** of 24/7 at the median
    price, **14%** at the 4-bit floor. With 4 h/month of ops time at €30/h [A]: **26% / 48%**
    (T2, 8 users, 10:1 input/output).
  - At 90% utilization: **€544/month** margin at the median price, **€266** at the floor.
- **Utilization, not cost, decides the outcome, and there is no evidence this machine
  could get any.**
  - **Channel.** OpenRouter admits providers through an application form. It pays by auto
    top-up or invoice and routes by price, uptime, throughput, tool-call error rate and
    benchmark accuracy [S]. All 16 providers of this model are companies. We found no route
    for a single consumer GPU: Vast.ai hosting is NVIDIA-only [S], RunPod lists no Radeon
    consumer card [S], and Darkbloom's consumer grid is Mac-only [S].
  - **Product fit.** In the 8-user mode each request gets 8,192 tokens of context. The mean
    OpenRouter request for this model was 18,500 prompt tokens [S]. Every listed provider
    offers 262k tokens or more [S]. Per-user speed in that mode is 19 tok/s [M], below every
    provider's p50 (26–159 tok/s, median 47) [S]. The 1-user mode takes the mean request, but
    its time to first token is ~15 s [D], against p50 latencies of 0.3–2.0 s [S].
  - **Demand is not the constraint.** One card at full load serves ~0.16% of this model's
    daily OpenRouter tokens [D]. The smallest listed providers each handle about one card's
    worth or less [S/D].
- **Renting the card out raw is not an option.** No RX 7900 XTX offers exist on Vast.ai [S],
  and RunPod does not list the card [S]. The closest proxy, an RTX 3090 at the Vast median
  of $0.15/h [S], would not cover German household power for this card at any utilization
  (T7: −€47 to −€49/month).
- **Verdict: neither, today.**
  - Don't buy hardware for this. On the machine you already own, the marginal economics are
    positive at any utilization if traffic existed (T4, "owned").
  - The work that would make it sellable: an aggregator or direct customer who admits a
    single-GPU, int4 endpoint; long context in parallel mode; and prefix caching. Only the
    last two are engineering.
  - Price decay is the long-run risk. Comparable ~30B dense models already sell at $0.25–0.45
    per 1M output tokens [S]. At that level the used-GPU box needs 40–86% utilization
    (1 user at 3:1: never), and never breaks even once ops time counts (T2, T4).

## 1. Market price (question 1)

### Qwen3.8-27B on OpenRouter (16 paid endpoints + 1 free)

[S] `https://openrouter.ai/api/frontend/v1/stats/endpoint?...qwen3.8-27b-20260814` (30-minute
window) and `.../stats/effective-pricing` (token share over the last 24 h), both read
2026-09-26 ~13:10 CEST; raw: [endpoint stats](2026-09-26-token-economics/openrouter-qwen3.8-27b-endpoint-stats.json),
[effective pricing](2026-09-26-token-economics/openrouter-qwen3.8-27b-effective-pricing.json),
[public endpoints API](2026-09-26-token-economics/openrouter-endpoints-qwen3.8-27b.json).
Prices are USD per 1M tokens as currently charged (DeepInfra and Phala include a promotion):

| provider | quant | input | output | cache read | p50 tok/s | p50 latency s | tokens, 24 h | share | cache hits |
|---|---|---|---|---|---|---|---|---|---|
| Darkbloom | fp4 | 0.100 | 1.800 | – | 53 | 0.81 | 42 M | 0.1% | 9% |
| DeepInfra (25% promo) | bf16 | 0.150 | 1.875 | 0.037 | 29 | 1.21 | 1,232 M | 4.3% | 32% |
| Phala (17% promo) | unknown | 0.199 | 2.075 | 0.042 | 62 | 1.00 | 620 M | 2.2% | 35% |
| Parasail | fp8 | 0.240 | 2.200 | 0.050 | 59 | 0.92 | 2,078 M | 7.3% | 62% |
| Chutes | fp8 | 0.240 | 2.200 | 0.024 | 32 | 2.01 | 807 M | 2.8% | 75% |
| AkashML | fp8 | 0.250 | 2.200 | 0.050 | 41 | 0.86 | 161 M | 0.6% | 35% |
| Mancer 2 | fp8 | 0.200 | 2.500 | – | 58 | 1.16 | 87 M | 0.3% | 0% |
| Ionstream | fp8 | 0.105 | 2.550 | 0.100 | 47 | 0.82 | 288 M | 1.0% | 48% |
| Alibaba (DashScope intl.) | unknown | 0.425 | 2.550 | 0.085 | 47 | 0.80 | 3,214 M | 11.3% | 70% |
| CoreWeave | fp8 | 0.400 | 3.000 | 0.150 | 58 | 0.30 | 204 M | 0.7% | 61% |
| Novita | unknown | 0.420 | 3.000 | 0.085 | 51 | 1.45 | 2,393 M | 8.4% | 79% |
| Venice | fp8 | 0.450 | 3.200 | – | 47 | 1.99 | 1,140 M | 4.0% | 2% |
| Cloudflare | unknown | 0.450 | 3.200 | 0.050 | 26 | 1.52 | 27 M | 0.1% | 0% |
| Reka | unknown | 0.090 | 4.400 | 0.085 | 34 | 0.71 | 4,812 M | 16.9% | 75% |
| Wafer | unknown | 0.091 | 4.400 | 0.085 | 33 | 1.22 | 8,458 M | 29.7% | 45% |
| DekaLLM | unknown | 0.096 | 4.400 | 0.087 | 159 | 0.73 | 2,905 M | 10.2% | 82% |
| ModelRun (`:free`, Modular) | fp4 | 0 | 0 | – | – | – | – | – | – |

- **Range and median [S/D].** Output $1.80–4.40, median **$2.55**. Input $0.09–0.45, median
  **$0.22**. Cache reads $0.024–0.15.
  - Token-weighted effective prices (the page's "Weighted Average"): input **$0.127**,
    output **$3.43**. Cache hits pull input down; output is pulled up by the $4.40 providers.
- **Recent moves [S].** Reka, Wafer and DekaLLM raised output from ~$2.50 to $4.40 on
  2026-09-23/24 (effective-pricing daily series). They list input at ~$0.09 and now carry
  57% of tokens.
  - The traffic is 20–30× more input than output, so the per-request cost is dominated by
    input and cache pricing. Output price alone does not predict share: Darkbloom has the
    lowest output price and 0.1% of tokens.
- **Quantization [S].** The market prices 4-bit low but not consistently:
  - the only paid 4-bit endpoint (Darkbloom fp4) is the cheapest, and the free endpoint is
    fp4;
  - but DeepInfra's BF16 endpoint sits at $1.875 (on promotion);
  - fp8 endpoints are at $2.20–3.20, and "unknown" ones span $2.075–4.40.
  - No int4 endpoint is listed. OpenRouter lets buyers filter by quantization, so a Q4_0
    endpoint loses the buyers who exclude it. The base case below uses the fp4 floor as the
    "4-bit tier" price.
- **Speed tiers [S].**
  - Groq serves this model at 450 tok/s for **$0.80 / $4.00** (console.groq.com/docs/models).
  - Fireworks prices its "Fast" variants at **1.5×** Standard (Kimi K3 $15.00 → $22.50 output,
    GLM 5.3 $4.40 → $6.60) and "Priority" at 1.25–1.5×.
  - On OpenRouter, price does not follow speed. The three $4.40 endpoints run at 33, 34 and
    159 tok/s.
- **Other providers [S].**
  - Together has no serverless Qwen3.8-27B (only fine-tuning). Its comparable dense models:
    Gemma 4 31B $0.39 / $0.97, Muse Glimmer 30B $0.35 / $1.50.
  - Fireworks does not list the model individually. Unlisted models above 16B parameters
    cost a flat **$0.90** per 1M, input and output alike. Its on-demand H100 is $8.00/h.
  - DeepInfra and Alibaba/DashScope (international) appear in the table above.
- **Same-generation and older ~30B models on OpenRouter [S]** (input / output, from
  [comparables](2026-09-26-token-economics/openrouter-models-comparables.json)):
  - Qwen3.6-27B: $0.30–0.45 / $2.00–3.25. Qwen3.5-27B: $0.195–0.30 / $1.56–2.60.
  - Qwen3-32B: $0.08–0.14 / $0.28–0.57. Gemma 4 31B: $0.09 / $0.34. Gemma 3 27B: $0.08 / $0.45.
    Mistral Small 3.2 24B: $0.094 / $0.25.
  - **Price decay** is therefore the long-run risk. The "decay" price scenario uses
    Gemma 4 31B's $0.09 / $0.34.
- **Consumer and decentralized markets [S].**
  - Darkbloom (Eigen Labs) runs a grid of 1,231 consumer Macs and is the fp4 endpoint above.
    It credits providers the full token price with no platform fee. Its calculator estimates
    **$204.89/month** for an M4 Max 128 GB at a 60% duty cycle (Qwen3.6-35B-A3B). It accepts
    Apple Silicon only.
  - AkashML, Chutes and Phala are also listed on OpenRouter. We did not check whether they
    accept third-party consumer GPUs.
  - Not checked: Salad, io.net, Nosana, inference.net.

## 2. Sellable volume (question 2)

**Market size [S]** ([activity json](2026-09-26-token-economics/openrouter-qwen3.8-27b-activity.json), 30 full days
2026-08-27..09-25):

| Measure | Range over 30 days |
| --- | --- |
| Prompt tokens per day | 17.9–61.9 B |
| Completion tokens per day | 0.70–2.86 B |
| Requests per day | 0.83–3.44 M |
| Spend per day | $4.4–14.3 k |
| Prompt/completion ratio | 10.6–29.3, median 23.7 |
| Share of prompt tokens served from cache | 52–79%, median 71% |
| Tool calls per request | median 0.33 |

On 2026-09-25 the mean request was **18,475 prompt + 976 completion tokens**, and 67% of
completion tokens were reasoning.

**One card against that [D].** At full load in the 8-user mode with a 20:1 mix, the card
produces 43.8 output + 875 input tok/s, or 79 M tokens/day. That is **0.16%** of 2026-09-25's
49.3 B tokens. The smallest listed providers handle 27–87 M tokens per 24 h: Cloudflare 27 M,
Darkbloom 42 M, Mancer 87 M [S]. That is **0.3–1.1 cards' worth**. If the machine were
admitted and routable at all, a sliver of this one model's traffic would fill it.

**What limits routability** (OpenRouter's rules [S]; our numbers [M/D]):

| Router signal | Market | zerv, 8 users | zerv, 1 user MTP |
| --- | --- | --- | --- |
| Context per request | 262k (14 endpoints), 1M (Alibaba, Novita) | **8,192** [M] | 44,896 [M] (60k tuned) |
| Output tok/s per request | p50 26–159, median 47 | **19** [M] | 77–127 [M] |
| TTFT, 3,000-token prompt | latency p50 0.30–2.0 s | 2.5 s [D] | 2.5 s [D] |
| TTFT, mean request (18.5k prompt) | same | does not fit | **15.4 s** [D] |
| Router-measured throughput, 3,000 + 500 tokens | as above | 18 tok/s [D] | 67 tok/s [D] |
| Cache pricing for repeated prompts | 13 of 16 endpoints | none (no prefix cache in `--parallel`) | prefix cache per slot |
| Quantization disclosed | fp4 / fp8 / bf16 | int4 | int4 |

- **max_tokens.** OpenRouter only routes a request to providers that support its
  `max_tokens` [S]. Reasoning clients often ask for large outputs, and 8,192 tokens total
  excludes them.
- **Throughput metric.** It includes TTFT and queueing [S].
- **Auto Exacto** (on every tool-calling request, [S]) deprioritizes:
  - endpoints more than 1.5σ below the median throughput;
  - endpoints scoring below the median − 2σ on OpenRouter's own benchmarks, or with no
    benchmark data.
  - The 8-user mode's 19 tok/s is likely below the throughput cutoff: not computed, since σ
    was not available.
  - How Q4_0 scores on those benchmarks is unknown. Quality against the BF16/FP8 endpoints
    is not measured here.
- **Uptime.** At least 95% for normal routing, 80–94% "degraded", below 80% fallback only
  [S]. One box on a residential line has no redundancy.
- **Determinism** (outputs identical under greedy decoding whatever the load) is not
  advertised by any provider and has no visible price. It is valued at $0.

**Utilization scenarios [A].** We found no public utilization figures for small providers.
Darkbloom's calculator assumes 60% duty for its Macs [S], and Vast asks hosts to expect "close
to max capacity" during rentals [S].
- **Our judgement:**
  - 0% until an aggregator admits the endpoint;
  - once admitted, 10–30% for the 8-user mode (8k context, slow per user), and more for a
    long-context, prefix-cached parallel mode.
- The tables use 10 / 30 / 60 / 90% and give break-even points so any estimate can be
  checked against them.

**Input/output mix.** The tables use 3:1 and 10:1 [A] (the handoff's chat range) and 20:1
[S] (OpenRouter's observed median ~23:1). In zerv's parallel mode every prompt token is
prefilled, because it has no prefix cache. Competitors served 71% of prompt tokens from cache
[S]. A zerv endpoint without cache pricing is more expensive per repeated request.

## 3. Revenue model (question 3)

- **Formula.** Revenue/month = u × 730 h × 3,600 × (R_in × price_in + R_out × price_out) / 10⁶.
- **The GPU is time-shared** between prefill at P and decode at D [D].
  - Full-load rates: R_out = 1 / (r/P + 1/D) and R_in = r × R_out, where r is uncached input
    tokens per output token.
  - P = 1,200 tok/s [D], from measured TTFT: 2.49 s at 3,223 tokens (1,294 tok/s) and
    10.14 s at 12,034 tokens (1,187 tok/s) ([report](../bench/2026-09-24-gemm-f16x-isa.md)).
    The multi-user interference scenario implies ~1,160 tok/s.
  - D is derived from the measured aggregate by removing the measurement's own short-prompt
    prefill (137-token prompts, 256 output tokens):
    [8 users](../bench/2026-09-25-multiuser.md): 150.9 → 161.8 tok/s; 4 users: 138.7 → 147.8.
  - The test checks that the model reproduces each measured aggregate.

**T1. Full-load capacity and revenue, USD per hour** (price scenarios: decay $0.09/$0.34,
floor4bit $0.10/$1.80, median $0.22/$2.55, high $0.45/$3.20):

| mode | in/out | out tok/s | in tok/s | per-user tok/s | context/request | $/h @decay | $/h @floor4bit | $/h @median | $/h @high |
|---|---|---|---|---|---|---|---|---|---|
| M8 | 3 | 115.2 | 346 | 19.3 | 8,192 | 0.25 | 0.87 | 1.33 | 1.89 |
| M8 | 10 | 68.9 | 689 | 19.3 | 8,192 | 0.31 | 0.69 | 1.18 | 1.91 |
| M8 | 20 | 43.8 | 875 | 19.3 | 8,192 | 0.34 | 0.60 | 1.10 | 1.92 |
| M4 | 3 | 107.9 | 324 | 35.6 | 8,192 | 0.24 | 0.82 | 1.25 | 1.77 |
| M4 | 10 | 66.2 | 662 | 35.6 | 8,192 | 0.30 | 0.67 | 1.13 | 1.84 |
| M4 | 20 | 42.7 | 854 | 35.6 | 8,192 | 0.33 | 0.58 | 1.07 | 1.87 |
| M1 | 3 | 80.0 | 240 | 100 | 44,896 | 0.18 | 0.60 | 0.92 | 1.31 |
| M1 | 10 | 54.5 | 545 | 100 | 44,896 | 0.24 | 0.55 | 0.93 | 1.51 |
| M1 | 20 | 37.5 | 750 | 100 | 44,896 | 0.29 | 0.51 | 0.94 | 1.65 |
| M8+ | 3 | 144.3 | 433 | 25.6 | 8,192 | 0.32 | 1.09 | 1.67 | 2.36 |
| M8+ | 10 | 78.3 | 783 | 25.6 | 8,192 | 0.35 | 0.79 | 1.34 | 2.17 |
| M8+ | 20 | 47.4 | 948 | 25.6 | 8,192 | 0.37 | 0.65 | 1.19 | 2.08 |

Modes:
- **M8**, 8 users [M].
- **M4**, 4 users [M]: measured with 8 slots, so its context stays at 8,192. A `--parallel 4`
  server with larger slots was not measured.
- **M1**, 1 user with MTP [M range 77–127, A point 100]. zerv refuses to combine speculation
  with `--parallel`.
- **M8+**, the roadmap's 190–220 tok/s at 8 users [A, not measured], an upside only.

**Many users against one fast user.**
- At chat mixes (r ≥ 3), prefill takes most of the GPU time, so the modes converge. At 10:1
  and the median price, M8 earns $1.18/h and M1 $0.93/h (−21%).
- M1 would need a 27% higher price to match. Its 77–127 tok/s would rank second on this
  model's OpenRouter list (after DekaLLM's 159). But the price data shows no speed premium
  there, and the fast tiers that do charge one run far faster:
  - Groq: 450 tok/s at $4.00;
  - Fireworks: Fast = 1.5× Standard.
- M1 is still the only mode that can take the market's mean request.

## 4. Cost base (question 4)

### Electricity

| Item | Value | Label |
| --- | --- | --- |
| GPU board power under decode | 339 W, the power cap; sustained GEMMs 310–339 W | [M] [decode overheads](../bench/2026-09-22-decode-overheads.md), [hardware](../hardware.md); not re-measured in `--parallel 8` |
| GPU idle | 34 W (`power1_average`, 10 samples, desktop session on the card) | [M] 2026-09-26 13:04 |
| Host (3900X, X570, 64 GB, NVMe) | 75 W load / 45 W idle, DC side | [A] |
| PSU efficiency | 90% | [A] |
| **Wall** | **460 W load, 88 W idle** | [D] |

**Electricity prices:**

| Case | Price per kWh | Label and source |
| --- | --- | --- |
| DE-home (base) | €0.35 | [A], handoff's range. Eurostat `nrg_pc_204` 2025-S2, DE households at 2.5–5 MWh, all taxes: €0.3869 [S] (includes fixed charges) |
| DE-high | €0.45 | [S], Eurostat 2025-S2, DE households at 1–2.5 MWh: €0.4383 |
| DE-biz | €0.26 | [S], Eurostat `nrg_pc_205` 2025-S2, DE non-household at 20–499 MWh, excluding VAT: €0.2618 |
| EU27 references | households €0.290; non-household (20–499 MWh, excl. VAT) €0.218 | [S], Eurostat 2025-S2 |
| US-ind | 9.77 ¢ (€0.086) | [S], EIA Table 5.6.A, July 2026 |
| US residential and commercial | 18.31 ¢ and 14.53 ¢ | [S], EIA Table 5.6.A, July 2026 |

Raw Eurostat files: [nrg_pc_204](2026-09-26-token-economics/eurostat-nrg_pc_204.json),
[nrg_pc_205](2026-09-26-token-economics/eurostat-nrg_pc_205.json). The machine runs at 460 W,
or 4.0 MWh/year at 24/7 full load.

**Measure it (recommended before any decision).**
- **GPU.** Sample `cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_average` (µW) once per
  second while `bench/run_multiuser.py` runs the 8-user steady scenario, and at idle.
  `tools/thermal_log.py` already records it along with clocks and throttle reasons.
- **CPU package.** RAPL `/sys/class/powercap/intel-rapl:0/energy_uj` needs root here.
- **Whole machine.** A wall meter on the machine's socket gives the number that is billed.
  Read it at idle and after 10 minutes of steady load. The difference between wall power
  and the GPU reading is the host plus PSU loss.

### Hardware, other costs, fees, tax

- **GPU price [S].**
  - New: one offer left on geizhals.de, Sapphire Nitro+ Vapor-X at **€1,799.99**. Every other
    7900 XTX listing shows "keine Angebote"; the card is out of production.
  - Used: 19 private Kleinanzeigen asking prices from 2026-09-14 to 26: **€600–1,200, median
    €900**. Asking prices are not transaction prices.
  - eBay sold listings need a login and were not read.
- **Cost bases [A]** (life 36 months, 24–48 plausible):

  | Basis | Hardware | Resale | Per month | Idle power |
  | --- | --- | --- | --- | --- |
  | **owned** | sunk | – | €0 | not counted (machine is on anyway) |
  | **used** (base) | €800 GPU + €500 host | €350 | €26.4 | counted |
  | **new** | €1,800 GPU + €500 host | €350 | €54.2 | counted |

- **Internet [A].** The existing line costs nothing extra. Bandwidth is negligible: ~1 kB
  of SSE per output token, and prompts of ~75 kB. A business line with a static IP and an
  SLA would be ~€30–50/month [A]. A residential line with no redundancy threatens the 95%
  uptime rule.
- **Ops time [A].** 4 h/month at €30/h = €120/month, shown separately as "incl. ops time".
- **Platform fees.**
  - OpenRouter passes provider prices through with no markup. Buyers pay 5.5% on credit
    purchases [S]. Provider-side commercial terms are not public [A: 0%].
  - Direct sales by card through Stripe cost 1.5% + €0.25 per EEA standard card, or
    3.15% + €0.25 (+2% FX) for foreign cards [S].
  - Darkbloom charges providers no fee [S], but accepts Macs only.
- **Tax and business (Germany, flagged, not computed).**
  - Selling inference is a trade: Gewerbeanmeldung, and income tax on the profit.
  - Trade tax has a €24,500 allowance for natural persons (§11 GewStG) [S].
  - VAT is 19% (§12 UStG) [S]. The small-business exemption applies below €25,000 revenue in
    the previous year and €100,000 in the current one (§19 UStG) [S].
  - B2B services to a non-EU business (e.g. OpenRouter, US) are generally not taxable in
    Germany (place of supply at the recipient). Confirm with a tax adviser.
  - Household electricity tariffs may not permit commercial use.
  - None of this changes the break-even points much at these revenues.
- **Opportunity cost: renting the GPU raw [S].**
  - **Vast.ai:** no RX 7900 XTX offers (`vast.ai/pricing/gpu/RX-7900-XTX`: "No current
    offers"; search API: 0). Its host setup installs NVIDIA drivers.
  - **RunPod** lists no Radeon consumer card.
  - **Nearest proxy:** the RTX 3090 (24 GB, 936 GB/s). Vast: 64 on-demand offers, median
    **$0.151/h**, $0.121–0.19 interquartile. RunPod: **$0.50/h** list (RunPod's own price, not
    a host payout).
  - At the Vast median, rental income stays below this card's German household power bill
    at every utilization (T7). Rental prices from [vast-offers.json](2026-09-26-token-economics/vast-offers.json);
    the API returns at most 64 offers per query.

## 5. Comparison with rented datacenter GPUs (question 5)

**Rental prices [S].** H100 SXM: Vast.ai min $1.97/h, p25 $2.27, median $3.14, p75 $3.21
(16 offers), RunPod $3.49/h, Fireworks on-demand $8.00/h. A100 SXM4: Vast median $0.80/h, RunPod $1.59/h.
L40S: Vast median $0.80/h, RunPod $1.09/h.

**Throughput [S].** GPUStack's Qwen3-32B benchmark on one H100 SXM
([docs.gpustack.ai](https://docs.gpustack.ai/2.0/performance-lab/qwen3-32b/h100/)).
The model is not the same: it is 32B dense with full attention, where Qwen3.8-27B is a hybrid.
The engines are vLLM v0.11.0 (BF16) and TensorRT-LLM (FP8), and every request is sent at once.

**T6. USD per 1M output tokens, all cost charged to output:**

| H100 workload (Qwen3-32B) | out tok/s | H100 SXM, Vast p25 | H100 SXM, Vast median | H100 SXM, RunPod | revenue $/h @median |
|---|---|---|---|---|---|
| ShareGPT, vLLM BF16 | 1132 | 0.56 | 0.77 | 0.86 | 11.35 |
| ShareGPT, TRT-LLM FP8 | 2060 | 0.31 | 0.42 | 0.47 | 20.68 |
| 4000 in / 200 out, vLLM BF16 | 102 | 6.17 | 8.53 | 9.48 | 2.56 |
| 4000 in / 200 out, TRT-LLM FP8 | 235 | 2.69 | 3.72 | 4.13 | 5.87 |

**Comparisons:**
- **Datacenter floor.** For chat-shaped traffic (ShareGPT is roughly 1:1), a rented H100
  produces output at **$0.31–0.86 per 1M**. That is far below the market's $1.80–4.40, so
  current Qwen3.8-27B prices carry a large margin over datacenter cost.
- **Input-heavy traffic.** At 20:1 without caching, prefill dominates. An FP8 H100 then
  earns $5.87/h at the median price against $2.27–3.49/h rent.
- **This card** ([T5](2026-09-26-token-economics/tables.md)), at €0.35/kWh with the used-GPU
  basis:
  - **€0.48–1.25 per 1M output tokens** at full load (8 users, 3:1 to 20:1);
  - €0.18–0.48 at US industrial power;
  - €0.68–1.46 in the 1-user mode.
  - At 20:1 ($1.43) that is 4–7× cheaper per output token than the vLLM BF16 H100 on the
    same shape, and ~2–3× cheaper than the FP8 H100. The bases differ: the H100 figures are
    rental prices including the host's margin, ours is depreciation plus power. The card's
    advantage is low capital and power per card, not throughput: an FP8 H100 moves ~5×
    more tokens per hour at 20:1.
- **Consumer-GPU floor** (HyperQwen on an RTX 3090 at Vast's median rent [S]):
  - 942 output tok/s at 64 concurrent (128 in / 512 out) → **$0.045 per 1M output**;
  - 298 tok/s at 8 concurrent (1,024-token answers) → $0.14 per 1M.
  - At RunPod's list price: $0.15 and $0.47.
- **Conclusion.** Short-prompt batch traffic on consumer GPUs is essentially free to produce.
  Market prices are set by demand for this new model and by provider margins, not by cost.

## 6. Results

**T2. Monthly revenue / cost / margin, EUR** (at €0.35/kWh, with the used-GPU basis):

*8 users, 10:1 input/output:*

| price | u=10% | u=30% | u=60% | u=90% | break-even u | incl. ops time |
|---|---|---|---|---|---|---|
| decay | 20 / 58 / **-39** | 59 / 77 / **-18** | 118 / 106 / **12** | 177 / 134 / **43** | 48% | never |
| floor4bit | 44 / 58 / **-14** | 133 / 77 / **56** | 267 / 106 / **161** | 400 / 134 / **266** | 14% | 48% |
| median | 75 / 58 / **17** | 226 / 77 / **149** | 453 / 106 / **347** | 679 / 134 / **544** | 7% | 26% |
| high | 122 / 58 / **64** | 367 / 77 / **289** | 734 / 106 / **628** | 1100 / 134 / **966** | 4% | 15% |

*1 user with MTP, 10:1 input/output:*

| price | u=10% | u=30% | u=60% | u=90% | break-even u | incl. ops time |
|---|---|---|---|---|---|---|
| decay | 16 / 58 / **-43** | 47 / 77 / **-31** | 94 / 106 / **-12** | 140 / 134 / **6** | 80% | never |
| floor4bit | 35 / 58 / **-23** | 106 / 77 / **28** | 211 / 106 / **105** | 317 / 134 / **182** | 19% | 66% |
| median | 60 / 58 / **1** | 179 / 77 / **102** | 358 / 106 / **252** | 537 / 134 / **403** | 10% | 34% |
| high | 97 / 58 / **38** | 290 / 77 / **213** | 581 / 106 / **475** | 871 / 134 / **737** | 6% | 19% |

**T3. Margin against electricity price** (8 users, 10:1, used-GPU basis; EUR/month):

| price | electricity | u=10% | u=30% | u=60% | u=90% | break-even u |
|---|---|---|---|---|---|---|
| decay | US-ind €0.086 | -15 | 20 | 72 | 124 | 18% |
| decay | DE-biz €0.26 | -30 | -5 | 33 | 71 | 34% |
| decay | DE-home €0.35 | -39 | -18 | 12 | 43 | 48% |
| decay | DE-high €0.45 | -48 | -33 | -10 | 12 | 74% |
| floor4bit | US-ind | 10 | 95 | 221 | 347 | 8% |
| floor4bit | DE-biz | -6 | 69 | 181 | 294 | 12% |
| floor4bit | DE-home | -14 | 56 | 161 | 266 | 14% |
| floor4bit | DE-high | -23 | 41 | 138 | 235 | 17% |
| median | US-ind | 41 | 187 | 407 | 626 | 4% |
| median | DE-biz | 25 | 162 | 367 | 572 | 6% |
| median | DE-home | 17 | 149 | 347 | 544 | 7% |
| median | DE-high | 8 | 134 | 324 | 514 | 9% |
| high | US-ind | 88 | 328 | 688 | 1048 | 3% |
| high | DE-biz | 72 | 303 | 648 | 994 | 4% |
| high | DE-home | 64 | 289 | 628 | 966 | 4% |
| high | DE-high | 55 | 275 | 605 | 935 | 5% |

**T4. Break-even utilization** (€0.35/kWh; owned / used / new):
- "any" means every utilization is profitable, because the machine's idle and hardware costs
  are sunk.
- "never" means even 100% loses.

| mode | in/out | decay | floor4bit | median | high |
|---|---|---|---|---|---|
| M8 | 3 | any / 73% / never | any / 11% / 17% | any / 6% / 10% | any / 4% / 7% |
| M8 | 10 | any / 48% / 75% | any / 14% / 22% | any / 7% / 12% | any / 4% / 7% |
| M8 | 20 | any / 40% / 63% | any / 17% / 27% | any / 8% / 13% | any / 4% / 7% |
| M1 | 3 | never / never / never | any / 17% / 26% | any / 10% / 15% | any / 7% / 10% |
| M1 | 10 | any / 80% / never | any / 19% / 30% | any / 10% / 15% | any / 6% / 9% |
| M1 | 20 | any / 54% / 85% | any / 21% / 33% | any / 10% / 15% | any / 5% / 8% |

M4 and M8+ rows, the cost per 1M output tokens (T5), T7 and T8 are in
[tables.md](2026-09-26-token-economics/tables.md).

**Effective cost per 1M output tokens at full load** (T5, used-GPU basis):

| Mode | 3:1 | 10:1 | 20:1 |
| --- | --- | --- | --- |
| M8, €0.35/kWh | €0.48 | €0.79 | €1.25 |
| M8, US industrial | €0.18 | €0.30 | €0.48 |

Revenue per 1M output tokens at the floor price is $1.80 plus the input revenue.

## 7. Competitors on this card class (llama-server, vLLM, HyperQwen)

These are the tracked competitors ("Tracked set", user decision 2026-09-26, in
[performance.md](../performance.md)). The fourth member of that set,
llama.cpp-RDNA3-7900xtx-opt, publishes no single-card numbers and has not been run here.

HyperQwen targets NVIDIA and cannot run here, so its numbers are its own README's, on an
RTX 3090 at 250 W, with its own requantization and workloads [S]. Those rows are not an equal
comparison.

| Engine | Card | 1 user | 8 users | 64 users |
| --- | --- | --- | --- | --- |
| zerv (Q4_0) | RX 7900 XTX | 47.9 plain, 77–127 MTP [M] | 150.8–150.9 total, 19.3 per user [M] | not supported (8 slots) |
| vLLM v0.30.0 (INT4 W4A16) | RX 7900 XTX | 29.6–31.1 median, best round 35.9 [M] | 151.9–158.2 [M] | not measured |
| llama-server b10964 (Q4_0) | RX 7900 XTX | 37.9–38.4 [M] | 125.7–131.0 [M] | not measured |
| HyperQwen @`1cf86656` | RTX 3090 | 46 plain, 127 MTP (single-user mode) | 298 e2e (cohort C8, 1,024-token answers); 111–144 decode at 8k context per request | 942 e2e at 128 in / 512 out |

- zerv, vLLM and llama-server were measured in the [multi-user report](../bench/2026-09-25-multiuser.md)
  and the [packed-prefill run](../bench/2026-09-26-packed-prefill.md): steady closed loop,
  256-token answers, short prompts.
- **What HyperQwen changes in the handoff's framing.** The handoff called this machine's
  result typical of consumer-GPU inference. HyperQwen reports roughly **2× zerv's 8-user
  throughput at 8 users and ~6× at 64 users** on a cheaper 24 GB card, on shorter-context
  work.
- **Why that matters here.** Aggregate throughput sets the revenue per GPU-hour (T1). The
  8-slot, 8k-context limit sets which requests the machine can take at all.
- **Missing comparison.** HyperQwen's C8/C64 cohort protocol has not been run on zerv. Until
  it is, the gap is a reported number, not a measured one.

## 8. Open questions and next measurements

1. **Electricity price and country** of the operator: the tables assume Germany at
   €0.35/kWh. Confirm or replace `base_electricity` in `inputs.json`.
2. **Wall power** at idle and under the 8-user load, measured as in section 4. Replace
   `power` in `inputs.json`.
3. **Admission.** Would OpenRouter, or another aggregator, list a single-GPU int4 endpoint
   with 8k or 45k context? Answering needs contact, which is the operator's decision.
4. **Request-length distribution** of target traffic: what share fits in 8,192 tokens,
   and what `max_tokens` clients send.
5. **Quality** of Q4_0 against the BF16/FP8 endpoints on the benchmarks OpenRouter uses
   (GPQA Diamond, τ-bench), which gate Auto Exacto routing.
6. **HyperQwen's cohort protocol** (C1–C8, C64) on zerv, for a like-for-like aggregate
   comparison.
7. **Engineering levers** that change the numbers most: long-context parallel slots, a
   prefix cache in `--parallel` (71% of market prompt tokens are cache hits), and higher
   concurrency (T1: revenue per GPU-hour scales with aggregate throughput).

Follow-up: [renting datacenter GPUs with a faster engine](2026-09-26-rented-gpu-economics.md).

## Limitations

- Prices are one day's snapshot, and they moved within the week (the $4.40 changes).
- "Share" comes from OpenRouter's page and covers only OpenRouter.
- The Vast.ai search API returns at most 64 offers per query, so medians are over a sample.
- The time-sharing model ignores queueing and scheduler overheads. The interference
  measurement suggests prefill runs ~3–10% below the isolated rate.
- M1's point value of 100 tok/s is a content mix [A]. Agentic and code traffic sits at the
  upper end.
- The H100 throughputs come from a different model, older engine versions and a
  saturating benchmark.
- No inference, power or throughput measurement was run for this report.
