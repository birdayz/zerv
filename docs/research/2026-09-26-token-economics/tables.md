Wall power (derived): 460 W under load, 88 W idle. Base: DE-home 0.35 EUR/kWh, cost basis 'used' (26.4 EUR/month fixed). 1 EUR = 1.1403 USD.

### T1. Full-load capacity and revenue per hour (USD)

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

### T2. 8 users (--parallel 8), input/output 10: revenue / cost / margin, EUR per month (DE-home, 'used')

| price | u=10% | u=30% | u=60% | u=90% | break-even u | incl. ops time |
|---|---|---|---|---|---|---|
| decay | 20 / 58 / **-39** | 59 / 77 / **-18** | 118 / 106 / **12** | 177 / 134 / **43** | 48% | never |
| floor4bit | 44 / 58 / **-14** | 133 / 77 / **56** | 267 / 106 / **161** | 400 / 134 / **266** | 14% | 48% |
| median | 75 / 58 / **17** | 226 / 77 / **149** | 453 / 106 / **347** | 679 / 134 / **544** | 7% | 26% |
| high | 122 / 58 / **64** | 367 / 77 / **289** | 734 / 106 / **628** | 1100 / 134 / **966** | 4% | 15% |

### T2. 1 user, MTP speculation, input/output 10: revenue / cost / margin, EUR per month (DE-home, 'used')

| price | u=10% | u=30% | u=60% | u=90% | break-even u | incl. ops time |
|---|---|---|---|---|---|---|
| decay | 16 / 58 / **-43** | 47 / 77 / **-31** | 94 / 106 / **-12** | 140 / 134 / **6** | 80% | never |
| floor4bit | 35 / 58 / **-23** | 106 / 77 / **28** | 211 / 106 / **105** | 317 / 134 / **182** | 19% | 66% |
| median | 60 / 58 / **1** | 179 / 77 / **102** | 358 / 106 / **252** | 537 / 134 / **403** | 10% | 34% |
| high | 97 / 58 / **38** | 290 / 77 / **213** | 581 / 106 / **475** | 871 / 134 / **737** | 6% | 19% |

### T3. Margin sensitivity to electricity, EUR per month (M8, in/out 10, 'used')

| price | electricity | u=10% | u=30% | u=60% | u=90% | break-even u |
|---|---|---|---|---|---|---|
| decay | US-ind | -15 | 20 | 72 | 124 | 18% |
| decay | DE-biz | -30 | -5 | 33 | 71 | 34% |
| decay | DE-home | -39 | -18 | 12 | 43 | 48% |
| decay | DE-high | -48 | -33 | -10 | 12 | 74% |
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

### T4. Break-even utilization (DE-home), per cost basis

| mode | in/out | decay (owned/used/new) | floor4bit (owned/used/new) | median (owned/used/new) | high (owned/used/new) |
|---|---|---|---|---|---|
| M8 | 3 | any / 73% / never | any / 11% / 17% | any / 6% / 10% | any / 4% / 7% |
| M8 | 10 | any / 48% / 75% | any / 14% / 22% | any / 7% / 12% | any / 4% / 7% |
| M8 | 20 | any / 40% / 63% | any / 17% / 27% | any / 8% / 13% | any / 4% / 7% |
| M4 | 3 | any / 86% / never | any / 11% / 18% | any / 7% / 11% | any / 5% / 7% |
| M4 | 10 | any / 52% / 81% | any / 15% / 23% | any / 8% / 12% | any / 5% / 7% |
| M4 | 20 | any / 42% / 66% | any / 18% / 27% | any / 8% / 13% | any / 4% / 7% |
| M1 | 3 | never / never / never | any / 17% / 26% | any / 10% / 15% | any / 7% / 10% |
| M1 | 10 | any / 80% / never | any / 19% / 30% | any / 10% / 15% | any / 6% / 9% |
| M1 | 20 | any / 54% / 85% | any / 21% / 33% | any / 10% / 15% | any / 5% / 8% |
| M8+ | 3 | any / 45% / 71% | any / 8% / 13% | any / 5% / 8% | any / 3% / 5% |
| M8+ | 10 | any / 38% / 59% | any / 12% / 19% | any / 6% / 10% | any / 4% / 6% |
| M8+ | 20 | any / 35% / 55% | any / 15% / 24% | any / 7% / 12% | any / 4% / 6% |

### T5. Cost per 1M output tokens at full load, EUR ('used', all cost on output)

| mode | in/out | US-ind | DE-biz | DE-home | DE-high |
|---|---|---|---|---|---|
| M8 | 3 | 0.18 | 0.38 | 0.48 | 0.59 |
| M8 | 10 | 0.30 | 0.63 | 0.79 | 0.98 |
| M8 | 20 | 0.48 | 0.99 | 1.25 | 1.54 |
| M4 | 3 | 0.19 | 0.40 | 0.51 | 0.63 |
| M4 | 10 | 0.32 | 0.65 | 0.83 | 1.02 |
| M4 | 20 | 0.49 | 1.01 | 1.28 | 1.58 |
| M1 | 3 | 0.26 | 0.54 | 0.68 | 0.84 |
| M1 | 10 | 0.38 | 0.79 | 1.00 | 1.24 |
| M1 | 20 | 0.56 | 1.15 | 1.46 | 1.80 |
| M8+ | 3 | 0.15 | 0.30 | 0.38 | 0.47 |
| M8+ | 10 | 0.27 | 0.55 | 0.70 | 0.86 |
| M8+ | 20 | 0.44 | 0.91 | 1.16 | 1.43 |

### T6. Rented datacenter GPU: USD per 1M output tokens (all cost on output) and revenue per hour at the median price

| H100 workload (Qwen3-32B) | out tok/s | H100 SXM, Vast p25 | H100 SXM, Vast median | H100 SXM, RunPod | revenue $/h @median |
|---|---|---|---|---|---|
| ShareGPT, vLLM BF16 | 1132 | 0.56 | 0.77 | 0.86 | 11.35 |
| ShareGPT, TRT-LLM FP8 | 2060 | 0.31 | 0.42 | 0.47 | 20.68 |
| 4000 in / 200 out, vLLM BF16 | 102 | 6.17 | 8.53 | 9.48 | 2.56 |
| 4000 in / 200 out, TRT-LLM FP8 | 235 | 2.69 | 3.72 | 4.13 | 5.87 |

### T7. Renting the card out instead (proxy: RTX 3090 on Vast.ai; no RX 7900 XTX market found)

| u | gross EUR/month | power EUR/month (DE-home) | margin EUR/month ('used') |
|---|---|---|---|
| 10% | 10 | 32 | -49 |
| 30% | 29 | 51 | -48 |
| 60% | 58 | 79 | -48 |
| 90% | 87 | 108 | -47 |

- 3090 + HyperQwen, hyperqwen_c64_output_tok_s: 942 out tok/s -> 0.045 USD/1M output at Vast median, 0.147 at RunPod list.

- 3090 + HyperQwen, hyperqwen_c8_output_tok_s: 298.4 out tok/s -> 0.141 USD/1M output at Vast median, 0.465 at RunPod list.

### T8. What a router would measure per request (unloaded server; estimate)

| prompt + output tokens | mode | TTFT s | output tok / total s |
|---|---|---|---|
| 3,000 + 500 | M8 | 2.5 | 18 |
| 3,000 + 500 | M4 | 2.5 | 30 |
| 3,000 + 500 | M1 | 2.5 | 67 |
| 3,000 + 500 | M8+ | 2.5 | 23 |
| 18,500 + 900 | M8 | does not fit | - |
| 18,500 + 900 | M4 | does not fit | - |
| 18,500 + 900 | M1 | 15.4 | 37 |
| 18,500 + 900 | M8+ | does not fit | - |

Ops time, not included above: 120 EUR/month.
