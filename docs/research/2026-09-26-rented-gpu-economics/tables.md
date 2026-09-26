
### A30. Today's best engines against the roofline, >= 30 tok/s per user

| GPU | Llama-70B measured tok/s/GPU (engine, TP) | Llama-70B roofline at that TP | fitted efficiency | Qwen3.8-27B roofline req/h/GPU | Qwen3.8-27B today's engine req/h/GPU |
|---|---|---|---|---|---|
| H100 SXM | 1,607 (vllm, TP4) | 7,386 | 0.35 | 9,149 | 2,982 |
| H200 SXM | 3,019 (trt, TP4) | 8,922 | 0.41 | 11,254 | 4,529 |
| B200 | 6,062 (trt, TP2) | 17,634 | 0.43 | 21,803 | 9,293 |
| MI300X | 1,598 (vllm, TP4) | 11,208 | 0.23 | 13,558 | 2,943 |
| MI325X | 1,574 (vllm, TP8) | 12,459 | 0.17 | 14,523 | 2,209 |
| MI355X | 4,665 (vllm, TP1) | 15,792 | 0.49 | 22,812 | 11,017 |

### B30. Cost per request at 100% busy, milli-USD (per 1M output tokens: x1.025 USD)

| GPU | today's engine, owned | today's, rented (InferenceX rate) | today's, rented (public) | ours eta=0.5, rented (public) | ours eta=0.7, rented (public) | ours eta=0.85, rented (public) | roofline, rented (public) |
|---|---|---|---|---|---|---|---|
| H100 SXM | 0.39 | 0.67 | 0.77 | 0.52 | 0.36 | 0.30 | 0.25 |
| H200 SXM | 0.27 | 0.64 | 0.75 | 0.62 | 0.44 | 0.36 | 0.30 |
| B200 | 0.19 | 0.40 | 0.38 | 0.33 | 0.23 | 0.19 | 0.16 |
| MI300X | 0.32 | 0.44 | 0.63 | 0.28 | 0.20 | 0.16 | 0.14 |
| MI325X | 0.50 | 0.72 | 0.91 | 0.28 | 0.20 | 0.16 | 0.14 |
| MI355X | 0.14 | 0.26 | 0.24 | 0.23 | 0.16 | 0.13 | 0.11 |

Lowest-cost producer with today's engines: owned **MI355X** at 0.136 m$ per request. To match it, a renter needs:

| GPU (rented) | efficiency needed, public rent (>1 = impossible) | efficiency needed, InferenceX rent | speedup over today's engine, public rent |
|---|---|---|---|
| H100 SXM | 1.85 | 1.61 | 5.7x |
| H200 SXM | 2.22 | 1.89 | 5.5x |
| B200 | 1.18 | 1.25 | 2.8x |
| MI300X | 1.00 | 0.70 | 4.6x |
| MI325X | 1.01 | 0.81 | 6.6x |
| MI355X | 0.83 | 0.93 | 1.7x |

### A50. Today's best engines against the roofline, >= 50 tok/s per user

| GPU | Llama-70B measured tok/s/GPU (engine, TP) | Llama-70B roofline at that TP | fitted efficiency | Qwen3.8-27B roofline req/h/GPU | Qwen3.8-27B today's engine req/h/GPU |
|---|---|---|---|---|---|
| H100 SXM | 1,071 (vllm, TP8) | 7,601 | 0.26 | 8,959 | 1,945 |
| H200 SXM | 2,254 (trt, TP4) | 8,152 | 0.41 | 11,100 | 4,328 |
| B200 | 4,529 (trt, TP4) | 18,115 | 0.33 | 21,624 | 6,925 |
| MI300X | 1,041 (vllm, TP8) | 11,414 | 0.17 | 13,379 | 1,914 |
| MI325X | 1,154 (vllm, TP8) | 12,074 | 0.17 | 14,353 | 2,026 |
| MI355X | 2,280 (vllm, TP4) | 19,293 | 0.22 | 22,618 | 4,561 |

### B50. Cost per request at 100% busy, milli-USD (per 1M output tokens: x1.025 USD)

| GPU | today's engine, owned | today's, rented (InferenceX rate) | today's, rented (public) | ours eta=0.5, rented (public) | ours eta=0.7, rented (public) | ours eta=0.85, rented (public) | roofline, rented (public) |
|---|---|---|---|---|---|---|---|
| H100 SXM | 0.60 | 1.03 | 1.18 | 0.54 | 0.38 | 0.30 | 0.26 |
| H200 SXM | 0.28 | 0.67 | 0.79 | 0.64 | 0.44 | 0.36 | 0.31 |
| B200 | 0.25 | 0.53 | 0.51 | 0.33 | 0.23 | 0.19 | 0.16 |
| MI300X | 0.50 | 0.68 | 0.97 | 0.29 | 0.20 | 0.16 | 0.14 |
| MI325X | 0.54 | 0.79 | 0.99 | 0.29 | 0.20 | 0.16 | 0.14 |
| MI355X | 0.33 | 0.64 | 0.57 | 0.23 | 0.17 | 0.14 | 0.11 |

Lowest-cost producer with today's engines: owned **B200** at 0.250 m$ per request. To match it, a renter needs:

| GPU (rented) | efficiency needed, public rent (>1 = impossible) | efficiency needed, InferenceX rent | speedup over today's engine, public rent |
|---|---|---|---|
| H100 SXM | 1.03 | 0.89 | 4.7x |
| H200 SXM | 1.23 | 1.05 | 3.1x |
| B200 | 0.65 | 0.68 | 2.0x |
| MI300X | 0.55 | 0.39 | 3.9x |
| MI325X | 0.56 | 0.45 | 4.0x |
| MI355X | 0.46 | 0.51 | 2.3x |

### C. Rented GPU at public prices, >= 50 tok/s per user: break-even utilization and monthly margin at 60% (USD, 730 h)

| GPU | engine | break-even u, paid-today | break-even u, median-list | break-even u, commodity | margin u=60%, paid-today | margin u=60%, median-list | margin u=60%, commodity |
|---|---|---|---|---|---|---|---|
| H100 SXM | today | 22% | 25% | never | 2,835 | 2,421 | -978 |
| H100 SXM | ours 0.5 | 10% | 11% | 66% | 8,187 | 7,281 | -147 |
| H100 SXM | ours 0.7 | 7% | 8% | 46% | 12,543 | 11,236 | 530 |
| H100 SXM | ours 0.85 | 6% | 6% | 37% | 15,834 | 14,225 | 1,041 |
| H200 SXM | today | 15% | 16% | 95% | 7,565 | 6,642 | -922 |
| H200 SXM | ours 0.5 | 12% | 13% | 77% | 9,943 | 8,802 | -552 |
| H200 SXM | ours 0.7 | 8% | 9% | 54% | 15,269 | 13,638 | 275 |
| H200 SXM | ours 0.85 | 7% | 8% | 44% | 19,265 | 17,267 | 896 |
| B200 | today | 10% | 11% | 61% | 13,520 | 12,043 | -58 |
| B200 | ours 0.5 | 6% | 7% | 40% | 22,013 | 19,756 | 1,261 |
| B200 | ours 0.7 | 4% | 5% | 28% | 32,259 | 29,060 | 2,852 |
| B200 | ours 0.85 | 4% | 4% | 23% | 39,948 | 36,043 | 4,046 |
| MI300X | today | 18% | 20% | never | 3,093 | 2,685 | -660 |
| MI300X | ours 0.5 | 5% | 6% | 35% | 13,680 | 12,299 | 984 |
| MI300X | ours 0.7 | 4% | 4% | 24% | 20,095 | 18,125 | 1,980 |
| MI300X | ours 0.85 | 3% | 3% | 20% | 24,904 | 22,492 | 2,727 |
| MI325X | today | 19% | 21% | never | 3,242 | 2,810 | -730 |
| MI325X | ours 0.5 | 5% | 6% | 35% | 14,716 | 13,230 | 1,052 |
| MI325X | ours 0.7 | 4% | 4% | 24% | 21,579 | 19,462 | 2,118 |
| MI325X | ours 0.85 | 3% | 3% | 20% | 26,714 | 24,126 | 2,916 |
| MI355X | today | 11% | 12% | 69% | 8,698 | 7,725 | -246 |
| MI355X | ours 0.5 | 4% | 5% | 28% | 23,811 | 21,449 | 2,101 |
| MI355X | ours 0.7 | 3% | 3% | 20% | 34,522 | 31,176 | 3,764 |
| MI355X | ours 0.85 | 3% | 3% | 16% | 42,561 | 38,477 | 5,013 |

Revenue per request: paid-today 5.300 m$, median-list 4.813 m$, commodity 0.823 m$.

### D. Whole OpenRouter Qwen3.8-27B demand (2,534,379 requests/day) in GPUs, 100% busy, >= 50 tok/s per user

| GPU | today's engine | ours eta=0.7 |
|---|---|---|
| H100 SXM | 54.3 | 17.2 |
| H200 SXM | 24.4 | 13.8 |
| B200 | 15.2 | 7.0 |
| MI300X | 55.2 | 11.4 |
| MI325X | 52.1 | 10.6 |
| MI355X | 23.2 | 6.7 |

### E. Price of speed: cost per request (m$), our engine at eta=0.7, rented (public), one GPU per replica (no tensor parallelism, no speculation)

| GPU | >= 30 tok/s/user | >= 50 tok/s/user | >= 100 tok/s/user | >= 150 tok/s/user | >= 250 tok/s/user |
|---|---|---|---|---|---|
| H100 SXM | 0.53 | 0.82 | infeasible | infeasible | infeasible |
| H200 SXM | 0.55 | 0.70 | 2.25 | infeasible | infeasible |
| B200 | 0.26 | 0.30 | 0.44 | 0.79 | infeasible |
| MI300X | 0.24 | 0.30 | 0.70 | infeasible | infeasible |
| MI325X | 0.24 | 0.28 | 0.53 | 3.60 | infeasible |
| MI355X | 0.19 | 0.21 | 0.31 | 0.58 | infeasible |
