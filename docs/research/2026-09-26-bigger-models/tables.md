### Market (OpenRouter, 30 days to 2026-09-25; price USD per 1M)

| model | license | params | providers | input median | output min / median / max | spend per 30 d | requests/day | prompt:completion | cached |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen3.8-27b | apache-2.0 | 28B | 16 | 0.22 | 1.80 / 2.55 / 4.40 | 284k | 1.87M | 22 | 71% |
| qwen/qwen3.6-35b-a3b | apache-2.0 | 36B | 10 | 0.14 | 0.70 / 1.00 / 1.80 | 56k | 1.84M | 8 | 40% |
| qwen/qwen3.5-122b-a10b | apache-2.0 | 125B | 5 | 0.29 | 2.08 / 2.40 / 3.20 | 26k | 0.22M | 7 | 0% |
| qwen/qwen3.5-397b-a17b | apache-2.0 | 403B | 10 | 0.55 | 2.34 / 3.55 / 4.50 | 126k | 0.45M | 7 | 40% |
| qwen/qwen3.8-2.4t-a95b | other | 2,446B | 7 | 2.00 | 6.00 / 6.00 / 6.00 | 113k | 0.13M | 16 | 85% |
| qwen/qwen3.8-flash | other | 180B | 1 | 0.15 | 0.47 / 0.47 / 0.47 | 61k | 1.42M | 28 | 88% |
| z-ai/glm-5.3-flash | mit | 321B | 33 | 0.15 | 0.14 / 0.50 / 1.00 | 2,822k | 49.08M | 37 | 84% |
| z-ai/glm-5.3 | other | 753B | 40 | 1.35 | 1.19 / 4.40 / 8.80 | 4,412k | 6.15M | 57 | 89% |
| moonshotai/kimi-k3 | other | 2,780B | 19 | 3.00 | 9.04 / 15.00 / 22.50 | 4,782k | 3.76M | 59 | 89% |
| minimax/minimax-m3 | other | 427B | 13 | 0.30 | 0.96 / 1.20 / 3.00 | 234k | 4.13M | 57 | 87% |
| deepseek/deepseek-v4.1-flash | mit | 763B | 28 | 0.22 | 0.29 / 0.87 / 1.50 | 2,598k | 35.58M | 63 | 89% |
| deepseek/deepseek-v4-flash-0731 | mit | 304B | 31 | 0.14 | 0.13 / 0.35 / 1.32 | 1,983k | 71.18M | 26 | 81% |
| deepseek/deepseek-v4-pro-0813 | mit | 1,650B | 22 | 1.09 | 0.75 / 3.73 / 4.95 | 972k | 4.20M | 40 | 86% |

### Serving (InferenceX profit estimator: agentic traces, OpenRouter default price, revenue per GPU-hour at 100% busy, USD)

| InferenceX model | target tok/s/user | price in / cached / out | MI355X best | MI355X revenue / InferenceX rent | best 8-GPU NVIDIA (B200/B300) | best rack-scale NVIDIA (GB200/GB300) | MI355X GPUs per decode replica |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 24 | 0.264 / 0.0088 / 0.792 | 4.74 (sglang) | 1.6x | 9.91 (vllm) | - | 4, 8 |
| DeepSeek-V4.1-Flash | 125 | 0.3 / 0.006 / 1.2 | 5.17 (atom) | 1.8x | 6.00 (sglang) | 7.63 (sglang) | 2, 4 |
| GLM-5.2 | 100 | 0.379 / 0.07 / 1.19 | 4.92 (atom) | 1.7x | 6.39 (dynamo-sglang) | 12.72 (dynamo-trt) | 4, 8 |
| Kimi-K3 | 45 | 3 / 0.3 / 15 | 15.78 (atom) | 5.4x | 17.36 (vllm) | 31.79 (dynamo-vllm) | 8 |
| MiniMax-M3 | 83 | 0.3 / 0.06 / 1.2 | 13.43 (atom) | 4.6x | 22.44 (trt) | 51.57 (dynamo-trt) | 2, 4, 8, 16, 64 |
| Qwen-3.5-397B-A17B | 45 | 0.55 / 0.22 / 3.5 | no row | - | - | - | - |
| Qwen3.8-Flash-Next | 45 | 0.15 / 0.016 / 0.47 | no row | - | - | - | - |
