# External Qwen reference bring-up

This is an independent compatibility/oracle experiment, **not zerv serving**.
Use the already-installed llama-server (build10964 / b29c606e), never as a production
proxy. It may load the now fully SHA-verified selected artifact onto Vulkan for
oracle generation. Bind only 127.0.0.1:18081, no Web UI, context 2048, one slot,
batch/ubatch 256, all layers offloaded, flash attention on, f16 KV, MTP/speculation
none and context shifting disabled. No driver/clock/power/package changes.

Record executable/library hashes, launch command/log/effective config and model
hash. Poll GET /health with a bounded wait. A deterministic short arithmetic
Chat Completions request uses temperature 0, max_tokens 32, enable_thinking=false,
stream=false and the explicit official template file. Save request, response,
HTTP status and wall time. Compare expected answer as a smoke check, not proof of
intermediate/logit correctness. Only report an actual successful request. Stop
reference process when oracle experiments finish so VRAM is available to zerv.

The same localhost process may supply /tokenize results (add_special=false,
parse_special explicitly set) and /apply-template for independent tokenizer/template
comparisons. First inspect its pinned API documentation; retain requests/responses.
No credentials/private inputs involved. Reference endpoint failure must not be
hidden by falling back to production proxying or fabricated output.

This is not a tuned benchmark. Full serving comparison still requires the repeated
matched workload matrix, strongest compatible competitors, numerical gates, and
TTFT/decode/throughput/tails/memory measurements specified in performance.md.
