# 2026-09-22 — external Qwen3.8 Vulkan compatibility smoke test

**External oracle only, not zerv inference, not a tuned serving benchmark.**
The installed llama-server build10964 / b29c606e successfully loaded the exact
SHA-verified Qwen3.8-27B Q4_0 artifact on this RX 7900 XTX and served one actual
`POST /v1/chat/completions` request. This establishes an executable compatibility
reference; it does not satisfy zerv's serving goal or prove numerical correctness.

[Launch command](data/2026-09-22-reference-bringup/command.txt),
[server log](data/2026-09-22-reference-bringup/server.log),
[effective properties](data/2026-09-22-reference-bringup/props.json),
[binary/model manifest](data/2026-09-22-reference-bringup/manifest.json).
No system package, driver or power/clock settings changed. The reference process
was stopped after tokenizer probes to release GPU resources.

Configuration: loopback port 18081, 2,048 context, one slot, b/ub 256, ngl 99,
flash attention on, FP16 KV, no speculation, no context shifting or Web UI,
Jinja with the **official pinned template**, thinking disabled for the request.
The loader reports ignoring all 15 optional MTP tensors, as intended.

Request: “What is 2 + 2? Reply with only the number.” Greedy temperature 0,
max_tokens 32, non-streaming. HTTP **200**, content **`4`**, finish_reason **stop**,
26 prompt tokens, 2 completion tokens including termination.
[Exact request](data/2026-09-22-reference-bringup/request.json) and
[response](data/2026-09-22-reference-bringup/response.json).

Observed one-request wall time: 793.983 ms; reference reports prompt 632.789 ms and
prediction 148.432 ms. No warmup/repetition, no tail/throughput statistics, no
quality or intermediate/logit differential test. These values are a smoke result,
**not competitive performance evidence**. Tuned llama-server plus the strongest
other compatible server, longer/repeated workloads, memory accounting and native
zerv comparison remain outstanding. [Bring-up contract](../specs/reference-bringup.md).

Tokenizer probing through this independent server confirmed the normalization
mismatch: official HF Tokenizers NFC-composes decomposed accents/Hangul; the
GGUF llama QWEN35 tokenizer does not. See
[raw probes](../research/2026-09-22/tokenizer-probes.json). Native tokenizer policy
must not silently inherit that discrepancy or use ASCII-only tests to hide it.
