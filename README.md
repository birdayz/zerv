# zerv

A general-purpose model-serving engine in Zig, built from scratch for maximum
performance and explicit control over resource use. No C++ dependencies.

First performance target: **Qwen3.8-27B on an AMD RX 7900 XTX, 24 GB VRAM**.

**Status:** independently validated native Q4_0/Q8_0/Q4_1/Q5_K/Q6_K CPU decoding, bounded GGUF
loading of the SHA-verified Qwen3.8 artifact, official text-only chat rendering and
Unicode-9 NFC and complete Qwen BPE tokenization/raw decoding, with repeatable
component benchmarks and an actual llama-server tokenizer comparison. Native Vulkan
memory/transfers/compute dispatch now pass independent hardware checks and repeated
matched driver benchmarks. Native GPU packed-weight projections now pass independent
checks across all complete dense model shapes, with repeatable benchmarks that
retain performance losses. The [matvec DFS](docs/bench/2026-09-22-matvec-optimization.md)
improves the full Q6 projection by ~4.6× over our scalar baseline, with unchanged
numerical gates and repeated reference comparisons. An external Vulkan oracle
served a real request. **zerv now runs Qwen3.8-27B natively** (`zig build server`,
then `zig-out/bin/zerv --model PATH`) and serves `POST /v1/chat/completions`;
outputs match llama.cpp under greedy decoding. With batched FP32 prefill
([report](docs/bench/2026-09-22-prefill.md)), split-K decode attention
([report](docs/bench/2026-09-22-decode-attention.md)) and small-row prefill plans
([report](docs/bench/2026-09-22-small-prefill.md)) and a scalar-X FP32 GEMM
([report](docs/bench/2026-09-22-gemm-throughput.md)), zerv decodes faster than tuned
llama-server at every measured context length and has lower TTFT up to ~100 prompt
tokens (102 vs 162 ms at 23 tokens). It is not yet the faster server overall: prompts of
~0.8–3.2K tokens take 1.7–2.3× longer than llama-server's default (Q8_1 activation)
prompt path, though less than its fully FP32 path.

[Development and test commands](docs/development.md).
[Matched tokenizer timings and verified optimizations](docs/bench/2026-09-22-tokenizer-matched.md)
include an audited libllama comparison, allocation control and retained regressions.

- [Controlled one-block-at-a-time work queue](TODO.md)
- [Agent instructions](AGENTS.md)
- [Research, design, verification, and implementation plan](docs/README.md)
