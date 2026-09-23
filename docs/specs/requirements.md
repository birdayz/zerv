# Mission and scope

## Requirements from the user

1. General-purpose model serving, implemented in Zig.
2. Rewrite the inference stack **entirely from scratch**. **No C++ dependencies**,
   including dependencies disguised behind a C ABI or subprocess proxy.
3. Extreme performance: aim to be the fastest correct serving implementation for
   **Qwen3.8-27B + RX 7900 XTX 24 GB** as the first competitive target.
4. Configurability for minimizing VRAM use or maximizing useful utilization, not
   a single opaque configuration optimized for a demo.
5. Grind through implementation, profiling, and comparison against the strongest
   reference available. Establish correctness, not just plausible text output.
6. `AGENTS.md` contains working instructions; all research/knowledge is in `docs/`;
   `docs/specs/` contains specifications of functionality.
7. Complete research and fully understand the relevant functionality before coding.
8. Where reference implementations exist, establish a concrete, reproducible
   correctness mechanism against them (differential tests or golden fixtures)
   before implementing the feature.
9. Code quality and tests are non-negotiable. The project must remain highly
   maintainable, with cohesive packages, explicit boundaries/interfaces, and
   deliberate dependency direction. Performance cannot excuse architectural debt.
10. The incoming inference interface is Chat Completions v1,
    `POST /v1/chat/completions`.

General-purpose does not mean implementing every architecture/backend before the
first usable model. Model descriptions and resource policy should be extensible;
execution plans and kernels should specialize aggressively to known shapes and
hardware. A Qwen-only global engine or a generic interpreter in the token loop
would each miss a different part of the requirement.

## Implementation boundary

Our code owns model/container parsing, tokenizer and template behavior, tensor
layouts, quantized operators, inference kernels, execution planning, sampling,
state/cache management, scheduling, and API integration. No llama.cpp/ggml,
PyTorch, ONNX Runtime, tokenizers library, BLAS engine, or other inference/math
implementation is imported, linked, vendored, or called to perform production
inference. No C++ code or C++ dependency in the engine.

Zig standard library, operating-system services, and raw GPU driver interfaces
are infrastructure. Author GPU kernels ourselves; consuming a GPU driver API is
not permission to consume a prebuilt operator library. A standalone shader
compiler or profiler is a development tool, not an engine dependency. Keep this
distinction visible in the build/dependency manifest. If a future dependency does
not clearly fit this boundary, ask before adopting it.

Reference implementations may run separately to generate test fixtures or compete
in benchmarks. Research their documented mathematics and behavior; implement our
own code. Preserve provenance of fixtures and respect upstream licenses.

## Proposed first usable scope

- Single model, single GPU, text generation from the target model's language path.
- Streaming and non-streaming chat completions, bounded requests/output/context,
  cancellation, health/readiness, model identification, and performance metrics.
- Localhost by default; never expose an unauthenticated server publicly by default.
- Correct official tokenization, chat template, thinking controls, and termination.
- Explicit precision, memory, scheduling, and context settings.
- Start correctness bring-up with one sequence; do not call the server competitive
  until bounded concurrent serving has been measured too.

Text-only first is a proposed sequencing decision, not a claim the model is
text-only. Vision/video, tool-call compatibility, other model architectures,
multi-GPU, and distributed serving are subsequent extensions. Requests for
unimplemented capabilities must fail explicitly, not be silently reinterpreted.

## What success means

There is no honest single "fastest" number across latency, throughput, context,
quality, and memory. Maintain separate low-latency, throughput, and low-VRAM
profiles, with published constraints and a reproducible comparison matrix.

The goal is to win against the best measured compatible competitor in each target
workload, at matched correctness/quality and resource limits. The first baseline
sets the numbers; do not invent tokens/second targets now. A win in one cell is
not a blanket claim across the matrix. GPU occupancy alone is not success;
useful work delivered at the requested latency/quality is.

**Current status:** charter/initial research established; feature research and
specifications not yet complete; fastest reference not yet established. No model
inference or correctness/performance claims. See [verification.md](verification.md)
for the mandatory research-to-implementation gate.
