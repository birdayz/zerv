# zerv — agent instructions

## Non-negotiable mission

- Build a **general-purpose model-serving engine in Zig**, with the inference
  stack implemented **entirely from scratch**.
- Performance is a primary requirement, not a later optimization pass. The first
  competitive target is **Qwen3.8-27B on this machine's RX 7900 XTX (24 GB)**.
  The objective is to be the fastest correct server for this model/card, not
  merely to get it running. Do not claim that objective is achieved without evidence.
- General-purpose architecture; specialized fast paths. Do not hard-code the
  entire engine to Qwen or AMD. Do not impose generality/portability overhead on
  the execution hot path. Resolve model, device, and kernel choices before execution.
- Configurability is part of the product: expose explicit, measurable tradeoffs
  among latency, throughput, VRAM, context capacity, concurrency, and quality.
- Incoming inference interface: **OpenAI-compatible Chat Completions v1**,
  `POST /v1/chat/completions`. Do not substitute a Responses API or custom protocol.
- **No C++ dependencies.** No importing C++ headers, linking C++ inference/math
  libraries, wrapping them behind a C ABI, copying their implementation into our
  engine, or delegating production inference to another process/service.
- Implement our own model loading, tensor representation, quantization handling,
  tokenizer, model execution, GPU kernels, cache/state management, sampling,
  scheduling, and serving integration. Existing engines are **external test
  oracles and benchmark competitors only**, never the implementation.
- Zig's standard library and OS/GPU driver interfaces are the infrastructure
  boundary. A C ABI for a system interface is not permission to import a C++
  implementation library. External compilers/profilers/reference programs may be
  development tools, not linked/vendored runtime dependencies. Record any proposed
  boundary expansion in docs and obtain approval before adding it.

## Read first / preserve knowledge

- Start with [docs/README.md](docs/README.md) and the controlled [TODO.md](TODO.md), then the documents relevant to the task.
- **All research, design knowledge, decisions, experiments, and benchmark findings
  belong under `docs/`.** **`docs/specs/` is the source of truth for specifications
  of functionality.** Keep this file focused on instructions.
- Distinguish observed facts, source-backed facts, estimates, proposals, and open
  questions. Record source URL, revision, date, commands, and limitations.
- Put third-party source used for research in the gitignored **`third_party/`**
  directory, with stable, versioned paths—not `/tmp`. Record origin, exact revision,
  local path, and hashes under `docs/`. These files are research/oracle material,
  never production dependencies; our findings still belong in `docs/`.
- Preserve negative results and correctness failures. Never leave important
  findings only in chat, temporary files, or an agent's memory.
- Keep the docs index and implementation status accurate. A proposal is not a
  feature; device enumeration is not successful inference; a benchmark recipe is
  not a measured result.

## How to work

- Inspect the actual machine and installed versions rather than assuming CUDA,
  ROCm, a Zig version, GPU features, free VRAM, or model compatibility.
- **Before coding, complete the research for the functionality being implemented
  and understand it in full.** Read the primary specification and relevant reference
  implementations; document the mathematics, layouts, precision, state transitions,
  edge cases, hardware constraints, and alternatives under `docs/`. Resolve open
  semantic questions; do not guess or implement around gaps in understanding.
- Write/update the functional specification in `docs/specs/` before implementation.
  The research must identify an executable correctness mechanism and acceptance
  criteria. If these are missing, the feature is not ready for coding.
- **When a reference implementation is available, differential testing or
  reproducible golden fixtures against it is mandatory.** Work out the mechanism
  before implementation: exact reference revision, inputs/artifacts, extraction
  method, comparison rules, tolerances, and reproducible commands. Self-comparison
  or plausible text is not a substitute.
- Then work in small, runnable increments: reference fixture → implementation →
  tests → measurement → record findings. Finish the verification loop.
- Keep exactly one active building block/package in `TODO.md`. Close its research,
  tests and benchmark gates before advancing; do not run parallel implementation
  tracks. Require actual llama-server comparisons at runnable integration milestones,
  and document non-equivalent component semantics instead of claiming false wins.
- **Code quality, tests, maintainability, package boundaries, and clear interfaces
  are non-negotiable.** Performance is not permission for tangled ownership,
  untestable code, dependency cycles, or undocumented cross-package coupling.
- Keep packages cohesive, their public APIs small, and dependencies directional.
  Specify ownership, lifetimes, errors, invariants, and concurrency at boundaries.
  Test through those interfaces; isolate hardware/protocol details from model math.
- Keep code Zig-first, explicit, and understandable. Use explicit ownership and
  allocators; bound queues, buffers, context, output, and scratch space.
- No per-token heap allocation in the steady-state execution path. No avoidable
  host/device transfers, device-wide synchronizations, or runtime graph rebuilding.
  Instrument and justify unavoidable costs rather than hiding them.
- Use checked arithmetic and validate dimensions, offsets, sizes, and resource
  limits before entering optimized code. Reject unsupported features explicitly;
  never silently use the wrong model operation, precision, template, or device.
- Keep changes scoped. No speculative framework building, unrelated cleanup,
  placeholder inference, fake metrics, or dead configuration knobs.
- Do not change drivers, GPU clocks/power limits, system packages, or start large
  weight downloads as incidental setup. Make such steps explicit and record them.

## Correctness and performance gates

- Follow [docs/specs/verification.md](docs/specs/verification.md) and
  [docs/performance.md](docs/performance.md). Correctness gates every optimization.
- Compare independent scalar/reference calculations, intermediate tensors,
  full-model logits, tokenization/templates, and request/state behavior. Meaningful
  output alone is not evidence of correct inference.
- Separate implementation error from quantization quality loss. Compare the same
  weights/tokens/precision for numerical correctness; evaluate lower-precision
  artifacts separately against a higher-precision reference.
- **Always include a tuned `llama-server` in serving benchmarks**, and also evaluate
  other servers that claim strong performance on a relevant model/hardware path.
  Compare against the **best tuned compatible reference we can find**, not just
  a convenient default or an old installed binary. Revisit the competitor set;
  a library microbenchmark does not replace the `llama-server` comparison.
- Maintain repeatable **component benchmarks for every performance-sensitive
  component**, not only end-to-end tests: kernels/operators, transfers/submission,
  allocation/state/cache operations, tokenizer/template/sampling, scheduler and
  protocol overhead. Use representative shapes, sizes, dtypes and concurrency;
  compare equivalent reference components when available. Keep numerical gates.
- Pair component evidence with full-serving measurements when that path exists;
  require both before claiming a serving speedup. Measure prefill, decode, TTFT,
  inter-token latency, aggregate throughput, tail latency, peak VRAM, and host
  memory separately. Match workload, quality, context, concurrency, cache state,
  and resource constraints; include warmup and repeated runs.
- **Benchmarks must be repeatable:** a checked-in harness/command, fixed versioned
  workloads/seeds, pinned model/reference/build hashes, explicit effective configs,
  controlled warmup/cache/load conditions, repeated trials, and raw results with
  variance. Another run must be reproducible without reconstructing steps from chat.
- **Benchmark reports go in `docs/bench/YYYY-MM-DD-topic.md`**, with the date,
  question, setup, exact commands, results/variance, comparison and interpretation.
  Link raw results and complete manifests stored alongside under `docs/bench/`.
  Report regressions and failed runs. Never substitute a theoretical roofline or
  a kernel microbenchmark for end-to-end evidence.
- Prefer the fastest *verified* design, even when that means replacing our own
  code. Do not lower quality, omit work, or narrow the test to manufacture a win.

## Commands and completion

Bazel builds and tests everything (`.bazelversion`; Zig **0.16.0** pinned in
`MODULE.bazel`). See [docs/development.md](docs/development.md) for verified toolchain setup
and commands. Harnesses build through `tools/zerv_build.py`, never another build tool.

Required checks after native changes (`zig fmt --check`, every CPU unit test in Debug and
ReleaseFast, the Python tests):

```sh
bazel test //...
```

After GPU, shader or kernel changes also run the device tests in both modes and the shader
spill gate:

```sh
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu   # the same tests on the host's driver (production's)
```

Run Python tools and harnesses with `tools/py SCRIPT` (the pinned interpreter; scripts refuse
the host's Python). Production benchmarks gate themselves on the host-driver GPU tests.

Reference-golden regeneration is separate and explicit; ordinary builds/tests must
work without external reference libraries or `third_party/`. Never report a test,
benchmark, or model run as passing without actually executing it.

For each completed change, state what changed, what was actually verified, and
what remains blocked. When a performance claim is involved, link its evidence.
