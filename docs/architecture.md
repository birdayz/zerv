# Architecture research and proposed direction

**Not a completed design.** Functional contracts live in [specs/](specs/README.md).
Every component needs full feature research and a concrete reference-validation
mechanism before implementation. No backend or speed claim is established yet.

## General engine, specialized execution

```text
HTTP / application API
  → validation, native tokenization/template rendering
  → admission + scheduler
  → compiled model execution plan
  → native operators/kernels + explicit memory/state management
  → raw GPU driver interface
```

Model import produces an architecture description with verified shapes, tensor
types, constants, and state requirements. Model-specific planners lower it to a
reusable device execution plan. Backend capability/shape dispatch occurs during
planning or at bounded batch-shape transitions, not as a generic graph interpreter
on every token. Shape-specific kernels are a strength, not a loss of generality.

Proposed responsibilities, not a mandate to build empty abstractions first:

| Component | Responsibility |
| --- | --- |
| Artifact importer | Bounded GGUF parsing, metadata validation, native quant decoding, future additional formats |
| Model family | Exact graph/norm/position/attention/recurrence semantics from metadata |
| Tensor/memory | Dtype/layout, ownership, residency, lifetimes, checked byte accounting |
| Planner | Shape specialization, kernel choice, buffer liveness/reuse, command recording |
| Backend | Device capabilities, allocations, transfers, dispatch, synchronization, timings |
| Session state | Full attention KV plus recurrence/convolution, sequence identity, snapshots |
| Scheduler | Admission, bounded prefill/decode scheduling, fairness, cancellation |
| Token/generation | Native tokenizer/template, sampling, detokenization, stop/EOS |
| Serving | Protocol, bounded buffering, accounting, observability |

A scalar CPU path is an independent diagnostic/reference aid, not a silent
production fallback. Keep a clear seam for later models/devices without requiring
multiple production backends in the initial release.

## GPU boundary: no C++ dependency

First candidate: raw **Vulkan C API**, called from Zig using minimal explicit
bindings, plus our own compute shaders. No Vulkan-Hpp, ggml, llama, BLAS, or
prebuilt inference operators. Shader compiler invocation is a build tool; shader
source, operation implementations, and runtime dispatch are ours. Pin tool versions
and generated shader hashes. Query actual device support rather than assuming
all Vulkan implementations execute the same path efficiently.

Vulkan is selected for initial feasibility because RADV works here, not because
HIP is assumed slower. A future alternative must use an allowed system-driver
interface without C++ wrappers/math dependencies and win a controlled comparison.
External HIP-based engines remain valid benchmark competitors.

Study API allocation/alignment limits, memory types/budgets, descriptor/buffer
addressing, barriers, queue submission, timeline synchronization, subgroup control,
cooperative matrix shapes, timestamp validity, and device-loss handling before
writing the relevant backend code. These details have **not** been fully researched.

## First kernel research areas

- **Quantized matvec:** short-context batch-one decode is a likely weight-bandwidth
  problem. Explore coalesced repacked blocks and fused dequantization/dot products.
  Never permanently inflate all Q4 weights to FP16 to simplify matmul.
- **Quantized matmul:** prefill and batched decode have different shapes/intensity.
  Study RDNA3 cooperative-matrix/WMMA lowering versus vector/integer-dot kernels;
  tile/occupancy/LDS/register pressure must be measured on this card.
- **Attention:** 256-dimensional full-attention heads with GQA, partial/interleaved
  RoPE, Q/K normalization, output gating, and long-context KV traffic. Fuse safely
  and avoid materializing the full attention score matrix.
- **DeltaNet:** correctly normalized Q/K, convolution, decay/update, gating, chunked
  prefill, and one-token recurrent updates. Chunk size trades workspace and speed;
  retain stable accumulation and verify long trajectories.
- **Elementwise/reductions:** residual/norm, activation/gates, softmax, embeddings,
  and the large-vocabulary output head. Fuse only where numerical tests and
  occupancy show a benefit.
- **Sampling:** measure device filtering/sampling versus host transfer. Downloading
  the whole 248,320-element FP32 logit vector every token costs about 0.95 MiB
  plus synchronization; avoid it in the fast path unless measured superior.

These are hypotheses for profiling, not a preset optimization order. Actual
profiles determine which kernel gets rewritten next. Inline assembly or specialized
layouts are options after compiler-generated code has been inspected and measured.

## State, scheduling, and memory

Start with a correct single-sequence plan and preallocated buffers. Then introduce
bounded multi-sequence batches, chunked prefill to protect decode latency, and
explicit total-token/sequence capacity. One owner schedules mutable device state;
network workers do not mutate KV/recurrent buffers directly.

Avoid per-token allocations, repeated weight uploads, unnecessary graph/command
rebuilds, and device-wide waits. Synchronize only where dependencies require;
measure command submission and GPU timestamps separately. Pre-record reusable
work where the driver model and shapes make this correct and faster.

Paged KV/prefix caching are not automatic requirements for the first decode loop.
Compare memory efficiency against contiguous allocation; hybrid recurrence makes
prefix branching/eviction/rollback a correctness problem as well as a performance
problem. Speculation/MTP is a later experiment, not free speed: account for draft
weights, verification batches, acceptance, and all recurrent rollback states.

## Before writing the first runtime component

Pin Zig and the shader toolchain; inspect normative GGUF and Vulkan behavior;
inspect candidate artifact tensor types; establish independent fixtures; choose
one bounded vertical slice (e.g. one real-shape quantized operation). Specify its
inputs/results/precision and prove the reference-extraction mechanism works. Do
not skip these steps to create a large unvalidated server skeleton.
