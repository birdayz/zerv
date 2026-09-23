# Verification specification

**Mandatory policy.** The quant-block golden mechanism is implemented; full-model
and serving mechanisms are not. References, research, and acceptance rules must
exist before the corresponding implementation is started.
Tests against our own implementation alone are insufficient when independent
reference implementations are available.

## 1. Research-readiness gate

For each feature, create a research record under `docs/` and link it from its
functional specification. It must establish:

- The full operation/protocol: mathematics, order of operations, shapes, strides,
  dtypes, packing/endianness, rounding, normalization, state updates, and errors.
- The exact reference implementation path/revision and relevant configuration.
  Read implementation behavior, not only its README. Document discrepancies between
  specifications, model metadata, converters, and reference code; resolve them.
- All relevant edge cases: padding, empty input, tails, overflow, invalid files,
  unsupported quantization, special tokens, sequence boundaries, long contexts,
  cancellation, aliasing, concurrent ownership, and applicable device constraints.
- Alternatives and expected performance/memory costs, including possible numerical
  changes. Record remaining questions explicitly. Unresolved semantics or absence
  of a credible oracle block the affected code, not just its release.
- An executable reference-extraction/comparison mechanism, fixture format, and
  acceptance thresholds. Validate that the reference can actually produce the
  required data. A list of future test ideas does not satisfy this gate.

Research is scoped to the next functionality, but must be complete for that scope.
Do not use project-wide ambition as an excuse for coding a partly understood kernel.

## 2. Independent references

Use two different reference roles:

1. **Semantic/quality oracle:** pinned official Transformers implementation and
   original checkpoint/tokenizer/template. It anchors model meaning and measures
   quantization loss. CPU/slow execution or layer-sized fixtures are acceptable;
   the oracle does not need to be fast or fit fully on this GPU.
2. **Quantized compatibility oracle:** a pinned, verified llama.cpp build loading
   the **identical GGUF artifact** and exact prompt token IDs. It anchors our
   container/quant compatibility and practical whole-model comparisons. It is not
   a substitute for original-model semantics when artifacts/templates differ.

External tools run in a separate development environment/process; the engine must
not link or call them for inference. Pin their versions and extraction code.
A wrapper process in a test harness is allowed; a wrapper production engine is not.
If the oracles disagree, investigate metadata, template, dtype, and operation
mapping rather than voting for whichever matches our code.

## 3. Concrete comparison mechanism to implement

The intended workflow is an **external reference runner → immutable fixtures →
native comparison runner**, with optional live differential tests. The feature
research must turn this workflow into tested exact commands before engine code.

- The reference runner consumes a canonical JSON case with exact input tokens or
  named tensors, model/artifact hashes, sampling settings, and requested captures.
- A reference-only adapter/hook captures named operations/layer boundaries and
  logits, copies to CPU, normalizes to a documented layout/dtype, and writes raw
  little-endian tensor payloads with JSON metadata. Use non-pickled data.
- Goldens carry shapes, dtype, strides/layout, byte length, payload SHA-256,
  reference commit/build, device, source artifact/hash, generator revision,
  generation command, random seed, input hash, and all effective settings.
- Our runner accepts the same case and produces corresponding outputs. A native
  comparator reports the first failing operation/index and aggregate error metrics;
  it exits nonzero on mismatch, malformed/missing fixture, or skipped required case.
- Keep small fixtures and manifests under `tests/fixtures/` once code exists;
  research/results remain in `docs/`. Large reference tensors live in an explicit
  artifact store with hashes and reproducible regeneration instructions, not git.
- Never automatically update expected outputs from our code. Golden refreshes
  require running the pinned reference, recording why, and reviewing the diff.

## 4. Required comparisons

| Area | Independent evidence / cases | Comparison |
| --- | --- | --- |
| Container and tensor loading | GGUF spec + reference tensor metadata and malformed files | Exact names, dims, type, payload offsets/bytes; deterministic rejection |
| Quant decoding / repacking | Reference dequantized blocks; scale/sign/extreme/tail cases | Exact when conversion arithmetic is exact; otherwise predeclared error bounds; pack/unpack invariants |
| Tokenizer / template | Official tokenizer/template goldens; Unicode, whitespace, special tokens, multi-turn, thinking, invalid roles | Exact bytes and token IDs, exact special-token policy |
| Primitive kernels | Reference ops + independent scalar calculations with seeded inputs and real model shapes | Shape/layout and finite-value checks, abs/relative/error-norm bounds |
| Model graph | Matched per-layer intermediates and full-vocabulary logits on fixed tokens | Error metrics and first divergence, not merely matching argmax |
| Recurrent state | Token-at-a-time vs chunked reference; interleaved sequences, snapshot/restore, long horizon | State tensors plus logits; reset/isolation/restore invariants |
| Generation | Fixed inputs/settings, EOS, output caps, stop strings and Unicode token pieces | Greedy tokens where stable; logit evidence at near ties; exact stopping/accounting behavior |
| Sampling | Synthetic logits + reference probability transformations | Probabilities, filters, penalties, deterministic seed within our RNG; statistical tests for stochastic outputs |
| Serving | Direct-engine output vs HTTP/SSE; cancellation, backpressure, overload | Same token stream, valid framing/order, correct usage, no leaks/cross-request state |

Explicitly compare **batched vs unbatched**, **chunked vs unchunked prefill**, and
**prefix-restored vs freshly computed** paths against the oracle. Arbitrary
sequence interleavings expose bugs a single uninterrupted conversation misses.

## 5. Numerical acceptance and quality

Declare tolerances per operation/dtype/shape **before** optimizing. Research must
measure the reference's own precision behavior and establish concrete thresholds;
no single blanket epsilon, no tolerance chosen after seeing a failing result.

Record max absolute error, max relative error with a specified denominator floor,
normalized error, and non-finite counts. For logits add top-k agreement and
KL/probability differences where meaningful. Reductions may differ in order;
bitwise FP equality is not generally a valid requirement. Do not hide large errors
behind averages or treat unstable near-tie argmax differences as automatic proof
of correctness/incorrectness. Compare teacher-forced logits on fixed tokens after
free-running generation diverges.

Separate:

- **Implementation correctness:** same quantized weights, same tokens, same intended
  arithmetic. Our operator must implement its specified math/layout faithfully.
- **Quality tradeoff:** lower-precision weights/KV/state vs higher-precision oracle;
  use fixed held-out text/log-likelihood and representative task suites, including
  long-context retrieval/reasoning and long recurrent trajectories.

A more aggressive quant, approximated operation, changed sampler, speculative
method, or truncated context must meet its own declared quality contract. Greedy
speed tests do not validate stochastic sampling or reasoning quality. Exact RNG
streams across independent engines are not assumed.

## 6. Release and optimization gates

1. CPU/scalar unit and malformed-input tests pass with allocation/leak checks.
2. Mandatory reference fixtures pass; GPU tests fail or are explicitly marked
   unavailable when no supported GPU exists, never silently counted as passing.
3. Real target model comparisons pass, including long context and concurrent state.
4. Server integration and repeated load/cancel/unload tests pass without unbounded
   VRAM/host-memory growth.
5. Only then compare speed at matched quality, per [performance protocol](../performance.md).

Every optimized kernel keeps a simple validated fallback for diagnosis until its
coverage is strong. Record failing seeds/tensors as regression cases. "Proving
correctness" here means reproducible evidence and explicit bounds/invariants,
not a claim that finite tests provide a formal proof for every possible input.

**Readiness today:** external Q4_0/Q8_0 goldens were generated and cross-checked
before native implementation; all finite scalar products match the native CPU
implementation bit-exactly in Debug/ReleaseFast. The component benchmark verifies
full input/output hashes against the external decoder before accepting results.
See [quantization.md](quantization.md) and [quant-benchmark.md](quant-benchmark.md).
GPU/model/tokenizer/template/protocol research, fixtures and concrete numerical
thresholds remain prerequisites to those implementations. No model comparison
or serving correctness claim has been made.
