# Research-to-serving roadmap

**Native Qwen3.8 serving works end-to-end (blocks 09–11 closed, 12 open).** Next: batched prefill.
Tokenizer measurement/optimization and baseline replay are recorded. All required
packed CPU decoders, native Vulkan driver primitives and GPU matrix-vector
projections are verified. [Matvec optimization](research/matvec-optimization.md)
retains correctness gates, repeated comparisons and losses. Normalization/gates
and serving remain queued and must not resume automatically. All inference code
must be ours, in the permitted Zig/system-driver boundary, without C++ deps.
The goal is not achieved yet. Do not replace it with a proxy or mock endpoint.

Execution is controlled by [TODO.md](../TODO.md): one building block/package at a
time, with no parallel implementation tracks. Every milestone begins with completed
scoped research + a functional spec + an executable independent oracle, and ends
with recorded correctness and measurements.
The list below is a plan, not implementation evidence.

## 0. Ground truth and project rules — initial pass done

- [x] Record general-purpose/from-scratch/no-C++/extreme-performance requirements.
- [x] Establish `AGENTS.md`, `docs/`, and `docs/specs/` conventions.
- [x] Inspect actual hardware, GPU API, installed references and missing tools.
- [x] Confirm official model identity, architecture and candidate artifact sizes.
- [x] Derive separate attention-KV and recurrent-state memory estimates.
- [x] Specify reference-validation and performance-comparison requirements.
- [ ] Complete operation-level, format, tokenizer/template, and driver research.

## 1. Reproducible references and first native component — partial

Completed bounded slice: Zig 0.16.0 pinned/verified locally; external Q4_0/Q8_0
finite-domain goldens generated and independently scalar-checked before native
code; native CPU decoding (then explicit SIMD) passes in Debug/ReleaseFast. A
repeatable CPU comparison harness rebuilds/tests, checks full outputs, records
raw trials/manifests and compares the installed ggml decoder. Initial losses and
repeat runs are [recorded](bench/2026-09-22-quant-decode.md). This does **not** complete
model format/GPU/tokenizer/graph research or the reference-serving tournament.

Additional completed slices: the selected artifact is downloaded/SHA-verified;
native mmap/GGUF indexing passes complete-file metadata/descriptor/payload-sample
comparison and repeatable benchmarks. Official text-only chat rendering passes
100 independent Jinja cases and component timing. An external Vulkan llama-server
served a real greedy request (then stopped); it is not a native implementation.
Tokenizer probes confirmed official NFC vs llama non-normalizing behavior. Native
Unicode-9 NFC now passes normative/exhaustive fixtures and two component benchmark
runs; [results and limitations](bench/2026-09-22-normalization.md). The subsequently
completed [text-splitter block](bench/2026-09-22-tokenizer-split.md) passes 47,919 HF
cases, exhaustive properties and two benchmark runs. The [complete tokenizer](bench/2026-09-22-tokenizer.md)
now passes exact official encoding/raw-piece fixtures and the actual-GGUF checker.
Two benchmark runs include actual llama-server; all 380 normalization differences
were independently reconciled by submitting NFC inputs.

Remaining:

- Native driver/shader-toolchain and all mixed diagnostic decoders are now verified;
  implement verified GPU kernels for the confirmed mixed tensor types. Streaming UTF-8 repair belongs at the future text-output boundary.
- Resolve `output_gate_type` semantics, EOS policy, artifact template differences,
  DeltaNet recurrence/chunking, and exact tensor layouts against pinned references.
- Build/test the external fixture extractor, capturing blocks/ops/intermediates/
  logits and tokenizer/template outputs; record manifests and concrete tolerances.
- Keep the verified artifact pin and actual inventory; quantify GPU workspace/cache
  budget during native backend bring-up. Do not download every precision variant.
- Run correctness-checked external baselines and tune qualifying candidates. Identify
  the fastest *measured* reference rather than picking one by reputation.
- Expand the verified quant-block slice through the required container parsing,
  remaining artifact tensor types, and scalar diagnostics after their research gates.

Exit: reproducible reference fixtures and a real native component pass; toolchain
and artifact identity are fixed. No C++ runtime dependency is introduced.

## 2. Native GPU primitives

Native Vulkan device/memory/transfers/dispatch completed with independent ABI,
scalar-output and hardware tests plus repeated matched raw-driver measurements;
[results and invalidated initial runs](bench/2026-09-22-gpu-driver.md).
Remaining research/specced slices: quantized
matvec/matmul → reductions/norm/gates → full attention → DeltaNet recurrence and
chunked prefill. Each slice gets real-shape, adversarial, and seeded reference cases.
Inspect generated shaders/ISA and benchmark only after numerical acceptance.

Exit: all target primitive semantics work natively, with error reports and timings;
peak scratch/residency is measured. No hot operator delegated to an external engine.

## 3. Complete one correct native model session

Wire verified loading, graph planning, native tokenization/template, prefill,
decode, sampling and termination. Reuse buffers/commands and keep weights resident.
Compare intermediate tensors and teacher-forced logits, then full generation;
exercise long recurrent trajectories and context boundaries.

Exit: Qwen3.8 text generation succeeds on the 7900 XTX, passes independent model
comparisons, and fits an explicit memory budget. Plausible output alone fails.

## 4. Serve Chat Completions v1

Research/spec the exact API subset, implement bounded HTTP/SSE and request
lifecycle, then add independent protocol tests and direct-engine/API comparisons.
Implement health/readiness, effective configuration, usage, metrics, limits,
cancellation/backpressure, overload and shutdown behavior.

Exit: a real client can call `POST /v1/chat/completions` and stream actual native
Qwen3.8 output; correctness, bounded resource use, and no-proxy/no-C++ constraints
are verified. This is the broader serving goal's functional milestone, not yet a
claim of competitive speed.

## 5. Competitive execution and serving

Use full end-to-end profiles to select work: kernels, layouts, synchronization,
GPU sampling, batching/chunked prefill, state/cache planning. Add tuned latency,
throughput, and low-VRAM profiles. Run repeated matched competitor matrices,
including tails, memory peaks and quality, not only tokens/s on an empty context.

Exit: publish wins/losses against the strongest qualified reference. Keep iterating
until the first target is genuinely competitive; document every important result.

## 6. Broaden functionality without losing the fast path

Additional architectures/devices, vision, tools/structured output, prefix caching,
MTP/speculation and multi-GPU each require their own complete research/spec/oracle
cycle. Order by measured user value, not by building a framework for its own sake.
