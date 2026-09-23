# Repeatable benchmark records

Benchmarks live in **`docs/bench/`**. Each report is a dated Markdown file:
`YYYY-MM-DD-topic.md`. Correctness-suite runtimes are not benchmark results.

- [2026-09-22: Q4/Q8 CPU decode, external reference, SIMD and repeat run](2026-09-22-quant-decode.md).

- [2026-09-22: native GGUF parse/index vs independent reader](2026-09-22-gguf-loading.md).
- [2026-09-22: native official text template vs Jinja](2026-09-22-chat-template.md).
- [2026-09-22: external Vulkan reference compatibility smoke](2026-09-22-reference-bringup.md).
- [2026-09-22: native Unicode-9 NFC vs HF, two runs and comparison limits](2026-09-22-normalization.md).
- [2026-09-22: native Qwen text splitter vs HF, repeated exact-boundary comparisons](2026-09-22-tokenizer-split.md).
- [2026-09-22: complete native tokenizer vs HF and actual llama-server](2026-09-22-tokenizer.md) — historical unequal-boundary timings, not relative speed evidence.
- [2026-09-22: matched direct libllama comparison and two native optimizations](2026-09-22-tokenizer-matched.md) — reference-call audit, allocation control, paired binaries and retained regressions.

- [2026-09-22: Q4_1 real-tensor diagnostic decoding vs direct ggml](2026-09-22-q4_1.md).

- [2026-09-22: Q5_K actual tensor decoding, packed-field goldens and retained losses](2026-09-22-q5_k.md).

- [2026-09-22: Q6_K output-weight slices, signed-scale goldens and installed scalar-reference limits](2026-09-22-q6_k.md).

- [2026-09-22: native Vulkan driver transfers/dispatch vs matched C, with invalidated allocation-policy runs retained](2026-09-22-gpu-driver.md).

- [2026-09-22: native resident packed-weight matvec vs FP32/default ggml Vulkan](2026-09-22-gpu-matvec.md) — original baseline, full dense shapes, repeated losses and separate activation-precision control.
- [2026-09-22: matvec DFS optimization and paired source-rebuilt baseline](2026-09-22-matvec-optimization.md) — ~4.6× Q6 improvement, packed/aligned kernels, repeated full-shape gates and retained reference losses.
- [2026-09-22: matvec push toward ≥1.30× reference](2026-09-22-matvec-push.md) — GPU timestamps, raw-read probe, bit-identical kernel change; target partly met.

- [2026-09-22: native Qwen3.8 forward pass vs FP64/libllama oracle](2026-09-22-model-forward.md) — all gates pass, ~23 ms/step.
- [2026-09-22: native zerv vs llama-server serving](2026-09-22-serving.md) — matched greedy workload, two runs; prefill loses badly, short-context decode wins.
- [2026-09-22: batched FP32 prefill](2026-09-22-prefill.md) — oracle gates, two serving runs incl. a fully FP32 llama control, GEMM component and per-phase profile.
- [2026-09-22: split-K decode attention](2026-09-22-decode-attention.md) — decode step at 3.2K halved; accuracy study of summation orders; retained failed repeat.
- [2026-09-22: small-row prefill plans](2026-09-22-small-prefill.md) — GEMM tile crossovers, chunk policy, short-prompt TTFT below llama-server in the same runs.
- [2026-09-22: accumulation accuracy](2026-09-22-accumulation-accuracy.md) — per-op/per-layer error studies on real data, decode score fix, quality before/after, two serving runs.
- [2026-09-22: decode step overheads](2026-09-22-decode-overheads.md) — kernel-chain microbenchmark, bit-identity investigation, two serving runs, retained concurrent-run failure.
- [2026-09-22: scalar-X prefill GEMM](2026-09-22-gemm-throughput.md) — ISA/ablation/power research, bit-identical kernel replacement, sustained component and serving runs.

Each report must include:

1. Date, question/hypothesis, component or serving workload, and correctness status.
2. Hardware/OS/driver/toolchain, exact code/build/reference/model hashes, effective
   configuration, input shapes/dtypes/layouts, workload/input hashes and seeds.
3. Exact reproducible commands and checked-in harness revision, setup, warmup,
   repetition/trial duration, synchronization, cache conditions and background load.
4. Raw-results location, units, sample counts, dispersion/confidence intervals,
   median and tail results where meaningful, resource use, and correctness reports.
5. Matched reference comparisons, failed/unsupported cases, regressions, limits
   of the measurement, interpretation and next action. No fabricated estimates.

Put machine-readable manifests, stdout/stderr, raw per-trial timings, placement /
allocation logs and correctness reports under `data/YYYY-MM-DD-topic/`, linked
from the Markdown report. Derive tables from raw data rather than editing them
independently. Large profiler/tensor artifacts stay outside git with hashes,
locations and regeneration instructions. Never store secrets/private prompts.

Both levels are mandatory for performance-sensitive work:

- **Component:** repeatable operator/kernel, data movement, command submission,
  cache/state, allocation, tokenizer/template, sampling, scheduler and protocol
  benchmarks. Representative real shapes/dtypes/working-set sizes; an equivalent
  reference component where available; numerical checks before timing.
- **Serving:** always include a tuned **`llama-server`**, plus other credible
  high-performance servers that can execute the workload; compare the best tuned
  qualifying configurations. End-to-end evidence is required once the path exists.

Follow [performance.md](../performance.md) and
[verification.md](../specs/verification.md). A kernel-only win does not establish a
serving win; a serving aggregate does not explain which components are fast or slow.
