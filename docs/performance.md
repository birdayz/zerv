# Performance research and measurement protocol

**No inference benchmarks run yet.** A first CPU component experiment is recorded
in [the dated quant report](bench/2026-09-22-quant-decode.md). The performance objective
is to beat the best compatible reference, not Python overhead or a weak baseline.

Every serving comparison must include tuned **`llama-server`**, plus other servers
claiming strong relevant performance that can execute the workload. Every
performance-sensitive component also needs its own repeatable benchmark; neither
component nor end-to-end evidence substitutes for the other.
Follow the mandatory [correctness gate](specs/verification.md) first.

## Find the strongest reference, then keep looking

| Candidate | Initial evidence | Required action |
| --- | --- | --- |
| llama.cpp Vulkan | Installed 0.4.1 enumerates this card; inspected source includes Qwen3.5/hybrid and Vulkan DeltaNet support | Verify exact Qwen3.8 artifact/semantics; benchmark installed build, then a pinned current optimized build |
| llama.cpp HIP | Upstream documents `GGML_HIP` and gfx1100; HIP not installed here | Evaluate a compatible toolchain and tuned build as an external competitor; don't assume slower/faster |
| vLLM / SGLang / TokenSpeed | Official Qwen3.8 card links deployment recipes | Check exact gfx1100 support, quant format, hybrid kernels, and 24 GB fit; run qualifying candidates, record exclusions |
| HyperQwen (patched vLLM, github.com/syv-ai/HyperQwen @ `1cf86656`, pinned in `third_party/research-serving/`; **NVIDIA-only in practice**: int8 Marlin GEMMs tuned on sm86, int8-QK prefill kernel; cannot run on this card, so it is compared by its published RTX 3090 numbers, always labeled as another GPU) | Own README (RTX 3090 at 250 W, own ~20 GB requantization, vLLM 0.27.1 measurements): 45–46 tok/s plain single stream, 111/120 tok/s with MTP (sampled/greedy), 122/131 with the DFlash2 drafter, 381 tok/s when the answer quotes the prompt, ~1,035 tok/s aggregate at 64 concurrent, prefill ~1,440 tok/s at 1k; 150k context with fp8 KV | NVIDIA only, not runnable here. Run its prompt protocol (`bench/prompts_real.jsonl`, 1,024-token answers, C1, default and greedy sampling) against zerv; different quantization, so not an equal-quality comparison. Its techniques (batched decode, prompt-lookup drafting, 8-bit KV) are queued in TODO |
| **Tracked set (user decision 2026-09-26): llama-server, vLLM, HyperQwen, llama.cpp-RDNA3-7900xtx-opt** | | Every serving claim is checked against all four (HyperQwen by its published numbers, see its row) |
| llama.cpp-RDNA3-7900xtx-opt (github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt @ `15995a12`, MIT, pinned in `third_party/research-serving/`) | README (2026-09-09): llama.cpp HIP build for gfx1100 with RDNA3 tuning of flash attention, MMQ/MMVQ and GDN (chunked gated-delta-net prefill), fused kernels, adaptive MTP draft depth; tested on one and two 7900 XTX with Qwen3.8-27B Q8_0 + MTP; reports ~1,600 tok/s prefill with two cards (tensor parallel), no single-card numbers | Build with HIP for gfx1100 (inside the pinned vLLM ROCm image: host has no HIP toolchain), run on our Q4_0 GGUF, with and without adaptive MTP, in `run_serving.py` / `run_multiuser.py` |
| ik_llama.cpp | Current README says only CPU and CUDA are fully functional/performant; warns against expecting ROCm/Vulkan support | Not established as a Radeon baseline; monitor/recheck, do not equate CUDA results with this card |

Model-family support, successful device enumeration, and the presence of a kernel
in source each fall short of a successful correctness-checked run. No reference
has yet earned the title "best" on this machine. Document candidates tried,
settings, failures, and why any were excluded; revisit before broad speed claims.
Latest upstream llama.cpp observed: `ec5a12b85ae32fbccfa4276051382330a8e6458b`.
Installed reference: `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`; its benchmark
startup warns that assertions are enabled. Do not use only that build to claim a win.

## Fair comparison contracts

Two separate comparisons:

- **Same artifact:** identical weight file/hash, tokenizer/template, token IDs,
  cache precision, context, output length, sampling and speculation settings.
  Isolates implementation/runtime performance.
- **Best quality-matched deployment:** allow each engine its strongest supported
  formats/optimizations, but enforce the same declared quality floor, usable
  context, memory cap, latency constraints, and request workload. Report artifact
  and quality differences explicitly. Same bit-width label is not equal quality.

Report first-token latency and first-*visible-answer* latency separately for
thinking models. Count all generated reasoning tokens as work and resource use.
Don't win by disabling thinking, shrinking context, suppressing EOS in only one
engine, dropping prompts, or benchmarking a shorter generation.

## Workload matrix

Start with fixed hashed input-token fixtures; later add representative natural
language/code/long-context chat traces. Keep held-out quality cases distinct from
tuning cases. Sweep incrementally, not a huge blind Cartesian product.

- Prompt lengths: 128, 512, 2K, 8K, 32K; add 64K+ only when budget and correctness permit.
- Generation lengths: 128 and 512; mixed realistic lengths for service tests.
- Concurrency: 1, 2, 4, 8, up to actual admitted capacity. Explicit OOM/overload
  outcomes count; unsupported cells must not disappear from the report.
- Profiles: low-latency, throughput under a declared latency SLO, low-VRAM under
  a declared quality floor. Set numeric SLO/quality thresholds before comparisons.
- Cold load separately from warm execution. Prefix-cache cold/hot separately.
- Long-context decode uses genuinely populated KV/recurrent state, not only a
  large allocated context. Include mixed short/long requests and cancellation.

Capture:

- Load time, warmup/compile time, tokenizer/template time, queue time.
- Prefill tokens/s, decode tokens/s per request, total output tokens/s.
- TTFT, inter-token latency, end-to-end latency: p50/p95/p99 where sample count supports it.
- Peak/steady VRAM, host RSS/pinned memory, total token/state/checkpoint capacity.
- GPU kernel time, dispatch/submit gaps, host/device transfer bytes, sync counts,
  memory bandwidth, occupancy/register/LDS indicators where tools expose them.
- Requests completed/failed/timed out, acceptance rate for speculation, fairness.

Use enough warmup to exclude shader compilation and stabilize clocks. At least
five repeated steady-state trials for initial engine tests; longer service trials
for tail percentiles. Publish sample counts, variation/confidence intervals, and
interleaved baseline/candidate run order. Small noisy differences are not wins.
Use open-loop offered-load service tests to expose queueing/tail behavior, plus
closed-loop concurrency tests for capacity; don't hide coordinated omission.

## Component benchmark obligations

Maintain versioned cases at meaningful package interfaces: quant/linear algebra,
attention and recurrence, norm/activation/reductions, transfers/dispatch/sync,
allocation and cache/state/snapshot operations, tokenization/template/sampling,
scheduling/admission, and HTTP/SSE serialization/backpressure. Include relevant
shapes, dtypes, alignments/tails, batch/context lengths, working-set sizes and
concurrency. Always validate results before timing; compare equivalent reference
operations when available, disclosing validation/API-overhead differences.

For GPU work, use device timestamps with required completion synchronization and
measure submit-to-completion separately. CPU timers around asynchronous dispatch
alone measure launch overhead, not kernel time. Exclude allocation/compile/setup
only when that is the explicit case; separately benchmark cold/setup-sensitive paths.
Prevent dead-code elimination, measure enough work above timer resolution, and
record warm/cold cache policy. Pair component changes with serving reruns when the
full path exists. Report **dated Markdown files under `docs/bench/`**, with exact
commands and linked manifests/raw trials under `docs/bench/data/`.

## Reproducible external baseline seed

These commands are **recipes, not executed model benchmarks**. Flag syntax was
checked against installed `--help`. Explicit artifact path only; no auto-download.
Before running, verify the hash, memory availability, template, and semantic gate.

```sh
MODEL=/absolute/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
llama-server --version
llama-server --list-devices
llama-bench -m "$MODEL" -dev Vulkan0 -ngl 99 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 128 -t 12 \
  -p 512 -n 128 -r 5 -o json > bench.json 2> bench.stderr
```

`llama-bench -p ... -n ...` produces separate prompt/generation measurements; do
not present that as long-context end-to-end service throughput. Inspect `-pg` and
`-d` for paired/depth cases at the pinned version, then validate what work occurs.

```sh
llama-server -m "$MODEL" --device Vulkan0 --gpu-layers all --fit off \
  --ctx-size 8192 --parallel 1 --batch-size 512 --ubatch-size 128 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --ctx-checkpoints 0 --cache-ram 0 --spec-type none \
  --jinja --host 127.0.0.1 --port 8081
```

This is a controlled starting point, **not a tuned winner**. Preserve startup logs,
actual GPU placement, resolved params, token counts, and observed memory. Then tune
FA on/off, batch/ubatch, threads, quant choices, cache precision, concurrency,
prefix reuse, and speculation within the comparison contract. Disabling caches /
speculation in the starting point does not justify omitting a stronger tuned run.
The target engine never shells out to these commands in production.

## Optimization loop

1. Understand the operation completely; specify semantics and build oracle cases.
2. Obtain a correct simple implementation and an end-to-end baseline.
3. Profile the real workload. Rank costs by total time, not interestingness.
4. Predict the gain and identify the bottleneck: bandwidth, arithmetic, occupancy,
   launch latency, synchronization, transfers, scheduling, or memory capacity.
5. Implement one measured change; run numerical/state/protocol regressions.
6. Repeat matched end-to-end benchmarks; retain or reject based on evidence.
7. Store raw data and explanation in [bench/](bench/README.md).

Batch-one dense decode is *likely* weight-bandwidth-limited at short contexts;
prefill/batched decode can be compute/launch-bound, long-context decode KV-bound.
Use a roofline based on **measured sustainable device bandwidth** and actual
bytes/operations, not advertised TFLOPS. `tokens/s ≤ bandwidth / bytes_per_token`
is an idealized bound, not a performance forecast; MTP/batching/cache reuse change
the amount of work per accepted token. GPU utilization alone cannot diagnose this.

Research candidates: quant-layout repacking, fused dequant/matvec, shape-specific
matmul, fused norm/gates/RoPE, efficient DeltaNet chunking, command reuse, reduced
synchronization, GPU sampling, scheduling and state/cache budgets, then MTP with
exact rollback. Don't assume a general-purpose baseline has already hit the hardware
limit, or that a handwritten kernel is faster than a tuned reference.
