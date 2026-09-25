# Project knowledge

Initial investigation: **2026-09-22**, on the target machine. Everything here is
research or a proposed implementation plan unless explicitly marked observed.
Native Q4_0/Q8_0/Q4_1/Q5_K/Q6_K CPU decoding passes independent reference goldens in Debug and
ReleaseFast, with repeatable CPU component comparisons. Native GGUF loading now
passes full-artifact independent comparison; the model download is SHA-verified.
Native official text-only chat rendering passes 100 Jinja cases. Unicode-9 NFC
passes normative/exhaustive goldens and two component benchmark runs. Native Qwen
text splitting passes 47,919 independent cases; complete BPE encoding/raw decoding
passes 19,033 encode cases, 70 decode cases and all 248,320 token pieces on the actual
GGUF, with two benchmark runs and actual llama-server comparison. An external Vulkan reference served a real request,
and native Vulkan memory/transfers/compute dispatch now pass independent hardware
checks and repeated matched driver benchmarks. Native packed-weight GPU matvec also
passes independent fixtures and every full base-model dense shape, with repeated
measurements retaining losses against the reference. Native Qwen3.8 execution passes FP64/libllama oracle gates and is served over
`POST /v1/chat/completions` with batched FP32 prefill and split-K decode attention (oracle
gates pass). Decode is faster than llama-server at every measured length and TTFT is lower up
to ~100 prompt tokens; prompts of ~0.8–3.2K tokens are still 1.7–2.3× slower than
llama-server's default Q8_1 prompt path (and faster than its fully FP32 path).

## Start here

[Controlled work queue](../TODO.md): one active building block/package at a time;
finish independent tests and benchmark gates before advancing.

1. [Functional specifications](specs/README.md) — requirements and behavior contracts.
1a. [Speed program design and decision log](design/speed.md) — plan, measured breakdown, open decisions.
2. [Target machine](hardware.md) — observed hardware/software and tooling gaps.
3. [Qwen3.8-27B](model.md) — exact identity, architecture, artifacts, compatibility risks.
4. [Memory budget](memory.md) — weight, KV, recurrent state, and workspace accounting.
5. [Architecture](architecture.md) — general-purpose Zig engine; scratch-built fast paths.
6. [Verification specification](specs/verification.md) — mandatory reference comparisons and gates.
7. [Performance](performance.md) — competitor selection, measurement, optimization work.
8. [Roadmap](roadmap.md) — executable milestones and exit conditions.
9. [Development](development.md) — pinned toolchain, tested commands, dependency boundaries.
10. [Research source storage](research-code.md) — stable gitignored `third_party/` paths.
11. [Quant-block research](research/quant-blocks.md) — completed scoped research and external oracle.
12. [Quant verification results](research/2026-09-22-quant-validation.md) — actual tests, not inference speeds.
13. [Quant component benchmark](bench/2026-09-22-quant-decode.md) — dated results, losses, SIMD experiment and repeated reference comparisons.
14. [GGUF validation](research/2026-09-22-gguf-validation.md) — complete artifact hash, native loading, independent inventory/payload comparison.
15. [GGUF component benchmark](bench/2026-09-22-gguf-loading.md) — actual model and tiny-file comparisons, ownership caveats.
16. [Tokenizer/template research](research/tokenizer.md) — normalization/template differences and completed text-rendering scope.
17. [Chat-template benchmark](bench/2026-09-22-chat-template.md) — exact official fixture outputs, specialized native vs Jinja.
18. [External reference bring-up](bench/2026-09-22-reference-bringup.md) — actual Vulkan request, explicitly not zerv serving.
19. [Unicode-9 NFC](research/normalization.md) — resolved backend version, normative data and native correctness gates.
20. [NFC component benchmark](bench/2026-09-22-normalization.md) — repeated native/HF timings, initial marks loss and comparison limits.
21. [Qwen text splitting](bench/2026-09-22-tokenizer-split.md) — exhaustive properties/independent boundaries and repeated HF comparisons, not yet BPE.
22. [BPE vocabulary/decode research](research/tokenizer-bpe.md) — completed vocabulary/merge equality, added-token semantics, raw decoding and corrected forward-rank dependencies.
23. [Complete tokenizer benchmark](bench/2026-09-22-tokenizer.md) — actual llama-server correctness and NFC reconciliation; unequal-boundary timing table corrected.
24. [Matched tokenizer benchmark and optimization](bench/2026-09-22-tokenizer-matched.md) — audited direct libllama use, allocation control, paired baselines, expanded cases and retained losses. Replay verified; tokenizer detour closed and native serving work resumed.

25. [Q4_1 decoding](bench/2026-09-22-q4_1.md) — independent exhaustive/actual-tensor checks, matched direct component timings and retained full-tensor repeat loss.

26. [Q5_K decoding](bench/2026-09-22-q5_k.md) — packed-scale/global-field goldens, actual tensors, matched timings and current Q4_1 performance regression.

27. [Q6_K decoding](bench/2026-09-22-q6_k.md) — trailing half/signed subscales, independent finite-domain goldens and matched bounded output-weight slices. All artifact packed types covered; GPU execution still separate.

28. [Native Vulkan driver](bench/2026-09-22-gpu-driver.md) — real transfers/dispatch, bounded ownership, independent ABI/scalar checks, corrected memory-feature policy and repeated matched measurements. Separate from GPU model operators.

29. [Native GPU matvec baseline](bench/2026-09-22-gpu-matvec.md) — independent CPU/GPU goldens, all full dense shapes including the vocabulary projection, byte-identical replay and repeated timings. Original native implementation loses every shape; retained as the DFS counterfactual.

30. [GPU normalization/gate research](research/gpu-primitives.md) — paused block09: converted norm/A parameters, correct DeltaNet epsilon placement, source-level gate mapping and V-head permutation. Executable intermediate gate and native operators still pending.

31. [Matvec optimization DFS](research/matvec-optimization.md) — block08b: compiler inspection, exact hardware-half ablation, packed traversal/scheduling/FMA experiments, retained regressions and source-rebuildable baseline. [Paired results](bench/2026-09-22-matvec-optimization.md). Serving remains paused.

32. [Matvec ≥1.30× push](bench/2026-09-22-matvec-push.md) — block08c: GPU timestamps, raw-read limits, bit-identical exact-FMA decode; target met on 5/11 shapes (GPU time), not uniformly.

## Evidence

- [Source ledger](research/2026-09-22/sources.json): exact URLs, revisions, and
  hashes of the locally retained model metadata.
- [Hardware capture](research/2026-09-22/hardware.txt): commands and outputs.
- [Official model config](research/2026-09-22/model-config.json).
- [Official generation config](research/2026-09-22/generation-config.json).
- [Quantization artifact metadata](research/2026-09-22/quant-artifacts.json): sizes
  and publisher-provided SHA-256 values. Only the selected Q4_0 artifact is now
  locally SHA-verified; other listed weight files are not downloaded.
- [Benchmark record requirements](bench/README.md).

## Documentation conventions

- `docs/specs/` contains specifications of functionality; research and design
  evidence belong elsewhere under `docs/` and are linked from those specifications.
- Complete the research and reference-comparison plan for a feature **before
  coding it**. Initial project research here is not a claim of full implementation
  readiness. Unresolved semantics block the affected implementation.

- **Observed:** command run here, with evidence. **Source:** upstream documentation
  or code inspected at a recorded revision. **Estimate:** formula and assumptions.
  **Proposed:** decision to test, not a claimed capability.
- Write benchmark reports as `docs/bench/YYYY-MM-DD-topic.md`; put reproducible
  experiment manifests/raw data under `docs/bench/data/<run-id>/`.
  Keep large model weights, build products, and profiler binaries outside docs;
  record their locations/hashes instead. Redact secrets and private prompt content.
- Record failed hypotheses, incompatible references, and unavailable tools too.
- Keep detailed knowledge here rather than growing the root README or AGENTS.md.

33. [Qwen3.8 execution semantics](research/qwen35-execution.md) and [native forward-pass gates](bench/2026-09-22-model-forward.md) — FP64 + libllama oracle, all declared tolerances met, greedy 269/269.
34. [Native serving vs llama-server](bench/2026-09-22-serving.md) — working Chat Completions server; decode competitive at short context, prefill 3–32× slower; retained invalid/failed runs.
35. [Batched FP32 prefill](bench/2026-09-22-prefill.md) ([spec](specs/prefill.md)) — oracle gates at chunks 1/13/512, TTFT 7–10× better; GEMM component and per-phase GPU profile.
36. [Split-K decode attention](bench/2026-09-22-decode-attention.md) ([spec](specs/decode-attention.md)) — decode at 3.2K 45.5 → 21.9 ms/step; summation-order accuracy study on real tensors.
37. [Small-row prefill plans](bench/2026-09-22-small-prefill.md) — 23-token TTFT 450 → 168 ms; GEMM tile crossovers.
38. [Scalar-X prefill GEMM](bench/2026-09-22-gemm-throughput.md) — LDS-bound diagnosis (ISA, ablations, power/data dependence); bit-identical 1.4–1.6× faster GEMM; TTFT below llama-server up to ~100 tokens.
39. [Decode step overheads](bench/2026-09-22-decode-overheads.md) — norm descriptor-reload and DeltaNet latency fixes, bit-identical; decode 18–22% faster than llama-server; unroll/fma-fusion hazard documented.
40. [Accumulation accuracy](bench/2026-09-22-accumulation-accuracy.md) — decode vs prefill error localization, FP32 emulation studies; two-level decode scores bring decode logits error to llama.cpp's level.
41. [Serving open items](bench/2026-09-22-serving-open-items.md) ([output parsing research](research/output-parsing.md)).
    - Graceful drain on SIGINT/SIGTERM, and real-socket tests for overload and
      disconnect.
    - Serving RSS falls from 15.5 GB to 60 MB.
    - Output text follows llama-server's parser: all 12 parity cases equal, and a
      negative control shows the check catches the old behavior.
42. [Cooperative-matrix research](bench/2026-09-23-coopmat-research.md) ([research](research/coopmat-prefill.md)).
    - RDNA3's f16 WMMA accumulation is not IEEE; its s8 WMMA is exact.
    - Peaks: WMMA 135 T, FP32 VALU 63–66 TFLOP/s.
    - No FP32-grade WMMA scheme beats the VALU, and the shipped GEMM runs at ~35% of
      its peak.
43. [Prefill GEMM efficiency](bench/2026-09-23-gemm-efficiency.md).
    - Ablations and ISA found three costs: SGPR spills, `lgkmcnt`-serialized X
      loads, and unaligned block reads.
    - The fix is bit-identical and 1.3–1.4× faster.
    - A 256×64 tile for 512-row chunks follows.
    - TTFT at 3223 tokens: 7.76 → 5.34 s.
    - A new 562-token oracle case gates 512-row chunks.
44. [GEMM round 2](bench/2026-09-23-gemm-round2.md) (research; no change).
    - Ablations of the wide kernel show LDS reads as the largest cost.
    - Compiler scheduling exposes SMEM latency, and A-read pipelining did not help.
45. [DeltaNet scan latency](bench/2026-09-23-delta-scan.md) — bit-identical; delta phase 62 → 45 ms per chunk.
46. [llama-server precision ladder](bench/2026-09-23-llama-precision.md).
    - llama's default prompt path here is f16 WMMA with f16 accumulation, not Q8_1
      (this corrects earlier reports).
    - At matched FP32, zerv is 1.9–3.8× faster.
48. [Explicit f16 prefill mode](bench/2026-09-24-f16-prefill.md) (block 14, open).
    - Opt-in WMMA prompt projections.
    - vs llama default: 2.1× faster at 23–81 tokens, 1.29× faster at 836, 4% slower
      at 3223.
    - More accurate than llama's matched config on average; worst-token gate unmet.
49. [Long context at the current cap](bench/2026-09-24-long-context.md).
    - 29,000-token prefill: 46 s (f16) / 63 s (fp32).
    - Decode at 29k: 38.7 tok/s.
    - 90k needs a lower-precision KV cache; estimated 4–5 min of prefill.
    - Serving at 29k vs llama-server: llama's default prefill is 1.40× faster than zerv
      f16 (flash attention); zerv is 1.55× faster at matched FP32. Decode: zerv 38.5,
      llama 37.0 tok/s.
    - KV cache at 90k: 11.8 GB in FP32, 5.9 GB in f16. Prefix caching is proposed.
50. [Tool calling](bench/2026-09-24-tool-calling.md) (block 12c).
    - 22/22 greedy cases equal to llama-server (text, calls, arguments bytes, finish, token counts).
    - Prompts byte-equal to an independent Jinja oracle; llama.cpp's float printing differs.
    - bruh `-p openai-compat --only bash` completes multi-step tasks against zerv.
51. [Serving robustness fixes](bench/2026-09-24-serving-fixes.md) — exclusive port, free-VRAM check,
    device-lost shutdown, prefix-cache shutdown panic; after a two-server GPU incident.
52. [f16 GEMM lab](bench/2026-09-24-gemm-f16-lab.md) (block 16b, rounds 1–2):
    - ablations, wave32, direct f16 X, BK=64 (bitwise-equal, 1.25–1.36× the block-14
      kernel in an interleaved race);
    - RADV/ACO constraints (source-backed), the 110 °C thermal limit, and negative
      results.
53. [f16 GEMM integrated for Q4_0](bench/2026-09-24-gemm-f16x.md) (block 16b):
    - logits, captures and served outputs bitwise-identical;
    - prefill GPU time −18%;
    - 3223-token TTFT 3.04 s against llama-server's 3.39 s; 12k: 12.9 s against 12.46 s.
55. [KV buffers](bench/2026-09-24-kv-buffers.md): the KV cache leaves the 4 GiB state
    buffer; bitwise identical; context bound by free VRAM (38.7–49.4k), 37.8k-token prompt
    answered correctly.
54. [Speculative verification ≡ decode](bench/2026-09-24-spec-verify.md) (block 17b.1):
    - N-row verify logits and committed state bitwise equal to N decode steps;
    - exact-count multi-row matvec modules: verify + commit of 3 / 5 rows in 24.1 / 32.4 ms
      against a 20.0 ms step (from 34.1 / 46.3).
56. [Fused prefill attention, batched MTP catch-up, verify grid order](bench/2026-09-24-flash-attention.md)
    (block 16a, inside 17): TTFT at 37.8k tokens 46.0 s against llama's 46.6 s; speculative
    decode at 38k 80.8 against llama MTP 3's 76.3 tok/s; all outputs identical.
57. [Decode baseline](bench/2026-09-24-decode-baseline.md) (block 17a): per-phase decode
    profile, per-projection read rates (805 GB/s against a 920 GB/s roofline), competitor.
58. [MTP speculative decoding](bench/2026-09-24-speculative.md) (block 17b.3): lossless in
    every run; acceptance equal to llama.cpp's where outputs agree; faster than llama's best
    on every decode-v1, serving-v2 and 38k case. Default since: 3 drafts, adaptive.
59. [Host-resident embedding and snapshots](bench/2026-09-24-host-memory.md): embedding in
    system RAM at no measured cost (682 MiB VRAM; now default); host snapshots cost 10–19 ms
    of TTFT for 1.2 GB (knob).
60. [`--context max`](bench/2026-09-24-context-max.md): the largest context per memory
    configuration, 39.8k–60.1k with a 1 GiB reserve, up to 67.8k without; all served.
61. [KV cache precision](bench/2026-09-24-kv-precision.md) (`--kv-type f16`, block 17c): context
    ×2 (88.6k default), 38k decode 39.5 tok/s against llama's 35.8, KL to f32 about 900× below
    llama's f16 KV; f32 stays the default (worst-token gate clause unmet).
    [Research](research/kv-precision.md): layout, value ranges, llama.cpp's options, q8 notes.
62. [Decode attention at long context](bench/2026-09-24-decode-attention-long.md): global-max and
    parallel-combine passes, bitwise identical; 64k decode step 30.64 → 26.97 ms (+0.07 ms at
    short context).
63. [FMA matvec accumulation](bench/2026-09-24-fma-matvec.md): decode closer to FP64 on every oracle
    case; with a re-tuned table, verify+commit of 3 / 4 / 5 rows −6 / −8 / −12%.
    `--matvec-accumulation separate` keeps the previous arithmetic (byte-identical).
64. [Token sampler](bench/2026-09-24-sampler.md): component benchmark; top-k off sorted the whole
    vocabulary (29 ms per token), now a nucleus prefix (1.2 ms top-p, 0.18 ms min-p), same draws.
    `--sampler-order sorted` keeps the previous order (golden-tested against its source).
65. [Decode phase fusion](bench/2026-09-24-decode-fusion.md): 1.7 ms (8%) of each decode step is
    dependency cost; gate+up+swiglu fused (`--decode-fusion`), plain decode +1.7%, bitwise identical.
66. [MTP draft cost and draft vocabulary](bench/2026-09-24-draft-vocab.md): drafting is 16% of a
    3-draft cycle; `--spec-draft-vocab 65536` gives +7–9% on English/code, but regresses
    multilingual text (Chinese below plain decode), so it is opt-in. Outputs unchanged.
67. [Verify FFN fusion](bench/2026-09-24-verify-fusion.md): gate+up+swiglu in one multi-row dispatch
    (`--verify-fusion`), verify −1 to −2%, speculative decode +0.4–1.2%, bitwise identical.
68. [RCA: ACO LDS-spill miscompile](bench/2026-09-24-aco-lds-spill.md): wrong K-quant multi-row
    results traced to Mesa 26.2.3's pre-RA scheduler reordering LDS spill slots; 5 whys; spill
    gate `tools/check_shader_spills.py`.
69. [Host overhead](bench/2026-09-24-host-overhead.md): per-request decode time split (backend
    calls / sampling / token handling, logged by the server); host work below 1% of decode.
70. [DeltaNet scratch spill](bench/2026-09-24-delta-spill.md): 120 store addresses kept live across
    the row loop; `--delta-state-out` removes the spill, byte-identical, decode +1.6%, TTFT −0.4–1%.
71. [Own RDNA3 code through Vulkan](research/native-isa-via-vulkan.md) (approved for experiments, D7): `VK_KHR_pipeline_binary`
    lets RADV run our own machine code; RADV's LLVM backend disables cooperative matrices.
72. [Hand-scheduled `gemm_f16x` machine code](bench/2026-09-24-gemm-f16x-isa.md): generated RDNA3
    kernel, bit-identical (23 configs, 91 M outputs); 1.20–1.23× the SPIR-V kernel at 512 rows,
    1.22–1.40× at 3,328 rows (component). Found: the SPIR-V epilogue store burst (15%), the
    host-memory `io` read at dispatch start (15 µs per dispatch). Integrated as
    `--gemm-code native` (default): outputs identical, f16 TTFT −12% (3223 tokens), −10% (12k).
73. [Tensile GEMM techniques](research/tensile-gemm-techniques.md): AMD's assembly GEMM generator,
    read as research; not usable as a component (dense only, ROCm ABI, other k-order); six
    candidate techniques for our generator, two of our findings confirmed (store remap).
74. [HyperQwen's single-user protocol](bench/2026-09-24-hyperqwen-protocol.md): zerv 91.3 / 79.5 tok/s
    (greedy / sampled, 3 drafts) vs llama-server MTP 68.2 / 59.8; HyperQwen (3090, README) 120 / 111.
    Our speculative cycle is +46% over a plain step (theirs +12%): draft cost and acceptance.
75. [Concurrent sequences](research/concurrent-sequences.md) (block 18a, research in progress): where zerv
    assumes one sequence, llama.cpp's batching (per-slot drafts, chunked prefill, one recurrent state cell
    per sequence), bruh's actual concurrency, memory model; six open items before the spec.
76. [Concurrency baseline](bench/2026-09-24-concurrency-baseline.md): llama-server `-np 8` 40 → 156 tok/s
    aggregate at 1 → 8 clients (MTP 3: 62 → 117); zerv flat at 50 / 87 (3 drafts), TTFT up to 72 s queued.
77. [Spec: concurrent sequences](specs/concurrent.md) (block 18): `--parallel N`, `--decode-precision f32|f16`,
    paged KV pool, batch invariance as the exactness contract, stages 18b–18e with gates.
78. [Paged KV addressing, single sequence](bench/2026-09-24-paged-kv.md) (18b.1): KV pages behind a page
    table; `--kv-page-tokens N|context` (specialization constant; `context` = the old layout); every gate
    byte-identical at 128/256/context; default 128: +0.06% / +0.07% decode (f32 / f16 KV) at 30k, −0.3% at 4k;
    a silently passing ReleaseFast test found and fixed; an older f16-KV baseline bisected to the FMA change.
79. [Slots and batched decode](bench/2026-09-25-batched-decode.md) (18b.2): per-slot state and page tables,
    `decodeBatch` with per-row slot entries; `zerv-batch-check` 399/399 rows bitwise equal to solo decoding;
    single-sequence path byte-identical; model-level 158.6 tok/s at 4 rows, 172.8 at 8 (49.8 at 1).
80. [`--parallel N` serving](bench/2026-09-25-parallel-serving.md) (18c): continuous batching scheduler; 60/60
    concurrent responses byte-identical to solo; 49.4 / 91.4 / 147.6 / 161.8 tok/s at 1/2/4/8 clients against
    llama-server's 39.8 / 65.2 / 99.1 / 143.0 (same session); TTFT p50 0.53 against 1.38 s at 8; llama-server
    gives 3–5 different greedy outputs per prompt across concurrency levels.
81. [Exact FP32 batched projection beyond 4 rows](bench/2026-09-25-fp32-batched-projection.md) (18e part 1,
    negative): component splitting and row groups stay bitwise exact but gain at most 5%; the multi-row kernel is
    stall-bound at ~5.2 TFMA/s from 4 rows on; the WMMA f16 decode mode comes next.
82. [`--decode-precision f16`, kernel v1](bench/2026-09-25-f16-decode-mode.md) (18e part 2): WMMA batched decode
    projections, batch-invariant within the mode (399/399), flat in rows (45.4 → 50.0 ms, 1 → 8 rows) but not yet
    faster than FP32; kernel v2 design recorded.
47. [Previous Ollama deployment](research/2026-09-23-previous-deployment.md) — Q4_K_M, KV q4_0, FA, 96–128k context.
