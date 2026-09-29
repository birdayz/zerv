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
83. [Multi-user latency, segmented prefill](bench/2026-09-25-multiuser.md) (18c.2): prefill in 4-layer segments
    with decode between (`--prefill-stall-ms`, `--prefill-order`), a 32-row f16 GEMM tile (`--f16-small-tile`,
    +5.3% on 128-row plans), `run_multiuser.py`; against vLLM and llama-server: zerv leads at 1–4 users, on TTFT
    and on long-prompt interference (40 tok/s kept against 5–13); vLLM leads 8-user throughput (158 vs 151) and
    steady gap p99 (51 vs 147 ms) by packing simultaneous prompts into one prefill.
86. [Packed multi-sequence prefill](bench/2026-09-26-packed-prefill.md) (18d.1): several prompts per chunk, each
    bitwise its solo prefill; `--prefill-pack`, `--f16-split`; 8-user 150.8 tok/s (vLLM 151.9), TTFT p95 1.05 s;
    a failure found by the gate (sub-128-row plans) fixed; steady gap p99 still behind vLLM.
85. [f16 decode kernel v2](bench/2026-09-26-decode-v2.md) (18e part 3): `gemm_f16d` bitwise v1, 1.3–1.5×
    faster; 8-row step 38.6 ms vs FP32 46.3; ablations show WMMA decode bounded near 600–700 GB/s on this card
    (the 16×16 tile's issue cost), so it cannot beat FP32 at 1 row; `zerv-decode-f16-bench`, batched-step profile.
84. [vLLM as a competitor](bench/2026-09-25-vllm.md): official ROCm image v0.30.0 with RedHatAI W4A16, FP8 KV;
    download verification and malware scan (`tools/fetch_hf.py`, ClamAV), hardened container, bring-up findings
    (a cold compile leaves no KV memory on 24 GB).
47. [Previous Ollama deployment](research/2026-09-23-previous-deployment.md) — Q4_K_M, KV q4_0, FA, 96–128k context.
87. [Unit economics of selling tokens](research/2026-09-26-token-economics.md) (research, no new measurements):
    Qwen3.8-27B OpenRouter prices ($1.80–4.40 output, median $2.55), traffic mix (~23:1 input, 71% cached),
    power/electricity/hardware costs, break-even 7–14% utilization at median/4-bit-floor prices but no route to
    market for one int4 8k-context endpoint; no rental market for the card; `tools/token_economics.py`.
88. [Rented GPUs with a faster engine](research/2026-09-26-rented-gpu-economics.md) (research, no new measurements):
    rental prices vs owning TCO (2.0–3.9×), InferenceX-fitted efficiency of today's engines, roofline model of
    the mean OpenRouter request; at today's prices renting is not the constraint (traffic is), at commodity prices
    a renter must beat owners' engines: MI355X needs ≥ 46% of roofline, MI300X 55%, B200 65% (corrected with Vultr's
    on-demand prices); AMD Developer Cloud and access programs; `tools/rented_gpu_economics.py`.
89. [Bigger open models on rented AMD GPUs](research/2026-09-26-bigger-models.md) (research, no new measurements):
    OpenRouter market per model (Kimi K3 / GLM-5.3 $4.4–4.8M per 30 days vs Qwen3.8-27B $0.28M), InferenceX MI355X
    economics (Kimi K3 5.4x rent, commodity MoEs ~1.7x), architecture fit (delta-rule hybrids), licenses, gated plan;
    `tools/model_market_summary.py`.
90. [AMD Instinct backend](research/2026-09-26-instinct-backend.md) (research + boundary proposal, awaiting approval):
    KFD ioctls + AQL queues + our own code objects from Zig, no ROCm userspace; LLVM assembler only as a dev tool
    (precedent: RDNA3 native gemm); MI300X/MI355X hardware facts (FNUZ vs OCP FP8, 64/160 KiB LDS, per-XCD L2);
    the KFD runtime can be built on the local RX 7900 XTX first. Pinned sources: [ledger](research/2026-09-26-instinct/sources.json).
91. [Big-model serving TODOs](research/2026-09-26-big-model-serving-todos.md): Qwen3.5-397B-A17B, Qwen3.8-2.4T-A95B and
    GLM-5.3-Flash on MI3xx (KDA, MLA+DSA, mHC, MoE, 4-bit quality gates, memory fits); no official Qwen 3.9 exists
    (`QwennAI/Qwen3.9-*` is a fake repository).
92. [Qwen3.8-27B image input (vision)](research/vision-qwen38.md) (block 19a, research first pass): HF vs llama.cpp
    semantics (encoder, erf vs tanh GELU in the merger, 2-D RoPE, interleaved M-RoPE positions, preprocessing:
    stretch vs pad, 64–16,384 vs 8–4,096 tokens), the verified `mmproj-BF16.gguf` (all 334 tensors bit-equal to
    the RedHat BF16 tower, `tools/vision_artifacts.py`), oracle design, cost estimates, open questions.
    Sources: [ledger](research/2026-09-26-vision/sources.json).
93. [Tests in parallel](bench/2026-09-26-test-parallelism.md): one test binary per file, concurrent rounds, golden SHA-256 in a
    ReleaseFast object, no TLS in the server test, stripped optimized tests; `zig build test` ~96 s → 9.4 s cold / ~4 s warm,
    `zig build check` (all required checks) 22.6 s cold; under load: a batcher cancel defect (fixed) and a test hang (fixed).
94. [Bazel as the only build tool](bench/2026-09-26-bazel-build.md) (merged 2026-09-26): `build.zig` removed, every harness
    builds through `tools/zerv_build.py`; the sandbox exposed undeclared test inputs; glibc 2.43 target; GPU driver as
    a test input; SPIR-V and native kernel code regenerated by Bazel and checked against the committed files (`zig
    clang` assembles the kernel byte-identically); GPU tests 41/41 in both modes and the spill gate (0 FAIL) under Bazel.
    Hermetic phases 3–5 ([spec](specs/hermetic-build.md)): GPU tests on a source-built Mesa RADV + Vulkan loader;
    fixtures regenerated with source-built oracles (llama.cpp captures byte-identical to the host package's); harnesses
    run no host program; llama-server built in the graph at the host package's speed within noise (A/B).
95. [KV admission and preemption in other engines](research/2026-09-27-kv-admission-preemption.md) (research, no new
    measurements): vLLM reserves the prompt and preempts the youngest by recompute; SGLang reserves prompt + a decaying
    0.7→0.1 share of the output and retracts by recompute; llama.cpp -kvu has no admission and fails every slot (observed
    exit at 93k). Nobody is bitwise exact across preemption; host swap is the exact option for zerv.
96. [Shared KV pool](bench/2026-09-26-shared-pool.md) (18d.2): no measurable cost against static KV (ABBA),
    one request may use the whole pool; default `--kv-pool shared`. 70–80k prompts with 6 users streaming against
    llama-server `-kvu` and vLLM (only zerv keeps the others streaming; llama prefills 70k 7% faster).
97. [Prompt admission with exact swap to host](bench/2026-09-27-kv-swap.md) (18d.3): `--kv-admit prompt`, bitwise
    exact swaps (batch-check and serving); clients without `max_tokens`: 109 tok/s, 16/16 finished, against reserve
    47 tok/s with 11/16 timeouts and vLLM ~21 tok/s unfinished after 2.7 h.
98. [Prefill stall sweep](bench/2026-09-27-prefill-stall.md): steady gap p99 vs TTFT for `--prefill-stall-ms`
    0/25/50/100. Stall 0: 8-user p99 65 ms (vLLM 51–55), TTFT p50 ~0.9 s (vLLM 1.07 s), but a 4.9k prompt under load 11.6 s
    (vLLM 4.9 s). No setting beats vLLM everywhere; default stays 100 ms.
99. [Tiered radix prefix cache](bench/2026-09-28-tiered-cache.md) (18d.5): `--prefix-cache-tier host`
    keeps evicted prefixes in host memory. 16 distinct 8.8k-token conversations: turn-1 TTFT p50 3.3–3.5 s,
    against 11.5–19.9 s with the tier off or the flat policy, and 26–30% less wall time. 192/192 outputs identical;
    GPU prefix gate 462/462 bitwise. Includes the hunt for the memory-pressure hangs and a restore renaming
    corruption: deep swap-out, slice wake, prompt victims, invariant checks.
100. [NVMe tier prerequisites](research/2026-09-28-nvme-tier.md) and
     [direct NVMe→VRAM P2P investigation](research/2026-09-28-nvme-p2p.md) (P2P parked):
     imported host buffers are still a two-transfer path. PCIe P2P is hardware-plausible;
     native AIS/KFD could retain filesystem I/O, but this driver lacks AIS. DMA-BUF/raw
     NVMe alternatives require a kernel helper and storage ownership. No P2P transfer
     demonstrated in that investigation; system/boundary changes await approval.
     The later RAM-staged archive is integrated (entry 106); it is not P2P.
101. [Imported host I/O buffers](bench/2026-09-28-nvme-buffers.md) (18d.6a complete):
     independent ABI/device gates and 24 exact disk/GPU trials; still RAM-staged,
     no transfer-overlap or serving claim.
102. [Bounded asynchronous disk transport](bench/2026-09-28-nvme-store.md) (18d.6b complete):
     independent syscall byte gates, bounded ownership and repeated component trials;
     equivalent-QD1 losses and write variance retained. Core uses generic alignment
     discovery/configuration; [filesystem preparation](deployment/nvme-scratch.md)
     is operator-owned. Later production integration: entry 106.
103. [Snapshot residency bookkeeping](bench/2026-09-28-snapshot-residency.md) (18d.6c first increment):
     independent snapshot/hot/disk identities and guarded pending transfers; 4,860 exact
     oracle transitions and 12 small disk snapshots through two resident slots. CPU
     79/79; two metadata benchmark runs. Existing serving cache is not yet switched over.
104. [Immutable disk prefix archive](bench/2026-09-28-prefix-archive.md): independent
     POSIX/hash fixture, bounded chunking/leases/drain, CPU 81/81, five measured
     CPU/disk trials. [Integration contract](specs/disk-prefix-cache.md).
105. [Model archive adapter](bench/2026-09-28-archive-model.md): poisoned/permuted
     state and full-vocabulary exactness at 257/80k; independent FP64/libllama gate,
     teardown fix, failed checker attempts retained.
106. [Production RAM-staged NVMe prefix archive](bench/2026-09-28-disk-prefix-serving.md):
     opt-in flags, bounded pending I/O and drained cancellation, 5.40 GB exact at
     80k, late-corruption cold fallback; 72 native HTTP turns identical. Three-round
     comparison: disk lowers reuse TTFT versus off but raises cold-write latency;
     host caching and tuned llama-server finish faster. Not P2P or restart-persistent.
107. [KV tiering papers: Mooncake, Pensieve, CachedAttention, LMCache](research/2026-09-28-kv-tier-papers.md)
     (research only): paper versus current SSD implementation, eager versus eviction-driven
     policies, bounded proactive copying/leases and scheduler-aware prefetch. The baseline
     full-image NVMe write-through is a poor measured fit; eager DRAM handoff is a different
     policy. Proposed replacement; subsequent execution-only changes are in entry 108.
108. [Asynchronous pressure-driven tiering plan](design/async-tiering.md): staged
     [GPU ownership prerequisite](research/2026-09-28-async-transfers.md) and archive
     submit/poll/drain implemented ([A report](bench/2026-09-28-async-ownership.md),
     [B report](bench/2026-09-28-async-archive.md)). 80k state/logits exact, 72 HTTP
     outputs identical; no serving speedup, long restore regression retained.
     Pressure policy/proactive preparation remain pending; write-through is unchanged.
109. [Radix cache source leases](bench/2026-09-28-cache-sources.md): 2,345 independent
     prefix-set transitions, both modes ×20, mixed host/GPU ancestor protection;
     24 bytes ownership metadata per hot slot. Pressure-triggered persistence and
     faster-or-on-par full-serving goal remain open.
110. [Canonical mixed cache-source capture](bench/2026-09-28-cache-source-bytes.md)
     (C.2 gates closed): independent 60-layout/4,804-window byte oracle, CPU 81/81,
     GPU/spill 3/3, host GPU 2/2, model oracle 337/337; 257/80k state/logits exact.
     Repeated component timings retain variance and the long-restore regression.
     Pressure admission and proactive prefetch are not integrated; no serving
     speedup claim. [C.3 specification](specs/tiering-pressure.md) precedes its code.
111. [Pressure-driven archive admission](bench/2026-09-28-tiering-pressure.md)
     (C.3 gates closed): checkpoint-triggered writes removed; bounded byte/slot
     policy, background drain, read priority and hard-pressure cancellation.
     Independent 2,048 decisions / 1,560 transitions; CPU 81/81, repeated interface
     tests, GPU/spill 3/3, host GPU 2/2, model oracle 337/337, exact 257/80k state.
     Three-round serving: 96 native responses/token counts exact; immediate-turn
     disk yields no restores and source retention reaches 29 s. Explicit idle
     host+disk gate restores 669 MB with eight exact responses. Native host+disk
     51.383 ± 4.020 s versus tuned RDNA3/HIP 49.783 ± 1.320 s; differing reference
     token streams, no disk speedup. RDNA3 fusion failures retained, supported
     nofusion follow-up passes. [D audit](design/tiering-preparation-audit.md)
     is active research, not code; the full plan/performance goal remains open.
112. [Archive progress between prefill units](bench/2026-09-28-tiering-progress.md)
     (D.0a evaluation closed, negative serving result): immutable-source-only hook,
     unchanged 1 MiB windows/read priority. Packed vocabulary/source bytes exact at
     257/80k; independent model 337/337, CPU 82/82, device gates pass. Adversarial
     shutdown test exposed/fixed uncompleted queued waiters. Matched component ~0.55%
     faster; 96 native HTTP responses/counts exact but zero disk restores, increased
     wasted writes, host+disk slower than host-only and tuned references. No serving
     win, proactive demotion or prefetch claim. D.0b window sizing is next.
113. [Bounded archive windows](bench/2026-09-28-tiering-window.md) (D.0b gates closed):
     1/2/4/8 MiB tickets, 8/16/32/64 MiB staging, two reserved read tickets.
     Independent POSIX/hash goldens, CPU 82/82, device gates and model oracle
     337/337 pass; every size passes packed/257/80k byte/vocabulary, cancellation
     and corruption gates. 120 native responses/counts exact; 8 MiB restores real
     disk state every round but still loses to host-only and tuned references
     (51.282 ± 1.534 s versus 47.476 ± 0.229 s / HIP 48.691 ± 0.985 s).
     Default unchanged; D.1 ownership research/specification is next.
114. [Preparation ownership](bench/2026-09-28-preparation-ownership.md) (D.1.1 component
     gates closed): bounded generation-qualified page-pool reservations, ack before
     release, atomic late-alias validation, live GPU masks preserved, copied versus
     freed accounting. Independent 1,362-case oracle, CPU 82/82, both page modes ×20,
     two negative controls and two metadata runs pass. No GPU adapter or serving
     speedup yet; D.1.2 cache/GPU integration research/specification is active.
115. [Preparation integration](bench/2026-09-29-preparation-integration.md) (D.1.2
     functionality/evaluation gates closed, performance goal open): qualified atomic cache finish, bounded GPU owner, packed-safe polling,
     host-only policy and opt-in knobs. Joint oracle 3,872 cases, initial CPU/device
     gates, exact 257/80k state/vocabulary/packed rows and independent 337/337 pass.
     Repetition found a swap-order livelock; deterministic negative control and fix
     recorded. Twelve final ownership/window/host/disk cases pass, including real
     concurrent reads. Clean serving: 96 exact native responses, disk still slower
     than references, no matched-quality parity. D.2 research active, not implemented.
116. [Queued-demand audit](design/queued-demand-audit.md) and
     [D.2 contract draft](specs/queued-demand.md): current active research. Separates
     evictable demand preference from immutable source leases; specifies bounded
     staging-window prefetch, not whole-image fetching. Token callback lifetime,
     request generation, record pin and foreground ticket reserve must be verified
     by an independent event fixture before native implementation. Not implemented.

- [Queued-demand implementation plan](design/queued-demand-implementation.md) — paper-derived ordered D.2 work and acceptance gates.
- [D.2 intermediate verification](bench/2026-09-29-queued-demand.md) — cache/scheduler tests, strengthened negative controls, window1/2 × 257/80k exact model checks and independent numerical oracle; serving/performance gates still open.
- [Fixed-history competitive control](bench/2026-09-29-fixed-history.md) — identical serialized requests/prompt counts, preserved native outputs, remaining competitor divergence and HIP reuse-TTFT loss; not quality-equivalent parity.
- Fixed-history report follow-up: full prompt/token-ID equivalence is now verified against both competitors; the reuse component profile isolates two costly small-prefill segments. Performance acceptance remains open.
- [Short-reuse checkpoint policy](bench/2026-09-29-reuse-join.md) — paper-motivated storage/recomputation tradeoff, controlled counterfactual, opt-in implementation, independent oracle and production tests; short-reuse gains and later-turn regressions retained. Competitive quality acceptance remains open.
- [D.2 held-out quality and pressure ablation](bench/2026-09-29-demand-quality.md) — preregistered strict gate fails equally on all engines (39/45); all180 output/count/finish signatures and12 prompt-token arrays match. More KV slack reduces later cold recomputation at a measured VRAM cost; C1 regression retained.
