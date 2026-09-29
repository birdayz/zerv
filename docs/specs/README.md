# Functional specifications

This directory is the source of truth for **what zerv must do**. Research, design
alternatives, and measurements live elsewhere under `docs/` and are linked here.

- [Requirements](requirements.md): user requirements and project boundaries.
- [Serving and configuration](serving.md): proposed first usable behavior contract.
- [Verification](verification.md): mandatory research readiness, reference fixtures,
  differential testing, and acceptance behavior.
- [GPU driver](gpu-driver.md): verified native Vulkan allocation/transfer/dispatch, bounded ownership and repeated raw-driver comparisons.
- [NVMe scratch transport](nvme-store.md): bounded asynchronous disk I/O through borrowed host staging; cache/scheduler integration is separate.
- [Snapshot residency](snapshot-residency.md): independent snapshot/hot/disk identities, pending-transfer ownership and read leases; metadata component, not disk-backed serving.
- [GPU matvec](gpu-matvec.md): verified resident packed-weight/FP32 projections, independent CPU/GPU goldens and repeated full-shape timings; losses retained.
- [Native model forward](model.md), [generation session](session.md): verified contracts.
- [Batched FP32 prefill](prefill.md): verified chunked prefill contract (block 13a); small-row plans (13d).
- [Split-K decode attention](decode-attention.md): verified three-pass decode attention (block 13b).
- [Matvec ≥1.30× push](matvec-push.md): bit-identity contract for exact-FMA decode, timestamp/read-probe diagnostics.
- [Matvec optimization DFS](matvec-optimization.md): fixed numerical gates, packed/aligned paths, F32 scheduling, baseline reconstruction and paired comparison protocol.
- [Quant decoding](quantization.md): implemented, independently verified CPU component.
- [Q6_K extension](q6_k.md): verified trailing-half/signed-subscale decoder and bounded output-weight slice measurements.
- [Q5_K extension](q5_k.md): verified 256-value packed-subscale decoder and matched real-model timings.
- [Q4_1 extension](q4_1.md): bit-exact native decoder and matched real-model measurements.
- [Quant component benchmark](quant-benchmark.md): implemented repeatable comparison harness.
- [GGUF loading](gguf.md): implemented bounded zero-copy parsing and Linux file mapping.
- [GGUF benchmark](gguf-benchmark.md): implemented independent full-container gate and parse/free comparison.
- [Official chat template](chat-template.md): implemented allocation-free text-only rendering with Jinja goldens and benchmark.
- [Tool calling](tool-calling.md): implemented `tools`/`tool_choice`/tool messages, Qwen3-Coder call parser and between-call constraint; llama-server parity and bruh end-to-end verified.
- [Unicode-9 NFC](normalization.md): implemented allocation-free normalization with independent normative/exhaustive goldens.
- [NFC benchmark](normalization-benchmark.md): measured native/HF comparison; no equivalent llama-server NFC operation.
- [Qwen text splitting](tokenizer-split.md): implemented borrowed-slice iterator with exhaustive property/golden checks and HF component timings; splitter only.
- [Complete Qwen tokenizer](tokenizer.md): implemented owned BPE/added-token tables, bounded encoding/raw decoding and actual-GGUF adapter; independent HF/raw-piece and actual llama-server comparisons.
- [Matched tokenizer benchmark](tokenizer-matched-benchmark.md): direct libllama comparison, allocation control, paired baseline and correctness-gated short-piece/byte-pair optimization.
- [External reference bring-up](reference-bringup.md): completed compatibility smoke experiment, not native serving.
- [Hermetic build](hermetic-build.md) (branch `bazel`): every build, test, oracle and harness input pinned by content or built in the graph; source-built shader tools, oracles, GPU test runtime (Mesa RADV, Vulkan loader) and competitors.

User requirements are settled constraints. Serving/API details are **draft** until
feature research resolves the open questions and a concrete schema/test suite is
recorded. Native quant decoding, container loading, text-only prompt rendering and
Unicode-9 NFC and complete Qwen tokenization/raw decoding are implemented, as is native
Vulkan memory/transfers/compute dispatch and packed-weight matvec; there is no native model
execution or serving yet.
Follow the [one-active-block queue](../../TODO.md) rather than starting packages in parallel.

Before a feature is ready for code, its specification must contain:

1. Inputs, outputs, exact semantics, supported capabilities, and explicit exclusions.
2. Limits, ownership/lifetimes, state transitions, failure and cancellation behavior.
3. Links to completed research: source revisions, equations/layouts, edge cases,
   hardware constraints, considered alternatives, resolved ambiguities.
4. An independent correctness oracle and executable comparison/fixture-generation
   plan whenever a reference implementation exists.
5. Acceptance criteria, numerical tolerances where applicable, negative tests,
   and the performance experiment that will judge the implementation.

A reference-validation harness is itself functionality: specify and research it
before coding it. Do not invent untested commands in docs and label them working.

- [RAM-staged disk prefix archive](disk-prefix-cache.md): immutable full checkpoint
  images, integrity, bounded pending I/O, model/scheduler ownership and 80k gates.

- [Asynchronous tiering plan](../design/async-tiering.md): GPU ownership and archive
  completion implemented; pressure policy still pending. Functional amendments in
  [GPU driver](gpu-driver.md#asynchronous-completion-ownership-18d7a) and
  [disk archive](disk-prefix-cache.md#asynchronous-device-quanta-18d7b-specified-before-implementation).
- [Radix cache source leases](cache-source-leases.md): generation/serial-qualified
  snapshot and ancestor ownership for background capture; byte adapter and pressure
  policy still separate. Independent prefix-set oracle and component gates pass.
- [Canonical mixed cache-source bytes](cache-source-bytes.md): immutable snapshot,
  mixed host/GPU capture, canonical partial tails and a bounded independent source
  job. CPU fixture/negative, device/model and component gates pass; no serving claim.
- [Pressure-driven archive admission](tiering-pressure.md): C.3 pre-code contract;
  independent policy oracle and integration remain pending.
- [Bounded archive progress during prefill](tiering-progress.md): D.0a pre-code
  contract; immutable-source-only unit-boundary callback, unchanged transfer window,
  independent ownership/byte anchors and packed-model interleaving gates.
- [Bounded archive transfer windows](tiering-window.md): D.0b pre-code limits,
  independent POSIX/SHA256 fixture and explicit staging/read-reservation tradeoffs.
- [Proactive preparation ownership](tiering-preparation.md): D.1.1 bounded pure
  page-pool transaction implemented/verified; D.1.2 joint cache/GPU adapter still
  requires its pre-code ownership oracle and scheduling contract.
- [Bounded short-reuse checkpoint suppression](reuse-join.md): opt-in cache-hit suffix joining, unchanged default and explicit future-recomputation tradeoff; independent policy and numerical/serving gates.
- [D.2 held-out quality gate](demand-quality-gate.md): predeclared synthetic record oracle, strict JSON scoring, resource limits and performance criteria; executed with shared model-answer failures, not passed.
- [Local InferenceX fixed-sequence benchmark](inferencex-local.md): security/build boundary,
  native ignore_eos contract, matched-work/resource gates, transparent timing observer,
  execution lifetime and checked resume. Three-round local matrix completed.
