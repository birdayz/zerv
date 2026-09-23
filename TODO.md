# Controlled work queue

Active goal (resumed by the user after 08c): native Qwen3.8-27B through
`POST /v1/chat/completions`, fully verified. Not achieved yet. No proxy, mock
inference, or external engine in the production path.

## Work controls

- **Exactly one active building block, in one package.** Finish its verification
  and measurement loop before starting the next. Supporting tests, tools and docs
  belong to that block; they are not permission to start another implementation.
- Research the primary sources and resolve semantics, then write the functional
  spec and executable independent oracle **before implementation**.
- Each block must pass Debug/ReleaseFast tests, negative/resource-limit tests and
  independent differential/golden checks. A plausible completion is not a test.
- Benchmark correctness-equivalent work against the pinned llama.cpp reference;
  **compare against actual `llama-server` at every runnable integration milestone**
  (tokenizer, model session, HTTP serving). Tune and repeat full-serving comparisons.
  If llama-server lacks the operation or has different semantics, record that
  limitation explicitly; use an independent equivalent component oracle and do
  not label the component timing a llama-server win.
- Record commands, versions, effective configuration, hashes, raw trials and
  variance under `docs/bench/`. No checkbox closes on unrun tests or estimates.
- If a gate fails, keep the block active and record the failure. Change this queue
  explicitly before changing scope. No parallel implementation tracks.

## Active — finish this before advancing

User goal (2026-09-24, later the same day): **"hardcore optimizations": faster than
llama-server, TTFT first, then tok/s; prefill first; never sacrifice correctness; real
trade-offs are config knobs.** This replaces "feature completeness first" for now; the
functional gaps below stay queued.

Queue change (2026-09-24, user request): block 14 is parked (see below). The
earlier "on par at 23/81/836/3223" session goal stays paused. 12c closed the same day.
Block 15 became active at the user's request ("fix the cache, priority 1"); its session
goal was then paused by the user. Queue change (2026-09-24): block 15 is **parked** with
its open gates (see Parked), and block 16 (prefill speed) is active.

Process note (2026-09-24): block 15's implementation started before this file was
updated to make it the active block. The research note and spec were written before the
policy code, but after the model primitive and its exactness run.

- [ ] **16 · prefill speed: faster than llama-server** (TTFT first; decode follows as
  block 17). [Design and decision log](docs/design/speed.md).
  - Steps, one at a time, each with its own gates and dated report:
    - [ ] 16b GEMM round 3, **first** (decision D4): the f16 WMMA path (target 80–90
      TFLOP/s including dequantization), then the FP32 path. Load-time weight
      repacking, LDS double-buffering, a device-keyed tile table.
      - [x] Lab rounds 1–2 ([report](docs/bench/2026-09-24-gemm-f16-lab.md)). k64 (wave32,
        128×256, BK=64, f16 X read directly) is bitwise-equal and 1.25–1.36× the shipped
        f16 kernel on the Q4_0 shapes (interleaved race; the card runs at 110 °C
        junction). Negative results, driver limits and occupancy variants are recorded.
      - [x] Integrate k64 for Q4_0 ([spec](docs/specs/prefill.md), "f16 GEMM, wave32
        kernel with f16 X"; [report](docs/bench/2026-09-24-gemm-f16x.md)): gates 1–4
        pass, with logits, captures and served outputs bitwise-identical.
        - 3223-token f16 TTFT: 3.62 → 3.04 s (llama default: 3.39 s).
        - 12k: 15.1 → 12.9 s (llama: 12.46 s).
        - Prefill GPU time −18%.
      - [ ] Port to Q4_1 and Q5_K; the 128-row plan; smaller plans; tail fill for the
        M = 5120 shapes (83% wave-quantization efficiency on 96 CUs).
    - [ ] 16a fused (flash-style) prefill attention: research (FlashAttention-2,
      llama.cpp `flash_attn.comp`), spec, FP64 component gate, model gates, serving at
      3223 and 29k.
    - [ ] 16c fusions: SwiGLU, residual + RMSNorm, conv, gates.
    - [ ] 16d chunkwise-parallel Gated DeltaNet.
    - [ ] 16e conv kernel, `--prefill-chunk` re-tune.
  - Baseline (measured): f16 3.54 s / 46.5 s TTFT at 3223 / 29k tokens against llama
    default's 3.40 / 33.3 s; FP32 5.21 / 62.8 s.
  - Open questions for the user: which precision mode carries the headline claim (Q1);
    MTP speculation for decode comparisons (Q2). See the decision log.

Process note (2026-09-24, block 15 prefix cache): a leftover zerv test server (pid 584034,
f16, 8 snapshot slots, from the bruh recording run) was still holding about 21 GB of VRAM when
the user started a second zerv on the same port 18080. The second bind succeeded, because
Zig 0.16's `listen(.{ .reuse_address = true })` sets SO_REUSEPORT as well as SO_REUSEADDR,
so the two processes shared the port. VRAM use reached 25.3 GB, the user's process hit a
compute ring timeout ("guilty of a hard recovery"; kernel log 16:21:25), and its
generation failed with DeviceLost. Stopping the leftover then exposed a shutdown panic
("snapshot store in use"): cleanup frees the snapshot buffer before the copy command that
still references it. **Fixed the same day** ([evidence](docs/bench/2026-09-24-serving-fixes.md)):
cleanup order; an exclusive listener bound before loading; a free-VRAM check before
allocating; 503 and exit status 3 once the engine is unusable.

Process note (2026-09-22, 13c): a serving benchmark was started concurrently with the
session check (two GPU tool calls in one step). VRAM oversubscription caused
compute-queue timeouts and resets. Both runs were discarded (logs retained), then rerun
one at a time.

Process note (2026-09-22, 13b/13d): 13d source edits began while 13b's serving
repeat was still queued, which broke that repeat's rebuild (retained failure; the repeat
was rerun on the pinned run1 binary). No 13d build, test or measurement ran until 13b's
measurements had finished.

Queue change (2026-09-22): 13b moved ahead of 12b because the per-phase profile
([prefill report](docs/bench/2026-09-22-prefill.md)) shows it is the largest measured
loss; 12b stays next after the 13-series speed items below.

Process note (2026-09-22): blocks 10–12 were implemented in overlap while the
block-09 oracle was generating, not strictly one at a time. The model spec and its
thresholds were written before native code and before any native-vs-FP64 run; the
session/serving specs were written after their implementation and after the
greedy-equality gate ran (they document the tested contract). Every gate listed was
actually executed.

## Queued — in order, one at a time

Proposed order (by impact on bruh use); the user chooses what comes next.

- **Long context (the 29,504-token cap).** The largest functional gap for agent work.
  - Why: bruh's first request is about 12.4k tokens (30 tools), so a session has about
    17k tokens of room before `400 context_length_exceeded`. The old Ollama setup served
    96–128k ([previous deployment](docs/research/2026-09-23-previous-deployment.md)).
  - Limits today ([long context](docs/bench/2026-09-24-long-context.md)):
    - FP32 KV, 128 KiB/token, in one state buffer under 4 GiB.
    - The materialized prefill score buffer (24 heads × 512 rows × context, FP32) puts
      the activation arena over the 4 GiB addressing limit above about 79k (estimate
      from the layout, before split-K partials). So long context needs fused
      (flash-style) prefill attention or smaller chunks, not only a smaller KV format.
  - Scope: KV formats f16 / q8 / q4 as explicit options, with quality measured against
    FP32 and against llama's f16/q4_0 KV; state split across buffers; prefill attention
    that does not materialize the scores; VRAM accounting (snapshots, context).
  - Related, on the client side: bruh's openai-compat provider never learns the context
    window (`context_window` is None), so it never compacts automatically; `:compact`
    works by hand. Consider exposing the context length from zerv (e.g. in `/v1/models`)
    and a bruh change to read it. The bruh change is in `~/projects/bruh`, outside this
    repo.
- **Prefix cache host-RAM tier** (only if block 15's benchmark shows the need). Today
  one conversation is cached: switching between sessions reuses only their shared
  tools/system prefix. llama-server keeps older conversations in RAM (`--cache-ram`,
  8 GiB by default).
- **Constrained decoding (grammar).** Needed for:
  - llama-server's in-call tool grammar (function and parameter names, schema-valid
    JSON values). Without it a malformed call is possible; it is handled and counted by
    `zerv_tool_call_parse_failures_total`, and none has been seen so far;
  - `tool_choice: "required"` and named functions (rejected with 400 today);
  - `response_format` `json_object` / `json_schema` (rejected with 400 today; bruh uses
    it only with `compaction.structured_output: true`, off by default).
- **Image input.** bruh's `browser` and `generate_image` tools can return images, which
  bruh sends as `image_url` user content; zerv rejects that with 400, so the turn fails.
  llama-server without mmproj fails the same way. Research first: the vision encoder
  for Qwen3.8-27B (the template has image/video tokens), its artifact, and the oracle.
- **More than one sequence at a time.** Today requests are serialized (a bounded wait
  queue). Research the VRAM, KV and scheduling tradeoffs of parallel sequences or
  batched decode, and what bruh actually sends concurrently.
- **Small API and usability items:**
  - `logprobs` / `top_logprobs` and `n > 1` (rejected with 400 today).
  - Model name: zerv accepts only its alias (`qwen3.8-27b`); bruh's default is
    `qwen3.8:27b` (Ollama naming). Decide whether to accept several aliases, or
    document `OPENAI_MODEL` / `--alias`.
  - The default port 8080 is taken on this host (`adp-work-server`); decide whether
    to document it or pick another default.
  - The startup free-VRAM check is a snapshot: two servers starting at the same moment
    can both pass it.

## Verification debt (no block yet)

- Long positions: FP64 oracle tensors cover prompts up to 562 tokens. At 29k tokens the
  evidence is output-text equality with llama-server at full FP32 (3/3 requests), not a
  tensor-level gate.
- The real device-loss path of the server (503, exit 3) is tested only with a simulated
  engine; a real GPU loss was not provoked on purpose.
- Block 14 (f16 prefill, opt-in): the worst-token quality gate is unmet (see Parked).

## Parked

- [ ] **15 · prefix cache** (reuse processed tokens across requests) — parked 2026-09-24
  for block 16; implemented and in use, gates below open.
  [Spec](docs/specs/prefix-cache.md), [research](docs/research/prefix-cache.md).
  - **Done:**
    - model snapshots (`saveSnapshot`/`loadSnapshot`, 150 MiB each);
    - policy `session.prefix` (keep / restore / reset, snapshot points at message
      boundaries, eviction);
    - `--prefix-cache-slots` (default 8, 1.2 GiB);
    - `usage.prompt_tokens_details.cached_tokens` and the cache metrics.
  - **Done, tests:**
    - 79 CPU tests including a randomized differential test against uncached runs
      (it catches both planted mutants);
    - split prefills pass the FP64 oracle gates, default modes 512:17/512:40 and long
      modes 512:17/300/555 ([data](docs/bench/data/2026-09-24-prefix-cache/)).
  - **Observed with real bruh (full tool set, f16):** cold 15.7 s TTFT at 12.4k
    tokens; later steps 0.59–0.79 s with 12.4–13k tokens reused; a new session reused
    the 12.4k-token tools prefix. The user's own sessions showed the same pattern.
  - Open gates:
    - [ ] `zerv-prefix-check`: compare restore against split directly (the rule in the
      spec) and exit 0 on it. The current tool compares against an unsplit run and
      exits 1, although the data shows restore == split at all 11 points.
    - [ ] Serving: the same multi-turn requests with the cache on and off; compare
      texts (the prompt split may change logits ≤ 5.3e-6) and check `cached_tokens`.
    - [ ] Benchmark against llama-server (defaults: prompt cache and checkpoints on).
      Replay recorded bruh sessions with `tools/record_proxy.py`; session A is recorded
      (4 requests), session B was cut off at 2 of its requests and needs re-recording.
      Also a fresh-session case. Report per-request TTFT and cached tokens, and peak
      VRAM.
    - [ ] Report `docs/bench/2026-09-24-prefix-cache.md` (already linked from the spec
      and research); serving spec (cached_tokens, metrics); `development.md` (new tools,
      `CHUNK:SPLIT` oracle modes); docs index.
    - [ ] HTTP-level test for the cache metrics and `cached_tokens`.
    - [ ] Decide from the benchmark: the slot default (VRAM vs reuse) and whether the
      host-RAM tier (queued below) is needed.

- [ ] **14 · explicit f16 prefill mode** — parked 2026-09-24 by queue change;
  implemented, gate 2 unmet.
  [Spec](docs/specs/prefill.md), [evidence](docs/bench/2026-09-24-f16-prefill.md).
  - **Implemented.** Gates 1 (component, RNE) and 3 (serving) are measured.
  - **Serving (run1) against llama default:**
    - zerv is 2.2× / 2.1× / 1.29× faster at 23 / 81 / 836 tokens;
    - zerv is 1.04× slower at 3223.
  - **Quality.** Better than llama nof16 (matched arithmetic) in mean, median, p90
    and p99, with 100% argmax agreement (llama default flips 13–24 tokens).
  - **Gate 2 unmet.** The single worst token is +20% (long-think) and +0.1%
    (long-prefill) against llama nof16.
  - Next options:
    1. The exact-integer-weight variant (per-block f32 scale; about 0.77× GEMM
       error, 17% slower GEMM).
    2. FP32 attention cost at long positions.


## Closed building blocks

- [x] **12c · tool calling, plus an end-to-end agent test.** [Spec](docs/specs/tool-calling.md),
  [research](docs/research/tool-calling.md), [evidence](docs/bench/2026-09-24-tool-calling.md).
  - `tools`, `tool_choice` auto/none, `parallel_tool_calls`, `tool` messages, assistant
    `tool_calls`; Qwen3-Coder XML parser with JSON and SSE `tool_calls`, finish reason
    `tool_calls`; llama's grammar after a complete call (whitespace, next call or EOS).
  - Prompts byte-equal to the independent Jinja oracle (26 cases, 830 number literals).
  - 22/22 greedy JSON/SSE cases equal to llama-server (llama-fp32-full), including
    argument bytes and token counts.
  - Real `bruh -p openai-compat --only bash` runs: 3 multi-step tasks completed,
    0 rejected, 0 malformed calls.
  - Not implemented (rejected explicitly): `tool_choice` required/named function,
    in-call grammar constraints.

- [x] **13j · DeltaNet prefill scan latency.** [Spec](docs/specs/prefill.md),
  [evidence](docs/bench/2026-09-23-delta-scan.md).
  - Gated norm split out, gate prologue, double-buffered q/k.
  - `delta` phase 62 → 45 ms per 512-row chunk; bit-identical on both oracles.
  - TTFT at 3223 tokens: 5.34 → 5.20 s.

- [x] **13i · prefill GEMM round 2 (research; no change).**
  [Evidence](docs/bench/2026-09-23-gemm-round2.md).
  - LDS reads are the largest remaining cost (+27% when removed).
  - The compiler sinks the "next X" SMEM loads to the loop latch, so SMEM latency
    is exposed once per 2 k.
  - Pipelining the A reads did not help (−1–3%).
  - Next levers need a compile-time X stride or hand scheduling.

- [x] **13h · FP32 prefill GEMM efficiency.** [Spec](docs/specs/prefill.md),
  [evidence](docs/bench/2026-09-23-gemm-efficiency.md).
  - Diagnosis: SGPR spills, `lgkmcnt`-serialized X loads, and unaligned block
    reads.
  - Part 1: bit-identical, 1.3–1.4× faster kernels.
  - Part 2: a 256×64 tile for 512-row chunks, bit-identical at equal split.
  - Result: the 512-row chunk takes 1178 → ~790 ms; TTFT at 3223 tokens drops
    7.76 → 5.34 s, and at 836 tokens 2.02 → 1.41 s. Served outputs are unchanged.
  - A new oracle case, `long-prefill` (562 tokens), gates 512-row chunks at model
    level for the first time.
  - The GEMM still runs at about 55% of the clock-adjusted FP32 peak.

- [x] **13g · cooperative-matrix prefill (research; negative for the FP32 default).**
  [Research](docs/research/coopmat-prefill.md),
  [evidence](docs/bench/2026-09-23-coopmat-research.md).
  - The f16 WMMA's f32 accumulation is not IEEE. Negative products are off by an
    ulp, and 16-term sums have a mean error 37× that of sequential fma.
  - The s8 WMMA is exact.
  - Peaks: WMMA ~135 T; scalar FP32 FMA 63–66 TFLOP/s.
  - Only 4 int8 limbs match FP32 GEMM accuracy on real data, with a ceiling of
    ~34 TOPS. That is below the VALU peak, so no default change.
  - The shipped GEMM runs at ~35% of the VALU peak.
  - Kept: a GPU-layer opt-in (`cooperative_matrix`) with a hardware test, and the
    research probes.

- [x] **12b · serving package — open items.** [Spec](docs/specs/serving.md),
  [evidence](docs/bench/2026-09-22-serving-open-items.md).
  - Real-socket tests: 503 overload, and disconnect cancellation.
  - Graceful drain on SIGINT/SIGTERM. The listener closes, new chat requests get
    503, admitted requests finish within `--drain-timeout`, and a second signal
    exits 130. Checked on the real binary.
  - Unmapping the model after load drops serving RSS from 15.5 GB to 60 MB. The
    load peak is unchanged.
  - Output text now follows llama-server's parser. Reasoning keeps trailing
    whitespace, stop strings match the raw text, delimiters are found in the text,
    and control tokens render as nothing. All 12 parity cases against llama-server
    are equal.
  - Serving speed is unchanged.

- [x] **13f · accumulation accuracy.** [Spec](docs/specs/decode-attention.md),
  [evidence](docs/bench/2026-09-22-accumulation-accuracy.md).
  - Research: model-level decode was less accurate than prefill; per operation the
    matvec is the most accurate projection. Localized to the decode attention score
    chain.
  - Fix: two-level score accumulation. Decode logits mean error 7.1e-7 → 4.5e-7, on par
    with llama.cpp FP32.
  - No speed cost; all gates pass; served outputs unchanged. A few worst-case samples
    grew (retained caveat).

- [x] **13c · decode step overheads.** [Spec](docs/specs/model.md),
  [evidence](docs/bench/2026-09-22-decode-overheads.md).
  - Norm: per-element bounds branches made the compiler reload descriptors (130
    reloads). A compile-time row width gives straight-line code: 1.36 → 0.23 ms per step.
  - DeltaNet: state column in registers, 1.00 → 0.58 ms.
  - Decode step 21.42 → 20.10 ms GPU; serving decode +7.7% (49.0 tok/s at 836 prompt
    tokens vs llama-server 41.4).
  - Bit-identical to 13e (34/34 capture files), after a failed first gate caused by
    unroll-dependent fma fusion (root-caused and fixed).

- [x] **13e · scalar-X prefill GEMM.** [Spec](docs/specs/prefill.md),
  [evidence](docs/bench/2026-09-22-gemm-throughput.md).
  - Diagnosis: the 128×128 kernel was LDS-read bound (ISA, wave32/VOPD probe, ablations),
    and the card is power-capped.
  - New kernel: X travels through wave-uniform scalar loads. Bit-identical to the old
    kernels at equal split for every format, 1.36–1.62× faster (sustained, realistic
    data). It replaces all 13d tiles.
  - Prefill 1.38–1.65× faster.
  - Serving TTFT: 23 tokens 102 ms (llama-server 162 ms), 81 tokens 243 ms (369),
    836 tokens 2.02 s (1.20 s).
  - All oracle gates pass.
  - Found: the earlier GEMM component numbers were optimistic (zero X).

- [x] **13d · small-row prefill plans.** [Spec](docs/specs/prefill.md),
  [evidence](docs/bench/2026-09-22-small-prefill.md). Tile-parameterized GEMM
  (128×128 / 128×32 / 256×16, bit-identical at equal split; 128×64 measured and
  dropped), one recorded command per plan, measured chunk policy. 23-token TTFT
  450 → 168 ms (llama-server 194–207 ms in the same runs), 81-token 487 → 390 ms;
  oracle gates pass in modes 0/1/13/29/60/512; greedy outputs unchanged.

- [x] **13b · split-K decode attention.** [Spec](docs/specs/decode-attention.md),
  [evidence](docs/bench/2026-09-22-decode-attention.md). Three passes (GQA-shared
  scores with chunk maxima, P·V partials under the exact global max, fixed-order
  combine); summation orders chosen by an FP32 emulation study on real tensors (more
  accurate than the old kernel). Decode step at ~3.2K 45.5 → 21.9 ms GPU; serving decode
  at 836 prompt tokens 34.8 → 45.7 tok/s (llama-server 41.3); all oracle gates pass;
  greedy outputs unchanged.

- [x] **13a · batched FP32 prefill.** [Spec](docs/specs/prefill.md),
  [evidence](docs/bench/2026-09-22-prefill.md). Tiled FP32 GEMM with exact dequant and
  split-K, batched operators, materialized causal attention, in-register DeltaNet/conv
  scans. Model-oracle bounds met at chunks 1/13/512 (worst 0.595 of bound), greedy
  equality through the server, deterministic. TTFT 7–10× better than token-by-token;
  ≈ parity with a fully FP32 llama-server control at 0.8–3.2K, 2.4–3.2× slower than
  its default Q8_1 prompt path; 23-token TTFT 437 ms vs 162 ms (tile floor, 13d).

- [x] **11 · generation session (`src/session`).** [Spec](docs/specs/session.md).
  Sampling, EOS/`</think>` handling, streaming UTF-8 and stop strings, cancellation;
  CPU tests plus greedy equality with libllama through the real server (JSON and SSE,
  both oracle cases). Matched llama-server comparison: [report](docs/bench/2026-09-22-serving.md) —
  decode +10% at short context, −44% at 3.2K; TTFT 3–32× slower (token-by-token prefill).

- [x] **10 · native Qwen3.8 forward (`src/model`).** [Spec](docs/specs/model.md),
  [evidence](docs/bench/2026-09-22-model-forward.md). Resident banked weights, shared
  matvec pipelines, operator shaders, FP32 KV/recurrent/conv state, one pre-recorded
  decode step. All declared gates pass vs the FP64 reference on 48+221-token
  sequences (worst ratio to bound 0.755; logits ≤3.5e-6; greedy 269/269), bit-identical
  across two runs; determinism/reset/bounds checked. ~23 ms/step at short context.
- [x] **09 · Qwen3.8 execution oracle.** [Semantics](docs/research/qwen35-execution.md).
  Pinned libllama per-layer capture + independent FP64 NumPy forward agree (≤1.05e-5
  intermediates, ≤2.4e-6 logits, no argmax disagreement). Fixture replays except two
  volatile fields (stderr log hashes, absolute build path).

- [x] **08c · Matvec push toward ≥1.30× reference (target partly met).** Bit-identical
  unrolled Q4 loads + exact-FMA decode; all outputs byte-identical to 08b, 48 fixtures
  both layouts, 42 CPU + 10 GPU tests both modes, 49 Python, replay. GPU kernel time
  ≥1.30× on 5/11 shapes, wall time on 3/11. Not met: Q6 (97% of measured read rate),
  larger Q4_0 (1.08–1.18×), ~40–55µs fixed per-call latency. FMA accumulation/scale
  factoring rejected because they change output bits. 18 experiments retained.
  [Evidence and limits](docs/bench/2026-09-22-matvec-push.md).

- [x] **08b · DFS matvec performance investigation/optimization.** Core half unpack,
  packed-word reuse, F32 scheduling and validated aligned Q4_1/Q5 paths. 33 tuning
  attempts/32 passed; wider tiles, FMA and 8-byte losses retained. Rebuilt scalar
  baseline from checked-in sources/shaders, inspected emitted GPU code, repeated
  paired full-shape comparisons twice before and twice after alignment specialization.
  Full Q6 ~5.61ms→1.22ms (~4.6×); every shape improves over our scalar baseline.
  Not uniformly faster than the reference: Q4 losses and offset2 Q5 loss remain;
  naturally aligned Q5 is near reference parity. No activation quantization or
  tolerance change. 42 CPU+8 GPU tests pass Debug/ReleaseFast, 48 Python and fmt;
  all48 independent cases run in both address layouts. Fresh replay rebuilds the
  independent fixture and all8 modules identically. 1348 final source/artifact hashes
  verified. [Results, failures and limits](docs/bench/2026-09-22-matvec-optimization.md).
  Serving/normalization remain paused; no new block starts automatically.

- [x] **08 · Native GPU packed-weight matrix-vector projections.** F32/Q4_0/Q4_1/
  Q5_K/Q6_K resident weights, FP32 input, checked byte views and bounded 2D dispatch.
  48 independent fixtures replay identically before native code; 380928 exact
  finite-half outputs, isolated columns, cancellation/zeros/tails and model samples.
  42 CPU +7 real-GPU tests pass Debug/ReleaseFast, 48 Python and fmt checks.
  Two full-shape runs cover all eleven base-model dense shape/type pairs, including
  the complete vocabulary projection; 939 source/artifact hashes verified.
  **Original scalar native loses every measured shape** (repeat Q6 ~4.5× slower). Separate default
  reference activation-quantization control retained; no false precision-matched
  speed claim. [Evidence](docs/bench/2026-09-22-gpu-matvec.md).
  `tools/replay_matvec.py` rebuilds independent fixtures/shaders and checks suites;
  benchmark runner rebuilds binaries. No native model/session/HTTP execution yet.

- [x] **07 · Native Vulkan driver/memory/transfer/dispatch.** Independent C ABI
  (43 structs/61 constants), scalar-checked real transfers/guarded dispatch goldens
  before native code; repeatable regeneration. Bounded buffers/kernels/commands,
  references, mapped spans, explicit barriers, replay/fences and error/state tests.
  39 CPU + 3 real-GPU tests pass Debug/ReleaseFast, 43 Python and fmt checks.
  Allocation audit invalidated two initial output-correct runs using an unenabled
  AMD memory feature. Core-only policy and cross-replay barriers corrected; two
  fresh matched runs and all 108-file snapshots verified. [Evidence/losses](docs/bench/2026-09-22-gpu-driver.md).
  GPU targets use system Vulkan/libc startup and bundled LLVM+LLD; CPU tests remain
  driver-free. No validation layer installed, no native model/HTTP execution yet.

- [x] **06 · Q6_K CPU decoding.** Independent scalar/external goldens regenerated
  identically before native code: 65,011,712 global-half values, 4,194,304 signed-
  subscale values, isolated payload bits, edge/seeded cases and 70 actual output-
  weight blocks. Trailing-half validation, signed zeros, unaligned/length/atomicity
  and payload-prefix negatives pass. 35 native tests Debug/ReleaseFast, 37 Python.
  Two matched output-weight-slice runs plus Q4_1/Q5_K compatibility runs; all full
  slice hashes and 86-file snapshots verified. [Evidence/reference limits](docs/bench/2026-09-22-q6_k.md).
  All artifact packed types covered; CPU diagnostic success is not GPU execution.

- [x] **05 · Q5_K CPU decoding.** Independent scalar/external goldens before code:
  97,517,568 finite-domain values, packed-scale byte fingerprint, high bitplanes,
  seeded/edge cases and 384 blocks across all 48 actual tensors. 32 native tests in
  Debug/ReleaseFast, 35 Python tests; bounds/nonfinite/atomicity/unaligned checks pass.
  Two matched real-model timing runs, plus Q4_1 compatibility rerun; snapshots verified.
  [Results/losses](docs/bench/2026-09-22-q5_k.md): full Q5_K tensor 3.2% slower on
  repeat; current small-Q4_1 timings regressed and are explicitly retained.

- [x] **04 · Q4_1 CPU decoding.** Spec/independent fixtures preceded native code;
  44,695,552 finite-field/coefficient values plus explicit patterns and all eight
  model tensors bit-exact. Both fields/all nonfinites/unaligned/bounds/atomicity pass;
  original Q4_0/Q8_0 gates retained. 29 native tests Debug/ReleaseFast, 33 Python.
  Two actual-shape direct ggml benchmark runs and source snapshots verified.
  [Results](docs/bench/2026-09-22-q4_1.md) retain the 8.3% full-tensor repeat loss;
  diagnostic decoding is not inference or a serving speed claim.

- [x] **03b · tokenizer matched benchmarking and two optimization steps.** Audited
  exact llama call path/library, paired original/optimized binaries and reference
  allocation control. 19,033 encode cases, 70 raw cases, all pieces; 26 native tests
  in Debug/ReleaseFast, 30 Python tests. Ten original runs plus a full replay using
  a source-rebuilt baseline; hashes verified. [Results/replay](docs/bench/2026-09-22-tokenizer-matched.md).
  User resumed serving work. Deferred: long-piece/NFC optimization and measured
  raw-decode regression investigation; retain losses and the 512 KiB table tradeoff.

- [x] **03 · `src/tokenizer`: complete BPE/added tokens/raw decoding.** Research,
  [spec](docs/specs/tokenizer.md) and independent fixtures preceded implementation.
  Exact official IDs for 1,240 encode cases, 70 decode cases and all 248,320 raw
  pieces pass on fixtures and actual GGUF. Table/ownership/limits/allocation-failure
  tests pass: 24/24 native Debug/ReleaseFast, 25/25 Python. Regenerated fixtures and
  independent raw-piece extraction are byte-identical. Corrected dependency count:
  **27 direct** forward-rank edges (the initial 66 included downstream rejections).
  Actual llama-server: 860 direct matches; all 380 NFC-changing cases match after
  normalization, with zero residual differences. Two runs, 64-file snapshots each,
  hashes verified; [report and timing/ownership caveats](docs/bench/2026-09-22-tokenizer.md).
  Raw-byte decoding is not streaming UTF-8 repair; no native inference/serving claim.

- [x] **02 · `src/tokenizer`: Qwen text splitting.** Research/spec and 47,919 HF
  fixtures preceded native code. Exhaustive scalar properties, exact split ends,
  UTF-8/limits/borrowed boundaries/state and byte-identical regeneration pass.
  Full suites: 20/20 native tests in Debug/ReleaseFast, 19/19 Python tests. Two
  benchmark runs with 53-file source/data snapshots verified; [results and API
  ownership caveats](docs/bench/2026-09-22-tokenizer-split.md). This is not BPE or a
  llama-server win; the actual server comparison remains a gate in block 03.

- [x] **01 · `src/text`: Unicode-9 NFC normalization.** Research/spec and independent
  fixtures preceded implementation. Table/generator provenance, 93,610 normative
  relations, both exhaustive fingerprints, buffer/UTF-8/atomicity tests and repeat
  fixture regeneration passed. All 17 native tests pass Debug/ReleaseFast; all 14
  Python tests pass. Two benchmark runs and 41-file source/data snapshots per run
  verified. [Results](docs/bench/2026-09-22-normalization.md) retain the first marks
  loss and sign-changing repeat; no stable marks or llama-server speedup claimed.
  Actual server lacks NFC; its tokenizer comparison remains open in block 03.

## Previously closed foundations

- [x] Q4_0/Q8_0 CPU decode: independent goldens and repeatable component comparisons.
- [x] Bounded GGUF/mmap loading: full-artifact independent comparison and timings.
- [x] Official text-only chat rendering: independent Jinja fixtures and timings.
- [x] Selected model downloaded and SHA-verified.
- [x] External llama-server compatibility smoke only; **not native goal completion**.

Evidence and limitations: [documentation index](docs/README.md),
[roadmap](docs/roadmap.md), [benchmark index](docs/bench/README.md).
