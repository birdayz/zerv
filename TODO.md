# Controlled work queue

Active block: **19 · image input (vision)** (below). Active goal (resumed by the user after 08c): native Qwen3.8-27B through
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
- **Always DFS into bugs** (user, 2026-09-24): an unexplained wrong result is top
  priority. Drill down to the root cause (reproduce, minimize, isolate the layer at
  fault: our code, compiler or driver) before resuming other work, and record each step.
  Avoiding the trigger is a workaround, not a fix; it does not close the bug.

## Active — finish this before advancing

### BUG (root cause found 2026-09-24): wrong multi-row matvec results on the K-quant path

- **Root cause:** Mesa 26.2.3 ACO. VGPR spills placed in LDS (single-wave workgroups)
  carry no memory-sync info, and the pre-RA scheduler (which runs after spilling) hoisted a
  spill store above a reload from the same, reused slot: one accumulator per workgroup
  read a weight value. Proof: `ACO_DEBUG=nosched` is correct with identical spilling, the
  scratch-spill builds (wave32, LDS padded) are correct, and exactly 1 of 225 reloads moves
  across an aliasing store. Full RCA with 5 whys: [report](docs/bench/2026-09-24-aco-lds-spill.md).
- [x] Workaround: no shipped shader spills VGPRs into LDS; gate
  `tools/check_shader_spills.py` (with negative control) in the required checks
  (docs/development.md, docs/specs/verification.md). The tuning tools reject spilling
  candidates and validate every candidate on all formats.
- [ ] Upstream: file a Mesa issue with the reproducer and the proposed fix (tag LDS spill
  ops `storage_vgpr_spill`). Optionally verify the fix with a locally built Mesa via
  `VK_ICD_FILENAMES` (no system change). Needs the user (network, large build).
- [x] Performance side finding: `K_DELTA`/`K_DELTAB` spilled 128/130 VGPRs to scratch.
  Cause: 120 store addresses kept live across the row loop. Fixed with a separate
  `state_out` store offset, knob `--delta-state-out on|off`: spills 7/9, byte-identical
  on both oracles, plain decode +1.6%, speculative +1.1–1.7%, TTFT −0.4 to −1.0%
  ([report](docs/bench/2026-09-24-delta-spill.md)). Also found: one scratch-heavy kernel
  slows later submissions on the queue (RADV's per-queue scratch ring; mechanism open).

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

Queue change (2026-09-24, user: "pivot to tok/s perf now"): block 16 is **parked** with
its open work listed under Parked; block 17 (decode tok/s) is active.

Queue change (2026-09-24, user: "yeah i suppose this is actual real work, without it's just
a toy", on serving several requests at once): block 17 is **parked** in place below with its
open items (status unchanged; the single-user levers found by the HyperQwen protocol run are
listed in it); block 18 is active.

Queue change (2026-09-26, user: "commit what we have. then, start working on vision support for
qwen 3.8:27b"): block 18 is **parked** in place below with its open items; the in-tree 18e
part 4 work (`gemm_rows.comp`, `DecodePrecision.split`, its gpu-test) was committed as WIP in
`c30c6d9` without a spec section, report or model gates. The queued "Image input" item
becomes block 19, active.

- [ ] **19 · image input: Qwen3.8-27B vision through `/v1/chat/completions`** (active).
  - Why: bruh sends tool screenshots as `image_url` user content (PNG data URLs); zerv answers
    400 and the turn fails. llama-server serves them with `--mmproj`.
  - Steps, one at a time, each with its gates:
    - [ ] 19a research ([note](docs/research/vision-qwen38.md), first pass 2026-09-26): HF and
      llama.cpp semantics read at the pinned revisions and cross-checked (encoder, merger,
      2-D RoPE, interleaved M-RoPE positions, preprocessing, template, server media);
      projector `mmproj-BF16.gguf` downloaded and verified, all 334 tensors bit-equal to the
      RedHat checkpoint's BF16 tower (`tools/vision_artifacts.py`); llama-mtmd-cli runs it
      here. Found: llama.cpp's merger uses tanh GELU (HF: erf), pads instead of stretching,
      caps images at 4,096 tokens (HF 16,384); the prefix cache would confuse two same-size
      images (all placeholders are one token id). Open: the note's section 9 (7 items, two
      need a user decision: the HF dev-tool venv, the default image token maximum).
    - [ ] Spec `docs/specs/vision.md` with gates and thresholds; FP64 NumPy vision reference
      and the llama.cpp embedding/logits capture, validated against each other, before code.
    - [ ] 19b image decode (PNG first) and preprocessing, bit-exact to PIL.
    - [ ] 19c vision encoder on the GPU vs FP64; component benchmark vs llama.cpp's encoder.
    - [ ] 19d language-model integration: embedding input rows, M-RoPE rows, per-sequence
      position offset, prefix-cache key with image identity, MTP draft input policy;
      whole-model logits vs FP64 and libllama.
    - [ ] 19e serving: `image_url` data URLs, template, limits and errors; served outputs
      and TTFT vs llama-server `--mmproj` on the same images.
    - [ ] 19f JPEG, then WebP/GIF or explicit rejection.

Process note (2026-09-26, user: "optimize our tests for parallelism"): tests now run one binary
per file in parallel (`zig build check` runs every required check; ~96 s → 9.4 s cold for
`zig build test`; [report](docs/bench/2026-09-26-test-parallelism.md)). The load exposed a
committed batcher defect (a canceled prompt ran all its remaining chunks; fixed), a committed
test hang (the packed-chunk test, 5/240 under load; fixed) and two timing flaws in the
uncommitted 18d.2 shared-pool work of a concurrent session (fixed in place, uncommitted).

Process note (2026-09-26, user: all tooling to Bazel only, branch `bazel`; then "everything
must be 100% hermetic", merge to main only once it all is): Bazel builds and tests everything;
`build.zig` is removed and the required checks are `bazel test //...`
([report](docs/bench/2026-09-26-bazel-build.md), [commands](docs/development.md),
[spec](docs/specs/hermetic-build.md)). Done on the branch: source-built shader tools, oracles
(ggml, llama.cpp), GPU test runtime (Mesa RADV, Vulkan loader), fixtures regenerated with
them, harnesses without host programs, llama-server built in the graph, the HIP competitor's
recipe (byte-identical rebuild), the GPU tests in the container (`tools/hermetic_check.sh
--gpu`, 70/70), host-driver GPU tests gating production benchmarks. Merged to main 2026-09-26.
Open (not a building block): the cold-build comparison (`bench/cold_build.py`), a recheck of
the llama-server A/B on an idle machine, the Zig caches outside the output base.

- [ ] **18 · concurrent sequences: batched decode for several requests** — parked 2026-09-26
  for block 19.
  - Why: zerv serves one request at a time; bruh's parallel subagents queue. Decode is
    weight-bandwidth bound (16.7 of 20 ms per step), so one weight pass can serve several
    sequences: the verify path already does 3 / 4 / 5 rows in 22.7 / 25.0 / 28.8 ms against
    20.0 ms for one. HyperQwen (3090) reports ~400 tok/s aggregate at 8 users.
  - Requirement (user, 2026-09-24): hardcore performance — saturate the GPU optimally
    (aggregate throughput, cost effectiveness, bin packing), and the HTTP/CPU side held to the
    same bar (comptime specialization, nothing on the GPU's critical path).
  - Measured (18a): today's FP32 multi-row projection kernel is compute-bound after ~5 rows;
    batching tops out at ~3.3× aggregate around 4–5 rows
    ([data](docs/bench/data/2026-09-24-rows-scaling/)). Saturation beyond needs a faster batched
    projection: a tuned FP32 kernel (exact) or batch-invariant WMMA (a precision knob).
  - Decision (user, 2026-09-24): build everything that is batch-invariant for multiple users;
    the batched projection arithmetic is a knob with both options available — exact FP32
    (default, a tuned multi-row kernel) and an opt-in WMMA f16 mode (batch-invariant within
    itself); KV moves to a shared paged pool.
  - Design direction (user, 2026-09-24): the scheduler is general-purpose serving code; the
    model supplies a narrow backend (per-slot state layout, slot-indexed kernels, recorded
    commands per batch size, a memory/cost model). The interface must express both per-token
    KV and fixed-size recurrent state (Qwen3.8: KV in 16 layers, DeltaNet state in 48), so
    vLLM-style paged-KV prefix sharing does not carry over as is.
  - Steps, one at a time:
    - Competitor baseline (18a, [report](docs/bench/2026-09-24-concurrency-baseline.md)):
      llama-server `-np 8` plain reaches 156 tok/s aggregate at 8 clients (3.9× its single
      stream; MTP 3 only 117); zerv stays at 50 / 87 (3 drafts) with TTFT up to 72 s queued.
    - [ ] 18a research — first pass written ([note](docs/research/concurrent-sequences.md):
      the code survey, llama.cpp's batching and per-sequence recurrent state, bruh's actual
      concurrency, a first memory model); its six open items remain. Scope as planned:
      llama-server `--parallel` and vLLM scheduling semantics (slots,
      continuous batching, chunked prefill, fairness), what bruh sends concurrently, where
      zerv assumes one sequence (model state, session, serve), VRAM per slot, the compute
      crossover (estimated 8–16 users on the current decode kernels). Research note, then the
      spec in docs/specs/ with the gates.
    - [x] Spec written: [docs/specs/concurrent.md](docs/specs/concurrent.md) (knobs, batch
      invariance, paged KV pool with 256-token pages, backend interface, scheduler, stages).
    - [x] 18b slots, paged KV and batched decode (f32) in the model; gate `zerv-batch-check`
      (every logits row bitwise equal to the sequence decoded alone, with join/leave).
      - [x] 18b.1 paged KV addressing, one sequence, identity page table
        ([report](docs/bench/2026-09-24-paged-kv.md)). Every gate is byte-identical (fp32 and
        f16 oracles, forced KV split, f16 KV, spec/mtp/prefix checks, serving-v2). Decode is
        −0.24% short and +0.14% / +0.33% (f32 / f16 KV) at 30k; prefill is −0.1%. The GPU tests
        use a permuted page table.
      - [x] 18b.1 follow-up: `--kv-page-tokens N|context` as specialization constant 0
        (`gpu.Kernel.Options.constants`; machine code identical to the `#define`). `context`
        selects the pre-paging layout. All gates are byte-identical at 128, 256 and `context`.
        Page-size sweep at 30k: 128 is closest to pre-paging (+0.06% f32 KV, +0.07% f16 KV,
        against +0.18% / +0.39% at 256); the page table itself is free. **Default is now
        128.** Found and fixed a GPU test that passed without running (a ReleaseFast assert is
        UB; rule in verification.md) ([report](docs/bench/2026-09-24-paged-kv.md)).
      - [x] 18b.2 slots and batched decode in the model
        ([report](docs/bench/2026-09-25-batched-decode.md)). `zerv-batch-check` is 399/399
        rows bitwise equal to solo decoding (joins/leaves, permuted rows and pages, B = 1..8)
        in 4 KV configurations. The single-sequence path is byte-identical and its speed
        unchanged. Model-level: 158.6 tok/s at B = 4 and 172.8 at B = 8, against 49.8 at B = 1
        (projection groups reread the weights past 4 rows; 18e).
    - [x] 18c scheduler and `--parallel N` serving ([report](docs/bench/2026-09-25-parallel-serving.md)).
      The gate passes: 60/60 concurrent responses are byte-identical to solo (greedy and sampled,
      1–8 clients). Aggregate throughput is 49.4 / 91.4 / 147.6 / 161.8 tok/s at 1/2/4/8 clients,
      against same-session llama-server `-np 8` at 39.8 / 65.2 / 99.1 / 143.0. TTFT p50 is
      0.53 s at 8 clients against 1.38 s. The GPU is 99% busy. llama-server is not
      batch-invariant (3–5 greedy outputs per prompt). The session is not yet a state machine
      (the spec's host item); host time is measured as not limiting.
    - [x] 18c.2 prefill without stalling the others ([report](docs/bench/2026-09-25-multiuser.md),
      spec concurrent.md "18c.2 design"):
      - Prefill chunks run as 16 four-layer segments, bitwise the chunk; decode batches run
        between the segments.
      - `--prefill-stall-ms N|chunk` (default 100) and `--prefill-order shortest|fifo`.
      - Ownership audit fixes, with tests.
      - `--f16-small-tile`: a 32 × 128 f16 GEMM tile for 128-row plans, bitwise gemm_f16,
        +5.3% on 128-row prefill.
      - Gates passed: gpu-test 39/39, batch-check 400/400 (f32 and f16 KV), f16 and FP32
        oracles byte-identical, serving identity 30/30, no decode regression (ABBA).
      - Final interleaved comparison (zerv, vLLM chunks 2048 and 512, llama `-b 2048` and `-b 512`):
        zerv leads at 1–4 users (+33–50%), on 8-user TTFT (0.44 s against 0.80 s) and on
        long-prompt interference. vLLM leads 8-user throughput (158.2 against 150.9) and
        steady gap p99 (51 against 147 ms): it packs simultaneous prompts into one prefill.
    - [ ] Competitor: vLLM ([report](docs/bench/2026-09-25-vllm.md)), user decision
      2026-09-25 ("the serious competitor").
      - Official ROCm image v0.30.0 (gfx1100 build, pinned digest) with RedHatAI's W4A16
        checkpoint. Different weights, FP8 KV: speed is compared, quality separately.
      - Downloads hash-verified, the safetensors structure checked, ClamAV scans clean. The
        container runs unprivileged with the GPU device nodes only.
      - Runs; KV pool 67–72k tokens; a cold graph compile fails for lack of KV memory, so
        every configuration is warmed once first.
      - Multi-user: in the 18c.2 final run. Pending: single user (serving-v2, MTP 3).
    - [ ] Competitor: SGLang (user goal 2026-09-25: "absolute kings in all metrics" against
      vLLM and SGLang). Check ROCm/gfx1100 and Qwen3.8 support, verify and scan the image,
      then the same runs.
    - [x] 18d.1 packed multi-sequence prefill ([report](docs/bench/2026-09-26-packed-prefill.md)):
      bitwise solo per sequence (batch-check pack 504/504 x4, serving 30/30). 8-user tok/s
      141.7 → 150.8 (vLLM 151.9), TTFT p95 1.58 → 1.05 s. Steady gap p99 unchanged (149 vs
      vLLM 53; max 193 vs 466–896): few chunks pack in the closed loop.
    - Queue decision 2026-09-25 (user goal: multi-user throughput and batching): 18e (batched
      projection kernels) goes before 18d. The 8-row step is the bottleneck: 46 ms, 2.3× one
      row, and the GPU is 99% busy. Within 18d, per-slot prefix caching comes first (bruh's
      multi-turn long prompts).
    - [ ] 18d speculation per slot, chunked prefill interleaving, per-slot prefix cache,
      preemption, async scheduling.
    - [ ] 18e batched projection kernels: tuned FP32 beyond 5 rows; WMMA `--decode-precision f16`.
      - [x] FP32 part 1, negative ([report](docs/bench/2026-09-25-fp32-batched-projection.md)):
        component splitting (`SPLIT`) and adjacent row groups (`ROWGROUPS`) stay bitwise
        exact. Neither is a material win: 8 rows cost 8.48 against 8.95 ms, a one-row pass
        3.48 ms. The kernel is stall-bound at ~5.2 TFMA/s from 4 rows on; the stall source is
        unidentified (no profiler). Recorded candidates: ROWGROUPS (+5%), a software-pipelined
        loop, LDS-shared X. Nothing shipped; the modules are byte-identical.
      - [ ] WMMA `--decode-precision f16` (opt-in).
        - [x] Kernel v1 ([report](docs/bench/2026-09-25-f16-decode-mode.md)): spec written;
          `gemm_f16n` (128 × 16 tile, shape-only split-K). Bitwise equal to `gemm_f16` per row
          and per split part (gpu-test). `zerv-batch-check` in f16 mode is 399/399.
          `Options.decode_precision`; no CLI yet. Slow: 45.4 ms at 1 row, 50.0 at 8 (FP32:
          20.1 / 46.3), about 35% of DRAM bandwidth; the prefill tile's LDS/barrier scheme
          buys nothing at 16 rows.
        - [x] Kernel v2 ([report](docs/bench/2026-09-26-decode-v2.md)): `gemm_f16d`,
          bitwise v1, 1.3–1.5× faster per projection; Q4_1/Q5_K stay FP32
          (`decode_f16_formats`). 8-row step 38.6 ms (FP32 46.3), 1 row 31 ms (FP32 20).
          WMMA decode is bounded near 600–700 GB/s by the tile's issue cost (estimate from
          ISA; ablations in the report): it cannot match FP32 at 1 row, so the mode stays
          internal (batch invariance forbids choosing arithmetic by batch size).
      - [ ] Next (the large lever): the exact FP32 multi-row projection at ~5.2 TFMA/s. With
        long per-row weight runs (the v2 study: 576-byte runs reach the ~800 GB/s harness
        ceiling) and the FMA work spread across lanes, an 8-row step near the 1-row memory
        time would roughly double 8-user throughput, bit-exact and in the default mode.

- [ ] **17 · decode speed: tok/s faster than llama-server** (single stream first) — parked
  2026-09-24 for block 18. [Design and decision log](docs/design/speed.md).
  - Baseline (measured, serving-v2, [report](docs/bench/2026-09-24-gemm-f16x.md)):
    zerv 54 / 49 / 49 / 56 tok/s against llama default 44 / 41 / 41 / 45; 38.5 against
    37.0 at 29k. Weight streaming bounds plain decode at about 60 tok/s (15.3 GB per
    token at a measured ~920 GB/s read rate).
  - Steps, one at a time, each with its own gates and dated report:
    - [x] 17a baseline and competitor: per-phase profile at positions 70 and 8,020,
      llama default and MTP 1–4 with acceptance; n-gram stalled
      ([report](docs/bench/2026-09-24-decode-baseline.md)).
    - [x] 17b MTP speculative decoding with the GGUF's nextn layer. **Closed
      2026-09-24: gates 1–4 passed** ([report](docs/bench/2026-09-24-speculative.md)).
      Default since then: 3 drafts, adaptive policy; `--spec-draft 0` turns it off.
      [Research](docs/research/speculative-mtp.md), [spec](docs/specs/speculative.md).
      - [x] Research, from the pinned llama.cpp source: MTP semantics and the draft driver.
      - [x] Competitor baseline: llama MTP 1–4 on decode-v1 reaches 105–112 tok/s on
        code/json; its output differs from its own non-speculative output.
      - [x] 17b.1 N-row verification ≡ decode (gate 1): `matvec_rows`, row-capable decode
        kernels, commit pass, `zerv-spec-check`. Gate passed. Verify + commit of 3 / 5
        rows costs 24.1 / 32.4 ms against a 20.0 ms step, after exact-count modules
        ([report](docs/bench/2026-09-24-spec-verify.md)).
      - [x] 17b.2 MTP layer (Q8_0 rows module, layer, catch-up, draft chain) with an FP64
        reference (gate 2, component part): `zerv-mtp-check` + `tests/reference/
        mtp_reference.py`. h' and draft logits within 2–5e-7 normalized L2 of FP64 on two
        oracle sequences, every draft equal to the FP64 argmax
        ([data](docs/bench/data/2026-09-24-mtp-gate/)). Batched prompt catch-up since
        ([flash report](docs/bench/2026-09-24-flash-attention.md), section 2); `step()`
        with MTP on runs the pending rows since (`zerv-mtp-check` scenario C).
      - [x] 17b.3 Engine loop with sample-matching acceptance, `--spec-draft`, adaptive
        verify count. Gate 3: every speculative output equals the plain one (greedy,
        sampled, 38k). Gate 2 acceptance: counts identical to llama MTP where outputs
        agree. Gate 4: faster than llama's best on every decode-v1, serving-v2 and 38k
        case. Counters in `/metrics` and the harness.
    - [ ] 17c decode and memory, one at a time:
      - [x] `--context max` with `--vram-reserve-mib` (default 1024, as llama's
        `--fit-target`): 44.9k by default, up to 60.1k (67.8k with no reserve) with
        knobs; every configuration loads and serves
        ([report](docs/bench/2026-09-24-context-max.md)). Contexts above the trained
        262,144 are rejected.
      - [x] KV precision knob `--kv-type f32|f16`
        ([report](docs/bench/2026-09-24-kv-precision.md)). f16: context ×2 (88.6k
        default, up to 135k), 38k decode 34.8 → 39.5 tok/s (llama 35.8), 3 drafts 84.6;
        KL to f32 3.4e-7 (llama's f16 KV: 3.2e-4); outputs identical. Gate 3's
        worst-token clause is not met (4–6% over llama's matched config), so f32 stays the
        default. 75.5k (long-v2): all engines answer correctly; llama 3.5% faster prefill
        and 6% faster plain decode there, zerv 3 drafts fastest decode (in the report).
        Open: q8 KV (research).
      - [ ] **In the tree, unfinished (2026-09-24):**
        - [x] FMA accumulation in `matvec.comp` / `matvec_rows.comp` with a re-tuned
          verify table ([report](docs/bench/2026-09-24-fma-matvec.md)): all gates pass,
          decode more accurate against FP64; verify+commit 3 / 4 / 5 rows 24.24 → 22.68 /
          27.30 → 25.03 / 32.60 → 28.77 ms (interleaved race); decode step unchanged.
          Serving decode-v1, 3 drafts: 120.5 / 124.8 / 107.7 / 75.6 tok/s (was 110 / 114 /
          99 / 71; llama MTP 4 105 / 109.5 / 90.5 / 57.5), outputs unchanged. Open: why
          G = 4 at 5 rows breaks bitwise equality for Q5_K / Q6_K (never shipped).
        - [x] Decode attention global-max pass (`attn_gmax`) and parallel combine
          (`attn_cblock`): bitwise identical on both oracles (42 files each), spec-check
          11/11; decode step at 64k 30.64 → 26.97 ms, at 8k −0.25 ms, at position 300
          +0.07 ms (two more dispatches per layer; folding the maxima into the scores pass
          would remove one). [Report](docs/bench/2026-09-24-decode-attention-long.md).
          Then the scores pass on key pairs (wider K loads, Q from LDS as vec4): bitwise
          identical (42 files), 64k step 26.97 → 26.75 ms, 300: +0.03 ms. Then P·V in
          two waves (pv2w): 300 back to 20.18 ms, 8k 20.65, 64k 26.57.
      - [ ] **Decode kernel opportunities** (user-approved list, 2026-09-24). Basis: the
        short-context step is 20.0 ms; the weights alone need 16.7 ms at the measured
        920 GB/s; a diagnostic build without barriers ran 18.3 ms. Gains are estimates.
        - [ ] Phase-dependency cost, **1.7 ms (8%) measured**: fuse dependent phases
          without changing values (a DeltaNet layer has 8 phases, an attention layer
          about 12).
          - [x] Gate+up+swiglu in one dispatch, knob `--decode-fusion on|off` (default on):
            byte-identical on both oracles, spec-check 11/11, 63 of 64 layers fused; step
            20.015 → 19.667 ms, plain decode 49.9 → 50.7 tok/s, speculation unchanged
            ([report](docs/bench/2026-09-24-decode-fusion.md)).
          - [x] The same fusion in the verify pass, knob `--verify-fusion on|off` (default
            on): verify+commit 1–4 rows −0.25 to −0.37 ms, decode-v1 3 drafts +0.4 to
            +1.2%, outputs identical ([report](docs/bench/2026-09-24-verify-fusion.md)).
            Count 5 stays unfused: 4 weight rows × 5 input rows is wrong on the K-quant
            path (second occurrence; cause not investigated).
          - [ ] Further fusions: find which other phases can merge (conv into the lin_in
            epilogue if its per-head norm allows, the small attention passes, norms).
        - [ ] Projection efficiency, about 1.6 ms: in-model projections read at 805–820
          GB/s against 923 GB/s streaming one role without barriers; Q5_K `ssm_out` and
          `attn_out` at about 720 GB/s ([baseline](docs/bench/2026-09-24-decode-baseline.md)).
          - [ ] Weight repacking at load (bit-identical values): Q4_0 blocks are 18 bytes
            and read through unaligned word handling; a layout with separate scales and
            16-byte-aligned quants allows wide loads. The largest untried kernel lever.
        - [ ] Prefetch the next matrix into the 96 MB Infinity Cache during latency-bound
          phases (norms, conv, DeltaNet step, attention), when DRAM is idle. Untested
          idea; maybe 0.5–1 ms; values unchanged.
        - [x] Speculation: profile the MTP draft passes. **1.55 ms per drafted token, 16%
          of a 3-draft cycle**; the Q6_K output head is 80% of the bytes and already at
          916 GB/s ([report](docs/bench/2026-09-24-draft-vocab.md)).
          - [x] Knob `--spec-draft-vocab N|full` (default full): the draft head covers
            ids 0..N-1. Output identical in every run; decode-v1 +7–9% at N = 65,536;
            multilingual text regresses (Chinese below plain decode), so opt-in only.
          - [ ] Frequency-ranked draft vocabulary (row indirection in the head matvec,
            index → id map in the argmax) to keep the gain across languages; needs a
            representative token-frequency source that is not fit to the benchmarks.
        - Target: 10–15% more plain tok/s; speculation inherits most of it.
      - [ ] Plain decode at long context: **75.5k serving now 35.7 tok/s against llama's
        32.4** (was 30.6), 3 drafts 60.6 against llama MTP 3's 49.9
        ([report](docs/bench/2026-09-24-decode-attention-long.md)). Still below the read
        roofline at 64k: scores read K at about 755 GB/s, P·V read V at about 590 GB/s.
        P·V in two waves with 4-dim loads (pv2w) shipped: bitwise identical, 64k P·V
        3.99 → 3.45 ms, short context back to the original 20.18 ms. At 64k the step is
        26.6 ms (from 30.7).
      - [ ] Prefill at very long context: 75.5k TTFT 119.9 s against llama's 115.7 s
        (f16 prefill, f16 KV). The fused attention kernel runs at 18.6 TFLOP/s against
        31–37 for the GEMMs ([flash report](docs/bench/2026-09-24-flash-attention.md)).
        Wave32 tried: 31% slower (negative, section 3b). Remaining lever: WMMA attention
        in the f16 prefill mode (opt-in precision change; research llama.cpp's
        `flash_attn_cm1.comp` first).
      - [ ] Host snapshots through a device staging slot and an async transfer (removes
        the 10–19 ms TTFT cost of `--prefix-cache-memory host`;
        [host memory](docs/bench/2026-09-24-host-memory.md)).
      - [x] Sampler component benchmark ([report](docs/bench/2026-09-24-sampler.md)):
        top-k configs 55–60 µs per token (reference 1.2 ms). Found: top-k off sorted the
        whole vocabulary, 29 ms per token; now a nucleus prefix only (top-p 3.2 ms, min-p
        0.18 ms), draw for draw equal to the full-sort reference.
      - [x] Sampler: own vectorized f64 exponential (`expNeg`): top-k-off top-p 3.2 →
        1.2 ms, untruncated 3.3 → 1.05 ms per token, draws unchanged
        ([report](docs/bench/2026-09-24-sampler.md)).
      - [ ] Reports not yet updated: `docs/design/speed.md` (FMA, gmax, cblock).

Work rule (user, 2026-09-24): performance or behavior changes go behind config knobs
with the previous behavior still selectable, not flat-out replacements. Applied to
`--decode-fusion` and `--matvec-accumulation fma|separate` (the FMA change had shipped
without a knob earlier the same day; `separate` reproduces the pre-FMA outputs byte
for byte), and `--sampler-order id|sorted` (`sorted` = the pre-2026-09-24 sampler,
golden-tested against its source). The even-context requirement (key-pair loads) stays
an input restriction, not a knob: it only matters with `--prefill-chunk 0` (chunked
prefill already needs a multiple of 32), and the error message now states it.

Work rule (user, 2026-09-24): also look for CPU-side optimizations in Zig, using
comptime specialization (Zig's strength) where it pays off: resolve per-configuration
choices at compile time instead of branching per token or per element. Only when it
makes sense: find the host cost first (wall clock minus GPU time per step, component
benchmarks), then specialize. Candidates to measure: the engine loop's host time per
decode/verify step (RoPE table writes, logits handling, sampling), tokenizer and
template, request parsing. **Measured 2026-09-24** ([report](docs/bench/2026-09-24-host-overhead.md)):
session host work is 0.5–0.9% of speculative decode and 0.4–0.5% of plain decode
(sampling about 60 µs per token, bandwidth-bound; token handling about 0.1%). Nothing to
specialize yet. Recorded candidates under 1% each: GPU argmax of greedy verify rows,
commit and next draft in one submission, a single-pass greedy scan.

Decision (user, 2026-09-24, D7 in docs/design/speed.md): GPU kernels may be written in
any language or compiler that makes the fastest kernel (C/C++/HIP, LLVM IR, RDNA3
assembly), compiled offline; the no-C++ rule covers the host runtime only. Own machine
code via `VK_KHR_pipeline_binary` is approved for experiments.
- [x] 2026-09-24: hand-scheduled `gemm_f16x` (Q4_0) through a pipeline binary
  ([report](docs/bench/2026-09-24-gemm-f16x-isa.md)): bit-identical on 23 configurations,
  1.20–1.23× the SPIR-V kernel at 512 rows, 1.22–1.40× at 3,328 rows (component races).
- [x] Integration (same day): `--gemm-code native|spirv` (default native, SPIR-V fallback on
  a driver-key mismatch); gates 1–4 passed: model captures and served outputs identical,
  f16 TTFT 3223 tokens 2.83 → 2.49 s (llama-server 3.40 s), 12k 11.25 → 10.14 s (llama 12.3 s).
- [ ] Native kernel follow-ups (ranked by value per effort; gains are estimates unless noted):
  - [ ] Q5_K `ssm_out` (lin_out: 205 ms of the 2,457 ms 3k prefill, now the largest unported
    GEMM phase) and Q4_1 (some ffn_down layers) on the generator: same pipeline, new
    dequantization (Q5_K: 256-weight super-blocks, 6-bit sub-scales, fifth bit plane; Q4_1:
    bit-identity needs care, its SPIR-V rounds twice).
  - [ ] Device-local copy of the `io` fields kernels read at dispatch start (~15 µs per
    dispatch in the first round of a large GEMM grid, measured). For decode:
    `zerv-kernel-chain` (129 dependent norms gated on `io[COUNT]`) measured **0.5–1.0 µs per
    dependent dispatch** host vs device `io` (adjacent pairs; the variant order skews results
    by the clock ramp, so the tool needs a warm-up phase before it is a proper benchmark).
    With an estimated 150–200 `io`-reading dispatches per decode step (not counted), ~0.5–1%
    of decode. Design question: `io` is also where kernels write logits and argmax for the
    host, so only the parameter words can move (a per-submission copy or a fifth binding).
  - [ ] Tail tiles for the M = 5120 shapes (ffn_down, attn_output: 520 tiles on 96 CUs at
    3,328 rows, ~7% lost to the last round; K-splitting is excluded by bit-identity).
  - [ ] Flash attention on the generator (18.6 TFLOP/s today; 13–20% of TTFT at 75k).
  - [ ] Check whether the SPIR-V store burst (15% in our GEMM, finding 3) also costs
    `gemm_f16` (wave64) and flash attention.
  - [ ] Epilogue stores started during the last stage (~1%); dequantization via 16-wave
    WGP-mode workgroups (~2%, halves dequant per WMMA) or VOPD (~1%); X through LDS once
    per workgroup to cut energy at the power cap (speculative).
  - [ ] Productionizing (needs a user decision: parsing RADV's private format at runtime):
    capture the placeholder binary on the running driver, splice our per-ISA code, check
    the ABI fields, bitwise self-test, cache per driver key; so a Mesa update keeps the
    speedup. Until then: after a Mesa update rerun `tools/build_native_gemm.py` and the
    gates. Also consider `--gemm-code native-required` (fail instead of falling back).
  - [x] Research: AMD Tensile's assembly GEMM generator, reading material only
    ([note](docs/research/tensile-gemm-techniques.md)). Candidates from it, cheapest first,
    all bit-identical by construction unless noted: align the hot loop (tried: no effect, noise);
    `WorkGroupMapping`-style tile order (weight reuse in L2/MALL); epilogue interleaved with
    the last stage's WMMAs; persistent kernel prefetching the next tile; load/store density
    tuning; `SourceSwap` (transposed WMMA; bit-identity must be tested first).
- [x] Bug (diagnostics, not results): with `--context max`, `InsufficientVram` printed
  "the model needs 0 MiB" (the requirement was computed after the context fit failed). Fixed:
  the message gives the context-independent requirement and names the limit that ran out
  (free VRAM, the `--vram-budget-gib` cap, or the `--vram-reserve-mib` reserve); checked by
  forcing each case on the real model.
- [x] Competitor HyperQwen, its single-user protocol run against zerv
  ([report](docs/bench/2026-09-24-hyperqwen-protocol.md)): zerv 3 drafts 91.3 / 79.5 tok/s
  (greedy / default sampling) vs llama-server MTP 68.2 / 59.8 here and HyperQwen 120 / 111
  (their README, 3090). Our speculative cycle costs +46% over a plain step (theirs +12%).
- [ ] From that report, single-user decode levers: (1) draft cost — the frequency-ranked draft
  vocabulary below is the largest lever (est. +10–15%); (2) opt-in distribution-exact
  speculative sampling (rejection sampling; est. +20% at T = 1; not seed-identical, needs
  research and a statistical gate).
- [ ] Found on the way, not yet acted on: every kernel reading the host-visible `io` buffer at
  dispatch start waits ~15 µs in its first round (all waves read one host address). Measure for
  decode (hundreds of dispatches per step); candidate for block 17.

Process note (2026-09-24, native GEMM): an interrupted measurement command left its
background zerv (`--context max`, port 18094, 23 GB VRAM) running; the user's server then
failed at startup with `InsufficientVram`. The leftover was stopped. Rule since then: a GPU
server started in the background is stopped explicitly before the turn ends, and `pgrep`
runs before and after every GPU job.

Process note (2026-09-24, host memory): the first two `zerv-prefix-check` runs of the
host-memory change used a stale binary. The default `zig build` step runs the tests
only; tools are installed by their `*-build` steps. Those runs were discarded. Every GPU
tool is now rebuilt with its step before a gate run.

Queue change (2026-09-24, user: "okay fix it", on the context cap): **KV buffers** is
done inside block 17 before 17b.3 closes. The KV cache moves out of the ≤4 GiB state
buffer into per-group buffers; the cap was 29,523 tokens (27,786 with MTP) regardless of
free VRAM. Spec: [model.md, "KV buffers"](docs/specs/model.md). **Done**
([report](docs/bench/2026-09-24-kv-buffers.md)): bitwise identical, 24 of 24 files, with
default and 6-way split buffers; a 37,827-token prompt answered correctly at context
38,400. Max context is now bound by free VRAM (38.7k without speculation and with 8
snapshot slots; 49.4k with 3 drafts, 0 slots, 256-row chunks).

Process note (2026-09-24, 16a): the first fused-attention build sized the activation
arena's score region to zero, but decode and verify attention also write their scores
there (24 × rows × context). Decode then wrote into the split-K partial region that
follows. It was harmless in every run made (the partials, about 20 M words at 512-row
chunks, exceed decode's 4.6 M at 38k and are rewritten by each prefill projection), but
smaller chunk plans could overrun into the decode scratch. The flash gates had run
prefill modes only. Found by reading the layout before the next change; fixed (the
region keeps decode's rows), with a layout test, an init check and mode 0 added to the
gate.

Process note (2026-09-24, 17b): the first multi-row matvec lab run
(`zerv-matvec-rows-bench`, `third_party/spec-lab/rows4-r1.jsonl`) ran while the user's
zerv (pid 1228571, port 18080) was loaded. The pre-run check printed the pid, but the
command did not stop on it. Its **timings are invalid**. Its bitwise results stand (every
row equal to the single-row module; 0 mismatches for Q4_0/Q4_1/Q5_K/Q6_K/F32, R = 1–4),
because they do not depend on contention. GPU runs now stop when that check finds a
process.

Process note (2026-09-24, 17a): the first decode benchmark hung for about 50 minutes.
llama-server with `--spec-type ngram-mod` stopped producing tokens mid-request (no GPU
fault in the kernel log), and `run_serving.py` had a 3600 s socket timeout and no request
deadline. Fixed with stall and total timeouts that record the failure and move to the next
engine, with a regression test
([incident](docs/bench/data/2026-09-24-decode-baseline/serving-decode-v1/INCIDENT.md)).

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

- **Long context** (the 29,504-token cap is gone: KV buffers, 2026-09-24; free VRAM
  binds now). The largest functional gap for agent work.
  - Why: bruh's first request is about 12.4k tokens (30 tools), so a session has about
    17k tokens of room before `400 context_length_exceeded`. The old Ollama setup served
    96–128k ([previous deployment](docs/research/2026-09-23-previous-deployment.md)).
  - Limits today ([long context](docs/bench/2026-09-24-long-context.md)):
    - ~~FP32 KV in one state buffer under 4 GiB~~ (fixed: KV buffers). FP32 KV at
      128 KiB/token remains.
    - ~~The materialized prefill score buffer~~ (fixed: fused prefill attention, 16a).
    - VRAM knobs so far: `--embedding-memory host` (default, 682 MiB),
      `--prefix-cache-memory host` (1.2 GB at 8 slots), `--prefix-cache-slots`,
      `--spec-draft 0`, `--prefill-chunk`.
  - Scope: KV formats f16 / q8 / q4 as explicit options, with quality measured against
    FP32 and against llama's f16/q4_0 KV (being done as block 17c); `--context max`;
    VRAM accounting (snapshots, context).
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
- ~~Image input~~: now block 19 (active, 2026-09-26).
- **More than one sequence at a time.** Today requests are serialized (a bounded wait
  queue). Research the VRAM, KV and scheduling tradeoffs of parallel sequences or
  batched decode, and what bruh actually sends concurrently.
  - Evidence (2026-09-24): decode is weight-bandwidth bound (16.7 of 20 ms per step); the
    multi-row verify path already does 3/4/5 rows in 22.7/25.0/28.8 ms against 20.0 ms for
    one, and its kernels are row-for-row bit-identical to single-row. HyperQwen (3090)
    reaches ~400 tok/s aggregate at 8 users and ~1,035 at 64.
  - Missing: per-slot state (KV region, DeltaNet/conv state, position) with a per-row slot
    index in the attention, DeltaNet and conv kernels; recorded commands per batch size; a
    step scheduler with join/leave; per-request sampling and streaming; memory split per
    slot. Gate: each request byte-identical to running it alone.
  - Stage 2: chunked prefill interleaved with decode, per-slot prefix cache, speculation
    with a few users (HyperQwen: speculation wins below ~8 users).
- **Prompt-lookup drafting** (drafts taken from the prompt/context): HyperQwen reaches
  381 tok/s when the answer quotes its input, relevant for a coding agent that edits files.
  llama's n-gram mode stalled in 17a (incident); needs its own research. Lossless like MTP.
- **8-bit KV** (fp8/q8): HyperQwen serves 150k context with fp8 KV; zerv has f32/f16 today
  (already listed under long context as "Open: q8 KV").
- **Small API and usability items:**
  - `logprobs` / `top_logprobs` and `n > 1` (rejected with 400 today).
  - Model name: zerv accepts only its alias (`qwen3.8-27b`); bruh's default is
    `qwen3.8:27b` (Ollama naming). Decide whether to accept several aliases, or
    document `OPENAI_MODEL` / `--alias`.
  - The default port 8080 is taken on this host (`adp-work-server`); decide whether
    to document it or pick another default.
  - The startup free-VRAM check is a snapshot: two servers starting at the same moment
    can both pass it.
- **AMD Instinct backend and big-model serving** (user direction 2026-09-26: AMD-first).
  Research only until the user makes it the active block.
  - Research note and dependency-boundary proposal (awaiting approval):
    [Instinct backend](docs/research/2026-09-26-instinct-backend.md). The runtime would
    use KFD ioctls with no ROCm userspace; LLVM's assembler only as a development tool.
    The KFD runtime can be built on the local RX 7900 XTX first.
  - Research TODOs, in order: platform → Qwen3.8-27B on MI300X → Qwen3.5 MoE up to 397B →
    GLM-5.3-Flash → Qwen3.8-2.4T
    ([big-model TODOs](docs/research/2026-09-26-big-model-serving-todos.md)).
  - "Qwen 3.9" does not exist officially; `QwennAI/Qwen3.9-*` is a fake repository with a
    pickle. Do not load it.

## Verification debt (no block yet)

- Long positions: FP64 oracle tensors cover prompts up to 562 tokens. At 29k tokens the
  evidence is output-text equality with llama-server at full FP32 (3/3 requests), not a
  tensor-level gate.
- The real device-loss path of the server (503, exit 3) is tested only with a simulated
  engine; a real GPU loss was not provoked on purpose.
- Block 14 (f16 prefill, opt-in): the worst-token quality gate is unmet (see Parked).

## Parked

- [ ] **16 · prefill speed** — parked 2026-09-24 for block 17 (user pivot to tok/s).
  [Design](docs/design/speed.md).
  - **Done:** 16b lab rounds 1–2 ([report](docs/bench/2026-09-24-gemm-f16-lab.md)) and
    the wave32 f16 GEMM for Q4_0 ([report](docs/bench/2026-09-24-gemm-f16x.md)).
    - Bitwise-identical end to end.
    - f16 TTFT at 3223 tokens: 3.04 s against llama default's 3.39 s.
    - 12k: 12.9 s against 12.46 s.
  - **Open, in order:**
    - **Prefill kernel opportunities** (user-approved list, 2026-09-24): 3,223-token f16
      prefill is 2.75 s GPU; shares ffn_in 32%, ffn_down 21%, lin_in 14%, delta 9.5%,
      lin_out 7.4%, attention 4.4% (about 40% of TTFT at 75k), attn_in 4%, conv 3.3%
      (`zerv-model-profile MODEL 4096 512 3223 4 f16`). Gains are estimates.
    - [ ] 16b: Q4_1 and Q5_K on the wave32 structure (ssm_out, Q5_K, is 203 ms of the
      3223-token prefill, about 1.7× slower per FLOP than Q4_0: ~4–5%); the 128-row plan;
      smaller plans; **tail fill for the M = 5120 shapes** (320 tiles on 96 CUs, the last
      of ~3.3 rounds a third full: ffn_down 8.9 ms per matrix against 6.8 for ffn_gate at
      equal FLOPs, ~5%); then the FP32 path. Device-keyed tile table once a second
      configuration is measured. GEMM efficiency is ~63% of WMMA peak per clock; the known
      remedies need scheduling control SPIR-V lacks (D5).
    - [ ] WMMA flash attention in the f16 prefill mode (13–20% of TTFT at 75k; tracked in
      17c, "Prefill at very long context").
    - [x] 16a fused (flash-style) prefill attention (done inside block 17, 2026-09-24):
      TTFT at 37.8k tokens 46.0 s against llama's 46.6 s (fp32 74 s; before: 70 / 112 s).
      [Report](docs/bench/2026-09-24-flash-attention.md). Further kernel tuning is open
      (18.6 TFLOP/s against 31–37 for the GEMMs).
    - [ ] 16c fusions (norm, swiglu, conv, gate: ~7% of prefill, ~3–4% to gain),
      16d chunkwise DeltaNet (delta 9.5% of prefill, ~5–6%), 16e conv and chunk size.
    - [ ] Thermal load test (`tools/thermal_log.py`, command in
      [hardware.md](docs/hardware.md)). The card runs GEMMs at its 110 °C junction
      limit; the user is adding case fans. Re-measure after the change.
  - Open questions Q1 and Q2 (design doc) are unchanged.

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
