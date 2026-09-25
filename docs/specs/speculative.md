# Speculative decoding with the MTP layer — block 17b (specified 2026-09-24, before implementation)

Research: [speculative-mtp.md](../research/speculative-mtp.md). Design log:
[speed.md](../design/speed.md).

## Contract

- **Lossless, bit for bit:** with speculation on, the emitted token sequence is identical
  to speculation off for the same request and seed, for greedy and sampled decoding.
  - Mechanism: verification rows are computed with arithmetic identical to single-token
    decode (below), and acceptance is by **sample matching**.
  - Row i's token is sampled from its logits with the random numbers non-speculative
    decode would use at that position. A draft is accepted while the sample equals it; the
    first mismatch (or the row after the last draft) supplies the step's final token.
- Speculation changes **only speed** and VRAM. Draft quality affects acceptance, never
  outputs.
- Knob: `--spec-draft N` (0 = off; 1..4 drafts per step; `Options.spec_draft`).
  Speculation is a configuration choice, never a requirement: with 0 no MTP weights,
  MTP cache, verify scratch or verify commands are allocated. Decode then runs exactly
  as without this block (bitwise, gate 1).
  - Its cost: MTP weights (0.265 GB), the MTP KV cache (8 KiB per context position, FP32),
    verify scratch and N + 1 logit rows.
  - **Default (decided 2026-09-24, after gates 1–4 passed): 3 drafts with the adaptive
    policy.** It is lossless and was faster than no speculation, and than every
    llama-server configuration, in every measured case (greedy and sampled decode-v1,
    serving-v2, 38k), at equal TTFT ([report](../bench/2026-09-24-speculative.md)).
    `--spec-draft 0` turns it off: no MTP memory, plain decode, bitwise as before.

## Target side: N-row verification (N = drafts + 1 ≤ 5)

The main model processes rows `[t, d1..dn]` at positions `p..p+n` in one command buffer.
Per row, every operation is the decode step's operation with the same module, the same
per-row code and the same summation order:

| Operation | Kernel | Rows |
| --- | --- | --- |
| Embedding | `embed_b` (tokens from `io.tokens`) | grid y. Exact dequantization; equal to `embed` bit for bit (gate 1) |
| RMS norms | `norm` (the decode module) | grid x = row, `stride` = width (already row-capable) |
| Projections | `matvec_rows.comp` | count = N input rows; per row the arithmetic of `matvec.comp` |
| q/k norm, RoPE, KV write | `qkprep` | row = grid y; position `io.position + r`; RoPE row r at push `rope + 64 r` |
| Attention (3 passes) | `attn_scores/pv/combine` | row = grid z; keys `0..p+r`; per-row scratch |
| Conv | `conv` | loops rows in order; window of the state plus earlier rows |
| DeltaNet | `delta` | loops rows in order, state in registers |
| SwiGLU | `swiglu` | `n = ffn · N` (elementwise) |
| Output head | `matvec_rows` into a host-visible logits buffer | N logit rows |

**Multi-row projection modules** (`matvec_rows.comp`, implemented 17b.1):

- One SPIR-V module per exact row count 1..5 and format (`rows<count>_<format>.spv`; the
  host validates count == ROWS). A runtime count made ACO reload the X descriptor per
  row and select every accumulator, and it was 1.45× slower at 5 rows
  ([measured](../bench/2026-09-24-spec-verify.md)).
- A workgroup computes G consecutive weight rows (`matvec.rows_groups` = 1, 2, 2, 3, 2
  for counts 1..5; CB = 4, 2, 2, 2, 4 blocks per chunk). Lane l does lane l's
  single-row work for each of them, so an X load serves G weight rows. Rows past M
  recompute row0 and are never stored.
- `RowsPipeline.init(format, aligned, max_count, ...)` creates the kernels for counts
  1..max_count. Its geometry holds the dispatch for every count.

**Decode itself uses these row-capable kernels with N = 1.**

- The decode modules change: the kernels gain the row index, the RoPE table pointer and
  the row loop.
- Gate 1 therefore requires the decode captures and logits to be **bitwise identical**
  to the pre-change build on the oracle cases (mode 0).
- The single-row matvec modules stay byte-identical, and `matvec_rows` must equal them
  per row.

**State.** `conv` and `delta` take `rows` and a `commit` flag.

- The verify pass runs with `commit = 0`: it reads the stored state and never writes it.
- After acceptance, a **commit pass** re-runs both over the first `m` rows (1 ≤ m ≤ N)
  with `commit = 1`. It writes the state after row m − 1, and it rewrites that row's
  outputs with identical values.
- Plain decode is `rows = 1, commit = 1`.
- The arena reuses its activation regions across layers, so a commit pass cannot read a
  layer's inputs from them afterwards.
  - In verify, each linear layer's lin_in outputs (`mixed`, `z`, `alpha`, `beta_raw`)
    and its conv outputs go to that layer's slot of a **spec scratch** region (48 layers
    × R rows × 26,720 words ≈ 25.6 MB at R = 5). Conv and delta read them from there.
  - The commit re-runs conv (reading `mixed`) and delta over the same slots. It has two
    phases: all conv dispatches, one barrier, all delta dispatches.
- Attention K/V written for rows ≥ m stay in the cache. Every position is written again
  before any row reads it (reads cover keys ≤ own position).

**Layout.**

- The activation arena's decode scratch (attention `amax/apart/asum`) holds R = 5 rows.
- Verify uses rows 0..N−1 of the existing activation regions (capacity ≥ 512 rows).
- Logits: a host-visible buffer of R × vocab floats.
- The final norm writes `hn` rows (the MTP input `h`).

## Draft side: the MTP layer

- **Semantics:** llama.cpp `graph_mtp` exactly (research note), in FP32. The MTP KV
  cache position q holds `(h_{q−1}, x_q)`, with h = the main model's `output_norm`
  output. Its K/V live in their own buffer: one attention layer, 4 KV heads × 256, FP32.
- **New operator: Q8_0 matvec** (`eh_proj`).
  - Block: f16 d, then 32 int8.
  - `w = d · q`, exact in FP32; the dot product follows matvec.comp's lane and summation
    structure.
  - A new format branch; existing modules must stay byte-identical.
- **Passes:**
  - *Prompt catch-up:* after each prefill chunk, the MTP runs on the chunk's rows
    `(h_{q−1}, x_q)` (h of the chunk's first row = the previous chunk's last h, zero at
    position 0). This uses a batched path (GEMM kernels at the model's prefill
    precision) and computes no logits.
  - *Step:* one MTP pass over the rows accepted in the previous verify (their true `h`),
    plus one new draft row `(h_{p+m−1}, y)`. Logits are computed only for the last row.
  - *Further drafts:* a single row `(h', d_i)` at the next position, where h' is the
    MTP's own `shared_head_norm` output.
- **Draft choice:** the argmax of the draft logits, computed on the GPU (no logits
  readback).

### Draft vocabulary (knob, 2026-09-24; specified before implementation)

- **Measured motive:** a draft costs 1.55 ms per drafted token (`zerv-mtp-check`
  timing: 1.59 / 3.12 / 4.65 / 6.18 ms for k = 1..4 after a step), 16% of a 3-draft
  cycle (4-row verify + commit 25.0 ms). It reads 253 MiB of MTP layer plus the
  995 MiB Q6_K output head, 1.31 GB, 1.42 ms at 920 GB/s. The head matvec already runs
  at 916 GB/s ([decode baseline](../bench/2026-09-24-decode-baseline.md)), so only
  reading fewer rows can make it cheaper.
- **Knob:** `Options.draft_vocab` (server `--spec-draft-vocab N|full`, default full =
  the previous behavior). The draft head computes logits for token ids `0..N-1` only:
  the same single-row matvec over the first N rows of `output.weight` (rows are
  contiguous per id), and the draft choice is the first index of the maximum over
  those N logits. The drafter's probability becomes the softmax probability of that
  maximum within the N ids. N = 0 or N = vocab means full.
- **Why a prefix of ids:** the byte-level BPE vocabulary numbers merged tokens in merge
  order, and merges are learned by pair frequency, so low ids are the frequent pieces
  of the tokenizer's training data. Measured coverage on our text (llama-tokenize, ids
  ≥ N as a share of tokens): decode-v1 greedy outputs, N = 32,768: 1.0–10.4%; N =
  65,536: 0.2–3.7%; N = 100,000: 0–0.2%; the long-v2 prompt (75.5k tokens): 6.6%,
  2.2%, 0.2%. Related published designs restrict the draft head to a frequency-ranked
  subset (FR-Spec, arXiv 2502.14856; EAGLE-3's reduced draft vocabulary). Cited from
  memory, not re-checked in this session; this design stands on our own measurements.
- **Consequence:** tokens at ids ≥ N can never be drafted. That includes every control
  token (`<|im_end|>` 248046, `</think>`, `<tool_call>`), so each occurrence costs one
  missed draft. Verification is unchanged, so output is unchanged (greedy: identical;
  sampled: the same draws, since sampling matches drafts against the sampled token).
  Only acceptance and speed change.
- **Gates:** (1) `zerv-mtp-check` with N: every dumped draft logits vector equals the
  first N values of the full-vocabulary run bitwise, and each draft is the first
  argmax of that prefix; scenario C passes. (2) Serving: outputs identical to
  `--spec-draft 0`; acceptance and tok/s per N on decode-v1 against full, with a dated
  report.
- **Result (2026-09-24, [report](../bench/2026-09-24-draft-vocab.md)):** both gates
  pass. Three drafts cost 4.66 / 3.03 / 2.09 / 1.63 ms for full / 131,072 / 65,536 /
  32,768. decode-v1 at N = 65,536: +7–9% tok/s with acceptance nearly unchanged. On a
  multilingual workload the prefix is a poor frequency proxy (Chinese: 82% of tokens at
  ids ≥ 65,536) and acceptance collapses: Chinese falls below plain decode. **Default
  stays full**; N = 65,536 is an opt-in for English/code deployments. Follow-up: a
  frequency-ranked subset (row indirection plus an index → id map).

## MTP runtime design (17b.2, decided before implementation)

- **Knob:** `Options.mtp` (needs `verify_rows` ≥ 2; drafts n ≤ `verify_rows − 1`). Off:
  no MTP tensors, regions, kernels or commands exist.
- **Weights:** the 15 `blk.64.*` tensors join the placement (params into bank 0). The
  projections use the multi-row pipelines (`eh_proj` through the Q8_0 rows module). The
  draft head is the single-row decode pipeline of `output.weight`.
- **State:** the MTP KV is attention cache index 16, after the trunk's 16, in the KV
  buffers ([model.md, "KV buffers"](model.md)). `hp`
  (the pending `h`, 5120 words) follows the conv state. `reset` zeroes it with the
  recurrent state, and snapshots include it (`Model.snapshotBytes`).
- **Activation arena** (MTP only):
  - `cat` (R × 10240; e then g per row);
  - `mh` (R × 5120: h inputs of a pass);
  - `mo` (R × 5120: the MTP's `h'` outputs);
  - `hrows` (prefill rows × 5120: all-row final norm of a prefill chunk);
  - `mlogits` (vocab);
  - argmax partials.
- **io block:**
  - `mtp_pos[R + N]` (positions), `mtp_tok[R]` (pass-1 tokens), `draft[N]` (argmax
    output), `hsrc[2]` (row copy sources), `mtp_rope[(R + N) × 64]`.
  - The trunk's words are unchanged.
- **New kernels** (`model.comp`; the other modules stay byte-identical):
  - `rowcopy`: copies words from a source whose word offset (bit 31: state arena) is
    read from io, to a static destination (a flag selects the state arena);
  - `argmax_a` / `argmax_b`: two-phase first-index argmax of the draft logits into an
    io word.
- **A pass of n ≤ R rows** (positions `io[mtp_pos] + r`, RoPE rows `mtp_rope + 64 r`):
  1. `mh[0] ← src0`, `mh[1..n) ← src1` (row copies);
  2. `embed_b` (tokens from io);
  3. per row, `norm` with `enorm` into `cat[r][0..H)` and with `hnorm` into
     `cat[r][H..2H)`;
  4. `eh_proj` into `x`, then the trunk's attention-layer sequence (verify kernels, MTP
     weights, KV index 16), `post_attention_norm`, FFN;
  5. the final norm with `shared_head_norm` into `mo`.
- **Chained passes** (1 row) read h from `mo` (the previous pass's last row) and the
  token from `draft[k − 1]`.
- **Commands:**
  - `draft[m][k]`: a pass of m rows, head plus argmax into `draft[0]`, then k − 1 chained
    passes, each with head plus argmax;
  - `catchup[n]`: a pass without head;
  - `save_hp`: row copy into `hp`.
- **Plain step with the MTP layer** (2026-09-24; the session steps when no draft fits:
  the last allowed token or the context end): `step(t)` first runs the pending rows plus
  t's row through a catch-up pass (`catchup[m]`, the rows a draft's first pass would
  run, KV only), then the trunk step; the step's final-norm row becomes the pending h.
  Before this, a step dropped the pending rows and left the pending h stale (speed only).
  Gate: `zerv-mtp-check` scenario C, drafts and probabilities after 1 step, after a
  2-row commit and a step, and after 2 steps are bitwise equal to the draft/verify/commit
  path; the old step fails all three
  ([data](../bench/data/2026-09-24-speculative/step-mtp/)).
- **Pending catch-up.** After `commit(m)`, rows `p+1..p+m` (the accepted drafts, then the
  new token) have h = verify `hn` rows `0..m−1`. The next `draft` runs them as its first
  pass.
  - A prefill first flushes them: m − 1 rows, then `hp ← hn[m−1]`.
  - After a prefill, the first pass is one row with h = `hp`.
  - `reset` and `loadSnapshot` clear the pending rows.
- **Prompt catch-up (batched, 2026-09-24; replaced the host-driven ≤ R-row passes):**
  each prefill chunk writes the all-row final norm into `hrows`, and the same prefill
  command then runs the catch-up for all B rows (`recordCatchupBatch`). Only what the MTP
  KV depends on runs; no logits:
  - `hrows` row 0 ← the pending h (`hp`); `embed_b` writes the tokens' embeddings into the
    first halves of `cat`; `copy2d` puts h rows 0..B−1 into the second halves, so MTP row
    i holds (h of row i − 1, token i);
  - `enorm` / `hnorm` in place, `eh_proj` with the Q8_0 GEMM (`gemm.comp` FORMAT 8) at
    the prefill precision, `attn_norm`, K and V projections, then `qk_b` writes KV cache
    16 (query heads run on stale rows: finite, unused);
  - `hp ← hrows[B−1]`.
  - Effect: TTFT with speculation equals TTFT without it (3223 tokens: 3.57 → 3.08 s;
    [report](../bench/2026-09-24-flash-attention.md), section 2). Gate: `zerv-mtp-check` and the FP64
    reference (below) re-passed; its h rows come from the prefill.
- **Gate 2 mechanism** (`zerv-mtp-check`):
  - The tool dumps the h rows and tokens zerv fed to the MTP (prefill `hrows`, verify
    `hn`), and zerv's `mo` rows and draft logits per chain step.
  - `tests/reference/mtp_reference.py` recomputes `graph_mtp` in FP64 from those same
    inputs, with the chain teacher-forced on zerv's drafts.
  - Bound: normalized L2 ≤ 1e-5 for h' and logits (FP32 accumulation over K ≤ 17408; the
    trunk's attention layers measure ≈ 1e-6 to 1e-7). Drafts must equal the FP64
    argmax, except where its top-2 margin is < 1e-4 relative.

## Engine loop (per step, `spec_draft = n`)

1. Draft `d1..dn` from `(h, t)` and the chained MTP (n passes).
2. Verify `[t, d1..dn]`, then read the N logit rows.
3. Host: sample rows in order with the request's sampler state until a mismatch;
   m = rows consumed.
4. Commit m rows (conv/delta state), then position += m.
5. Emit the accepted drafts plus the final sampled token. Stop conditions (EOS,
   max_tokens, stop strings) apply token by token, exactly as without speculation: tokens
   after a stop are discarded and the state is committed only up to the stop.
6. The next step's first MTP pass re-runs the accepted rows with their true h.

- **Near the context end,** n is clamped so that p + n < context.
- **As implemented** (`session.Generation`, a backend with `speculative/draft/verify/
  commit`; `--spec-draft N` in the server):
  - k = min(N, tokens still allowed − 1, context room); k = 0 falls back to `step`.
  - Row i + 1 is used only if the token sampled from row i equals draft i + 1;
    otherwise `commit(i + 1)`. The sampler call sequence is exactly the
    non-speculative one.
  - When the generation ends inside verified rows (EOS, stop, length), the consumed
    rows are committed and recorded in the prefix cache, so the model state matches the
    non-speculative run. An error with an uncommitted verify invalidates the prefix
    cache.
  - CPU gate: `tests/session.zig` checks the loop on a hash "model" with wrong drafts,
    greedy and seeded top-k/penalty sampling, EOS and the length limit inside verified
    rows, and context clamping, for 1–4 drafts. Output and processed history equal the
    non-speculative run.
- **Counters** (`session.SpecStats` in the result; summed in `/metrics`, one log line per
  request): verifies, drafts produced, drafts verified and decided, drafts accepted.
  Drafts left over when a generation ends inside verified rows count as neither
  verified nor accepted, so accepted / verified is the acceptance rate. The benchmark
  harness records per-request deltas (zerv) and llama-server's `timings.draft_n` /
  `draft_n_accepted`.
- **Prefix cache** (decided 2026-09-24): a snapshot stores the committed state plus the
  MTP's pending h (`snapshotBytes` adds hidden × 4 bytes), and `saveSnapshot` flushes
  the pending MTP rows first. The MTP KV below a restore point stays valid in KV cache 16,
  like the trunk's KV. Every committed position's MTP row was written from its true h:
  by a draft's first pass, a step's catch-up or the prompt catch-up; draft-chain rows
  written from h' are rewritten when their positions are committed. Nothing is re-run on
  restore. Outputs never depend on it (sample matching); only acceptance does.

## Gates

1. **Verification ≡ decode (component and model):**
   - `matvec_rows` = `matvec` per row, bitwise, on all 48 matvec fixtures plus the real
     shapes, for R = 1..5 (**passed**: `gpu-test`, and `zerv-matvec-rows-bench` on every
     layer of 8 roles);
   - decode captures and logits (verify_model mode 0, default and long oracles) bitwise
     identical before and after the row-capable kernels;
   - an N-row verify of the tokens decode produced gives logits rows bitwise equal to the
     N decode steps, and the committed state after m rows is bitwise equal to m decode
     steps (new `zerv-spec-check` tool, several positions including chunk edges of the
     attention split and 1..5 rows).
   - **Passed** 2026-09-24 ([report](../bench/2026-09-24-spec-verify.md)). Verify +
     commit then cost 20.6 / 21.4 / 24.1 / 27.1 / 32.4 ms for 1..5 rows, against a
     20.0 ms decode step.
2. **MTP layer:**
   - an FP64 reference of `graph_mtp` on real weights and oracle h/token rows
     (tests/reference; normalized L2 bound as in the model spec);
   - acceptance on the decode-v1 workload in the range llama.cpp measures (a large
     shortfall means a semantic error).
3. **Lossless serving:**
   - `run_serving.py` with `--spec-draft` 0 and N on serving-v2 and decode-v1;
   - outputs byte-identical (greedy), plus sampled requests with fixed seeds.
4. **Speed:** decode-v1 and serving-v2 against llama-server default and `draft-mtp`
   1–4, with raw data and a dated report. A speed claim needs gate 3 passed first.
