# Concurrent sequences — block 18 (specified 2026-09-24, before implementation)

Research: [concurrent-sequences.md](../research/concurrent-sequences.md). Baseline:
[concurrency report](../bench/2026-09-24-concurrency-baseline.md). User decisions (2026-09-24):
everything batch-invariant for multiple users; both batched-projection arithmetics available as
knobs; the scheduler is general-purpose serving code over a narrow model backend; hardcore
performance on the GPU and the host.

## Knobs

| Knob | Default | Meaning |
| --- | --- | --- |
| `--parallel N` | 1 | at most N sequences decode together (N = 1: today's behaviour, byte for byte) |
| `--decode-precision f32\|f16` | f32 | batched decode projections: `f32` = the FP32 multi-row kernels (exact: equal to single-stream decode); `f16` = WMMA with f16 inputs and f32 accumulation (opt-in; its own arithmetic, batch-invariant within itself) |
| `--kv-page-tokens` | 128 | KV page size, a multiple of 128 (whole scores chunk pairs and flash key tiles); `context` = one page, the pre-paging layout. Specialization constant 0 of the KV kernels. Default 256 until 2026-09-24, changed on [measurement](../bench/2026-09-24-paged-kv.md) |
| `--kv-pool` | all free memory after the fixed parts | KV pages shared by all sequences (replaces the per-sequence `--context` budget; `--context` stays the per-request maximum) |

The previous behaviour stays selectable (`--parallel 1`, `--decode-precision f32`).

## Contract: batch invariance

For a fixed precision mode, a request's logits, sampled tokens and output bytes do not depend on
which other requests share its steps, their number, arrival times or lengths, or preemption and
resumption. In `f32` mode they equal today's single-stream output (the same seed, the same
sampler, speculation on or off).

Why it holds: every per-row computation is independent of the other rows. The FP32 multi-row
projection kernels are row-for-row bit-identical to the single-row module (measured to 16 rows;
the 5-row K-quant `GROUP = 4` exception is never used); attention, conv and DeltaNet run per
sequence with the same arithmetic; sampling is per request. WMMA rows are independent too (each
output element is its own dot product in a fixed k order).

Anything that would break it is excluded: split-K or any reduction across rows; batch-size
dependent kernel selection within one precision mode (the per-count module table must give every
row the same arithmetic — to be verified per count); shared random state.

## Memory

- **Slot** (one per running sequence, up to N): its DeltaNet + conv state (149.6 MiB), MTP hidden
  row, its KV page table, its position and pending speculative state.
- **KV pool:** pages of `P` tokens holding K and V of all 16 attention layers (+ MTP when on);
  f32: 16 MiB per 128-token page, f16: 8 MiB. Layout per page and layer: K `[kvh][dim][P]`, V
  `[kvh][P][dim]` (today's per-layer layout, in P-token pieces). A slot's page table maps its token
  t to page `table[t / P]`.
- **Addressing (18b.1, exact):** a KV buffer holds `per_buffer` attention layers (as today) for
  every physical page; page `q` starts at element `q * pstride`, `pstride = per_buffer * 2 * 4 *
  256 * P`; layer piece `l = attention % per_buffer` has K at `l * 2048 * P` and V at `l * 2048 *
  P + 1024 * P` within the page. Token `j` of a sequence lives in physical page
  `ptab[j / P]` at in-page index `jj = j % P`: K element `(kh * 256 + d) * P + jj`, V element
  `(kh * P + jj) * 256 + d`. The page table is `u32` words in the device-local activation arena
  (never in the host-visible `io`: host reads at dispatch start measured 0.5–15 µs), written by a
  copy when it changes. P is a multiple of 128, so a scores chunk pair (128 keys) and a P·V chunk
  (64 keys) never straddle a page. With one sequence and the identity table (`ptab[i] = i`) the
  memory equals today's and every value is unchanged (addresses only): the existing gates
  (`verify_model.py`, spec-, prefix-, mtp-check, gpu-test) must pass byte-identically.
  **Implemented 2026-09-24** ([report](../bench/2026-09-24-paged-kv.md)). All gates are
  byte-identical at P = 128, 256 and `context`. `KV_PAGE` is specialization constant 0; its
  machine code equals the former `#define`. At 30k, decode against the pre-paging layout is
  +0.06% (f32 KV) / +0.07% (f16 KV) at P = 128, and +0.18% / +0.39% at 256. At 4k both are
  0.3% faster. The page table itself costs nothing. P = 128 is the default.
- **Admission (18b/18c):** a request is admitted when a slot is free and the pool can reserve
  pages for its whole possible length (prompt + `max_tokens`, capped by `--context`); pages are
  handed out as the sequence grows, so a running request never runs out. Requests that do not fit
  wait (bounded queue).
- **Preemption (18d):** reserving the worst case wastes pages; 18d replaces it by on-demand
  allocation with preemption (host copy of the slot's state and pages, or recompute) — policy
  decided there, under the same batch-invariance gate.

## Model backend interface (model package)

- `openSlots(n)`, per slot `reset`, `prefill(slot, tokens)` (chunked), `saveSnapshot /
  loadSnapshot(slot, …)`.
- `decode(batch)`: rows `(slot, token)` (one row per sequence; with speculation, a slot's rows are
  consecutive: token + drafts) → logits rows. One recorded command per row count B (1 .. max);
  per-row tables in `io`: slot, position, token, RoPE row, page-table base.
- `commit(slot, m)` for speculative rows, `draft(slots, …)` for MTP per slot (18d).
- A memory/cost model: fixed bytes, bytes per slot, bytes per page, and the measured step cost
  per row count (for the scheduler).

Kernels that change: `qkprep` (per-row K/V write to the row's page and position), the five
attention passes (per-row page table and length), `conv` / `delta` (per-row slot state; a slot's
consecutive rows run sequentially as in verify), `embed_b`, RoPE rows; projections use the
multi-row pipelines (f32) or a new small-M WMMA path (f16). Prefill runs into a slot's pages.

### 18b.2 design (exact; specified 2026-09-24, before implementation)

- **Slots in the state arena.** `Model.Options.slots` (N ≥ 1). Slot `s`'s recurrent and conv
  state (and the MTP's pending h) start at word `s * slot_words`, where `slot_words` is slot 0's
  size rounded up to 64 words. Slot 0 is today's layout exactly. N > 1 requires the MTP off until
  18d (`error.MtpNeedsOneSlot`).
- **KV pool.** `Model.Options.kv_pages` physical pages (0: N × pages for `context`). Each slot has
  a page table of `ptabWords(context)` entries at `Act.ptab + s * ptabWords(context)`. The default
  table of slot `s` maps logical page `i` to physical page `s * pagesFor(context) + i` (a static
  split; with N = 1 that is the identity). `setPages(slot, pages)` replaces a prefix of a slot's
  table with copies checked against the pool (each entry < `kv_pages`). The scheduler owns
  allocation (18c). `context` stays the per-sequence maximum.
- **Slot entry.** 4 io words: position, page-table offset (words from `Act.ptab`), state offset
  (words), and a reserved word. The single-slot entry is at io word 68 (`io.slot`). The batch table
  `io.batch(rows)` holds one entry per batch row, at most `io.batch_max` (32).
- **Kernels.** The decode family (`qkprep`, `attn_scores`, `attn_gmax`, `attn_pv`, `attn_cblock`,
  `attn_combine`, `conv`, `delta`) gains two push words: `slots` (io word of the entry, 0 = none)
  and `slot_rs` (entry stride per row).
  - `slots = 0` is today's behaviour exactly: page-table offset 0, state offset 0, no io read.
  - `slot_rs = 0`: every row uses the entry at `slots` (its offsets), position `io[pos] + r`,
    and conv/delta scan the rows in order (decode and verify of one slot).
  - `slot_rs = 4`: row r uses the entry at `slots + 4 r` for its position, page table and state.
    conv/delta run one row per workgroup (grid y = row) with commit. This is the batched decode;
    rows must belong to distinct slots.
  - The prefill family (`qk_b`, `flash`, `conv_b`, `delta_b`) gains `slots` (0 or `io.slot`) for
    the page-table and state offsets.
  - Per row, the arithmetic is the same source expressions in the same order; only addresses and
    loop bounds come from the entry.
- **Commands.**
  - With N = 1, every command is recorded with `slots = 0`, byte for byte today's behaviour.
  - With N > 1, the single-slot commands (prefill plans, step, reset) use `slots = io.slot`, and
    the host writes the selected slot's entry.
  - Batched decode commands exist for B = 1 .. `Options.batch_rows`. Projections use the
    multi-row pipelines, in groups of at most `matvec.max_rows` rows; the fused SwiGLU multi-row
    module is used where it supports the group's count. Logits go to rows 0..B−1 of the
    host-visible verify logits.
- **API.**
  - `select(slot)` makes a slot current: `reset`, `prefill`, `step` and the snapshots act on it,
    and the model keeps each slot's position.
  - `resetSlot(slot)`.
  - `decodeBatch(rows)` takes rows of (slot, token), with distinct slots at their current
    positions, and returns B logits rows. Those rows are borrowed until the next model call; each
    slot's position advances by 1.
  - `setPages(slot, first, pages)`.
- **Gate `zerv-batch-check`** (real model, f32 and f16 KV):
  - Setup: K sequences with different prompts. Each is prefilled and decoded alone in slot 0 for
    T teacher-forced steps (the reference logits), then prefilled in its own slot and decoded in
    batches.
  - Batch variations: sequences join late and leave early, rows are permuted, B spans 1..8
    (above 5 the projections run in groups), and slot pages are permuted non-identity tables.
  - Every logits row must be **bitwise** equal to the reference, and the single-slot operations
    in slot s must equal slot 0.
  - It also times `decodeBatch` per B and context.
- **Unchanged-at-N=1 gate.** `verify_model.py` default and long oracles byte-identical;
  spec-, mtp- and prefix-check; the spill gate.

## Scheduler (serve package, general purpose)

### 18c design (exact; specified 2026-09-25, before implementation)

A first serving step that keeps `session.Generation` (sampling, stops, tools, streaming)
unchanged per request. That change to the host state machine (below) waits until measurements
show host overhead matters.

- **Knob `--parallel N`** (1..`layout.max_slots`, default 1).
  - N = 1 is today's server byte for byte: one generation under the server lock, MTP and the
    prefix cache as configured.
  - N > 1: the model gets N slots and batch commands for 1..N rows, and the KV pages are split
    statically (each slot owns the pages of `--context` tokens). `--context max` fits N sequences.
  - N > 1 turns speculative decoding and the prefix cache off. Both need per-slot versions
    (18d). The startup log says so.
- **HTTP:** with N > 1, up to N generations run at once, bounded by a semaphore, and the wait
  queue (`--max-waiting`, 503 beyond it) is unchanged.
- **Batcher** (`serve/batcher.zig`), generic over a narrow backend: `select`, `reset`,
  `prefillChunk(tokens) → (consumed, logits?)`, `decodeBatch(rows)`. It owns the model on one
  scheduler task.
  - Each generation holds a slot. Its session backend's `reset`, `prefill` and `step` submit an
    operation and wait for the result.
  - Every decode row's logits (and the final prefill logits) stay valid until the generation
    calls `sampled()`. Before that, no later batch (or prefill chunk) overwrites them.
  - `session.Generation` calls `backend.sampled()` right after sampling, when the backend has
    it, so slow client writes never hold the GPU.
- **Scheduling loop:**
  - A decode batch runs when steps are pending and no row of the previous batch is still held,
    and either every decoding slot has submitted or `gather` (2 ms) has passed since the last
    row was released.
  - Prompts are prefilled in chunks (the model's plans), FCFS. When both kinds of work are
    ready, chunks and batches alternate, so a long prompt delays running sequences by one chunk
    per step, not by the whole prompt.
- **Cancellation** (client gone, drain deadline): a waiting generation withdraws its operation.
  A row already in flight completes and is discarded; the slot is freed after that. Engine
  errors fail every waiting operation. Unusable-device handling is unchanged.
- **Gate:**
  - A set of chat requests (greedy and seeded sampling, varied lengths) served with
    `--parallel N` at 1/2/4/8 concurrent clients gives the same `output_sha256` as each request
    alone on the same server and on `--parallel 1 --spec-draft 0 --prefix-cache-slots 0`.
  - Host tests of the batcher with a fake backend: routing, holds, gather, joins and leaves,
    cancellation, errors.
  - Serving throughput and latency at 1/2/4/8 clients against the baseline report and a tuned
    llama-server.

- One scheduler thread owns the device. Each step: decode rows of running requests (1 + chosen
  drafts each) + at most one prefill chunk of an admitting request, within a row budget.
- Row budget and drafts: chosen to maximize expected accepted tokens per step from the measured
  step cost per row count and each request's acceptance estimate, with a per-request minimum rate
  (the multi-user generalization of the adaptive speculation policy).
- Prefill chunks are sized so the step time stays under a latency ceiling for running requests.
- The next step is recorded/submitted before the host finishes the previous step's
  post-processing when its inputs are known (async scheduling); per-request sampling and output
  processing run in parallel across CPU cores.
- FCFS admission; bounded queue (`--max-waiting`); cancellation frees the slot at the next step;
  drain as today.

### 18e design: `--decode-precision f16` (specified and implemented 2026-09-25; kernel v1, not yet faster — [report](../bench/2026-09-25-f16-decode-mode.md))

- **Arithmetic.** In the batched decode, the projections the f16 prefill mode covers are
  computed exactly as that mode computes them: Q4_0, Q4_1 and Q5_K with M ≥ 4096 and
  M % 128 == 0 give `Y[t][m] = Σ_k f16(W[m][k]) · f16(X[t][k])`, one 16×16×16 f16→f32 WMMA
  per 16-k slice in k order ([prefill.md](prefill.md), block 14). Everything else stays FP32
  exactly as in the FP32 mode: attn_k/v, ssm_alpha/beta, the output head, attention, conv,
  DeltaNet and norms. `step`, `verify` and prefill do not change.
- **Batch invariance within the mode.** Each output element is its own chain; tile position
  and batch size never change which WMMAs feed it.
- **Split-K.** Workgroup z computes the chain over `[z·c, (z+1)·c)` from zero, and the reduce
  adds the parts in z order. The chunk `c = gemm.f16nChunk(M, K)` is a function of the shape
  only (about 384 workgroups). The per-row arithmetic is therefore fixed per projection;
  it differs from the unsplit prefill chain, and that is part of the mode's definition.
- **Kernel v1.** `gemm_f16.comp -DSMALLN=1` (`gemm_f16n_*`): a 128 × 16 tile with 4
  subgroups of 32 × 16, the prefill kernel's dequantize-to-LDS scheme, and a split-K z
  dimension. Rows past the batch read the last row; the span is the batch rounded up to 16.
- **Knob.** `Model.Options.decode_precision = .f32 | .f16`. The CLI `--decode-precision` is
  not exposed until the mode beats FP32 (see the report). `.f16` needs `batch_rows > 0` and a
  device with cooperative matrices and subgroups of 64.
- **Gates.**
  1. Component: every `gemm_f16n` row equals `gemm_f16` bitwise for the same X row (unsplit),
     and every split part equals the unsplit kernel over its K range (gpu-test).
  2. Batch invariance: `zerv-batch-check … f16` compares against the same mode on a one-slot
     model, one row per batch.
  3. Quality against FP64: not yet run.
  4. Serving: after a faster kernel.

## Session and HTTP (host performance)

- `session.Generation` becomes a per-request state machine: `feed(logits rows) → tokens, output
  events`, with no model calls inside; the scheduler calls the backend.
- Rendering and tokenization on the connection's thread, never the scheduler's; incremental
  detokenization and SSE framing with fixed per-request buffers; no per-token heap allocation;
  comptime-specialized writers; per-request memory bounds.
- Metrics: running/waiting requests, rows per step, pages in use, preemptions, host time per step,
  tokens per GPU-second, TTFT/ITL histograms.

## Stages and gates (one at a time)

- **18b — slots and batched decode (f32), static test harness.** Model slots, paged KV,
  slot-indexed kernels, `decode(batch)`. Gate: `zerv-batch-check` on the real model — N sequences
  prefilled, then decoded together for T steps with varied join/leave: every logits row
  bitwise equal to the same sequence decoded alone; plus `verify_model.py` unchanged at
  `--parallel 1`, all existing gates.
- **18c — scheduler, session state machine, `--parallel N` serving.** Gate: concurrent requests
  (run_concurrent.py mixes) byte-identical to the same requests served alone; cancellation,
  overload, drain and resource-limit tests; throughput and latency at 1/2/4/8 against the
  baseline report.
- **18d — speculation per slot in batches, chunked prefill interleaving, prefix cache per slot,
  preemption, async scheduling.** Each with its own gate.
- **18e — batched projection kernels:** a tuned FP32 multi-row kernel beyond 5 rows (bitwise
  per row) and the WMMA `f16` mode (batch-invariance gate within the mode; quality against FP64
  as the f16 prefill mode). Component benchmarks against the cost curve, then serving.

## Open (to settle within the stages)

- Preemption: recompute from the prompt vs host copy of pages; choice of victim.
- The WMMA f16 decode path's operand layout for small M (rows padded to 16).
- Page-table size bound: settled in 18b.2 (`ptabWords(context)` entries per slot in the
  activation arena, written by copies).
