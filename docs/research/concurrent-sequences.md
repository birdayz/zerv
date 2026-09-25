# Concurrent sequences: batched decode for several requests (block 18a research, 2026-09-24)

Status: **research in progress** (third pass: cost curve, vLLM, competitor baseline, exact memory
model, proposals for the exactness contract and scheduling; decisions pending with the user:
batched projection arithmetic, paged KV). No spec yet; open questions at the end must be
closed before `docs/specs/` gets a section and before any code.

Question: how should zerv serve several requests at once on one GPU, and what does it take?
User direction (2026-09-24): the scheduler is general-purpose serving code; the model supplies a
narrow backend.

## Why it pays (measured)

- A decode step is weight-bandwidth bound: 15.3 GB of weights per token at ~920 GB/s ≈ 16.7 ms of
  the 20.0 ms step; the ~54 GFLOP of arithmetic would take ~0.5 ms.
- The speculative-verify path already runs several rows per weight pass: 1 / 3 / 4 / 5 rows in
  20.0 / 22.7 / 25.0 / 28.8 ms ([FMA report](../bench/2026-09-24-fma-matvec.md)), i.e. 50 /
  132 / 160 / 174 tok/s total for rows of *one* sequence. Rows of different sequences cost the same
  in the projections (the weight-bound part); attention and the DeltaNet/conv steps are per
  sequence.
- The multi-row kernels are row-for-row bit-identical to the single-row module (Q4_0, Q4_1, Q5_K,
  Q6_K, F32 for 1–4 rows; the known exception: G = 4 at 5 rows on the K-quant path, never shipped).
- Competitor reference: HyperQwen (RTX 3090, vLLM) reports ~400 tok/s aggregate at 8 concurrent
  requests and ~1,035 at 64, and finds speculation better below ~8 users, plain batching above
  ([protocol report](../bench/2026-09-24-hyperqwen-protocol.md)).

## Requirements (user, 2026-09-24)

- "Hardcore performance": the server must **saturate the GPU optimally** — aggregate
  throughput, cost effectiveness (tokens per GPU-second at a given per-user latency), and
  **bin packing** of requests into the GPU's memory and compute.
- The HTTP/CPU side is part of that: no host work on the critical path of the GPU, comptime
  specialization where it removes per-token or per-request branching, and everything else a
  production server needs ("must be a gud server").
- The scheduler is general-purpose serving code; the model supplies a narrow backend.

## Batch-size scaling of today's decode projections (measured, 2026-09-24)

`zerv-matvec-rows-bench` on the real weights, all 64 (48 for attn_qkv) layers resident, Q4_0,
R input rows per weight pass, FP32 FMA kernel `matvec_rows.comp`
([data](../bench/data/2026-09-24-rows-scaling/)); every row bitwise equal to the single-row
module (0 mismatches for all R up to 16):

| R | Cost vs 1 row (ffn_gate / attn_qkv), best GROUP | Aggregate throughput vs 1 user |
| --- | --- | --- |
| 1 | 1.00 | 1.0× |
| 2 | 1.04 / 1.04 | 1.9× |
| 4 | 1.26 / 1.26 | 3.2× |
| 5 | 1.50 / 1.48 | 3.3× |
| 6 | 2.48 / 2.35 | 2.4× |
| 8 | 2.57 / 2.56 | 3.1× |
| 12 | 3.98 / 3.91 | 3.0× |
| 16 | 6.99 / 6.96 (GROUP 2 spills to scratch) | 2.3× |

**Consequence:** the FP32 vector-ALU kernel becomes compute-bound after ~5 rows (each extra row
then costs ~0.3 of a full weight pass, ~11 TFLOP/s effective). With it, batching tops out at
about **3.3× aggregate around 4–5 concurrent rows** (speculative drafts compete for the same
rows). Saturating the GPU beyond that needs a compute-efficient batched projection:

1. a better FP32 multi-row kernel (tuned beyond 5 rows; the vector peak is ~2.5–5× higher than
   the 11 TFLOP/s reached) — keeps the decode arithmetic exact; or
2. WMMA (f16 inputs, f32 accumulation) for batched decode: compute-cheap enough to stay
   bandwidth-bound to ~16–30 rows, but it changes the arithmetic, so it is an explicit precision
   mode (like `--prefill-precision f16`). WMMA results per row do not depend on the other rows,
   so such a mode can still be **batch-invariant** (a request's output independent of who else
   is batched), which is the property the exactness gate needs — not FP32 as such.

## Design consequences of the requirements (proposals, to settle in the spec)

- **Keep the GPU busy between steps:** the next step can only be recorded once every row's next
  token is known. Host sampling costs ~60 µs per row; at 8 rows that is ~0.5 ms per step if done
  serially. Options: sample the rows in parallel on CPU cores; overlap one batch's sampling with
  another batch's GPU step (two alternating batches); GPU-side argmax for greedy rows (exists for
  drafts). Measure host time per step as a first-class metric.
- **Goodput-driven scheduling:** the scheduler knows the cost curve c(R) (above) and the
  speculation acceptance per request; it chooses rows per step (users × drafts) to maximize
  aggregate tokens/s subject to a per-user latency floor — the multi-user generalization of the
  adaptive spec policy.
- **Bin packing:** per-slot memory = fixed recurrent state + KV that grows with the sequence. A
  static `n_ctx / n_seq` split (llama.cpp) wastes memory; paged KV blocks allocated on demand,
  admission control by free blocks, and preemption by snapshotting a slot's state to host memory
  (the snapshot mechanism exists) pack variable-length requests. Prefill chunks are sized to fill
  steps without starving decode.
- **HTTP/CPU:** tokenization and template rendering on connection threads, never on the scheduler;
  incremental detokenization and SSE framing per request with fixed buffers; no per-token heap
  allocation; comptime-specialized response writers and backend dispatch; bounded queues and
  per-request memory limits; cancellation that frees the slot at the next step.
- **Cost model and metrics:** tokens per GPU-second, step occupancy (rows per step), host time per
  step, per-user TTFT/ITL percentiles, memory utilization; exported in `/metrics`.

## Where zerv assumes one sequence (code survey)

| Layer | Assumption | Change needed |
| --- | --- | --- |
| `model` (`runtime.zig`, `layout.zig`) | one state: `position`, one KV cache (per KV buffer group), one DeltaNet/conv state (`State.ssm_words + conv_words`), MTP hidden state; `io` holds one position, one token list, one set of logits rows; recorded commands for 1 row (decode), 2–5 rows of the same sequence (verify), prefill plans | per-slot state regions; a per-row (slot, position) table; attention, DeltaNet, conv, RoPE and MTP kernels indexed by slot per row; logits rows per slot; commands recorded per batch size |
| `matvec` | `max_rows = 5` (verify: drafts + 1) | batch sizes beyond 5 need more rows per weight pass (extend the multi-row kernels or a small-M WMMA GEMM; the f16x GEMM needs 256-row plans) — to measure |
| `session` (`Generation.run`) | a blocking loop per request that owns the model from prefill to the last token (sampling, stops, tool-call parsing, streaming, speculative bookkeeping, prefix-cache records) | invert into a per-request state machine advanced by a scheduler: "here are your logits → next token(s) and output events" |
| `serve` (`http.zig`, `engine.zig`) | `busy` mutex + bounded wait counter: one generation at a time | a scheduler thread that owns the device; requests enqueue, stream from their state machine, cancel |
| prefix cache (`session.prefix`) | one live conversation plus snapshot slots | per-slot live state; snapshots stay the reuse mechanism (recurrent state cannot be shared per token) |

## How llama.cpp does it (source, pinned `b29c606e`)

- `--parallel N` sets `n_seq_max`. Unless the KV cache is unified (`--kv-unified`), each sequence
  gets `n_ctx_seq = n_ctx / n_seq_max`, padded to 256 (`src/llama-context.cpp:290–302`).
- Recurrent memory (`src/llama-memory-recurrent.cpp`): a fixed number of state cells, one per
  sequence (`rs_idx` per `seq_id`, error "seq_id ≥ n_seq_max … use a bigger --parallel").
- Server loop (`tools/server/server-context.cpp`): each iteration builds **one batch**: every
  generating slot's sampled token **plus its speculative drafts** (per-slot verify rows,
  `handle_last_sampled_token`), then pending prompt tokens of slots in prefill up to `n_batch`
  (chunked prefill in the same forward pass), then one decode. So speculation stays on with
  several users.
- Prefix reuse for recurrent models works through state checkpoints saved near the prompt end
  (`checkpoint_offsets`), comparable to our snapshots.

## How vLLM does it (source, pinned `v0.29.0` = `98dff2a8`, `third_party/research-serving/vllm/`)

HyperQwen pins vllm==0.29.0; read 2026-09-24, research material only.

- **One token budget per step** (`v1/core/sched/scheduler.py`, `schedule()`): each step has
  `max_num_batched_tokens` tokens and at most `max_num_seqs` running requests. RUNNING requests are
  scheduled first (a decode is 1 token + its speculative drafts; an unfinished prefill takes up to
  the remaining budget, capped by `long_prefill_token_threshold`), then WAITING requests are
  admitted with the rest (chunked prefill is on by default). Policies: FCFS or PRIORITY.
- **Preemption** when blocks run out: the lowest-priority / last running request is preempted
  (its blocks freed, recomputed later), and its scheduled tokens are returned to the budget.
- **Async scheduling** (`async_scheduler.py`, `async_scheduling` "avoids gaps in GPU
  utilization"): step N+1 is scheduled before step N's sampled tokens are known, with placeholder
  output tokens filled in on the worker. The host never sits between two GPU steps.
- **Hybrid memory in one block pool** (`kv_cache_coordinator.py`,
  `single_type_kv_cache_manager.py`): attention layers get paged KV blocks; recurrent (mamba)
  layers get a state per request drawn from the same pool (`MambaManager`, page sizes unified
  across groups). Prefix caching for recurrent layers (`mamba_cache_mode` "align", the default
  with prefix caching): the state is saved at block-size boundaries at the end of a step and
  reused on a hash hit — the same idea as our snapshots, generalized to many requests.
- `stream_interval`: the number of tokens buffered per SSE flush (1 = smooth; larger = less host
  overhead at high concurrency).

## What bruh sends concurrently (source, `~/projects/bruh`, read-only)

- No agent spawning today: `spawn` is designed in its RFC 0002 ("create and drive an agent",
  scheduled last) but not implemented. `Concurrency::Parallel` in `crates/core/src/tool.rs` is
  about tools running beside each other, not model requests.
- Concurrent model requests therefore come from several bruh sessions (its ACP runtime admits up to
  64) or other clients. The expected near-term load is a few (2–4) concurrent streams, rising when
  `spawn` lands.

## Memory model (exact, from `layout.zig` / `config.zig`)

- **Recurrent state per sequence: 149.6 MiB** (DeltaNet 48 layers × 48 heads × 128 × 128 f32 =
  144.0 MiB, conv 48 × 10,240 × 3 f32 = 5.6 MiB), fixed, independent of the context. With MTP,
  one hidden row more (20 KiB).
- **KV per token: 128 KiB (f32) or 64 KiB (f16)** — 16 attention layers × K and V × 4 KV heads ×
  256. Logits row: 0.95 MiB per decode row.
- Today's single-slot budget: 43,712 tokens with f32 KV = 5.48 GiB of KV + state. Splitting that
  budget statically (llama.cpp's `n_ctx / n_seq`):

| Slots | f32 KV, tokens per slot | f16 KV, tokens per slot |
| --- | --- | --- |
| 2 | 21,258 | 42,515 |
| 4 | 10,030 | 20,060 |
| 8 | 4,417 | 8,833 |

  (before the batch's larger activation arena and the per-row decode attention scratch, which
  are small next to KV).
- **Consequence for bruh:** its first request is ~12.4k tokens (30 tools) and sessions grow; a
  static split at 4 slots (10k with f32 KV) could not even hold one session's first request.
  Useful concurrency on 24 GB therefore needs **shared, on-demand KV allocation** (pages from one
  pool, as vLLM; a long session takes what it needs, short ones fit around it), f16 or 8-bit KV,
  and a policy for running out (queue new requests; preempt by moving a slot's state to host
  memory). The fixed 150 MiB state per sequence is cheap by comparison (a 1k-token KV is 125 MiB).

## Open before the spec

1. ~~vLLM's scheduler semantics~~ read (section above). Still open: HyperQwen's patches to it for
   this model (secondary; their "recurrent-state pages the pool has few of" finding).
2. ~~Multi-row cost beyond 5 rows~~ measured (section above): compute-bound after ~5 rows on the
   FP32 kernel. Still open: how far a tuned FP32 multi-row kernel goes, and a small-M WMMA
   variant's cost curve (the batch-invariant f16 option).
3. ~~Exact per-slot VRAM~~ computed (section above): a static split cannot hold bruh's sessions;
   the spec needs shared paged KV. Still open: page size, the KV kernels' addressing (today one
   contiguous cache per layer), and the preemption/eviction policy.
4. Exactness contract — **proposal:** *batch invariance*: a request's output (logits, sampled
   tokens, bytes) is independent of which other requests share its steps, and equals the
   single-stream output in the same precision mode. Basis: the FP32 multi-row kernels are
   row-for-row bit-identical to the single-row module (measured up to 16 rows here, all formats up
   to 4–5 rows earlier; the 5-row K-quant exception stays excluded); attention, DeltaNet and conv
   run the same per-sequence arithmetic whether one or several sequences are in the step.
   Speculation per slot keeps its sample-matching rule, so speculative output stays identical too.
   A batched WMMA decode would be its own precision mode, batch-invariant within itself.
   Gate: N requests run concurrently (with varied arrival and length mixes) produce the bytes of
   the same requests run alone.
5. Scheduling policy — **proposal** (vLLM-like, goodput-driven):
   - one scheduler thread owns the device; each step = decode rows of running requests (1 +
     chosen drafts each) + a prefill chunk of at most one admitting request, within a row budget
     chosen from the measured cost curve (rows are nearly free up to ~5 today);
   - drafts vs users: allot rows to maximize expected accepted tokens per step (the adaptive
     policy's acceptance estimate per request), with a per-user minimum rate;
   - admission by free KV pages; preemption (state to host memory, resume later) instead of
     failing; FCFS within priority;
   - prefill chunks sized so a step never exceeds a latency ceiling for the running users;
   - the next step is recorded and submitted before the host finishes post-processing of the
     previous one (async scheduling), with sampling done in parallel per request;
   - cancellation frees a slot at the next step boundary; drain as today.
6. ~~Competitor baseline~~ measured ([report](../bench/2026-09-24-concurrency-baseline.md)):
   llama-server `-np 8` plain 40 / 70 / 104 / 156 tok/s aggregate at 1 / 2 / 4 / 8 clients (MTP 3:
   62 / 81 / 88 / 117); zerv flat at 50 (plain) / 82–87 (3 drafts) with TTFT up to 72 s from
   queueing. Still open: repeats, longer answers, mixed long prompts, open-loop arrivals.
