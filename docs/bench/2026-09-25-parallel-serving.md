# `--parallel N`: continuous batching in the server (block 18c) — 2026-09-25

**Question.** Several users at once: does zerv now batch their decode steps, with every
response byte-identical to serving it alone? How does it compare to llama-server?

**Answer.**
- **Batch-invariant:** 60 of 60 concurrent responses are byte-identical to the solo reference
  (greedy and seeded sampling, 1–8 clients).
- **Throughput:** 49.4 / 91.4 / 147.6 / 161.8 tok/s aggregate at 1 / 2 / 4 / 8 clients, against a
  same-session llama-server `-np 8` at 39.8 / 65.2 / 99.1 / 143.0. That is +24% / +40% / +49% / +13%.
- **TTFT p50:** 0.17 / 0.35 / 0.42 / 0.53 s, against llama's 0.17 / 0.70 / 1.35 / 1.38 s.
- **llama-server is not batch-invariant:** with greedy decoding it gave 3–5 different outputs
  per prompt across concurrency levels. zerv gave exactly one per prompt at every level.

## Change ([spec](../specs/concurrent.md), "18c design")

- `src/serve/batcher.zig` (new): a scheduler task over a narrow backend (`reset`,
  `prefillChunk`, `decodeBatch`).
  - Generations `join` a slot and submit reset, prefill and step operations, then wait on a
    futex.
  - Decode steps of all waiting slots run as one batch.
  - Prompts are prefilled in the model's chunks, FCFS, alternating with batches.
  - A batch waits for every decoding slot, or until `gather` (2 ms) after the last row was
    released, so a slow client never holds the GPU.
  - Returned logits stay valid until the generation calls `sampled`; no later batch or prefill
    overwrites them.
  - Cancellation withdraws a waiting operation; a running one completes, then the slot is freed.
  - Backend errors go to the affected operations only.
  - Accounting: GPU-busy time, and batches by row count, logged at shutdown.
- `src/serve/engine.zig`: `ModelBackend`, the model under the batcher (select, runChunk,
  decodeBatch). A failure that leaves a command pending or loses the device marks the engine
  unusable. There is also a per-slot session backend.
- `session.Generation` calls `backend.sampled()` right after sampling when the backend has it.
  Nothing else changed.
- `serve/http.zig`: the generation lock became a semaphore of `Options.parallel` permits. The wait
  queue and 503 overload are unchanged.
- `main.zig`: `--parallel N` (1..32, default 1). N > 1 gives the model N slots and batch commands
  for 1..N rows, and turns speculative decoding and the prefix cache off (logged; per-slot
  versions are block 18d). `--context` is per request.
- `bench/run_concurrent.py`:
  - zerv engines with `@parallel=M` get `--context` = context-per-slot.
  - `--reference RAW` is the batch-invariance gate: per-case `output_sha256`, exit 1 on any
    mismatch.
  - The GPU-busy check ignores the harness's own ancestor processes.
- `tests/batcher.zig`: three host tests with a deterministic fake backend.
  1. 12 generations over 4 slots with joins, leaves, holds and pauses: every logits row equals
     its solo computation.
  2. A paused client does not stall the others (partial batches).
  3. Cancellation, backend errors, slot exhaustion and stop.

## Setup

- zerv `82044d43…` (`third_party/paged-ab/zerv-18c-2`), f16 prefill with native GEMM, f32 KV,
  `--parallel 8 --context 4096`. VRAM needed: 20,396 MiB.
- Reference: the same binary with `--parallel 1 --spec-draft 0 --prefix-cache-slots 0`, one
  client, 8 requests (every case).
- llama-server build 10964 (`b29c606e`): `-fa on -b 2048 -ub 512 -np 8 -c 32768` (4,096 tokens
  per slot), as in the [baseline](2026-09-24-concurrency-baseline.md). It ran in the same session,
  directly after zerv.
- Workloads:
  - HyperQwen's 8 real prompts, greedy
    ([workload](../../bench/workloads/hyperqwen-real-v1-greedy.json)).
  - The same prompts with default sampling (T 1.0, top-p 0.95, top-k 20, seed 1234).
  - `max_tokens` 512, closed loop, 2 requests per client, one warm-up per level.
- Commands (from `bench/`; data and manifests in
  [data/2026-09-25-parallel-serving](data/2026-09-25-parallel-serving/)):
  ```sh
  python3 run_concurrent.py --output $D/reference-greedy --zerv-binary $Z --engines "zerv-f16@prefix-cache-slots=0" --concurrency 1 --max-tokens 512 --requests-per-client 8
  python3 run_concurrent.py --output $D/parallel8-greedy --zerv-binary $Z --engines "zerv-f16@parallel=8" --concurrency 1,2,4,8 --max-tokens 512 --requests-per-client 2 --reference $D/reference-greedy/raw.jsonl
  python3 run_concurrent.py --output $D/llama-greedy --zerv-binary $Z --engines llama-fa-ub512 --concurrency 1,2,4,8 --max-tokens 512 --requests-per-client 2 --reference $D/reference-greedy/raw.jsonl
  # and the same two zerv runs with --workload workloads/hyperqwen-real-v1-default.json
  ```

## Results

| Clients | zerv aggregate tok/s | llama aggregate | zerv per-request decode (median) | llama per-request | zerv TTFT p50 / p95 | llama TTFT p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **49.4** | 39.8 | 50.1 | 40.5 | 0.17 / 0.17 s | 0.17 / 0.32 s |
| 2 | **91.4** | 65.2 | 46.9 | 33.8 | 0.35 / 0.41 s | 0.70 / 0.72 s |
| 4 | **147.6** | 99.1 | 38.2 | 26.5 | 0.42 / 0.81 s | 1.35 / 1.67 s |
| 8 | **161.8** | 143.0 | 20.7 | 19.5 | 0.53 / 1.53 s | 1.38 / 4.39 s |

- **Sampled workload:** zerv gives 49.4 / 91.4 / 147.8 / 161.6 tok/s and passes the gate.
- **Baseline for context:** the [1-day-old baseline](2026-09-24-concurrency-baseline.md) had
  zerv flat at 50 tok/s with queued TTFT up to 72 s. llama-server in this run measured 3–8%
  below its baseline figures (40.3 / 69.7 / 103.8 / 155.7).
- **Gate:**
  - `parallel8-greedy` and `parallel8-default` are 30 of 30 identical to the reference each.
  - Every prompt has exactly one output hash across all levels.
  - llama-server (not a gate, different arithmetic) has 3, 5, 4, 3, 3, 3, 3 and 3 distinct
    greedy outputs for the 8 prompts across its levels.
- **Scheduler accounting** (the greedy run, 124.8 s): the GPU was busy 99.3% (decode 94.0%,
  prefill 5.3%). There were 4,242 batches of 15,454 rows, and only 27 were partial (`gather`
  expired). By row count: 1,166 × 1, 1,030 × 2, 1,012 × 4, 998 × 8 rows, plus a few of other
  sizes at level ends.
- **`--parallel 1` is unchanged:** serving-v2 `zerv-f16` and `zerv-f16-spec3` (2 repeats each)
  are 16/16 byte-identical to the recorded outputs
  ([data](data/2026-09-25-parallel-serving/serving-v2-parallel1/)).

## Interpretation

- The host side is not the limit. The GPU is 99% busy, and per-request decode at 4 clients
  (38.2 tok/s) matches the model-level step (25.2 ms → 39.7 tok/s per row).
- The limit is the batched step: 8 rows cost 46 ms, 2.3× one row. The projections run as two
  4-row groups, and each group rereads all weights
  ([18b.2](2026-09-25-batched-decode.md)).
- At 8 clients zerv leads by only 13%. llama's batched matmul scales better past 4 rows, but it
  is not batch-invariant.
- Next lever (18e): a multi-row FP32 projection that stays bitwise equal per row and reads the
  weights once for 8+ rows, and the opt-in WMMA `--decode-precision f16`.
- **Not measured yet:** long prompts under load. A prefill chunk delays running sequences by
  one chunk time, 0.2–0.3 s per 512-token chunk.
- Speculation and the prefix cache are off with `--parallel > 1`. For bruh-style multi-turn
  requests with long shared prefixes the prefix cache matters, so per-slot prefix caching is a
  high-priority 18d item.
