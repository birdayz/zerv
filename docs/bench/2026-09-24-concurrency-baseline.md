# Concurrent requests: llama-server `--parallel` vs zerv today (block 18a baseline, 2026-09-24)

Question: what do several simultaneous users get on this card today, and what does the competitor
reach by batching? This is the baseline block 18 is judged against.

## Setup

- Harness: [bench/run_concurrent.py](../../bench/run_concurrent.py) (new): closed loop, C client
  threads, each sends its next request when the previous one ends; 2 requests per client; prompts
  round-robin from HyperQwen's 8 real prompts
  ([workload](../../bench/workloads/hyperqwen-real-v1-greedy.json), greedy), `max_tokens` 512;
  one untimed warm-up request per level; engines one at a time, GPU checked free before and after.
- Engines (commands in the manifest): llama-server build 10964 (`b29c606e`) `-fa on -b 2048 -ub
  512 -np 8 -c 32768` (4,096 tokens per slot), plain and with `--spec-type draft-mtp
  --spec-draft-n-max 3`; zerv `c88642a1` (f16 prefill, native GEMM), plain and 3 drafts (one
  request at a time, the rest wait in its queue).
- Data: [raw, summary, manifest, server logs](data/2026-09-24-concurrency-baseline/). A first
  run is retained under `third_party/gemm-native/concurrency-baseline-nousage/` (same timings,
  but token counts missing because the client did not request `stream_options.include_usage`;
  fixed, then rerun).

## Results

Every request produced 512 tokens; no errors.

| Clients | Engine | Aggregate tok/s | Per-request decode (median / min) | TTFT p50 / p95 |
| --- | --- | --- | --- | --- |
| 1 | llama-server | 40.3 | 41.1 / 41.0 | 0.15 / 0.38 s |
| 2 | llama-server | 69.7 | 36.7 / 36.5 | 0.76 / 0.85 s |
| 4 | llama-server | 103.8 | 27.2 / 26.7 | 1.05 / 1.43 s |
| 8 | llama-server | **155.7** | 21.1 / 19.2 | 1.25 / 3.77 s |
| 1 | llama-server MTP 3 | 61.7 | 63.8 / 58.7 | 0.15 / 0.34 s |
| 2 | llama-server MTP 3 | 80.8 | 43.8 / 37.4 | 0.64 / 0.67 s |
| 4 | llama-server MTP 3 | 88.0 | 24.2 / 19.9 | 0.80 / 1.43 s |
| 8 | llama-server MTP 3 | 117.3 | 17.0 / 12.6 | 0.55 / 3.25 s |
| 1 | zerv | 50.4 | 50.9 / 50.7 | 0.07 / 0.19 s |
| 2 | zerv | 50.2 | 51.0 / 50.9 | 10.4 / 10.5 s |
| 4 | zerv | 50.1 | 51.2 / 50.8 | 30.8 / 31.0 s |
| 8 | zerv | 50.1 | 51.2 / 50.6 | **71.8 / 71.8 s** |
| 1 | zerv 3 drafts | 81.8 | 83.7 / 78.8 | 0.07 / 0.19 s |
| 2 | zerv 3 drafts | 83.2 | 87.1 / 79.4 | 6.4 / 6.8 s |
| 4 | zerv 3 drafts | 87.3 | 88.8 / 79.9 | 17.6 / 19.0 s |
| 8 | zerv 3 drafts | 87.2 | 88.9 / 79.0 | 40.9 / 42.4 s |

## Interpretation

- zerv serves one stream at a time: aggregate stays at its single-stream rate and waiting clients
  see TTFTs of tens of seconds. From 2 clients on, llama-server's plain batching beats zerv's best
  (3 drafts) in aggregate; at 8 clients it is 1.8× zerv's best and 3.1× zerv plain.
- llama-server's plain batching reaches 3.9× its single stream at 8 clients. Our projection kernel's
  measured curve ([rows scaling](data/2026-09-24-rows-scaling/)) gives ~3.1× at 8 rows for the
  projections alone, so matching and beating it needs a faster batched projection than today's
  FP32 multi-row kernel (research note, "Batch-size scaling").
- llama-server's MTP stops paying off from 2–4 clients (at 8: 117 vs 156 tok/s plain), in line with
  HyperQwen's finding (speculation wins below ~8 users there) — drafts compete with other users
  for rows. A goodput-driven scheduler should choose between drafts and users per step.
- Limitations: one run per engine (no repeats); 512-token answers only; closed loop with equal
  requests (no arrival process, no long prompts mixed in); greedy only.
