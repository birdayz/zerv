# Run stalled (2026-09-24, about 20:16–21:06 CEST)

- Command: `cd bench && python3 run_serving.py --output ../docs/bench/data/2026-09-24-decode-baseline/serving-decode-v1 --workload workloads/decode-v1.json --zerv-binary ../third_party/gemm-f16x/zerv-72b986c3 --engines zerv,llama-fa-ub512,llama-fa-ub512-mtp1,llama-fa-ub512-mtp2,llama-fa-ub512-mtp3,llama-fa-ub512-mtp4,llama-fa-ub512-ngram --repeats 2`
- Engines zerv, llama-fa-ub512 and mtp1–4 completed (2 repeats × 4 cases each).
- **llama-fa-ub512-ngram (`--spec-type ngram-mod`) stalled** on its 7th request (think,
  repeat 1, task 2136). Its log's last line is `n_gen = 115, tg = 37.86 t/s` at 1:08 after
  start; nothing followed. The kernel log shows no amdgpu fault or reset, so the server
  process stopped producing tokens.
- The harness had a 3600 s socket timeout and no request deadline, so it waited until
  the user interrupted it about 50 minutes later. No `manifest.json` / `summary.json` was
  written. The 54 completed rows in `raw.jsonl` are valid.
- **Fix:** `bench/run_serving.py` now fails a request after 300 s without data or
  1800 s in total (`--stall-timeout`, `--request-timeout`). It records the failure
  (`raw.jsonl` row with `error`, `manifest.failures`), abandons that engine and continues.
  Regression test: `tests/test_serving_harness.py`.
