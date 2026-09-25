# Multi-user latency: prefill without stalling the others (block 18c.2) — 2026-09-25

**Question.** With `--parallel 8`, what does each user see while others share the server?
Specifically: token gaps, TTFT, the impact of a long prompt on the users already streaming,
and queueing. Can zerv beat llama-server on all of them at once, with every response still
byte-identical to serving it alone?

**Answer.**
**zerv leads on most multi-user metrics; vLLM leads on two.** Final interleaved run: 2 ABBA rounds,
all engines cold (no prompt cache), `--parallel 8`.
- **zerv wins:**
  - 1–4 users: +33–50% throughput against vLLM, +25–50% against llama-server.
  - TTFT: 0.44 s p50 at 8 users, against 1.07 s (vLLM) and 3.3 s (llama).
  - A long prompt arriving while 6 users stream: running users keep 40 tok/s (vLLM 5.7–13,
    llama 5–11). Their worst gap is 169 ms (vLLM 597–1,888, llama 692–2,065). A short prompt
    behind it gets its first token in 1.4 s (vLLM 4.8–5.4 s, llama 6.0–6.6 s).
  - Worst token gap under steady load: 191 ms (vLLM 430–791, llama 798–1,253).
  - Every response is byte-identical to serving it alone.
- **zerv loses:**
  - **8-user throughput:** 150.9 against vLLM's 158.2 tok/s (−4.6%). llama gets 126–131.
  - **Steady gap p99:** 147 ms against 51–53 (vLLM) and 55–58 (llama). 4.5% of zerv's
    gaps are stalls of ~134 ms each; vLLM has 0.1–0.7% stalls of 250–790 ms.
- **Cause, the same for both:** zerv prefills arriving prompts one at a time. When 8 users'
  prompts arrive together, that is 8 × ~170 ms on 128-row plans. vLLM packs them into one
  ~750 ms pass. Both engines' 8-row decode step is 46 ms (gap p50). Next lever: packed
  multi-sequence prefill (below).

## Workload and harness

`bench/run_multiuser.py` (new). It streams every request and records the arrival time of every
token-bearing SSE delta. Prompts are fixed, decoding is greedy with a fixed seed, and requests
send `cache_prompt: false` (cold prefill, see "Failed run" below). Three scenarios:

- **steady:** 1 / 2 / 4 / 8 closed-loop clients. Each sends HyperQwen's short real prompts
  (`bench/workloads/hyperqwen-real-v1-greedy.json`) for 256 tokens, 2 requests per client.
  Reported: aggregate tok/s, TTFT p50/p95, and inter-token gap p50/p99/max. "Stalled" is the
  share of gaps above 1.5× the median, with the mean of those gaps.
- **interference:** P − 2 = 6 users stream 512-token answers. Once each has 20 tokens, a
  4,936-token prompt arrives (`bench/workloads/long-6k-v1.json`, 64 tokens out), then 100 ms
  later a short prompt (280 tokens, 32 out). Both get a slot. Reported: the long and the short
  prompt's TTFT, and the running users' gaps and tok/s while the long prompt is prefilled.
- **queue:** P = 8 users stream 512 tokens; 2 more arrive with no slot free. Reported: their
  TTFT, their TTFT counted from the first slot freeing, and the running users' gaps.

A delta can carry several tokens: servers hold back text that could start a stop string. Such
requests are left out of the gap statistics, with the same rule for both servers.
`--rounds R` starts every engine R times, in alternating order (ABBA); steady runs once per
level per round, interference and queue `--reps` times per round.
`bench/multiuser_table.py SUMMARY` prints the tables below: medians with [min–max] over
rounds (steady) or rounds × reps.

## Baseline (before this block)

zerv `a7c51dd4` (block 18c: prefill in whole 512-token chunks between decode batches, FIFO)
against llama-server `-np 8`, with P − 1 = 7 users in the interference scenario
([data](data/2026-09-25-multiuser/baseline/)):

| Metric | zerv | llama-server |
| --- | --- | --- |
| steady tok/s, 1 / 2 / 4 / 8 users | 47 / 85 / 137 / 151 | 39 / 66 / 96 / 147 |
| TTFT p50 at 8 users | 894 ms | 1330 ms |
| gap p99 / max at 8 users | 210 / 404 ms | 53 / 1435 ms |
| running users' gaps during the long prefill, p50 / max | 422 / 450 ms | 683 / 2031 ms |
| long prompt TTFT | 4.25 s | 5.4 s |
| running users' tok/s during it | 16.5 | 6.6 |

The problem: while a ~5k-token prompt prefills, every running user stalls for a whole
512-token chunk (0.42 s), 10 times in a row. A short prompt arriving just after it waits for
the whole long prefill. (In this run the short prompt also queued for a slot, 9 requests for
8 slots, so its TTFT is not comparable and is left out.)

## Change ([spec](../specs/concurrent.md), "18c.2 design"; [prefill.md](../specs/prefill.md), "32-row tile")

- **Segmented prefill (model).**
  - With slots > 1, every prefill chunk is also recorded as 16 segments of 4 layers
    (`prefill_segment_layers`, `Model.recordPrefillRange`). A segment runs the chunk's
    kernels, pushes and order exactly, so the results are bitwise the chunk's.
  - `Model.prefillSegment(tokens)` runs the next segment of the slot's chunk and returns
    `{consumed, logits}`; `abortChunk()` drops a chunk in flight.
  - Batched decode may run between segments:
    - the batch keeps its own r/f rows (`layout.Act.brf`, `b_down`/`d_down`), so it never
      overwrites the prefill's live rows;
    - `decodeBatch` marks the shared io words dirty (`io_dirty`), and the next segment
      rewrites the count, p0 and RoPE rows.
  - Errors: `ChunkInFlight` (another slot's prefill, or a reset mid-chunk) and
    `SlotNeedsReset` (after an abort).
  - `gpu.max_commands` 64 → 256.
- **Scheduler (`serve/batcher.zig`).**
  - `--prefill-stall-ms N|chunk` (default 100): while a prompt prefills and other users wait,
    a decode step runs at the first segment boundary N ms after the previous one. `chunk` is
    the previous behaviour.
  - `--prefill-order shortest|fifo` (default shortest): when a chunk ends, the pending prompt
    with the fewest remaining tokens goes next. At most `--parallel − 1` others can overtake a
    prompt.
  - Stats: prefill units, batches inside a chunk, aborted chunks.
- **Audit fixes** (ownership rules written into the spec; each with a host test in
  `tests/batcher.zig`):
  - A use-after-free when a running prefill was cancelled: the waiter now waits for the
    running unit to end (`canceled` flag).
  - A data race in `usable()`: batch mode reads only the atomic `fatal`.
  - Per-row `checkRow`: a bad logits row fails its own request, not the batch.
  - `Native.attachBatcher` refuses a shared prefix cache or speculation policy
    (`SharedStateWithBatching`).
  - The only process-wide mutable state is the atomic `stop_requested`.
- **Short-prompt f16 GEMM tile (`--f16-small-tile on|off`, default on).**
  - `gemm_f16.comp -DSMALLM`: a 32 × 128 tile, bitwise the 128 × 128 tile.
  - Used for 128-row plans whose 128-tile grid would have fewer than 64 workgroups. On this
    model that is the 5120-wide outputs: ffn_down, lin_out and attn_out.
- **Harness:** `bench/run_multiuser.py` and `bench/multiuser_table.py` (new).
  `run_concurrent.py`'s GPU-busy check ignores its own ancestors.

## Correctness gates (binary `7022f1aa`)

- **gpu-test:** 39/39 under the spill gate, 0 spill failures. `gemm_f16m` is bitwise equal to
  `gemm_f16` in 7 cases: Q4_0 / Q4_1 / Q5_K, one and two row tiles, row tails 1..256, K 512..5120.
- **Host tests:** 96/96 in Debug and ReleaseFast. Python: 63 OK.
- **`zerv-batch-check`, f32 and f16 KV:**
  - 400/400 logits rows bitwise equal to solo decoding, batch sizes 1..8;
  - joins run segment by segment, with 108 decode batches between segments;
  - error cases pass: chunk in flight, abort, then reset.
- **f16 oracle, long fixture** (`verify_model.py --precision f16 --gemm-code native`):
  - modes 512 and 256: logits, tensors and serving logits byte-identical to the captures
    before this block;
  - mode 128, `--f16-small-tile on` against `off`: byte-identical.
- **FP32 oracles** (default and long fixtures, every mode): pass, and 57 of 57 capture files
  are byte-identical to before.
- **Serving identity:** `run_concurrent.py --reference` with `--parallel 8` at 1/2/4/8
  clients: 30/30 outputs equal to the solo reference
  ([data](data/2026-09-25-multiuser/gate-parallel8-greedy/)). 8 more runs of the decode A/B
  below also pass.
- **No decode regression.** Interleaved ABBA runs against `a7c51dd4`, 2 rounds, with
  `run_concurrent.py` at 1 and 8 clients and 512 tokens
  ([data](data/2026-09-25-multiuser/ab-decode/)):

  | | 1 client, tok/s (4 runs) | 8 clients, tok/s (4 runs) |
  | --- | --- | --- |
  | before (`a7c51dd4`) | 45.4 median (43.8–46.9) | 151.4 median (148.0–152.9) |
  | after (`7022f1aa`) | 46.0 median (45.0–46.8) | 152.3 median (150.8–154.2) |

  Both binaries drifted together from run to run (±3%). The 49.4 tok/s at one client in the
  [18c report](2026-09-25-parallel-serving.md) was an earlier machine state, not a
  regression.

## Short-prompt tile (component, model level)

`race_profile.py`: interleaved `zerv-model-profile` runs, f16 prefill (native), f16 KV. A
600-token prompt on the 128-row plan runs as 4 × 128 + 88 rows; the 256-row plan gets 1,000
tokens. GPU ms for the whole prefill, median [min–max] of 4 runs
([data](data/2026-09-25-multiuser/)):

| Run | small tile off | small tile on |
| --- | --- | --- |
| plan 128, rule "grid < 192" | 896.8 [892.3–902.3] | 991.4 [986.4–994.8] (**−10.5%**) |
| plan 128, rule "grid < 64" (shipped) | 891.3 [890.3–892.6] | **844.1** [844.0–845.4] (**+5.3%**) |
| plan 256, rule "grid < 64" | 865.4 [863.8–866.9] | 1008.1 [1006.2–1015.1] (**−16%**) |

- Per phase, on the 128-row plan with the shipped rule: ffn_down 61.3 → 51.8 ms, lin_out
  17.6 → 15.3, attn_out 5.5 → 4.2.
- ffn_in and lin_in keep their kernel but read 1–3 ms higher. That is probably timestamp
  attribution next to the changed phases; it is included in the totals.
- With the first rule (below 192 workgroups), ffn_in (136 workgroups) went 46.5 → 70.7 ms
  and lin_in 22.2 → 28.7 ms.
- At 256 rows the tile replaces the wave32 `gemm_f16x` and loses.
- Shipped rule: 128-row plans only, 128-tile grid below 64 workgroups.

## Stall budget sweep (binary `9c060c90`, interference with P − 1 users)

[data](data/2026-09-25-multiuser/stall-sweep/). This ran before the scenario was fixed to
P − 2, so the short prompt queued for a slot and its TTFT includes the whole long prefill.

| `--prefill-stall-ms` | running gaps during the long prefill, p50 / max | running tok/s during it | long TTFT | steady gap p99 / max, 8 users | steady tok/s, 8 users |
| --- | --- | --- | --- | --- | --- |
| 50 | 110 / 119 ms | 66.9 | 6.58 s | 104 / 226 ms | 148.0 |
| 100 | 158 / 169 ms | 45.7 | 5.36 s | 150 / 404 ms | 150.7 |
| 200 | 254 / 265 ms | 29.7 | 4.70 s | 219 / 642 ms | 152.3 |
| chunk + fifo (previous) | 424 / 450 ms | 16.4 | 4.27 s | 219 / 513 ms | 152.3 |

The budget trades the long prompt's TTFT against the running users' gaps, almost linearly.
Steady throughput moves by ~3% (single runs).

## Final comparison: zerv against vLLM and llama-server (binary `7022f1aa`)

`run_multiuser.py --engines "zerv-f16@parallel=8,kv-type=f16;vllm;vllm-b512;llama-fa-ub512;llama-fa-b512"
--reps 3 --rounds 2` (from `bench/`; [data](data/2026-09-25-multiuser/final/)). Engines ran one at a
time in the order zerv, vLLM, vLLM-b512, llama, llama-b512, then reversed. 8 slots each, 8,192
tokens of context per request, requests with `cache_prompt: false`.
- **zerv:** `--parallel 8 --kv-type f16 --prefill-precision f16 --spec-draft 0`, defaults
  otherwise (`--prefill-stall-ms 100 --prefill-order shortest --f16-small-tile on`).
- **vLLM** v0.30.0, W4A16 checkpoint, FP8 KV cache ([setup](2026-09-25-vllm.md)):
  `--max-num-seqs 8 --max-model-len 8192 --no-enable-prefix-caching`, prefill chunk 2048
  (default) or 512 (`--max-num-batched-tokens 512`).
- **llama-server** build 10964: `-np 8 -c 65536 -fa on -ub 512`, with `-b 2048` or `-b 512`.

Different weights: vLLM runs RedHat's INT4 group-128 GPTQ; zerv and llama-server run the Q4_0 GGUF.
Medians [min–max] over 2 rounds (steady) or 6 runs (interference, queue):

#### Steady (closed loop, short prompts, 256 tokens)

| Engine | Users | tok/s | TTFT p50 ms | TTFT p95 ms | gap p50 ms | gap p99 ms | gap max ms | stalled % | stalled mean ms | slowest user tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zerv-f16@parallel=8,kv-type=f16 | 1 | 47.9 [47.2–48.5] | 167 [166–167] | 169 [168–171] | 20 [20–20] | 22 [20–23] | 23 [23–24] | 0.0 [0.0–0.0] | 0 [0–0] | 49.1 [48.4–49.8] |
| zerv-f16@parallel=8,kv-type=f16 | 2 | 87.0 [86.6–87.4] | 310 [308–313] | 373 [372–373] | 21 [21–22] | 23 [22–24] | 128 [128–128] | 0.7 [0.7–0.7] | 86 [85–87] | 45.2 [44.9–45.5] |
| zerv-f16@parallel=8,kv-type=f16 | 4 | 138.7 [138.7–138.8] | 373 [373–373] | 834 [833–834] | 25 [25–26] | 126 [124–127] | 149 [149–150] | 2.1 [2.1–2.1] | 118 [118–118] | 35.6 [35.5–35.6] |
| zerv-f16@parallel=8,kv-type=f16 | 8 | 150.9 [150.5–151.3] | 437 [388–485] | 1546 [1544–1548] | 46 [46–46] | 147 [147–147] | 191 [191–191] | 4.5 [4.5–4.6] | 134 [134–135] | 19.3 [19.2–19.4] |
| vLLM (chunk 2048) | 1 | 29.6 [23.4–35.9] | 183 [181–186] | 192 [186–197] | 33 [27–40] | 45 [31–60] | 49 [33–65] | 0.6 [0.0–1.2] | 31 [0–62] | 29.2 [21.8–36.7] |
| vLLM (chunk 2048) | 2 | 51.2 [38.9–63.5] | 371 [357–384] | 411 [359–463] | 41 [30–52] | 51 [36–67] | 118 [40–196] | 0.1 [0.0–0.2] | 86 [0–171] | 25.1 [17.9–32.2] |
| vLLM (chunk 2048) | 4 | 99.1 [92.3–105.9] | 670 [641–699] | 828 [678–979] | 35 [35–36] | 57 [40–73] | 385 [380–391] | 9.6 [0.1–19.1] | 209 [66–352] | 24.2 [20.8–27.6] |
| vLLM (chunk 2048) | 8 | 158.2 [158.0–158.3] | 1067 [1067–1067] | 1178 [1142–1214] | 46 [46–46] | 53 [52–53] | 791 [785–797] | 0.1 [0.0–0.1] | 787 [785–790] | 20.3 [20.3–20.3] |
| vLLM (chunk 512) | 1 | 31.1 [26.3–35.9] | 178 [172–183] | 191 [190–193] | 31 [27–36] | 46 [30–62] | 52 [32–71] | 1.4 [0.0–2.7] | 31 [0–61] | 30.8 [25.0–36.6] |
| vLLM (chunk 512) | 2 | 59.4 [55.8–63.0] | 363 [345–381] | 402 [400–404] | 31 [30–32] | 47 [35–59] | 207 [206–208] | 2.6 [0.2–5.1] | 119 [59–179] | 28.8 [26.1–31.5] |
| vLLM (chunk 512) | 4 | 108.3 [108.1–108.4] | 600 [598–603] | 637 [633–640] | 35 [35–35] | 38 [37–40] | 419 [417–421] | 0.1 [0.1–0.1] | 412 [409–415] | 27.5 [27.4–27.5] |
| vLLM (chunk 512) | 8 | 157.9 [157.8–158.0] | 802 [780–824] | 1052 [1048–1056] | 46 [46–46] | 51 [50–53] | 430 [429–431] | 0.7 [0.7–0.7] | 248 [247–249] | 20.1 [20.1–20.1] |
| llama -b 2048 | 1 | 37.9 [36.7–39.2] | 451 [361–541] | 483 [361–604] | 24 [24–25] | 27 [26–29] | 33 [30–36] | 0.0 [0.0–0.0] | 0 [0–0] | 40.2 [39.2–41.1] |
| llama -b 2048 | 2 | 64.0 [62.7–65.3] | 978 [943–1014] | 1076 [1035–1117] | 27 [27–28] | 32 [31–32] | 148 [128–168] | 0.1 [0.1–0.2] | 146 [125–168] | 35.7 [35.1–36.3] |
| llama -b 2048 | 4 | 92.2 [90.3–94.2] | 2096 [1860–2333] | 2396 [2141–2651] | 35 [35–35] | 40 [40–40] | 328 [320–336] | 0.3 [0.3–0.3] | 200 [159–241] | 27.3 [27.2–27.5] |
| llama -b 2048 | 8 | 131.0 [124.4–137.6] | 3292 [2992–3592] | 3623 [3165–4080] | 47 [45–49] | 55 [52–57] | 1253 [1057–1450] | 0.4 [0.3–0.4] | 408 [352–465] | 18.9 [18.8–18.9] |
| llama -b 512 | 1 | 38.4 [37.9–38.8] | 372 [362–383] | 440 [385–495] | 24 [24–24] | 28 [28–28] | 35 [31–40] | 0.1 [0.0–0.2] | 20 [0–40] | 40.4 [40.1–40.7] |
| llama -b 512 | 2 | 64.3 [64.1–64.5] | 908 [889–927] | 1024 [1017–1032] | 27 [27–27] | 32 [32–33] | 131 [127–134] | 0.2 [0.2–0.3] | 102 [71–134] | 35.5 [35.1–36.0] |
| llama -b 512 | 4 | 92.6 [90.6–94.5] | 1751 [1651–1850] | 1923 [1845–2001] | 36 [35–36] | 41 [39–43] | 315 [230–400] | 0.3 [0.2–0.4] | 190 [152–227] | 26.7 [26.1–27.4] |
| llama -b 512 | 8 | 125.7 [123.4–128.0] | 3406 [3029–3783] | 3729 [3487–3972] | 49 [49–49] | 58 [58–59] | 798 [747–848] | 0.6 [0.4–0.7] | 383 [314–452] | 19.1 [18.9–19.3] |

#### Interference (P-2 users streaming; a 4,936-token prompt, then a 280-token prompt 100 ms later)

| Engine | runs | long TTFT ms | short TTFT ms | running users' gap before, p50 ms | gap during prefill p50 / p99 / max ms | running tok/s during prefill |
| --- | --- | --- | --- | --- | --- | --- |
| zerv-f16@parallel=8,kv-type=f16 | 6 | 5838 [5783–5932] | 1397 [1369–1427] | 41 [41–42] | 155 [154–159] / 169 [167–170] / 169 [167–172] | 40.0 [39.3–41.0] |
| vLLM (chunk 2048) | 6 | 4879 [4833–6811] | 4776 [4732–6710] | 43 [43–43] | 1256 [1238–1304] / 1888 [1883–3795] / 1888 [1883–3795] | 5.7 [4.1–6.1] |
| vLLM (chunk 512) | 6 | 5268 [5251–5285] | 5442 [5426–5459] | 43 [43–43] | 490 [488–490] / 597 [596–602] / 597 [596–602] | 13.0 [12.5–13.5] |
| llama -b 2048 | 6 | 6120 [5872–6523] | 6022 [5775–6427] | 57 [39–78] | 1056 [994–1149] / 2065 [2038–2114] / 2065 [2038–2114] | 4.9 [4.6–5.6] |
| llama -b 512 | 6 | 6740 [6458–7431] | 6641 [6359–7332] | 64 [39–98] | 572 [545–616] / 692 [664–787] / 692 [664–789] | 10.8 [9.8–11.3] |

#### Queue (P users streaming, 2 more arrive with no slot free)

| Engine | runs | queued TTFT ms | queued TTFT after the first slot frees, ms | running gap p50 / p99 / max ms | stalled % |
| --- | --- | --- | --- | --- | --- |
| zerv-f16@parallel=8,kv-type=f16 | 6 | 22663 [22397–22917] | 485 [290–675] | 46 [46–46] / 142 [135–144] / 189 [187–192] | 2.0 [2.0–2.0] |
| vLLM (chunk 2048) | 6 | 24083 [23893–24190] | 419 [256–491] | 48 [48–48] / 54 [53–56] / 666 [617–719] | 0.2 [0.2–0.2] |
| vLLM (chunk 512) | 6 | 24054 [23912–24192] | 391 [270–499] | 48 [48–48] / 53 [53–55] / 423 [405–434] | 0.4 [0.4–0.4] |
| llama -b 2048 | 6 | 24178 [23537–26003] | 1071 [995–1271] | 47 [46–50] / 55 [51–59] / 582 [509–778] | 0.2 [0.2–0.3] |
| llama -b 512 | 6 | 23997 [23438–24475] | 1113 [1019–1213] | 46 [46–47] / 54 [51–58] / 584 [253–738] | 0.3 [0.2–0.4] |

Notes:
- **vLLM's second round was slower at 1–2 users** (22–26 tok/s against 36; its own log
  shows 22 tok/s generation for one request). Its best round is 35.9 tok/s at 1 user.
- **zerv's 1-user gap is 20.0 ms (50 tok/s)**, against vLLM's 27 ms and llama's 24 ms.
- **Queue:** after a slot frees, zerv takes 485 ms [290–675] to the first token, vLLM 391–419
  and llama ~1,100. zerv serves the two waiting prompts one at a time.
- **The steady stall pattern** comes from lockstep arrivals. All 8 clients start together
  and stop at 256 tokens, so their next prompts arrive within a few ms of each other.
  - vLLM prefills them in one pass: one stall of ~790 ms (chunk 2048) or two of ~430 ms
    (chunk 512) per user and request, under 1% of gaps.
  - zerv spreads 8 separate prefills over ~12 stalls of ~134 ms, 4.5% of gaps.

**What wins both p99 and max:** less prefill work per round. With packed multi-sequence
prefill, zerv would prefill the 8 prompts together, as vLLM does. The GEMM kernels already
give each output element the same result at any plan size (`gemm_f16`, `gemm_f16m`,
`gemm_f16x` are bitwise equal per element). The per-sequence kernels (attention, DeltaNet,
conv) then run per segment, with each prompt's chunk grid kept exactly as in solo prefill.
That keeps batch invariance. It is the next block (18d).

## Failed run (kept): llama-server prompt-cache hits

The first run of the final comparison (`failed-cachehits/`) is invalid for two reasons:
- **Cache hits.** Requests did not send `cache_prompt: false`. llama-server reuses a slot's
  KV for a repeated prompt, so its interference reps after the first served the 4,936-token
  prompt from its cache: long TTFT 0.22–0.27 s instead of 6.2 s.
  - zerv `--parallel N` has no prefix cache yet (block 18d).
  - The harness now sends `cache_prompt: false` by default (`--prompt-cache on` restores
    it).
  - llama's per-slot prefix reuse is a real feature zerv lacks in parallel mode. It matters
    for multi-turn chats, and it is the first item of 18d.
- **Crash.** In llama-server's 5th queue rep, both queued requests were closed without a
  response (`RemoteDisconnected`). The exception killed the client threads unrecorded, and
  the summary step crashed. `stream()` now records connection failures as errors, and the
  tables report failed runs. The cause on llama's side is not investigated.

Its cold runs agree with the final run: zerv's interference numbers are within 1–3%, and
llama's cold reps (0 and 3) gave long TTFT 6.2–6.3 s, running gaps p50 900–936 ms and max
2.1 s, at 4.8–5.0 tok/s.

## Other runs kept

- **`final-cancelled/`:** the second attempt, with vLLM not yet in the set. The user stopped it
  for a break after zerv's first round. Its zerv numbers match the final run within 1–3%.
- **`failed-vllm-cold-compile/`:** the first attempt with vLLM. zerv's round 0 completed.
  vLLM exited at startup: a cold graph compile leaves no memory for its KV cache on 24 GB
  ([vLLM report](2026-09-25-vllm.md), "Bring-up findings"). Each vLLM configuration was
  warmed once before the final run.
