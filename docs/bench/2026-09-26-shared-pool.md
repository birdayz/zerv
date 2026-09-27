# Shared KV pool (18d.2): capacity, steady A/B, long context against llama-server and vLLM (2026-09-26/27)

Question: does one shared KV pool (`--kv-pool shared`, docs/specs/concurrent.md "18d.2 design")
cost anything against the static per-slot KV of 18d.1? What does the capacity it frees buy
when one user sends a 70–80k-token prompt while others stream? And how do llama-server
(`-kvu`, its unified KV) and vLLM behave in the same situation?

## Setup

- RX 7900 XTX (24 GiB), Mesa RADV, PCIe 4.0 x16; Qwen3.8-27B-Q4_0 (sha256 `ede16c7b…`) for
  zerv and llama-server; vLLM on RedHatAI/Qwen3.8-27B-INT4 (W4A16, FP8 KV, `c063053e…`).
- zerv:
  - `third_party/multiuser/zerv-3738213f` for the long-context runs (sha256 `3738213f…`,
    built with `zig build` before the Bazel migration);
  - `zerv-9307bc44` for the steady A/B (sha256 `8742672a…`, `tools/zerv_build.py`, same
    serving code). f16 KV, `--parallel 8`, f16 prefill, packed prefill (18d.1 defaults).
- llama-server: `/usr/bin/llama-server` build 10964 (Vulkan; sha256 `0f401589…`), engine
  `llama-fa-kvu` (`-fa on -b 2048 -ub 512 -kvu`), `-np 8 -c 94208`.
- vLLM v0.30.0 ROCm image `sha256:2e7da1ad…`, hardened container, `--max-num-seqs 8`,
  `--max-model-len 98304` for the long runs, 8192 for the steady A/B. It reported a KV pool
  of 106,313 tokens.
- Every manifest (commands, hashes, host) is next to its raw data in
  [data/2026-09-26-shared-pool](data/2026-09-26-shared-pool/).

## Correctness (gates of the spec)

| Gate | Result |
| --- | --- |
| Host test: shared memory pool (admission waits, no starvation, pages returned) | passes |
| `zerv-batch-check MODEL {f32,f16} {128,256} 2048 {f32,split} shared` | 168/168 rows bitwise equal, joins waited, all pages returned, in 4 configurations |
| Serving identity, `run_concurrent.py --reference` (18d.1 reference), `kv-pool=shared` | every output equal at 1/2/4/8 clients ([gate-greedy](data/2026-09-26-shared-pool/gate-greedy/)) |

## Capacity (f16 KV, measured from the server logs)

| Configuration | Tokens one request may hold |
| --- | --- |
| static, `--parallel 1 --context max` | 96k |
| static, `--parallel 8 --context max` | 12,288 each |
| shared, `--parallel 8 --context max` | the whole pool: 91,904 (718 pages) in the 60k run, 101,120 (790 pages) in the 80k rerun |

The pool is sized to the VRAM free at startup. One 80k attempt started while about 1 GB of
VRAM was held elsewhere. It got 603 pages (77,184 tokens) and refused the 78,653-token
prompt (HTTP 400, `context_length_exceeded`). Kept as
`third_party/multiuser/zerv-80k-vram-short-603pages`; the rerun got 790 pages.

## Steady A/B: static vs shared vs vLLM (ABBA, 2 rounds)

```
tools/py bench/run_multiuser.py --output docs/bench/data/2026-09-26-shared-pool/steady-ab \
  --zerv-binary third_party/multiuser/zerv-9307bc44 \
  --engines "zerv-f16@parallel=8,kv-type=f16,kv-pool=static;zerv-f16@parallel=8,kv-type=f16,kv-pool=shared;vllm" \
  --context-per-slot 8192 --rounds 2 --reps 3
```

Steady (HyperQwen real prompts, 256 tokens each, 2 requests per client). Values are
round 1 / round 2:

| Users | Engine | tok/s | TTFT p50 ms | gap p99 ms | gap max ms |
| --- | --- | --- | --- | --- | --- |
| 1 | static | 48.7 / 47.5 | 161 / 168 | 20.1 / 23.9 | 20 / 26 |
| 1 | shared | 48.2 / 47.3 | 167 / 165 | 23.1 / 24.4 | 25 / 27 |
| 1 | vLLM | 28.9 / 35.9 | 183 / 182 | 37.4 / 30.6 | 40 / 32 |
| 2 | static | 88.5 / 84.1 | 311 / 280 | 21.4 / 41.1 | 128 / 129 |
| 2 | shared | 87.5 / 85.5 | 314 / 319 | 27.5 / 24.9 | 126 / 128 |
| 2 | vLLM | 52.4 / 63.0 | 374 / 367 | 39.0 / 35.0 | 241 / 142 |
| 4 | static | 140.4 / 137.9 | 551 / 573 | 122 / 44 | 145 / 149 |
| 4 | shared | 140.2 / 139.9 | 581 / 445 | 50 / 124 | 146 / 151 |
| 4 | vLLM | 93.2 / 107.8 | 620 / 618 | 44.3 / 40.0 | 431 / 411 |
| 8 | static | 155.7 / 151.0 | 553 / 588 | 145 / 146 | 191 / 188 |
| 8 | shared | 155.6 / 155.8 | 558 / 560 | 146 / 145 | 188 / 187 |
| 8 | vLLM | 143.9 / 158.7 | 1083 / 1063 | 54.9 / 51.2 | 917 / 874 |

Interference (6 users stream, a 4,936-token prompt arrives, then a 280-token one) and queue
(8 running + 2 waiting); medians of 6 repetitions:

| Engine | long TTFT ms | short TTFT ms | background gap p50 / max while it fills | queued TTFT after a slot frees | running gap p99 |
| --- | --- | --- | --- | --- | --- |
| static | 5,940 | 1,426 | 153 / 171 ms | 442 ms | 67 ms |
| shared | 5,882 | 1,415 | 154 / 170 ms | 489 ms | 71 ms |
| vLLM | 4,895 | 4,795 | 1,256 / 1,887 ms | 379 ms | 54 ms |

**Static vs shared: no difference beyond round-to-round noise** in any scenario. For
example 155.7/151.0 vs 155.6/155.8 tok/s at 8 users, and 145/146 vs 146/145 ms gap p99.

## Long context: a 70k or 80k prompt while 6 users stream

Workloads:
- `bench/workloads/long-60k-v1.json`: 195,000 characters of docs, 69,558 prompt tokens;
- `long-80k-v2.json`: 224,000 characters, 78,653 tokens.

Both carry a needle whose answer is "HERON-2958" and "314". The 6 users stream 512-token
answers to short prompts. Once each has 20 tokens the long prompt arrives (64 tokens out),
then 100 ms later a 280-token prompt. 2 repetitions per engine. Commands are in
`third_party/multiuser/long-run.sh` and `long-run2.sh` (copied into each manifest's argv),
for example:

```
tools/py bench/run_multiuser.py --output docs/bench/data/2026-09-26-shared-pool/long-80k-zerv \
  --zerv-binary third_party/multiuser/zerv-3738213f \
  --engines "zerv-f16@parallel=8,kv-type=f16,kv-pool=shared,context=max" --context-per-slot 65536 \
  --long-workload bench/workloads/long-80k-v2.json --scenarios interference --reps 2
```

| Prompt | Engine | long TTFT s | short TTFT s | background gap p50 / max while it fills | background tok/s meanwhile |
| --- | --- | --- | --- | --- | --- |
| 69,558 | zerv shared | 119.5 / 121.6 | 1.37 / 1.39 | 158 / 203 ms | 24.5 |
| 69,558 | llama `-kvu` | **111.2 / 111.8** | 6.84 / 7.15 | 2.93 / 4.68 s | 2.1 |
| 69,558 | vLLM | 208.9 / 202.9 | 227.4 / 221.4 | 4.92 / 10.13 s | 1.2 |
| 78,653 | zerv shared | 139.9 / 138.4 | **1.38 / 1.44** | 157 / 193 ms | 21.1 |
| 78,653 | llama `-kvu` | **131.7** / 140.6 | 131.6 / 140.5 | 3.17 / 4.85 s | 1.9 |
| 78,653 | vLLM | 311.4 / 254.8 | 311.3 / 254.7 | (the long prompt waited for the others to finish) | 10.6 |

- **All seven engine configurations answered the needle correctly in both repetitions**
  (the `text_head` field of raw.jsonl).
- **vLLM at 78,653 tokens** did not start the long prompt while the 6 users held memory.
  All 2,951 background tokens streamed before its first token, and the short prompt waited
  behind it: 255–311 s. At 69,558 tokens it ran the long prompt alongside, with 5–10 s
  background gaps.
- **llama-server at 78,653 tokens** blocked the short prompt behind the long one (131–141 s
  TTFT).

## Failed and discarded runs (kept, not in the tables)

- **long-80k v1** (260,000 characters, about 93k tokens): too big for both zerv's 91.9k
  pool (HTTP 400) and llama's `-c 94208`. llama-server logged
  `decode() failed: Context size has been exceeded` and **exited**. v1 was never committed;
  its results are in `third_party/multiuser/invalid-80k-v1/`.
- **vLLM at `--max-model-len 65536`** refused the 69,558-token prompt (HTTP 400):
  [long-60k-vllm-len65536-refused](data/2026-09-26-shared-pool/long-60k-vllm-len65536-refused/).
- **vLLM cold starts** failed with "No available memory for the cache blocks" twice
  (`third_party/multiuser/vllm-60k-65536-try1-coldfail`, `vllm-60k-fail1`). The harness
  retries; this is the known cold-compile problem.
- **zerv 80k with 603 pages** (VRAM held elsewhere), above.
- The first runner scripts waited on `pgrep -f` loops that matched their own parent shell
  and never started; replaced by sequential scripts.

## Interpretation

- **The shared pool costs nothing measurable and gives one request up to the whole pool**
  (7.5–8.2× the static per-slot context at `--parallel 8`). **Default: `--kv-pool shared`**
  (changed with this report; `static` stays selectable).
- **Long prompts:** zerv is the only engine that keeps the other users streaming.
  - Background gaps stay at the 100 ms stall rule plus one batch: 157 ms p50, ≤ 203 ms max,
    and ~23 tok/s keeps flowing.
  - A short prompt arriving meanwhile gets its first token in 1.4 s, against 7–141 s for
    llama-server and 221–311 s for vLLM.
- **Where zerv loses:**
  1. **Raw long-prompt prefill:** llama-server fills 69,558 tokens in 111 s against our
     120 s (7% faster), with nearly all GPU time given to it. At 78,653 tokens it is
     132–141 s against our 138–140 s, a tie within its variance.
  2. **Steady gap p99 at 4–8 users:** 145 ms for us, 51–55 ms for vLLM. Ours is the
     prefill stall rule (decode at least every 100 ms during a prefill). vLLM's max is
     worse: 874–917 ms.
  3. **8-user throughput:** vLLM's second round reached 158.7 tok/s against our 155.8. Its
     first round was 143.9, so this is a tie within vLLM's round-to-round variance.
  4. **Mid-size prompt under load** (4,936 tokens): vLLM 4.9 s vs our 5.9 s. vLLM stops the
     other streams for it: 1.3 s p50 and 1.9 s max background gaps, against our 154/170 ms.
- **18d.2's remaining weakness** is admission of requests without `max_tokens` (they reserve
  the whole context). That is block 18d.3 (docs/specs/concurrent.md).
