# Prefill stall sweep: steady gap p99 against TTFT (2026-09-27)

Question: at 4–8 users vLLM keeps the p99 inter-token gap at 51–55 ms, and zerv's is
145 ms (docs/bench/2026-09-26-shared-pool.md). How much of that is the prefill stall rule
(`--prefill-stall-ms`, docs/specs/concurrent.md "18c.2 design")? What does lowering it cost
in TTFT, throughput and long-prompt latency?

## Setup

RX 7900 XTX, Qwen3.8-27B-Q4_0, zerv `third_party/multiuser/zerv-swap2` (sha256 `577d791f…`),
f16 KV, `--parallel 8`, shared pool (default), reserve admission (default). Two rounds with
alternating engine order.

```
tools/py bench/run_multiuser.py --output docs/bench/data/2026-09-27-stall/sweep --zerv-binary third_party/multiuser/zerv-swap2 \
  --engines "zerv-f16@parallel=8,kv-type=f16,prefill-stall-ms=0;…=25;…=50;…=100" \
  --context-per-slot 8192 --rounds 2 --reps 2 --levels 4,8
```

(`third_party/multiuser/stall-sweep.sh`; manifest in [data](data/2026-09-27-stall/sweep/)). The
vLLM rows are from the 2026-09-26 ABBA run (same workloads and harness).

## Results (round 1 / round 2)

| Stall | Users | tok/s | TTFT p50 ms | TTFT p95 ms | gap p99 ms | gap max ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 137 / 136 | 631 / 670 | 1264 / 920 | 43 / 45 | 58 / 66 |
| 25 | 4 | 139 / 135 | 663 / 663 | 849 / 1155 | 63 / 54 | 87 / 76 |
| 50 | 4 | 146 / 138 | 342 / 445 | 745 / 745 | 70 / 82 | 88 / 103 |
| 100 | 4 | 145 / 144 | 343 / 319 | 631 / 573 | 48 / 114 | 145 / 142 |
| vLLM | 4 | 93 / 108 | 620 / 618 | 651 / — | 44 / 40 | 431 / 411 |
| 0 | 8 | 150 / 149 | 938 / 912 | 1850 / 1772 | 65 / 66 | 106 / 100 |
| 25 | 8 | 154 / 149 | 733 / 852 | 1258 / 1391 | 84 / 86 | 128 / 113 |
| 50 | 8 | 155 / 145 | 612 / 652 | 1127 / 1131 | 103 / 106 | 140 / 154 |
| 100 | 8 | 158 / 156 | 642 / 638 | 824 / 981 | 144 / 146 | 169 / 187 |
| vLLM | 8 | 144 / 159 | 1083 / 1063 | 1115 / — | 55 / 51 | 917 / 874 |

Interference (4,936-token prompt while 6 stream) and queue, medians:

| Stall | long TTFT ms | short TTFT ms | background gap p50 / max | queued TTFT after free | running gap p99 |
| --- | --- | --- | --- | --- | --- |
| 0 | 11,614 | 2,995 | 66 / 80 | 995 | 61 |
| 25 | 9,629 | 2,110 | 69 / 99 | 671 | 72 |
| 50 | 7,180 | 1,673 | 102 / 123 | 483 | 84 |
| 100 | 5,825 | 1,394 | 155 / 171 | 425 | 55 |
| vLLM | 4,895 | 4,795 | 1,256 / 1,887 | 379 | 54 |

## Interpretation

- **The stall rule is the knob, as designed:** gap ≈ stall + one prefill segment + one
  8-row batch (46 ms).
  - Stall 0 brings the 8-user gap p99 to 65 ms and the max to ~100 ms (vLLM: 51–55 and
    874–917).
  - It still beats vLLM's TTFT p50 (912–938 vs 1,063–1,083 ms) and every background-gap
    number under interference.
  - It pays with prompt latency when prefills compete with decode: the 4.9k prompt takes
    11.6 s, and queued TTFT after a slot frees is 995 ms.
- **No stall setting beats vLLM on every angle.**
  - vLLM's decode rows ride inside its prefill steps (mixed batches), so a step with prefill
    tokens costs about what a decode step costs.
  - Ours alternate: prefill segment, then decode batch.
  - So zerv must choose between vLLM's p99 (stall 0, slower prompts under load) and its
    prompt speed (stall 100, 145 ms p99). The only structural ways past that are a cheaper
    8-row decode step (`.split` is 36 ms against 46 ms, but slower at one row:
    docs/bench/2026-09-26-decode-v2.md) or mixed steps that keep every row's arithmetic
    unchanged.
- **Default stays 100 ms** (the best TTFT, long-prompt and queue numbers; p99 is the
  cost). `--prefill-stall-ms 0` is the documented latency setting.
