# Tiered radix prefix cache (host tier) and the memory-pressure hang — 2026-09-28

**Question.** When many long conversations exceed the KV pool, does keeping evicted prefix
checkpoints in host memory (`--prefix-cache-tier host`, spec: `docs/specs/concurrent.md`,
"18d.5 design") beat dropping them (`off`) and the flat policy, exactly? And why did the
first benchmark runs hang with the tier off?

## Setup

- Binary `third_party/multiuser/zerv-tier7`, sha256 `3308c16f…331bcac` (release build of
  the working tree before the 18d.5 commit). Model sha256 `ede16c7b…384e671d` (the f16
  prefill/f16 KV engine `zerv-f16`).
- RX 7900 XTX, Linux 7.2.6-arch2-1. No other GPU job ran.
- Workload `bench/workloads/multiturn-distinct-v1.json` (sha256 `58e61754…edb7341a`),
  made by `bench/make_multiturn_workload.py --conversations 16 --turns 2 --distinct`:
  16 conversations, each with its own ~8.8k-token document, 2 turns. Phased: all turn-0
  requests, then all turn-1 requests, so each conversation's prefix must survive 15
  others.
- Engine options for all three: `--parallel 8 --kv-type f16 --prefix-cache-slots 48
  --kv-swap-mib 16384`, context 12288 per slot. KV pool: 789 pages of 128 tokens; host
  swap store: 1898 pages.
- Command (manifest and raw results: [data/2026-09-28-tiered-cache/phased](data/2026-09-28-tiered-cache/phased)):

```sh
E="zerv-f16@parallel=8,kv-type=f16,prefix-cache-slots=48,kv-swap-mib=16384"
tools/py bench/run_multiturn.py --output docs/bench/data/2026-09-28-tiered-cache/phased \
  --workload bench/workloads/multiturn-distinct-v1.json --zerv-binary third_party/multiuser/zerv-tier7 \
  --engines "$E,prefix-cache=radix,prefix-cache-tier=host;$E,prefix-cache=radix,prefix-cache-tier=off;$E,prefix-cache=flat" \
  --context-per-slot 12288 --levels 16 --phased --rounds 2 \
  --reference third_party/multiuser/tier-run2-leafonly/raw.jsonl
```

Rounds are interleaved (host, off, flat, then flat, off, host).

## Results

Turn 1 is the cache test: its prompt is turn 0's plus the reply and a new question.

| config | round | turn-1 TTFT p50 | turn-1 TTFT max | cached tokens mean (of 8931) | restores | wall |
|---|---|---|---|---|---|---|
| radix, tier host | 0 | 3.28 s | 10.1 s | 8823 | 16/16 | 154.6 s |
| radix, tier host | 1 | 3.45 s | 10.5 s | 8823 | 16/16 | 156.7 s |
| radix, tier off | 0 | 11.47 s | 64.3 s | 5063 | 9/16 | 208.8 s |
| radix, tier off | 1 | 13.77 s | 72.5 s | 4479 | 8/16 | 217.5 s |
| flat | 0 | 19.91 s | 78.7 s | 4147 | 7/16 | 221.8 s |
| flat | 1 | 12.50 s | 70.7 s | 4691 | 8/16 | 217.5 s |

Turn 0 (cold, identical work in all configs): TTFT p50 74.4–76.4 s, max 140–144 s.

- Host tier: 25–26 demotions (1145–1240 pages) and 11–12 promotions (736–820 pages) per
  round. Nothing dropped for memory or from the host. GPU busy 99.8%.
- Tier off and flat: 31–36 checkpoints dropped for memory per round.
- **Identity gate: 192/192 turns** equal the reference outputs (same greedy tokens as the
  earlier leaf-only runs). No request errors.
- **Interpretation:** with the host tier, every conversation's prefix survives. Turn-1 TTFT
  p50 is 3.3–6.1× lower than without it, and the worst turn-1 wait falls from 64–79 s to
  about 10 s. The wall time for the same work falls by 26–30%. The copy's own cost was not
  measured separately here.

**GPU exactness gate** (`zerv-batch-check MODEL {f16 128|f16 128 split|f16 256|f32 128} 2048
… prefix`, release build of the same tree): 462/462 rows bitwise in all 4 configurations,
including 2 tier round trips (demote all pages, overwrite the freed pool pages, promote,
restore into a fresh slot) ([data](data/2026-09-28-tiered-cache/batch-check/)).

## The hang (failed runs, kept)

- Earlier runs (`zerv-tier6`, raw in `third_party/multiuser/tier-run6-demotedrops`,
  `tier-run7-hang`): radix with the tier off hung in both rounds. 11 of 16 turn-0 requests
  finished, then the rest timed out. At shutdown 5 requests were admitted, GPU busy 14.8%,
  0 swaps, and only 2 checkpoints dropped for memory: eviction could not free anything.
  Most likely cause (from these counters, not isolated on GPU): leaves with empty or
  demoted segments were never dropped, so their ancestors' reclaimable pages stayed
  pinned. The drop fallback's second choice (a leaf whose ancestor has reclaimable pages)
  was added after that binary. With all fixes, the same runs drop 35–36 checkpoints for
  memory and complete.
- A CPU reproduction of the benchmark's shape (`tests/kv_system.zig`, "distinct long
  conversations…") then found more: it hung with every policy, flat included. The
  deadlocks and fixes are listed in the spec ("Progress and invariants under memory
  pressure"):
  - swapped sequences holding shared pages: deep swap-out;
  - swap-in never made room;
  - the time slice not armed with nothing swapped;
  - prompts holding pages waiting on each other: prompt victims, which at first livelocked.
  Each fix alone, reverted, makes the test hang.
- A 400-seed sweep of the tight-pool test found a rare corruption. The restore's renaming
  used ids overwritten by `makeRoom` (shared scratch buffer), which caused an orphaned pin
  and `WrongLogits`. It also found host pages that `evictHost` could not reach. Both are
  fixed. Invariant checks (ownership, pin accounting) now run after every cache call in the
  test. 2,400 seeds pass; the original code fails within 15–150.
- An earlier round-1 host-tier result showed 15 of 16 turn-1 requests. It does not recur
  (16/16 in both rounds here). It was most likely the renaming corruption or a timeout; not
  investigated further.

## Limitations

- One workload shape and one concurrency level (16 conversations, `--parallel 8`). The
  benefit grows with how far the conversations' prefixes exceed the pool.
- No llama-server or vLLM run in this report: the question is our cache policies against
  each other. The multi-turn comparison with competitors is still open (docker unavailable
  for vLLM on this date).
- The deep swap-out and prompt-victim paths are exercised on CPU (`tests/kv_system.zig`,
  real batcher, real page accounting and cache policies). No sequence was swapped in this
  GPU benchmark, so they did not run on GPU. The deep swap-out's copy uses
  `Model.copyPages`, the path demotion uses (checked bitwise by the batch-check prefix
  gate).
