# D.2 later-turn pressure investigation

2026-09-29, before the pool-size ablation. No production code changes proposed yet.

Observed in the original join128 three-turn run: C4 round0 conv2 reused0/10305 tokens,
TTFT12367.7ms; rounds1/2 reused10100 with1696.0/1170.4ms. Join0 reused10219 in
all three rounds. The slow round has113prefill chunks versus93/92; one admission
wait versus0/0; two disk restores versus3/3. All output/count signatures are exact.
Round0 join128 subsequent conv1/conv0 TTFT13611/14863ms is consistent with waiting
behind recomputation, not proof of their individual cache I/O costs. Raw per-request
analysis: `docs/bench/data/2026-09-29-demand-quality/analysis.txt`.

Source-backed possible mechanisms (revision28583d1):
- `src/session/kvcache.zig:Radix.restore` returns0 if promotion/attachment fails and
  `makeRoom` cannot reclaim. Zero reported cached tokens does NOT prove entry eviction.
- `src/serve/engine.zig:beginKey` can reset a shorter hot restore to load a longer
  disk record; if admission fails it returns0. Cold prefill later needs at least the
  same prefix capacity. Read-ahead admission failure instead cancels/drains, suppresses
  restart, and retries; the later direct path can still return0.
- Urgent page reclamation may explicitly override queued-demand policy protection;
  immutable source leases remain protected. Existing tests intentionally cover this.

Thus distinguish actual absence, failed promotion, and failed destination admission
before changing retry behavior. Blindly returning CacheReclaimPending on every miss
could create a no-progress loop; any remedy needs event/retry semantics and tests.

## Controlled existing-knob ablation

Compare join128 at192 vs256 KV pages, all other settings identical, on the SAME
three-turn own-answer workload. Keep parallel2/context12288, host4096MiB, disk8192MiB,
snapshots8, prefetch2; include tuned Vulkan/HIP unchanged. No new code/kernel/math.
Three alternating-order rounds, phased C1/C4, warmup, no concurrent diagnostics.
The larger pool intentionally spends more VRAM on cache slack, still within the common
24GiB physical ceiling. This is NOT equal-allocation performance. Require exact native
outputs/counts, report memory peaks and all trials; no claim of universal speedup.

Hypothesis: additional pool slack reduces failed restoration/cold recomputation under
pressure. Acceptance of the hypothesis requires a measured reduction in zero-reuse
later turns and admission waits; absence of an outlier in three trials alone is weak
evidence and cannot identify the exact prior failing branch. Retain third-turn small
suffix overhead and any other regression. Do not promote the threshold or pool default.
