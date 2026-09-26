# Packed multi-sequence prefill (block 18d.1) — 2026-09-26

**Question.** In the [18c.2 comparison](2026-09-25-multiuser.md), vLLM led zerv on 8-user
throughput and steady gap p99 because it prefills simultaneous prompts in one pass. Does
packing several prompts into one prefill chunk, each bitwise identical to its solo
prefill, close that gap?

**Answer.**
- **Correct:** every packed sequence's logits are bitwise equal to its solo prefill, and so
  is every later decode row:
  - `zerv-batch-check … pack`: 504/504 in 4 configurations (f32 and f16 KV, f16 decode,
    256-token pages);
  - serving: 30/30 responses identical to the same server serving one request at a time.
- **Faster where prompts coincide.** Measured against `--prefill-pack 1` in the same
  interleaved run:
  - 8-user throughput 141.7 → **150.8** tok/s (+6%);
  - TTFT p95 1,584 → **1,054** ms;
  - queue running-gap p99 146 → **82** ms;
  - stalled gaps 4.9% → 3.2%.
- **Against vLLM** (same run):
  - 8-user throughput ties: 150.8 against 151.9 tok/s for its best configuration.
  - TTFT p50 is 592 ms against 814–1,103 ms, and the long-prompt interference result is
    unchanged: 39.5 tok/s kept against 5.0–12.7.
  - **Steady gap p99 stays behind:** 149 against 53 ms. The max is 193 against 466–896 ms.
- **Why p99 did not move:**
  - Only 19 of 77 chunks were packed. In the closed loop, prompts arrive staggered by a
    few decode steps, and a new chunk starts as soon as one prompt is pending.
  - The gaps during any prefill are set by `--prefill-stall-ms 100` plus a segment.
  - vLLM stalls rarely but long: under 1% of gaps, 470–900 ms each. zerv stalls in 3% of
    gaps, ~140 ms each.
  - p99 rewards vLLM's pattern and max rewards zerv's. Both are reported.

## Change ([spec](../specs/concurrent.md), "18d.1 design", "As implemented")

- **Model:**
  - packed kernels `qk_p`, `attn_flash_p`, `conv_p`, `delta_p` (+ KV16), built from the
    existing sources with `-DPACKED`; all existing modules byte-identical;
  - io row entries and a sequence table (`layout.io.packRows`, `seqs`);
  - `Model.prefillPackedSegment`, `packable`, `packSpan`;
  - an output head over the sequences' last rows (`seg_tail` commands).
- **Arithmetic:**
  - `Options.f16_split = .shape` (default; CLI `--f16-split shape|plan`): the FP32 split-K
    of the small f16-mode projections no longer depends on the plan.
  - In parallel mode, prefill plans have at least 128 rows.
  - Together these make a row's arithmetic independent of packing. They also change f16-mode
    prefill outputs once, so the serving reference was regenerated.
- **Scheduler:**
  - `prefillUnit(items)` and `packFits`; `--prefill-pack N` (default 8, `1` = the previous
    behaviour);
  - a canceled member completes at the next unit boundary without aborting the others' chunk.
- **Tests:**
  - host: a pack test (solo logits, cancellation inside a pack); 97/97, stable over 5
    ReleaseFast runs;
  - `tests/model.zig`: io layout;
  - `tests/test_model.py`: the new modules.

## Gates (binary `7a6e3f06`; the gates ran on `36f328f2`, which differs only in a log line)

- `zerv-batch-check MODEL {f32,f16} 128 2048 f32 pack`, `f32 128 2048 f16 pack` and
  `f32 256 4096 f32 pack`: 504/504 bitwise, packs of 1–4, 160 decode batches between segments.
  The existing non-packed batch-check (the K-slot model's single-sequence prefill, now on the
  packed kernels, against the 1-slot model): 400/400 with f32 and f16 KV.
- **A failure found and fixed.** The first packed run had 126 mismatches, on the 40- and
  61-token prompts only. Plans under 128 rows run FP32 projections, so such a chunk's
  arithmetic depended on packing. Fix: plans of at least 128 rows in parallel mode.
- **Serving** ([data](data/2026-09-26-packed-prefill/)): `run_concurrent.py` with
  `zerv-f16@parallel=8`. A reference at concurrency 1 (8 requests), then 1/2/4/8 clients with
  `--reference`: passed.

## Multi-user comparison

`run_multiuser.py --engines "zerv-f16@parallel=8,kv-type=f16;zerv-f16@parallel=8,kv-type=f16,prefill-pack=1;vllm;vllm-b512" --reps 3 --rounds 2`
(from `bench/`, 2026-09-26 09:59–10:40 UTC; [data](data/2026-09-26-packed-prefill/multiuser/)).
vLLM as in the [vLLM report](2026-09-25-vllm.md); its configurations were warm in the compile
cache. llama-server was not rerun; see the [18c.2 run](2026-09-25-multiuser.md).

#### Steady (closed loop, short prompts, 256 tokens)

| Engine | Users | tok/s | TTFT p50 ms | TTFT p95 ms | gap p50 ms | gap p99 ms | gap max ms | stalled % | stalled mean ms | slowest user tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zerv, pack 8 | 1 | 46.6 [46.2–47.0] | 169 [167–171] | 171 [168–173] | 21 [21–21] | 23 [21–25] | 24 [22–25] | 0.0 [0.0–0.0] | 0 [0–0] | 47.5 [46.8–48.2] |
| zerv, pack 8 | 2 | 84.8 [84.7–85.0] | 320 [317–323] | 386 [376–397] | 22 [22–22] | 24 [23–24] | 130 [129–131] | 0.7 [0.7–0.7] | 92 [88–96] | 44.0 [43.8–44.2] |
| zerv, pack 8 | 4 | 137.9 [136.0–139.9] | 374 [332–417] | 691 [683–699] | 26 [26–26] | 77 [28–126] | 154 [152–155] | 1.5 [1.0–2.0] | 113 [111–115] | 35.4 [34.7–36.2] |
| zerv, pack 8 | 8 | 150.8 [150.7–150.9] | 592 [590–595] | 1054 [1051–1057] | 48 [48–48] | 149 [148–150] | 193 [192–193] | 3.2 [3.2–3.2] | 143 [142–143] | 19.2 [19.2–19.2] |
| zerv, pack 1 | 1 | 46.9 [46.9–47.0] | 170 [170–170] | 171 [171–172] | 21 [21–21] | 21 [21–21] | 22 [22–22] | 0.0 [0.0–0.0] | 0 [0–0] | 48.3 [48.2–48.3] |
| zerv, pack 1 | 2 | 84.1 [83.5–84.6] | 320 [318–321] | 381 [381–381] | 22 [22–22] | 27 [25–28] | 130 [129–131] | 0.7 [0.7–0.7] | 93 [90–95] | 43.1 [42.6–43.6] |
| zerv, pack 1 | 4 | 133.5 [132.7–134.2] | 410 [382–438] | 862 [856–868] | 26 [26–26] | 126 [126–126] | 155 [154–155] | 2.2 [2.2–2.2] | 111 [110–111] | 34.5 [34.3–34.7] |
| zerv, pack 1 | 8 | 141.7 [136.0–147.3] | 501 [407–595] | 1584 [1583–1585] | 49 [48–51] | 151 [150–152] | 202 [196–208] | 4.9 [4.6–5.2] | 136 [135–138] | 17.7 [16.7–18.8] |
| vLLM (chunk 2048) | 1 | 21.8 [19.1–24.6] | 199 [190–207] | 205 [192–218] | 44 [40–48] | 64 [45–82] | 69 [46–92] | 4.7 [0.0–9.4] | 39 [0–77] | 21.4 [18.3–24.6] |
| vLLM (chunk 2048) | 2 | 45.4 [44.1–46.6] | 469 [386–552] | 542 [449–634] | 43 [41–44] | 66 [44–87] | 176 [155–196] | 3.9 [0.2–7.6] | 132 [80–184] | 21.3 [19.2–23.4] |
| vLLM (chunk 2048) | 4 | 91.8 [82.4–101.1] | 637 [633–641] | 668 [649–687] | 42 [37–46] | 47 [41–53] | 340 [254–426] | 0.1 [0.1–0.2] | 331 [240–422] | 23.5 [20.8–26.1] |
| vLLM (chunk 2048) | 8 | 151.9 [150.5–153.3] | 1103 [1093–1112] | 1261 [1172–1349] | 48 [48–48] | 53 [51–54] | 896 [878–913] | 0.0 [0.0–0.0] | 876 [854–897] | 19.4 [19.3–19.5] |
| vLLM (chunk 512) | 1 | 27.4 [20.3–34.6] | 190 [180–200] | 197 [184–211] | 36 [28–44] | 65 [30–101] | 67 [31–104] | 5.8 [0.0–11.6] | 42 [0–85] | 27.8 [20.4–35.2] |
| vLLM (chunk 512) | 2 | 49.6 [38.7–60.5] | 384 [357–410] | 436 [414–458] | 39 [32–46] | 71 [34–108] | 169 [159–179] | 4.9 [0.2–9.6] | 124 [91–157] | 24.9 [19.4–30.3] |
| vLLM (chunk 512) | 4 | 89.4 [75.5–103.3] | 651 [646–656] | 709 [703–715] | 43 [36–49] | 71 [39–102] | 430 [418–442] | 1.6 [0.1–3.1] | 265 [106–424] | 22.4 [18.5–26.3] |
| vLLM (chunk 512) | 8 | 137.8 [123.3–152.3] | 814 [779–849] | 1165 [1138–1192] | 50 [48–51] | 86 [52–120] | 466 [446–486] | 9.5 [0.7–18.3] | 178 [100–257] | 16.4 [13.4–19.4] |

#### Interference (P-2 users streaming; a 4,936-token prompt, then a 280-token prompt 100 ms later)

| Engine | runs | long TTFT ms | short TTFT ms | running users' gap before, p50 ms | gap during prefill p50 / p99 / max ms | running tok/s during prefill |
| --- | --- | --- | --- | --- | --- | --- |
| zerv, pack 8 | 6 | 6078 [6024–6125] | 1421 [1418–1466] | 43 [42–45] | 150 [148–151] / 171 [170–172] / 171 [170–172] | 39.5 [38.8–40.4] |
| zerv, pack 1 | 6 | 6247 [6070–6462] | 1472 [1420–1553] | 45 [42–47] | 151 [148–154] / 171 [170–173] / 172 [171–176] | 39.5 [39.0–39.9] |
| vLLM (chunk 2048) | 6 | 5199 [4990–5939] | 5098 [4890–5838] | 44 [44–45] | 1277 [1249–1448] / 1967 [1961–2119] / 1967 [1961–2119] | 5.0 [4.9–5.2] |
| vLLM (chunk 512) | 6 | 5490 [5446–5555] | 5676 [5629–5743] | 45 [44–46] | 509 [504–514] / 626 [617–632] / 626 [617–632] | 12.7 [12.3–13.5] |

#### Queue (P users streaming, 2 more arrive with no slot free)

| Engine | runs | queued TTFT ms | queued TTFT after the first slot frees, ms | running gap p50 / p99 / max ms | stalled % |
| --- | --- | --- | --- | --- | --- |
| zerv, pack 8 | 6 | 23726 [23317–24095] | 489 [283–736] | 48 [48–48] / 82 [50–144] / 172 [167–176] | 1.0 [0.7–1.2] |
| zerv, pack 1 | 6 | 23889 [23045–24971] | 507 [297–816] | 49 [48–50] / 146 [144–150] / 195 [189–199] | 2.0 [2.0–2.1] |
| vLLM (chunk 2048) | 6 | 25010 [24863–25154] | 394 [265–508] | 50 [50–50] / 56 [55–56] / 703 [676–719] | 0.2 [0.2–0.2] |
| vLLM (chunk 512) | 6 | 25217 [24827–25616] | 390 [296–524] | 51 [50–51] / 58 [55–60] / 376 [344–422] | 0.4 [0.4–0.4] |

Notes:
- vLLM's 1–2-user numbers vary across rounds (19–35 tok/s at 1 user), as seen before.
  Its best rounds are 25–35 tok/s against zerv's 47.
- zerv's prefill share of GPU time dropped from 12.1% to 10.7–10.8% with packing (batcher logs).

## Next

- **Steady p99.** A chunk could wait briefly for more pending prompts to pack (a knob
  trading first-prompt TTFT for fewer stalls). Or the stall budget could be matched to
  vLLM's pattern for p99-focused deployments; the budget sweep already exists.
- **Throughput.** The batched decode step (8 rows: 46 ms FP32) remains the limiter at 8+
  users; see [decode v2](2026-09-26-decode-v2.md) for why the lever is the exact FP32
  multi-row kernel.
