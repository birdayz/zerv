# Slots and batched decode in the model (block 18b.2) — 2026-09-25

**Question.** Can one command decode several sequences at once, each row bit for bit what
the sequence gives when decoded alone? Is the single-sequence path unchanged? What does a
batched step cost?

**Answer.**
- Yes: 399 of 399 logits rows are bitwise equal in each of four KV configurations.
  `zerv-batch-check` covers slots other than 0, staggered joins and leaves, row orders
  permuted every step, B = 1..8 and non-identity page tables.
- The single-sequence path is unchanged: both oracles byte-identical, all checks equal, and
  decode within ±0.06%.
- In the model (not yet served), a batched step of 4 sequences runs at 158.6 tok/s and one of
  8 at 172.8, against 49.8 for one.

## Change ([spec](../specs/concurrent.md), "18b.2 design")

- **Layout:**
  - `layout.stateWith(.., slots, kv_pages)`: slot s's state starts at `s * slot_words`, and
    slot 0 keeps the old layout.
  - A KV pool of `pages` physical pages; each sequence needs at most `seq_pages`.
  - `layout.actWith(.., verify, slots)`: one page table of `ptab_words` entries per slot.
  - io gains a single-slot entry at word 68 and a batch table of up to 32 four-word entries
    (position, page-table offset, state offset, reserved).
- **Kernels:**
  - The decode family gains push words `slots, slot_rs`: `qkprep`, the five attention passes,
    `conv` and `delta`.
  - The prefill family gains `slots`: `qk_b`, `flash`, `conv_b` and `delta_b`.
  - `slots = 0` means no io read and offsets 0, which is how every command is recorded with one
    slot.
  - With `slot_rs = 4`, row r reads its own position, page table and state from the batch table,
    and conv/delta run one row per workgroup.
  - The per-row arithmetic is the same expressions.
  - 19 modules changed.
- **matvec:** `RowsPipeline.projectionSpan` and `recordAt`, and the same for the fused SwiGLU rows.
  A span of up to 32 rows is validated once; groups of at most 5 rows (at most 4 or 2 when fused)
  are dispatched at row offsets. The push offsets are words; a first version shifted them by
  bytes and the new GPU test caught it.
- **Runtime:**
  - `Options.slots`, `batch_rows` and `kv_pages`.
  - `select(slot)`: the existing single-sequence operations act on the selected slot, and each
    slot keeps its own position.
  - Per-slot `reset`, and the snapshots act on the selected slot.
  - `mapPages` / `releasePages` / `mappedPages`: the model owns page ownership, so a pool page
    belongs to one slot, and it checks that every KV write and read is at a mapped position
    (`PagesMissing`, `PageInUse`).
  - `decodeBatch(rows)` with the batch commands for B = 1..`batch_rows`.
  - Projections go in balanced row groups (`RowGroups`).
  - More than one slot requires the MTP off (`MtpNeedsOneSlot`) until 18d.
- **Tools:** `bench/batch_check.zig` (`zig build batch-check-build`).

## Correctness (all run)

| Gate | Result |
| --- | --- |
| `zig fmt`, `zig build test` | clean, 89/89 (a new layout test for slots, pool and io entries) |
| gpu-test, ReleaseFast under the spill gate | 36/36, 0 FAIL, 0 WARN. New: multi-row matvec over an 11-row span in groups at row offsets equals the single-row module on every fixture (17.6M values, plain and fused) |
| `zerv-batch-check` f32 KV / 128 pages | 399/399 bitwise; 8 batch sizes seen; error cases correct |
| same, f16 KV / 128, f32 KV / `context` pages, f16 KV / 256 | 399/399 each |
| `verify_model.py` default oracle (modes 0/1/13/512/512:17) | 0 failures, 48 data files **identical** to `2026-09-24-stateout-default` |
| long oracle (0/13/128/512/512:300) | 24 files **identical** |
| long oracle f16 prefill + native GEMM (512/256); f16 KV (0/512) | **identical** to the 18b.1 captures |
| `zerv-spec-check` f32 and f16 KV | 11/11 each |
| `zerv-mtp-check`, `zerv-prefix-check` fp32/f16 | 19/19 dumps and data lines identical; the gate passes |

What `zerv-batch-check` does (`bench/batch_check.zig`, raw output in
[data](data/2026-09-25-batched-decode/batch-check/)):

- **Phase A (the one-slot model):** 8 sequences with prompts of 40, 61, 127, 128, 200, 255,
  511 and 700 random tokens, then 20 teacher-forced steps each. The prefill and step logits
  are the reference, from exactly the production single-sequence configuration.
- **Phase B (an 8-slot model with batch commands for 1..8 rows):**
  1. Prefill and steps in slots 0, 3 and 7 equal the reference.
  2. Sequence i joins at global step 2i: it is prefilled into slot σ(i) while the others are
     mid-decode. Every global step batches all active sequences in a new random row order,
     B = 1..8 (6–8 rows run as two projection groups).
  3. Step 2 again after all pages are released and remapped to a random permutation of a
     133-page pool.
  4. Error cases.

## Speed

**Single-sequence path, ABBA against the pre-change profiler** (`race_profile.py`,
[data](data/2026-09-25-batched-decode/)):

| Decode step (ms) | before | after |
| --- | --- | --- |
| 4k context, 600 prompt, 3 rounds | 19.673 [19.649, 19.696] | 19.684 [19.649, 19.716] |
| 32k context, 30000 prompt, 2 rounds | 24.626 [24.581, 24.662] | 24.611 [24.586, 24.638] |

Prefill totals are within ±0.06%.

**Batched decode** (`zerv-batch-check` timing section, f32 KV, 128-token pages, all 8 slots
at position ~700–740, median of 5 wall-clock `decodeBatch` calls; a model-level timing, not
a serving benchmark):

| Rows B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms per step | 20.06 | 21.17 | 23.03 | 25.23 | 33.62 | 40.98 | 43.64 | 46.29 |
| tok/s | 49.8 | 94.5 | 130.3 | 158.6 | 148.7 | 146.4 | 160.4 | 172.8 |

The other KV configurations agree within 1% (f16 KV: 158.7 at B = 4, 174.2 at B = 8).

**Interpretation.**
- The step is dominated by the projections, and each projection group rereads all weights.
  Up to 4 rows one group costs little more than one row: 20 → 25 ms.
- 5 rows use the 5-row module, which is 1.5× the single-row cost (the known saturation of the
  FP32 multi-row kernel, [rows scaling](2026-09-24-concurrency-baseline.md)).
- 6–8 rows use two groups, so the weights are read twice.
- Per-row attention, conv and delta add about 1.3 ms per row at this position.
- More sequences per weight pass need 18e (a multi-row kernel efficient beyond 4 rows, or the
  WMMA mode).
- For comparison, llama-server reached 156 tok/s at 8 concurrent clients in the
  [baseline](2026-09-24-concurrency-baseline.md), but that is a serving measurement. The serving
  comparison comes with 18c.

## Next

18c: the scheduler and `--parallel N` serving over this backend, with its gate (concurrent
requests byte-identical to solo) and the serving comparison at 1/2/4/8 clients.
