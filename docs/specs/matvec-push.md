# Matvec ≥1.30× reference push — block08c

User explicitly resumed optimization and requires unchanged behavior. Serving and
block09 remain paused. Target is reference time / native time >=1.30 on each full
shape; also report the stricter 30%-lower-latency threshold separately. Do not
claim aggregate or universal success from a subset of shapes. Keep the same FP32
inputs, exactly decoded weights, all48 independent fixtures (both alignments),
all11 complete dense shapes and the existing exact/mixed-error acceptance gates.
No dropped work, pruning, activation quantization or precision/tolerance changes.
08b final-repeat source snapshot is the frozen counterfactual. Changes are not
accepted merely because a diagnostic operation is faster than matvec.

## Research and bounded diagnostic stage (before code)

Q6 reads1,042,944,000 packed bytes for one vector. At1.253ms the1.30× target is
0.964ms and implies1.082TB/s of packed payload, before input/output/metadata traffic.
This is an estimate, not a measured bandwidth limit. A lossless-representation
feasibility scan is allowed, not silently reducing bit depth. Initial full Q6
coefficient histogram has5.792 bits/symbol entropy,30.8% outside[-16,15], no zero
subscales/global blocks; simple sparse/5-bit exceptions are unattractive. Histogram
entropy is not an impossibility proof for higher-order compression.

Measure separately:
- Uninstrumented existing submit/fence wall time (unchanged benchmark boundary).
- Diagnostic GPU timestamps around the same Plan dispatch, bracketed at TOP and
  BOTTOM of pipe. Query queue-family timestampValidBits and device timestampPeriod;
  reject unsupported/invalid values. Mask wraparound before converting to ns.
  Reset two queries inside the reusable command, read64-bit available results only
  after a completed fence; no host query-reset feature needed. Query pool outlives
  its command and is destroyed before the device. Instrumentation is test/bench
  code, not a new production GPU API or a replacement performance denominator.
- A raw-weight streaming/XOR checksum shader over the same full packed payload.
  Independent Python uint32 XOR per row gates every output, and all output guards
  remain checked. This deliberately omits matvec arithmetic: it is a diagnostic
  read-throughput floor, NEVER a candidate matvec or claimed inference speedup.
  Require4-byte-aligned views and row byte counts for this scoped probe.

Pinned Khronos query specification/header define the timestamp semantics/ABI.
Generate a separate independent C size/offset/constant fixture before the private
Zig diagnostic bindings; do not mutate historical ABI fixtures or golden sources.
Core queries add no optional feature. Test query reset/replay and wrapping math.

## Candidate stage

Explore further packed traversal, accumulator/register pressure, load schedules
and work assignment. Process-local RADV cswave32 can test a compiler hypothesis;
record its environment explicitly, run all independent gates, and never deploy
that environment override as an unqueried production capability. A subgroup-size
production path requires a separate queried/validated capability contract first.
No runtime knob for a losing experiment. Any lossless repacking needs its own
layout spec, exact round-trip/reference decoder gates, explicit setup/memory costs
and native owned implementation before it can be accepted.

Each candidate must pass original fixtures/full shapes, including guards/replay/
changed input; keep failures and source/module/binary hashes. Only verified gains
may change production. Compare current08b, candidate and external FP32 reference
with3 warmups,7 trials,3 alternating rounds, two fresh runs. Keep aligned/stress
placements separate and default-reference precision control. CPU/GPU tests in
both modes, Python, fmt, shader/golden replay and source-rebuildable counterfactual
remain final gates. If the30% target cannot be achieved without behavior changes,
record it as unmet with measured limits rather than manufacturing a win.

## Selected change and exactness contract (after candidate stage)

Accepted only: (1) Q4_0/Q4_1 inner loop issues U=4 blocks of loads before use,
with unchanged per-lane ascending block/accumulator order; (2) weight decode as
`fma(d, q, c)` from unsigned byte-to-FP32 conversion. Dot products stay strict
separate FP32 multiply and add (`precise`), exactly as in 08b.

Bit-identity argument over the specified finite domain. Let the strict decode
compute `round(A*B) (+) C` (or `A*(q-k)`), where A is the already-rounded FP32
scale computed by the same expression as before:
- Q4_0 `d*(q-8)`: d has <=11 significant bits (finite half), |q-8|<=8, so the
  exact value needs <=15 bits; `-8d` is an exact power-of-two scaling. The exact
  result is representable, so `fma(d,q,-8d)` returns it.
- Q4_1 `d*q + m`: d*q needs <=15 bits, hence exact; fma(d,q,m) = round(d*q+m) =
  strict `round(round(d*q)+m)`.
- Q5_K `(d*sc)*q - dmin*mn`: the scale product has <=17 bits, q<=31 (5 bits), so the
  product (<=22 bits) is exact; fma equals strict product-then-subtract.
- Q6_K `(d*s)*(q-32)`: scale has <=18 bits (|s|<=128), q-32 in [-32,31]; the exact
  result needs <=23 bits and `-32*scale` is exact, so fma returns it.
Because each product term is itself exact, a driver that splits a non-fused fma
into multiply+add also returns the same value. Only the sign of an exact zero
decoded weight can differ. Products with finite x are then ±0; an accumulator
that starts at +0 can only become -0 by adding -0 to -0, so by induction it is
never -0 and a zero product never changes it. Reduction sums therefore match
bit-for-bit. Empirically, all 48 fixtures pass and every full-shape output is
byte-identical to 08b. FP32 inputs, rows, reduction trees and gates unchanged.

Rejected, retained in data: multi-row activation-reuse tiles (NR=2/4/8, with and
without exact decode), U=2/8, RADV cswave32 (process-local compiler experiment),
and FMA dot accumulation. FMA accumulation passed the tolerance gates but changes
output bits for ~3-4% GPU time, so it violates the no-behavior-change requirement.

## FMA accumulation — block 17c (specified 2026-09-24, replacing "strict separate multiply and add"; implemented, gates passed, verify table re-tuned — [evidence](../bench/2026-09-24-fma-matvec.md))

- **Change:** every dot-product step of `matvec.comp` and `matvec_rows.comp` becomes
  `sum = fma(w, x, sum)` (one rounding) instead of `round(sum + round(w*x))`. The weight
  decode, the lane partition, the block order per lane, the two partial vectors and the
  reduction tree are unchanged. `accum()` is the single place; both kernels share its
  text.
- **Why now:** 08c rejected it only because it changed output bits under a
  no-behavior-change rule. Since block 10 the criterion is the FP64 model gates. The
  lab measured it at 3–4% for decode and 5-row verify 1.70× → 1.37× a decode pass (with
  G = 4; [spec-verify](../bench/2026-09-24-spec-verify.md)). One rounding per step is also
  at least as accurate.
- **Exactness that remains:** multi-row ≡ single-row, bitwise, per row (the same
  sequence of fused steps), so speculative verify stays ≡ decode. Fixtures whose
  products and sums are exactly representable (isolated columns, cancellation, zero
  input, half domains, row ramp) are unchanged, because fma equals multiply-then-add
  there; the other fixture cases keep their FP64 bounds.
- **Knob (2026-09-24, user rule: changes stay selectable):**
  `Options.matvec_accumulation` / `--matvec-accumulation fma|separate`, default fma.
  `separate` selects the pre-FMA arithmetic (`ACCUM_FMA 0`, modules in
  `src/matvec/shaders/separate/`) with its own verify table (`matvec.rowsGroups`), and
  reproduces the pre-FMA captures and logits byte for byte on both oracles.
- **Gates:** `zig build gpu-test` (all 48 fixtures in both layouts; multi-row ≡ single
  row on all fixtures and shapes); `tools/verify_model.py` FP64 gates on the default and
  long oracles in every mode (captures change; the bounds decide); `zerv-spec-check`
  11/11; `zerv-mtp-check` scenario C and the FP64 MTP reference; `zerv-prefix-check`.
  Then decode step, verify + commit per count, and serving decode-v1 against the
  previous numbers, and a per-count re-tune of GROUP/CB.
