# Matvec push toward ≥1.30× reference — 2026-09-22 (block08c)

**Result: target met on some shapes, not all.** One bit-identical kernel change
(unrolled Q4 loads + exact FMA weight decode) is in production. Every output is
**byte-identical** to 08b on all 11 full shapes (66/66 per run) and passes all 48
independent fixtures in both layouts. Against the FP32-input reference:

- **GPU kernel time** (device timestamps; what counts inside a model graph): ≥1.30×
  on 5 of 11 shapes (F32 ~3.0×, Q4_0 5120×1024 ~2.9×, Q4_1 1.44×, Q4_0 6144×5120
  ~1.39×, Q4_0 5120×6144 1.29–1.30×), Q5_K aligned 1.27–1.28×, other Q4_0
  1.08–1.18×, Q6_K 1.03×.
- **Submit-to-completion wall time** (the earlier benchmark boundary): ≥1.30× on 3
  of 11 (Q4_1, F32, Q4_0 5120×1024), 1.03–1.22× elsewhere.

**Not achieved:** a uniform 30% win. Evidence for the limits is below. No
serving/model claim; serving and block09 remain paused.

## Why some shapes cannot reach 30%

1. **Fixed per-call latency.** Both engines spend ~40–55µs outside the kernel per
   standalone synchronous call (wall minus GPU timestamp, e.g. Q4_0 5120×6144: native
   74µs wall / 34µs GPU; reference 84µs / 44µs). For small matrices this caps any
   wall-clock ratio. A model decode step submits hundreds of dispatches at once, so
   GPU time is the operative per-matvec number there; that path is not built yet.
2. **Q6_K is at the memory-read limit.** A raw XOR read probe over the same
   1,042,944,000 bytes takes ~1133µs (~920GB/s); the matvec takes ~1166µs (97%).
   1.30× over the reference (1200µs GPU) needs ~923µs ⇒ ~1.13TB/s, above what this
   card delivered when simply reading those bytes. Only reading fewer bytes could
   help; order-0 coefficient entropy is 5.79 of 6 bits
   ([scan](data/2026-09-22-matvec-push/entropy.json)), i.e. ≤~3.5% even for an
   ideal lossless code that the GPU could not decode cheaply. Not pursued.
3. **Larger Q4_0 shapes are instruction/issue-limited**, ~2× above their raw-read
   probe (e.g. 82.8 vs 40µs). Tried and rejected, all correctness-gated: multi-row
   activation-reuse tiles (NR 2/4/8), unroll U=2/8, RADV `cswave32`, and FMA dot
   accumulation (passes tolerances, ~3–4% faster, but **changes output bits**).
   Factoring scales out of the dot (the reference's approach) also changes
   rounding, so it was not adopted under the no-behavior-change requirement.

Caveat on realism: every shape except Q6 is small enough that repeated calls read
it from on-die cache (probe rates of 0.96–1.4TB/s on those matrices exceed the
~920GB/s full-Q6 probe). In real decoding each matrix is read once per token from
DRAM; both engines benefit equally here, so the comparison is fair, but absolute
per-matvec times will differ in a model. A cold-cache benchmark is future work.

## Selected change

`src/matvec/matvec.comp`: Q4_0/Q4_1 issue four blocks' loads per trip with the
per-lane accumulation order unchanged; all quant formats decode via
`fma(scale, unsigned byte value, offset)`. The exactness argument (each decoded
value is exactly representable and its product term is exact, so fused or split
evaluation equals the old strict decode; zero signs cannot change a +0-started sum)
is in [the spec](../specs/matvec-push.md). Dot products remain strict FP32
multiply-then-add. F32 modules are byte-unchanged.

## Results (median µs; ratios are reference/native, higher is better)

Wall: submit→fence, 21 trial means/engine/run (3 warmups, 7 trials, 3 alternating
rounds). GPU: TOP/BOTTOM-of-pipe timestamps around the same dispatch, 32 samples;
reference GPU from its own `GGML_VK_PERF_LOGGER` (adds its own timestamps/barriers;
diagnostic). "stress" = original offset-2 weights; "aligned" = offset 0.

### run1

| Shape | 08b wall | New wall (stress) | New wall (aligned) | Ref wall | Wall ratio | New GPU (aligned) | New GPU (stress) | Ref GPU | GPU ratio | Read probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_k-5120-248320 | 1216.5 | 1210.4 | 1210.8 | 1247.3 | 1.03× | 1165.1 | 1166.9 | 1199.6 | 1.03× | 1132.8 |
| q4_0-5120-6144 | 81.9 | 76.8 | 74.3 | 84.1 | 1.13× | 33.6 | 34.1 | 43.4 | 1.29× | 23.3 |
| q4_0-5120-10240 | 112.3 | 98.5 | 98.1 | 113.9 | 1.16× | 51.9 | 52.2 | 61.0 | 1.18× | 31.6 |
| q4_1-17408-5120 | 140.8 | 126.1 | 122.0 | 167.2 | 1.37× | 85.6 | 91.9 | 123.1 | 1.44× | 39.4 |
| q4_0-5120-17408 | 135.3 | 119.9 | 119.8 | 135.3 | 1.13× | 84.3 | 85.0 | 91.3 | 1.08× | 46.4 |
| f32-5120-48 | 50.3 | 50.2 | 50.3 | 67.0 | 1.34× | 10.3 | 10.3 | 31.0 | 3.00× | 10.3 |
| q5_k-6144-5120 | 100.3 | 98.9 | 79.1 | 85.4 | 1.08× | 39.0 | 51.7 | 49.4 | 1.27× | 26.5 |
| q4_0-5120-1024 | 56.3 | 51.8 | 51.8 | 67.8 | 1.31× | 11.5 | 11.6 | 33.5 | 2.92× | 12.7 |
| q4_0-6144-5120 | 89.8 | 76.1 | 75.5 | 84.2 | 1.11× | 32.8 | 33.0 | 45.3 | 1.38× | 21.4 |
| q4_0-5120-12288 | 120.3 | 107.3 | 106.6 | 116.4 | 1.09× | 61.2 | 61.8 | 70.3 | 1.15× | 37.0 |
| q4_0-17408-5120 | 141.8 | 122.1 | 123.1 | 149.4 | 1.22× | 82.8 | 83.4 | 96.7 | 1.17× | 40.1 |

### repeat

| Shape | 08b wall | New wall (stress) | New wall (aligned) | Ref wall | Wall ratio | New GPU (aligned) | New GPU (stress) | Ref GPU | GPU ratio | Read probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q6_k-5120-248320 | 1214.9 | 1211.0 | 1210.6 | 1248.1 | 1.03× | 1167.7 | 1167.2 | 1200.8 | 1.03× | 1132.5 |
| q4_0-5120-6144 | 81.3 | 76.6 | 74.1 | 84.0 | 1.13× | 33.5 | 33.9 | 43.7 | 1.30× | 23.1 |
| q4_0-5120-10240 | 111.9 | 98.3 | 98.3 | 113.7 | 1.16× | 51.8 | 52.2 | 61.3 | 1.18× | 31.8 |
| q4_1-17408-5120 | 142.3 | 126.9 | 124.4 | 167.4 | 1.34× | 85.6 | 91.8 | 122.9 | 1.44× | 39.6 |
| q4_0-5120-17408 | 135.4 | 120.5 | 119.3 | 135.2 | 1.13× | 84.2 | 84.9 | 91.2 | 1.08× | 46.4 |
| f32-5120-48 | 50.3 | 50.6 | 50.5 | 67.0 | 1.33× | 10.1 | 10.2 | 31.0 | 3.06× | 10.1 |
| q5_k-6144-5120 | 100.3 | 98.3 | 78.8 | 84.7 | 1.08× | 39.0 | 51.8 | 49.8 | 1.28× | 26.4 |
| q4_0-5120-1024 | 55.4 | 51.2 | 51.4 | 67.5 | 1.32× | 11.4 | 11.7 | 33.6 | 2.93× | 12.8 |
| q4_0-6144-5120 | 89.1 | 75.8 | 75.5 | 84.2 | 1.12× | 32.7 | 33.0 | 45.8 | 1.40× | 21.5 |
| q4_0-5120-12288 | 122.6 | 107.7 | 107.4 | 116.0 | 1.08× | 61.2 | 61.7 | 70.6 | 1.15× | 36.8 |
| q4_0-17408-5120 | 140.7 | 123.8 | 123.1 | 149.4 | 1.21× | 82.8 | 83.3 | 96.6 | 1.17× | 39.9 |

Wall ratio uses the faster native placement. Q5_K and Q4_1 stress placements are
slower (Q5 GPU ~52µs, 0.96× reference) because two-byte-aligned rows cannot use the
direct-word path; aligned views are the expected model layout.

## Verification

- 42 CPU + 10 GPU tests (Debug and ReleaseFast), 49 Python tests, fmt: pass. New
  GPU tests cover timestamp query reset/replay/lifetime around an independently
  checked matvec; the private timestamp ABI matches an independent C-header fixture
  generated twice byte-identically.
- All 48 fixtures × both layouts; 380928 exact finite-half outputs per layout.
- Both paired runs: all native outputs byte-identical to the source-rebuilt 08b
  counterfactual; max error/sumabs 1.136e-8 (unchanged); default-reference
  normalized L2 still up to 0.00407 (precision control, not a denominator).
- Fresh replay regenerated the independent fixtures, timestamp ABI fixture and all
  8 SPIR-V modules byte-identically and reran all suites
  ([logs](../research/2026-09-22/matvec-push-replay/)).
- 138 snapshot sources/run match the live tree; 542 artifacts/run rehashed
  ([audit](data/2026-09-22-matvec-push/final-hash-verification.json)).
- Raw-read probe outputs checked against an independent NumPy per-row XOR.

## Commands

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
python3 bench/rebuild_matvec_baseline.py --run docs/bench/data/2026-09-22-matvec-final-repeat \
  --output third_party/matvec-push/rebuilt-08b
python3 bench/run_gpu_matvec.py --cpu 10 --aligned --baseline-run third_party/matvec-push/rebuilt-08b \
  --output docs/bench/data/2026-09-22-matvec-push-run1        # and -repeat
python3 bench/profile_matvec.py --output docs/bench/data/2026-09-22-matvec-push-gpu-run1
python3 bench/profile_reference_matvec.py --run docs/bench/data/2026-09-22-matvec-push-run1 \
  --output docs/bench/data/2026-09-22-matvec-push-refgpu-run1
python3 tools/replay_matvec.py --output-dir .tools/matvec-push-replay
```

Candidate experiments: `bench/tune_matvec.py` (now also records GPU timestamps and
supports per-format rows per group); [ledger of 18 push experiments](data/2026-09-22-matvec-push/experiments.json),
sources under `data/2026-09-22-matvec-push/source/`. Raw data:
[run1](data/2026-09-22-matvec-push-run1/), [repeat](data/2026-09-22-matvec-push-repeat/),
[GPU run1](data/2026-09-22-matvec-push-gpu-run1/), [GPU repeat](data/2026-09-22-matvec-push-gpu-repeat/),
[reference GPU run1](data/2026-09-22-matvec-push-refgpu-run1/), [repeat](data/2026-09-22-matvec-push-refgpu-repeat/).
Same pinned model, reference build, driver and tools as the [08b report](2026-09-22-matvec-optimization.md).
