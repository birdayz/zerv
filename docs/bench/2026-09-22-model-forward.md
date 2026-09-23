# Native Qwen3.8 forward pass — oracle gates and step timing (blocks 09–10)

2026-09-22. **Result: the native Vulkan forward pass passes every declared gate**
against an independent FP64 reference on two teacher-forced sequences (48 and 221
tokens, thinking off/on), with greedy argmax identical at all 269 positions and
bit-identical outputs across runs. Decode step ≈ 23 ms (≈43 tok/s) at short context.
No serving-speed claim here; see the [serving benchmark](2026-09-22-serving.md).

## Oracle (block 09)

[Execution semantics](../research/qwen35-execution.md). `tests/reference/generate_model_oracle.py`:
- renders two official-template chats with the pinned Jinja2 (hash-checked venv),
- captures 33 named per-layer tensors + logits per token from the pinned libllama
  build 10964 (Vulkan, one token per decode, `GGML_VK_DISABLE_MMVQ=1`, F32 KV,
  flash attention off; greedy continuation of 24/40 tokens),
- runs an independent NumPy FP64 forward (HF modeling + converter semantics, own
  vectorized dequantizers cross-checked bit-exact against the earlier scalar decoders
  on 155,648 values) over the same tokens.

libllama vs FP64 (fixture `llama_vs_fp64`): worst intermediate normalized L2
6.35e-6 (short) / 1.05e-5 (long), logits 2.05e-6 / 2.36e-6, no argmax disagreement.
This confirms RoPE pairing, attention scaling/GQA, DeltaNet head tiling, conv tap
order, norm/gate conversions and residual structure before native code was judged.
Minimum FP64 top-1/top-2 margin: 0.122 (short) and 0.0092 (long).

Replay: a fresh second generation produced identical tokens, reference tensors,
logits and metrics; the fixture differed only in two volatile fields (hashes of the
libllama stderr logs, which contain timings, and the absolute work-directory path in
the recorded build command). The committed fixture is from run `2026-09-22-a`.

Failures retained: the first generation attempt stopped because Jinja2 is not
installed system-wide (fixed by rendering through the existing hash-verified venv);
a second was cancelled because derived ggml views (`Kcur-3 (view)`) broke the strict
name check (loader now skips them); the single-threaded reference BLAS (2.7 GFLOP/s)
made the FP64 pass take >25 min, fixed with a row-parallel worker pool (identical
results on the 7-token probe: worst 9.14e-6 before and after).

## Native gates (block 10)

`python3 tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-22-a
--work-dir third_party/model-native/NEW --report docs/bench/data/.../report.json`
rebuilds `zerv-model-capture` (ReleaseFast) and applies `tests/reference/compare_model.py`
with the thresholds declared in [the spec](../specs/model.md) before any native-vs-FP64 run.

| Case | Captured tensor samples | Worst ratio to bound | Logits nL2 (bound) | Greedy = FP64 | Near ties |
|---|---:|---:|---:|---:|---|
| short-nothink (48 tok) | 1,362 names × 48 | 0.755 (`attn_output`) | 3.53e-6 (8.21e-6) | 48/48 | none |
| long-think (221 tok) | 433 names × 221 | 0.260 (`attn_gated`) | 2.50e-6 (9.43e-6) | 221/221 | none |

Embedding rows are bit-exact. Native error is comparable to libllama's own FP32
error (e.g. long case `ffn_out` 1.01e-5 native vs 1.05e-5 libllama). Plain steps
equal capture steps bit-for-bit, a reset replays identical logits, token ≥ vocab
and a full context (after 976/803 extra steps) are rejected, and a reset after a
full context reproduces position 0. Two complete gate runs (`gate4`, `gate5`)
produced bit-identical tensors/logits. Reports: [run1](data/2026-09-22-model-gate/report-run1.json),
[repeat](data/2026-09-22-model-gate/report-repeat.json); native manifests/summaries
under [native/](data/2026-09-22-model-gate/native/).

An earlier native-vs-libllama smoke comparison (short case, before the FP64
reference existed) gave the same picture (≤1.1e-5, logits ≤4.6e-7); it was not used
to set thresholds.

## Step timing (submit → fence, one token)

| Run/case | Steps | Median ms | Min | Max |
|---|---:|---:|---:|---:|
| gate4 short | 96 | 22.96 | 22.58 | 29.05 |
| gate4 long | 442 | 23.39 | 22.21 | 35.87 |
| gate5 short | 96 | 22.77 | 21.92 | 29.06 |
| gate5 long | 442 | 23.30 | 22.30 | 33.57 |

≈15.8 GB of weights read per step ⇒ ≈0.68 TB/s effective; ~820 dispatches and ~570
barriers per step. Load (validate 16 GB + upload) took 30–34 s here because the FP64
replay saturated the CPU concurrently; 6.6 s on an idle machine. Prefill currently
reuses the one-token step (no batching) — the dominant serving gap.

## Limitations

Two prompts of one artifact; context ≤ 1024 in the gate runs (attention at long
contexts is exercised only by the serving benchmark, not by an FP64 comparison).
FP64 reference is HF-semantics NumPy, not PyTorch/BF16 execution; quality relative to
the BF16 checkpoint is unmeasured.
