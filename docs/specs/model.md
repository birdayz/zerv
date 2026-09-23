# Native Qwen3.8 forward pass — `src/model` (block 10)

Contract before native model code. Semantics: [execution research](../research/qwen35-execution.md),
verified executable oracle: block 09 (`tests/reference/generate_model_oracle.py`,
[fixture](../../tests/fixtures/model/qwen38-oracle.json)). Serving is block 12; this
block produces logits for one token at a time with persistent per-sequence state.

## Scope and interface

- Artifact: the SHA-verified Qwen3.8-27B-Q4_0 GGUF only (`general.architecture=qwen35`,
  64 trunk layers, the exact hyperparameters in the research note). Any other
  architecture, dimension, tensor type/shape or missing tensor is rejected before any
  GPU allocation. `blk.64.*` (MTP) is not loaded.
- `Model.load(device, container, options)`: validates every used tensor (type, dims,
  byte size, finite halves / FP32 per quant rules), packs trunk tensors into ≤4 GiB
  device-local weight banks (32-byte aligned; small F32 parameters first in bank 0),
  uploads through a bounded host staging buffer, creates shared pipelines and records
  one reusable decode-step command. `options.context` (1..max) sizes the KV cache;
  max is bounded by one ≤4 GiB state buffer (FP32 KV, 131072 B/token).
- `Model.step(token, position) → logits` (one token; FP32 logits in host-visible
  memory, borrowed until the next step). Positions must be consecutive from 0 within
  a sequence; `reset()` clears recurrent/conv state (KV beyond position is ignored).
  Out-of-range token/position/context → error, no GPU work. Single sequence, externally
  serialized. No per-step allocation, descriptor update or command re-recording.

## Numerics (FP32 on device)

Weights decode exactly (same exact-FMA decoders as block 08). All activations,
KV cache and recurrent/conv state are FP32. RMS norm uses the GGUF eps (FP32
9.99999997e-7). DeltaNet L2 norm `x/sqrt(Σx²+1e-6)`; softplus stable
(`max(x,0)+log1p(exp(-|x|))`); RoPE cos/sin are computed on the host in FP64 per
position and rounded to FP32 (more accurate than GPU sin/cos at large angles).
Attention: scale 1/16, max-subtracted FP32 softmax. Reduction orders are free;
correctness is judged by the gates below, not bit equality with llama.cpp.

## Correctness gates (declared from the block-09 measurements, before native code)

Native capture tool writes the same format as the libllama capture (named tensors per
token + logits) for the identical token sequences of the oracle fixture.

1. Every captured intermediate, per (name, token): native-vs-FP64 normalized L2 ≤
   `max(4 · E_llama(name), 2e-6)`, where `E_llama(name)` is the worst llama-vs-FP64
   normalized L2 for that name in the same case (fixture `llama_vs_fp64`). No
   non-finite values. `model.input_embed` must be exactly equal.
2. Logits at every position: normalized L2 ≤ `max(4 · max_t E_llama(logits), 2e-6)`.
3. Greedy agreement: wherever the FP64 top-1/top-2 logit margin exceeds
   `M = 10 · max_t max_abs(llama − FP64 logits)` (same case), native argmax equals
   FP64 argmax. Near-tie positions are reported, never silently accepted as proof.
   Declared 2026-09-22 before any native-vs-FP64 comparison was run. (A native-vs-
   libllama smoke comparison on the short case had already been run while the FP64
   reference was computing: all intermediates ≤1.1e-5 normalized L2, logits ≤4.6e-7,
   argmax equal at all 48 positions; it is not used to choose these thresholds.)
4. Stateful paths: capture run matches a plain (no-capture) step run bit-for-bit on
   logits; `reset()` followed by re-running a sequence reproduces identical logits.
5. Negative tests (driver-free where possible): wrong architecture/dims/types/missing
   tensors, nonfinite scales, position/token/context bounds, bank packing limits.

Oracle cases (the same rules apply to each):

- **Default fixture** (`qwen38-oracle.json`): short-nothink (48 tokens, all
  intermediates) and long-think (221 tokens).
- **Long fixture** (`qwen38-oracle-long.json`, added in 13h): long-prefill,
  546 prompt tokens + 16 generated, capturing l_out, attn_output and result_norm.
  It is the only case that runs a full 512-row prefill chunk.
- Run it with `tools/verify_model.py --fixture tests/fixtures/model/qwen38-oracle-long.json
  --oracle-dir third_party/model-oracle/2026-09-23-long --modes 0,512,64`.
- Changes to 512-row chunks or plans must pass it.

## Performance obligations

Component timings per step (GPU timestamps and submit→fence wall) for decode at
several positions; weight bytes/step and implied bandwidth; dispatch/barrier counts.
Actual tuned llama-server comparison happens at the session milestone (block 11).

## Decode step overheads — block 13c (specified 2026-09-22, before implementation; implemented and verified, [evidence](../bench/2026-09-22-decode-overheads.md))

**Measured** (per-phase profile, decode at position ~40, 21.4 ms/step): weight-streaming
phases run at ~730–910 GB/s (a raw read probe measured ~920 GB/s). A barrier plus tiny
dispatch costs ~1.7–1.9 µs (qkprep, conv and attention-combine phases), so ~610 links
account for only ~1.2 ms. Two kernels are disproportionately slow:
- **RMS norm:** 10.4 µs per call, 1.34 ms for 129 calls.
- **DeltaNet decode:** 21 µs per layer, 1.0 ms for 48 layers, against ~7 µs of state
  traffic.

**Cause (from the source):** both loop over loads and stores to the same storage
buffer. The compiler must assume aliasing, so it cannot issue later loads before earlier
stores, and each iteration pays full memory latency:
- **Norm:** each iteration loads x and a, stores the sum, then a second pass re-reads
  the sum.
- **DeltaNet:** each thread reads its 128-value state column twice, and the second loop
  interleaves loads and stores.

**Change:** load first into registers, then compute in the same order, then store:
- **Norm:** up to 32 values per thread (width ≤ 8192, validated on the host).
- **DeltaNet:** the column is held in registers, as the prefill scan already does.

Every arithmetic expression and summation order is unchanged. **Gate: bit-identical
outputs.** Every capture file of `verify_model` modes 0/1/13/29/60/512 must be
byte-identical to the 13e gate run (`third_party/model-native/2026-09-22-scalarx-gate2`).
The CPU/GPU tests and the serving greedy check must also pass.

**Measurement:** per-phase profile (norm and delta phases, step time) and a serving
benchmark, two runs.
