# Native Qwen3.8 forward pass — `src/model` (block 10)

Contract before native model code. Semantics: [execution research](../research/qwen35-execution.md),
verified executable oracle: block 09 (`tests/reference/generate_model_oracle.py`,
[fixture](../../tests/fixtures/model/qwen38-oracle.json)). Serving is block 12; this
block produces logits for one token at a time with persistent per-sequence state.

## Scope and interface

- Artifact: the SHA-verified Qwen3.8-27B-Q4_0 GGUF only (`general.architecture=qwen35`,
  64 trunk layers, the exact hyperparameters in the research note). Any other
  architecture, dimension, tensor type/shape or missing tensor is rejected before any
  GPU allocation. `blk.64.*` (MTP) is loaded only with `Options.mtp`
  ([speculative decoding](speculative.md)).
- `Model.load(device, container, options)`: validates every used tensor (type, dims,
  byte size, finite halves / FP32 per quant rules), packs trunk tensors into ≤4 GiB
  device-local weight banks (32-byte aligned; small F32 parameters first in bank 0),
  uploads through a bounded host staging buffer, creates shared pipelines and records
  one reusable decode-step command. `options.context` (1..max) sizes the KV cache
  (FP32, 131072 B/token for the 16 trunk layers).
- **KV buffers** (2026-09-24, user request: the single ≤4 GiB state buffer capped the
  context at 29,523 tokens regardless of free VRAM).
  - The attention KV caches live in their own device buffers, not in the state arena.
    Cache i covers the 16 trunk attention layers, then the MTP layer's.
  - Since 18b.1 ([concurrent.md](concurrent.md), "Addressing";
    [report](../bench/2026-09-24-paged-kv.md)) the caches are paged. Every page of P tokens
    (`--kv-page-tokens`, default 128; `context` = one page) holds, per layer, K [kv
    head][dim][P] then V [kv head][P][dim]. A token reaches its page through the page table
    `Act.ptab`, which is the identity with one sequence.
  - Cache i goes to KV buffer i / c. Within every page it sits at element (i mod c) ×
    `page_piece`, and pages are `pstride` = (layers in that buffer) × `page_piece` apart.
    Here c = floor(buffer capacity / cache bytes). The capacity is the bank capacity
    (3.75 GiB), or `Options.kv_capacity` to force a split in tests.
  - The attention kernels that touch the cache (`qkprep`, `qk_b`, `attn_scores`,
    `attn_pv`, and the fused prefill attention `flash`) exist once per KV buffer, with
    that buffer bound where the state arena was (binding 2). The shaders are unchanged:
    those kernels reach nothing else through that binding.
  - The state arena keeps the recurrent and conv state and the MTP's pending h.
  - Max context:
    - one cache must fit one buffer (491,520 tokens at 3.75 GiB);
    - the activation arena must stay below 4 GiB. Its context-dependent term is the
      decode/verify score region, 24 heads × verify rows × context (prefill attention is
      fused and has no score matrix since 16a);
    - everything must fit free VRAM (the startup check).
  - Gate:
    - logits and captured intermediates bitwise identical to the single-buffer build
      (decode and 512-row prefill, default and long oracles), with default buffers and
      with a forced split;
    - a load at the new maximum context and a long-prompt run beyond the old cap.
- **`--context max`** (`Options.context = context_max`, 2026-09-24): `init` places the
  weights, then picks the largest context (a multiple of 32) whose context-dependent
  buffers (KV caches, state arena, activation arena) fit both
  - free VRAM (`VK_EXT_memory_budget`) less every other device buffer, `vram_headroom`
    (256 MiB) and `Options.context_reserve` (`--vram-reserve-mib`, default 1024 MiB: the
    margin llama-server's `--fit-target` uses, for the desktop and other GPU users), and
  - the device allocation cap (`--vram-budget-gib`) less every device and host buffer
    and 16 MiB of alignment slack,
  capped at the trained context (`qwen35.context_length`, 262,144; explicit contexts
  above it are rejected with `ContextTooLarge`). A binary search over multiples of 32 on
  the runtime's own layout functions (`model.fitContext`, tested in `tests/model.zig`).
  Without a driver budget, `max` fails (`VramBudgetUnknown`). Measured table:
  [context max](../bench/2026-09-24-context-max.md).
- **Decode FFN fusion** (`Options.decode_fusion`, `--decode-fusion on|off`, default on;
  2026-09-24): where a layer's `ffn_gate` and `ffn_up` share a bank, format and shape,
  the decode step runs one `matvec.SwigluPipeline` dispatch (workgroup i: row i of both,
  the single-row arithmetic, then g, u and silu(g)·u) instead of two projections, a
  barrier and the swiglu kernel. Values are identical; off records the separate path.
  Gate: byte identity on both oracles, spec-check with both settings
  ([report](../bench/2026-09-24-decode-fusion.md)).
- **DeltaNet state store** (`Options.delta_state_out`, `--delta-state-out on|off`, default
  on; 2026-09-24): `K_DELTA`/`K_DELTAB` store the final state at the push constant
  `state_out` (== `ssm`), so the compiler does not keep 120 load addresses live across
  the row loop (128 VGPRs spilled to scratch before, 7–9 now). Off selects the previous
  modules (`delta_legacy`, `delta_b_legacy`). Values are identical: byte identity on both
  oracles ([report](../bench/2026-09-24-delta-spill.md)).
- **Verify FFN fusion** (`Options.verify_fusion`, `--verify-fusion on|off`, default on;
  2026-09-24): the same fusion in the speculative verify pass, for the same layers.
  `matvec.SwigluRowsPipeline` (matvec_rows.comp with `SWIGLU`): a workgroup takes GROUP
  gate rows and the same GROUP up rows (2 GROUP weight rows sharing each X load), with
  the multi-row arithmetic per row, then writes g, u and silu(g)·u for every input row
  (the single-row SWIGLU expression). The table `matvec.swigluRowsGroups` /
  `SWIGLU_ROWS_CONFIG` is tuned per row count by an in-model race
  (`tools/race_swiglu_rows.py`); count 5 has no fused module, and Q5_K/Q6_K fused
  modules stop at count 2 (their larger modules spill VGPRs into LDS, which Mesa 26.2.3's
  ACO miscompiles: [RCA](../bench/2026-09-24-aco-lds-spill.md)). The verify pass records
  the separate path for those counts. Values are identical; off records the separate path.
  Gates: `gpu-test` (every quantized fixture, both accumulations and alignments, counts
  1..5: g, u and y bitwise equal to the single-row fused module per row; rows past the
  count untouched) and `zerv-spec-check` 11/11 with verify fusion on and off.
- **Host-resident data** (2026-09-24; knobs for VRAM, measured in
  [host memory](../bench/2026-09-24-host-memory.md)):
  - `Options.embedding_memory` (`--embedding-memory`, default `.host`): the Q4_0 token
    embedding (682 MiB) lives in a mapped host buffer, bound as binding 0 of `embed` and
    `embed_b` only (offset 0); it takes no bank space. The kernels read one 2,880-byte
    row per token over PCIe. Bitwise identical to `.device`; no measured TTFT or decode
    cost (serving-v2, 2 repeats).
  - `Options.snapshot_memory` (`--prefix-cache-memory`, default `.device`): the prefix
    cache's snapshot slots (157 MB each, 8 by default) in VRAM or in host memory. On
    the host, save takes 9.8 ms and load 15.3 ms (device: 0.5 ms each); serving TTFT
    rises by 10–19 ms per request (one or two saves). Frees 1.2 GB of VRAM at 8 slots.
  - The allocation cap (`--vram-budget-gib`) counts device memory; the server adds the
    host buffers to the driver-level cap. The free-VRAM check counts device buffers only.
- `Model.step(token, position) → logits` (one token; FP32 logits in host-visible
  memory, borrowed until the next step). Positions must be consecutive from 0 within
  a sequence; `reset()` clears recurrent/conv state (KV beyond position is ignored).
  Out-of-range token/position/context → error, no GPU work. Single sequence, externally
  serialized. No per-step allocation, descriptor update or command re-recording.

## KV precision — block 17c (specified 2026-09-24, before implementation; implemented: gates 1, 2, 4 and 5 passed, gate 3's worst-token clause not met, so f32 stays the default — [evidence](../bench/2026-09-24-kv-precision.md))

Research: [kv-precision.md](../research/kv-precision.md).

- **Knob:** `Options.kv_type` (`--kv-type f32|f16`), `layout.KvType`. f32 is the exact
  configuration and stays byte-for-byte what it was (all captures and logits bitwise
  equal). f16 is an explicit precision trade: half the KV bytes (VRAM and attention
  reads). The default is decided from the measurements below.
- **Storage:** the same layouts (K `[kv head][dim][page]`, V `[kv head][page][dim]` per
  page, caches packed per KV buffer; before 18b.1 the pieces spanned the context, which
  `--kv-page-tokens context` still selects). Every size and offset is in elements of the KV type
  (4 or 2 bytes). `cacheElements`, `kcache`, `vcache` are element offsets; KV buffer
  bytes = elements × element bytes; `maxContext(capacity, kv)` and
  `per_buffer = capacity / cache bytes` use the element size. The context must be even
  for both types (flash and the decode scores pass read key pairs as one load); with
  prefill it is a multiple of 32 anyway.
- **Writers** (`qkprep`, `qk_b`; `KV16` variants): store `f16(x)` of the same FP32 K
  (normed, roped) and V values the f32 path stores, where `f16` is the device's f32→f16
  conversion. It must round to nearest even; a hardware test checks it, including
  subnormal results. The FP32 copies in the activation arena (`kr`, `kn`, …) are
  unchanged.
- **Readers** (`attn_scores`, `attn_pv`, `flash`; `KV16` variants): load halves and
  convert to FP32 exactly. All arithmetic, the summation order and the grids are the f32
  kernels'. `flash` reads key pairs and 4-dim V rows as raw 32/64-bit words and converts
  them with `unpackHalf2x16` at their use, so its K prefetch stays a pure load (see the
  report: converting at the load made prefill attention 2.1× slower).
- **Device:** `storageBuffer16BitAccess` (SPIR-V `StorageBuffer16BitAccess` only; no
  `shaderFloat16`), requested by `gpu.Device.open(.{ .storage16 = true })`. Without it,
  f16 fails with `UnsupportedDevice`.
- The MTP layer's cache uses the same type. Snapshots are unaffected (no KV).
- **Gates:**
  1. f32: every capture and logits file of `verify_model` (default and long oracles,
     modes 0/1/13/512/512:17) byte-identical to the previous build; `zerv-spec-check`,
     `zerv-prefix-check`, `zerv-mtp-check` pass.
  2. Component (GPU tests): the cache holds exactly RNE-f16 of the FP32 values written
     by `qkprep` and `qk_b`; split-K decode attention and fused prefill attention with f16
     KV against FP64 over the same f16 values, within the f32 tests' bounds.
  3. Oracle quality, matched: `tools/prefill_quality.py` (decode mode 0 and prefill 512):
     zerv f16-KV errors against FP64 (logits and `l_out-63`, mean and max) no worse than
     llama's with FP32 arithmetic and f16 KV (`fp32-kvf16`), and no argmax flips outside
     near ties.
  4. Long context: KL(f32-KV ‖ f16-KV) over 256 teacher-forced decode steps after a
     ~37k-token real-text prefix (`zerv-kv-quality`, and llama's through
     `llama_batch_capture` in decode mode on the same tokens). zerv's mean and p99 KL
     no worse than llama's, within the run-to-run noise measured on llama.
  5. Serving and speed: long-v1 answers correct; 38k decode (plain and speculative),
     TTFT and `--context max` measured against f32 and llama.

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
