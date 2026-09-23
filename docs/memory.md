# VRAM accounting and configuration research

**Estimates, not allocation measurements.** Never use `27B × 4 bits` as a fit test.
GiB = 2^30 bytes; MiB = 2^20 bytes. The card reports 23.984375 GiB physical VRAM.

## Budget equation

```text
weights_device + attention_KV + recurrent_state + recurrent_checkpoints
+ graph_workspace + activations + logits/sampling + staging/repacking
+ allocator_padding/fragmentation + runtime_overhead
<= physical_VRAM - external_use_reserve - safety_margin
```

Count the peak during load/repacking as well as steady-state, prefill, decode,
and cancellation. Host mmap/file cache is not device residency. GTT is not free
VRAM; spilling hot weights/state over PCIe can destroy throughput. A partially
resident model is an explicit low-memory tradeoff, never a silent fallback.

A driver budget is dynamic. Record physical capacity, current usage, budget, our
allocated bytes, and our required reserve separately. Do not subtract desktop use
twice when applying a budget already net of that usage. A startup fit estimate is
not protection against future allocations by other applications.

## Full-attention KV (16 layers, not 64)

For one stored token, with FP16 K and V and no padding:

```text
16 layers × 4 KV heads × 256 elements × (2 bytes K + 2 bytes V)
= 65,536 bytes = 64 KiB per token
```

| Total retained tokens across independent sequences | FP16 KV |
| ---: | ---: |
| 8,192 | 512 MiB |
| 32,768 | 2 GiB |
| 65,536 | 4 GiB |
| 131,072 | 8 GiB |
| 262,144 | 16 GiB |

This excludes recurrent state, MTP, paging overhead, alignment, and scratch.
A 32K-per-sequence limit at concurrency four can require 128K aggregate token
capacity; distinguish per-sequence limits from the total pool. Prefix sharing only
reduces storage when implemented correctly for both attention and recurrence.

For GGML-style `q8_0` blocks, 32 values occupy 34 bytes; `q4_0` uses 18 bytes.
Under compatible layouts, K+V q8_0 would take 34 KiB/token, and q4_0 18 KiB/token,
not exactly half/a quarter of FP16. These are format arithmetic, not a promise
that any chosen kernel supports quantized KV or that quality is unchanged.
Research/validate KV quantization independently of weight quantization.

## Gated DeltaNet state (48 layers)

Logical recurrent matrix per sequence at FP32:

```text
48 layers × 48 value heads × 128 key dimension × 128 value dimension × 4 bytes
= 150,994,944 bytes = 144 MiB
```

Minimal convolution history retains `kernel_width - 1 = 3` prior steps:

```text
channels = 2 × 16 × 128 + 48 × 128 = 10,240
48 layers × 3 steps × 10,240 channels × 4 bytes = 5.625 MiB
```

Combined logical lower bound: **149.625 MiB per independent state**. An
implementation retaining four convolution steps instead of three needs more.
Temporary chunk states, speculative copies, allocation rows, and cached prefix
checkpoints multiply this. Inspected llama.cpp allocates recurrent rows according
to `mem_size × (1 + n_rs_seq)`; do not assume its physical allocation equals this
logical lower bound or interpret context checkpoints as free.

Initial recurrent precision should preserve FP32 behavior. Lower precision requires
long-horizon stability tests, not just short-prompt top-1 agreement. Recurrent
state cannot be reconstructed from arbitrary paged attention KV alone.

## Illustrative fit, one sequence

Assume file size approximates resident weights; one FP32 recurrent state; FP16 KV;
**2 GiB workspace/runtime allowance**; **2 GiB total external-use/safety reserve**.
Both allowances are hypothetical, not measured. No MTP, projector, or checkpoints.

| Candidate | 8K context total incl. allowances | 32K context total incl. allowances |
| --- | ---: | ---: |
| Q4_0 | 19.60 GiB | 21.10 GiB |
| UD-Q4_K_M | 19.98 GiB | 21.48 GiB |
| UD-Q5_K_M | 23.06 GiB | 24.56 GiB |
| UD-Q6_K | 25.12 GiB | 26.62 GiB |

Q4 is a sensible first artifact class. Q5 may fit short contexts but leaves
limited margin under these assumptions. Q6 may fit with a different workspace /
reserve strategy; these numbers are not a proof it cannot run. Native 256K FP16
KV plus Q4 weights already exceeds this card before workspace. Context defaults
must not blindly inherit the model's advertised maximum.

## Knobs to study, not all to implement at once

- Weight format and per-tensor precision/layout; device/host placement policy.
- Hard device-byte budget and reserve; host/pinned/staging-memory caps.
- Per-request context and output caps, total token capacity, maximum live sequences.
- KV dtype, page/block size, prefix-cache budget, recurrent checkpoint budget.
- Prefill token budget, physical microbatch size, workspace cap, decode batch cap.
- Full attention kernel variant; recurrent chunking; device-side sampling.
- Optional speculation with its draft weights/state, acceptance, and rollback cost.

Settings must map to real mechanisms, be validated against capabilities, and
appear in the effective configuration. If fitting fails, report the breakdown and
explicit alternatives; don't silently lower precision, offload, or shrink context.

Sources: [model config](research/2026-09-22/model-config.json), artifact metadata,
and pinned llama.cpp `llama-hparams.cpp`, `llama-memory-recurrent.cpp`,
`llama-model.cpp`, and `ggml-common.h` in the source ledger. Recalculate from
loaded metadata in the engine; these model-specific calculations are research only.
