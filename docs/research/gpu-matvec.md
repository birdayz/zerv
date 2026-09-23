# GPU single-vector projection research — block 08

Date: 2026-09-22. Scope: F32-input, row-major F32/Q4_0/Q4_1/Q5_K/Q6_K
matrix-vector products, not batching, attention, recurrent state or model execution.
[Source ledger](2026-09-22/gpu-matvec-sources.json), retained under
`third_party/ggml/456172ec733a135778adcd32d00e576a58232e45/`.
[Independent actual-artifact shape inventory](2026-09-22/gpu-matvec-shapes.json).
Existing exact CPU quant specifications define the packed weight equations.

## Source-backed conclusions

- `ggml_mul_mat`: A dimensions [K,M,1,1], B [K,1,1,1], output [M,1,1,1].
  Contiguous row-major weights; each K is a whole number of quant blocks.
- Base model needs F32 [5120,48], Q4_0 seven dense shapes, Q4_1 [17408,5120],
  Q5_K [6144,5120], Q6_K [5120,248320]. Token embedding is lookup, conv weights
  belong to convolution. The sole Q8_0 tensor is `blk.64.nextn.eh_proj.weight`,
  [10240,5120], MTP-only: excluded from this base-model block explicitly.
- `ggml_vk_should_use_mmvq` and `ggml_vk_mul_mat_vec_q_f16` (~9719/9800): AMD
  normally quantizes F32 activations to Q8_1 for Q4_0/Q4_1/Q5_K when K>=2048.
  Q6_K stays on floating-input matvec on AMD. `GGML_VK_DISABLE_MMVQ=1` disables
  this extra activation quantization. Contiguous F32 input then stays F32 despite
  the misleading `_f16` function name. The shader generator's matvec variants use
  FLOAT_TYPE=float, B_TYPE=float; no global F16 disable is necessary.
- Matvec selection does not consult the deprecated `ggml_mul_mat_set_prec` flag.
  Do not confuse matrix-matrix/flash-attention precision controls with this path.
  Matvec shaders use FP32 reductions/FMA/reassociation and sometimes factor global
  scales out of sums. They need not match sequential decoded-F32 CPU dot bits.
- `ggml_backend_graph_compute` calls async compute then synchronizes. Vulkan has
  null graph-plan callbacks. It does graph/context/command management each call;
  there is no exported arbitrary pre-recorded matvec dispatch. Thus a one-node
  prebuilt graph is the closest public component boundary, not identical host work
  to a native reusable command buffer. Report this difference, and a default-MMVQ
  quality/timing control. No component ratio is a llama-server serving result.
- The installed ggml reports a dirty build; binary identity, not a clean-source
  reconstruction claim, pins the oracle. GPU library is in `/usr/lib/ggml/`, not
  the root ldconfig library directory. C adapters are external tests only.

## Original native baseline (superseded by block08b DFS)

One 64-invocation workgroup per output row; strided FP32 partial sums followed by
six shared-memory binary-tree levels. Flatten a bounded 2D dispatch for vocabulary
rows beyond maxComputeWorkGroupCount[0]. Compile one shader per type before use.
Read packed bytes through uint32 storage (including 2-byte row alignment); avoid
optional 8/16-bit storage/arithmetic and subgroup features. Convert binary16 bits
exactly into FP32, including half subnormals. Strict decoded weight equations;
FP32 product/add without contraction initially. No repacking or expanded weights,
per-call allocation/upload/readback, or activation quantization. Resident buffers
may be shared with nonoverlapping byte views; descriptor ranges remain <= queried
maxStorageBufferRange. Host API validates arithmetic/extents before recording.

Alternatives: subgroup reductions/packed loads/cooperative quantized activation
paths might be faster, but change capability/precision requirements. Start with the
small diagnosable implementation and retain measured losses. Full-vocabulary Q6
weights alone are 1,042,944,000 bytes, within the actual descriptor range. Whole
model residency will need multiple weight banks, not one >4 GiB descriptor.

The later [matvec DFS](matvec-optimization.md) replaces scalar traversal/manual
half conversion with packed-word reuse/core half unpack, widens the F32 workgroup
when supported, and balances a genuinely 2D capped grid. The original scalar
source and independent numerical contract are preserved. The original65537-row
fixture did not force2D on this device because its queried X limit is larger.

## Independent executable mechanism / readiness

[Pre-implementation contract](../specs/gpu-matvec.md). A standalone C adapter links
installed ggml Vulkan only as an external oracle. Load a bounded binary case,
construct one-node graph, upload once, compare GPU output against row-at-a-time
external CPU dequantization and long-double dot/sum-absolute-products. Independent
Python scalar decoding cross-checks synthetic case construction. Pin binaries and
headers. Fixture generation precedes native shader code; repeat into a fresh path.

Exact cases isolate each coefficient/column with dyadic inputs, and exhaust every
finite half encoding in every global field with a single nonzero activation.
Canonicalize signed output zeros only. Seeded/cancellation/multi-block/actual-width
cases use a predeclared mixed absolute / sum-of-absolute-products bound (not unstable
relative-to-near-zero-only comparison). Report reference precision errors first.
Full real-model dense shapes are later correctness-gated component workloads;
small native fixtures do not imply whole-shape coverage. F32 activation subnormals,
nonfinite arithmetic and overflow are outside this initial GPU numerical domain;
finite normal/zero inputs and finite normal/zero FP32 intermediates are required.
Half subnormals decode to normal FP32 and are in scope.

Status: independent extraction and native implementation passed; all eleven full
dense shape/type pairs were checked and benchmarked twice. [Validation](2026-09-22-gpu-matvec-validation.md)
and [measured losses](../bench/2026-09-22-gpu-matvec.md). No native model or serving success asserted.

## Model-planner integration limitation

The current Plan is a fixed-buffer/view single-projection primitive. Existing GPU
infrastructure caps live kernels at64 and retained kernels per command at32. A
whole-model planner cannot blindly create a unique pipeline/command object for
every projection: bank weights and share pipelines/validated dispatch geometry
during graph recording. This integration is not implemented by block08; do not
raise caps or add per-token command rebuilding to disguise it.
