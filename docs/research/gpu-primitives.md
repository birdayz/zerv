# GPU normalization, gates and reductions — paused block 09

Paused by user request during the matvec DFS. These findings do not authorize
resuming implementation; follow [the controlled queue](../../TODO.md).

2026-09-22. **Research in progress, no native block09 implementation/spec/goldens
yet.** Block08 is closed with verified full-shape matvecs and retained performance
losses. [Source ledger](2026-09-22/gpu-primitives-sources.json) retains pinned files
under third_party; existing model/HF source identities also appear in the original
[ledger](2026-09-22/sources.json).

## Source-backed mathematics and actual artifact observations

### Norm weights are already converted

Pinned llama.cpp `conversion/qwen.py`, Qwen3NextModel.modify_tensors, inherited by
Qwen3_5TextModel through _LinearAttentionVReorderBase:

- `.A_log` becomes **`-exp(A_log)`**, not the original log. GGUF `ssm_a` is already
  negative. Do not negate/exponentiate it a second time in the execution graph.
- Ordinary `*norm.weight` tensors get **+1 during conversion**, except
  `linear_attn.norm.weight` (gated RMS scale) which is used as-is.
- `.dt_bias` is renamed to `.dt_proj.bias`, then tensor-name mapping produces
  `blk.N.ssm_dt.bias`. Graph adds it to the alpha projection before softplus.
- llama graph's `build_norm` applies `ggml_rms_norm(x, eps)` then multiplies by the
  GGUF scale directly. No further +1 in the native GGUF execution path.

HF Qwen3_5RMSNorm performs FP32 `x * rsqrt(mean(x*x)+eps) * (1+weight.float())`,
then casts to input dtype. HF Qwen3_5RMSNormGated instead uses an ordinary weight
initialized to ones, and normalizes before multiplication by SiLU(gate). Its BF16
cast points differ from all-FP32 diagnostic execution; don't claim identical BF16
rounding from an FP32 ggml operator comparison.

Observed actual F32 tensor summaries/hashes:
[gpu-primitives-tensors.json](2026-09-22/gpu-primitives-tensors.json). For example,
blk.0.attn_norm ranges .868164..1.198242; blk.0.ssm_norm .785156.. .929688;
blk.0.ssm_a is entirely negative (-.337585..-.003839). This supports the conversion
trace, but is not an original-BF16-weight differential check. The post-attention
norm's actual name is `blk.N.post_attention_norm.weight`. An initial inspection
script guessed `attn_post_norm.weight` from the C++ field and stopped with
StopIteration; corrected by enumerating actual returned tensor names. No file was
changed and no numerical fixture was produced by that failed probe.

### Gated DeltaNet L2 normalization is not ggml_l2_norm

HF `l2norm(x)` is `x * rsqrt(sum(x*x) + eps)` with eps=1e-6. llama's
`src/models/models.h:build_gdn_l2_norm` implements the equivalent mathematical
operation using **`ggml_rms_norm(x, eps/n) * (1/sqrt(n))`**. FP32 operation ordering
can differ from direct evaluation, so compare with declared numerical bounds.
In contrast, pinned Vulkan `l2_norm.comp` implements
`x / max(sqrt(sum(x*x)), eps)`: this is a **different function**, especially near
zero. Do not choose that exported operation as the GDN oracle merely by name.
Q/K widths are128, 16 heads each. Gated output RMS width128, 48 V heads. Ordinary
hidden norm width5120; full-attention Q/K norm width256, 24/4 heads. eps=1e-6.

### Gate semantics: strong source evidence, executable gate still pending

Exact official checkpoint config retains `output_gate_type="swish"`, but neither
pinned HF `qwen35-modeling.py` nor `qwen35-configuration.py` contains a reference to
that field. Full-attention forward **unconditionally multiplies the pre-output
attention by sigmoid(gate)**. Qwen3_5RMSNormGated sets `self.activation="silu"`
and multiplies normalized output by SiLU(gate); this is independent of that field.
The pinned eager attention core also does not inspect it. llama qwen35 graph agrees:
`gate_sigmoid`/`attn_gated` for full attention, RMS then SiLU gate for linear output.
FFN is `down(silu(gate_projection) * up_projection)`.

This pins intended operations for those inspected implementations; do not turn the
checkpoint field into a global sigmoid→SiLU rewrite. **Still required before native
block09 math:** executable independent intermediate probes, with explicit handling
of the original config, and a documented scope for fallback/eager execution versus
optional HF hub kernels. torch and transformers are not installed in the system
Python (NumPy is); no packages were installed. A source-derived NumPy adapter must
not be mislabeled as executing the actual PyTorch model/checkpoint. Full-model
teacher-forced intermediates/logits remain a later mandatory gate.

### Other precision traps

Pinned Vulkan unary shaders use sigmoid `1/(1+exp(-x))`, SiLU
`x/(1+exp(-x))`, softplus `x>20 ? x : log(1+exp(x))`. The latter loses relative
accuracy for negative x where adding exp(x) to1 rounds toward1; a stable scalar
log1p reference can therefore disagree near zero. Measure the reference error
before choosing native tolerances or a stable implementation. Do not silently
reinterpret a reference rounding difference as a layout bug or hide a true error.
Weighted RMS can fuse with multiply in the external Vulkan graph. Keep default
reference fusion enabled and record composition/host-boundary differences.

Linear gate preprocessing should consume GGUF `ssm_a` directly:
`g[h] = ssm_a[h] * softplus(alpha[h] + dt_bias[h])`,
`beta[h] = sigmoid(beta_projection[h])`. The subsequent exp(g) belongs to the
recurrent update, not a second conversion of ssm_a.

## Model-layout finding to preserve for block11

The pinned converter explicitly permutes V heads from original HF grouped order
`[K0_v0,K0_v1,K0_v2,K1_v0,...]` into ggml tiled order
`[K0_v0,K1_v0,...,K15_v0,K0_v1,...]` for K16/V48. It permutes V portions of QKV,
Z gate rows, alpha/beta rows, A/dt parameters, convolution V channels and output
projection columns consistently. The shared 128-wide gated RMS scale is not a
per-head array. **Do not apply HF repeat_interleave indexing directly to these
already-converted GGUF tensors**; capture/verify the actual permutation when
implementing the recurrent block. No recurrent code has been started.

## Next executable gate (proposal, not implemented)

Specify a small cohesive resident FP32 primitive package: weighted RMS, GDN L2,
add/multiply, SiLU/sigmoid, sigmoid/SiLU multiplication, gated RMS and linear gate
preprocessing; allow checked strided head rows/broadcast weight vector where needed.
Independent scalar FP64 expectations plus actual external ggml Vulkan intermediate
captures from equivalent composed graphs; use RMS(eps/n)+scale for GDN L2, not
`ggml_l2_norm`. Include all-zero/tiny/cancellation/extreme/seeded inputs, actual
norm/ssm weights, per-head strides and sentinels. Declare tolerances only after
measuring the independent reference and before native implementation. Pin all
sources/generator/binaries, repeat extraction and preserve any precision discrepancy.
Then native tests and matched repeated timings. No attention/recurrent/session or
HTTP implementation starts in parallel with this block.
