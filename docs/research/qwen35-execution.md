# Qwen3.8-27B forward pass — consolidated execution semantics (block 09)

2026-09-22. Source-backed from pinned files already in the ledgers:
HF `qwen35-modeling.py`/`qwen35-configuration.py` and llama.cpp
`llama-qwen35.cpp`, `delta-net-base.cpp`, `models.h`, `llama-graph.cpp`,
converter `conversion/qwen.py` at b29c606e (same commit as the installed libllama
build 10964). GGUF metadata/tensor inventory is observed from the SHA-verified
artifact ([inventory](../bench/data/2026-09-22-gguf-qwen38/container.json)).
Earlier findings on converted norms/A/dt and gate semantics:
[gpu-primitives.md](gpu-primitives.md). **Status: source research; the
executable agreement gate is the block09 oracle below.**

## Hyperparameters (GGUF metadata, observed)

n_embd 5120, n_ff 17408, vocab 248320 (token_embd Q4_0, output Q6_K, untied),
64 trunk layers (`block_count` 65 includes one MTP block, skipped), full attention
every 4th layer (il % 4 == 3: 16 layers), 48 GatedDeltaNet layers. Attention:
24 query heads, 4 KV heads, head dim 256, rotary dims 64, freq_base 1e7, sections
[11,11,10,0], IMROPE. DeltaNet: 16 K heads, 48 V heads, head dims 128, conv kernel 4,
inner 6144. RMS eps 1e-6 (stored as FP32 9.99999997e-7). BOS 248044, EOS 248046.

## Layer (il), ggml column-vector notation, x is one token's FP32 hidden state

```
h   = rmsnorm(x) * attn_norm            # weights already +1 by the converter
a   = attn_full(h) if il%4==3 else attn_linear(h)
r   = x + a                              # "attn_residual"
f   = ffn_down( silu(ffn_gate h2) * ffn_up h2 ),  h2 = rmsnorm(r) * post_attention_norm
x'  = r + f                              # "l_out"
```
Final: `logits = output · (rmsnorm(x_64) * output_norm)`. rmsnorm(v) =
`v / sqrt(mean(v²) + eps)`. No embedding scaling; row `token_embd[token]` is the input.

### Full attention (16 layers)

- `attn_q` [5120→12288] rows are per head `[q(256), gate(256)]`, 24 heads.
- `attn_k`, `attn_v` [5120→1024]: 4 heads × 256.
- q, k: per-head RMS norm (width 256) × `attn_q_norm`/`attn_k_norm` (already +1).
- RoPE on the first 64 dims of each head, NEOX pairing (i, i+32), angle
  `pos · 1e7^(-2i/64)`, i=0..31. Text tokens use equal t/h/w positions, so the
  interleaved-MRoPE section assignment does not change any angle (HF: apply
  interleaved mrope with identical position rows). Dims 64..255 unrotated.
  (IMROPE pairing and position replication must be confirmed by the capture.)
- Causal softmax attention, scale 1/16, GQA: query head h uses KV head h / 6.
- `attn_gated = attn · sigmoid(gate)` (flattened in head order), then `attn_output`
  [6144→5120]. The config field `output_gate_type="swish"` is not read by either
  implementation (see gpu-primitives.md).

### GatedDeltaNet (48 layers)

- `attn_qkv` [5120→10240] = conv channels `[q(2048) | k(2048) | v(6144)]`,
  `attn_gate` (z) [5120→6144], `ssm_beta`, `ssm_alpha` F32 [5120→48].
- Causal depthwise conv, kernel 4, per channel c: `u_t[c] = Σ_{j=0..3} w[c][j] ·
  s[c][t-3+j]` (tap 3 = current token; zero history at sequence start), then SiLU.
  Conv state keeps the last 3 pre-conv inputs per channel.
- q, k split into 16 heads × 128; each L2-normalized `v/sqrt(Σv²+1e-6)`
  (llama computes it as rms_norm(eps/128)/sqrt(128)); q scaled by 1/sqrt(128).
- `beta = sigmoid(ssm_beta·h)`, `g = ssm_a · softplus(ssm_alpha·h + ssm_dt.bias)`
  (`ssm_a` is already `-exp(A_log)`).
- Per V head hv (48), K head `hv % 16` (converter tiles V heads; HF's
  `repeat_interleave` applies only to the unconverted order). State S[j][i]
  (j value dim, i key dim), FP32, zero at sequence start:
  `S ← S·exp(g);  d_j = (v_j − Σ_i S[j][i] k_i)·beta;  S[j][i] += k_i d_j;
  o_j = Σ_i S[j][i] q_i` (HF identical with transposed storage).
- `final = rmsnorm_128(o_head) * ssm_norm * silu(z_head)` per V head
  (ssm_norm used as-is, not +1), then `ssm_out` Q5_K [6144→5120].

## State per sequence

KV: 16 layers × 2 × 4 × 256 values per token. DeltaNet: 48 × 48 × 128 × 128 FP32
recurrent state + 48 × 3 × 10240 conv history. Prefix reuse/rollback must restore
both. Our first path keeps KV in FP32 (no quantized cache).

## Oracle design (executable gate for this research)

1. **libllama capture** (external dev tool, pinned binaries): Vulkan0, all layers
   offloaded, one token per `llama_decode` so every projection is a matvec,
   `GGML_VK_DISABLE_MMVQ=1` (FP32 activations), F32 K/V cache, flash attention
   off. `cb_eval` records named tensors (`attn_norm-N`, `Qcur_full-N`, `l_out-N`,
   `result_output`…) for every token; strided views are gathered to contiguous FP32.
2. **Independent NumPy FP64 forward** written from the HF modeling semantics and
   the converter mapping above, with its own vectorized dequantizers (cross-checked
   against the existing scalar quant goldens). It processes a whole teacher-forced
   sequence per layer.
3. Semantic agreement gate: NumPy vs libllama on every captured tensor/token, with
   errors consistent with FP32 rounding only. Any structural mismatch (wrong RoPE
   pairing, head mapping, permutation, sign) blocks native code.
4. Native tolerances are declared from the measured libllama-vs-FP64 error.

HF PyTorch itself is not executed: torch/transformers are not installed and the
BF16 checkpoint is ~55 GB. Quality relative to the original BF16 model is a
separate, currently unmeasured question (it needs that download and approval).
