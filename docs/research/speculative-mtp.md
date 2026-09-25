# Speculative decoding with the Qwen3.8 MTP layer (block 17b research, 2026-09-24)

Question: how can zerv use the GGUF's multi-token-prediction (MTP / "nextn") layer to beat
llama-server's decode tok/s **without changing any output bit**?

## Sources (pinned)

llama.cpp commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, the same build as the installed
`llama-server` (build 10964). Files fetched to
`third_party/llama.cpp/b29c606e…/` and checked against the git blob hashes in
`research-tree.json` (all match):

- `src/models/qwen35.cpp`: MTP tensor loading and `graph_mtp`;
- `common/speculative.cpp`: `common_speculative_impl_draft_mtp`, the draft/process/accept
  driver;
- `docs/speculative.md`, `src/llama-context.cpp`, `src/llama-model.cpp`,
  `src/llama-arch.cpp`, `src/llama-memory-recurrent.cpp` (read for context).

## The MTP layer (source-backed)

GGUF tensors `blk.64.*` (`qwen35.nextn_predict_layers = 1`, 0.265 GB):

| Tensor | Type | Shape (K × M) |
| --- | --- | --- |
| `nextn.eh_proj` | **Q8_0** | 10240 × 5120 |
| `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` | F32 | 5120 |
| `attn_norm`, `post_attention_norm` | F32 | 5120 |
| `attn_q` (gated, as the main attention layers) | Q4_0 | 5120 × 12288 |
| `attn_k`, `attn_v` | Q4_0 | 5120 × 1024 |
| `attn_q_norm`, `attn_k_norm` | F32 | 256 |
| `attn_output` | Q4_0 | 6144 × 5120 |
| `ffn_gate`, `ffn_up` / `ffn_down` | Q4_0 | 5120 × 17408 / 17408 × 5120 |

There are no own embedding or head tensors: `embed_tokens` and `shared_head_head` are
optional, and the model's `token_embd` and `output.weight` (Q6_K, 1.04 GB) are used.

`graph_mtp` for one row with token `x` at position `q` and input hidden `h`:

1. `e = rmsnorm(embed(x)) * enorm`, `g = rmsnorm(h) * hnorm`;
2. `u = eh_proj([e ; g])`. The concatenation puts the **embedding first**
   (`ggml_concat(e_norm, h_norm, dim 0)`);
3. one full-attention block, as the main model's attention layers:
   - `attn_norm`;
   - q (gated, 2 × 6144) / k / v;
   - q/k RMS norm and multi-section RoPE at position `q`;
   - attention over the **MTP layer's own KV cache**, scale 1/16;
   - `sigmoid(gate)` × attention, `attn_output`, residual `+ u`;
4. `post_attention_norm`, SwiGLU FFN, residual;
5. `h' = rmsnorm(·) * shared_head_norm` (the MTP's `h_nextn`);
6. `logits = output.weight · h'`.

**Main model output fed to the MTP:** `t_h_nextn` = `output_norm(l_out)`, i.e. the
normalized final hidden state, the head's input (`result_norm` in our captures), at
every position.

## How llama.cpp drives it (`draft-mtp`, single head)

- **MTP KV position `q` holds the pair `(h_{q−1}, x_q)`**: the main model's `h_nextn` at
  the previous position and the token at `q` ("pair (h_p, x_{p+1}) at MTP pos p+1").
- **Prompt (`process`):**
  - after each target batch, the MTP runs over the same tokens and positions, with the
    target's `h` rows shifted right by one;
  - row 0 takes `pending_h`, which is zero for the first prompt token;
  - no logits (catch-up only fills the MTP KV).
- **Draft (`draft`):**
  - step 0: token = last sampled `id_last` at `pos0`, `h = pending_h` (the target `h` of
    the last verified row);
  - step i > 0: the drafted token at `pos0 + i`, with `h` = the MTP's own `h'` from the
    previous step;
  - candidates are the top-1 of a top-k(10) draft sampler; `p_min` (default 0) can stop
    early; `--spec-draft-n-max` drafts (default 3).
- **Verify:**
  - the target decodes `[id_last, d1..dn]` in one batch; `process` then catches the MTP up
    on that batch with the true target `h` rows;
  - `accept(n)` sets `pending_h` to the target `h` row of the last accepted row;
  - rejected positions are removed from both caches.

## Measured competitor (2026-09-24, [data](../bench/data/2026-09-24-decode-baseline/serving-decode-v1/))

`bench/workloads/decode-v1.json`: 4 cases, greedy, 512 tokens, 2 repeats. llama-server
default flags plus `--spec-type draft-mtp --spec-draft-n-max N`. Decode tok/s:

| Case | zerv (no spec) | llama | MTP 1 | MTP 2 | MTP 3 | MTP 4 |
| --- | --- | --- | --- | --- | --- | --- |
| code | 49.3 | 41.4 | 68.7 | 86.8 | 91.1 | 105.3 |
| json | 49.2 | 41.4 | 69.2 | 88.5 | 94.7 | 111.6 |
| think | 49.2 | 41.3 | 66.6 | 81.6 | 82.3 | 90.7 |
| prose | 49.3 | 41.3 | 59.7 | 62.7 | 55.6 | 58.5 |

- Draft acceptance per drafted token (llama's log):
  - N = 1: code 0.97, json 0.98, think 0.91, prose 0.70;
  - N = 4: 0.82 / 0.88 / 0.66 / 0.33.
- **llama's MTP output is not its non-speculative output:** json, think and prose differ
  from `llama-fa-ub512` under greedy decoding. Its batched verification uses different
  arithmetic than single-token decode.
- The ngram-mod run stalled mid-request (see the incident note in the data directory).

## Losslessness design (zerv)

1. **Verification is bitwise single-token decode.**
   - The target processes `N = n + 1` rows `[t, d1..dn]` with kernels whose arithmetic
     per row is exactly the decode step's.
   - Every row's logits are then bitwise equal to those of N sequential decode steps (as
     long as the drafts are the tokens decode would have produced, which is exactly the
     accepted prefix).
2. **Acceptance by sample matching**, not rejection sampling.
   - For row i, draw the sampler's random numbers exactly as non-speculative decode would
     for that position and sample `y_i` from the target logits.
   - Accept while `y_i == d_{i+1}`; the first mismatch emits `y_i` and ends the step. If all
     n drafts match, row n's sample is a bonus token.
   - The emitted sequence is **identical to non-speculative decode with the same seed**,
     for greedy and sampled decoding alike.
   - The acceptance probability of a deterministic draft is `p(d)`, the same as speculative
     rejection sampling with a point-mass draft, so nothing is lost against it.
3. **State commit.**
   - DeltaNet and conv states must end after the `m` consumed rows.
   - The verify pass leaves the stored state untouched; a commit pass recomputes it from
     the stored per-row inputs for `m` rows, with the same per-token arithmetic.
   - Attention K/V of rejected positions stay in the cache, but nothing reads a position
     before it is written again.
4. The **draft quality affects only speed**, never output. MTP arithmetic can therefore
   be chosen for speed, but its semantics must match the trained layer, or acceptance
   drops.

## Cost model (estimates, to be replaced by measurements)

- **Decode step (measured):** 20.1 ms, of which about 18.5 ms is weight-streaming matvecs
  (15.1 GB) and about 1.6 ms is everything else.
- **Verify of N rows:** weights are read once, but each row needs its own X.
  - The shipped matvec reads 32 bytes of X per 4-byte weight word per row from L0.
  - At about 900 GB/s of Q4_0 weights, N = 4 needs about 25 TB/s of X loads, which is at
    or above the L0 limit (96 CUs × 128 B/clk × about 2.5 GHz ≈ 30 TB/s).
  - The multi-row kernel therefore gives each lane G weight rows, reusing each X load G
    times; each lane's arithmetic is unchanged.
- **Draft step:** the MTP layer (0.27 GB) plus the shared head (1.04 GB) is about
  1.5 ms per drafted token. The head dominates.
  - A reduced draft vocabulary (EAGLE-3 style) is a later speed knob. It changes
    acceptance only.
- **Catch-up:** the rows of the accepted prefix are fused with the next step's first draft
  row into one MTP pass (logits for the last row only). This saves an MTP weight pass per
  step compared with llama's separate catch-up.
- **Commit:** about 0.6 ms (one DeltaNet state pass).
- **Example, n = 3 on code** (mean accepted length about 3.6 including the bonus
  token): step about 20–24 + 4.5 + 0.7 ≈ 25–29 ms, so about 125–145 tok/s, against llama's
  91 (MTP 3) and 105 (MTP 4). Prose (about 2.1): about 72–84 against llama's best 63.

## Open questions resolved here

- **Q2 (design log):** MTP counts for decode comparisons, because llama-server's MTP is
  benchmarked alongside with the same prompts and flags recorded. zerv's variant is
  additionally lossless; llama's is not.
- **Draft arithmetic precision:** FP32 like decode (the reference is the trained layer).
  Its correctness gate is an FP64 component reference of `graph_mtp` on real weights
  and inputs. llama.cpp's draft tokens are a secondary check (acceptance of the same
  magnitude).
