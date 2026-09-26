# Qwen3.8-27B image input (vision) — research, first pass (block 19a)

2026-09-26. Status: **research in progress**; no zerv code yet. Scope: still images through
`POST /v1/chat/completions` (`image_url` content parts). Video is out of scope and must be
rejected explicitly. Labels: **observed** (run or inspected here), **source** (read at the
pinned revision), **estimate**, **open**.

Sources, all pinned and hashed in the [ledger](2026-09-26-vision/sources.json), local copies
under `third_party/research/2026-09-26-vision/`:

- Transformers `87d34bfa` (the revision of the block-09 modeling file; `modeling_qwen3_5.py`
  is byte-identical to `third_party/research/2026-09-22/qwen35-modeling.py`): the vision
  model, `vision_utils.py`, the Qwen2-VL image processor, the Qwen3-VL processor.
- llama.cpp `b29c606e` (the installed build 10964 and the block-09 oracle): `tools/mtmd`
  (clip, the qwen3vl graph, preprocessing, M-RoPE positions), the converter, the server's
  media handling, the Vulkan RoPE shader, `common/speculative.cpp`.
- `Qwen/Qwen3.8-27B` `1d4bf0f2`: `chat_template.jinja`, `preprocessor_config.json`,
  `model.safetensors.index.json`.

## 1. Artifacts (observed)

- **Projector:** `unsloth/Qwen3.8-27B-GGUF` `4ca72078` `mmproj-BF16.gguf`, 931,146,432 bytes,
  sha256 `83ee4f4f…5c2d53`, downloaded 2026-09-26 on purpose for this block (`.part`, then
  `sha256sum --check` against the pinned manifest) to `models/qwen3.8-27b/`. Same repo
  revision as the Q4_0 text GGUF. The repo also has `mmproj-F16.gguf` (lossy BF16→F16;
  not downloaded). Ollama's projector blob for `qwen3.8:27b` is 931,146,016 bytes, i.e. the
  BF16 variant ([deployment note](2026-09-23-previous-deployment.md)).
- **Inventory** ([JSON](2026-09-26-vision/mmproj-inventory.json), `tools/vision_artifacts.py`):
  GGUF v3, architecture `clip`, `clip.projector_type = qwen3vl_merger`, 334 tensors: 110 BF16
  matrices, 224 F32 (norms, biases, position table, both patch-kernel slices). Metadata:
  `image_size 768`, `patch_size 16`, `embedding_length 1152`, `feed_forward_length 4304`,
  `block_count 27`, `head_count 16`, `projection_dim 5120`, `image_mean/std 0.5`,
  `use_gelu true`, `spatial_merge_size 2`, `layer_norm_epsilon 1e-6` (as FP32
  9.99999997e-7), `is_deepstack_layers` all false.
- **Provenance check (observed):** every GGUF tensor equals exactly one tensor of the
  `model.visual.*` tower in the RedHat INT4 checkpoint already on disk (vLLM competitor; the
  vision tower is excluded from its quantization, BF16): 110 raw-byte BF16 matches, 224
  exact BF16→F32 widenings, the Conv3d kernel `[1152, 3, 2, 16, 16]` as its two temporal
  slices `[:, :, 0]` / `[:, :, 1]`; no tensor unmatched on either side. Two independent
  derivations of Qwen's upstream weights agree bit for bit. Upstream keeps all 333 vision
  tensors in `model-00001-of-00018.safetensors` (not downloaded).
- Size: 460,730,096 parameters; 921 MB as BF16, 1.84 GB as F32 (estimate for device
  residency).

## 2. Model semantics (source: HF `Qwen3_5VisionModel`, cross-read with llama.cpp `qwen3vl.cpp`)

Config: depth 27, hidden 1152, 16 heads × 72, MLP 4304 with `gelu_pytorch_tanh`, patch 16,
temporal patch 2, spatial merge 2, `num_position_embeddings` 2304 (48 × 48), output 5120,
no deepstack layers. Image tokens: `<|vision_start|>` 248053, `<|vision_end|>` 248054,
`<|image_pad|>` 248056 (`<|video_pad|>` 248057 unused here).

For one image with a patch grid of `gh × gw` patches (`gh`, `gw` even), `N = gh·gw` patches,
`N/4` output tokens:

1. **Pixels → patches.** Input normalized to `(x/255 − 0.5)/0.5` per channel (section 3).
   Each patch is `3 × 2 × 16 × 16`: a still image is repeated along time (processor
   `patchify`), so the Conv3d equals a 2-D conv with `W[:,:,0] + W[:,:,1]`, plus bias.
   llama.cpp runs two 2-D convs and adds them (different FP32 rounding, same value).
   **Patch order is merge-block-major:** blocks of 2 × 2 patches in raster order over the
   `(gh/2) × (gw/2)` block grid, each block's four patches row-major `(dy, dx)`.
2. **Learned position embedding**, bilinear, `align_corners=True`: source coordinate
   `row·47/(gh−1)` (and `col·47/(gw−1)`; 0 when the size is 1), 4 taps clamped to the
   48 × 48 table, weights `(1 − |d|)`; added to the patch embedding. Same in llama.cpp
   (`GGML_SCALE_MODE_BILINEAR | ALIGN_CORNERS`).
3. **27 pre-norm blocks:** `x += proj(attn(LN1(x)))`, `x += fc2(gelu_tanh(fc1(LN2(x))))`.
   LayerNorm with weight and bias, eps 1e-6. `qkv` has bias; q, k, v are 16 heads × 72.
   **Full bidirectional attention** over all `N` patches of the image (`cu_seqlens` per
   image; images never attend to each other), scale `72^-0.5`, no mask.
4. **2-D rotary embedding** on q and k before attention, over the whole head: NEOX pairs
   `(i, i+36)`; pairs 0–17 rotate by `row · 10000^(−2i/36)`, pairs 18–35 by
   `col · 10000^(−2(i−18)/36)` (row, col: the patch's grid coordinates). HF computes
   cos/sin in FP32 and rotates in FP32. llama.cpp: `ggml_rope_multi` vision mode, sections
   18 × 4, positions `(y, x, y, x)`, identical angles. Base 10000: the checkpoint sets no
   vision rope parameters, so HF uses `RotaryEmbeddingConfigMixin.default_theta = 10_000.0`
   (`modeling_rope_utils.py:739`); llama.cpp passes the constant 10000.
5. **Merger:** LayerNorm(1152, eps 1e-6) per patch, then the four patches of a merge block
   concatenated in order (→ 4608), `fc1` 4608→4608, **exact (erf) GELU** (`nn.GELU()`),
   `fc2` 4608→5120. Output: one 5120-vector per image token, in block raster order.
6. **Into the language model:** each `<|image_pad|>` placeholder row takes the next image
   embedding instead of `token_embd[248056]`; everything else (DeltaNet layers, attention
   layers, MTP layer input) is unchanged.

**llama.cpp deviation (source):** its merger uses `FFN_GELU` (`ggml_gelu`, the tanh
approximation), not erf. The vision blocks are equal (both tanh). Magnitude: **open**, to be
measured with the FP64 reference computing both variants. llama.cpp is therefore not a
bitwise-semantic oracle for the merger; the reference must reproduce its variant to use it
for cross-checking.

## 3. Image preprocessing

**HF (source; the model's own processor, `Qwen2VLImageProcessorFast` =
`Qwen2VLImageProcessor(TorchvisionBackend)` at this revision):**

- Decode with PIL, `convert("RGB")` (RGBA: alpha dropped, not composited; palette and gray
  expanded).
- `smart_resize(h, w, factor 32, min_pixels 65,536, max_pixels 16,777,216)`: round each side
  to a multiple of 32; if the area exceeds max, scale by `sqrt(hw/max)` and floor to 32; if
  below min, scale up and ceil to 32; aspect ratio above 200 rejected. Token range
  **64 … 16,384** per image.
- Resize to exactly that size (stretch, no padding) with `tvF.resize(BICUBIC, antialias=True)`
  on the uint8 tensor from `pil_to_tensor` (result uint8; `image_processing_backends.py`).
- Rescale and normalize are fused (`_fuse_mean_std_and_rescale_factor`): FP32
  `(x − 127.5) / 127.5`, one rounding per value. llama.cpp computes `x/255` then
  `(v − 0.5)/0.5` in FP32 (two or three roundings; may differ in the last bit).
- A 1280 × 800 screenshot (bruh's browser/desktop tools, PNG) is already a multiple of 32:
  **no resize**, 80 × 50 patches, 1,000 image tokens.

**llama.cpp (source, `mtmd-image.cpp`, `clip.cpp`):**

- Decode with stb_image (3 channels; alpha dropped), WebP via ffmpeg if built with it.
- The same size rule in float (`calc_size_preserved_ratio`) but token limits **8 … 4,096**
  (`--image-min-tokens` / `--image-max-tokens` override); warns below 1,024 tokens.
- Resize with a Pillow-compatible separable bicubic (`resize_pillow`) and **`PAD_CEIL`**:
  keeps the aspect ratio, scales by `min(sw, sh)`, and pads with black. Example: the
  640 × 488 test image becomes 630 × 480 plus 5-pixel black bars (HF: stretched to 640 × 480).
  No resize when the size already matches, as for aligned screenshots.

**Consequences.** For images whose sides are multiples of 32 and within both token limits,
the two pipelines give identical pixels up to decoding; PNG decoding is lossless, so aligned
PNGs give **identical encoder inputs** in both (the first oracle cases). Otherwise the
encoder inputs differ (padding, token limits, resize implementation, JPEG decoder), and
llama-server comparisons on such images compare different preprocessing, not arithmetic.

**Our implementation (proposal):** follow the HF processor (stretch, the model's token
limits, bicubic antialias on uint8) with explicit `--image-min-tokens` /
`--image-max-tokens` knobs. The resize is to be bit-exact to PIL's `Image.resize(BICUBIC)`
(PIL is installed and is llama.cpp's stated model). **Open:** whether torchvision's uint8
antialiased bicubic equals PIL bit for bit (needs torchvision, section 7).

**Decoders (all ours, from the format specs; Zig std has inflate):** PNG first (bruh's
screenshots; all bit depths, palette, gray, alpha, 16-bit, Adam7; reference: PIL, exact).
JPEG second (baseline and progressive; PIL uses libjpeg-turbo, ISLOW IDCT and fancy
upsampling by default; stb_image uses its own IDCT and upsampling, so llama.cpp and HF can
see different pixels for the same JPEG — **open**, to be measured). WebP and GIF later or
rejected with 400. Limits before decoding: encoded bytes, declared dimensions, decoded
bytes, aspect ratio.

## 4. Positions: interleaved M-RoPE in the language model (source)

Text attention layers rotate 64 of 256 head dims in 32 NEOX pairs. With `mrope_section
[11, 11, 10]` interleaved: pair `j` uses the **h** position if `j % 3 == 1` (11 pairs, `j` <
33), **w** if `j % 3 == 2 and j < 30` (10 pairs), else **t** (11 pairs). HF
`apply_interleaved_mrope` and the llama.cpp Vulkan `rope_multi` imrope branch agree.
For text tokens t = h = w, so today's single-position RoPE is the special case.

Positions (`get_rope_index`; llama.cpp `MTMD_POS_TYPE_MROPE`, identical): text continues from
`p`; an image of `gh' × gw'` merged tokens (`gh' = gh/2`) starting at `p` gives token `(r, c)`
the triple `(t, h, w) = (p, p + r, p + c)`; the next text token gets
`p + max(gh', gw')`. So after an image, **position ≠ KV index**: the RoPE position is the
KV index minus `Σ (gh'·gw' − max(gh', gw'))` over earlier images.

zerv writes RoPE cos/sin rows on the host per row (`Model.writeRope`, FP64 angles): M-RoPE
changes only those rows, not the GPU kernels. The KV index (`pos`) stays the row index.

## 5. Chat template and request format (source)

- The Qwen3.8 template renders content arrays via `render_content`: an item with `image`,
  `image_url` or `type == 'image'` becomes `<|vision_start|><|image_pad|><|vision_end|>`
  (optionally `Picture n: ` with `add_vision_id`, default off); `text` items verbatim;
  system messages cannot contain images (exception). The processor then expands the single
  `<|image_pad|>` to `N/4` placeholders. An image-bearing user message counts as a user
  query for `last_query_index` (thinking retention), like any user text.
- bruh (`crates/openai-compat/src/wire.rs` @ `800d45ef`) sends tool images as a following
  **user** message: text `Image output of tool call {id}:` plus
  `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,…"}}` per image; browser
  and desktop screenshots are `image/png`, other tools PNG/JPEG/WebP.
- llama-server accepts `http(s)://` (downloads, 10 MB, 10 s), `file://` (with
  `--media-path`), `data:` base64 and raw base64. Proposal for zerv: `data:` base64 only;
  remote URLs rejected with 400 (no network egress from the engine; an explicit opt-in knob
  could come later). `detail` accepted and ignored (llama-server also ignores it; **open**).
- zerv's request body limit is 4 MiB (`api.Limits.max_body_bytes`); one base64 screenshot is
  0.3–3 MB, so multi-image conversations need a larger, explicit limit.

## 6. Interactions with existing zerv features (source: zerv tree; decisions open)

- **Prefix cache:** it matches token ids, and every image placeholder is id 248056, so two
  different images of equal size would match. The key must include each image's identity
  (hash of the decoded pixels after preprocessing, as llama.cpp's bitmap id) and the
  position offset. **Correctness requirement.**
- **MTP speculation:** the draft layer consumes `embed(next input)` and the hidden state.
  llama.cpp skips image batches in its MTP hook (`common/speculative.cpp`: "TODO: how to make
  it work with vision tokens?"), leaving a hole in the draft KV. Lossless either way (drafts
  are verified); acceptance differs. Options: feed image embeddings as the draft input
  (natural analog), or skip. To be measured.
- **Batched decode / slots:** images affect prefill only; each slot needs its own position
  offset for decode RoPE rows.
- **Prefill input:** the embed kernels read `token_embd` rows (Q4_0; host-mapped by default).
  Image rows need an input path from the vision output buffer; text rows stay Q4_0.
- **VRAM:** + 0.92 GB (BF16 weights) or 1.84 GB (F32), plus encoder scratch; must enter the
  `--context max` fit and `--vram-reserve-mib` accounting. A `--vision off` default keeps the
  text server's capacity unchanged.

## 7. Oracle design (proposal; the gate before engine code)

Mirrors block 09 (FP64 NumPy + libllama), with the llama.cpp variant made explicit:

1. **Semantic reference, FP64 NumPy** (`tests/reference/`, new): the HF vision model from
   the BF16 weights (exact in FP64), per-image captures (patch embed, position embed, block
   outputs, merger output); switch `merger_gelu = erf | tanh` to reproduce llama.cpp.
   Preprocessing reference: PIL decode + PIL resize + normalize + patchify.
2. **Official HF processor/model (proposed dev tool, needs a decision):** a venv under
   `third_party/` with CPU torch, torchvision and transformers `87d34bfa` to run the actual
   processor and vision model once, anchoring (1) and settling the torchvision-vs-PIL
   resize question. Not a runtime dependency; CPU wheels are several hundred MB.
3. **llama.cpp oracle** (installed build 10964, same commit): `libmtmd` final image
   embeddings (`mtmd_encode_chunk` + `mtmd_get_output_embd`) and whole-model logits with an
   image in the prompt (extending `tests/reference/model_oracle.c`). For tight agreement, an
   **F32 mmproj** made from the verified BF16 tensors by exact widening, run on the CPU
   backend (the BF16 CPU matmul rounds activations to BF16). **Observed feasibility:**
   `llama-mtmd-cli -m Q4_0 --mmproj mmproj-BF16.gguf --image test-1.jpeg -n 300 --temp 0
   -ngl 99 -c 8192` (Vulkan) described the 1969 NYT moon-landing front page correctly;
   "mtmd batch encoding done in 291 ms" (first encode, 300 image tokens; one run, not a
   benchmark). Logs: `third_party/vision-feasibility/`.
4. **Gates (to be fixed in the spec with thresholds measured from FP32-emulation noise, as
   in block 10):** decoders and resize bit-exact to PIL; encoder output vs FP64 (normalized
   L2 and worst element, per stage); FP64 reference vs llama.cpp (tanh variant) and vs HF;
   M-RoPE rows vs an FP64 construction of HF `get_rope_index`; whole-model logits with an
   image vs FP64 and libllama; served greedy text vs llama-server with the same projector on
   aligned PNGs (merger GELU difference quantified first); prefix-cache test with two
   different same-size images.

## 8. Cost estimates (estimate, from the shapes)

Per image with `N` patches: linear layers `27 · 30.45 MFLOP · N`, attention
`27 · 4,608 · N²` FLOP, merger `22.4 MFLOP · N`, total ≈ `0.845 GFLOP·N + 124,416·N²`.

| image | patches N | tokens | TFLOP |
|---|---:|---:|---:|
| 640 × 480 | 1,200 | 300 | 1.2 |
| 1280 × 800 screenshot | 4,000 | 1,000 | 5.4 |
| llama.cpp max (4,096 tokens) | 16,384 | 4,096 | 47 |
| HF max (16,384 tokens) | 65,536 | 16,384 | 590 |

At the 30–35 TFLOP/s our f16 GEMMs reach, a screenshot's encoder is ~0.2 s next to ~1 s of
text prefill for its 1,000 tokens (estimate). The HF maximum is impractical on this card
(~20 s of encoder, 16k tokens of context): the default maximum is a real trade-off (**open
decision**, knob either way). Attention over up to 16k patches needs a flash-style kernel
for head dim 72 (ours is written for 256).

Precision options (knobs, FP32 default until measured): FP32 on BF16 weights (exact weight
values); f16 WMMA (weights BF16→F16 lossy, activations F16); BF16 WMMA (weights exact,
activations rounded to BF16). HF usually runs the tower in BF16.

## 9. Open questions

1. ~~Vision `rope_theta` default~~ resolved: 10000 (section 2, item 4).
2. torchvision uint8 antialiased bicubic vs PIL `resize(BICUBIC)`: bit-identical?
3. Magnitude of llama.cpp's tanh-GELU merger deviation on real images.
4. JPEG: PIL (libjpeg-turbo) vs stb_image pixel differences on real JPEGs.
5. Default `--image-max-tokens` (model: 16,384; llama.cpp: 4,096) and whether images above
   the context are rejected or downscaled.
6. MTP draft input for image rows (embedding vs skip), by measured acceptance.
7. Whether to install the HF dev-tool venv (section 7.2).
8. Where the tower's weights live (device BF16 / device F32 / host with upload per request).
