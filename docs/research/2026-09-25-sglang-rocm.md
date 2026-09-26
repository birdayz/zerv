# SGLang on the RX 7900 XTX with Qwen3.8-27B — 2026-09-25 (source research)

**Question.** The user asked for SGLang as a competitor next to vLLM ("absolute kings in all
metrics against vLLM and SGLang"). Can SGLang serve Qwen3.8-27B on this card (gfx1100, 24 GB)
in a configuration worth benchmarking?

**Answer (source-backed, not yet measured): no fast path exists.**
- SGLang 0.5.20 runs the model's architecture on ROCm. There is no prebuilt image for gfx1100;
  a source build for it is plausible.
- 16-bit weights do not fit (54 GB).
- Every 4-bit dense-linear kernel in SGLang is either CUDA-only (Marlin) or, on HIP,
  dequantizes the whole weight matrix to fp16 on every forward call. Decode would move ~54 GB
  of fp16 per token, about 50× zerv's 15.3 GB/token of Q4_0 weight reads (estimate, not
  measured).
- A measured number needs a source build (~1 h) and an AWQ checkpoint (~20 GB). Both are
  possible; see "Open".

## Source

- `sgl-project/sglang` tag `v0.5.20`, GitHub tarball, sha256
  `b3fa51d654d52962c5deb754999ae18aeea4d06f13d497fdc5cf0246a1dfac9b`, extracted to
  `third_party/sglang/sglang-0.5.20/` (research only).
- Docker Hub, 2026-09-25: `lmsysorg/sglang` and `rocm/sgl-dev` publish ROCm images for
  `mi30x` (gfx942), `mi35x` (gfx950), `mi45x` and `gfx1151` (Strix Halo). A tag search for
  `gfx110`, `rdna`, `navi` and `gfx11` returns only the gfx1151 images. gfx1151 code objects
  do not run on gfx1100.

## Findings (paths under `python/sglang/`)

- **Architecture:** `srt/models/qwen3_5.py`, `qwen3_5_text.py` and `qwen3_5_mtp.py` exist.
  The DeltaNet layers use Triton kernels, which should run on RDNA3 (not checked).
- **gfx1100 build:** `docker/rocm-gfx1151.Dockerfile` builds from `rocm/pytorch` ROCm
  7.2.4 / PyTorch 2.9.1 (digest `7fe531fa…`) with `ARG GPU_ARCH`. It then patches
  `kernels/aot/setup_rocm.py`: the architecture allowlist, and `WARP_SIZE` 32 on both
  compiler passes (`docker/patches/sgl-kernel-gfx1151.sh`).
  - It installs aiter JIT-only with `SGLANG_USE_AITER=0` and needs
    `--attention-backend triton`.
  - gfx1100 is wave32-capable RDNA3 with the same 64 KB workgroup LDS, so the same recipe
    with `gfx1100` added to the patch is plausible.
- **compressed-tensors W4A16** (RedHat's checkpoint, the one vLLM runs):
  - `srt/layers/quantization/compressed_tensors/compressed_tensors.py:698` selects
    `CompressedTensorsWNA16` for every group or channel INT-N weight scheme.
  - Its `apply_weights` (`…/schemes/compressed_tensors_wNa16.py:327`) calls
    `apply_gptq_marlin_linear`. `kernels/ops/quantization/gptq_marlin.py:24` JIT-compiles
    `jit/csrc/gemm/marlin/gptq_marlin.cuh`, whose template issues PTX
    `mma.sync.aligned.m16n8k16` (`marlin_template.h:88`). There is no HIP path.
  - Only the MoE variant (`CompressedTensorsWNA16TritonMoE`) has a ROCm branch.
- **GPTQ:** `srt/hardware_backend/gpu/quantization/gptq_kernels.py` offers the Marlin kernel
  only.
- **AWQ:** on HIP, `srt/hardware_backend/gpu/quantization/awq_kernels.py:43` imports
  `awq_dequantize_triton`. `AWQLinearKernel.apply` (line 97) dequantizes the full `qweight`
  to fp16 on every call, then runs `torch.matmul`: a full fp16 weight write and read per step.
- **GGUF:** `srt/layers/quantization/gguf.py:69/91` warns that only CUDA, MUSA and NPU are
  supported. The ggml kernels come from `sgl_kernel` (CUDA or MUSA).
- **FP8:** RDNA3 has no FP8 matrix hardware, and the official FP8 checkpoint does not fit
  in 24 GB with KV.

## Consequence for the benchmark plan

- vLLM is the strong competitor that runs on this card. Its W4A16 path on ROCm uses its own
  kernels; see [vLLM report](../bench/2026-09-25-vllm.md).
- SGLang would only be measurable on the AWQ fallback, far below vLLM by construction.

## Open

- Measure the AWQ fallback anyway? It would take a gfx1100 source build (pip, rustup and aiter
  downloads, then a ClamAV scan of the image) and an AWQ checkpoint of ~20 GB. The user
  decides.
- Recheck each SGLang release for a HIP W4A16 or GGUF dense kernel.
