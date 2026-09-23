# Stable third-party research code

Third-party source used for research belongs in the gitignored **`third_party/`**
folder, not `/tmp`. Findings/specifications/measurements still belong under `docs/`.
The native build must never import or link research code from this directory.

## Current stable paths

Downloaded source snapshots were moved to **`third_party/research/2026-09-22/`**:

| Path relative to that folder | Purpose |
| --- | --- |
| `llama-qwen35.cpp` | Reference model graph and gating |
| `llama-model.cpp` | Memory/backend construction |
| `llama-hparams.cpp` | KV/recurrent dimensions |
| `llama-memory-recurrent.cpp` | State allocation/rollback storage |
| `llama-vulkan.cpp` | Reference Vulkan operator/capability handling |
| `llama-build.md` | Vulkan/HIP baseline build requirements |
| `ggml-common.h`, `ggml-quants.c` | Block format/decoding in inspected llama.cpp revision |
| `oracle-ggml-common.h`, `oracle-ggml-quants.c` | Block/decoder source at installed oracle's reported source commit |
| `qwen35-modeling.py`, `qwen35-configuration.py` | Transformers architecture/field semantics |
| `ik-README.md`, `ik-build.md` | Alternative reference support/limitations |
| `qwen38-*`, `unsloth-*` | Model metadata/cards, not weight files |

[The source ledger](research/2026-09-22/sources.json) records origin, revision, stable
local path, and SHA-256. These are selected files, **not complete buildable source
checkouts**. Read upstream licenses before using any reference in tooling; research
access does not permit copying implementation into our engine.

## Reproducing / extending this folder

For a recorded file, use the ledger's exact URL and local path, for example:

```sh
mkdir -p third_party/research/2026-09-22
curl --fail --location \
  https://raw.githubusercontent.com/ggml-org/ggml/456172ec733a135778adcd32d00e576a58232e45/src/ggml-quants.c \
  -o third_party/research/2026-09-22/oracle-ggml-quants.c
sha256sum third_party/research/2026-09-22/oracle-ggml-quants.c
```

Compare the hash with the ledger. For future complete checkouts, prefer stable
revision-qualified paths such as `third_party/llama.cpp/<full-commit>/`, detached
at that commit, with baseline build outputs separately named by backend/profile.
Record dirty patches and toolchain identity in docs; never silently update a
reference underneath existing fixtures or benchmark results.

Explicit candidate goldens can live here during review. Accepted small data-only
goldens live under `tests/fixtures/`; ordinary native tests consume those without
third-party libraries. Oracle programs may be built/run externally for research,
fixture generation and comparisons only. `git check-ignore third_party/...` must
confirm exclusion; do not force-add large checkouts or model weights.

Additional revision-qualified research snapshots now include:

- `third_party/ggml/456172ec733a135778adcd32d00e576a58232e45/`: GGUF spec, C ABI,
  reader/writer implementation (external research/oracle only).
- `third_party/llama.cpp/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/`: tokenizer,
  Unicode, public ABI, server docs and captured installed help. Added for the output
  pipeline study: the chat/PEG parser sources (`common/chat*.{h,cpp}`,
  `common/peg-parser.{h,cpp}`, `common/parsers/{qwen3-coder,parsers}.cpp`,
  `parsers.h`) and `tools/server/server-{context,task,common}.cpp`. Each git blob hash
  matches that commit's `research-tree.json`. SHA-256 values are in
  [output parsing](research/output-parsing.md).
- `third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/`:
  official tokenizer/config and template extracted verbatim from that config.
- `third_party/unsloth/Qwen3.8-27B-GGUF/4ca720788d1e01f1bff70c033e0d0028fd02e502/`:
  embedded template extracted verbatim from the selected SHA-verified GGUF.

Source-file hashes/URLs are in the ledger; derived-template hashes are in
`tests/fixtures/chat-template.json`. No native build reads these paths.
