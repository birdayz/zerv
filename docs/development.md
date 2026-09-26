# Development

## Current executable scope

`zerv` is a Zig module exporting `quant` (Q4_0/Q8_0/Q4_1/Q5_K/Q6_K CPU rows),
`artifact` (bounded GGUF parsing and Linux mmap), `chat` (official Qwen3.8
text-only rendering), `text` (Unicode-9 NFC), and `tokenizer` (Qwen splitting, bounded BPE encoding, raw-byte decoding and GGUF vocab
loading), plus `gpu` (bounded native Vulkan memory/transfers/dispatch, not model
operators). `zig build` defaults to running native tests. **There is
no server executable or model-serving command yet.**

The native build imports only Zig standard library and our modules. Explicit GPU
executables additionally link the system Vulkan loader and OS libc startup/TLS;
CPU tests/tools remain driver-free. No C/C++ inference/math library is linked.
External ggml
and isolated Jinja/HF tokenizers are used only by explicit reference/benchmark
tools; neither those libraries nor
`third_party/` is read/linked by native tests. Compiled test executables were checked
as statically linked ELF files with no dynamic `NEEDED` libraries for CPU targets.
GPU executables have only system Vulkan/libc/ELF-loader direct dependencies.

## Pinned toolchain

Zig **0.16.0**, recorded in `.zig-version`. A project-local compiler was downloaded,
SHA-256 verified, extracted, and executed successfully on 2026-09-22. Nothing was
installed into the system. Reproduce on Linux x86_64:

```sh
mkdir -p .tools
curl --fail --location --retry 2 \
  -o .tools/zig-0.16.0.tar.xz \
  https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
printf '%s  %s\n' \
  70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00 \
  .tools/zig-0.16.0.tar.xz | sha256sum --check
tar -xJf .tools/zig-0.16.0.tar.xz -C .tools
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
zig version
```

The local `.tools/` directory and build products are ignored by git. The diagnostic GPU shader is compiled/validated with pinned `glslc`/SPIR-V tools;
its checked-in SPIR-V fixture removes those development tools from ordinary tests.
GPU model kernels are not implemented yet. GPU targets select the bundled LLVM+
LLD: Zig0.16's default linker rejects system GCC16 CRT `.sframe` relocations.

## Verified native commands

From the repository root, with the pinned Zig on PATH:

```sh
zig fmt --check build.zig src bench/*.zig tools/*.zig tests/*.zig
zig build test --summary all
zig build test -Doptimize=ReleaseFast --summary all
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

All four at once, in one parallel build graph (the same checks; 2026-09-26):

```sh
zig build check        # fmt, Debug and ReleaseFast unit tests, Python tests
```

### Tests in parallel (2026-09-26)

Zig 0.16 runs the tests of one binary one at a time ([ziglang/zig#15953](https://github.com/ziglang/zig/issues/15953),
open); the build runner runs independent steps on every core. So:

- **One test binary per file.** `build.zig` lists the CPU unit test files (`unit_tests`); each
  is its own compile and run step. `tests/test_build_lists.py` fails if a `tests/*.zig` file is
  neither listed, nor imported by a listed file, nor part of the GPU aggregate
  (`tests/gpu.zig`, which stays one serial binary: the device is shared).
- **Independent rounds run concurrently** inside a test through `tests/parallel.zig`:
  `parallel.rounds(n, ctx, f)` (each round seeds its own generator, `parallel.seed`), and
  `parallel.hashChunks` for golden fingerprints (chunks produced concurrently, hashed in
  order: the same digest). `std.testing.io` has one thread per CPU.
- **Golden SHA-256 in ReleaseFast.** Debug `std.crypto` SHA-256 runs at ~134 MB/s here
  (ReleaseFast 1.75 GB/s); the quant goldens hash up to 390 MB. The hashing lives in a
  ReleaseFast object (`tests/support/fast_sha256.zig`, drop-in `tests/fast_sha256.zig`) linked
  into every unit test; the code under test keeps the test's mode.
- **Optimized test binaries are stripped** (debug info doubles their LLVM time:
  an empty ReleaseFast test compiles in 14.4 s with it, 6.4 s without). Failures still print
  their messages; `-Dtest-symbols` keeps symbols for stack traces.
- **Keep a binary's run near 2 s or less** and do not let correctness depend on timing: a
  test that waits for a state must arrange it (see the batcher tests), not sleep and hope.
  Split a file when one binary dominates (`sampler_nucleus.zig`).
- **Test results are cached.** The build runner passes its per-invocation random `--seed` to
  every test binary, which made every run a cache miss; `build.zig` fixes it (`-Dtest-seed`,
  default 0), so an unchanged test binary is not run again: a no-change `zig build test` takes
  ~0.1 s. The CPU unit tests read no files at run time, so the binary is the whole input.
- **Only dependents rebuild.** Packages are modules and each unit test declares its packages
  ("Package and interface discipline" below), so an edit rebuilds and reruns only the tests
  that depend on the edited package; before the split any `src/` edit rebuilt all 19.
- `tools/test_profile.py [--optimize Debug|ReleaseFast] [name ...]`: per-test and per-binary
  times (each binary run alone), to find the next one to split or parallelize.

Measured on this machine (Ryzen 9 3900X, 12 cores / 24 threads;
[report](bench/2026-09-26-test-parallelism.md)): `zig build test` 9.4 s cold, ~4.1 s warm,
~4–7 s after a `src/` edit (was ~96 s every time); `zig build check` 22.6 s cold, ~19–22 s after
a `src/` edit (the ReleaseFast LLVM compiles, ~12 s each in parallel, are the critical path).
Build system decision (user, 2026-09-26): **`zig build` only, no Bazel for now.** If Bazel
comes later, it builds **everything** in one graph (Zig packages and tests, GLSL shaders and
their manifests, the native RDNA3 kernels, the Python goldens/oracles), not only the non-Zig
parts. The package modules above map one-to-one onto Bazel targets.
The warm run uses 64 s of CPU in 4 s: it is at the machine's throughput. Not usable:
`-fincremental` (0.16 produced a test binary that aborts).

Both modes pass the same independent goldens and malformed-input tests. Test
allocations use the testing allocator; implementation decoding allocates nothing.
See [quant validation](research/2026-09-22-quant-validation.md) for evidence and scope.
Do not treat a test run's duration as throughput. Actual component benchmark
results and limitations are in [the dated report](bench/2026-09-22-quant-decode.md).

## Repeatable CPU component benchmark

```sh
python3 bench/run_quant.py \
  --zig .tools/zig-x86_64-linux-0.16.0/zig \
  --reference /usr/lib/libggml-base.so.0.24.0 \
  --cpu 0 \
  --output docs/bench/data/YYYY-MM-DD-quant-unique-run
```

Use a **new** output directory and a permitted CPU. The runner reruns both native
correctness suites, rebuilds the native benchmark in ReleaseFast for the host CPU,
pins its own process/children to that CPU, validates full workload/output hashes
against the external decoder, and records 3 alternating comparison rounds with 5
trials per format/engine per round. It records raw results, dispersion, binary /
source hashes, compiler/host/oracle identity and exact commands. It neither changes
system clocks/power policy nor imports foreign code into the native binary.

Native-only executable build: `zig build bench-build -Doptimize=ReleaseFast -Dcpu=native`.
Its fixed workload contract is [specified here](specs/quant-benchmark.md). The separate
Python runner adds external reference validation and repeatability. This is not a
GPU/inference benchmark; a `llama-server` comparison remains mandatory once serving
exists. Report conventions: [bench/](bench/README.md).

## External goldens: explicit, not a build dependency

The reference generator's operation/ABI and oracle were researched first in
[quant-blocks.md](research/quant-blocks.md). It uses Python stdlib `ctypes` in a
separate process, checks the external results with independent scalar arithmetic,
and records the exact oracle binary and generator hash. No network/package install.

```sh
python3 tests/reference/generate_quant_goldens.py \
  --library /usr/lib/libggml-base.so.0.24.0 \
  --output third_party/quantization-candidate.json
diff -u tests/fixtures/quantization.json third_party/quantization-candidate.json
```

The output must not already exist. Do not regenerate expected data from native
outputs or blindly replace the committed golden after a failure. A changed library
hash/dirty revision requires review even if scalar outputs remain the same. Ordinary
native tests need only the committed JSON, not Python or the oracle library.

## Package and interface discipline

- **Packages are Zig modules (2026-09-26).** Each `src/NAME/` directory is one module rooted
  at `src/NAME/root.zig`, declared in `build.zig` `packages` with the packages it imports
  (like a Bazel `go_library` and its deps). Dependencies are directional:

  ```
  quant   gpu   artifact   text   chat          (std only)
  matvec    → gpu
  tokenizer → text, artifact
  model     → artifact, gpu, matvec
  session   → chat
  serve     → chat, session, model, tokenizer
  ```

  - Across packages import the module (`@import("gpu")`, `@import("artifact").gguf`), never
    a path (`../gpu/…`); within a package, relative files. Zig resolves imports lazily, so
    an unreferenced import of an undeclared package would compile: `tests/test_build_lists.py`
    checks that each package imports only what it declares, and that the directories and
    the list agree.
  - `src/root.zig` is the `zerv` umbrella re-exporting every package, for executables that use
    several (server, benchmarks, tools, GPU tests). Packages never import it.
  - A unit test declares the packages it imports (`unit_tests` in `build.zig`), so an edit
    rebuilds and reruns only the tests of the edited package's dependents (measured: a
    `session` edit rebuilds session, sampler, sampler_nucleus, prefix, serve, batcher, tools;
    a `quant` edit only the four quant tests).
  - Tests exercise each package's public API from `tests/`.
- Keep artifact parsing, model semantics, backend/device management, state/scheduling,
  generation, and HTTP as cohesive boundaries as they become real functionality.
- Core tensor/quant math must not depend on HTTP, configuration parsing, global
  state, or external oracles. An artifact parser must not allocate GPU resources.
  Model planning may depend on typed tensor/backend capabilities, not driver handles
  leaked throughout the graph. HTTP maps protocol to service requests, not kernels.
- Publish small APIs with ownership, aliasing, allocation, lifetime, precision,
  error, and concurrency contracts. Keep internals private; avoid dependency cycles,
  hidden singleton allocators, and bidirectional callbacks that bypass ownership.
- Introduce packages when a cohesive implementation needs them, not empty directory
  scaffolding. Prefer static dispatch or planning-time specialization without
  scattering model/device checks across unrelated layers.
- Maintain independent reference tests and performance-sensitive component benches
  at these boundaries; add integration tests where data/state crosses them. Clarity
  and tests are requirements, not sacrifices made for an unmeasured optimization.

## Native artifact inspection and new component benchmarks

```sh
zig build inspect-build -Doptimize=ReleaseFast
zig-out/bin/zerv-inspect models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf
python3 bench/run_gguf.py --library /usr/lib/libggml-base.so.0.24.0 \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --output docs/bench/data/YYYY-MM-DD-gguf-unique --cpu 2
.tools/tokenizer-oracle-venv/bin/python bench/run_chat.py \
  --config third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json \
  --output docs/bench/data/YYYY-MM-DD-chat-unique --cpu 2
```

The exact actual run directories/results are linked in the
[GGUF report](bench/2026-09-22-gguf-loading.md) and
[chat report](bench/2026-09-22-chat-template.md). Chat, normalization and tokenizer
oracles/benchmarks need isolated development Python packages; ordinary tests do not. Setup used:

```sh
python3 -m venv .tools/tokenizer-oracle-venv
.tools/tokenizer-oracle-venv/bin/python -m pip install \
  tokenizers==0.22.2 Jinja2==3.1.6 regex==2026.9.10
```

No system package changes. Full installed versions/hashes are in fixture/benchmark
manifests. Unicode-9 NFC has its own completed verification and benchmark loop:

```sh
zig build nfc-bench-build -Doptimize=ReleaseFast
.tools/tokenizer-oracle-venv/bin/python bench/run_nfc.py \
  --cpu 2 --output docs/bench/data/YYYY-MM-DD-nfc-unique
.tools/tokenizer-oracle-venv/bin/python tests/reference/generate_nfc.py \
  --ucd third_party/unicode/9.0.0 \
  --data third_party/nfc-candidate/nfc9.bin \
  --fixtures third_party/nfc-candidate/fixtures
```

Use fresh output paths. Compare regenerated data/fixtures to `src/text/data/nfc9.bin`
and `tests/fixtures/nfc9/` rather than replacing expected results. The
[NFC report](bench/2026-09-22-normalization.md) records two actual runs, full source/
embedded-data snapshots and the boundary between NFC timing and llama-server.
Follow [TODO.md](../TODO.md): one active building block/package at a time.
The subsequent Qwen splitter is independently verified and measured as well:

```sh
zig build split-bench-build -Doptimize=ReleaseFast
.tools/tokenizer-oracle-venv/bin/python bench/run_split.py \
  --cpu 2 --output docs/bench/data/YYYY-MM-DD-split-unique
```

Its [report](bench/2026-09-22-tokenizer-split.md) contains the exact independent
fixture-regeneration command, two actual runs and the llama-server comparison
boundary. The subsequent complete tokenizer is also verified and measured:

```sh
zig build tokenizer-bench-build -Doptimize=ReleaseFast
zig-out/bin/zerv-tokenizer-bench models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  tests/fixtures/tokenizer/manifest.json
```

See the [complete tokenizer report](bench/2026-09-22-tokenizer.md) for fixture
regeneration and two benchmark runs including actual llama-server launch/config
records. Raw decoding deliberately does not perform per-piece UTF-8 replacement.

Regenerate fixtures explicitly into new paths:

```sh
python3 tests/reference/gguf_oracle.py --library /usr/lib/libggml-base.so.0.24.0 \
  fixtures --output third_party/gguf-candidate
.tools/tokenizer-oracle-venv/bin/python tests/reference/generate_chat_goldens.py \
  --config third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json \
  --publisher-template third_party/unsloth/Qwen3.8-27B-GGUF/4ca720788d1e01f1bff70c033e0d0028fd02e502/embedded-template.jinja \
  --output third_party/chat-candidate.json
# Tool-calling prompt oracle (tests/fixtures/chat-tools.json); --compare RAW.json diffs it
# against llama-server /apply-template prompts recorded by tools/check_tool_parity.py.
.tools/tokenizer-oracle-venv/bin/python tests/reference/render_tools.py \
  --output third_party/chat-tools-candidate.json
```

Tool-calling parity with llama-server (one engine at a time; GPU):

```sh
python3 tools/check_tool_parity.py --engines llama-fp32-full --output docs/bench/data/YYYY-MM-DD-tool-parity/llama
python3 tools/check_tool_parity.py --engines zerv --zerv-binary PATH \
  --reference-raw docs/bench/data/YYYY-MM-DD-tool-parity/llama/raw.json --output docs/bench/data/YYYY-MM-DD-tool-parity/zerv
```

## Matched tokenizer comparison and optimization

The old direct-native/HTTP table is **not** relative tokenizer-speed evidence.
The direct reference harness checks the same installed libllama used by the server:

```sh
.tools/tokenizer-oracle-venv/bin/python bench/run_tokenizer_matched.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --tokenizer third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json \
  --cpu 10 --output docs/bench/data/YYYY-MM-DD-tokenizer-matched-unique
```

No server needs to run for these **component** timings. Encode includes owned-output
allocation/free for both engines; raw decode uses preallocated buffers. An additional
reference-preallocated encode control favors llama. `--baseline-run PATH` interleaves
a saved verified native binary after checking its corpus/binary/source identities.
Binaries are retained in ignored `third_party/tokenizer-matched/`, not docs or native
runtime dependencies. Use fresh output paths. [Contract](specs/tokenizer-matched-benchmark.md),
[fairness audit](research/tokenizer-comparison-fairness.md).

Additional independent fixtures can be regenerated with
`tests/reference/generate_tokenizer_optimization.py --tokenizer PATH --output NEW`.
They exercise all ASCII pairs, merge-length boundaries and long/unique workloads;
compare regenerated bytes rather than replacing expectations after a test failure.

If the ignored saved baseline binary is gone, rebuild from the archived sources:

```sh
python3 bench/rebuild_tokenizer_baseline.py \
  --run docs/bench/data/2026-09-22-tokenizer-matched-wide-baseline \
  --output third_party/tokenizer-restored/NEW-UNIQUE-RESTORE \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf
```

Then use `--baseline-run third_party/tokenizer-restored/NEW-UNIQUE-RESTORE` with the
matched runner above. This was actually rebuilt and rerun; [evidence](bench/data/2026-09-22-tokenizer-replay/manifest.json).
No original executable is required. Pinned model/compiler/HF configuration and
installed reference library remain explicit prerequisites; no silent downloads.

## Mixed actual-model quant decoders

The matched direct-loop harness supports `--format q4_1`, `q5_k`, and `q6_k`.
It rebuilds both binaries, runs both native test modes, checks actual model/library/
header identities, and validates full slice outputs before accepting timings.

```sh
python3 tests/reference/generate_q6_k_goldens.py \
  --library /usr/lib/libggml-base.so.0.24.0 \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --output third_party/NEW-q6_k-candidate.json
cmp tests/fixtures/q6_k.json third_party/NEW-q6_k-candidate.json
python3 bench/run_model_quant.py --format q6_k \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --cpu 10 --output docs/bench/data/NEW-q6_k-run
```

Use fresh paths. Q6_K measures 1/64/**4096-row slices** of output.weight, not the
entire 248320-row vocabulary projection. CPU diagnostic decoding is not GPU matvec
or a serving comparison. Earlier formats retain their existing row/tile/full-tensor
workloads and their own `generate_q4_1_goldens.py`/`generate_q5_k_goldens.py` tools.

## Explicit native GPU checks and repeatable driver benchmark

```sh
zig build gpu-test --summary all
zig build gpu-test -Doptimize=ReleaseFast --summary all
zig build gpu-driver-bench-build -Doptimize=ReleaseFast -Dcpu=native
zig-out/bin/zerv-gpu-driver-bench
python3 bench/run_gpu_driver.py --cpu 10 --output docs/bench/data/NEW-gpu-driver
```

Shader spill gate (required when shaders, compile defines or tuning tables change;
[RCA](bench/2026-09-24-aco-lds-spill.md)). Each command must report 0 FAIL (no VGPR
spills in LDS); WARN lines are scratch spills (performance items):

```sh
python3 tools/check_shader_spills.py --log third_party/spill-gate/gpu-test.txt -- zig build gpu-test -Doptimize=ReleaseFast
python3 tools/check_shader_spills.py --log third_party/spill-gate/spec.txt -- zig-out/bin/zerv-spec-check MODEL
python3 tools/check_shader_spills.py --log third_party/spill-gate/mtp.txt -- zig-out/bin/zerv-mtp-check MODEL third_party/mtp-check/tokens-short-nothink.json third_party/spill-gate/mtp-dump
python3 tools/check_shader_spills.py --log third_party/spill-gate/f16.txt -- zig-out/bin/zerv-model-profile MODEL 4096 512 600 2 f16 f16
```

GPU tests exercise actual transfers/compute and resource/state failures; default
`zig build test` still needs no GPU/driver. Diagnostic shader compilation occurs
only in the explicit external generator, not at runtime. The runner rebuilds both
native and independent C binaries; no old executable is required. CPU+GPU tests,
full output hashes and matching device/queue/allocation metadata gate all timings.
No tuned llama-server/inference claim follows from this raw driver benchmark.

[GPU validation, failures, fixture/binding regeneration commands](research/2026-09-22-gpu-driver-validation.md).
Regeneration/benchmarking require the pinned research C headers/XML (not native
runtime dependencies). Restore/verify those small research files explicitly:

```sh
python3 - <<'PY'
from pathlib import Path
import json, hashlib, urllib.request
for source in json.loads(Path('docs/research/2026-09-22/vulkan-sources.json').read_text()):
    path = Path(source['local_path'])
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(source['url']).read()
    assert hashlib.sha256(data).hexdigest() == source['sha256'], path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
PY
```

This verified all 26 retained files here; the pinned URLs were downloaded during
research. Existing mismatched files are refused, not replaced. Toolchain/driver
identity changes require deliberate review; no automatic package install or weight
download is part of this path.

## Native machine code (`src/model/native/`)

The f16 `gemm_f16x` kernel also ships as our own RDNA3 machine code in a RADV pipeline binary
([spec](specs/prefill.md), [report](bench/2026-09-24-gemm-f16x-isa.md)). Ordinary builds only
embed the files. Regenerating them needs the GPU with RADV, clang and the lab binary, and must
be followed by the gates in the spec (`zig build gpu-test`, `verify_model.py --gemm-code`,
serving outputs):

```sh
cc -std=gnu11 -O2 -Wall -Wextra -Ithird_party/vulkan/1.4.354/include \
  bench/isa_lab/pipeline_binary_lab.c -lvulkan -lm -o third_party/isa-lab/pipeline_binary_lab
python3 tools/build_native_gemm.py --output-dir third_party/NEW-native   # then review and copy
```

The binary is valid only where the driver's global pipeline key equals
`gemm_f16x_q4_0.global` (same Mesa build, GPU and compiler options); elsewhere the server logs
the fallback and runs the SPIR-V. `--gemm-code spirv` selects the SPIR-V explicitly.

## Resident GPU matvec — verified replay and measurements

```sh
# Ordinary checks use committed fixtures/SPIR-V, not ggml or shader compilers.
zig build test
zig build gpu-test
zig build test gpu-test -Doptimize=ReleaseFast

# Explicit independent replay: fresh paths, existing pinned model/tools/libraries.
python3 tools/replay_matvec.py --output-dir .tools/NEW-matvec-replay --restore-sources

# Rebuild both competitors and benchmark all eleven complete dense shape/type pairs.
python3 bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-run1
python3 bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-repeat
```

The replay tool verifies 62 retained ggml research files, optionally fetching only
missing pinned small sources. It rejects modified existing files, compares freshly
regenerated independent fixture and eight SPIR-V modules byte-for-byte, and runs fmt,
Debug/ReleaseFast CPU+GPU and Python tests. No stale binary, package installation
or weight download is needed. Original replay actually executed and passed; see
[validation/logs](research/2026-09-22-gpu-matvec-validation.md).

To rebuild shaders alone into a review directory:
`python3 tools/compile_matvec.py --output-dir .tools/NEW-matvec-shaders`.
Nothing automatically overwrites expected outputs or production shader files.
The benchmark separately records FP32-input and default (activation-quantized)
reference precision. [Measured losses and boundary caveats](bench/2026-09-22-gpu-matvec.md).
These are component commands, **not** a native Qwen server.

The [block08b DFS](research/matvec-optimization.md) adds a source-rebuilt scalar
counterfactual, correctness-gated candidate harness and separately labelled aligned
views. Full repeated protocol (fresh output directories; no concurrent GPU work):

```sh
python3 bench/rebuild_matvec_baseline.py --output third_party/matvec-dfs/NEW-baseline
python3 bench/run_gpu_matvec.py --cpu 10 --aligned \
  --baseline-run third_party/matvec-dfs/NEW-baseline \
  --output docs/bench/data/NEW-dfs-run1
python3 bench/run_gpu_matvec.py --cpu 10 --aligned \
  --baseline-run third_party/matvec-dfs/NEW-baseline \
  --output docs/bench/data/NEW-dfs-repeat
```

`native` retains the original offset2 quantized-weight stress placement;
`native-aligned` uses offset0. Do not silently interchange them when quoting
speedup ratios. Both use the same weights, FP32 x and independent numerical gates.
`bench/tune_matvec.py --help` describes exploratory shader replacement, outside
all timing regions; every accepted candidate must still pass all48 fixtures and
all11 full-shape outputs. [Dated results](bench/2026-09-22-matvec-optimization.md).

## Native model, session and server (blocks 09–12)

```sh
export PATH="$PWD/.tools/zig-x86_64-linux-0.16.0:$PATH"
zig build server -Doptimize=ReleaseFast            # zig-out/bin/zerv
zig-out/bin/zerv --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf --port 8080 --context 8192
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hi"}]}'

# Oracle (explicit; needs pinned libllama/ggml, the model, and the tokenizer venv
# for template rendering). Fresh paths; the fixture replays except two volatile
# fields (libllama stderr hashes, absolute build path).
python3 tests/reference/generate_model_oracle.py --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir third_party/model-oracle/NEW --output .tools/NEW-oracle.json

# The >512-token case (one full 512-row chunk; block 13h) is a separate fixture.
python3 tests/reference/generate_model_oracle.py --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir third_party/model-oracle/NEW-long --output .tools/NEW-oracle-long.json --case-set long

# Native forward gates against the oracle work dirs (thresholds in docs/specs/model.md).
# The default fixture was regenerated on 2026-09-23 with the current generator. Its data
# files are byte-identical to 2026-09-22-a; only oracle.stderr differs. The fixture's
# file hashes refer to the new directory. The old fixture is kept in
# docs/bench/data/2026-09-23-gemm-efficiency/qwen38-oracle-2026-09-22-a.json.
python3 tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-23-default-regen \
  --modes 0,1,13,29,60,512 --work-dir third_party/model-native/NEW --report docs/bench/data/NEW/report.json
python3 tools/verify_model.py --fixture tests/fixtures/model/qwen38-oracle-long.json \
  --oracle-dir third_party/model-oracle/2026-09-23-long --modes 0,512,64 \
  --work-dir third_party/model-native/NEW-long --report docs/bench/data/NEW/report-long.json

# Rerun a gate with an archived capture binary (every work dir keeps its copy), e.g. to
# attribute a difference to a build: same flags plus --tool.
python3 tools/verify_model.py --tool third_party/model-native/OLD/zerv-model-capture \
  --oracle-dir third_party/model-oracle/2026-09-23-default-regen --modes 0 \
  --work-dir third_party/model-native/NEW-rerun --report third_party/model-native/NEW-rerun.json

# Interleaved (ABBA) A/B of zerv-model-profile binaries: per-phase decode/prefill GPU time.
# An engine may carry its own arguments after "|", e.g. KV page sizes of one binary.
python3 bench/race_profile.py --engine old=PATH --engine new=PATH --rounds 2 \
  --output docs/bench/data/NEW-race --args 32768 512 30000 32 f16@native f32
python3 bench/race_profile.py --engine "p128=PATH|32768 512 30000 32 f16@native f32@page=128" \
  --engine "ctx=PATH|32768 512 30000 32 f16@native f32@page=context" --rounds 2 \
  --output docs/bench/data/NEW-pages --args 32768 512 30000 32 f16@native f32
# verify_model.py --kv-page-tokens N|context gates another KV page size (default 128).

# End-to-end greedy equality through the real server (JSON and SSE).
python3 tools/check_session.py --output docs/bench/data/NEW/session.json

# Matched serving benchmark vs llama-server (engines run one at a time).
python3 bench/run_serving.py --output docs/bench/data/NEW-serving
```

`zig build model-capture-build` builds the native capture tool used by
`verify_model.py`. The oracle's FP64 pass uses all CPU cores for ~25 minutes; do
not run GPU benchmarks concurrently with it.
