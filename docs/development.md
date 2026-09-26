# Development

## Current executable scope

`zerv` is a Zig module exporting `quant` (Q4_0/Q8_0/Q4_1/Q5_K/Q6_K CPU rows),
`artifact` (bounded GGUF parsing and Linux mmap), `chat` (official Qwen3.8
text-only rendering), `text` (Unicode-9 NFC), and `tokenizer` (Qwen splitting, bounded BPE encoding, raw-byte decoding and GGUF vocab
loading), plus `gpu` (bounded native Vulkan memory/transfers/dispatch, not model
operators). Bazel builds and tests everything (below).

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

Bazel builds and tests everything: the Zig packages, tests, server, benchmarks and tools, and
the Python tests (user decision 2026-09-26: **Bazel only**, one graph with exact caching;
conventions follow `../fdb-go`), **hermetically** ([spec](specs/hermetic-build.md)): every
compiler, library, header and interpreter an action uses is a sha256-pinned download or built
in the graph; nothing from the host's `/usr` besides the platform (kernel, glibc, bash,
coreutils). Nothing is installed into the system besides `bazelisk`, which downloads the Bazel
of `.bazelversion` and runs it through `tools/bazel` (`--nohome_rc --nosystem_rc`: another
project's `~/.bazelrc` injected `-I`/`-L` host paths into our C++ actions). Proof:
`tools/hermetic_check.sh` runs `bazel test //...` in a slim Debian container without compilers,
Python, Vulkan or shader tools, from empty caches. Pins:

- Bazel **9.2.0** (`.bazelversion`, latest stable on 2026-09-26); `MODULE.bazel` and its lock
  file `MODULE.bazel.lock` pin every module.
- Zig **0.16.0** through rules_zig 0.16.0: the SDK tarball sha256
  `70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00`, the same as the former
  project-local `.tools/zig-x86_64-linux-0.16.0` (its `zig` executable, sha256 `2317bbb9…`, is
  byte-identical, so manifests' `zig_sha256` stay comparable). `bazel run //bazel:zig -- …`
  runs it.
- Python **3.14** (rules_python 2.3.4, hermetic interpreter, started directly: `bootstrap_impl
  = script`, the default bootstrap runs the host's `python3` first) with hash-locked packages
  (`requirements_lock.txt`: numpy 2.5.3) for the Python tests.
- C/C++ for development tools only: hermetic_cc_toolchain 4.3.0 (clang of a pinned Zig 0.15.2
  SDK, bundled libc/libc++ headers, glibc 2.28 target); the host's C/C++ toolchain is never
  configured (`BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN=1`). `//src:no_cc` (an aspect over the
  dependencies of `//src:zerv`) fails if a C/C++ rule reaches what zerv ships.
- Vulkan at link time: `//src/gpu:vulkan`, a stub `libvulkan.so.1` generated from the pinned
  Vulkan registry (Vulkan-Headers vulkan-sdk-1.4.357.0; the 234 core commands, each traps if
  ever called). The dynamic linker loads the real loader by that SONAME at run time;
  `tests/test_linkage.py` checks every executable: static, or only libvulkan/glibc, no
  RPATH/RUNPATH.
- Zig target `x86_64-linux-gnu.2.43` (`bazel/BUILD.bazel`): rules_zig's default glibc 2.17
  lacked symbols the host Vulkan loader references when it was linked (glibc 2.34/2.38; `zig
  test` links with `--no-allow-shlib-undefined`) and links libm/libpthread/libdl separately. 2.43 is the newest
  Zig 0.16 provides (the host has 2.44). Every Zig compile uses `-mcpu=native` (as `zig build`
  did; a cache shared across machines would need an explicit CPU model).

- GPU tests: a test-only Vulkan runtime built in the graph (`//tests:gpu_runtime`: Mesa 26.2.3
  RADV, Vulkan-Loader 1.4.357.0, libdrm 2.4.133); the tests use it and check that they do
  (`ZERV_TEST_GPU_RUNTIME`). Production zerv and benchmarks use the host's driver.
- Shader tools (glslc, glslang, SPIRV-Tools), oracles (ggml, llama.cpp), vulkaninfo and the
  llama-server competitor are built from pinned sources; harness scripts execute no host
  program (`tests/test_build_lists.py`). Spec: [hermetic-build.md](specs/hermetic-build.md).

## Verified native commands

From the repository root (`just` recipes in `justfile`):

```sh
bazel test //...        # required checks: zig fmt, CPU unit tests Debug + ReleaseFast, Python tests,
                        # generated files up to date (SPIR-V, native code, vk.zig, Vulkan fixtures;
                        # tools built from source)
bazel test --config=quick //...   # Debug unit tests and Python tests only
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills   # real device, test-only runtime
tools/py tools/zerv_build.py --test-host-gpu   # the GPU tests on the host's driver (production's; cached
                                               # per driver identity; production benchmarks run it)
tools/hermetic_check.sh --gpu //... //tests:gpu   # the same in a slim container, empty caches
bazel build --config=release //...   # ReleaseFast, native CPU: server, benchmarks, tools
tools/zerv_build.py zerv-spec-check  # build (release) and print an executable's path
```

- **One Bazel package per directory, explicit deps.** `src/NAME/BUILD.bazel` is the
  package's `zig_library` (module `NAME`) with its dependencies and embedded data;
  `//src:zerv_lib` the umbrella module `zerv`; `//src:zerv` the server. Tests:
  `tests/BUILD.bazel`, macro `zerv_test` (`bazel/defs.bzl`) = the test in Debug plus
  `NAME_release_fast` (a `zig_configure_test`, stripped, tag `release_fast`), each declaring the
  packages it imports, the files it embeds and the files it opens at run time (`data`). The
  golden SHA-256 is a ReleaseFast `zig_static_library` (`//tests/support:fast_sha256`). Python
  tests are `py_test`s declaring every script and file they read (runfiles).
- **GPU tests** (`//tests:gpu`, `//tests:gpu_release_fast`; tags `manual`, `exclusive`, `gpu`,
  so `//...` does not run them) load the host's Vulkan loader and RADV at run time until the
  loader and Mesa are built in the graph (spec, phase 4); meanwhile a driver update does not
  invalidate their cached results (`--nocache_test_results` reruns them).
  `//tests:gpu_spills` is the shader spill gate (`tools/check_shader_spills.py`) over the
  ReleaseFast GPU tests; its RADV statistics land in
  `bazel-testlogs/tests/gpu_spills/test.outputs/shaderstats.txt`.
- **Harnesses build through `tools/zerv_build.py`**: `binary(name)` / `build(...)` run
  `bazel build --config=release` and take the path from `bazel cquery` of the same
  configuration; `test_command()` gates a measurement on `bazel test //...` (GPU runners add
  the GPU tests); `provenance()` records Bazel's version, the Zig executable's version and
  sha256 and the hash of every build definition file (`build_files()`: `.bazelversion`,
  `.bazelrc`, `.bazelignore`, `MODULE.bazel`, its lock file, `requirements_lock.txt`, `bazel/*.bzl`, every
  `BUILD.bazel`), which the runners also copy with their source snapshots.
  `tests/test_build_lists.py` fails if a bench/tools script builds any other way; the
  exceptions are `bench/rebuild_*_baseline.py`, which rebuild archived trees with their own
  `build.zig` (default compiler: the toolchain's Zig, the same executable).
- **Caching:** action results in a disk cache shared by all worktrees
  (`~/.cache/bazel/zerv-disk`, 20 GB bound); test results are cached per test binary and its
  declared inputs (the test runner's seed is 0 without `--seed`, as `-Dtest-seed` defaulted).
  `zig fmt --check` is a test (`//:zig_fmt`) over every package's `zig_srcs`. One output base
  for all configurations: switching between the default and `--config=release` discards
  Bazel's analysis cache, measured ~150 ms here (not the per-config output bases fdb-go
  needs). `.bazelignore` keeps `docs/bench/data` (benchmark source snapshots, which contain
  BUILD files of their tree), `third_party/` and `models/` out of `//...`.
- **The sandbox found undeclared inputs** the Zig build had missed silently: `matvec` embeds
  `shaders/separate/*.spv`; `tests/gguf.zig` and `tests/model.zig` open
  `tests/fixtures/gguf/default.gguf` at run time; `matvec_gpu.zig` opens
  `src/matvec/shaders/f32_small.spv`. Correction to the `zig build` result caching below: "the
  CPU unit tests read no files at run time" was wrong for `gguf` and `model`; a change to that
  fixture would not have invalidated their cached results. The Python tests' modules under test
  resolved their own path (`Path(__file__).resolve()`) out of the runfiles into the source tree
  and so read undeclared files there; with `absolute()` six tests failed until their inputs
  were declared (`gguf_oracle.py`, `generate_vulkan_goldens.py`, the NFC, split and tokenizer
  golden binaries, `bench/workloads/long-v1.json`).
- **Measured** (59 tests: 19 Zig files × 2 modes, 20 Python, fmt; nice'd, another session's GPU
  benchmark running): after `bazel clean` without disk cache 42.9 s; no-op 0.2 s; an edit in
  `session` 21.5 s (the ReleaseFast LLVM compiles of its dependents); reverting it 1.0 s (disk
  cache); `--config=quick` no-op 0.24 s.
- **Generated files are committed and checked** (as fdb-go commits generated code and fails
  CI on a diff): Bazel regenerates the SPIR-V modules and the native kernel code and tests
  that the committed files are exactly that output ("Shaders" and "Native machine code"
  below). The build embeds the committed files: reviewable diffs of what ships, the native
  kernel's key hashes the SPIR-V bytes, and editors see the files.

### Shaders (`src/matvec/shaders`, `src/model/shaders`)

`//src/matvec:generated_shaders` and `//src/model:generated_shaders` run
`tools/compile_matvec.py` / `tools/compile_model.py` (py_binaries carrying their GLSL sources;
the variant tables stay in those scripts) with glslc and spirv-val built from source:
shaderc v2026.3, glslang and SPIRV-Tools of vulkan-sdk-1.4.357.0 (168d452a, 9a49b088),
SPIRV-Headers 29981f65 — the revisions of the host packages that produced the committed
modules, all 172 of which the source-built tools reproduce byte for byte (only the manifests'
`tools` entries changed: the tools' version lines instead of host binary hashes).
SPIRV-Tools and SPIRV-Headers carry upstream BUILD files; glslang and shaderc use ours
(`bazel/third_party/*.BUILD`, sources and defines as upstream BUILD.gn/CMake, no HLSL). One
action per package, compiling in parallel (8 jobs; matvec 100 modules 33 s → 3.2 s, model 72
modules 16 s → 1.9 s standalone, byte-identical output). `generated_shaders_test` (tag
`shaders`, part of `//...`) fails on any missing, extra or differing file; after an intended
change of a source, the tools or a variant table:

```sh
bazel run //src/matvec:update_shaders    # writes src/matvec/shaders (and manifest.json)
bazel run //src/model:update_shaders
```

then the GPU gates (spill gate, GPU tests, model gates). Harnesses that compile experimental
variants get the same tools from `tools/zerv_build.py` `shader_tools()`, never the host's.

### Tests in parallel (2026-09-26)

Zig 0.16 runs the tests of one binary one at a time ([ziglang/zig#15953](https://github.com/ziglang/zig/issues/15953),
open); the build tool runs independent tests on every core (first `zig build`, now Bazel).
So:

- **One test binary per file.** `tests/BUILD.bazel` has a test per CPU unit test file (then
  `build.zig`'s `unit_tests`). `tests/test_build_lists.py` fails if a `tests/*.zig` file is
  not a test's main or source or a helper library (the GPU aggregate `tests/gpu.zig` stays one
  serial binary: the device is shared).
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
  their messages; drop `-fstrip` from the `zerv_test` macro for stack traces.
- **Keep a binary's run near 2 s or less** and do not let correctness depend on timing: a
  test that waits for a state must arrange it (see the batcher tests), not sleep and hope.
  Split a file when one binary dominates (`sampler_nucleus.zig`).
- **Test results are cached.** `zig build`'s runner passed a per-invocation random `--seed`
  to every test binary, which made every run a cache miss; `build.zig` fixed it (`-Dtest-seed`,
  default 0), so an unchanged test binary was not run again (a no-change `zig build test` took
  ~0.1 s). Bazel runs the binaries without `--seed` (seed 0) and caches results per binary and
  declared inputs.
- **Only dependents rebuild.** Packages are modules and each unit test declares its packages
  ("Package and interface discipline" below), so an edit rebuilds and reruns only the tests
  that depend on the edited package; before the split any `src/` edit rebuilt all 19.
- `tools/test_profile.py [--optimize Debug|ReleaseFast] [name ...]`: per-test and per-binary
  times (each binary run alone, in its runfiles), to find the next one to split or
  parallelize; per-target times: `bazel test --test_summary=detailed`.

Measured on this machine (Ryzen 9 3900X, 12 cores / 24 threads;
[report](bench/2026-09-26-test-parallelism.md)): `zig build test` 9.4 s cold, ~4.1 s warm,
~4–7 s after a `src/` edit (was ~96 s every time); `zig build check` 22.6 s cold, ~19–22 s after
a `src/` edit (the ReleaseFast LLVM compiles, ~12 s each in parallel, are the critical path).
Build system decisions (user, 2026-09-26): first **`zig build` only, no Bazel for now**, and if
Bazel comes, it builds **everything** in one graph (Zig packages and tests, GLSL shaders and
their manifests, the native RDNA3 kernels, the Python goldens/oracles); then the move to
Bazel only (above). The package modules map one-to-one onto Bazel targets.
The warm run uses 64 s of CPU in 4 s: it is at the machine's throughput. Not usable:
`-fincremental` (0.16 produced a test binary that aborts).

Both modes pass the same independent goldens and malformed-input tests. Test
allocations use the testing allocator; implementation decoding allocates nothing.
See [quant validation](research/2026-09-22-quant-validation.md) for evidence and scope.
Do not treat a test run's duration as throughput. Actual component benchmark
results and limitations are in [the dated report](bench/2026-09-22-quant-decode.md).

## Repeatable CPU component benchmark

```sh
tools/py bench/run_quant.py \
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

Native-only executable build: `bazel build --config=release //bench:zerv-quant-bench`.
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
tools/py tests/reference/generate_quant_goldens.py \
  --output third_party/quantization-candidate.json
diff -u tests/fixtures/quantization.json third_party/quantization-candidate.json
```

The output must not already exist. Do not regenerate expected data from native
outputs or blindly replace the committed golden after a failure. A changed library
hash/dirty revision requires review even if scalar outputs remain the same. Ordinary
native tests need only the committed JSON, not Python or the oracle library.

## Package and interface discipline

- **Packages are Zig modules (2026-09-26).** Each `src/NAME/` directory is one module rooted
  at `src/NAME/root.zig`, declared by the `zig_library` in `src/NAME/BUILD.bazel` with the
  packages it imports (`deps`). Dependencies are directional:

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
    checks that each package imports only what it declares, and that every package directory
    has its BUILD file and library.
  - `src/root.zig` is the `zerv` umbrella re-exporting every package, for executables that use
    several (server, benchmarks, tools, GPU tests). Packages never import it.
  - A unit test declares the packages it imports (its `deps` in `tests/BUILD.bazel`), so an edit
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
bazel build --config=release //tools:zerv-inspect
bazel-bin/tools/zerv-inspect models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf
tools/py bench/run_gguf.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --output docs/bench/data/YYYY-MM-DD-gguf-unique --cpu 2
tools/py bench/run_chat.py \
  --config third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json \
  --output docs/bench/data/YYYY-MM-DD-chat-unique --cpu 2
```

The exact actual run directories/results are linked in the
[GGUF report](bench/2026-09-22-gguf-loading.md) and
[chat report](bench/2026-09-22-chat-template.md). Chat, normalization and tokenizer
oracles/benchmarks need development Python packages (tokenizers 0.22.2, Jinja2 3.1.6, regex
2026.9.10 and their dependencies, exactly the former `.tools/tokenizer-oracle-venv`, which
produced the goldens): they are hash-locked in `requirements_lock.txt` (from
`requirements.in`; relock with `bazel run //:requirements.update`), and every script runs
under the pinned interpreter with them through `tools/py SCRIPT ARGS` (`bazel run //tools:py`).
Scripts refuse the host's Python.

No system package changes. Full installed versions/hashes are in fixture/benchmark
manifests. Unicode-9 NFC has its own completed verification and benchmark loop:

```sh
tools/py bench/run_nfc.py \
  --cpu 2 --output docs/bench/data/YYYY-MM-DD-nfc-unique
tools/py tests/reference/generate_nfc.py \
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
tools/py bench/run_split.py \
  --cpu 2 --output docs/bench/data/YYYY-MM-DD-split-unique
```

Its [report](bench/2026-09-22-tokenizer-split.md) contains the exact independent
fixture-regeneration command, two actual runs and the llama-server comparison
boundary. The subsequent complete tokenizer is also verified and measured:

```sh
bazel build --config=release //bench:zerv-tokenizer-bench
bazel-bin/bench/zerv-tokenizer-bench models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  tests/fixtures/tokenizer/manifest.json
```

See the [complete tokenizer report](bench/2026-09-22-tokenizer.md) for fixture
regeneration and two benchmark runs including actual llama-server launch/config
records. Raw decoding deliberately does not perform per-piece UTF-8 replacement.

Regenerate fixtures explicitly into new paths:

```sh
tools/py tests/reference/gguf_oracle.py \
  fixtures --output third_party/gguf-candidate
tools/py tests/reference/generate_chat_goldens.py \
  --config third_party/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json \
  --publisher-template third_party/unsloth/Qwen3.8-27B-GGUF/4ca720788d1e01f1bff70c033e0d0028fd02e502/embedded-template.jinja \
  --output third_party/chat-candidate.json
# Tool-calling prompt oracle (tests/fixtures/chat-tools.json); --compare RAW.json diffs it
# against llama-server /apply-template prompts recorded by tools/check_tool_parity.py.
tools/py tests/reference/render_tools.py \
  --output third_party/chat-tools-candidate.json
```

Tool-calling parity with llama-server (one engine at a time; GPU):

```sh
tools/py tools/check_tool_parity.py --engines llama-fp32-full --output docs/bench/data/YYYY-MM-DD-tool-parity/llama
tools/py tools/check_tool_parity.py --engines zerv --zerv-binary PATH \
  --reference-raw docs/bench/data/YYYY-MM-DD-tool-parity/llama/raw.json --output docs/bench/data/YYYY-MM-DD-tool-parity/zerv
```

## Matched tokenizer comparison and optimization

The old direct-native/HTTP table is **not** relative tokenizer-speed evidence.
The direct reference harness checks the same installed libllama used by the server:

```sh
tools/py bench/run_tokenizer_matched.py \
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
tools/py bench/rebuild_tokenizer_baseline.py \
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
tools/py tests/reference/generate_q6_k_goldens.py \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --output third_party/NEW-q6_k-candidate.json
cmp tests/fixtures/q6_k.json third_party/NEW-q6_k-candidate.json
tools/py bench/run_model_quant.py --format q6_k \
  --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --cpu 10 --output docs/bench/data/NEW-q6_k-run
```

Use fresh paths. Q6_K measures 1/64/**4096-row slices** of output.weight, not the
entire 248320-row vocabulary projection. CPU diagnostic decoding is not GPU matvec
or a serving comparison. Earlier formats retain their existing row/tile/full-tensor
workloads and their own `generate_q4_1_goldens.py`/`generate_q5_k_goldens.py` tools.

## Explicit native GPU checks and repeatable driver benchmark

```sh
bazel test //tests:gpu //tests:gpu_release_fast
bazel build --config=release //bench:zerv-gpu-driver-bench
bazel-bin/bench/zerv-gpu-driver-bench
tools/py bench/run_gpu_driver.py --cpu 10 --output docs/bench/data/NEW-gpu-driver
```

Shader spill gate (required when shaders, compile defines or tuning tables change;
[RCA](bench/2026-09-24-aco-lds-spill.md)). Each command must report 0 FAIL (no VGPR
spills in LDS); WARN lines are scratch spills (performance items). The GPU tests' gate is a
Bazel test; the model checks need the model, which is not a Bazel input:

```sh
bazel test //tests:gpu_spills   # log: bazel-testlogs/tests/gpu_spills/test.outputs/shaderstats.txt
tools/py tools/check_shader_spills.py --log third_party/spill-gate/spec.txt -- "$(tools/zerv_build.py zerv-spec-check)" MODEL
tools/py tools/check_shader_spills.py --log third_party/spill-gate/mtp.txt -- "$(tools/zerv_build.py zerv-mtp-check)" MODEL third_party/mtp-check/tokens-short-nothink.json third_party/spill-gate/mtp-dump
tools/py tools/check_shader_spills.py --log third_party/spill-gate/f16.txt -- "$(tools/zerv_build.py zerv-model-profile)" MODEL 4096 512 600 2 f16 f16
```

GPU tests exercise actual transfers/compute and resource/state failures; default
`bazel test //...` still needs no GPU/driver. Diagnostic shader compilation occurs
only in the explicit external generator, not at runtime. The runner rebuilds both
native and independent C binaries; no old executable is required. CPU+GPU tests,
full output hashes and matching device/queue/allocation metadata gate all timings.
No tuned llama-server/inference claim follows from this raw driver benchmark.

[GPU validation, failures, fixture/binding regeneration commands](research/2026-09-22-gpu-driver-validation.md).
The Vulkan fixtures are generated by Bazel from the pinned headers (`@vulkan_headers`): the C
ABI (`abi.json`, `timing-abi.json`, compiled by the hermetic C toolchain) and the diagnostic
shader (`affine.spv`, source-built glslc); `//tests:vulkan_fixtures_test` (part of `//...`)
checks the committed files, `bazel run //tests:vulkan_fixtures_update` rewrites them. The
real-device fixture (`dispatch.json`) needs the GPU and runs the source-built C program
(`//tests:oracle_vulkan_driver`) on the test-only GPU runtime:

```sh
tools/py tests/reference/generate_vulkan_goldens.py --output third_party/NEW-dispatch.json --work third_party/NEW-dispatch-work
```

The driver benchmark (`run_gpu_driver.py`) runs both engines on the host's driver (what
production uses) and records it (`tools/host_info.py`: loader, ICD and driver hashes, CPU,
source-built vulkaninfo); it executes no host tool.

## Native machine code (`src/model/native/`)

The f16 `gemm_f16x` kernel also ships as our own RDNA3 machine code in a RADV pipeline binary
([spec](specs/prefill.md), [report](bench/2026-09-24-gemm-f16x-isa.md)). Ordinary builds only
embed the files. `//src/model:native_code` regenerates the kernel's assembly
(`bench/isa_lab/gen_f16x.py`), assembles it with the toolchain's Zig (`zig clang`; the object
is byte-identical to the host's clang 22.1.8) and splices the code into the committed
pipeline binary; `native_code_test` (part of `//...`) checks that the committed `.s` and
`.bin` are exactly that. The rest of the binary (the driver's config and keys) and the
bitwise sweep come from the GPU. A new binary therefore needs the GPU (the lab,
`//bench/isa_lab:pipeline_binary_lab`, and `zig clang` are built by Bazel), and must be followed
by the gates in the spec (`bazel test //tests:gpu`, `verify_model.py --gemm-code`, serving
outputs). One binary per driver build: `--runtime host` for the host's Mesa (production,
`src/model/native/`), `--runtime test` for the test-only GPU runtime built from source
(`src/model/native/test_radv/`, what the GPU tests run on):

```sh
tools/py tools/build_native_gemm.py --runtime host --output-dir third_party/NEW-native        # then review and copy
tools/py tools/build_native_gemm.py --runtime test --output-dir third_party/NEW-native-test   # -> src/model/native/test_radv/
```

A binary is valid only where the driver's global pipeline key equals its `.global` (same
Mesa build, GPU and compiler options); the model picks the one matching the device, elsewhere
the server logs the fallback and runs the SPIR-V. `--gemm-code spirv` selects the SPIR-V
explicitly.

## Resident GPU matvec — verified replay and measurements

```sh
# Ordinary checks use committed fixtures/SPIR-V (not ggml); //... also checks that the SPIR-V
# is what the pinned shader tools produce.
bazel test //... //tests:gpu //tests:gpu_release_fast

# Explicit independent replay: fresh paths, existing pinned model/tools/libraries.
tools/py tools/replay_matvec.py --output-dir .tools/NEW-matvec-replay --restore-sources

# Rebuild both competitors and benchmark all eleven complete dense shape/type pairs.
tools/py bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-run1
tools/py bench/run_gpu_matvec.py --cpu 10 --output docs/bench/data/NEW-matvec-repeat
```

The replay tool verifies 62 retained ggml research files, optionally fetching only
missing pinned small sources. It rejects modified existing files, compares freshly
regenerated independent fixture and eight SPIR-V modules byte-for-byte, and runs fmt,
Debug/ReleaseFast CPU+GPU and Python tests. No stale binary, package installation
or weight download is needed. Original replay actually executed and passed; see
[validation/logs](research/2026-09-22-gpu-matvec-validation.md).

To rebuild shaders alone into a review directory:
`tools/py tools/compile_matvec.py --output-dir .tools/NEW-matvec-shaders` (or `bazel build
//src/matvec:generated_shaders`). Only `bazel run //src/matvec:update_shaders` overwrites the
production shader files.
The benchmark separately records FP32-input and default (activation-quantized)
reference precision. [Measured losses and boundary caveats](bench/2026-09-22-gpu-matvec.md).
These are component commands, **not** a native Qwen server.

The [block08b DFS](research/matvec-optimization.md) adds a source-rebuilt scalar
counterfactual, correctness-gated candidate harness and separately labelled aligned
views. Full repeated protocol (fresh output directories; no concurrent GPU work):

```sh
tools/py bench/rebuild_matvec_baseline.py --output third_party/matvec-dfs/NEW-baseline
tools/py bench/run_gpu_matvec.py --cpu 10 --aligned \
  --baseline-run third_party/matvec-dfs/NEW-baseline \
  --output docs/bench/data/NEW-dfs-run1
tools/py bench/run_gpu_matvec.py --cpu 10 --aligned \
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
bazel build --config=release //src:zerv            # bazel-bin/src/zerv
bazel-bin/src/zerv --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf --port 8080 --context 8192
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hi"}]}'

# Oracle (explicit; needs the GPU and the model). The capture program is built from source
# (//tests:oracle_model: llama.cpp + ggml Vulkan) and runs on the test-only GPU runtime;
# templates render under the pinned interpreter. Fresh paths; the fixture replays except
# llama's stderr hash.
tools/py tests/reference/generate_model_oracle.py --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir third_party/model-oracle/NEW --output .tools/NEW-oracle.json

# The >512-token case (one full 512-row chunk; block 13h) is a separate fixture.
tools/py tests/reference/generate_model_oracle.py --model models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf \
  --work-dir third_party/model-oracle/NEW-long --output .tools/NEW-oracle-long.json --case-set long

# Native forward gates against the oracle work dirs (thresholds in docs/specs/model.md),
# on the test-only GPU runtime (--runtime host: the host's driver).
# The fixtures were regenerated on 2026-09-26 with the source-built oracle on the source-built
# runtime: llama's captures (tensors, logits) are byte-identical to the 2026-09-23 ones (host
# libllama, host Mesa); the FP64 reference files differ by <= 7e-13 (BLAS summation order: one
# OpenBLAS thread per worker process now). Older fixtures are kept in
# docs/bench/data/2026-09-23-gemm-efficiency/qwen38-oracle-2026-09-22-a.json.
tools/py tools/verify_model.py --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --modes 0,1,13,29,60,512 --work-dir third_party/model-native/NEW --report docs/bench/data/NEW/report.json
tools/py tools/verify_model.py --fixture tests/fixtures/model/qwen38-oracle-long.json \
  --oracle-dir third_party/model-oracle/2026-09-26-hermetic-long --modes 0,512,64 \
  --work-dir third_party/model-native/NEW-long --report docs/bench/data/NEW/report-long.json

# Rerun a gate with an archived capture binary (every work dir keeps its copy), e.g. to
# attribute a difference to a build: same flags plus --tool.
tools/py tools/verify_model.py --tool third_party/model-native/OLD/zerv-model-capture \
  --oracle-dir third_party/model-oracle/2026-09-26-hermetic --modes 0 \
  --work-dir third_party/model-native/NEW-rerun --report third_party/model-native/NEW-rerun.json

# Interleaved (ABBA) A/B of zerv-model-profile binaries: per-phase decode/prefill GPU time.
# An engine may carry its own arguments after "|", e.g. KV page sizes of one binary.
tools/py bench/race_profile.py --engine old=PATH --engine new=PATH --rounds 2 \
  --output docs/bench/data/NEW-race --args 32768 512 30000 32 f16@native f32
tools/py bench/race_profile.py --engine "p128=PATH|32768 512 30000 32 f16@native f32@page=128" \
  --engine "ctx=PATH|32768 512 30000 32 f16@native f32@page=context" --rounds 2 \
  --output docs/bench/data/NEW-pages --args 32768 512 30000 32 f16@native f32
# verify_model.py --kv-page-tokens N|context gates another KV page size (default 128).

# End-to-end greedy equality through the real server (JSON and SSE).
tools/py tools/check_session.py --output docs/bench/data/NEW/session.json

# Matched serving benchmark vs llama-server (engines run one at a time).
tools/py bench/run_serving.py --output docs/bench/data/NEW-serving
```

`verify_model.py` builds its native capture tool (`//tools:zerv-model-capture`) itself. The oracle's FP64 pass uses all CPU cores for ~25 minutes; do
not run GPU benchmarks concurrently with it.
