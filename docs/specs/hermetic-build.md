# Hermetic build — specification

Status: **in progress** (branch `bazel`). Done: phases 1–2 (build graph, container proof),
3a (scripts under the pinned Python), 3b (oracle libraries and reference programs from
source; fixtures that need the GPU or the model not yet regenerated), 4 on the host (GPU tests
on the source-built runtime; the container run with `/dev/dri` is open). Open: GPU and model
fixtures, harness host tools, competitors (phase 5). User requirement 2026-09-26: "everything must be 100%
hermetic. go hardcore all in on this"; merge to main only once it all is.

## Definition

An action or test is hermetic when everything it executes or reads is a declared Bazel input
that is pinned by content: our sources, or an external archive fetched by URL with a sha256
(`MODULE.bazel`, `MODULE.bazel.lock`), or an output built from those inside the graph. No
action or test may execute or read a tool, library, header or data file of the host (`/usr`,
`/opt`, `$HOME`, the host `PATH`), and no result may depend on which host packages exist.

The execution platform (not an input) is: x86_64 Linux kernel, the Bazel binary of
`.bazelversion` (launched by bazelisk), and what Bazel's own test/action wrappers require (a
POSIX shell and coreutils: `/bin/bash`, `/usr/bin/env`). Python comes from rules_python's
pinned interpreter (python-build-standalone), which needs the platform's glibc at run time;
Zig-built executables are static or link only libc through Zig's bundled glibc stubs.

Hardware is not an input: GPU tests need `/dev/dri` and the kernel's amdgpu driver. Everything
above the kernel that they load at run time — the Vulkan loader, the ICD and the user-mode
driver (Mesa RADV) — must be built in the graph like any other input (phase 4), so that a GPU
test result depends on pinned driver sources, not on the host's Mesa package.

## Scope and phases

1. **Build graph (`bazel build //...`, `bazel test //...`).**
   - C/C++ for development tools: `hermetic_cc_toolchain` (clang via a pinned Zig SDK, bundled
     libc/libc++ headers); the host's C/C++ toolchain is never configured
     (`BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN=1`). No production target may depend on C/C++
     (AGENTS.md "No C++ dependencies"; a test checks `deps(//src:zerv)`).
   - Vulkan at link time: a stub `libvulkan.so.1` generated from `src/gpu/vk.zig`'s `extern`
     declarations (the pattern of Zig's glibc stubs); at run time the loader is resolved by
     the dynamic linker as before.
   - Shader tools: glslc (shaderc v2026.3), glslang (vulkan-sdk-1.4.357.0 = 168d452a),
     SPIRV-Tools (vulkan-sdk-1.4.357.0 = 9a49b088) and SPIRV-Headers (29981f65) built from
     source. These are the versions of the host tools that produced the committed modules;
     acceptance: every committed `.spv` is reproduced byte for byte.
   - Native kernel assembly: the toolchain Zig's clang (done).
2. **Proof.** `bazel test //...` passes in a container that has no compiler, no Python, no
   Vulkan, no shader tools (a slim distribution image plus the Bazel binary), with the
   repository cache and disk cache empty. Checked-in harness: `tools/hermetic_check.sh`.
3. **Tools.** Harnesses and reference programs run from the graph: Python scripts as
   `py_binary` with locked requirements (`bazel run //bench:…`), measured binaries as data
   deps in their ReleaseFast configuration, reference C programs as `cc_binary`, oracle
   libraries (ggml/llama.cpp at the pinned revisions) built from source, Vulkan headers from a
   pinned archive.
4. **GPU runtime.** Vulkan-Loader and Mesa RADV built from source at pinned revisions; GPU
   tests select them (`VK_ICD_FILENAMES`, loader path) and pass in the container with only
   `/dev/dri` passed through.
5. **Competitors** (llama-server builds, vLLM): pinned artifacts (image digests, source
   revisions built in the graph where feasible); recorded per benchmark.

## Acceptance

- Phase 1–2: `tools/hermetic_check.sh` passes (container, empty caches); the host glslc,
  spirv-val, clang, cc, python3 and `/usr/lib/libvulkan.so` are referenced by no BUILD or bzl
  file (`tests/test_build_lists.py`); the shader drift tests pass with the source-built tools.
- Phase 3: no harness invokes a host interpreter or compiler; `test_build_lists` enforces it.
- Phase 4: `//tests:gpu`, `//tests:gpu_release_fast`, `//tests:gpu_spills` pass in the
  container with only the device nodes from the host.

## Decisions

- **C/C++ in the build graph** (boundary note, 2026-09-26): C/C++ is compiled only for
  development tools and test-time runtime pieces (shader compilers, reference oracles,
  Vulkan loader and driver for tests). It is never linked into zerv; the check above
  enforces this. Approved by the user's instruction for a fully hermetic build.
- Generated artifacts (SPIR-V, native kernel code) stay committed with drift tests
  (docs/development.md, "Shaders").
- **GPU runtime is test-only** (user, 2026-09-26, option "1a"): the Vulkan loader and Mesa
  RADV built in the graph are what GPU tests and GPU oracle captures run on. zerv in
  production keeps using the host's driver (the OS/GPU driver interface boundary of
  AGENTS.md; no C++ ships). The native gemm_f16x binary is keyed to one exact driver build, so
  a native binary valid for the test driver is separate from the one for the host driver:
  both are carried (below, phase 4).
- **Competitors** (user, 2026-09-26): pinned container image digests plus pinned source
  revisions count as hermetic (the tuned llama.cpp HIP build needs ROCm); llama-server's
  Vulkan build is built in the graph.

## Phase 4 — test-only GPU runtime (implemented 2026-09-26)

**Pieces** (pinned in `MODULE.bazel`, built in the graph):

| piece | source | build |
| --- | --- | --- |
| Mesa RADV `libvulkan_radeon.so` | Mesa 26.2.3 (the host's vulkan-radeon version), archive sha256 1628058a… | Mesa's own meson build in one action: `meson_project` (`bazel/meson.bzl`, driver `bazel/meson_build.py`), options in `bazel/third_party/mesa.BUILD` (RADV only; no LLVM, no platform, no shader cache) |
| libdrm (static, inside the driver) | 2.4.133 (the version of Mesa's wrap), sha256 fc68f9d0… | meson subproject of the same action; `amdgpu.ids` exported |
| Vulkan loader `libvulkan.so.1` | Vulkan-Loader vulkan-sdk-1.4.357.0, sha256 54f2537d… | native BUILD (`bazel/third_party/vulkan_loader.BUILD`): no assembler trampolines, no X11/Wayland, search directories `/nonexistent` |
| meson, ninja | Bazel registry (meson 1.12.1, ninja 1.13.2) | |
| glslangValidator (Mesa's build compiles internal shaders) | glslang 16.4.0 (phase 1 pin) | `bazel/third_party/glslang.BUILD` |

`//tests:gpu_runtime` puts loader, driver, `amdgpu.ids` and an ICD manifest naming the driver
relative to itself into one directory. `GPU_ENV` (`tests/BUILD.bazel`) makes a test and its
children use it: `LD_LIBRARY_PATH`, `VK_DRIVER_FILES`, `VK_LOADER_LAYERS_DISABLE=~all~`,
`AMDGPU_ASIC_ID_TABLE_PATHS`, `ZERV_TEST_GPU_RUNTIME=test_radv`. Harnesses get the same with
absolute paths from `tools/zerv_build.py gpu_runtime()` (and `host_vulkan_env()` for runs that
must use the host driver: production benchmarks, the host's native binary).

**The meson action is hermetic.** Its source tree is a symlink tree of declared inputs plus
the subprojects (no wrap downloads); `PATH` holds only the named tools, the pinned Python,
ninja and the platform's `rm`/`tr`; the C/C++ compiler is the Bazel toolchain (Zig clang) via
a wrapper; `PYTHONHASHSEED=0`, `SOURCE_DATE_EPOCH=0`. Findings that needed a fix (negative
results kept):

1. meson asks the compiler for its library search directories (`--print-search-dirs`); Zig's
   clang answered with the host's, and meson found and linked **the host's
   `/usr/lib/libelf.so`**. The wrapper now answers with no directories.
2. Zig's compiler runtime lacks `__cpu_model` (`__builtin_cpu_supports`), used by addrlib's
   x86 AVX2 swizzle path (host image copies; compute never uses it): addrlib is built without
   SIMD on x86 (`bazel/third_party/mesa-addrlib-no-x86-simd.patch`).
3. Zig's linker rejects `-Wl,--fatal-warnings` and a separate `-Wl,--version-script PATH`
   argument pair; the wrapper drops the former and joins the latter. `ar` gets no thin-archive
   `T` flag (the wrapper strips it).
4. **Crash at exit:** the loader unloads the driver at `vkDestroyInstance`, and the
   statically linked C++ runtime's exit-time destructors then segfaulted the process after the
   last test. Linked with `-z nodelete`.
5. **Reproducibility:** two output bases gave drivers differing only in debug sections (the
   static C++ runtime's absolute build paths) and hence build ID, and with it RADV's
   pipeline key. Linked with `--strip-debug`: byte-identical across output bases
   (sha256 06234597…, build ID 850abf79…), so the committed native binary for it stays valid.
6. libdrm's meson file requires `nm` (for its symbol tests only): `//bazel:nm_unavailable`,
   a stub that fails if run.
7. Fetch-time host tools: `patch_cmds` (host `sed`) were replaced by patch files that Bazel's
   own patcher applies (Mesa, SPIRV-Tools); `test_build_lists` rejects `patch_cmds`,
   `patch_tool` and repository-rule `execute`. The refetched sources hit the action cache
   (identical files).

**Native `gemm_f16x` binary per driver build** (docs/specs/prefill.md, "One binary per driver
build"): `src/model/native/` for the host's Mesa (production), `src/model/native/test_radv/`
for this runtime, both captured and swept bitwise by `tools/build_native_gemm.py --runtime
host|test` with the lab built in the graph (`//bench/isa_lab:pipeline_binary_lab`) and
`zig clang`. The two differ only in `.global` (the driver key); code, binary key, assembly
and the 23-configuration sweep result are identical. Drift tests `//src/model:native_code_test`
and `native_code_test_radv_test` (negative control: an edited `.s` fails).

**Which driver ran is tested, not assumed.** With `ZERV_TEST_GPU_RUNTIME=test_radv` the
native-kernel GPU test requires the device's pipeline key to be the `test_radv` binary's (it
hashes the driver's build ID). Negative control: the ReleaseFast GPU test binary run against
the host driver with that variable fails (`WrongDriver`, 40 passed, 1 failed). This check
found that `//tests:gpu_release_fast` had run on the **host** driver: rules_zig's
`zig_configure_test` forwards no `RunEnvironmentInfo`, so `env` of `:gpu` was lost. It is now
wrapped by `env_test` (`bazel/defs.bzl`), which sets `GPU_ENV`.

**Result (host, 2026-09-26):** `//tests:gpu` 41/41, `//tests:gpu_release_fast` 41/41 (native
kernel run and bitwise equal, no skip), `//tests:gpu_spills` pass, all on the source-built
runtime; `bazel test //...` 65/65.
