# Hermetic build — specification

Status: **in progress** (branch `bazel`). Done: phases 1–2 (build graph, container proof),
3a (scripts under the pinned Python). In progress: 3b (oracles from source). User requirement 2026-09-26: "everything must be 100%
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
  a native binary valid for the test driver is separate from the one for the host driver
  (phase 4 resolves how both are carried or which path each exercises).
- **Competitors** (user, 2026-09-26): pinned container image digests plus pinned source
  revisions count as hermetic (the tuned llama.cpp HIP build needs ROCm); llama-server's
  Vulkan build is built in the graph.
