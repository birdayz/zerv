# Hermetic build — specification

Status: **in progress** (branch `bazel`). Done: phases 1–2 (build graph, container proof),
3a (scripts under the pinned Python), 3b (oracle libraries and reference programs from
source), 4 (GPU tests on the source-built runtime, also in the container with only `/dev/dri`),
3 (fixtures regenerated, harnesses without host programs), 5 in part (llama-server built in the
graph, the HIP competitor built by its recipe byte-identically). Open: merge to main. User requirement 2026-09-26: "everything must be 100%
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

**Host-driver GPU tests (specified 2026-09-26).** The hermetic GPU tests no longer exercise the
driver production runs on, so a host Mesa update would go unnoticed. `//tests:gpu_host` and
`//tests:gpu_host_release_fast` run the same test binaries (`:gpu`, the ReleaseFast one) on
the host's installed Vulkan stack (`ZERV_TEST_GPU_RUNTIME=host`: the device's pipeline key
must be the one of `src/model/native/`, else the test fails and names the recapture command).
They are deliberately not hermetic, so they are not part of `//...` and their inputs include
the host driver's identity: `tools/zerv_build.py` passes `--test_env=ZERV_HOST_VULKAN_ID=<sha256
of tools/host_info.py host_vulkan()>` (loader, ICD manifests, driver libraries), which makes
the result cacheable exactly as long as the driver is unchanged; the test recomputes the
identity and fails when it is missing or differs (a stale id cannot yield a pass). Production
benchmarks run them first (`zerv_build.test_host_gpu()`): the GPU component benchmarks
(`run_gpu_driver`, `run_gpu_matvec`, `replay_matvec`) and the serving benchmarks
(`run_serving`, `run_multiuser`, `run_concurrent`, `run_long_context`). Diagnostics and
profilers are not gated.

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

**Acceptance (container, 2026-09-26):** `tools/hermetic_check.sh --gpu //... //tests:gpu
//tests:gpu_release_fast //tests:gpu_spills` at 448b26f: debian:trixie-slim (pinned digest)
with none of cc, gcc, clang, python3, glslc, spirv-val, zig and no libvulkan; empty repository,
disk and output caches; only `/dev/dri` from the host. 70/70 tests passed (the GPU tests with
their driver-identity check), 2352 actions, 25.8 min wall.

## Phase 3 — tools and fixtures without host programs (2026-09-26)

**Rule.** Harnesses and generators execute only the pinned interpreter (`tools/py`), programs
built in the graph, and the execution platform (glibc's dynamic linker, a POSIX shell). What a
benchmark records about the host it reads from the kernel and from files, never by running a
host program. `test_build_lists` enforces the host-Python guard on every script of `bench/`,
`tools/` and `tests/reference/`, and that none names a host tool as the program of a command
(or uses `shutil.which`, `shell=True`, `os.system`); negative control: an added
`["git", "status"]` fails it.

| host program | replacement |
| --- | --- |
| `cc` (reference C programs) | `cc_binary` oracles: `oracle_vulkan_driver` (BoringSSL SHA-256), `oracle_model`, `oracle_llama_batch_capture` (llama.cpp with the Vulkan backend: `@llama_cpp//:llama_vulkan`, statically registered, so no backend is loaded from the file system), plus 3b's |
| `/usr/lib` libllama, libggml, headers, hash pins | the source-built libraries (`gguf_oracle.library()`, `zerv_build.oracle()`) |
| `ldd` | `host_info.loaded_libraries()`: `ld.so --list` (the platform's dynamic linker; the program is not run) |
| `readelf -d` | `host_info.elf_dynamic()` (ELF parser; also used by `test_linkage`) |
| `lscpu` | `host_info.cpu()` (`/proc/cpuinfo`, sysfs) |
| `vulkaninfo` | `@vulkan_tools//:vulkaninfo` (Vulkan-Tools 1.4.357.0, no WSI; dlopens the loader of the environment) |
| `pacman -Q mesa` | `host_info.host_vulkan()`: loader, ICD manifests and driver libraries with hashes |
| `pgrep -a -f` | `host_info.processes()` (`/proc/*/cmdline`) |
| `taskset` | CPU affinity set in the child (`os.sched_setaffinity`) |
| `git status/rev-parse` | `zerv_build.source_revision()` (reads `.git`; manifests hash the sources) |
| `curl` | `fetch_hf.download()` (urllib, resume, retries) |
| host `clang` (isa_tool) | `ZERV_CLANG` (required) = the toolchain's `zig clang` |
| `.tools/tokenizer-oracle-venv` | the pinned interpreter (Jinja2 3.1.6 / MarkupSafe 3.0.3 files byte-identical) |

**Fixtures regenerated** (the data identical unless noted):

- Vulkan C ABI (`abi.json`, `timing-abi.json`) and the diagnostic shader (`affine.spv`) are now
  Bazel actions with a drift test (`//tests:vulkan_fixtures_test`): headers from
  `@vulkan_headers` (1.4.357.0 instead of the research copy of 1.4.354; the 64 scoped
  structures, 86 constants and the generated C source are identical, only header hashes
  changed), compiled by the hermetic toolchain; `affine.spv` byte-identical.
- `dispatch.json` (real device): the source-built C program on the test runtime; cases and
  device/allocation metadata identical.
- `gpu/matvec.json`: ggml-vulkan on the test runtime; all 48 cases bitwise identical to the
  host driver's.
- `chat-template.json`: all 100 renderings identical under the pinned interpreter.
- Model oracle (`qwen38-oracle*.json`): the source-built llama.cpp on the test runtime. Its
  captured tensors and logits are **byte-identical** to the host package's on the host driver;
  the FP64 reference changed by <= 7e-13 (top-1 logits; summation order, below).

**Negative result (performance of a generator):** the FP64 reference forks 22 worker
processes; under the pinned numpy wheel each started 24 OpenBLAS threads (528 runnable
threads on 24 CPUs, load average 394). The reference now sets one BLAS thread per process
before numpy loads (`qwen35_reference.py`, `generate_model_oracle.py`, `mtp_reference.py`).
The whole default-set run (captures and FP64) took 7.7 min (run-manifest 19:15:27–19:23:10
UTC) against 31.0 min for the 2026-09-23 run (host numpy); the aborted oversubscribed run
had not finished the FP64 pass of the first case after 4 minutes.

**GPU correctness tools use the test runtime** (`verify_model.py --runtime test` default,
`kv_quality.py`, `prefill_quality.py`, the fixture generators); production benchmarks use the
host driver and record it.

## Phase 5 — competitors (2026-09-26)

- **llama-server (Vulkan)**: `@llama_cpp//:llama-server`, built in the graph from the pinned
  llama.cpp (b29c606e, the host package's commit) and ggml (456172ec) with upstream's Linux
  defaults, except no OpenSSL (HTTPS downloads) and no embedded web UI (npm); reports
  `version: 0.4.1 (build 10964, commit b29c606e28)` like the host binary. The harnesses use
  `//bench:llama-server`: the same build compiled for this CPU (`-march=native` through a
  configuration transition, `native_cpu_binary`; zerv's release build is `-mcpu=native` too),
  which serves identical outputs at the host package's speed within noise (decode −0.4% mean;
  the generic build was −1.0%): A/B in docs/bench/2026-09-26-bazel-build.md.
  `/usr/bin/llama-server` is referenced only as an explicit `--llama-server` comparison.
- **llama.cpp-RDNA3-7900xtx-opt (HIP)**: `tools/build_competitor_rdna3.py` builds it from the
  pinned source archive (commit 15995a12, sha256 952679df…) in the pinned vLLM ROCm image
  (digest), network off, as our uid, with the fork's README configuration plus what the image
  and the archive require (no ccache, no curl/OpenSSL; the build number and commit that git
  supplied, ggml's through a two-answer git stand-in). Result (2026-09-26, 4.5 min): the five
  binaries recorded for the build the competitor benchmarks used (llama-server, libggml,
  libggml-base, -cpu, -hip) are **byte-identical**; `run_serving.py --engines rdna3` serves
  from it (smoke run, hash recorded in the manifest). Negative results: CMake picked up the
  image's ccache (not writable as our uid); without the stand-in ggml embeds commit
  "unknown" (libggml-base then differs).
- **vLLM**: the pinned image digest and model revision (unchanged).
