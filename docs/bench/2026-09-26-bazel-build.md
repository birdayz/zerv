# Bazel as the only build tool — 2026-09-26

**Question.** Can Bazel replace `zig build` and every ad-hoc build step of the harnesses
(user decision 2026-09-26: "rework ALL tooling to Bazel only", exact caching, conventions of
`../fdb-go`), with the same checks, and at what cost in check latency?

**Answer.** Yes: the Zig code, the tests, every harness, the SPIR-V modules and the native
kernel's code; `build.zig` and `.zig-version` are gone (branch `bazel`). Generated files stay
committed and Bazel tests that they are exactly what the sources and pinned tools produce
(section "Generated files"). Only the driver capture of the native kernel (GPU) stays a manual
tool. Latency: the no-change and edit cases are about the same as `zig build`; the fully cold
build was slower, but that comparison is not matched (below).

| `bazel test //...` (59 tests: 19 Zig files × Debug/ReleaseFast, 20 Python, `zig fmt`) | time |
|---|---:|
| after `bazel clean`, no disk cache | 42.9 s |
| no change | 0.2 s |
| an edit in `src/session` (reruns its dependents; ReleaseFast LLVM compiles) | 21.5 s |
| reverting that edit (disk cache hit) | 1.0 s |
| `--config=quick` (Debug + Python), no change | 0.24 s |
| switching `-c fastbuild` ↔ `--config=release`, `bazel build //...` up to date | ~0.30 s (0.17 s without the switch) |

For comparison, `zig build check` (the same four required checks) measured 22.6 s cold and
~19–22 s after a `src/` edit ([test parallelism](2026-09-26-test-parallelism.md)). **Not a
matched comparison:** the Bazel runs were `nice`d with `--jobs=4` or unrestricted at
different times while another session's GPU serving benchmark (llama-server, zerv, vLLM) was
running, and "cold" differs (Bazel: no action cache at all; zig build: a fresh local cache
with Zig's global cache present). A matched cold run of both on an idle machine is open.

Setup: AMD Ryzen 9 3900X (12 cores, 24 threads), Bazel 9.2.0 via bazelisk, rules_zig 0.16.0
(Zig 0.16.0, the same executable as `.tools/`, sha256 `2317bbb9…`), rules_python 2.3.4
(Python 3.14), times from bash `date`/`time` around the commands.

## Commands

The table's first five rows were single runs during the migration; their shape (the exact
edit in `src/session` was not recorded, which makes that row not reproducible as is):

```sh
bazel clean && bazel test //... --disk_cache=      # cold
bazel test //...                                   # no change
# (an edit in src/session), then
bazel test //...
git checkout src/session && bazel test //...
bazel test --config=quick //...
```

The configuration switch (last row), exactly:

```sh
for c in "-c fastbuild" "--config=release" "-c fastbuild" "--config=release" "--config=release"; do
  s=$(date +%s%N); bazel build $c //... >/tmp/bz.log 2>&1; e=$(date +%s%N)
  echo "$c $(( (e-s)/1000000 )) ms; $(grep -c discarding /tmp/bz.log) discard"; done
# 300, 305, 313, 331 ms with "discarding analysis cache"; 170 ms without
```

## Findings

1. **Undeclared inputs.** The sandbox exposed inputs the Zig build had missed silently:
   `matvec` embeds `shaders/separate/*.spv`; `tests/gguf.zig` and `tests/model.zig` open
   `tests/fixtures/gguf/default.gguf` at run time (so `zig build`'s cached results would not
   have been invalidated by a change of that fixture); `matvec_gpu.zig` opens
   `src/matvec/shaders/f32_small.spv`. The Python tests' modules under test escaped the
   runfiles through `Path(__file__).resolve()` into the source tree; after switching to
   `absolute()`, six tests failed until their inputs were declared (`gguf_oracle.py`,
   `generate_vulkan_goldens.py`, the NFC/split/tokenizer golden binaries,
   `bench/workloads/long-v1.json`).
2. **glibc target.** rules_zig's default `x86_64-linux-gnu.2.17` cannot link the GPU test
   (`zig test` uses `--no-allow-shlib-undefined`; the host Vulkan loader references
   `dlopen@GLIBC_2.34`, `__isoc23_strtol@GLIBC_2.38`, …), and it links libm/libpthread/libdl
   separately. The registered target is `x86_64-linux-gnu.2.43` (Zig 0.16's newest; the host
   has 2.44); the ReleaseFast server then needs exactly `libvulkan.so.1`, `libc.so.6` and
   `ld-linux-x86-64.so.2`, as the `zig build` binaries did. CPU executables and tests are
   static.
3. **Driver as a test input.** The GPU tests declare the loader, RADV
   (`libvulkan_radeon.so`) and its ICD manifest, so a Mesa update reruns them.
4. **One output base.** fdb-go keeps per-configuration output bases because toggling its race
   and coverage configurations re-executes everything; here a toggle only discards the
   analysis cache (row above), so zerv uses one.
5. **Benchmark source snapshots contain BUILD files.** `docs/bench/data/*/source` copies of
   the tree would become packages of `//...`; `.bazelignore` excludes them (and `third_party/`,
   `models/`).

## Generated files

| | before | Bazel |
|---|---|---|
| SPIR-V, `src/matvec/shaders` (100 modules) | `tools/compile_matvec.py` by hand, serial, 33.1 s | action `//src/matvec:generated_shaders` (the script, 8 parallel compiles; 3.2 s standalone), test `generated_shaders_test` |
| SPIR-V, `src/model/shaders` (72 modules) | `tools/compile_model.py` by hand, 16.2 s | `//src/model:generated_shaders` (1.9 s standalone), test |
| native `gemm_f16x_q4_0` code | `tools/build_native_gemm.py` (GPU, system clang) | `//src/model:native_code`: generator → `zig clang` → splice into the committed binary, test `native_code_test`; the GPU capture stays manual |

- Both compile scripts, run in parallel, reproduce the committed modules and manifests byte
  for byte (`diff -r`), and so do the Bazel actions (the tests pass). The first Bazel build of
  both packages took 6.1 s critical path (`--jobs=4`, nice'd).
- `zig clang -target amdgcn-mesa-mesa3d -mcpu=gfx1100 -c -x assembler` (Zig 0.16.0) produces
  an object byte-identical to the host's clang 22.1.8 for the committed kernel assembly, so
  the machine-code step needs no system compiler.
- Negative controls: a flipped byte in a committed `.spv`, a comment appended to
  `src/model/gemm.comp` (changes the manifest's source hash) and an edited committed `.s` each
  fail their test with the file named; `bazel run //src/matvec:update_shaders` restored the
  flipped module.
- The shader tools are the host's packages, declared as inputs and sha256-pinned, not
  hermetic downloads (the LunarG SDK 1.4.357.0 tarball, 330 MB, is the hermetic option; its
  output identity is untested).

## Not done / limitations

- The GPU targets (`//tests:gpu`, `//tests:gpu_release_fast`, `//tests:gpu_spills`) build but
  were not run under Bazel yet: the GPU was busy with another session's benchmarks.
- No harness was re-run end to end after the migration (each needs the GPU or a reference
  library and minutes to hours); their Bazel build paths and provenance were exercised
  (`tools/zerv_build.py`, `tools/test_profile.py`).
- The native kernel's driver capture and bitwise sweep (`tools/build_native_gemm.py`, the C
  lab against research Vulkan headers in `third_party/`) need the GPU and stay manual.
