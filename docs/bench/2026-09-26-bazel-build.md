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

## GPU targets

Run once the GPU was idle (VRAM 0.9 GB used, no other GPU process), in Bazel's linux-sandbox
(the device nodes are reachable; no sandbox flag needed):

| target | result | wall |
|---|---|---:|
| `//tests:gpu` (Debug) | 41/41 passed | 169 s incl. build |
| `//tests:gpu_release_fast` | 41/41 passed | 716 s for both rows (serial: `exclusive`) |
| `//tests:gpu_spills` | exit 0; 3040 pipelines; 0 FAIL (VGPR spills in LDS), 0 WARN (scratch), 45 SGPR-spilling | (above) |

The spill gate compiles every pipeline with the shader cache disabled, most of the 716 s.

## Not done / limitations

- No harness was re-run end to end after the migration (each needs the GPU or a reference
  library and minutes to hours); their Bazel build paths and provenance were exercised
  (`tools/zerv_build.py`, `tools/test_profile.py`).
- The native kernel's driver capture and bitwise sweep (`tools/build_native_gemm.py`, the C
  lab against research Vulkan headers in `third_party/`) need the GPU and stay manual.

## Hermetic phases 3–5: runtime, fixtures, harnesses, competitors (2026-09-26, evening)

Question: do the source-built oracles and the source-built GPU runtime reproduce the
committed fixtures and gates, and does the in-graph llama-server perform like the host's?
Spec and details: [hermetic-build.md](../specs/hermetic-build.md) (phases 3–5). Host: Ryzen 9
3900X, RX 7900 XTX, Mesa 26.2.3 (Arch) on the host and in the graph; model
`Qwen3.8-27B-Q4_0.gguf` (sha256 ede16c7b…).

**Byte-identity results** (committed fixtures regenerated with source-built programs on the
source-built runtime):

| artifact | result |
| --- | --- |
| Vulkan C ABI, timestamp ABI (`abi.json`, `timing-abi.json`) | same 64 structures / 86 constants and the same generated C source from headers 1.4.357.0 (was the 1.4.354 research copy); only header hashes changed |
| `affine.spv`, `src/gpu/vk.zig` | byte-identical (vk.zig: the header comment names the new registry release) |
| `dispatch.json` (real device) | 10 cases and device/allocation metadata identical |
| `gpu/matvec.json` (ggml-vulkan) | 48 cases bitwise identical to the host driver's |
| `chat-template.json` | 100 renderings identical (Jinja2/MarkupSafe files byte-identical to the old venv) |
| model oracle, default and long sets | llama.cpp's captured tensors and logits byte-identical to the host package on the host driver; FP64 reference within 7e-13 (top-1 logits) |
| native `gemm_f16x` for the test runtime | `.bin`/`.key`/`.s` identical to the host's; only `.global` (driver build) differs; bitwise sweep PASS (23 configurations) |

**Gates on the test runtime** (`verify_model.py`, oracle 2026-09-26-hermetic):

- FP32, modes 0, 1, 13, 29, 60, 512: every tensor within its bound (worst/bound ≤ 0.552),
  greedy tokens all equal, 0 failures.
- zerv on the host driver vs the source-built driver (FP32 modes 0 and 512; f16/native
  mode 512): tensors, logits and serving logits bitwise identical.
- f16 prefill, `--gemm-code native` (the `test_radv` binary was selected: `gemm_f16x: native`)
  vs `spirv`, modes 13 and 512: bitwise identical. (f16 exceeds the FP32 bounds on the long
  case at 512 rows, worst/bound 318.5, on both drivers; f16 is not gated by FP32 bounds.)
- `//tests:gpu` and `//tests:gpu_release_fast` 41/41, `//tests:gpu_spills` pass; the native
  GPU test requires the device key of the source-built driver (negative control: the host
  driver fails with `WrongDriver`). `bazel test //...` 67/67.

**Failures and negative results on the way** (kept in the spec): Mesa's meson linking the host
`libelf` through Zig's search-dir answer; a segfault at exit without `-z nodelete`; build IDs
differing by output base until `--strip-debug`; `gpu_release_fast` silently on the host driver
(`zig_configure_test` drops `env`); the FP64 reference oversubscribed with 22 × 24 OpenBLAS
threads (load average 394) under the pinned numpy.

### llama-server built in the graph vs the host package (A/B)

Question: may the in-graph llama-server (`//bench:llama-server`) replace `/usr/bin/llama-server`
(Arch llama.cpp-vulkan, same commit b29c606e, GCC 16) as the serving competitor without making
it slower? Engine `llama-fa-ub512` of `bench/run_serving.py` (workload `serving-v2.json`, 4
cases × 3 repeats, greedy), one engine at a time; raw data
[data/2026-09-26-llama-server-ab/](data/2026-09-26-llama-server-ab/) (per run: `summary.json`,
`raw.jsonl`, `manifest.json` with the binary's hash).

```sh
M=models/qwen3.8-27b/Qwen3.8-27B-Q4_0.gguf; D=docs/bench/data/2026-09-26-llama-server-ab
tools/py bench/run_serving.py --model $M --engines llama-fa-ub512 --output $D/native-1              # in-graph
tools/py bench/run_serving.py --model $M --engines llama-fa-ub512 --llama-server /usr/bin/llama-server --output $D/host-7
```

Median decode tok/s and TTFT per case; ratio = in-graph / host, means of the warm runs:

| build (runs) | short decode | think decode | medium decode | long decode | TTFT ratios (short, think, medium, long) |
| --- | ---: | ---: | ---: | ---: | --- |
| in-graph, generic x86-64 (medians over graph-2..4 / host-2..4) | 0.984 | 0.988 | 0.987 | 1.017 | 1.079, 1.018, 1.008, 0.999 |
| in-graph, `-march=native` experiment (march-1,2 vs host-5,6) | 0.993 | 1.004 | 0.998 | 0.991 | (host-5 TTFT outliers) |
| **`//bench:llama-server`** (`-march=native`; native-1,2 vs host-7,8, ABBA) | **0.997** | **0.999** | **0.994** | **0.993** | 0.996, 0.985, 1.002, 1.002 |

- Outputs (text hashes) identical in all 13 runs of both builds.
- The first run after an idle period is slow for either build (graph-1, host-3, host-5: TTFT
  up to +100%); later runs are preceded by a discarded warm-up run.
- The generic build decoded 1.0% slower on average in the warm adjacent pairs (graph-2/host-2,
  host-4/graph-4: −1.7%…+1.6% per case). Compiled for the host CPU (`native_cpu_binary`, bazel/defs.bzl: a configuration
  transition adding `-march=native` to the competitor's subtree only; the oracles keep their
  build), the difference is −0.4% on average (−0.7%…−0.1%), smaller than the spread between
  two runs of the same binary (up to 1.8%).
- Verdict: `//bench:llama-server` replaces the host package as the competitor. A residual
  decode difference below ~1% is not resolved at this sample size; serving claims within 1%
  of llama-server remain undecidable either way.
