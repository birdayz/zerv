# Standing project instructions

Read and follow [AGENTS.md](AGENTS.md) for the full project requirements.

## All builds go through Bazel

**Bazel is the only build system for this project.** This applies to production
code, tests, tools, benchmarks, external benchmark clients, reference programs,
and their dependencies—not just the zerv binary.

- Use the pinned toolchains and dependency graph in `MODULE.bazel` and
  `.bazelversion`; follow [docs/development.md](docs/development.md).
- Build from harnesses through `tools/zerv_build.py`.
- Run Python tools and harnesses with `tools/py SCRIPT` (the Bazel-pinned interpreter).
- Do not bypass Bazel with direct compiler invocations, `zig build`, standalone
  CMake/Make/Ninja builds, `pip install`, `uv sync`, or ad-hoc virtual environments.
  Upstream setup instructions are not an exception: integrate their build and
  dependencies into Bazel first. A compiler/build tool invoked by a declared Bazel
  action is permitted; launching an independent build outside the graph is not.
- During stabilization, run the smallest relevant Bazel test subset first (for
  InferenceX harness/resume/report changes: `bazel test //tests:test_serving_harness`).
  Do not rerun the entire suite or GPU benchmarks on every harness edit.
- Once stable, use `bazel test //...` for the required final checks, plus the
  GPU/host-driver gates specified in AGENTS.md when applicable.

## Long benchmark execution

For the full local InferenceX matrix, set the outer command/tool timeout to at least
**three hours (`timeout_ms=10800000`)**. The measured matrix exceeded a one-hour
caller deadline. Keep the harness's bounded startup/request/client stall timeouts;
do not confuse them with the whole-run budget or disable them. Preserve interrupted
artifacts and use the checked `--resume-from` path in `bench/run_inferencex.py` if
needed. Do not detach jobs to bypass the runtime's lifetime tracking.

## InferenceX result integrity

Use **100% of the pinned upstream result-processing path unchanged** for InferenceX
results. No patched processor, copied/reimplemented formulas, observer-derived metric
substitution, or corrections to its outputs. Supply only actual metadata and original
artifacts; preserve emitted results/audits exactly. Discuss caveats separately. Custom
adjusted metrics are not authoritative InferenceX results. No publication without approval.

## Honest streaming timing

Do not emit a standalone role-only or empty opening SSE event. Attach role to the
first actual generated payload (or terminal finish for an empty response). Never
present headers/role-only events as first generated tokens or optimize for that
measurement artifact. Keep upstream benchmark/client/result processing unchanged.
