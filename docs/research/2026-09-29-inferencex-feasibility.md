# InferenceX / AgentX local comparison feasibility

2026-09-29; user asked whether we can run https://inferencex.semianalysis.com/.
Feasibility research only: no benchmark, package installation, dataset/weight download,
or autonomous resumption of the paused performance goal.

Primary source: https://github.com/SemiAnalysisAI/InferenceX at
f437f7bfd164422036b0de7e3818f8afb5bc70d7 (HEAD observed by git ls-remote).
Downloaded using curl -fLsS from raw.githubusercontent.com at that exact revision:

- `README.md`, SHA2567370227cb2c54a056835732a2495108cec21abf14ff48307fab7e1f38eb6c177
- `inferencex-e2e/docs/agentx-standalone.md`,
  SHA256907680b85a22bcc86a270d741d330009ce8dcf1142ff5c03bfdfb801a3b95dd6

Local research-only root:
`third_party/research-serving/InferenceX-f437f7bfd164422036b0de7e3818f8afb5bc70d7/`.
No runtime dependency. Source documentation, not verified local execution.

Observed upstream contract:
- Website is a dashboard; benchmark source is InferenceX. AgentX has a standalone
  client for an existing OpenAI-compatible `/v1/chat/completions` streaming server.
- Standalone document pins https://github.com/SemiAnalysisAI/agentx-harness to
  754356e9a39acc6cc6afb242d123bb57c3fb6f75, CLI `aiperf profile`, scenario
  `inferencex-agentx-mvp`.
- Recipe uses public dataset `semianalysis_cc_traces_weka_062126`,393 entries,
  recorded assistant responses (`AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0`),
  one-hour measured duration plus preparation/warmup/drain, concurrency8 example,
  server token counts and no GPU telemetry. No dataset contents examined yet.
- Hardware support table lists data-center devices and RTX PRO6000 Server, not
  RX7900XTX. README requires unofficial runs/forks explicitly labelled unofficial.
- Upstream installation uses uv/venv/Python3.11 and datasets>=4.7.0. This is NOT our
  project's pinned Bazel/tooling recipe and must be adapted/reviewed before use.

Feasibility judgment: protocol matches zerv, so a local unofficial AgentX comparison
is plausible. Not yet verified: client dependency closure, tokenizer choice versus
our GGUF, request fields/tool schemas and zerv support, trace lengths versus configured
context, error semantics, and dataset access/size. Do not enable trust_remote_code
blindly just because the example does. Inspect before installation/execution.

Suggested bounded next task if requested: pin/review client and dependencies, build
with the project's tooling, inspect trace lengths and request schema, run a small
compatibility smoke test, then identical trace/concurrency sweeps on zerv with tiering,
tuned llama-server and compatible vLLM. Match hardware, model/precision where possible,
resource ceilings, context and warm/cold cache; record mismatches and reject silently
truncated/out-of-context/empty responses. Report TTFT/ITL distributions, completed
trajectories, throughput, memory and tier-I/O. Locally scaled traces/models are adapted
unofficial runs, not direct comparisons to the dashboard's multi-GPU configurations.

## Standard fixed-sequence client follow-up

User specifically asked for standard InferenceX vs llama-server. Inspected the SAME
revision's fixed-sequence launcher, not just AgentX. `benchmarks/benchmark_lib.sh`
constructs `python3 -m infx.bench_serving.benchmark_serving` with random dataset,
explicit input/output lengths, concurrency, infinite request rate,2×concurrency
warmups, `--ignore-eos`, and TTFT/TPOT/ITL/end-to-end percentiles. The request module
has an `openai-chat` backend posting streaming `/v1/chat/completions` with
`max_completion_tokens`, and sends `ignore_eos:true` when requested. Native source
currently has no ignore_eos handling. Thus protocol compatibility alone is not enough
to claim an unmodified fixed-output benchmark works; verify forced-length semantics
on BOTH servers before measuring. Disabling the flag and allowing early stops is not
the same fixed-output benchmark. Fixed-sequence tests also do not substitute for
multi-turn tier-hit/miss tests; use AgentX separately for that question.

Additional downloaded files under the same pinned local root, SHA256:
- inferencex-e2e/README.md: afeed8d20ff495a711d44dbec569181b2dcfec02d3b663f624bbf7469efa1a69
- inferencex-e2e/docs/index.md: a1012a0239a0e648ba741e8f2d9c5880457d2f60c9ed5d666b6aa89b0baefa4e
- inferencex-e2e/docs/architecture.md: 68fe2166bccd3f9c3ad88f14292d07fe567c915e53277b448dfd39d0267308cc
- inferencex-e2e/benchmarks/benchmark_lib.sh: 23360f930517ce3b7c913bc693beeb914ada06126c70d7e14192bc09f9d6e768
- inferencex-e2e/infx/bench_serving/backend_request_func.py: b8c410db4036eac4f3b2bae1d08d32b9627acae70a3d83c2acbff0df642a4a3f

Fetched with curl -fLsS from raw.githubusercontent.com at the pinned revision;
installation/execution still not attempted. This is feasibility work, not resumption
of the paused full optimization goal.

## Execution follow-up

The earlier feasibility-only status above is historical. The actual pinned standard
fixed-sequence client was integrated through Bazel after source/wheel security review.
Native ignore_eos compatibility and the three-engine HTTP/smoke gates pass; the full
36-point matrix is now complete locally. See
[measured report](../bench/2026-09-29-inferencex-local.md) and
[security limitations](2026-09-29-inferencex-security.md).
AgentX has not been run; no results published or pushed.
