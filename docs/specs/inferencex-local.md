# Local InferenceX fixed-sequence comparison

**Current result authority:** the [unchanged-upstream processing requirement](#authoritative-upstream-result-processing-user-requirement)
below supersedes earlier custom-metric reporting instructions. Use upstream processor
outputs without corrections; observer calculations are separate historical diagnostics.

2026-09-29. Active user goal replaces the paused D.2 optimization goal. Results remain
LOCAL: no git push, PR, upload, dashboard submission or external publication without
explicit approval. User requires security review before execution/installation.

## Plan / acceptance

1. Pin official InferenceX source (f437f7bfd164422036b0de7e3818f8afb5bc70d7), inspect
   the fixed-sequence client and complete import/dependency closure. Fetch data/source
   only; no upstream install scripts or setup.py. Hash-lock wheel-only dependencies,
   scan source/wheels with existing pinned ClamAV, inspect network/subprocess/dynamic
   code surfaces and archive paths. Record scanner age, skipped files and limitations;
   a clean scan is not a guarantee. No remote tokenizer code, no weight download.
2. Integrate the unchanged upstream client through Bazel/pinned Python as a development
   tool only. No external inference/math library becomes a native runtime dependency.
   Offline/local tokenizer, no credentials in client env, loopback-only destination;
   disable network/model downloads and telemetry during execution. No upstream CI.
3. Implement and test native ignore_eos compatibility as below, then HTTP differential
   forced-length tests against the pinned tuned llama servers before timing.
4. Run upstream random fixed-sequence generation and measurement using openai-chat,
   fixed seed, lengths/concurrency,2*C warmups and fixed output length. Same GPU/GGUF,
   f16KV, non-MTP, context/slot count and documented memory caps. Zerv tiering enabled;
   reference cache RAM explicit. No claims of identical cache representation or a disk
   speedup unless actual I/O supports it. No concurrency with other GPU jobs.
5. Publish only LOCAL raw client artifacts, server logs, configs/hashes, length/error
   gates and repeated measurements with variance. Timings need request success and
   matching actual input/output lengths; distinguish raw random tokens from rendered
   chat length. Preserve failures; no postselection of favorable runs. State scope and
   adaptations. Standard fixed-sequence does not test multi-turn tier reuse (AgentX
   would be separate). No claim of official dashboard equivalence.

## ignore_eos contract (before implementation)

Primary reference: pinned llama common/common.cpp collects all EOG IDs into -inf biases;
server-schema.cpp ignore_eos activates them. This suppresses selection, NOT merely
ignoring a stop after sampling EOS. Upstream openai-chat client sends this Boolean
and max_completion_tokens. Native profile's EOG IDs are im_end and endoftext.

Add optional Boolean API ignore_eos, defaultfalse, strict type validation. Carry through
Native.generate→session.Request. When true, construct a bounded per-request ascending
allowed-ID list excluding all special.eos IDs and use existing sampler.sampleFrom;
no logits mutation, per-token allocation, precision change, or default-path change.
Validate IDs and reject an empty allowed set. Explicit text stops, context cap, client
cancellation still apply. Tool mode + ignore_eos is explicitly unsupported initially;
reject before inference rather than bypass tool-stop semantics. Existing speculative
sample-matching still samples through the same filtered sampler (no draft logits change).

Executable checks: independent exhaustive tiny greedy logit fixtures with multiple EOG
IDs and ties; public generation tests with EOS-highest logits, length/context/stops,
false/default behavior and invalid all-EOS sets; API bool/default/rejection; baseline
CPU gates Debug/ReleaseFast. Differential HTTP: same short task whose ordinary output
ends early, both native and llama with ignore_eos produce exactly requested count and
length finish at several limits. Do not compare natural-language hashes as a numerical
oracle. Existing sampleFrom tests provide sampler-chain coverage; this feature only
constructs the static exclusion set. Performance measurement includes its real costs.

## Measurement audit discovered before execution

The pinned upstream openai-chat reader starts TTFT on ANY choices chunk, including
role-only, and counts finish/role chunks in ITL. Zerv emits role before generation.
Therefore preserve upstream raw TTFT/ITL but do NOT present them as first-token latency.
Use an identical loopback streaming observer for both engines: forward request/SSE
bytes unchanged, record role arrival separately from first nonempty content/reasoning,
all nonempty text-event times, usage and finish/DONE. Supplementary TTFT includes
observer forwarding overhead; not used to rewrite upstream results. Throughput/E2E
remain upstream. Observe prompt/output counts and reject incomplete responses, errors,
wrong output length or missing length termination even if upstream marks success.
This is a measurement adapter, not a substitute workload/load generator.

Initial local matrix: upstream random fixed-length input1024/8192, output256,
concurrency1/4,8 measured prompts per point,2*C upstream warmups, seed42, three
alternating engine-order rounds. These are explicitly local workload settings, NOT
a full dashboard/model/hardware reproduction. Parallel2/context12288 per slot,
native192pages of128 tokens, f16KV,8 snapshots; native host swap8192MiB, disk8192MiB;
llama cacheRAM8192MiB/checkpoints8. Different recurrent snapshot accounting disclosed;
common ceiling24GiB device and32GiB host RSS, same24576 live GPU token positions.
Keep join default0 to benchmark tiering without that extra policy change. Reference
Vulkan b512 and HIP nofusion b2048, both ub512. Source pin/default thinking behavior
unchanged: no double application of chat template. Capture actual request/rendered
prompt IDs and report chat overhead rather than pretending raw input length is total.

## Execution lifetime and interrupted runs

The first full matrix exceeded the caller's 3600-second deadline after 29/36
points; it did not hit a per-point timeout. Full-matrix agent tool invocations must
set **timeout_ms=10800000 (three hours)**, not the default or one-hour deadline.
This is the outer shell/tool setting, not a CLI flag; the runner has no whole-run
one-hour alarm. Keep bounded per-client (1200s), startup (900s), and observer socket
(600s) limits so a stalled point still fails. Larger matrices may need a larger
explicit outer budget. Do not detach jobs or remove stall limits.

`--resume-from OLD_DIRECTORY` writes a new output directory, verifies binaries,
client/observer/source pins and byte-identical workload/token captures, and inherits
only passed points. Never count interrupted output. Preserve the original manifest
and logs unchanged; identify the interruption in a sidecar note and final report.
The report must verify every expected matrix tuple exactly once and recheck raw
response/count/resource gates, then report arithmetic means and sample standard
deviations across three trials (not across pooled requests). Tail summaries from
only eight requests per trial are descriptive, not robust service SLO estimates.

## Authoritative upstream result processing (user requirement)

Use the pinned upstream fixed-sequence result-processing layer **byte-for-byte
unchanged**. No local formula implementation, monkey patch, metric substitution,
TTFT correction, observer input, or output rewriting. Fetch its source/import closure
at the same InferenceX revision, review/scan before execution, build through Bazel.
The local adapter may only supply actual deployment metadata/environment and paths,
copy original client artifacts byte-for-byte into fresh processing directories, and
invoke upstream `infx.results.fixed_sequence` and `infx.results.collect_results`.
Use the full processor CLI including its own power-validation/audit behavior—not
just a locally copied formula or a call that bypasses processing stages.

Existing runs lack power telemetry: keep upstream `power_valid=false` and its reasons;
never manufacture power measurements or substitute VRAM samples. Power is optional
for this existing fixed-sequence throughput comparison. Preserve all source/client
artifacts, output/audit/log bytes and record hashes, metadata, commands and exit codes.
The upstream collector receives only processor result files, not arbitrary JSON sidecars.
No official publication/submission workflow runs. No inference/GPU rerun is necessary.

Acceptance: Bazel source-download hash checks; wrapper tests through the full CLI,
including missing required metadata and missing-power behavior; complete36-point
processing with input SHA equality before/after and exact emitted JSON preservation
through the collector. No postprocessor execution from research checkout. Previously
adjusted/custom summaries remain historical diagnostics only and are superseded as
InferenceX results. Present native upstream metrics even when unfavorable; discuss
known measurement limitations separately, never repair the metrics.

## Authorized rerun: native tiered versus untiered

User explicitly requested commit/push and a rerun, clarified **only ours**, then
requested tiered versus untiered. No llama-server execution in this rerun. Existing
reference measurements stay historical, not contemporaneous controls.

Freeze two existing configurations: same model/f16KV, two slots,192×128 live GPU
positions,8 prefix checkpoints, same prefill precision and no speculation/join policy.
Tiered retains host8192MiB+disk8192MiB/16entries,8MiB chunks,prefetch2. Untiered sets
`kv-swap-mib=0,prefix-cache-tier=off` and no disk directory: GPU-resident prefix cache
still enabled, evicted entries dropped instead of tiered. Recurrent/snapshot bookkeeping
is still part of the engine; this is not a claim of zero host memory. Host/disk budget
is the intentional treatment difference, not a hidden equal-memory comparison.

Same fixed upstream matrix:1024/8192 nominal input,256 output,8 measured requests,
C1/C4,2*C warmups,3 alternating configuration-order rounds (24points total), new
servers per point. All normal count/prompt/resource gates stay enabled. Read actual
host/disk counters; random fixed-sequence reuse may not exercise disk restores. Check
native output signatures across treatments and rounds; disclose any mismatch before
claiming quality-equivalent speedup. Check role arrival coincides with first nonempty
text in the retained observer traces; no observer-derived metric replaces upstream.
Use `--engines zerv-tiered,zerv-untiered` and the full unchanged upstream processor /
collector. Pass upstream RECIPE_FINGERPRINT as SHA256 of the actual command/environment
to distinguish configurations without changing its result schema or metrics.

Outer tool timeout10800000ms; do not resume old-binary trials into this experiment.
Original historical artifacts are immutable. User's explicit git-push authorization
covers committing/pushing the current work; no InferenceX/dashboard submission.
