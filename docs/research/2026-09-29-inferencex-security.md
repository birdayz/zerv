# InferenceX client security review (local benchmark only)

2026-09-29. Not a guarantee of absence of malware. No result publication authorized.
Upstream pin and inspected surfaces: `2026-09-29-inferencex-feasibility.md` and
`docs/bench/data/2026-09-29-inferencex-security/source-surfaces.txt`.

Only six unchanged fixed-sequence client modules are used; hashes and original raw
GitHub URLs in `bazel/inferencex.bzl`. No upstream CI, uploader, installer, fleet runner,
AgentX, server_watch subprocess wrapper, or repository-level agent instructions run.
Reviewed request construction, tokenizer loading, random generation, measurements,
output writes and dependency imports. HTTP uses aiohttp (trust_env=True upstream),
which makes removing inherited proxy/credential configuration important. Optional Hub
snapshot downloads and remote tokenizer code must not run. Dynamic result filenames
are confined by our own runner's output choice. Multimodal/DeepSeek encoding is not
selected. Upstream's5% request-failure allowance is NOT our acceptance: require zero
failed requests and exact forced output counts outside the unchanged client.

Dependency resolution: separate development-only `requirements_inferencex.in` and
hash-locked closure, constrained by existing project pins, Python3.14 through Bazel.
`--only-binary=:all:` excludes setup.py/build hooks. No torch/vLLM/GPU dependency.
36 wheel artifacts (including existing dependencies) fetched as inert data from PyPI
metadata and files.pythonhosted.org, matched to both metadata and lockfile hashes,
compatible interpreter/platform tags selected. `tools/review_inferencex_wheels.py`
checks ZIP paths, symlinks, uncompressed size and rejects .pth startup files before
extracting. Downloaded names/versions, complete archive members, URLs/hashes and
requires_dist: `docs/bench/data/2026-09-29-inferencex-security/wheels.json`.
Local copies: `third_party/research-serving/inferencex-wheels-v1/`.

Review limitations: not a line-by-line audit of every dependency/native extension.
Native wheel code includes numpy, tokenizers, regex, aiohttp, frozenlist, multidict,
yarl, propcache, safetensors and hf-xet. These are dev-only package artifacts from
canonical PyPI project names, not production inference dependencies. No pickle/model
loading is selected. Transformers is used only for an existing local tokenizer;
USE_TORCH/TF/FLAX=0, offline Hub and telemetry disabled. No trust_remote_code.

ClamAV: existing official image pinned
`clamav/clamav@sha256:0e31ce089574268aefa0b543767d66b70240ab51ed49eec53e07f18d5629d817`,
engine1.5.4,3,628,071 signatures, DB28129 dated2026-09-20 (nine days old; warning
retained). No scanner pull or system installation. Source and wheels/unpacked closure:
**4446 files, zero infected, exit0**, 257.17MiB scanned. Read-only source mount,
networknone, cap-dropALL, no-new-privileges, writable noexec512MiB tmpfs for extraction.
First attempt lacked writable temporary space and exited2; retained, NOT a clean scan.
Logs `source-clamav.log` and `scan-complete.log`. Old signatures cannot rule out novel
malware, and static scanning does not prove safety.

Execution defense: `tools/inferencex_client.py` clears inherited environment, provides
an empty temporary HOME/HF_HOME, disables downloads/telemetry/remote code, and installs
an audit hook rejecting non-loopback sockets/DNS and child processes. Serial tokenizer
generation avoids multiprocessing (outside timed serving). This is defense in depth,
not an OS sandbox against malicious native code. It imports reviewed dependencies only
after these controls. No benchmark data is sent externally; package/source fetches are
GETs for public artifacts. Client remains a separate Bazel target and dependency hub.

No package installed on the host and no global driver/system configuration changed.
No uploaded results or git push. Preserve all failures, hashes and logs locally.

## Unchanged result processor follow-up

User now requires the actual full upstream result-processing layer unchanged.
Same InferenceX revision. Additional fetched execution closure: results package init,
fixed_sequence, collect_results, metadata, topology and power init/common/single_node/
audit/window/multinode/native_multinode. Canonical raw GitHub URLs use that revision
and the exact `inferencex-e2e/infx/results/` paths. Full hashes:
`docs/bench/data/2026-09-29-inferencex-processing/sources.sha256`.

Reviewed imports, environment-dependent dispatch, file reads/writes, missing-power
handling and collector discovery. This closure uses Python standard library and the
already pinned benchmark_outcome only; no new wheels/install hooks/network client,
process execution, pickle or dynamic downloaded code. Single-node process_result
loads original JSON, calls build_result, executes power validation and audit, writes
JSON and prints it. Missing CSV emits telemetry_file_missing with invalid power.
The collector recursively loads JSON, so its input directory must contain only
processor outputs. Optional multinode branches are retained unchanged but not selected.
The adapter restricts environment/paths and denies network/subprocess audit events;
this remains defense in depth, not a native-code OS sandbox.

Existing scanner image and old DB unchanged. Before execution, scanned the results
research directory (includes one extra research-only Agentic module): **13 files,
zero infected, exit0**,278.72KiB scanned. Warning that signatures are older than seven
days retained; the same nine-day-old DB limitation applies. No fresh image pull.
Command: `docker run --rm --pull=never --network=none --cap-drop=ALL
--security-opt=no-new-privileges --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m
-v "$PWD/third_party/research-serving/InferenceX-f437f7bfd164422036b0de7e3818f8afb5bc70d7/inferencex-e2e/infx/results:/scan:ro"
--entrypoint clamscan
clamav/clamav@sha256:0e31ce089574268aefa0b543767d66b70240ab51ed49eec53e07f18d5629d817
--recursive --infected /scan`.
Log: `docs/bench/data/2026-09-29-inferencex-processing/source-scan.log`.
No claim of a complete line-by-line audit or absence of malware. Bazel fetch/hash
validation of these exact files is required; never execute the research checkout.
