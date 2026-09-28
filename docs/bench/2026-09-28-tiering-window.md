# D.0b: bounded archive transfer windows

Date: **2026-09-28 (UTC)**; parent `472992c`. Functionality, verification and serving
evaluation gates closed. Performance target remains unmet.
This is neither proactive GPU→RAM preparation nor queued-demand prefetch, and does
not close the full plan or the faster-or-on-par goal.
[Pre-code contract](../specs/tiering-window.md), [paper-based plan](../design/async-tiering.md).

## Question and implementation

D.0a's source-only unit-boundary progress left immediate-turn disk caching slower
than host-only and tuned references. Would larger bounded transfers reduce
operation overhead without changing bytes, priority, ownership or numerical work?

`--prefix-cache-disk-chunk-mib 1|2|4|8` now resolves the store's ticket size and
imported staging allocation at startup. Eight tickets consume 8/16/32/64 MiB; six
optional-write tickets leave two physically available for reads. Device budgeting
includes actual imported staging exactly once. Startup reports the effective
window. Reject unsupported sizes, zero/non-multiple files and explicit use without
the disk tier before model loading. No new arithmetic, queue, runtime dependency,
steady-state allocation, admission rule or permission to mutate a packed source.
Default remains **1 MiB**; an optional larger window is not a general speed claim.

Larger tickets mean less metadata/submission work but more RAM, up to C−1 bytes
of per-record padding, and a larger indivisible copy/hash/cancellation quantum.
The model checker now locates pending/corrupt blocks using the actual ticket size;
its independent capture/poison windows remain 1 MiB. The four allowed sizes all
retain existing source-only progress, immutable leases and read priority.

## Independent oracle and executed correctness gates

Before native changes, `tests/reference/generate_archive_windows.py` generated
Python POSIX positional-I/O/hashlib fixtures for all four chunk sizes. Each image
has B=2C+17 valid bytes, byte(i)=(73i+floor(i/257)+19) mod 256, zero padding, reverse
positional writes and forward readback. Native tests compare both catalog digests
and direct physical-file reads, exact restored bytes, delayed-device cancellation,
no new issue with `allow_start=false`, and acquisition of two reserved tickets.

Fixture SHA256: `e7b6fe67e272c6a73dfb0031090a88d9c5a149bba54f3f39296366ba37ac43aa`.
Generator SHA256: `394430fda0b6f759dd1fac8aa1e366e99b3156a9f8a9ec42ed4e194a3a95c180`.
Ordinary tests embed the fixture and generator; they do not invoke external engines
or regenerate their expected outputs.

Executed, with logs in [the gate directory](data/2026-09-28-tiering-window/):

- CPU/Python/format **82/82**; archive and serve tests in both modes ×5 pass.
- GPU/spill **3/3**, host-driver GPU **2/2**. Debug executed (445.86/406.96 s);
  unchanged optimized targets were cache hits, as disclosed by the raw logs.
- Independent FP64/libllama **337/337** greedy rows, modes 0/512; every numerical
  bound passes ([oracle](data/2026-09-28-tiering-window/oracle.json)).
- Every size: first smoke, then six 257-token trials and one 80,000-token trial.
  Full canonical bytes, four continuation vocabulary rows, independent live decode,
  two packed vocabulary rows, cancellation/retry and last-block corruption pass.
  The long image is **5,399,773,184 bytes**. Positive small-source in-pack GPU
  quanta are 10/8/4/2 for 1/2/4/8 MiB respectively; the gate still requires progress.
- Ten CPU-only startup cases pass: explicit default without disk, invalid window
  sizes/non-multiple disk budgets reject early; every valid window reaches the
  intentionally nonexistent model and returns `FileNotFound`, without opening GPU.

Source/manifests: [1 MiB](data/2026-09-28-window-model-1/manifest.json),
[2 MiB](data/2026-09-28-window-model-2/manifest.json),
[4 MiB](data/2026-09-28-window-model-4/manifest.json),
[8 MiB](data/2026-09-28-window-model-8/manifest.json).
After those runs, three diagnostic-only assertions were added for actual ticket
size, imported byte count and eight allocated slots. The rebuilt checker passes
fresh 257-token runs at all four sizes ([final manifests](data/2026-09-28-window-final-1/manifest.json);
sibling directories `window-final-{2,4,8}`), including full byte/vocabulary,
packed, cancel/retry and corruption gates. Final CPU/Python/format **82/82**, GPU/spill
**3/3** and host GPU **2/2** pass; final device checks are cache hits, not additional
executions. No production code changed during the serving matrix.

Negative control after the serving matrix: temporarily setting optional write
tickets to eight instead of six fails the reserve assertion in both serve-test
modes (`expected 6, found 8`). Restored before final checks/model runs; retained
[negative log](data/2026-09-28-tiering-window/negative-reserve.log). This tests the
production limit constant, not timing-dependent physical disk-queue saturation.

## Setup and reproducible commands

RX 7900 XTX, 25,753,026,560 observed VRAM bytes; Bazel 9.2.0, Zig 0.16.0, pinned
Python via `tools/py`. Model Q4_0 SHA256
`ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`.
GPU jobs serialized, operator-prepared `third_party/nvme-probe` unchanged. No driver,
filesystem, clocks, power or system package changes. Unrelated CPU activity exists;
this is not an isolated-host experiment. Manifests retain build/runtime/source hashes.

```sh
tools/py tests/reference/generate_archive_windows.py
bazel test //...
bazel test //tests:archive //tests:archive_release_fast \
  //tests:serve //tests:serve_release_fast --runs_per_test=5
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py tools/zerv_build.py --test-host-gpu
for C in 1 2 4 8; do
  tools/py bench/run_archive_model.py --prefill --chunk-mib "$C" \
    --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
    --tokens 257 257 257 257 257 257 80000 \
    --output docs/bench/data/2026-09-28-window-model-$C
done
tools/py tools/verify_model.py \
  --oracle-dir third_party/model-oracle/2026-09-26-hermetic \
  --work-dir third_party/window-model-oracle \
  --report docs/bench/data/2026-09-28-tiering-window/oracle.json \
  --modes 0,512 --runtime host
tools/py docs/bench/data/2026-09-28-tiering-window/check_cli.py
```

Serving: fresh servers, warmup, three rounds with alternating order, two slots,
context 12,288, four conversations/two phased turns with zero idle. Native: 192 f16
KV pages, eight snapshots, 4,096 MiB host swap; disk adds 8,192 MiB/16 records,
4,096-byte asserted alignment and the selected staging budget. Compare host-only,
four window sizes, tuned Vulkan `llama-fa-b512`, and pinned RDNA3/HIP
`rdna3-nofusion` (batch 2,048, supported `GGML_CUDA_DISABLE_FUSION=1` inside container).
References use 8,192 MiB RAM cache/eight checkpoints. Exact executable commands,
artifacts and resources: [serving manifest](data/2026-09-28-window-serving/manifest.json).

```sh
E='zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=8,kv-swap-mib=4096,prefix-cache-tier=host'
for C in 1 2 4 8; do
  E="$E;zerv-f16@parallel=2,kv-type=f16,kv-pool-pages=192,prefix-cache-slots=8,kv-swap-mib=4096,prefix-cache-tier=host,prefix-cache-disk-dir=third_party/nvme-probe,prefix-cache-disk-mib=8192,prefix-cache-disk-entries=16,prefix-cache-disk-alignment=4096,prefix-cache-disk-chunk-mib=$C"
done
E="$E;llama-fa-b512;rdna3-nofusion"
tools/py bench/run_multiturn.py --output docs/bench/data/2026-09-28-window-serving \
  --workload bench/workloads/multiturn-distinct-v1.json --parallel 2 \
  --context-per-slot 12288 --levels 4 --rounds 3 --phased --warmup \
  --llama-cache-ram-mib 8192 --llama-checkpoints 8 --engines "$E"
# After the complete run, validate native text AND prompt/generated token counts:
tools/py docs/bench/data/2026-09-28-tiering-window/report.py
```

## Initial long component results (not a serving comparison)

Single long trial per size, seconds except maximum callback in milliseconds:

| MiB/ticket | capture s | write including packed computation s | restore s | max poll ms |
|---|---:|---:|---:|---:|
| 1 | 3.921 | 8.113 | 6.421 | 3.769 |
| 2 | 2.907 | 6.775 | 5.202 | 4.195 |
| 4 | 5.838 | 7.669 | 5.194 | 5.952 |
| 8 | 3.817 | 6.060 | 4.489 | 8.075 |

Different-time single runs with substantial host variance are not a controlled
causal estimate. The 8 MiB short smoke's 3.633 s capture outlier is retained in
[its log](data/2026-09-28-window-smoke-8/case-0-257.log). Lower operation count alone
does not close the historical restore regression or demonstrate faster serving.

## Repeated small component results

Five fresh-process trials after one warmup, mean ± sample SD, milliseconds:

| MiB/ticket | capture | write including identical packed computation | restore | maximum poll per trial |
|---|---:|---:|---:|---:|
| 1 | 81.265 ± 7.466 | 473.726 ± 7.849 | 190.569 ± 8.315 | 0.826 ± 0.058 |
| 2 | 150.728 ± 12.566 | 486.533 ± 6.892 | 183.152 ± 6.643 | 2.677 ± 1.206 |
| 4 | 132.952 ± 57.883 | 470.664 ± 44.756 | 163.564 ± 8.891 | 3.395 ± 0.595 |
| 8 | 126.492 ± 15.378 | 479.865 ± 23.032 | 161.936 ± 5.776 | 6.651 ± 0.974 |

Restore timings improve with larger windows; write/capture timings are noisy and
not monotonically better. The maximum callback grows materially. These are grouped
component runs, not interleaved pairs. The same pack executes in every write timer;
comparison with old write-only timers would misrepresent the work performed.

## Serving results

All **120 native responses and prompt/generated token counts match** each other
and the historical baseline exactly. Every request completed. Reference streams
are different: text-and-count identity is 15/24 for Vulkan and 1/24 for HIP.
This is the same weight artifact but not identical generated work or a newly
established quality-equivalent win. Mean ± sample SD across three rounds:

| configuration | wall s | reuse TTFT p50 s | aggregate output tok/s | stream gap p99 ms |
|---|---:|---:|---:|---:|
| native host-only | 47.476 ± 0.229 | 4.401 ± 2.656 | 12.512 ± 0.060 | 144.754 ± 0.245 |
| native host+disk 1 MiB | 55.472 ± 0.727 | 9.746 ± 0.271 | 10.709 ± 0.139 | 145.070 ± 0.309 |
| native host+disk 2 MiB | 53.704 ± 3.886 | 9.818 ± 0.055 | 11.101 ± 0.838 | 144.769 ± 0.409 |
| native host+disk 4 MiB | 57.913 ± 1.137 | 10.275 ± 0.212 | 10.259 ± 0.199 | 147.174 ± 1.647 |
| native host+disk 8 MiB | 51.282 ± 1.534 | 7.995 ± 3.536 | 11.590 ± 0.345 | 144.815 ± 0.180 |
| tuned Vulkan batch 512 | 48.871 ± 0.199 | 3.789 ± 0.704 | 11.895 ± 0.048 | 562.827 ± 1.818 |
| tuned RDNA3/HIP nofusion | 48.691 ± 0.985 | 3.202 ± 0.700 | 12.072 ± 0.294 | 1417.215 ± 675.542 |

The 8 MiB configuration is ~7.6% faster in wall time than 1 MiB, but ~8.0%
slower than host-only and ~5.3% slower than HIP. Four MiB regresses relative to
1 MiB. All disk configurations lose to host-only in this matrix; no overall win.
Reuse latency remains worse than both tuned references. Native streaming gaps are
smaller, but that does not erase wall/TTFT losses.

### Disk work and ownership

Each cell lists rounds 0/1/2. Physical bytes are decimal GB; hold time is seconds.

| MiB | completed writes | restores | GB written | GB read | maximum source hold | cancellations |
|---|---|---|---|---|---|---|
| 1 | 2 / 2 / 2 | 0 / 0 / 0 | 1.188 / 1.183 / 1.166 | 0 / 0 / 0 | 14.331 / 14.020 / 15.012 | 1 / 1 / 1 |
| 2 | 3 / 3 / 3 | 0 / 0 / 0 | 1.627 / 2.145 / 2.154 | 0 / 0 / 0 | 7.165 / 7.338 / 7.326 | 1 / 1 / 1 |
| 4 | 5 / 5 / 5 | 0 / 0 / 0 | 3.330 / 3.490 / 3.389 | 0 / 0 / 0 | 4.638 / 5.655 / 4.893 | 1 / 1 / 2 |
| 8 | 5 / 6 / 4 | 1 / 1 / 1 | 2.911 / 4.102 / 2.609 | 0.671 / 0.671 / 0.671 | 3.017 / 5.143 / 10.612 | 1 / 1 / 1 |

**Eight MiB produces a real immediate-turn restore in every round** (671,088,640
physical bytes each), unlike prior zero-idle runs. Shorter source holds can make
backing available in time, but more data is written and often never read. The
third 8 MiB source hold still reaches 10.612 s. Selection and preservation of
soon-needed host paths remain problems; faster copying is not demand-aware policy.
All disk runs report zero failures, extent evictions and capacity skips.

Peak sampled VRAM: native host-only 18.596–18.605 GB; disk variants
18.596–18.605 GB; Vulkan 18.158 GB; HIP 18.701 GB. Imported staging is host memory,
not an increase in the configured GPU KV capacity. Server VmHWM: host-only
15,472,932–15,479,472 KiB, disk 1/2/4/8 MiB respectively approximately
15,483–15,487 / 15,487–15,495 / 15,511–15,512 / 15,544–15,545 thousand KiB;
Vulkan 15,818,360–15,818,588 KiB; HIP 16,125,396–16,126,688 KiB. Exact per-round
values are retained in the manifest/validated summary. VmHWM includes startup
weight mapping; VmRSS sampled at shutdown can be much lower and is not a peak.
HIP samples the actual container process, not its Docker launcher. These counters
do not measure every driver/GTT allocation or page-cache byte.

## Decision and remaining plan

Keep **1 MiB default**. Expose larger windows as an explicit memory/latency tradeoff;
8 MiB is a measured candidate when disk capacity is needed, not a universally
better setting. The successful restores close the zero-idle functionality gap
for this configuration, not the performance objective. Final checks pass. Next is
D.1 GPU→RAM ownership research/specification and independent transaction oracle,
then its implementation, followed by D.2 queued-demand protection/prefetch. Neither
feature exists yet.

RDNA3's original fused path crashes and only its supported nofusion configuration
is compared; retained evidence/tuning is in [C.3](2026-09-28-tiering-pressure.md).
Prior vLLM W4A16/FP8 runs generated empty answers and remain non-equivalent;
HyperQwen is NVIDIA-only and SGLang remains untested. No broad best-server claim.
