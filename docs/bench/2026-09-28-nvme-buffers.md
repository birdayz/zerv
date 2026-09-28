# Imported host buffers for NVMe staging — 2026-09-28

18d.6a component gate, completed. This is **not an integrated NVMe prefix cache or
P2P result**, and not a serving speedup. The user approved the RAM-staged path.

## Question and setup

Can disk direct I/O use our Vulkan staging memory without an extra CPU copy?
Samsung 990 PRO, existing NOCOW scratch directory on btrfs, Ryzen 3900X, RX 7900 XTX,
RADV 26.2.3 host driver, kernel 7.2.6. One GiB temporary file, 8 MiB requests;
depths 1/8/8/1 (ABBA), one warmup and three alternating buffer-order trials per
run. Six retained trials per buffer/depth. Files exclusively created, then removed.
No driver, mount or directory-attribute changes in this run. Filesystem cache
bypassed with O_DIRECT; fsync outside write timing. Not a sustained SSD-cache
exhaustion benchmark. Other host work was not stopped; variance is retained.

Commands actually run:

```sh
bazel test //...
bazel test //tests:gpu //tests:gpu_release_fast //tests:gpu_spills
tools/py bench/run_disk_probe.py --scratch-dir third_party/nvme-probe \
  --output docs/bench/data/2026-09-28-nvme-buffers
```

Harness gates production-driver tests first (the same as
`tools/py tools/zerv_build.py --test-host-gpu`), builds through Bazel, records
source/build hashes, driver identity, mountinfo, RAM, commands and all raw trials.
[Manifest/raw logs/summary](data/2026-09-28-nvme-buffers/).

## Results

Decimal GB/s, arithmetic mean ± sample standard deviation, n=6 each:

| Buffer | QD | Disk read | Disk write | GPU upload | GPU download |
|---|---:|---:|---:|---:|---:|
| Vulkan host | 1 | 3.00 ± 0.67 | 2.86 ± 0.72 | 8.05 ± 0.76 | 10.66 ± 1.68 |
| Imported anonymous | 1 | 6.70 ± 0.30 | 6.08 ± 0.66 | 7.77 ± 0.92 | 9.81 ± 1.84 |
| Vulkan host | 8 | 3.13 ± 1.32 | 2.64 ± 0.94 | 7.55 ± 0.86 | 9.81 ± 1.66 |
| Imported anonymous | 8 | 6.90 ± 0.14 | 6.28 ± 0.68 | 7.67 ± 1.01 | 10.06 ± 1.87 |

All **24 retained trials** passed disk byte comparison and actual GPU round-trip
comparison (destination zeroed before GPU download, transfer→host barrier).
Imported buffers give substantially better disk throughput in this experiment;
GPU transfer means overlap with considerable noise, not evidence of a GPU-copy
speedup. This test measures each stage separately, not their overlap or impact
on live inference. Default 1 GiB fits the SSD's write cache; no sustained-write
claim. Earlier chat-only figures are superseded for this gate.

## Verification and limits

- `bazel test //...`: 75/75 pass (74 cached; Python/build-list changes rerun).
- GPU Debug/ReleaseFast/spills: 3/3 pass, cached for identical sources/runtime.
- Production-driver gate: 2/2 pass, cached for identical source/driver identity.
- Vulkan ABI independently regenerated from pinned C headers: 71 structs / 95
  constants. Imported-buffer tests cover borrowed lifetime, alignment, limits,
  retained destruction, option disabled and exact distinct-buffer copies.
- Initial failures: stale ABI counts (64/86) and Zig declaration placement fixed
  before these passes. Original probe lacked GPU-byte verification; the current
  probe corrects that and retains every trial.

The disk byte oracle is the known CPU pattern; imported-buffer ABI uses the
independent Vulkan C-header fixture. The ordinary host allocation is an equivalent
in-tree allocation-path control, not an independent model engine. llama-server
has no equivalent arbitrary staging-buffer operation; comparison belongs at the
future runnable disk-cache serving milestone. No C++ production dependencies.

Decision: use caller-owned anonymous staging memory, imported into Vulkan, in the
next bounded asynchronous disk-store increment. That store and cache/scheduler
integration are not provided by this buffer gate.
