# Preparing RAM-staged NVMe scratch storage

Status: imported-host buffers, asynchronous file transport and opt-in production
prefix-cache/scheduler integration are implemented. Real-model 80k state/logit and
serving gates pass; [results and limitations](../bench/2026-09-28-disk-prefix-serving.md).
Component commands and production flags are distinguished below.

## Engine boundary

The storage core has no filesystem-specific code or attribute changes. It creates
an exclusive mode-0600 scratch file in a caller-selected directory, immediately
unlinks it, preallocates the requested budget, and uses O_DIRECT with borrowed RAM
staging buffers. The file consumes disk space while open and is reclaimed on close.
It never mounts, formats, rebinds or opens a raw disk. Choose a directory controlled
by the service user, not an adversarial shared writable directory.

`storage.Options.alignment` may explicitly set **memory** and **offset/length**
alignment, in bytes, when the filesystem does not report `STATX_DIOALIGN`.
Both must be nonzero powers of two. Omitted configuration uses the reported values
or fails `DirectIoUnsupported`; it does not guess. Configured values may strengthen
but cannot weaken reported requirements. Reported unsupported direct I/O cannot
be overridden. The buffer base and slot stride must also satisfy the Vulkan import
alignment; the GPU capability query remains separate from filesystem policy.

The benchmark exposes a common alignment for both as `--direct-alignment BYTES`.
Zero/omitted means use filesystem-reported alignment. File budget (`--mib`) and the
bounded RAM staging size (chunk size × queue depth) are explicit. These are component
probe options; production server flags are listed below.

O_DIRECT requests direct I/O; it cannot prove that every filesystem/device path
avoids internal buffering. Filesystem preparation and its performance/quality
tradeoffs are deployment decisions. Nothing in core automatically disables checksums,
compression or copy-on-write.

## This host: btrfs on the Samsung 990 PRO

Observed 2026-09-28: `/home` is btrfs, mount compression `zstd:3`. The inspected
btrfs direct-write path can fall back to buffered I/O for checksummed files, and
its `getattr` does not report `STATX_DIOALIGN`. A **new empty NOCOW scratch directory**
is the measured setup. Files created inside inherit NOCOW, disabling data checksums
and compression for those scratch files. This is suitable for disposable cache
bytes, not a recommendation for other data.

If you choose this tradeoff, prepare a **new dedicated** directory explicitly:

```sh
# Examples for the operator; not performed automatically by zerv.
mkdir /chosen/path/new-zerv-scratch
chattr +C /chosen/path/new-zerv-scratch
lsattr -d /chosen/path/new-zerv-scratch
```

Do not recursively change an existing directory. Do not assume changing a flag
on a populated file retroactively changes its extents. Do not disable compression
on the whole filesystem or change mounts for this feature.

On this inspected volume, 4096-byte memory and offset alignment is appropriate;
the primary btrfs `check_direct_IO` checks `FS_INFO.sectorsize`. The observed
fundamental filesystem block size is also 4096 (`stat -f -c '%T %S'`). This is
host guidance, not a portable rule equating any filesystem's block size with its
DIO alignment. For another filesystem use its reported DIO alignment or establish
its own documented requirements before specifying an override.

Existing explicitly prepared local scratch directory: `third_party/nvme-probe`.
To measure without any new filesystem-attribute change:

```sh
tools/py bench/run_storage.py \
  --scratch-dir third_party/nvme-probe --direct-alignment 4096 \
  --output docs/bench/data/NEW_RUN_DIRECTORY
```

The runner requires a new output directory and uses at most 1 GiB scratch by
default. It records commands, hashes, configuration, raw trials and errors. Ordinary
CPU correctness tests may run on a differently configured filesystem; their byte
checks do not claim its I/O was physically direct. Actual NVMe throughput measurements
use the explicitly prepared deployment directory.

## Other filesystems and devices

No XFS/ext4 special path is needed in the core; use reported alignment where
available and verify the actual direct-I/O behavior on the selected volume.
For containers, grant only the required directory and kernel io_uring permission;
unsupported io_uring is an initialization error, not permission to change host policy.

The unmounted WD disk is not used by these commands. Its contents remain unknown.
The RAM-staged design requires neither exclusive storage nor a replacement GPU driver.

## Prefix archive integration (opt-in)

The serving flags now connect the immutable archive to checkpoint capture and
pending scheduler I/O (not the older standalone residency table):

```
--parallel 2 --kv-pool shared --spec-draft 0 --kv-type f16 \
--prefix-cache-disk-dir /operator/prepared/scratch \
--prefix-cache-disk-mib 8192 --prefix-cache-disk-entries 64 \
--prefix-cache-disk-alignment 4096
```

Alignment above is an **operator assertion**, not portable filesystem detection;
omit it when the filesystem reports suitable DIO alignment. Directory and disk
budget must be supplied together. No attributes/ioctls/mounts are changed. A new
0600, exclusive scratch file is unlinked immediately and preallocated; capacity is
bounded and returned at shutdown. The archive is not persistent across restarts.
Eight MiB of anonymous imported staging is included in the allocation budget;
metadata is separately bounded at 256 MiB. Record count, context and chunk-map
capacity determine metadata use. MTP and static/single-slot models are unsupported.

Cold checkpoints write full state and logical KV images, duplicating shared
prefix bytes on disk. Hot cache hits win ties. Longer disk hits allocate private
pages; if there is insufficient space, inference starts cold instead. Read errors
or failed integrity checks invalidate the record and reset the partial restore.
I/O cancellation drains before a slot or staging memory can be reused. Unknown
GPU DMA ownership is fail-stop, never silent fallback.

This is a capacity/latency tradeoff, not a claim of raw NVMe bandwidth: SHA256,
CPU bookkeeping and bounded synchronous GPU fences are in the path. Shutdown logs
archive writes/restores, physical bytes, evictions, skips, failures and cancellations.
See the [integration contract](../specs/disk-prefix-cache.md) and
[measured tradeoffs](../bench/2026-09-28-disk-prefix-serving.md): disk improves reuse
versus a small hot cache but adds cold-write latency; host caching was faster on
the measured workload. No asynchronous GPU copy/compute overlap or P2P claim.
