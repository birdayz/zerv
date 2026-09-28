//! NVMe prerequisite probe; docs/bench/2026-09-28-nvme-buffers.md.
//! Usage: zerv-disk-probe NEW_FILE TOTAL_MIB CHUNK_MIB DEPTH
//! Creates exclusively; never overwrites an existing file. Caller removes it afterwards.
//! Limits: 1..4096 MiB, whole chunks, depth 1..32. Three alternating-order trials plus
//! one unreported warmup. O_DIRECT is required, no buffered fallback. Scratch, not durable.
const std = @import("std");
const gpu = @import("zerv").gpu;
const linux = std.os.linux;

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn pattern(i: usize) u8 {
    return @truncate(i *% 2654435761 >> 7);
}

fn check(buf: []const u8) !void {
    for (buf, 0..) |b, i| if (b != pattern(i)) return error.WrongBytes;
}

// An unexpected ring submission/wait failure has an unknown number of outstanding DMA
// operations. Terminate instead of freeing/reusing buffers while the kernel may own them.
fn ringFailure(e: anyerror) noreturn {
    std.debug.print("fatal io_uring failure (no buffer reuse): {}\n", .{e});
    std.process.exit(2);
}

/// Bounded batches. Drain *every* submitted request before returning an I/O error; no
/// buffer or file extent can be reused until its completion is consumed.
fn run(io: std.Io, ring: *linux.IoUring, fd: i32, buf: []u8, chunk: usize, depth: usize, write: bool) !u64 {
    const n = buf.len / chunk;
    var next: usize = 0;
    const t0 = now(io);
    while (next < n) {
        const count = @min(depth, n - next);
        for (0..count) |j| {
            const k = next + j;
            const s = buf[k * chunk ..][0..chunk];
            if (write) {
                _ = ring.write(j, fd, s, k * chunk) catch |e| ringFailure(e);
            } else {
                _ = ring.read(j, fd, .{ .buffer = s }, k * chunk) catch |e| ringFailure(e);
            }
        }
        var submitted: usize = 0;
        while (submitted < count) {
            const got = ring.submit() catch |e| ringFailure(e);
            if (got == 0) ringFailure(error.NoSubmissionProgress);
            submitted += got;
        }
        var failed = false;
        var seen: u32 = 0;
        for (0..count) |_| {
            const cqe = ring.copy_cqe() catch |e| ringFailure(e);
            if (cqe.user_data >= count) ringFailure(error.InvalidCompletion);
            const bit = @as(u32, 1) << @intCast(cqe.user_data);
            if (seen & bit != 0) ringFailure(error.DuplicateCompletion);
            seen |= bit;
            if (cqe.res != chunk) {
                std.debug.print("request {d}: result {d}, wanted {d}\n", .{ next + cqe.user_data, cqe.res, chunk });
                failed = true;
            }
        }
        if (failed) return error.ShortOrFailedIo;
        next += count;
    }
    return @intCast(now(io) - t0);
}

fn transfer(io: std.Io, host: *gpu.Buffer, dev: *gpu.Buffer, up: bool) !u64 {
    var c = try gpu.Commands.init(host.device);
    defer c.deinit() catch @panic("commands retained after probe");
    try c.reset();
    try c.begin();
    if (up) try c.copy(host, 0, dev, 0, host.size) else try c.copy(dev, 0, host, 0, host.size);
    try c.barrier(.transfer, if (up) .transfer else .host);
    try c.end();
    const t0 = now(io);
    c.run(10_000_000_000) catch |e| ringFailure(e); // may still be pending
    return @intCast(now(io) - t0);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) return error.Usage;
    const total_mib = try std.fmt.parseInt(usize, args[2], 10);
    const chunk_mib = try std.fmt.parseInt(usize, args[3], 10);
    const depth = try std.fmt.parseInt(usize, args[4], 10);
    if (total_mib == 0 or total_mib > 4096 or chunk_mib == 0 or chunk_mib > total_mib or total_mib % chunk_mib != 0 or depth == 0 or depth > 32) return error.InvalidOptions;
    const total = total_mib << 20;
    const chunk = chunk_mib << 20;
    const path = try a.dupeZ(u8, args[1]);
    defer a.free(path);
    const rc = linux.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true, .DIRECT = true }, 0o600);
    if (linux.errno(rc) != .SUCCESS) return error.OpenFailed;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    if (linux.errno(linux.fallocate(fd, 0, 0, @intCast(total))) != .SUCCESS) return error.FallocateFailed;
    var ring = try linux.IoUring.init(64, 0);
    defer ring.deinit();
    var device = try gpu.Device.open(.{ .max_allocated_bytes = 3 * total + (1 << 30), .host_import = true });
    defer device.deinit() catch @panic("device retained after probe");
    var host = try gpu.Buffer.init(&device, total, .host);
    defer host.deinit() catch @panic("host retained after probe");
    var dev = try gpu.Buffer.init(&device, total, .device);
    defer dev.deinit() catch @panic("device buffer retained after probe");
    const mapped = try host.mapped();
    // Overallocate to satisfy the queried import alignment, retaining the original mmap
    // range for munmap. Advice is before touching pages, not after importing/pinning them.
    const alignment: usize = @intCast(device.host_import_alignment);
    if (alignment > total or total % alignment != 0) return error.UnsupportedAlignment;
    const anon_rc = linux.mmap(null, total + alignment, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (linux.errno(anon_rc) != .SUCCESS) return error.MmapFailed;
    const raw = @as([*]u8, @ptrFromInt(anon_rc))[0 .. total + alignment];
    defer _ = linux.munmap(raw.ptr, raw.len);
    const anon = @as([*]u8, @ptrFromInt(std.mem.alignForward(usize, anon_rc, alignment)))[0..total];
    const advice = linux.errno(linux.madvise(anon.ptr, total, linux.MADV.HUGEPAGE));
    for (anon, 0..) |*b, i| b.* = pattern(i);
    var imported = try gpu.Buffer.initImported(&device, anon);
    defer imported.deinit() catch @panic("import retained after probe");
    std.debug.print("device={s} import_alignment={d} madvise={s} total_mib={d} chunk_mib={d} depth={d}\n", .{ device.name(), alignment, @tagName(advice), total_mib, chunk_mib, depth });
    // Not a disk-cache implementation. Trial -1 warms both paths, then order alternates.
    for (0..4) |round| {
        for (0..2) |j| {
            const use_import = (j + round) % 2 != 0;
            const b = if (use_import) &imported else &host;
            const buf = if (use_import) anon else mapped;
            for (buf, 0..) |*x, i| x.* = pattern(i);
            const w = try run(io, &ring, fd, buf, chunk, depth, true);
            if (linux.errno(linux.fsync(fd)) != .SUCCESS) return error.SyncFailed;
            @memset(buf, 0);
            const r = try run(io, &ring, fd, buf, chunk, depth, false);
            try check(buf);
            const up = try transfer(io, b, &dev, true);
            @memset(buf, 0); // a no-op download must fail the byte gate
            const down = try transfer(io, b, &dev, false);
            try check(buf);
            if (round > 0) std.debug.print("{{\"trial\":{d},\"buffer\":\"{s}\",\"bytes\":{d},\"write_ns\":{d},\"read_ns\":{d},\"upload_ns\":{d},\"download_ns\":{d},\"exact\":true}}\n", .{ round - 1, if (use_import) "imported-anon" else "vulkan-host", total, w, r, up, down });
        }
    }
}
