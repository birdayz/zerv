//! Bounded disk-worker component benchmark, not a serving benchmark.
//! Usage: zerv-storage-bench SCRATCH_DIRECTORY TOTAL_MIB CHUNK_MIB DEPTH [ALIGNMENT]
//! ALIGNMENT is an explicit filesystem contract for both memory and offsets; 0/omitted
//! requires filesystem-reported alignment. Directory preparation belongs to deployment.
//! All operations use the same unlinked, preallocated file and aligned staging slots.
const std = @import("std");
const storage = @import("storage");
const linux = std.os.linux;

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}
fn pattern(i: usize) u8 {
    return @truncate((i *% 2654435761) >> 7);
}
fn check(bytes: []const u8) !void {
    for (bytes, 0..) |b, i| if (b != pattern(i)) return error.WrongBytes;
}

fn synchronous(s: *storage.Store, io: std.Io, write: bool, verify: bool) !u64 {
    const bytes = s.memory[0..s.slot_bytes];
    if (!write) @memset(bytes, 0);
    const start = now(io);
    var offset: u64 = 0;
    while (offset < s.file_bytes) : (offset += bytes.len) {
        const n = if (write) linux.pwrite(s.fd, bytes.ptr, bytes.len, @intCast(offset)) else linux.pread(s.fd, bytes.ptr, bytes.len, @intCast(offset));
        if (n != bytes.len) return error.ShortOrFailedReferenceIo;
        if (verify and !write) try check(bytes);
    }
    return @intCast(now(io) - start);
}

fn asynchronous(s: *storage.Store, io: std.Io, write: bool, verify: bool) !u64 {
    var tickets: [storage.max_depth]?storage.Ticket = @splat(null);
    if (!write) @memset(s.memory, 0);
    var next: u64 = 0;
    var finished: u64 = 0;
    const start = now(io);
    while (finished < s.file_bytes) {
        for (tickets[0 .. s.memory.len / s.slot_bytes]) |*t| {
            if (t.* == null and next < s.file_bytes) {
                const ticket = try s.acquire();
                try s.submit(ticket, if (write) .write else .read, next, s.slot_bytes);
                t.* = ticket;
                next += s.slot_bytes;
            }
            if (t.*) |ticket| {
                if (try s.poll(ticket)) |c| {
                    if (!c.exact()) return error.ShortOrFailedWorkerIo;
                    if (verify and !write) try check(try s.buffer(ticket));
                    try s.release(ticket);
                    t.* = null;
                    finished += s.slot_bytes;
                }
            }
        }
        // Deliberate polling benchmark: records throughput including caller bookkeeping,
        // not a production scheduler policy. Worker sleeps on a futex when idle.
        std.atomic.spinLoopHint();
    }
    return @intCast(now(io) - start);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.gpa);
    if (args.len != 5 and args.len != 6) return error.Usage;
    const alignment = if (args.len == 6) try std.fmt.parseInt(u32, args[5], 10) else 0;
    if (alignment != 0 and (!std.math.isPowerOfTwo(alignment) or alignment > 64 << 20)) return error.InvalidOptions;
    const total_mib = try std.fmt.parseInt(usize, args[2], 10);
    const chunk_mib = try std.fmt.parseInt(usize, args[3], 10);
    const depth = try std.fmt.parseInt(usize, args[4], 10);
    if (chunk_mib == 0 or chunk_mib > 64 or total_mib == 0 or total_mib > 4096 or total_mib % chunk_mib != 0 or depth == 0 or depth > 32) return error.InvalidOptions;
    const chunk = chunk_mib << 20;
    const total = total_mib << 20;
    const bytes = chunk * depth;
    const dir = try init.gpa.dupeZ(u8, args[1]);
    defer init.gpa.free(dir);
    const fd_rc = linux.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.OpenDirectoryFailed;
    const dir_fd: i32 = @intCast(fd_rc);
    defer _ = linux.close(dir_fd);
    const map_alignment: usize = @max(4096, alignment);
    const rc = linux.mmap(null, bytes + map_alignment, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (linux.errno(rc) != .SUCCESS) return error.MmapFailed;
    const raw = @as([*]u8, @ptrFromInt(rc))[0 .. bytes + map_alignment];
    defer _ = linux.munmap(raw.ptr, raw.len);
    const mem = @as([*]u8, @ptrFromInt(std.mem.alignForward(usize, rc, map_alignment)))[0..bytes];
    _ = linux.madvise(mem.ptr, mem.len, linux.MADV.HUGEPAGE);
    const s = try storage.Store.create(init.gpa, .{ .dir_fd = dir_fd, .name = "zerv-storage-probe", .file_bytes = total, .slot_bytes = chunk, .alignment = if (alignment == 0) null else .{ .memory = alignment, .offset = alignment } }, mem);
    defer s.destroy() catch @panic("benchmark failure with outstanding DMA; no early free");
    for (0..depth) |i| for (mem[i * chunk ..][0..chunk], 0..) |*b, j| {
        b.* = pattern(j);
    };
    // Independent synchronous write -> worker read, worker write -> synchronous read.
    _ = try synchronous(s, init.io, true, false);
    _ = try asynchronous(s, init.io, false, true);
    _ = try asynchronous(s, init.io, true, false);
    _ = try synchronous(s, init.io, false, true);
    // Unreported warmup then five trials; alternate the reference/worker order.
    for (0..6) |round| {
        for (0..2) |j| {
            const worker = (round + j) % 2 == 0;
            const w = if (worker) try asynchronous(s, init.io, true, false) else try synchronous(s, init.io, true, false);
            if (linux.errno(linux.fsync(s.fd)) != .SUCCESS) return error.FsyncFailed;
            const r = if (worker) try asynchronous(s, init.io, false, false) else try synchronous(s, init.io, false, false);
            if (round > 0) std.debug.print("{{\"trial\":{d},\"path\":\"{s}\",\"depth\":{d},\"bytes\":{d},\"staging_bytes\":{d},\"write_ns\":{d},\"read_ns\":{d},\"exact\":true}}\n", .{ round - 1, if (worker) "worker" else "pread-pwrite", if (worker) depth else 1, total, bytes, w, r });
        }
    }
    _ = try asynchronous(s, init.io, false, true);
}
