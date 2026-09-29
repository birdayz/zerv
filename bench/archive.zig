//! CPU archive component: bytes + SHA256 + real worker, not model/GPU/serving.
const std = @import("std");
const storage = @import("storage");
const archive = @import("session").archive;
const linux = std.os.linux;
const size = 64 << 20;
const Device = struct {
    source: []u8,
    target: []u8,
    fn start(ctx: *anyopaque, _: u32, offset: u64, bytes: []u8, importing: bool) !void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        if (importing) @memcpy(d.target[@intCast(offset)..][0..bytes.len], bytes) else @memcpy(bytes, d.source[@intCast(offset)..][0..bytes.len]);
    }
    fn poll(_: *anyopaque) !bool {
        return true; // CPU memcpy completed at start; still acknowledge the same protocol.
    }
};
fn drain(a: *archive.Archive, d: *Device, io: std.Io) !void {
    while (true) {
        const p = try a.advance(.{ .ctx = d, .start = Device.start, .poll = Device.poll }, 0, false);
        if (p.done) return;
        if (!p.progressed) try std.Io.sleep(io, .fromMicroseconds(10), .awake);
    }
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.gpa);
    if (args.len != 3 and args.len != 4) return error.Usage;
    const demand = args.len == 4 and std.mem.eql(u8, args[3], "demand");
    if (args.len == 4 and !demand) return error.Usage;
    const alignment = try std.fmt.parseInt(u32, args[2], 10);
    const dir = try init.gpa.dupeZ(u8, args[1]);
    defer init.gpa.free(dir);
    const fd_rc = linux.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
    if (linux.errno(fd_rc) != .SUCCESS) return error.OpenDirectoryFailed;
    const fd: i32 = @intCast(fd_rc);
    defer _ = linux.close(fd);
    const mem = try init.gpa.alignedAlloc(u8, .fromByteUnits(4096), 8 << 20);
    defer init.gpa.free(mem);
    const store = try storage.Store.create(init.gpa, .{ .dir_fd = fd, .name = "archive-bench", .file_bytes = 4 * size, .slot_bytes = 1 << 20, .alignment = if (alignment == 0) null else .{ .memory = alignment, .offset = alignment } }, mem);
    defer store.destroy() catch @panic("disk still pending");
    var a = try archive.Archive.init(init.gpa, store, .{ .records = 4, .context = 16, .slots = 1, .max_bytes = size });
    defer a.deinit() catch @panic("archive still pending");
    const source = try init.gpa.alloc(u8, size);
    defer init.gpa.free(source);
    const target = try init.gpa.alloc(u8, size);
    defer init.gpa.free(target);
    for (source, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 13);
    var d: Device = .{ .source = source, .target = target };
    if (demand) return demandBench(&a, &d, init.io);
    for (0..6) |round| {
        const begin = std.Io.Clock.awake.now(init.io);
        for (0..4) |i| {
            const tokens = [_]u32{ @intCast(round), @intCast(i) };
            if (!try a.startWrite(0, &tokens, size)) return error.SkippedWrite;
            try drain(&a, &d, init.io);
        }
        const write_ns = begin.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
        if (linux.errno(linux.fsync(store.fd)) != .SUCCESS) return error.SyncFailed;
        var read_ns: i96 = 0;
        for (0..4) |i| {
            @memset(target, 0xa5);
            const tokens = [_]u32{ @intCast(round), @intCast(i), 99 };
            const record = a.lookup(&tokens) orelse return error.MissingRecord;
            const start = std.Io.Clock.awake.now(init.io);
            try a.startRead(0, record);
            try drain(&a, &d, init.io);
            read_ns += start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds;
            if (!std.mem.eql(u8, source, target)) return error.WrongBytes;
        }
        if (round > 0) std.debug.print("{{\"trial\":{d},\"bytes\":{d},\"chunk\":1048576,\"depth\":8,\"write_ns\":{d},\"read_ns\":{d},\"exact\":true}}\n", .{ round - 1, 4 * size, write_ns, read_ns });
    }
}

fn demandBench(a: *archive.Archive, d: *Device, io: std.Io) !void {
    const Owner = @import("session").readahead.Owner;
    const key: @import("session").readahead.Key = .{ .slot = 0, .order = 1 };
    if (!try a.startWrite(0, &.{1}, size)) return error.SkippedWrite;
    try drain(a, d, io);
    if (linux.errno(linux.fsync(a.store.fd)) != .SUCCESS) return error.SyncFailed;
    const record = a.lookup(&.{ 1, 2 }) orelse return error.MissingRecord;
    for (0..6) |trial| for (0..3) |index| {
        const window: u32 = @intCast(if (trial % 2 == 0) index else 2 - index);
        var owner: Owner = .{};
        @memset(d.target, 0xa5);
        const starts = a.stats.device_starts;
        const begin = std.Io.Clock.awake.now(io);
        if (window == 0) {
            try a.startRead(0, record);
        } else {
            try owner.start(a, key, record, window);
            while (owner.completed_bytes < @as(u64, window) * a.store.slot_bytes) {
                _ = try owner.poll(a, key, true);
                try std.Io.sleep(io, .fromMicroseconds(10), .awake);
            }
            if (a.stats.device_starts != starts or owner.occupancy(a) != window) return error.EarlyUpload;
            if (try owner.take(a, key) != record) return error.WrongRecord;
        }
        const handed = std.Io.Clock.awake.now(io);
        try drain(a, d, io);
        const end = std.Io.Clock.awake.now(io);
        if (!std.mem.eql(u8, d.source, d.target)) return error.WrongBytes;
        if (a.store.freeSlots() != 8 or a.entries[record].readers != 0) return error.LeakedOwner;
        if (trial > 0) std.debug.print("{{\"trial\":{d},\"window\":{d},\"bytes\":{d},\"stage_ns\":{d},\"foreground_ns\":{d},\"total_ns\":{d},\"submitted_bytes\":{d},\"completed_bytes\":{d},\"exact\":true}}\n", .{ trial - 1, window, size, begin.durationTo(handed).nanoseconds, handed.durationTo(end).nanoseconds, begin.durationTo(end).nanoseconds, owner.submitted_bytes, owner.completed_bytes });
        if (window > 0) {
            var canceled: Owner = .{};
            try canceled.start(a, key, record, window);
            while (canceled.completed_bytes < @as(u64, window) * a.store.slot_bytes) {
                _ = try canceled.poll(a, key, true);
                try std.Io.sleep(io, .fromMicroseconds(10), .awake);
            }
            const cancel_start = std.Io.Clock.awake.now(io);
            while (canceled.held != null) _ = try canceled.poll(a, null, false);
            const cancel_ns = cancel_start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
            if (a.store.freeSlots() != 8 or a.entries[record].readers != 0 or canceled.discarded_bytes != @as(u64, window) * a.store.slot_bytes) return error.LeakedOwner;
            if (trial > 0) std.debug.print("{{\"trial\":{d},\"window\":{d},\"cancel_ns\":{d},\"discarded_bytes\":{d},\"exact\":true}}\n", .{ trial - 1, window, cancel_ns, canceled.discarded_bytes });
        }
    };
}
