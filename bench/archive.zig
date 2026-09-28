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
    if (args.len != 3) return error.Usage;
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
